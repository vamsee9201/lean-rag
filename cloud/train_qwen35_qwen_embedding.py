#!/usr/bin/env python3
"""Train a fresh Qwen3.5 9B LoRA adapter on Qwen-embedding-hybrid contexts."""

from __future__ import annotations

from importlib.metadata import version
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

from huggingface_hub import HfApi
from packaging.version import Version

from qwen3_adapter_metadata import sanitize_adapter


MODEL_ID = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
OUTPUT_REPO = "vamsee9201/qwen35-9b-qwen-embedding-rag-lora"
TRAIN_DATA = Path("/mnt/data/qwen_embedding/sft/train_sft.jsonl")
VALIDATION_DATA = Path("/mnt/data/qwen_embedding/sft/validation_sft.jsonl")
OUTPUT_DIR = Path("/tmp/qwen35-qwen-embedding-full")


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def verify_reload(rows: list[dict], validation_path: Path) -> None:
    expected = {
        json.dumps([m for m in json.loads(line)["messages"] if m["role"] in {"system", "user"}], sort_keys=True)
        for line in validation_path.read_text().splitlines() if line.strip()
    }
    seen = set()
    for row in rows:
        prompt = json.dumps(
            [m for m in row.get("messages", []) if m.get("role") in {"system", "user"}],
            sort_keys=True,
        )
        if prompt not in expected or prompt in seen or not str(row.get("response", "")).strip():
            raise RuntimeError("Adapter reload returned a duplicate, unmatched, or empty response")
        seen.add(prompt)


def preflight() -> dict:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    for path in (TRAIN_DATA, VALIDATION_DATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    run(["nvidia-smi"])
    versions = {
        package: version(package)
        for package in ("ms-swift", "torch", "transformers", "peft", "flash-attn", "flash-linear-attention")
    }
    if Version(versions["transformers"]) < Version("5.9"):
        raise RuntimeError("Qwen3.5 requires transformers>=5.9")
    for module in ("torch", "transformers", "swift", "flash_attn", "fla", "causal_conv1d"):
        importlib.import_module(module)
    counts = {
        "train_records": sum(bool(line.strip()) for line in TRAIN_DATA.read_text().splitlines()),
        "validation_records": sum(bool(line.strip()) for line in VALIDATION_DATA.read_text().splitlines()),
    }
    if counts != {"train_records": 2256, "validation_records": 111}:
        raise ValueError(f"Unexpected hybrid dataset counts: {counts}")
    return {"versions": versions, "model": MODEL_ID, "base_revision": BASE_REVISION, **counts}


def main() -> None:
    preflight_result = preflight()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run([
        "swift", "sft", "--model", MODEL_ID, "--model_revision", BASE_REVISION,
        "--use_hf", "true",
        "--dataset", str(TRAIN_DATA), "--val_dataset", str(VALIDATION_DATA),
        "--split_dataset_ratio", "0", "--tuner_type", "lora", "--target_modules", "all-linear",
        "--freeze_vit", "true", "--freeze_aligner", "true", "--lora_rank", "8",
        "--lora_alpha", "32", "--lora_dropout", "0", "--torch_dtype", "bfloat16",
        "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "4", "--gradient_checkpointing", "true",
        "--learning_rate", "1e-4", "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.05",
        "--num_train_epochs", "1", "--eval_steps", "100", "--save_steps", "100",
        "--save_total_limit", "3", "--load_best_model_at_end", "true",
        "--metric_for_best_model", "loss", "--greater_is_better", "false", "--logging_steps", "1",
        "--max_length", "4096", "--packing", "false", "--padding_free", "false",
        "--attn_impl", "flash_attn", "--add_non_thinking_prefix", "true",
        "--loss_scale", "default+ignore_empty_think", "--dataset_num_proc", "2",
        "--dataloader_num_workers", "2", "--save_only_model", "true", "--report_to", "none",
        "--seed", "20260906", "--output_dir", str(OUTPUT_DIR),
    ])
    checkpoints = sorted(OUTPUT_DIR.glob("**/checkpoint-*"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError("Training completed without a checkpoint")
    state = json.loads((checkpoints[-1] / "trainer_state.json").read_text())
    checkpoint = Path(state.get("best_model_checkpoint") or checkpoints[-1])
    if not list(checkpoint.glob("adapter_model*.safetensors")):
        raise RuntimeError(f"No LoRA adapter found in {checkpoint}")
    api = HfApi()
    api.create_repo(OUTPUT_REPO, repo_type="model", private=False, exist_ok=True)
    selected_copy = OUTPUT_DIR / "selected_checkpoint" / checkpoint.name
    shutil.copytree(checkpoint, selected_copy, dirs_exist_ok=True)
    sanitize_adapter(
        selected_copy, MODEL_ID, title="GovInfo Qwen3.5-9B RAG adapter",
        description="Selected LoRA checkpoint for the Lean RAG fourth experiment.",
        tags=("lora", "text-generation"),
    )
    training_manifest = OUTPUT_DIR / "training_manifest.json"
    training_manifest.write_text(json.dumps({
        **preflight_result, "status": "training_complete",
        "checkpoint": str(checkpoint), "checkpoint_name": checkpoint.name,
    }, indent=2) + "\n")
    api.upload_folder(
        repo_id=OUTPUT_REPO, repo_type="model", folder_path=selected_copy,
        path_in_repo=f"run1/selected_checkpoint/{checkpoint.name}",
        commit_message="Checkpoint selected Qwen embedding RAG adapter",
    )
    api.upload_file(
        repo_id=OUTPUT_REPO, repo_type="model", path_or_fileobj=training_manifest,
        path_in_repo="run1/training_manifest.json",
        commit_message="Checkpoint completed Qwen embedding RAG training",
    )
    inference = OUTPUT_DIR / "validation_inference.jsonl"
    run([
        "swift", "infer", "--adapters", str(checkpoint), "--use_hf", "true",
        "--infer_backend", "transformers", "--val_dataset", str(VALIDATION_DATA),
        "--val_dataset_sample", "20", "--result_path", str(inference), "--max_batch_size", "1",
        "--max_new_tokens", "220", "--temperature", "0", "--enable_thinking", "false", "--stream", "false",
    ])
    inference_rows = [
        json.loads(line) for line in inference.read_text().splitlines() if line.strip()
    ] if inference.is_file() else []
    if len(inference_rows) != 20:
        raise RuntimeError(f"Adapter reload inference produced {len(inference_rows)} records, expected 20")
    verify_reload(inference_rows, VALIDATION_DATA)
    for saved_checkpoint in checkpoints:
        sanitize_adapter(
            saved_checkpoint, MODEL_ID, title="GovInfo Qwen3.5-9B RAG adapter",
            description="LoRA checkpoint for the Lean RAG fourth experiment.",
            tags=("lora", "text-generation"),
        )
    (OUTPUT_DIR / "full_manifest.json").write_text(json.dumps({
        **preflight_result, "checkpoint": str(checkpoint), "validation_inference": str(inference),
    }, indent=2) + "\n")
    api.upload_folder(
        repo_id=OUTPUT_REPO, repo_type="model", folder_path=OUTPUT_DIR, path_in_repo="run1",
        commit_message="Upload Qwen3.5 9B Qwen-embedding RAG LoRA run",
    )
    print("QWEN_EMBEDDING_RAG_TRAINING_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN_EMBEDDING_RAG_TRAINING_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

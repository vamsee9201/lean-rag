#!/usr/bin/env python3
"""Run the Qwen3.5 9B tuned-local-hybrid LoRA compatibility and reload gate."""

from __future__ import annotations

import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi
from packaging.version import Version


MODEL_ID = "Qwen/Qwen3.5-9B"
OUTPUT_REPO = "vamsee9201/qwen35-9b-local-hybrid-rag-lora"
TRAIN_DATA = Path("/mnt/data/local_retriever/sft/train_sft.jsonl")
VALIDATION_DATA = Path("/mnt/data/local_retriever/sft/validation_sft.jsonl")
OUTPUT_DIR = Path("/tmp/qwen35-local-hybrid-smoke")


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def preflight() -> dict:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    for path in (TRAIN_DATA, VALIDATION_DATA):
        if not path.is_file():
            raise FileNotFoundError(path)
    run(["nvidia-smi"])
    versions = {package: version(package) for package in (
        "ms-swift", "torch", "transformers", "peft", "flash-attn", "flash-linear-attention"
    )}
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
    return {"versions": versions, "model": MODEL_ID, **counts}


def main() -> None:
    preflight_result = preflight()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    run([
        "swift", "sft", "--model", MODEL_ID, "--use_hf", "true",
        "--dataset", str(TRAIN_DATA), "--val_dataset", str(VALIDATION_DATA),
        "--split_dataset_ratio", "0", "--tuner_type", "lora", "--target_modules", "all-linear",
        "--freeze_vit", "true", "--freeze_aligner", "true", "--lora_rank", "8",
        "--lora_alpha", "32", "--lora_dropout", "0", "--torch_dtype", "bfloat16",
        "--per_device_train_batch_size", "1", "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "4", "--gradient_checkpointing", "true",
        "--learning_rate", "1e-4", "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.05",
        "--max_steps", "10", "--eval_steps", "5", "--save_steps", "5", "--save_total_limit", "2",
        "--logging_steps", "1", "--max_length", "4096", "--packing", "false",
        "--padding_free", "false", "--attn_impl", "flash_attn", "--add_non_thinking_prefix", "true",
        "--loss_scale", "default+ignore_empty_think", "--dataset_num_proc", "2",
        "--dataloader_num_workers", "2", "--save_only_model", "true", "--report_to", "none",
        "--seed", "20260906", "--output_dir", str(OUTPUT_DIR),
    ])
    checkpoints = sorted(OUTPUT_DIR.glob("**/checkpoint-*"), key=lambda path: path.stat().st_mtime)
    if not checkpoints or not list(checkpoints[-1].glob("adapter_model*.safetensors")):
        raise RuntimeError("Smoke training did not produce a LoRA adapter")
    inference = OUTPUT_DIR / "validation_inference.jsonl"
    run([
        "swift", "infer", "--adapters", str(checkpoints[-1]), "--use_hf", "true",
        "--infer_backend", "transformers", "--val_dataset", str(VALIDATION_DATA),
        "--val_dataset_sample", "20", "--result_path", str(inference), "--max_batch_size", "1",
        "--max_new_tokens", "220", "--temperature", "0", "--enable_thinking", "false", "--stream", "false",
    ])
    inference_rows = [line for line in inference.read_text().splitlines() if line.strip()] if inference.is_file() else []
    if len(inference_rows) != 20:
        raise RuntimeError(f"Adapter reload inference produced {len(inference_rows)} records, expected 20")
    (OUTPUT_DIR / "smoke_manifest.json").write_text(json.dumps({
        **preflight_result, "checkpoint": str(checkpoints[-1]), "validation_inference": str(inference),
    }, indent=2) + "\n")
    api = HfApi()
    api.create_repo(OUTPUT_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=OUTPUT_REPO, repo_type="model", folder_path=OUTPUT_DIR,
                      path_in_repo="smoke", commit_message="Upload hybrid LoRA smoke test")
    print("LOCAL_HYBRID_SMOKE_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"LOCAL_HYBRID_SMOKE_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

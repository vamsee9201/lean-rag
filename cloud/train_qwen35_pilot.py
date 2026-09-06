#!/usr/bin/env python3
"""Run the 50-step Qwen3.5 9B LoRA throughput pilot on Hugging Face Jobs."""

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
PILOT_REPO = "vamsee9201/qwen35-9b-rag-lora-pilot"
TRAIN_DATA = Path("/mnt/data/full/train_sft.jsonl")
VALIDATION_DATA = Path("/mnt/data/full/validation_sft.jsonl")
OUTPUT_DIR = Path("/tmp/qwen35-rag-pilot50")


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
    versions = {
        package: version(package)
        for package in ("ms-swift", "torch", "transformers", "peft", "flash-attn", "flash-linear-attention")
    }
    if Version(versions["transformers"]) < Version("5.9"):
        raise RuntimeError("Qwen3.5 requires transformers>=5.9 for correct sample boundaries")
    for module in ("torch", "transformers", "swift", "flash_attn", "fla", "causal_conv1d"):
        importlib.import_module(module)
    train_count = sum(1 for line in TRAIN_DATA.read_text().splitlines() if line.strip())
    validation_count = sum(1 for line in VALIDATION_DATA.read_text().splitlines() if line.strip())
    result = {
        "versions": versions,
        "train_records": train_count,
        "validation_records": validation_count,
        "model": MODEL_ID,
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    preflight_result = preflight()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    command = [
        "swift", "sft",
        "--model", MODEL_ID,
        "--use_hf", "true",
        "--dataset", str(TRAIN_DATA),
        "--val_dataset", str(VALIDATION_DATA),
        "--split_dataset_ratio", "0",
        "--tuner_type", "lora",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--lora_rank", "8",
        "--lora_alpha", "32",
        "--lora_dropout", "0",
        "--torch_dtype", "bfloat16",
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--gradient_checkpointing", "true",
        "--learning_rate", "1e-4",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.1",
        "--max_steps", "50",
        "--eval_steps", "25",
        "--save_steps", "25",
        "--save_total_limit", "2",
        "--logging_steps", "1",
        "--max_length", "4096",
        "--packing", "false",
        "--padding_free", "false",
        "--attn_impl", "flash_attn",
        "--add_non_thinking_prefix", "true",
        "--loss_scale", "default+ignore_empty_think",
        "--dataset_num_proc", "2",
        "--dataloader_num_workers", "2",
        "--save_only_model", "true",
        "--report_to", "none",
        "--seed", "20260906",
        "--output_dir", str(OUTPUT_DIR),
    ]
    run(command)
    checkpoints = sorted(OUTPUT_DIR.glob("**/checkpoint-*"), key=lambda path: path.stat().st_mtime)
    if not checkpoints:
        raise RuntimeError("Training completed without a checkpoint")
    checkpoint = checkpoints[-1]
    adapter_files = list(checkpoint.glob("adapter_model*.safetensors"))
    if not adapter_files:
        raise RuntimeError(f"No LoRA adapter found in {checkpoint}")

    inference_path = OUTPUT_DIR / "validation_inference.jsonl"
    run(
        [
            "swift", "infer",
            "--adapters", str(checkpoint),
            "--use_hf", "true",
            "--infer_backend", "transformers",
            "--val_dataset", str(VALIDATION_DATA),
            "--val_dataset_sample", "10",
            "--result_path", str(inference_path),
            "--max_batch_size", "1",
            "--max_new_tokens", "220",
            "--temperature", "0",
            "--enable_thinking", "false",
            "--stream", "false",
        ]
    )
    if not inference_path.is_file() or not inference_path.read_text().strip():
        raise RuntimeError("Adapter reload inference produced no records")

    manifest = {
        **preflight_result,
        "checkpoint": str(checkpoint),
        "adapter_files": [str(path) for path in adapter_files],
        "validation_inference": str(inference_path),
    }
    (OUTPUT_DIR / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    HfApi().upload_folder(
        repo_id=PILOT_REPO,
        repo_type="model",
        folder_path=OUTPUT_DIR,
        path_in_repo="pilot50",
        commit_message="Upload Qwen3.5 9B RAG LoRA 50-step pilot",
    )
    print("PILOT_TEST_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PILOT_TEST_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

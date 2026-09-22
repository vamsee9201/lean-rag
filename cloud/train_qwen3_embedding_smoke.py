#!/usr/bin/env python3
"""Run a short Qwen3-Embedding-8B LoRA compatibility and reload test."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from huggingface_hub import HfApi

from qwen3_embedding_utils import TRAINING_MAX_LENGTH, encode, load_model


DATA = Path("/mnt/data/train_swift.jsonl")
OUTPUT = Path("/tmp/qwen3-embedding-smoke")
REPO = "vamsee9201/qwen3-embedding-8b-govinfo-retriever-pilot"


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    rows = [line for line in DATA.read_text().splitlines() if line.strip()][:16]
    sample = OUTPUT / "smoke.jsonl"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample.write_text("\n".join(rows) + "\n")
    env = dict(os.environ)
    env.update({
        "INFONCE_USE_BATCH": "false",
        "INFONCE_HARD_NEGATIVES": "4",
        "INFONCE_MASK_FAKE_NEGATIVE": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    run(["nvidia-smi"])
    run([
        "swift", "sft", "--model", "Qwen/Qwen3-Embedding-8B", "--use_hf", "true",
        "--task_type", "embedding", "--model_type", "qwen3_emb",
        "--tuner_type", "lora", "--target_modules", "all-linear",
        "--dataset", str(sample), "--split_dataset_ratio", "0",
        "--loss_type", "infonce", "--label_names", "labels",
        "--torch_dtype", "bfloat16", "--max_length", str(TRAINING_MAX_LENGTH),
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "2",
        "--learning_rate", "1e-4", "--warmup_ratio", "0",
        "--max_steps", "2", "--save_steps", "2", "--logging_steps", "1",
        "--gradient_checkpointing", "true", "--dataloader_drop_last", "true",
        "--lora_rank", "8", "--lora_alpha", "32", "--lora_dropout", "0",
        "--save_only_model", "true", "--report_to", "none", "--seed", "20260918",
        "--output_dir", str(OUTPUT / "training"),
    ], env)
    checkpoints = sorted((OUTPUT / "training").glob("**/checkpoint-*"))
    if not checkpoints:
        raise RuntimeError("Smoke training produced no checkpoint")
    checkpoint = checkpoints[-1]
    merged = OUTPUT / "merged"
    run(["swift", "export", "--adapters", str(checkpoint), "--use_hf", "true", "--merge_lora", "true",
         "--output_dir", str(merged)])
    model = load_model(merged)
    vectors = encode(model, ["A short government retrieval passage.", "A different passage."])
    if np.allclose(vectors[0], vectors[1]):
        raise RuntimeError("Reloaded adapter produced identical test vectors")
    manifest = {
        "status": "complete", "records": len(rows), "steps": 2,
        "checkpoint": str(checkpoint), "reload_source": str(merged),
        "dimensions": int(vectors.shape[1]),
        "finite": bool(np.isfinite(vectors).all()),
    }
    (OUTPUT / "smoke_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(merged)
    shutil.rmtree(OUTPUT / "training" / "runs", ignore_errors=True)
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="model", folder_path=OUTPUT,
                      commit_message="Upload Qwen3 embedding LoRA smoke test")
    print("QWEN3_EMBEDDING_SMOKE_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN3_EMBEDDING_SMOKE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

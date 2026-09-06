#!/usr/bin/env python3
"""Run the selected Qwen3.5 RAG LoRA adapter on the frozen held-out benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi, snapshot_download


MODEL_REPO = "vamsee9201/qwen35-9b-rag-lora"
DATASET = Path("/mnt/data/final/qwen_inference.jsonl")
OUTPUT = Path("/tmp/qwen35-final-eval")


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    if not DATASET.is_file():
        raise FileNotFoundError(DATASET)
    run(["nvidia-smi"])
    repo_dir = Path(
        snapshot_download(
            repo_id=MODEL_REPO,
            allow_patterns="run1/v0-*/checkpoint-500/**",
            token=os.environ["HF_TOKEN"],
        )
    )
    adapters = list(repo_dir.glob("run1/v0-*/checkpoint-500/adapter_model*.safetensors"))
    if len(adapters) != 1:
        raise RuntimeError(f"Expected one selected adapter, found {len(adapters)}")
    checkpoint = adapters[0].parent
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT / "bm25_test_answers.jsonl"
    run(
        [
            "swift", "infer",
            "--adapters", str(checkpoint),
            "--use_hf", "true",
            "--infer_backend", "transformers",
            "--val_dataset", str(DATASET),
            "--val_dataset_sample", "50",
            "--result_path", str(result_path),
            "--max_length", "8192",
            "--max_batch_size", "1",
            "--max_new_tokens", "220",
            "--temperature", "0",
            "--enable_thinking", "false",
            "--stream", "false",
        ]
    )
    rows = [json.loads(line) for line in result_path.read_text().splitlines() if line.strip()]
    if len(rows) != 50:
        raise RuntimeError(f"Expected 50 inference records, found {len(rows)}")
    manifest = {
        "model_repo": MODEL_REPO,
        "checkpoint": str(checkpoint),
        "records": len(rows),
        "dataset": str(DATASET),
        "max_length": 8192,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    HfApi().upload_folder(
        repo_id=MODEL_REPO,
        repo_type="model",
        folder_path=OUTPUT,
        path_in_repo="evaluation/bm25-test",
        commit_message="Upload tuned Qwen BM25 held-out answers",
    )
    print("FINAL_INFERENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FINAL_INFERENCE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

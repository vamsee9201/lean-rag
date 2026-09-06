#!/usr/bin/env python3
"""Run the untuned Qwen3.5 9B blind judge over 50 complete matrix prompts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi


MODEL = "Qwen/Qwen3.5-9B"
DATASET = Path("/mnt/data/hybrid/judge/qwen_judge_inputs.jsonl")
OUTPUT = Path("/tmp/qwen35-hybrid-judge")


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    if sum(bool(line.strip()) for line in DATASET.read_text().splitlines()) != 50:
        raise ValueError("Qwen judge input must contain exactly 50 prompts")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = OUTPUT / "raw_scores.jsonl"
    started = time.perf_counter()
    subprocess.run([
        "swift", "infer", "--model", MODEL, "--use_hf", "true",
        "--infer_backend", "transformers", "--val_dataset", str(DATASET),
        "--val_dataset_sample", "50", "--result_path", str(result), "--max_length", "16384",
        "--max_batch_size", "1", "--max_new_tokens", "1800", "--temperature", "0",
        "--enable_thinking", "false", "--stream", "false",
    ], check=True)
    rows = [line for line in result.read_text().splitlines() if line.strip()]
    if len(rows) != 50:
        raise RuntimeError(f"Qwen judge produced {len(rows)} rows")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "model": MODEL, "records": len(rows), "elapsed_seconds": time.perf_counter() - started,
    }, indent=2) + "\n")
    HfApi().upload_folder(
        repo_id="vamsee9201/qwen35-9b-hybrid-rag-lora", repo_type="model",
        folder_path=OUTPUT, path_in_repo="evaluation/qwen-judge",
        commit_message="Upload blinded Qwen matrix judge output",
    )
    print("QWEN_JUDGE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN_JUDGE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

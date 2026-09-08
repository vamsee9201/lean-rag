#!/usr/bin/env python3
"""Run untuned Qwen3.5 9B as the blind judge for the third experiment."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi


MODEL = "Qwen/Qwen3.5-9B"
OUTPUT = Path("/tmp/qwen35-local-judge")


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    expected_prompts = int(os.environ.get("QWEN_JUDGE_EXPECTED_PROMPTS", "50"))
    max_new_tokens = int(os.environ.get("QWEN_JUDGE_MAX_NEW_TOKENS", "1800"))
    upload_subdir = os.environ.get("QWEN_JUDGE_UPLOAD_SUBDIR", "evaluation/qwen-judge")
    dataset = Path(os.environ.get(
        "QWEN_JUDGE_INPUT", "/mnt/data/local_retriever/judge/qwen_judge_inputs.jsonl"
    ))
    if sum(bool(line.strip()) for line in dataset.read_text().splitlines()) != expected_prompts:
        raise ValueError(f"Qwen judge input must contain exactly {expected_prompts} prompts")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = OUTPUT / "raw_scores.jsonl"
    started = time.perf_counter()
    subprocess.run([
        "swift", "infer", "--model", MODEL, "--use_hf", "true",
        "--infer_backend", "transformers", "--val_dataset", str(dataset),
        "--val_dataset_sample", str(expected_prompts), "--result_path", str(result),
        "--max_length", "16384",
        "--max_batch_size", "1", "--max_new_tokens", str(max_new_tokens), "--temperature", "0",
        "--enable_thinking", "false", "--stream", "false",
    ], check=True)
    rows = [line for line in result.read_text().splitlines() if line.strip()]
    if len(rows) != expected_prompts:
        raise RuntimeError(f"Qwen judge produced {len(rows)} rows")
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "model": MODEL, "records": len(rows), "elapsed_seconds": time.perf_counter() - started,
        "max_new_tokens": max_new_tokens,
    }, indent=2) + "\n")
    HfApi().upload_folder(
        repo_id="vamsee9201/qwen35-9b-local-hybrid-rag-lora", repo_type="model",
        folder_path=OUTPUT, path_in_repo=upload_subdir,
        commit_message="Upload blinded third-experiment Qwen judge output",
    )
    print("QWEN_JUDGE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN_JUDGE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

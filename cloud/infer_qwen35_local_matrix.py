#!/usr/bin/env python3
"""Run four Qwen states across the four frozen third-experiment retrievers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from huggingface_hub import HfApi, snapshot_download


BASE_MODEL = "Qwen/Qwen3.5-9B"
BM25_REPO = "vamsee9201/qwen35-9b-rag-lora"
HYBRID_REPO = "vamsee9201/qwen35-9b-hybrid-rag-lora"
LOCAL_REPO = "vamsee9201/qwen35-9b-local-hybrid-rag-lora"
DATASETS = {
    name: Path(f"/mnt/data/local_retriever/inference/{name}.jsonl") for name in (
        "bm25_top5", "vertex_hybrid_top5", "untuned_local_hybrid_top5", "tuned_local_hybrid_top5"
    )
}
OUTPUT = Path("/tmp/qwen35-local-matrix")


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def find_bm25_adapter() -> Path:
    root = Path(snapshot_download(
        repo_id=BM25_REPO, allow_patterns="run1/v0-*/checkpoint-500/**",
        token=os.environ["HF_TOKEN"],
    ))
    adapters = list(root.glob("run1/v0-*/checkpoint-500/adapter_model*.safetensors"))
    if len(adapters) != 1:
        raise RuntimeError(f"Expected one BM25 checkpoint-500 adapter, found {len(adapters)}")
    return adapters[0].parent


def find_hybrid_adapter() -> Path:
    root = Path(snapshot_download(
        repo_id=HYBRID_REPO, allow_patterns=["run1/full_manifest.json", "run1/**/checkpoint-*/**"],
        token=os.environ["HF_TOKEN"],
    ))
    manifest_path = root / "run1/full_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Hybrid repository has no completed full-run manifest")
    checkpoint_name = Path(json.loads(manifest_path.read_text())["checkpoint"]).name
    adapters = list(root.glob(f"run1/**/{checkpoint_name}/adapter_model*.safetensors"))
    if len(adapters) != 1:
        raise RuntimeError(f"Expected one selected hybrid adapter, found {len(adapters)}")
    return adapters[0].parent


def find_local_adapter() -> Path:
    root = Path(snapshot_download(
        repo_id=LOCAL_REPO, allow_patterns=["run1/full_manifest.json", "run1/**/checkpoint-*/**"],
        token=os.environ["HF_TOKEN"],
    ))
    manifest_path = root / "run1/full_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Local-hybrid repository has no completed full-run manifest")
    checkpoint_name = Path(json.loads(manifest_path.read_text())["checkpoint"]).name
    adapters = list(root.glob(f"run1/**/{checkpoint_name}/adapter_model*.safetensors"))
    if len(adapters) != 1:
        raise RuntimeError(f"Expected one selected local adapter, found {len(adapters)}")
    return adapters[0].parent


def infer(state: str, setup: str, dataset: Path, adapter: Path | None) -> tuple[Path, float]:
    path = OUTPUT / f"{state}__{setup}.jsonl"
    command = ["swift", "infer"]
    if adapter:
        command += ["--adapters", str(adapter)]
    else:
        command += ["--model", BASE_MODEL]
    command += [
        "--use_hf", "true", "--infer_backend", "transformers", "--val_dataset", str(dataset),
        "--val_dataset_sample", "50", "--result_path", str(path), "--max_length", "8192",
        "--max_batch_size", "1", "--max_new_tokens", "220", "--temperature", "0",
        "--enable_thinking", "false", "--stream", "false",
    ]
    started = time.perf_counter()
    run(command)
    elapsed = time.perf_counter() - started
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 50:
        raise RuntimeError(f"{state}/{setup} produced {len(rows)} rows")
    return path, elapsed


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    for path in DATASETS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bm25 = find_bm25_adapter()
    hybrid = find_hybrid_adapter()
    local_adapter = find_local_adapter()
    states = {"qwen-base": None, "qwen-bm25-lora": bm25, "qwen-vertex-hybrid-lora": hybrid,
              "qwen-local-hybrid-lora": local_adapter}
    outputs = {}
    for state, adapter in states.items():
        for setup, dataset in DATASETS.items():
            path, elapsed = infer(state, setup, dataset, adapter)
            outputs[f"{state}:{setup}"] = {"path": str(path), "elapsed_seconds": elapsed}
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "bm25_adapter": str(bm25), "hybrid_adapter": str(hybrid),
        "local_adapter": str(local_adapter),
        "outputs": outputs, "max_length": 8192, "max_new_tokens": 220,
    }, indent=2) + "\n")
    HfApi().upload_folder(
        repo_id=LOCAL_REPO, repo_type="model", folder_path=OUTPUT, path_in_repo="evaluation/matrix",
        commit_message="Upload third-experiment Qwen matrix answers",
    )
    print("MATRIX_INFERENCE_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"MATRIX_INFERENCE_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

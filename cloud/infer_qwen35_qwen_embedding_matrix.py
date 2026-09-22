#!/usr/bin/env python3
"""Run three Qwen states across five frozen fourth-experiment retrievers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError


BASE_MODEL = "Qwen/Qwen3.5-9B"
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
PREVIOUS_REPO = "vamsee9201/qwen35-9b-local-hybrid-rag-lora"
NEW_REPO = "vamsee9201/qwen35-9b-qwen-embedding-rag-lora"
SETUPS = (
    "bm25_top5", "vertex_hybrid_top5", "qwen3_untuned_dense_top5",
    "qwen3_untuned_hybrid_top5", "qwen3_tuned_hybrid_top5",
)
DATASETS = {
    name: Path(f"/mnt/data/qwen_embedding/inference/{name}.jsonl") for name in SETUPS
}
OUTPUT = Path("/tmp/qwen35-qwen-embedding-matrix")


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def selected_adapter(repo: str) -> Path:
    root = Path(snapshot_download(
        repo_id=repo, allow_patterns=[
            "run1/full_manifest.json", "run1/selected_checkpoint/**",
            "run1/**/checkpoint-*/**",
        ],
        token=os.environ["HF_TOKEN"],
    ))
    manifest_path = root / "run1/full_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"{repo} has no completed full-run manifest")
    checkpoint_name = Path(json.loads(manifest_path.read_text())["checkpoint"]).name
    preferred = root / "run1" / "selected_checkpoint" / checkpoint_name
    if list(preferred.glob("adapter_model*.safetensors")):
        return preferred
    adapters = list(root.glob(f"run1/**/{checkpoint_name}/adapter_model*.safetensors"))
    if len(adapters) != 1:
        raise RuntimeError(f"Expected one selected adapter in {repo}, found {len(adapters)}")
    return adapters[0].parent


def combined_dataset() -> tuple[Path, list[dict]]:
    path = OUTPUT / "combined_inputs.jsonl"
    order = []
    with path.open("w", encoding="utf-8") as output:
        for setup, dataset in DATASETS.items():
            rows = [line for line in dataset.read_text().splitlines() if line.strip()]
            if len(rows) != 50:
                raise ValueError(f"{dataset} contains {len(rows)} records, expected 50")
            for position, line in enumerate(rows):
                json.loads(line)
                output.write(line + "\n")
                order.append({"setup": setup, "position": position})
    if len(order) != 250:
        raise ValueError(f"Combined inference dataset contains {len(order)} records")
    return path, order


def infer(state: str, dataset: Path, adapter: Path | None) -> tuple[Path, float]:
    path = OUTPUT / f"{state}.jsonl"
    command = ["swift", "infer"]
    command += ["--adapters", str(adapter)] if adapter else [
        "--model", BASE_MODEL, "--model_revision", BASE_REVISION,
    ]
    command += [
        "--use_hf", "true", "--infer_backend", "transformers", "--val_dataset", str(dataset),
        "--val_dataset_sample", "250", "--result_path", str(path), "--max_length", "8192",
        "--max_batch_size", "1", "--max_new_tokens", "220", "--temperature", "0",
        "--enable_thinking", "false", "--stream", "false",
    ]
    started = time.perf_counter()
    run(command)
    elapsed = time.perf_counter() - started
    rows = [line for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 250:
        raise RuntimeError(f"{state} produced {len(rows)} rows")
    return path, elapsed


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prompts_match(path: Path, expected: list[dict]) -> bool:
    try:
        rows = read_jsonl(path)
        if len(rows) != len(expected):
            return False
        for row, source in zip(rows, expected, strict=True):
            actual_messages = row.get("messages") or []
            expected_messages = source.get("messages") or []
            actual_prompt = [
                item for item in actual_messages if item.get("role") in {"system", "user"}
            ]
            if actual_prompt != expected_messages:
                return False
            if not str(row.get("response", "")).strip():
                return False
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def reusable_output(state: str, expected: list[dict]) -> tuple[Path, float] | None:
    try:
        raw = Path(hf_hub_download(
            NEW_REPO, f"evaluation/matrix/{state}.jsonl", token=os.environ["HF_TOKEN"]
        ))
        state_manifest = Path(hf_hub_download(
            NEW_REPO, f"evaluation/matrix/{state}_manifest.json",
            token=os.environ["HF_TOKEN"],
        ))
    except EntryNotFoundError:
        return None
    if not prompts_match(raw, expected):
        return None
    metadata = json.loads(state_manifest.read_text())
    if metadata.get("state") != state or metadata.get("records") != 250:
        return None
    destination = OUTPUT / f"{state}.jsonl"
    shutil.copyfile(raw, destination)
    print(f"RESUME verified uploaded {state} responses", flush=True)
    return destination, float(metadata["elapsed_seconds"])


def upload_state(api: HfApi, state: str, path: Path, elapsed: float) -> None:
    state_manifest = OUTPUT / f"{state}_manifest.json"
    state_manifest.write_text(json.dumps({
        "state": state, "records": 250, "elapsed_seconds": elapsed,
    }, indent=2) + "\n")
    for local in (path, state_manifest):
        api.upload_file(
            repo_id=NEW_REPO, repo_type="model", path_or_fileobj=local,
            path_in_repo=f"evaluation/matrix/{local.name}",
            commit_message=f"Checkpoint {state} matrix responses",
        )


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    for path in DATASETS.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dataset, order = combined_dataset()
    expected_inputs = read_jsonl(dataset)
    previous = selected_adapter(PREVIOUS_REPO)
    new = selected_adapter(NEW_REPO)
    states = {
        "qwen-base": None,
        "qwen-previous-local-lora": previous,
        "qwen-qwen-embedding-lora": new,
    }
    api = HfApi()
    outputs = {}
    for state, adapter in states.items():
        reusable = reusable_output(state, expected_inputs)
        if reusable is None:
            path, elapsed = infer(state, dataset, adapter)
            if not prompts_match(path, expected_inputs):
                raise RuntimeError(f"{state} output prompts or responses failed verification")
            upload_state(api, state, path, elapsed)
        else:
            path, elapsed = reusable
        outputs[state] = {"path": str(path), "elapsed_seconds": elapsed, "records": 250}
    (OUTPUT / "manifest.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "previous_adapter": str(previous), "new_adapter": str(new),
        "setups": SETUPS, "dataset_order": order, "outputs": outputs,
        "model_loads": 3, "max_length": 8192, "max_new_tokens": 220,
    }, indent=2) + "\n")
    api.upload_folder(
        repo_id=NEW_REPO, repo_type="model", folder_path=OUTPUT,
        path_in_repo="evaluation/matrix",
        commit_message="Upload Qwen embedding experiment matrix answers",
    )
    print("QWEN_EMBEDDING_MATRIX_COMPLETE", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN_EMBEDDING_MATRIX_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

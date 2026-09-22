#!/usr/bin/env python3
"""Recover two durably uploaded SWIFT LoRA checkpoints without retraining."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import sys

import torch
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

from qwen3_adapter_metadata import sanitize_adapter
from qwen3_embedding_adapter_utils import (
    encode, load_engine, query_text, retrieval_metrics, sha256, tokenizer_for,
)
from qwen3_embedding_utils import MODEL_ID, positive_map, read_jsonl


DATA = Path("/mnt/data")
OUTPUT = Path("/tmp/qwen3-embedding-validation-recovery")
REPO = "vamsee9201/qwen3-embedding-8b-govinfo-retriever"
BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
FILES = {
    "training": DATA / "train_swift.jsonl",
    "train_chunks": DATA / "train_chunks.jsonl",
    "train_questions": DATA / "train_questions.jsonl",
    "validation_chunks": DATA / "validation_chunks.jsonl",
    "validation_questions": DATA / "validation_questions.jsonl",
    "test_v4_chunks": DATA / "test_v4_chunks.jsonl",
    "test_v4_questions": DATA / "test_v4_questions.jsonl",
}


def metric_key(row: dict) -> tuple:
    return (
        row["all_gold_recall_at_5"], row["mean_passage_recall_at_5"],
        row["ndcg_at_5"], row["mrr_at_5"], -row["step"],
    )


def reusable_candidate(token: str, name: str, expected: dict) -> dict | None:
    try:
        path = hf_hub_download(
            REPO, f"adapter_aware_run2/validation_candidates/{name}.json",
            token=token,
        )
    except EntryNotFoundError:
        return None
    value = json.loads(Path(path).read_text())
    if (
        value.get("checkpoint") != name
        or value.get("base_revision") != BASE_REVISION
        or value.get("input_sha256") != expected["input_sha256"]
        or value.get("adapter_weights_sha256") != expected["adapter_weights_sha256"]
    ):
        raise RuntimeError(f"Uploaded {name} validation does not match the staged input")
    return value


def evaluate(
    base_path: Path, adapter: Path, tokenizer, questions: list[dict],
    chunks: list[dict], positives: dict[str, list[str]],
) -> dict:
    engine = load_engine(base_path, adapter)
    corpus, corpus_stats = encode(engine, tokenizer, [row["text"] for row in chunks])
    queries, query_stats = encode(
        engine, tokenizer, [query_text(row["question"]) for row in questions]
    )
    metrics = retrieval_metrics(corpus, queries, questions, chunks, positives)
    del corpus, queries, engine
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "checkpoint": adapter.name,
        "step": int(adapter.name.split("-")[-1]),
        **metrics,
        "corpus_encoding": corpus_stats,
        "query_encoding": query_stats,
    }


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable")
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if sum(bool(line.strip()) for line in FILES["training"].open()) != 2560:
        raise RuntimeError("Expected 2,560 training records")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base = Path(snapshot_download(
        MODEL_ID, revision=BASE_REVISION, token=token,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "modules.json"],
    ))
    remote = Path(snapshot_download(
        REPO, token=token,
        allow_patterns=["adapter_aware_run2/candidate_adapters/*/*"],
    ))
    questions = read_jsonl(FILES["validation_questions"])
    chunks = read_jsonl(FILES["validation_chunks"])
    positives = positive_map(questions, chunks)
    tokenizer = tokenizer_for(base)
    api = HfApi()
    metrics = []
    input_sha256 = {name: sha256(path) for name, path in FILES.items()}
    for step in (80, 160):
        name = f"checkpoint-{step}"
        adapter = OUTPUT / name
        shutil.copytree(
            remote / "adapter_aware_run2" / "candidate_adapters" / name,
            adapter, dirs_exist_ok=True,
        )
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"Missing durable adapter weights for {name}")
        sanitize_adapter(adapter, MODEL_ID)
        expected = {
            "input_sha256": input_sha256,
            "adapter_weights_sha256": sha256(adapter / "adapter_model.safetensors"),
        }
        result = reusable_candidate(token, name, expected)
        if result is None:
            result = evaluate(base, adapter, tokenizer, questions, chunks, positives)
            result.update(expected)
            result["base_revision"] = BASE_REVISION
            result_path = OUTPUT / f"validation-{step}.json"
            result_path.write_text(json.dumps(result, indent=2) + "\n")
            api.upload_file(
                repo_id=REPO, repo_type="model", path_or_fileobj=result_path,
                path_in_repo=f"adapter_aware_run2/validation_candidates/{name}.json",
                commit_message=f"Persist {name} adapter-aware validation",
            )
        metrics.append(result)
        print("VALIDATED", name, json.dumps(result), flush=True)
    winner = max(metrics, key=metric_key)
    selected = OUTPUT / winner["checkpoint"]
    api.upload_folder(
        repo_id=REPO, repo_type="model", folder_path=selected,
        path_in_repo="adapter_aware_run2/selected_adapter",
        commit_message="Upload selected adapter-aware embedding checkpoint",
    )
    manifest = {
        "status": "validation_complete", "adapter_aware": True,
        "base_model": MODEL_ID, "base_revision": BASE_REVISION,
        "seed": 20260918, "training_records": 2560,
        "checkpoint_metrics": metrics,
        "selected_checkpoint": winner["checkpoint"],
        "selected_validation": winner,
        "input_sha256": input_sha256,
        "recovered_from_job": "6aadd49c51992417dfcc8310",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = OUTPUT / "validation_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    api.upload_file(
        repo_id=REPO, repo_type="model", path_or_fileobj=path,
        path_in_repo="adapter_aware_run2/validation_manifest.json",
        commit_message="Finalize recovered adapter-aware validation",
    )
    print("QWEN3_EMBEDDING_VALIDATION_RECOVERED", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN3_EMBEDDING_VALIDATION_RECOVERY_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

#!/usr/bin/env python3
"""Retrain deterministically, select adapters correctly, and build exact indexes."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download

from qwen3_adapter_metadata import sanitize_adapter
from qwen3_embedding_adapter_utils import (
    encode, load_engine, query_text, retrieval_metrics, save_index, save_query_cache,
    sha256, tokenizer_for,
)
from qwen3_embedding_utils import MODEL_ID, positive_map, read_jsonl


DATA = Path("/mnt/data")
OUTPUT = Path("/tmp/qwen3-embedding-adapter-aware")
REPO = "vamsee9201/qwen3-embedding-8b-govinfo-retriever"
BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
BASE_VALIDATION_EMBEDDINGS_SHA256 = (
    "1e9d0e558fc3c8f86cb182beab3381d07c6dcd4dc6aed76336bbb4e06339a4ca"
)
FILES = {
    "training": DATA / "train_swift.jsonl",
    "train_chunks": DATA / "train_chunks.jsonl",
    "train_questions": DATA / "train_questions.jsonl",
    "validation_chunks": DATA / "validation_chunks.jsonl",
    "validation_questions": DATA / "validation_questions.jsonl",
    "test_v4_chunks": DATA / "test_v4_chunks.jsonl",
    "test_v4_questions": DATA / "test_v4_questions.jsonl",
}


def run(command: list[str], env=None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def upload_folder(api: HfApi, folder: Path, remote: str, message: str) -> None:
    """Persist each expensive stage so a later failure can resume without retraining."""
    api.upload_folder(
        repo_id=REPO, repo_type="model", folder_path=folder,
        path_in_repo=remote, commit_message=message,
    )


def metric_key(row: dict) -> tuple:
    return (
        row["all_gold_recall_at_5"], row["mean_passage_recall_at_5"],
        row["ndcg_at_5"], row["mrr_at_5"], -row["step"],
    )


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable")
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    api = HfApi()
    rows = {name: read_jsonl(path) for name, path in FILES.items() if name != "training"}
    if sum(1 for line in FILES["training"].open() if line.strip()) != 2560:
        raise ValueError("Expected 2,560 embedding training records")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(["nvidia-smi"])
    base_path = Path(snapshot_download(
        repo_id=MODEL_ID, revision=BASE_REVISION, token=token,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "modules.json"],
    ))
    tokenizer = tokenizer_for(base_path)
    env = dict(os.environ)
    env.update({
        "INFONCE_USE_BATCH": "false", "INFONCE_HARD_NEGATIVES": "4",
        "INFONCE_MASK_FAKE_NEGATIVE": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    training_dir = OUTPUT / "training"
    run([
        "swift", "sft", "--model", str(base_path), "--task_type", "embedding",
        "--model_type", "qwen3_emb", "--tuner_type", "lora",
        "--target_modules", "all-linear", "--dataset", str(FILES["training"]),
        "--split_dataset_ratio", "0", "--loss_type", "infonce",
        "--label_names", "labels", "--torch_dtype", "bfloat16", "--max_length", "8192",
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "16",
        "--learning_rate", "1e-4", "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.05", "--weight_decay", "0.01", "--num_train_epochs", "1",
        "--save_steps", "80", "--logging_steps", "1", "--save_total_limit", "2",
        "--gradient_checkpointing", "true", "--dataloader_drop_last", "true",
        "--lora_rank", "8", "--lora_alpha", "32", "--lora_dropout", "0",
        "--save_only_model", "true", "--report_to", "none", "--seed", "20260918",
        "--output_dir", str(training_dir),
    ], env)
    checkpoints = sorted(
        training_dir.glob("**/checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if [path.name for path in checkpoints] != ["checkpoint-80", "checkpoint-160"]:
        raise RuntimeError(f"Unexpected checkpoints: {checkpoints}")
    positives = positive_map(rows["validation_questions"], rows["validation_chunks"])
    validation_texts = [row["text"] for row in rows["validation_chunks"]]
    validation_queries = [query_text(row["question"]) for row in rows["validation_questions"]]
    candidates = []
    for checkpoint in checkpoints:
        engine = load_engine(base_path, checkpoint)
        corpus, corpus_stats = encode(engine, tokenizer, validation_texts)
        queries, query_stats = encode(engine, tokenizer, validation_queries)
        metrics = retrieval_metrics(
            corpus, queries, rows["validation_questions"], rows["validation_chunks"], positives
        )
        candidates.append({
            "checkpoint": checkpoint.name, "step": int(checkpoint.name.split("-")[-1]),
            **metrics, "corpus_encoding": corpus_stats, "query_encoding": query_stats,
        })
        del corpus, queries
        del engine
        release()
    winner = max(candidates, key=metric_key)
    selected = next(path for path in checkpoints if path.name == winner["checkpoint"])
    for checkpoint in checkpoints:
        candidate_dir = OUTPUT / "candidate_adapters" / checkpoint.name
        shutil.copytree(
            checkpoint, candidate_dir, dirs_exist_ok=True
        )
        sanitize_adapter(candidate_dir, MODEL_ID)
    selected_dir = OUTPUT / "selected_adapter"
    shutil.copytree(selected, selected_dir, dirs_exist_ok=True)
    sanitize_adapter(selected_dir, MODEL_ID)
    validation_manifest = {
        "status": "validation_complete", "adapter_aware": True,
        "base_model": MODEL_ID, "base_revision": BASE_REVISION, "seed": 20260918,
        "training_records": 2560, "checkpoint_metrics": candidates,
        "selected_checkpoint": winner["checkpoint"], "selected_validation": winner,
        "input_sha256": {name: sha256(path) for name, path in FILES.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "validation_manifest.json").write_text(
        json.dumps(validation_manifest, indent=2) + "\n"
    )
    upload_folder(
        api, OUTPUT / "candidate_adapters", "adapter_aware_run2/candidate_adapters",
        "Checkpoint adapter-aware embedding candidates",
    )
    upload_folder(
        api, selected_dir, "adapter_aware_run2/selected_adapter",
        "Checkpoint selected adapter-aware embedding model",
    )
    api.upload_file(
        repo_id=REPO, repo_type="model", path_or_fileobj=OUTPUT / "validation_manifest.json",
        path_in_repo="adapter_aware_run2/validation_manifest.json",
        commit_message="Checkpoint adapter-aware validation decision",
    )
    engine = load_engine(base_path, selected_dir)
    manifests = {}
    query_manifests = {}
    for split in ("train", "validation", "test_v4"):
        chunks = rows[f"{split}_chunks"]
        questions = rows[f"{split}_questions"]
        corpus, corpus_stats = encode(engine, tokenizer, [row["text"] for row in chunks])
        query_vectors, query_stats = encode(
            engine, tokenizer, [query_text(row["question"]) for row in questions]
        )
        manifests[split] = save_index(
            corpus, chunks, OUTPUT / "indexes" / split, split, REPO, corpus_stats
        )
        query_manifests[split] = save_query_cache(
            query_vectors, questions, OUTPUT / "query_embeddings" / f"{split}.jsonl",
            REPO, query_stats,
        )
        upload_folder(
            api, OUTPUT / "indexes" / split,
            f"adapter_aware_run2/indexes/{split}",
            f"Checkpoint adapter-aware {split} index",
        )
        api.upload_file(
            repo_id=REPO, repo_type="model",
            path_or_fileobj=OUTPUT / "query_embeddings" / f"{split}.jsonl",
            path_in_repo=f"adapter_aware_run2/query_embeddings/{split}.jsonl",
            commit_message=f"Checkpoint adapter-aware {split} query vectors",
        )
        api.upload_file(
            repo_id=REPO, repo_type="model",
            path_or_fileobj=OUTPUT / "query_embeddings" / f"{split}_manifest.json",
            path_in_repo=f"adapter_aware_run2/query_embeddings/{split}_manifest.json",
            commit_message=f"Checkpoint adapter-aware {split} query manifest",
        )
        del corpus, query_vectors
    del engine
    release()
    if manifests["validation"]["embeddings_sha256"] == BASE_VALIDATION_EMBEDDINGS_SHA256:
        raise RuntimeError("Adapter-aware validation index is identical to the base index")
    manifest = {
        "status": "complete", "adapter_aware": True, "base_model": MODEL_ID,
        "base_revision": BASE_REVISION, "seed": 20260918,
        "training_records": 2560, "checkpoint_metrics": candidates,
        "selected_checkpoint": winner["checkpoint"], "selected_validation": winner,
        "indexes": manifests, "query_caches": query_manifests,
        "invalidated_artifacts": [
            "root training_manifest checkpoint_metrics",
            "root indexes/*", "root query_embeddings/*",
        ],
        "invalidated_reason": "BF16 merge rounded the small LoRA deltas away; direct adapter inference is required.",
        "input_sha256": {name: sha256(path) for name, path in FILES.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(training_dir)
    upload_folder(
        api, OUTPUT, "adapter_aware_run2",
        "Finalize adapter-aware Qwen3 embedding evaluation and indexes",
    )
    print("QWEN3_ADAPTER_AWARE_RUN_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN3_ADAPTER_AWARE_RUN_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

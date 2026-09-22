#!/usr/bin/env python3
"""Fine-tune Qwen3-Embedding-8B and build reproducible GovInfo indexes."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from huggingface_hub import HfApi

from qwen3_embedding_utils import (
    DIMENSIONS, ENCODING_MAX_LENGTH, MODEL_ID, QUERY_INSTRUCTION, TRAINING_MAX_LENGTH,
    audit_lengths, load_model, positive_map, read_jsonl, retrieval_metrics, save_index,
    save_query_cache, sha256,
)


DATA = Path("/mnt/data")
OUTPUT = Path("/tmp/qwen3-embedding-govinfo")
REPO = "vamsee9201/qwen3-embedding-8b-govinfo-retriever"
FILES = {
    "training": DATA / "train_swift.jsonl",
    "train_chunks": DATA / "train_chunks.jsonl",
    "train_questions": DATA / "train_questions.jsonl",
    "validation_chunks": DATA / "validation_chunks.jsonl",
    "validation_questions": DATA / "validation_questions.jsonl",
    "test_v4_chunks": DATA / "test_v4_chunks.jsonl",
    "test_v4_questions": DATA / "test_v4_questions.jsonl",
}


def run(command: list[str], env: dict[str, str] | None = None) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True, env=env)


def metric_key(row: dict) -> tuple:
    return (
        row["all_gold_recall_at_5"], row["mean_passage_recall_at_5"],
        row["ndcg_at_5"], row["mrr_at_5"], -row["step"],
    )


def release() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(["nvidia-smi"])
    rows = {name: read_jsonl(path) for name, path in FILES.items() if name != "training"}
    counts = {name: len(value) for name, value in rows.items()}
    training_count = sum(bool(line.strip()) for line in FILES["training"].read_text().splitlines())
    if training_count != 2560 or counts["validation_questions"] != 100:
        raise ValueError(f"Unexpected input counts: training={training_count}, rows={counts}")
    positives = positive_map(rows["validation_questions"], rows["validation_chunks"])

    base = load_model()
    training_rows = read_jsonl(FILES["training"])
    training_texts = []
    for row in training_rows:
        training_texts.append(row["messages"][0]["content"])
        training_texts.extend(group[0]["content"] for group in row["positive_messages"])
        training_texts.extend(group[0]["content"] for group in row["negative_messages"])
    training_length_audit = audit_lengths(
        base, training_texts, "embedding training", TRAINING_MAX_LENGTH
    )
    base_metrics = retrieval_metrics(
        base, rows["validation_questions"], rows["validation_chunks"], positives
    )
    base_indexes = {}
    for split in ("validation", "test_v4"):
        base_indexes[split] = save_index(
            base, rows[f"{split}_chunks"], OUTPUT / f"base_indexes/{split}", split, MODEL_ID
        )
    base_query_caches = {
        split: save_query_cache(
            base, rows[f"{split}_questions"], OUTPUT / f"base_query_embeddings/{split}.jsonl",
            MODEL_ID,
        )
        for split in ("validation", "test_v4")
    }
    del base
    release()

    env = dict(os.environ)
    env.update({
        "INFONCE_USE_BATCH": "false",
        "INFONCE_HARD_NEGATIVES": "4",
        "INFONCE_MASK_FAKE_NEGATIVE": "true",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    training_dir = OUTPUT / "training"
    run([
        "swift", "sft", "--model", MODEL_ID, "--use_hf", "true",
        "--task_type", "embedding", "--model_type", "qwen3_emb",
        "--tuner_type", "lora", "--target_modules", "all-linear",
        "--dataset", str(FILES["training"]), "--split_dataset_ratio", "0",
        "--loss_type", "infonce", "--label_names", "labels",
        "--torch_dtype", "bfloat16", "--max_length", str(TRAINING_MAX_LENGTH),
        "--per_device_train_batch_size", "1", "--gradient_accumulation_steps", "16",
        "--learning_rate", "1e-4", "--lr_scheduler_type", "cosine", "--warmup_ratio", "0.05",
        "--weight_decay", "0.01",
        "--num_train_epochs", "1", "--save_steps", "80", "--logging_steps", "1",
        "--save_total_limit", "2", "--gradient_checkpointing", "true",
        "--dataloader_drop_last", "true", "--lora_rank", "8", "--lora_alpha", "32",
        "--lora_dropout", "0", "--save_only_model", "true", "--report_to", "none",
        "--seed", "20260918", "--output_dir", str(training_dir),
    ], env)
    checkpoints = sorted(
        training_dir.glob("**/checkpoint-*"), key=lambda path: int(path.name.split("-")[-1])
    )
    if not checkpoints:
        raise RuntimeError("Training produced no checkpoints")
    checkpoint_metrics = []
    for checkpoint in checkpoints:
        step = int(checkpoint.name.split("-")[-1])
        merged = OUTPUT / f"merged-{step}"
        run(["swift", "export", "--adapters", str(checkpoint), "--use_hf", "true", "--merge_lora", "true",
             "--output_dir", str(merged)])
        model = load_model(merged)
        metrics = retrieval_metrics(
            model, rows["validation_questions"], rows["validation_chunks"], positives
        )
        checkpoint_metrics.append({"checkpoint": checkpoint.name, "step": step, **metrics})
        del model
        release()
        shutil.rmtree(merged)
    winner = max(checkpoint_metrics, key=metric_key)
    selected_source = next(path for path in checkpoints if path.name == winner["checkpoint"])
    selected_dir = OUTPUT / "selected_adapter"
    shutil.copytree(selected_source, selected_dir, dirs_exist_ok=True)
    selected_merged = OUTPUT / "selected_merged"
    run(["swift", "export", "--adapters", str(selected_dir), "--use_hf", "true", "--merge_lora", "true",
         "--output_dir", str(selected_merged)])
    selected = load_model(selected_merged)
    tuned_indexes = {}
    for split in ("train", "validation", "test_v4"):
        tuned_indexes[split] = save_index(
            selected, rows[f"{split}_chunks"], OUTPUT / f"indexes/{split}", split, REPO
        )
    tuned_query_caches = {
        split: save_query_cache(
            selected, rows[f"{split}_questions"], OUTPUT / f"query_embeddings/{split}.jsonl",
            REPO,
        )
        for split in ("train", "validation", "test_v4")
    }
    reload_vectors = selected.encode(
        ["Instruct: " + QUERY_INSTRUCTION + "\nQuery:What is the report date?"],
        normalize_embeddings=True, convert_to_numpy=True,
    )
    if reload_vectors.shape != (1, DIMENSIONS):
        raise RuntimeError("Selected adapter reload check failed")
    del selected
    release()
    shutil.rmtree(selected_merged)

    manifest = {
        "status": "complete",
        "base_model": MODEL_ID,
        "base_revision": HfApi().model_info(MODEL_ID).sha,
        "output_repo": REPO,
        "seed": 20260918,
        "query_instruction": QUERY_INSTRUCTION,
        "dimensions": DIMENSIONS,
        "training_maximum_length": TRAINING_MAX_LENGTH,
        "encoding_maximum_length": ENCODING_MAX_LENGTH,
        "training_records": training_count,
        "training_length_audit": training_length_audit,
        "hard_negatives": 4,
        "loss": "InfoNCE",
        "in_batch_negatives": False,
        "mask_false_negatives": True,
        "train_type": "LoRA",
        "lora_rank": 8,
        "lora_alpha": 32,
        "target_modules": "all-linear",
        "epochs": 1,
        "physical_batch_size": 1,
        "gradient_accumulation": 16,
        "learning_rate": 1e-4,
        "warmup_ratio": 0.05,
        "base_validation": base_metrics,
        "checkpoint_metrics": checkpoint_metrics,
        "selected_checkpoint": winner["checkpoint"],
        "selected_validation": winner,
        "base_indexes": base_indexes,
        "base_query_caches": base_query_caches,
        "tuned_indexes": tuned_indexes,
        "tuned_query_caches": tuned_query_caches,
        "input_counts": counts,
        "input_sha256": {name: sha256(path) for name, path in FILES.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.rmtree(training_dir)
    api = HfApi()
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=REPO, repo_type="model", folder_path=OUTPUT,
                      commit_message="Upload Qwen3 GovInfo retriever LoRA and exact indexes")
    print("QWEN3_EMBEDDING_TRAINING_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN3_EMBEDDING_TRAINING_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

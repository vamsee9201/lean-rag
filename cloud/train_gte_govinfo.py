#!/usr/bin/env python3
"""Fine-tune GTE ModernBERT for GovInfo retrieval and publish the run."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from datasets import Dataset
from huggingface_hub import HfApi
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers, SentenceTransformerTrainingArguments

from retriever_training_utils import resolve_positives, valid_negative


MODEL_ID = "Alibaba-NLP/gte-modernbert-base"
OUTPUT_REPO = "vamsee9201/gte-modernbert-govinfo-retriever"
DATA = Path("/mnt/data/local_retriever/embedding_training")
OUTPUT = Path("/tmp/gte-govinfo")
SEED = 20260907


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 96) -> np.ndarray:
    return model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    ).astype(np.float32)


def audit_truncation(model: SentenceTransformer, rows: list[dict], label: str) -> dict:
    maximum = 0
    truncated = []
    for start in range(0, len(rows), 256):
        batch = rows[start : start + 256]
        encoded = model.tokenizer(
            [row["text"] for row in batch], add_special_tokens=True, truncation=False,
            padding=False, return_length=True,
        )
        lengths = encoded["length"]
        maximum = max(maximum, max(lengths, default=0))
        truncated.extend(
            {"split": label, "chunk_id": row["chunk_id"], "source_tokens": length,
             "encoded_tokens": model.max_seq_length}
            for row, length in zip(batch, lengths, strict=True)
            if length > model.max_seq_length
        )
    return {"maximum_source_tokens": maximum, "truncated_chunks": len(truncated),
            "total_chunks": len(rows), "records": truncated}


def top_indices(corpus: np.ndarray, queries: np.ndarray, top_k: int) -> list[list[int]]:
    results = []
    for start in range(0, len(queries), 64):
        scores = corpus @ queries[start : start + 64].T
        for column in range(scores.shape[1]):
            values = scores[:, column]
            indices = np.argpartition(values, -top_k)[-top_k:]
            results.append(sorted(indices.tolist(), key=lambda i: (-float(values[i]), i)))
    return results


def build_training_records(
    questions: list[dict], chunks: list[dict], candidate_rows: list[dict], base_model: SentenceTransformer
) -> tuple[list[dict], dict[str, list[str]], list[dict]]:
    by_id = {row["chunk_id"]: row for row in chunks}
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in chunks:
        by_page[(row["document_id"], row["page_start"])].append(row)
    positives = {q["question_id"]: resolve_positives(q, by_page) for q in questions}
    question_by_id = {q["question_id"]: q for q in questions}
    ranked = {row["question_id"]: row["candidates"] for row in candidate_rows}

    corpus_vectors = encode(base_model, [row["text"] for row in chunks])
    query_vectors = encode(base_model, [q["question"] for q in questions])
    local_ranked = top_indices(corpus_vectors, query_vectors, 50)

    records = []
    mining_audit = []
    for q_index, question in enumerate(questions):
        qid = question["question_id"]
        positive_ids = set(positives[qid])
        candidates = ranked[qid]
        pools = {
            "bm25": [x["chunk_id"] for x in sorted(candidates, key=lambda x: (x["bm25_rank"] is None, x["bm25_rank"] or 10**9, x["chunk_id"])) if x["bm25_rank"] is not None and x["bm25_rank"] <= 50],
            "vertex": [x["chunk_id"] for x in sorted(candidates, key=lambda x: (x["dense_rank"] is None, x["dense_rank"] or 10**9, x["chunk_id"])) if x["dense_rank"] is not None and x["dense_rank"] <= 50],
            "local": [chunks[i]["chunk_id"] for i in local_ranked[q_index]],
            "bm25_extended": [x["chunk_id"] for x in sorted(candidates, key=lambda x: (x["bm25_rank"] is None, x["bm25_rank"] or 10**9, x["chunk_id"])) if x["bm25_rank"] is not None and 50 < x["bm25_rank"] <= 100],
        }
        selected = []
        selected_sources = []
        for source in ("bm25", "bm25", "vertex", "local"):
            choice = next((cid for cid in pools[source] if cid not in selected and valid_negative(by_id[cid], question, positive_ids)), None)
            actual_source = source
            if choice is None:
                for fallback in ("bm25", "vertex", "local"):
                    choice = next((cid for cid in pools[fallback] if cid not in selected and valid_negative(by_id[cid], question, positive_ids)), None)
                    if choice:
                        actual_source = fallback
                        break
            if choice is None:
                choice = next((
                    cid for cid in pools["bm25_extended"]
                    if cid not in selected and valid_negative(by_id[cid], question, positive_ids)
                ), None)
                if choice:
                    actual_source = "bm25_extended"
            if choice is None:
                raise ValueError(f"Could not mine four safe negatives for {qid}")
            selected.append(choice)
            selected_sources.append(actual_source)
        candidate_metadata = {item["chunk_id"]: item for item in candidates}
        mining_audit.append({
            "question_id": qid, "positive_chunk_ids": sorted(positive_ids),
            "negative_chunk_ids": selected, "negative_sources": selected_sources,
            "bm25_beyond_top_50_fallback": "bm25_extended" in selected_sources,
            "negative_ranks": [candidate_metadata.get(cid, {}) for cid in selected],
        })
        for positive_id in positives[qid]:
            records.append({
                "query": question["question"],
                "positive": by_id[positive_id]["text"],
                "negative_1": by_id[selected[0]]["text"],
                "negative_2": by_id[selected[1]]["text"],
                "negative_3": by_id[selected[2]]["text"],
                "negative_4": by_id[selected[3]]["text"],
            })
    return records, positives, mining_audit


def retrieval_metrics(
    model: SentenceTransformer, questions: list[dict], chunks: list[dict], positives: dict[str, list[str]]
) -> dict:
    corpus = encode(model, [row["text"] for row in chunks])
    queries = encode(model, [row["question"] for row in questions])
    ranked = top_indices(corpus, queries, 5)
    chunk_ids = [row["chunk_id"] for row in chunks]
    all_gold = passage_recall = reciprocal = ndcg = 0.0
    for question, indices in zip(questions, ranked):
        returned = [chunk_ids[i] for i in indices]
        gold = set(positives[question["question_id"]])
        hits = [index + 1 for index, cid in enumerate(returned) if cid in gold]
        all_gold += float(gold.issubset(returned))
        passage_recall += len(gold & set(returned)) / len(gold)
        reciprocal += 1 / min(hits) if hits else 0.0
        dcg = sum(1 / np.log2(rank + 1) for rank in hits)
        ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(5, len(gold)) + 1))
        ndcg += dcg / ideal
    count = len(questions)
    return {
        "all_gold_recall_at_5": all_gold / count,
        "mean_passage_recall_at_5": passage_recall / count,
        "mrr_at_5": reciprocal / count,
        "ndcg_at_5": ndcg / count,
        "questions": count,
    }


def save_index(
    model: SentenceTransformer, chunks: list[dict], directory: Path, split: str, model_label: str
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    ids = [row["chunk_id"] for row in chunks]
    started = time.perf_counter()
    vectors = encode(model, [row["text"] for row in chunks])
    elapsed = time.perf_counter() - started
    if vectors.shape != (len(chunks), 768) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid {split} embeddings: {vectors.shape}")
    np.save(directory / "embeddings.npy", vectors)
    (directory / "chunk_ids.json").write_text(json.dumps(ids) + "\n")
    manifest = {
        "complete": True, "split": split, "model": model_label,
        "base_model": MODEL_ID, "dimensions": 768, "dtype": "float32",
        "chunks": len(chunks), "normalized": True, "indexing_seconds": elapsed,
        "chunk_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "embeddings_sha256": sha256(directory / "embeddings.npy"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN secret is unavailable")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = {name: DATA / f"{name}.jsonl" for name in (
        "train_chunks", "validation_chunks", "train_questions", "validation_questions",
        "train_candidates", "validation_candidates", "test_v3_chunks",
    )}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    base = SentenceTransformer(MODEL_ID, trust_remote_code=True)
    base.max_seq_length = 1024
    truncation_audits = {
        name: audit_truncation(base, rows[name], name)
        for name in ("train_chunks", "validation_chunks", "test_v3_chunks")
    }
    with (OUTPUT / "truncation_audit.jsonl").open("w", encoding="utf-8") as stream:
        for audit in truncation_audits.values():
            for record in audit.pop("records"):
                stream.write(json.dumps(record) + "\n")
    save_index(base, rows["validation_chunks"], OUTPUT / "base_indexes/validation", "validation", MODEL_ID)
    save_index(base, rows["test_v3_chunks"], OUTPUT / "base_indexes/test_v3", "test_v3", MODEL_ID)
    training, train_positives, mining_audit = build_training_records(
        rows["train_questions"], rows["train_chunks"], rows["train_candidates"], base
    )
    training_path = OUTPUT / "mined_training_records.jsonl"
    with training_path.open("w", encoding="utf-8") as stream:
        for row in training:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    mining_audit_path = OUTPUT / "hard_negative_audit.jsonl"
    with mining_audit_path.open("w", encoding="utf-8") as stream:
        for row in mining_audit:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_page = defaultdict(list)
    for row in rows["validation_chunks"]:
        by_page[(row["document_id"], row["page_start"])].append(row)
    validation_positives = {
        q["question_id"]: resolve_positives(q, by_page) for q in rows["validation_questions"]
    }
    base_metrics = retrieval_metrics(base, rows["validation_questions"], rows["validation_chunks"], validation_positives)
    dataset = Dataset.from_list(training)
    training_dir = OUTPUT / "training"
    args = SentenceTransformerTrainingArguments(
        output_dir=str(training_dir), num_train_epochs=3,
        per_device_train_batch_size=16, gradient_accumulation_steps=4,
        learning_rate=2e-5, weight_decay=0.01, warmup_ratio=0.10,
        bf16=True, batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="epoch", save_total_limit=3, logging_steps=5,
        report_to="none", seed=SEED, data_seed=SEED,
    )
    trainer = SentenceTransformerTrainer(
        model=base, args=args, train_dataset=dataset,
        loss=CachedMultipleNegativesRankingLoss(base),
    )
    trainer.train()
    checkpoint_metrics = []
    for checkpoint in sorted(training_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        model = SentenceTransformer(str(checkpoint), trust_remote_code=True)
        model.max_seq_length = 1024
        metrics = retrieval_metrics(model, rows["validation_questions"], rows["validation_chunks"], validation_positives)
        checkpoint_metrics.append({"checkpoint": checkpoint.name, **metrics})
    if not checkpoint_metrics:
        raise RuntimeError("Training produced no checkpoints")
    winner = max(checkpoint_metrics, key=lambda x: (
        x["all_gold_recall_at_5"], x["mean_passage_recall_at_5"],
        x["ndcg_at_5"], x["mrr_at_5"], -int(x["checkpoint"].split("-")[-1]),
    ))
    selected_source = training_dir / winner["checkpoint"]
    selected_dir = OUTPUT / "selected_model"
    shutil.copytree(selected_source, selected_dir, dirs_exist_ok=True)
    selected = SentenceTransformer(str(selected_dir), trust_remote_code=True)
    selected.max_seq_length = 1024
    save_index(selected, rows["train_chunks"], OUTPUT / "indexes/train", "train", OUTPUT_REPO)
    save_index(selected, rows["validation_chunks"], OUTPUT / "indexes/validation", "validation", OUTPUT_REPO)
    save_index(selected, rows["test_v3_chunks"], OUTPUT / "indexes/test_v3", "test_v3", OUTPUT_REPO)
    manifest = {
        "status": "complete", "base_model": MODEL_ID, "output_repo": OUTPUT_REPO,
        "base_revision": HfApi().model_info(MODEL_ID).sha,
        "seed": SEED, "training_records": len(training), "epochs": 3,
        "max_sequence_length": 1024,
        "truncation_policy": "explicit tokenizer truncation at the frozen 1,024-token limit",
        "physical_batch_size": 16,
        "gradient_accumulation": 4, "effective_batch_size": 64,
        "learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.10,
        "loss": "CachedMultipleNegativesRankingLoss", "base_validation": base_metrics,
        "truncation_audit": truncation_audits,
        "mined_training_sha256": sha256(training_path),
        "hard_negative_audit_sha256": sha256(mining_audit_path),
        "bm25_beyond_top_50_fallback_questions": sum(
            row["bm25_beyond_top_50_fallback"] for row in mining_audit
        ),
        "checkpoint_metrics": checkpoint_metrics, "selected_checkpoint": winner["checkpoint"],
        "selected_validation": winner, "created_at": datetime.now(timezone.utc).isoformat(),
        "input_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    (OUTPUT / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # The selected checkpoint and its metrics are retained. Removing the other
    # checkpoint copies reduces upload time and storage without losing results.
    shutil.rmtree(training_dir)
    api = HfApi()
    api.create_repo(OUTPUT_REPO, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(repo_id=OUTPUT_REPO, repo_type="model", folder_path=OUTPUT,
                      commit_message="Upload GovInfo embedding fine-tune and indexes")
    print("GTE_GOVINFO_TRAINING_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GTE_GOVINFO_TRAINING_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise

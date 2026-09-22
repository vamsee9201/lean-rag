#!/usr/bin/env python3
"""Summarize the sealed fourth-experiment retrieval results and paired gain."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/qwen_embedding_experiment"
RETRIEVAL = EXP / "retrieval"
SEED = 20260918
RESAMPLES = 10_000
FILES = {
    "bm25": "test_bm25.jsonl",
    "vertex_hybrid": "test_vertex_hybrid.jsonl",
    "qwen3_untuned_dense": "test_qwen3_untuned_dense.jsonl",
    "qwen3_untuned_hybrid": "test_qwen3_untuned_hybrid.jsonl",
    "qwen3_tuned_hybrid": "test_qwen3_tuned_hybrid.jsonl",
}
METRICS = ("all_gold_found", "gold_passage_recall", "reciprocal_rank", "ndcg")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def paired_bootstrap(left: dict, right: dict, question_ids: list[str]) -> dict:
    rng = np.random.default_rng(SEED)
    output = {}
    draws = rng.integers(0, len(question_ids), size=(RESAMPLES, len(question_ids)))
    for metric in METRICS:
        differences = np.asarray([
            float(left[qid]["metrics"][metric]) - float(right[qid]["metrics"][metric])
            for qid in question_ids
        ])
        samples = differences[draws].mean(axis=1)
        output[metric] = {
            "difference": float(differences.mean()),
            "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        }
    return output


def main() -> None:
    questions_path = EXP / "benchmark/questions.jsonl"
    frozen_path = RETRIEVAL / "frozen_qwen3_configs.json"
    questions = read_jsonl(questions_path)
    question_by_id = {row["question_id"]: row for row in questions}
    question_ids = sorted(row["question_id"] for row in questions)
    if len(question_ids) != 50 or len(set(question_ids)) != 50:
        raise ValueError("Expected 50 unique sealed test questions")
    frozen = json.loads(frozen_path.read_text())
    recorded_hash = frozen.pop("sha256")
    actual_hash = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if recorded_hash != actual_hash:
        raise ValueError("Frozen retrieval configuration checksum mismatch")
    if frozen["sealed_test_questions_sha256"] != hashlib.sha256(questions_path.read_bytes()).hexdigest():
        raise ValueError("Sealed question checksum changed")

    by_setup = {}
    table = {}
    for setup, filename in FILES.items():
        rows = read_jsonl(RETRIEVAL / filename)
        keyed = {row["question_id"]: row for row in rows}
        if len(rows) != 50 or set(keyed) != set(question_ids):
            raise ValueError(f"Incomplete or duplicate retrieval records: {filename}")
        if any(len(row["results"]) != 5 for row in rows):
            raise ValueError(f"Retriever did not return five passages: {filename}")
        by_setup[setup] = keyed
        table[setup] = {
            "all_gold_recall_at_5": float(np.mean([
                row["metrics"]["all_gold_found"] for row in rows
            ])),
            "mean_gold_passage_recall_at_5": float(np.mean([
                row["metrics"]["gold_passage_recall"] for row in rows
            ])),
            "mrr": float(np.mean([row["metrics"]["reciprocal_rank"] for row in rows])),
            "ndcg_at_5": float(np.mean([row["metrics"]["ndcg"] for row in rows])),
            "cross_document_all_gold_recall_at_5": float(np.mean([
                row["metrics"]["all_gold_found"] for row in rows
                if question_by_id[row["question_id"]]["category"] == "cross_document"
            ])),
        }
    tuned_summary = json.loads((RETRIEVAL / "test_qwen3_tuned_hybrid_summary.json").read_text())
    table["qwen3_tuned_dense"] = {
        "all_gold_recall_at_5": tuned_summary["retriever_metrics"]["local_dense"]["all_gold_recall"],
        "mean_gold_passage_recall_at_5": tuned_summary["retriever_metrics"]["local_dense"]["mean_gold_passage_recall"],
        "mrr": tuned_summary["retriever_metrics"]["local_dense"]["mrr"],
        "ndcg_at_5": tuned_summary["retriever_metrics"]["local_dense"]["mean_ndcg"],
        "cross_document_all_gold_recall_at_5": tuned_summary["retriever_metrics"]["local_dense"]["cross_document_all_gold_recall"],
    }
    output = {
        "questions": 50,
        "frozen_config_sha256": recorded_hash,
        "sealed_questions_sha256": frozen["sealed_test_questions_sha256"],
        "table": table,
        "paired_tuned_minus_untuned_hybrid": paired_bootstrap(
            by_setup["qwen3_tuned_hybrid"], by_setup["qwen3_untuned_hybrid"], question_ids
        ),
        "paired_bootstrap_samples": RESAMPLES,
        "paired_bootstrap_seed": SEED,
    }
    path = RETRIEVAL / "sealed_test_report.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

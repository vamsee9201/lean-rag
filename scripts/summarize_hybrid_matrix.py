#!/usr/bin/env python3
"""Summarize dual-judge BM25/Vertex-hybrid results with paired uncertainty."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


SEED = 20260906
RESAMPLES = 10_000
MARGIN = 0.05
MODELS = (
    "google/gemini-3.8-flash",
    "qwen/qwen3.5-9b-base",
    "qwen/qwen3.5-9b-bm25-lora-step500",
    "qwen/qwen3.5-9b-vertex-hybrid-lora",
)
SETUPS = ("bm25_top5", "vertex_hybrid_top5")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def interval(values: list[float]) -> dict:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(SEED)
    boot = rng.choice(data, size=(RESAMPLES, len(data)), replace=True).mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "one_sided_95_lower": float(np.percentile(boot, 5)),
        "n": len(data),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--gemini-judge", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    automatic = {
        (r["model"], r["setup"], r["question_id"]): r for r in read_jsonl(args.automatic)
    }
    judge_files = [read_jsonl(args.gemini_judge), read_jsonl(args.qwen_judge)]
    judges = []
    for rows in judge_files:
        judges.append({(r["model"], r["setup"], r["question_id"]): r for r in rows})
    keys = sorted(set(automatic) & set(judges[0]) & set(judges[1]))
    expected = {
        (model, setup, question_id)
        for model in MODELS for setup in SETUPS
        for question_id in {key[2] for key in automatic}
    }
    if set(automatic) != expected or set(judges[0]) != expected or set(judges[1]) != expected:
        raise ValueError(
            f"Incomplete 2x4 matrix: automatic={len(automatic)}, "
            f"gemini_judge={len(judges[0])}, qwen_judge={len(judges[1])}, expected={len(expected)}"
        )
    rows = []
    for key in keys:
        row = dict(automatic[key])
        scores = [judge[key]["answer_score"] / 2 for judge in judges]
        citation_scores = [judge[key]["citation_score"] for judge in judges]
        row["judge_accuracy"] = sum(scores) / len(scores)
        row["judge_citation_accuracy"] = sum(citation_scores) / len(citation_scores)
        row["gemini_judge_accuracy"] = scores[0]
        row["qwen_judge_accuracy"] = scores[1]
        row["judge_disagreement"] = abs(scores[0] - scores[1])
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["setup"])].append(row)
    aggregates = {}
    categories = {}
    for (model, setup), records in grouped.items():
        aggregates.setdefault(model, {})[setup] = {
            "primary_dual_judge": interval([r["judge_accuracy"] for r in records]),
            "gemini_judge": interval([r["gemini_judge_accuracy"] for r in records]),
            "qwen_judge": interval([r["qwen_judge_accuracy"] for r in records]),
            "citation_judge": interval([r["judge_citation_accuracy"] for r in records]),
            "citation_recall": interval([r["citation_recall"] for r in records]),
            "token_f1": interval([r["token_f1"] for r in records]),
            "judge_mean_absolute_disagreement": float(np.mean([r["judge_disagreement"] for r in records])),
            "mean_elapsed_seconds": float(np.mean([
                r["elapsed_seconds"] for r in records if r["elapsed_seconds"] is not None
            ])),
        }
        for category in sorted({r["category"] for r in records}):
            subset = [r for r in records if r["category"] == category]
            categories.setdefault(model, {}).setdefault(setup, {})[category] = {
                "primary_dual_judge": interval([r["judge_accuracy"] for r in subset]),
                "citation_recall": interval([r["citation_recall"] for r in subset]),
                "token_f1": interval([r["token_f1"] for r in subset]),
            }
    lookup = {(r["model"], r["setup"], r["question_id"]): r["judge_accuracy"] for r in rows}
    questions = sorted({r["question_id"] for r in rows})
    comparisons = [
        ("hybrid_qwen_minus_gemini", "qwen/qwen3.5-9b-vertex-hybrid-lora", "google/gemini-3.8-flash", "vertex_hybrid_top5"),
        ("bm25_qwen_minus_gemini", "qwen/qwen3.5-9b-bm25-lora-step500", "google/gemini-3.8-flash", "bm25_top5"),
    ]
    paired = {}
    for name, left, right, setup in comparisons:
        values = [
            lookup[(left, setup, qid)] - lookup[(right, setup, qid)]
            for qid in questions
            if (left, setup, qid) in lookup and (right, setup, qid) in lookup
        ]
        result = interval(values)
        result["noninferiority_margin"] = -MARGIN
        result["noninferior"] = (
            result["one_sided_95_lower"] is not None and result["one_sided_95_lower"] > -MARGIN
        )
        paired[name] = result
    models = sorted({r["model"] for r in rows})
    for model in models:
        values = [
            lookup[(model, "vertex_hybrid_top5", qid)] - lookup[(model, "bm25_top5", qid)]
            for qid in questions
            if (model, "vertex_hybrid_top5", qid) in lookup and (model, "bm25_top5", qid) in lookup
        ]
        paired[f"hybrid_minus_bm25:{model}"] = interval(values)
    for setup in SETUPS:
        left = "qwen/qwen3.5-9b-vertex-hybrid-lora"
        right = "qwen/qwen3.5-9b-bm25-lora-step500"
        values = [lookup[(left, setup, qid)] - lookup[(right, setup, qid)] for qid in questions]
        paired[f"hybrid_trained_minus_bm25_trained:{setup}"] = interval(values)
    result = {
        "scored_answers": len(rows), "bootstrap_resamples": RESAMPLES,
        "noninferiority_margin": MARGIN, "aggregates": aggregates,
        "category_results": categories, "paired": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

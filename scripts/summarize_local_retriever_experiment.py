#!/usr/bin/env python3
"""Summarize the 4x5 local-versus-cloud experiment with paired uncertainty."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


SEED = 20260907
RESAMPLES = 10_000
MARGIN = 0.05
MODELS = (
    "google/gemini-3.8-flash", "qwen/qwen3.5-9b-base",
    "qwen/qwen3.5-9b-bm25-lora-step500", "qwen/qwen3.5-9b-vertex-hybrid-lora",
    "qwen/qwen3.5-9b-local-hybrid-lora",
)
SETUPS = ("bm25_top5", "vertex_hybrid_top5", "untuned_local_hybrid_top5", "tuned_local_hybrid_top5")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def bootstrap(values: list[float]) -> tuple[dict, np.ndarray]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(data, size=(RESAMPLES, len(data)), replace=True).mean(axis=1)
    result = {"mean": float(data.mean()),
              "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
              "one_sided_95_lower": float(np.percentile(samples, 5)), "n": len(data)}
    return result, samples


def mean(records: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def weighted_kappa(left: list[int], right: list[int]) -> float:
    matrix = np.zeros((3, 3), dtype=float)
    for a, b in zip(left, right, strict=True): matrix[a, b] += 1
    matrix /= matrix.sum()
    observed = sum(matrix[i, j] * (1 - abs(i - j) / 2) for i in range(3) for j in range(3))
    expected_matrix = np.outer(matrix.sum(axis=1), matrix.sum(axis=0))
    expected = sum(expected_matrix[i, j] * (1 - abs(i - j) / 2) for i in range(3) for j in range(3))
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--gemini-judge", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    automatic = {(r["model"], r["setup"], r["question_id"]): r for r in read_jsonl(args.automatic)}
    judge_maps = [
        {(r["model"], r["setup"], r["question_id"]): r for r in read_jsonl(path)}
        for path in (args.gemini_judge, args.qwen_judge)
    ]
    question_ids = sorted({key[2] for key in automatic})
    expected = {(model, setup, qid) for model in MODELS for setup in SETUPS for qid in question_ids}
    for name, mapping in (("automatic", automatic), ("gemini_judge", judge_maps[0]), ("qwen_judge", judge_maps[1])):
        if set(mapping) != expected:
            raise ValueError(f"Incomplete {name}: {len(mapping)} records, expected {len(expected)}")
    rows = []
    for key in sorted(expected):
        row = dict(automatic[key]); judges = [mapping[key] for mapping in judge_maps]
        row.update({
            "primary_score": sum(j["answer_score"] / 2 for j in judges) / 2,
            "gemini_judge_score": judges[0]["answer_score"] / 2,
            "qwen_judge_score": judges[1]["answer_score"] / 2,
            "judge_citation": sum(j["citation_score"] for j in judges) / 2,
            "unsupported_claim_rate": sum(j["unsupported_claim"] for j in judges) / 2,
        }); rows.append(row)
    grouped = defaultdict(list)
    for row in rows: grouped[(row["model"], row["setup"])].append(row)
    aggregates, categories = {}, {}
    for (model, setup), records in grouped.items():
        aggregates.setdefault(model, {})[setup] = {
            "primary_score": bootstrap([r["primary_score"] for r in records])[0],
            "gemini_judge": bootstrap([r["gemini_judge_score"] for r in records])[0],
            "qwen_judge": bootstrap([r["qwen_judge_score"] for r in records])[0],
            "judge_citation": mean(records, "judge_citation"), "citation_recall": mean(records, "citation_recall"),
            "token_f1": mean(records, "token_f1"), "unsupported_claim_rate": mean(records, "unsupported_claim_rate"),
            "mean_elapsed_seconds": mean(records, "elapsed_seconds"),
        }
        for category in sorted({r["category"] for r in records}):
            subset = [r for r in records if r["category"] == category]
            categories.setdefault(model, {}).setdefault(setup, {})[category] = {
                "primary_score": bootstrap([r["primary_score"] for r in subset])[0],
                "citation_recall": mean(subset, "citation_recall"), "token_f1": mean(subset, "token_f1"),
                "unsupported_claim_rate": mean(subset, "unsupported_claim_rate"),
            }
    lookup = {(r["model"], r["setup"], r["question_id"]): r["primary_score"] for r in rows}
    comparisons = [
        ("generator_local_qwen_minus_gemini_same_context", MODELS[4], "tuned_local_hybrid_top5", MODELS[0], "tuned_local_hybrid_top5", True),
        ("complete_local_minus_cloud", MODELS[4], "tuned_local_hybrid_top5", MODELS[0], "vertex_hybrid_top5", False),
        ("local_qwen_minus_base_same_context", MODELS[4], "tuned_local_hybrid_top5", MODELS[1], "tuned_local_hybrid_top5", False),
        ("local_qwen_minus_bm25_adapter_same_context", MODELS[4], "tuned_local_hybrid_top5", MODELS[2], "tuned_local_hybrid_top5", False),
        ("local_qwen_minus_vertex_adapter_same_context", MODELS[4], "tuned_local_hybrid_top5", MODELS[3], "tuned_local_hybrid_top5", False),
    ]
    for model in MODELS:
        comparisons.append((f"tuned_minus_untuned_local:{model}", model, "tuned_local_hybrid_top5", model, "untuned_local_hybrid_top5", False))
    paired, raw_p = {}, {}
    for name, left_model, left_setup, right_model, right_setup, noninferiority in comparisons:
        values = [lookup[(left_model, left_setup, qid)] - lookup[(right_model, right_setup, qid)] for qid in question_ids]
        result, samples = bootstrap(values)
        raw_p[name] = min(1.0, 2 * min(float(np.mean(samples <= 0)), float(np.mean(samples >= 0))))
        result.update({"left": [left_model, left_setup], "right": [right_model, right_setup],
                       "noninferiority_margin": -MARGIN if noninferiority else None,
                       "noninferior": result["one_sided_95_lower"] > -MARGIN if noninferiority else None})
        paired[name] = result
    adjusted = holm(raw_p)
    for name in paired: paired[name].update({"p_value_bootstrap": raw_p[name], "holm_adjusted_p": adjusted[name]})
    gemini_scores = [judge_maps[0][key]["answer_score"] for key in sorted(expected)]
    qwen_scores = [judge_maps[1][key]["answer_score"] for key in sorted(expected)]
    result = {"answers": len(rows), "questions": len(question_ids), "bootstrap_resamples": RESAMPLES,
              "seed": SEED, "noninferiority_margin": MARGIN,
              "judge_weighted_kappa": weighted_kappa(gemini_scores, qwen_scores),
              "aggregates": aggregates, "category_results": categories, "paired": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

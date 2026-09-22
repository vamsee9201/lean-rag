#!/usr/bin/env python3
"""Summarize the fourth experiment's 4x5 matrix with paired uncertainty."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


SEED = 20260918
RESAMPLES = 10_000
MARGIN = 0.05
MODELS = (
    "google/gemini-3.8-flash",
    "qwen/qwen3.5-9b-base",
    "qwen/qwen3.5-9b-local-hybrid-lora",
    "qwen/qwen3.5-9b-qwen-embedding-lora",
)
SETUPS = (
    "bm25_top5",
    "vertex_hybrid_top5",
    "qwen3_untuned_dense_top5",
    "qwen3_untuned_hybrid_top5",
    "qwen3_tuned_hybrid_top5",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def keyed_records(path: Path, label: str, expected_count: int = 1000) -> dict[tuple[str, str, str], dict]:
    records = read_jsonl(path)
    keys = [(row["model"], row["setup"], row["question_id"]) for row in records]
    if len(records) != expected_count or len(set(keys)) != len(keys):
        raise ValueError(f"{label} must contain {expected_count} unique ratings or answers")
    if label != "automatic":
        for row in records:
            if (row.get("answer_score") not in (0, 1, 2)
                or row.get("citation_score") not in (0, 1)
                or row.get("unsupported_claim") not in (0, 1, None)):
                raise ValueError(f"{label} contains an invalid judge rating")
    return dict(zip(keys, records, strict=True))


def bootstrap(values: list[float]) -> tuple[dict, np.ndarray]:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(data, size=(RESAMPLES, len(data)), replace=True).mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "one_sided_95_lower": float(np.percentile(samples, 5)),
        "n": len(data),
    }, samples


def mean(records: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in records if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def weighted_kappa(left: list[int], right: list[int]) -> float:
    matrix = np.zeros((3, 3), dtype=float)
    for a, b in zip(left, right, strict=True):
        matrix[a, b] += 1
    matrix /= matrix.sum()
    weight = lambda i, j: 1 - abs(i - j) / 2
    observed = sum(matrix[i, j] * weight(i, j) for i in range(3) for j in range(3))
    expected_matrix = np.outer(matrix.sum(axis=1), matrix.sum(axis=0))
    expected = sum(expected_matrix[i, j] * weight(i, j) for i in range(3) for j in range(3))
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted, running = {}, 0.0
    count = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_values[name]))
        adjusted[name] = running
    return adjusted


def paired_randomization_p(values: list[float], salt: int) -> float:
    data = np.asarray(values, dtype=float)
    observed = abs(float(data.mean()))
    rng = np.random.default_rng(SEED + salt)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(RESAMPLES, len(data)))
    null = np.abs((signs * data).mean(axis=1))
    return float((1 + np.count_nonzero(null >= observed)) / (RESAMPLES + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--gemini-judge", type=Path, required=True)
    parser.add_argument("--qwen-judge", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    automatic = keyed_records(args.automatic, "automatic")
    if any(row.get("citation_policy") != "exact" for row in automatic.values()):
        raise ValueError("Fourth-experiment scores require exact citation policy")
    judges = [
        keyed_records(args.gemini_judge, "gemini_judge"),
        keyed_records(args.qwen_judge, "qwen_judge"),
    ]
    question_ids = sorted({key[2] for key in automatic})
    if len(question_ids) != 50:
        raise ValueError(f"Expected 50 questions, found {len(question_ids)}")
    expected = {(model, setup, qid) for model in MODELS for setup in SETUPS for qid in question_ids}
    for name, mapping in (("automatic", automatic), ("gemini_judge", judges[0]), ("qwen_judge", judges[1])):
        if set(mapping) != expected:
            raise ValueError(f"Incomplete {name}: {len(mapping)} records, expected {len(expected)}")
    rows = []
    for key in sorted(expected):
        row = dict(automatic[key])
        ratings = [mapping[key] for mapping in judges]
        row.update({
            "primary_score": sum(rating["answer_score"] / 2 for rating in ratings) / 2,
            "gemini_judge_score": ratings[0]["answer_score"] / 2,
            "qwen_judge_score": ratings[1]["answer_score"] / 2,
            "judge_citation": sum(rating["citation_score"] for rating in ratings) / 2,
            "unsupported_claim_labels": [
                rating["unsupported_claim"] for rating in ratings
                if rating["unsupported_claim"] is not None
            ],
        })
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["setup"])].append(row)
    aggregates, categories = {}, {}
    for (model, setup), records in grouped.items():
        aggregates.setdefault(model, {})[setup] = {
            "primary_score": bootstrap([r["primary_score"] for r in records])[0],
            "gemini_judge": bootstrap([r["gemini_judge_score"] for r in records])[0],
            "qwen_judge": bootstrap([r["qwen_judge_score"] for r in records])[0],
            "judge_citation": mean(records, "judge_citation"),
            "citation_recall": mean(records, "citation_recall"),
            "citation_validity": mean(records, "citation_validity"),
            "reference_hit": mean(records, "reference_hit"),
            "deterministic_correct": mean(records, "deterministic_correct"),
            "token_f1": mean(records, "token_f1"),
            "unsupported_claim_rate": float(np.mean([
                label for r in records for label in r["unsupported_claim_labels"]
            ])),
            "unsupported_claim_ratings": sum(
                len(r["unsupported_claim_labels"]) for r in records
            ),
            "answerable_abstention_rate": mean(
                [r for r in records if r["answerable"]], "abstained"
            ),
            "unanswerable_abstention_accuracy": mean(
                [r for r in records if not r["answerable"]], "abstained"
            ),
            "mean_elapsed_seconds": mean(records, "elapsed_seconds"),
        }
        for category in sorted({r["category"] for r in records}):
            subset = [r for r in records if r["category"] == category]
            categories.setdefault(model, {}).setdefault(setup, {})[category] = {
                "primary_score": bootstrap([r["primary_score"] for r in subset])[0],
                "citation_recall": mean(subset, "citation_recall"),
                "citation_validity": mean(subset, "citation_validity"),
                "reference_hit": mean(subset, "reference_hit"),
                "token_f1": mean(subset, "token_f1"),
                "unsupported_claim_rate": float(np.mean([
                    label for r in subset for label in r["unsupported_claim_labels"]
                ])),
                "unsupported_claim_ratings": sum(
                    len(r["unsupported_claim_labels"]) for r in subset
                ),
                "answerable_abstention_rate": mean(subset, "abstained"),
            }
    lookup = {(r["model"], r["setup"], r["question_id"]): r["primary_score"] for r in rows}
    comparisons = [
        (
            "new_qwen_minus_gemini_same_tuned_context",
            MODELS[3], "qwen3_tuned_hybrid_top5", MODELS[0], "qwen3_tuned_hybrid_top5", True,
        ),
        (
            "complete_qwen_local_stack_minus_vertex_gemini",
            MODELS[3], "qwen3_tuned_hybrid_top5", MODELS[0], "vertex_hybrid_top5", False,
        ),
        (
            "new_qwen_minus_base_same_context",
            MODELS[3], "qwen3_tuned_hybrid_top5", MODELS[1], "qwen3_tuned_hybrid_top5", False,
        ),
        (
            "new_qwen_minus_previous_adapter_same_context",
            MODELS[3], "qwen3_tuned_hybrid_top5", MODELS[2], "qwen3_tuned_hybrid_top5", False,
        ),
    ]
    for model in MODELS:
        comparisons.extend([
            (
                f"tuned_hybrid_minus_untuned_hybrid:{model}",
                model, "qwen3_tuned_hybrid_top5", model, "qwen3_untuned_hybrid_top5", False,
            ),
            (
                f"untuned_hybrid_minus_dense:{model}",
                model, "qwen3_untuned_hybrid_top5", model, "qwen3_untuned_dense_top5", False,
            ),
        ])
    paired, raw_p = {}, {}
    primary_name = "new_qwen_minus_gemini_same_tuned_context"
    for comparison_index, (
        name, left_model, left_setup, right_model, right_setup, noninferiority
    ) in enumerate(comparisons):
        values = [
            lookup[(left_model, left_setup, qid)] - lookup[(right_model, right_setup, qid)]
            for qid in question_ids
        ]
        result, _samples = bootstrap(values)
        raw_p[name] = paired_randomization_p(values, comparison_index)
        result.update({
            "left": [left_model, left_setup],
            "right": [right_model, right_setup],
            "noninferiority_margin": -MARGIN if noninferiority else None,
            "noninferior": result["one_sided_95_lower"] > -MARGIN if noninferiority else None,
        })
        paired[name] = result
    adjusted = holm({name: value for name, value in raw_p.items() if name != primary_name})
    for name in paired:
        paired[name].update({
            "p_value_paired_randomization": raw_p[name],
            "holm_adjusted_p": adjusted.get(name),
        })
    result = {
        "answers": len(rows),
        "questions": len(question_ids),
        "matrix": {"generators": len(MODELS), "retrievers": len(SETUPS), "cells": 20},
        "judge_field_coverage": {
            "answer_score": 2000,
            "citation_score": 2000,
            "unsupported_claim": sum(
                len(r["unsupported_claim_labels"]) for r in rows
            ),
            "qwen_unsupported_claim_missing": sum(
                judges[1][key]["unsupported_claim"] is None for key in expected
            ),
        },
        "bootstrap_resamples": RESAMPLES,
        "seed": SEED,
        "noninferiority_margin": MARGIN,
        "abstention_scope": (
            "The 50 sealed fourth-experiment questions are answerable. "
            "Answerable abstentions are measured; correct abstention on unanswerable "
            "questions is not estimable from this benchmark."
        ),
        "judge_weighted_kappa": weighted_kappa(
            [judges[0][key]["answer_score"] for key in sorted(expected)],
            [judges[1][key]["answer_score"] for key in sorted(expected)],
        ),
        "aggregates": aggregates,
        "category_results": categories,
        "paired": paired,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare two blind judges and aggregate their normalized answer scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, default=ROOT / "data/runs/judge_scores.jsonl")
    parser.add_argument(
        "--gemini", type=Path, default=ROOT / "data/runs/gemini_judge_scores.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/runs/dual_judge_summary.json"
    )
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/benchmark/questions.jsonl"
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def interval(values: list[float], seed: int = 20260905) -> dict:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(20_000, len(data)), replace=True).mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n": len(data),
    }


def main() -> None:
    args = parse_args()
    key = lambda row: (row["model"], row["setup"], row["question_id"])
    qwen = {key(row): row for row in read_jsonl(args.qwen)}
    gemini = {key(row): row for row in read_jsonl(args.gemini)}
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    common = sorted(qwen.keys() & gemini.keys())
    if len(common) != len(qwen) or len(common) != len(gemini):
        raise ValueError("Judge files do not contain the same answer keys")

    by_model: dict[str, list[tuple[tuple[str, str, str], dict, dict]]] = defaultdict(list)
    for item in common:
        by_model[item[0]].append((item, qwen[item], gemini[item]))

    models = {}
    combined_by_key = {}
    for model, records in sorted(by_model.items()):
        qwen_scores = [q["answer_score"] / 2 for _key, q, _g in records]
        gemini_scores = [g["answer_score"] / 2 for _key, _q, g in records]
        combined = [(q + g) / 2 for q, g in zip(qwen_scores, gemini_scores)]
        for (item, _q, _g), value in zip(records, combined):
            combined_by_key[item] = value
        categories: dict[str, list[float]] = defaultdict(list)
        for (item, _q, _g), value in zip(records, combined):
            categories[questions[item[2]]["category"]].append(value)
        models[model] = {
            "qwen_judge_accuracy": interval(qwen_scores),
            "gemini_judge_accuracy": interval(gemini_scores),
            "dual_judge_accuracy": interval(combined),
            "exact_answer_score_agreement": sum(
                q["answer_score"] == g["answer_score"] for _key, q, g in records
            ) / len(records),
            "dual_judge_by_category": {
                category: interval(values) for category, values in categories.items()
            },
        }

    qwen_values = np.asarray([qwen[item]["answer_score"] for item in common], dtype=float)
    gemini_values = np.asarray([gemini[item]["answer_score"] for item in common], dtype=float)
    citation_keys = [
        item for item in common
        if qwen[item]["citation_score"] is not None and gemini[item]["citation_score"] is not None
    ]
    local_models = [model for model in models if model != "google/gemini-3.8-flash"]
    paired = {}
    paired_subgroups = {}
    for local_model in local_models:
        model_name = f"gemini-minus-{local_model.rsplit('-', 1)[-1]}"
        differences = {
            question_id: (
                combined_by_key[("google/gemini-3.8-flash", "bm25_top5", question_id)]
                - combined_by_key[(local_model, "bm25_top5", question_id)]
            )
            for question_id in sorted({item[2] for item in common})
        }
        paired[model_name] = interval(list(differences.values()))
        subgroup_ids = {
            "answerable": [qid for qid in differences if questions[qid]["answerable"]]
        }
        for category in sorted({questions[qid]["category"] for qid in differences}):
            subgroup_ids[category] = [
                qid for qid in differences if questions[qid]["category"] == category
            ]
        paired_subgroups[model_name] = {
            name: interval([differences[qid] for qid in ids])
            for name, ids in subgroup_ids.items()
        }

    result = {
        "scored_answers": len(common),
        "overall_exact_answer_score_agreement": float(np.mean(qwen_values == gemini_values)),
        "overall_within_one_point_agreement": float(np.mean(np.abs(qwen_values - gemini_values) <= 1)),
        "answer_score_pearson_correlation": float(np.corrcoef(qwen_values, gemini_values)[0, 1]),
        "citation_agreement": float(np.mean([
            qwen[item]["citation_score"] == gemini[item]["citation_score"]
            for item in citation_keys
        ])),
        "models": models,
        "paired_dual_judge_differences": paired,
        "paired_dual_judge_differences_by_subgroup": paired_subgroups,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

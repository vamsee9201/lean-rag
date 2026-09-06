#!/usr/bin/env python3
"""Create paired bootstrap summaries from automatic and blind-judge scores."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import combinations
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--automatic", type=Path, default=ROOT / "data" / "runs" / "automatic_scores.jsonl")
    parser.add_argument("--judge", type=Path, default=ROOT / "data" / "runs" / "judge_scores.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runs" / "final_summary.json")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def interval(values: list[float], seed: int = 20260906) -> dict:
    data = np.asarray(values, dtype=float)
    if not len(data):
        return {"mean": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(10_000, len(data)), replace=True).mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
        "n": len(data),
    }


def main() -> None:
    args = parse_args()
    automatic = {
        (row["model"], row["setup"], row["question_id"]): row for row in read_jsonl(args.automatic)
    }
    judges = {
        (row["model"], row["setup"], row["question_id"]): row for row in read_jsonl(args.judge)
    }
    common = sorted(automatic.keys() & judges.keys())
    rows = []
    for key in common:
        row = {**automatic[key], **judges[key]}
        row["judge_accuracy"] = float(row["answer_score"]) / 2.0
        rows.append(row)

    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["setup"], "all")].append(row)
        grouped[(row["model"], row["setup"], row["category"])].append(row)
    aggregates = {}
    for (model, setup, category), records in grouped.items():
        aggregates.setdefault(model, {}).setdefault(setup, {})[category] = {
            "judge_accuracy": interval([record["judge_accuracy"] for record in records]),
            "judge_citation_accuracy": interval([
                float(record["citation_score"])
                for record in records if record["citation_score"] is not None
            ]),
            "deterministic_accuracy": interval([
                float(record["deterministic_correct"])
                for record in records if record["deterministic_correct"] is not None
            ]),
            "citation_recall": interval([
                float(record["citation_recall"])
                for record in records if record["citation_recall"] is not None
            ]),
            "mean_elapsed_seconds": (
                float(np.mean(values)) if (values := [
                    record["elapsed_seconds"] for record in records if record["elapsed_seconds"] is not None
                ]) else None
            ),
            "mean_tokens_per_second": (
                float(np.mean(values)) if (values := [
                    record["tokens_per_second"]
                    for record in records if record["tokens_per_second"] is not None
                ]) else None
            ),
        }

    lookup = {
        (row["model"], row["setup"], row["question_id"]): row["judge_accuracy"] for row in rows
    }
    paired = {}
    models = sorted({row["model"] for row in rows})
    setups = sorted({row["setup"] for row in rows})
    question_ids = sorted({row["question_id"] for row in rows})
    for setup in setups:
        for smaller, larger in combinations(models, 2):
            values = [
                lookup[(larger, setup, question_id)] - lookup[(smaller, setup, question_id)]
                for question_id in question_ids
                if (larger, setup, question_id) in lookup
                and (smaller, setup, question_id) in lookup
            ]
            smaller_name = smaller.rsplit("-", 1)[-1]
            larger_name = larger.rsplit("-", 1)[-1]
            paired[f"{setup}: {larger_name}-minus-{smaller_name}"] = interval(values)

    result = {"scored_answers": len(rows), "aggregates": aggregates, "paired_differences": paired}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

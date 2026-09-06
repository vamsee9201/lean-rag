#!/usr/bin/env python3
"""Analyze completed blinded human ratings with paired confidence intervals."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path

import numpy as np


GEMINI = "google/gemini-3.8-flash"
QWEN = "qwen/qwen3.5-9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    return parser.parse_args()


def bootstrap(values: list[float], samples: int, seed: int = 20260905) -> dict:
    data = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(data, size=(samples, len(data)), replace=True).mean(axis=1)
    return {
        "mean": float(data.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n": len(data),
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / (2**n)
    return min(1.0, 2 * tail)


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))["questions"]
    valid_ids = None
    if args.audit:
        with args.audit.open(encoding="utf-8", newline="") as stream:
            audit = list(csv.DictReader(stream))
        accepted = {"yes", "y", "1", "true"}
        rejected = {"no", "n", "0", "false"}
        audit_fields = ("question_valid", "reference_valid", "evidence_supports_reference")
        incomplete = [
            row["question_id"] for row in audit
            if any(row[field].strip().casefold() not in accepted | rejected for field in audit_fields)
        ]
        if incomplete:
            raise ValueError(f"Question audit is incomplete for {len(incomplete)} questions")
        valid_ids = {
            row["question_id"] for row in audit
            if all(row[field].strip().casefold() in accepted for field in audit_fields)
        }

    with args.review.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = []
    for row in rows:
        qid = row["question_id"]
        if valid_ids is not None and qid not in valid_ids:
            continue
        try:
            scores = {
                mapping[qid]["A"]: int(row["answer_score_A_0_to_2"]),
                mapping[qid]["B"]: int(row["answer_score_B_0_to_2"]),
            }
        except (KeyError, ValueError):
            continue
        if any(score not in {0, 1, 2} for score in scores.values()):
            raise ValueError(f"Invalid answer score for {qid}")
        records.append({
            "question_id": qid,
            "category": row["category"],
            "gemini": scores[GEMINI] / 2,
            "qwen": scores[QWEN] / 2,
        })
    if not records:
        raise ValueError("No completed, valid human ratings found")
    completed_ids = {row["question_id"] for row in records}
    expected_ids = valid_ids if valid_ids is not None else {row["question_id"] for row in rows}
    if completed_ids != expected_ids:
        raise ValueError(
            f"Human ratings are incomplete: expected={len(expected_ids)}, completed={len(completed_ids)}"
        )

    differences = [row["gemini"] - row["qwen"] for row in records]
    by_category = defaultdict(list)
    for row in records:
        by_category[row["category"]].append(row["gemini"] - row["qwen"])
    gemini_only = sum(row["gemini"] == 1 and row["qwen"] < 1 for row in records)
    qwen_only = sum(row["qwen"] == 1 and row["gemini"] < 1 for row in records)
    paired = bootstrap(differences, args.bootstrap_samples)
    result = {
        "rated_valid_questions": len(records),
        "models": {
            GEMINI: bootstrap([row["gemini"] for row in records], args.bootstrap_samples),
            QWEN: bootstrap([row["qwen"] for row in records], args.bootstrap_samples),
        },
        "gemini_minus_qwen": paired,
        "by_category_gemini_minus_qwen": {
            category: bootstrap(values, args.bootstrap_samples)
            for category, values in sorted(by_category.items())
        },
        "full_correct_discordant_pairs": {
            "gemini_only": gemini_only,
            "qwen_only": qwen_only,
            "exact_mcnemar_p": exact_mcnemar_p(gemini_only, qwen_only),
        },
        "decision_rule": {
            "minimum_practical_gain": 0.05,
            "statistically_superior": paired["ci95"][0] > 0,
            "practically_superior": paired["mean"] >= 0.05,
            "claim_supported": paired["ci95"][0] > 0 and paired["mean"] >= 0.05,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Combine successful answer files while enforcing exact model-question alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--setups", nargs="+", default=["bm25_top5"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    question_ids = {row["question_id"] for row in read_jsonl(args.questions)}
    records = {}
    errors = []
    for path in args.inputs:
        for row in read_jsonl(path):
            if row.get("status") != "ok":
                errors.append(row)
                continue
            key = (row["model"], row["setup"], row["question_id"])
            if key in records:
                raise ValueError(f"Duplicate successful answer: {key}")
            records[key] = row
    expected = {
        (model, setup, question_id)
        for model in args.models
        for setup in args.setups
        for question_id in question_ids
    }
    missing = expected - set(records)
    extra = set(records) - expected
    if missing or extra:
        raise ValueError(f"Alignment failed: missing={len(missing)}, extra={len(extra)}")
    ordered = [records[key] for key in sorted(records)]
    for setup in args.setups:
        for question_id in question_ids:
            hashes = {
                records[(model, setup, question_id)].get("prompt_sha256")
                for model in args.models
            }
            if None in hashes or len(hashes) != 1:
                raise ValueError(f"Prompt hashes differ for {setup}/{question_id}: {hashes}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in ordered:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"answers": len(ordered), "errors_ignored": len(errors)}, indent=2))


if __name__ == "__main__":
    main()

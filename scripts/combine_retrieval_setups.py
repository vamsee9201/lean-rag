#!/usr/bin/env python3
"""Combine frozen retrieval files and verify complete question/setup alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--setups", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    question_ids = {row["question_id"] for row in read_jsonl(args.questions)}
    records = {}
    for path in args.inputs:
        for row in read_jsonl(path):
            key = (row["setup"], row["question_id"])
            if key in records: raise ValueError(f"Duplicate retrieval record: {key}")
            records[key] = row
    expected = {(setup, qid) for setup in args.setups for qid in question_ids}
    if set(records) != expected:
        raise ValueError(f"Retrieval alignment failed: missing={len(expected-set(records))}, extra={len(set(records)-expected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for key in sorted(records): stream.write(json.dumps(records[key], ensure_ascii=False) + "\n")
    manifest = {"records": len(records), "questions": len(question_ids), "setups": args.setups,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "input_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.inputs}}
    args.output.with_name(args.output.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

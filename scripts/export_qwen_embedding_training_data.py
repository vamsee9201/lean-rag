#!/usr/bin/env python3
"""Convert verified GovInfo retrieval pairs to the official SWIFT InfoNCE format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/local_retriever_experiment/models/gte/mined_training_records.jsonl"
DEFAULT_OUTPUT = ROOT / "data/qwen_embedding_experiment/embedding_training/train_swift.jsonl"
QUERY_INSTRUCTION = (
    "Given a search query about United States government publications, "
    "retrieve passages that contain the evidence needed to answer the query."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def query_text(query: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{query}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force")
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    converted = []
    duplicate_negatives_removed = 0
    for index, row in enumerate(rows, 1):
        original_negatives = [row[f"negative_{number}"] for number in range(1, 5)]
        negatives = list(dict.fromkeys(original_negatives))
        duplicate_negatives_removed += len(original_negatives) - len(negatives)
        values = [row["query"], row["positive"], *negatives]
        if any(not value.strip() for value in values):
            raise ValueError(f"Blank training text at record {index}")
        if not negatives or row["positive"] in negatives:
            raise ValueError(f"Invalid hard-negative group at record {index}")
        converted.append({
            "messages": [{"role": "user", "content": query_text(row["query"])}],
            "positive_messages": [[{"role": "user", "content": row["positive"]}]],
            "negative_messages": [
                [{"role": "user", "content": negative}] for negative in negatives
            ],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in converted:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "base_model": "Qwen/Qwen3-Embedding-8B",
        "format": "ms-swift InfoNCE with one positive and up to four explicit hard negatives",
        "query_instruction": QUERY_INSTRUCTION,
        "records": len(converted),
        "duplicate_negatives_removed": duplicate_negatives_removed,
        "minimum_negatives_per_record": min(len(row["negative_messages"]) for row in converted),
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "output_sha256": sha256(args.output),
    }
    args.output.with_name("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

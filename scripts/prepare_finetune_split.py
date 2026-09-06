#!/usr/bin/env python3
"""Create deterministic, document-level train/validation/test splits for SFT."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "data" / "processed" / "corpus500" / "documents.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "finetuning" / "document_splits.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "finetuning" / "document_splits_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--train", type=int, default=350)
    parser.add_argument("--validation", type=int, default=50)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_key(seed: int, document_id: str) -> str:
    return hashlib.sha256(f"{seed}:{document_id}".encode()).hexdigest()


def allocate_counts(sizes: dict[tuple[str, bool], int], target: int) -> dict[tuple[str, bool], int]:
    """Proportionally allocate an exact target using largest remainders."""
    total = sum(sizes.values())
    if target < 0 or target > total:
        raise ValueError(f"Cannot allocate {target} records from {total}")
    raw = {key: target * size / total for key, size in sizes.items()}
    allocation = {key: min(size, int(raw[key])) for key, size in sizes.items()}
    remaining = target - sum(allocation.values())
    order = sorted(sizes, key=lambda key: (-(raw[key] - int(raw[key])), key))
    while remaining:
        progressed = False
        for key in order:
            if allocation[key] < sizes[key] and remaining:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError("Unable to complete proportional allocation")
    return allocation


def split_documents(
    documents: list[dict], train: int, validation: int, test: int, seed: int
) -> list[dict]:
    if train + validation + test != len(documents):
        raise ValueError(
            f"Requested {train + validation + test} documents, but input contains {len(documents)}"
        )
    groups: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for document in documents:
        searchable = document.get("status") == "ok" and document.get("character_count", 0) > 0
        groups[(document.get("collection", "UNKNOWN"), searchable)].append(document)
    for group in groups.values():
        group.sort(key=lambda row: stable_key(seed, row["document_id"]))

    original_sizes = {key: len(group) for key, group in groups.items()}
    test_counts = allocate_counts(original_sizes, test)
    remaining_sizes = {key: original_sizes[key] - test_counts[key] for key in groups}
    validation_counts = allocate_counts(remaining_sizes, validation)

    rows: list[dict] = []
    for key in sorted(groups):
        group = groups[key]
        test_count = test_counts[key]
        validation_count = validation_counts[key]
        for index, document in enumerate(group):
            if index < test_count:
                split = "test"
            elif index < test_count + validation_count:
                split = "validation"
            else:
                split = "train"
            rows.append(
                {
                    "document_id": document["document_id"],
                    "split": split,
                    "collection": document.get("collection"),
                    "searchable": key[1],
                    "sha256": document.get("sha256"),
                }
            )
    rows.sort(key=lambda row: row["document_id"])
    counts = Counter(row["split"] for row in rows)
    expected = {"train": train, "validation": validation, "test": test}
    if counts != expected:
        raise AssertionError(f"Incorrect split counts: {dict(counts)} != {expected}")
    return rows


def main() -> None:
    args = parse_args()
    for path in (args.output, args.summary):
        if path.exists() and not args.force:
            raise SystemExit(f"Output exists: {path}; pass --force to replace it")
    documents = [json.loads(line) for line in args.documents.read_text().splitlines() if line.strip()]
    rows = split_documents(documents, args.train, args.validation, args.test, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)

    by_split = Counter(row["split"] for row in rows)
    searchable = Counter(row["split"] for row in rows if row["searchable"])
    by_collection = Counter((row["split"], row["collection"]) for row in rows)
    collections = sorted({row["collection"] for row in rows})
    summary = {
        "seed": args.seed,
        "source": str(args.documents),
        "counts": dict(sorted(by_split.items())),
        "searchable_counts": dict(sorted(searchable.items())),
        "collection_counts": {
            split: {
                collection: by_collection[(split, collection)]
                for collection in collections
                if by_collection[(split, collection)]
            }
            for split in ("train", "validation", "test")
        },
        "output": str(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Write split-isolated page files and BM25 indexes for fine-tuning."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from build_bm25 import create_index


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits", type=Path,
        default=ROOT / "data" / "finetuning" / "document_splits.jsonl",
    )
    parser.add_argument(
        "--pages", type=Path,
        default=ROOT / "data" / "processed" / "corpus500" / "pages.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "finetuning")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_by_document = {
        row["document_id"]: row["split"]
        for row in (json.loads(line) for line in args.splits.read_text().splitlines() if line.strip())
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    page_paths = {split: args.output_dir / split / "pages.jsonl" for split in SPLITS}
    index_paths = {split: args.output_dir / split / "bm25.sqlite3" for split in SPLITS}
    for path in (*page_paths.values(), *index_paths.values()):
        if path.exists() and not args.force:
            raise SystemExit(f"Output exists: {path}; pass --force to replace all split artifacts")
    temporary = {split: path.with_suffix(".jsonl.tmp") for split, path in page_paths.items()}
    for split in SPLITS:
        page_paths[split].parent.mkdir(parents=True, exist_ok=True)
        temporary[split].unlink(missing_ok=True)
    streams = {split: temporary[split].open("w", encoding="utf-8") for split in SPLITS}
    page_counts: Counter[str] = Counter()
    searchable_documents: dict[str, set[str]] = {split: set() for split in SPLITS}
    try:
        with args.pages.open(encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                split = split_by_document.get(row["document_id"])
                if split is None:
                    raise ValueError(f"Document missing from split manifest: {row['document_id']}")
                streams[split].write(line)
                page_counts[split] += 1
                if row.get("text"):
                    searchable_documents[split].add(row["document_id"])
    finally:
        for stream in streams.values():
            stream.close()
    for split in SPLITS:
        temporary[split].replace(page_paths[split])
        create_index(page_paths[split], index_paths[split], chunk_words=350, overlap_words=50)
    summary = {
        "page_counts": dict(page_counts),
        "searchable_documents": {
            split: len(searchable_documents[split]) for split in SPLITS
        },
        "pages": {split: str(page_paths[split]) for split in SPLITS},
        "indexes": {split: str(index_paths[split]) for split in SPLITS},
    }
    (args.output_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

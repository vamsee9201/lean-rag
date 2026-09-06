#!/usr/bin/env python3
"""Materialize a complete deterministic subset from an extracted corpus JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from extract_corpus import DEFAULT_MANIFEST, read_manifest, select_stratified


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-pages", type=Path, required=True)
    parser.add_argument("--source-documents", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--documents", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [args.output_dir / name for name in ("pages.jsonl", "documents.jsonl", "selection.json", "summary.json")]
    if any(path.exists() for path in outputs) and not args.force:
        raise SystemExit(f"Output exists under {args.output_dir}; pass --force to replace it")

    selected_rows = select_stratified(read_manifest(args.manifest), args.documents)
    selected_ids = [row["package_id"] for row in selected_rows]
    wanted = set(selected_ids)

    document_records = {}
    with args.source_documents.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["document_id"] in wanted:
                document_records[record["document_id"]] = record
    missing = wanted - document_records.keys()
    if missing:
        raise ValueError(f"Source extraction is missing {len(missing)} selected documents")

    pages_path, documents_path, selection_path, summary_path = outputs
    pages_tmp = pages_path.with_suffix(".jsonl.tmp")
    documents_tmp = documents_path.with_suffix(".jsonl.tmp")
    page_count = flagged_count = character_count = 0
    with args.source_pages.open(encoding="utf-8") as source, pages_tmp.open("w", encoding="utf-8") as target:
        for line in source:
            page = json.loads(line)
            if page["document_id"] not in wanted:
                continue
            target.write(line)
            page_count += 1
            flagged_count += bool(page["quality_flags"])
            character_count += page["character_count"]

    collections: dict[str, int] = {}
    with documents_tmp.open("w", encoding="utf-8") as stream:
        for document_id in selected_ids:
            record = document_records[document_id]
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            collection = record["collection"]
            collections[collection] = collections.get(collection, 0) + 1

    pages_tmp.replace(pages_path)
    documents_tmp.replace(documents_path)
    selection = {
        "selection_method": "collection round-robin with page-count quantiles",
        "requested_documents": args.documents,
        "selected_documents": len(selected_ids),
        "document_ids": selected_ids,
        "materialized_from": str(args.source_pages),
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    summary = {
        "documents": len(selected_ids),
        "pages": page_count,
        "flagged_pages": flagged_count,
        "characters": character_count,
        "collections": dict(sorted(collections.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

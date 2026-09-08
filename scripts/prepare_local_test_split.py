#!/usr/bin/env python3
"""Extract a deterministic, document-isolated test corpus from unused PDFs."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from extract_corpus import extract_document, read_manifest, select_stratified


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "local_retriever_experiment" / "test"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifest.csv")
    parser.add_argument(
        "--previous-documents",
        type=Path,
        default=ROOT / "data" / "processed" / "corpus500" / "documents.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--searchable-documents", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_order(rows: list[dict[str, str]], seed: int) -> list[dict[str, str]]:
    ordered = select_stratified(rows, len(rows))
    return sorted(
        ordered,
        key=lambda row: (
            hashlib.sha256(f"{seed}:{row['package_id']}".encode()).hexdigest(),
            row["package_id"],
        ),
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        args.output_dir / "pages.jsonl",
        args.output_dir / "documents.jsonl",
        args.output_dir / "document_split.jsonl",
        args.output_dir / "summary.json",
    ]
    if any(path.exists() for path in outputs) and not args.force:
        raise SystemExit(f"Output exists under {args.output_dir}; pass --force to replace it")

    previous = read_jsonl(args.previous_documents)
    previous_ids = {row["document_id"] for row in previous}
    previous_hashes = {row["sha256"] for row in previous}
    candidates = [
        row for row in read_manifest(args.manifest)
        if row["package_id"] not in previous_ids and row["sha256"] not in previous_hashes
    ]
    candidates = stable_order(candidates, args.seed)

    selected: list[tuple[list[dict], dict]] = []
    attempted = 0
    for row in candidates:
        attempted += 1
        pages, document = extract_document(row)
        searchable = document["status"] == "ok" and document["character_count"] > 0
        if searchable:
            selected.append((pages, document))
            print(
                f"selected {len(selected)}/{args.searchable_documents} "
                f"{document['document_id']} ({len(pages)} pages)",
                flush=True,
            )
        if len(selected) == args.searchable_documents:
            break
    if len(selected) != args.searchable_documents:
        raise RuntimeError(
            f"Found {len(selected)} searchable unused documents after {attempted} attempts"
        )

    pages_path, documents_path, split_path, summary_path = outputs
    with pages_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as stream:
        for pages, _ in selected:
            for page in pages:
                stream.write(json.dumps(page, ensure_ascii=False) + "\n")
    pages_path.with_suffix(".jsonl.tmp").replace(pages_path)
    with documents_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as stream:
        for _, document in selected:
            stream.write(json.dumps(document, ensure_ascii=False) + "\n")
    documents_path.with_suffix(".jsonl.tmp").replace(documents_path)
    with split_path.with_suffix(".jsonl.tmp").open("w", encoding="utf-8") as stream:
        for _, document in selected:
            stream.write(json.dumps({
                "document_id": document["document_id"],
                "split": "test_v3",
                "collection": document["collection"],
                "searchable": True,
                "sha256": document["sha256"],
            }) + "\n")
    split_path.with_suffix(".jsonl.tmp").replace(split_path)

    all_pages = [page for pages, _ in selected for page in pages]
    summary = {
        "seed": args.seed,
        "selection_method": "sha256 ordering over PDFs excluded by package ID and PDF hash",
        "requested_searchable_documents": args.searchable_documents,
        "selected_searchable_documents": len(selected),
        "attempted_documents": attempted,
        "pages": len(all_pages),
        "characters": sum(page["character_count"] for page in all_pages),
        "collections": dict(sorted(Counter(doc["collection"] for _, doc in selected).items())),
        "previous_document_ids": len(previous_ids),
        "previous_pdf_hashes": len(previous_hashes),
        "document_ids_sha256": hashlib.sha256(
            "\n".join(doc["document_id"] for _, doc in selected).encode()
        ).hexdigest(),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

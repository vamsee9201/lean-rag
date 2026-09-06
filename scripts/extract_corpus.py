#!/usr/bin/env python3
"""Select and extract a reproducible, page-level GovInfo PDF pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
from pathlib import Path
import re

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data" / "manifest.csv"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "pilot30"
logging.getLogger("pypdf").setLevel(logging.ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--documents", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"package_id", "collection", "local_path", "pages", "sha256"}
    missing = required - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return rows


def select_stratified(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """Round-robin collections while sampling short, median, and long PDFs."""
    if count < 1:
        raise ValueError("--documents must be positive")
    by_collection: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_collection.setdefault(row["collection"], []).append(row)
    for collection_rows in by_collection.values():
        collection_rows.sort(key=lambda row: (int(row["pages"]), row["package_id"]))

    selected: list[dict[str, str]] = []
    used: set[str] = set()
    quantiles = (0.2, 0.5, 0.8, 0.35, 0.65, 0.05, 0.95)
    collections = sorted(by_collection)
    round_number = 0
    while len(selected) < min(count, len(rows)):
        progress = False
        q = quantiles[round_number % len(quantiles)]
        for collection in collections:
            candidates = by_collection[collection]
            center = round(q * (len(candidates) - 1))
            offsets = [0]
            for distance in range(1, len(candidates)):
                offsets.extend((-distance, distance))
            chosen = None
            for offset in offsets:
                index = center + offset
                if 0 <= index < len(candidates):
                    candidate = candidates[index]
                    if candidate["package_id"] not in used:
                        chosen = candidate
                        break
            if chosen is not None:
                selected.append(chosen)
                used.add(chosen["package_id"])
                progress = True
            if len(selected) == count:
                break
        if not progress:
            break
        round_number += 1
    return selected


def normalize_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def page_flags(text: str, extraction_error: str | None) -> list[str]:
    flags: list[str] = []
    if extraction_error:
        flags.append("extraction_error")
    if not text:
        flags.append("empty")
    elif len(text) < 100:
        flags.append("low_text")
    if text and text.count("\ufffd") / len(text) > 0.002:
        flags.append("replacement_characters")
    printable = sum(character.isprintable() or character in "\n\t" for character in text)
    if text and printable / len(text) < 0.98:
        flags.append("nonprintable_characters")
    return flags


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_document(row: dict[str, str]) -> tuple[list[dict], dict]:
    pdf_path = ROOT / row["local_path"]
    actual_hash = file_sha256(pdf_path)
    if actual_hash != row["sha256"]:
        raise ValueError(f"Hash mismatch for {row['package_id']}")
    reader = PdfReader(pdf_path, strict=False)
    page_records: list[dict] = []
    flagged_pages = 0
    characters = 0
    for page_number, page in enumerate(reader.pages, start=1):
        error = None
        try:
            text = normalize_text(page.extract_text(extraction_mode="layout") or "")
        except Exception as exc:  # retain the page and make the failure auditable
            text = ""
            error = f"{type(exc).__name__}: {exc}"[:300]
        flags = page_flags(text, error)
        if flags:
            flagged_pages += 1
        characters += len(text)
        page_records.append(
            {
                "document_id": row["package_id"],
                "title": row["title"],
                "collection": row["collection"],
                "date": row["date"],
                "page": page_number,
                "text": text,
                "character_count": len(text),
                "quality_flags": flags,
                "extraction_error": error,
            }
        )
    expected_pages = int(row["pages"])
    summary = {
        "document_id": row["package_id"],
        "title": row["title"],
        "collection": row["collection"],
        "date": row["date"],
        "local_path": row["local_path"],
        "sha256": actual_hash,
        "expected_pages": expected_pages,
        "extracted_pages": len(page_records),
        "flagged_pages": flagged_pages,
        "character_count": characters,
        "status": "ok" if len(page_records) == expected_pages else "page_count_mismatch",
    }
    return page_records, summary


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pages_path = args.output_dir / "pages.jsonl"
    documents_path = args.output_dir / "documents.jsonl"
    selection_path = args.output_dir / "selection.json"
    summary_path = args.output_dir / "summary.json"
    outputs = (pages_path, documents_path, selection_path, summary_path)
    if any(path.exists() for path in outputs) and not args.force:
        raise SystemExit(f"Output exists under {args.output_dir}; pass --force to replace it")

    rows = read_manifest(args.manifest)
    selected = select_stratified(rows, args.documents)
    selection = {
        "selection_method": "collection round-robin with page-count quantiles",
        "requested_documents": args.documents,
        "selected_documents": len(selected),
        "document_ids": [row["package_id"] for row in selected],
    }
    pages_temporary = pages_path.with_suffix(pages_path.suffix + ".tmp")
    documents_temporary = documents_path.with_suffix(documents_path.suffix + ".tmp")
    total_pages = 0
    total_flagged_pages = 0
    total_characters = 0
    collection_counts: dict[str, int] = {}
    with (
        pages_temporary.open("w", encoding="utf-8") as pages_stream,
        documents_temporary.open("w", encoding="utf-8") as documents_stream,
    ):
        for index, row in enumerate(selected, start=1):
            pages, document_summary = extract_document(row)
            for page in pages:
                pages_stream.write(json.dumps(page, ensure_ascii=False) + "\n")
            documents_stream.write(json.dumps(document_summary, ensure_ascii=False) + "\n")
            total_pages += len(pages)
            total_flagged_pages += document_summary["flagged_pages"]
            total_characters += document_summary["character_count"]
            collection = document_summary["collection"]
            collection_counts[collection] = collection_counts.get(collection, 0) + 1
            print(
                f"{index}/{len(selected)} {row['package_id']}: "
                f"{len(pages)} pages, {document_summary['flagged_pages']} flagged",
                flush=True,
            )
    pages_temporary.replace(pages_path)
    documents_temporary.replace(documents_path)
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    summary = {
        "documents": len(selected),
        "pages": total_pages,
        "flagged_pages": total_flagged_pages,
        "characters": total_characters,
        "collections": dict(sorted(collection_counts.items())),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and seal the third experiment's document-isolated benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"direct": 20, "numeric_date": 10, "multi_passage": 15, "cross_document": 5}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compact(text: str) -> str:
    return " ".join(text.split())


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", compact(text).casefold()))


def similar(left: str, right: str) -> bool:
    a, b = compact(left).casefold(), compact(right).casefold()
    jaccard = len(tokens(a) & tokens(b)) / max(1, len(tokens(a) | tokens(b)))
    return SequenceMatcher(None, a, b).ratio() >= 0.82 or jaccard >= 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "data/local_retriever_experiment"
    parser.add_argument("--sources", type=Path, default=base / "benchmark/question_sources.jsonl")
    parser.add_argument("--pages", type=Path, default=base / "test/pages.jsonl")
    parser.add_argument("--documents", type=Path, default=base / "test/documents.jsonl")
    parser.add_argument("--output", type=Path, default=base / "benchmark/questions.jsonl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force")
    sources = read_jsonl(args.sources)
    counts = Counter(row["category"] for row in sources)
    if counts != Counter(EXPECTED) or len(sources) != 50:
        raise ValueError(f"Unexpected benchmark composition: {counts}")
    documents = {row["document_id"]: row for row in read_jsonl(args.documents)}
    pages = {(row["document_id"], row["page"]): compact(row["text"]) for row in read_jsonl(args.pages)}
    previous_paths = [
        ROOT / "data/final_benchmark/questions.jsonl",
        ROOT / "data/hybrid_experiment/benchmark/questions.jsonl",
        ROOT / "data/benchmark/questions.jsonl",
    ]
    previous = [row for path in previous_paths if path.exists() for row in read_jsonl(path)]
    old_documents = {
        row["document_id"] for row in read_jsonl(ROOT / "data/processed/corpus500/documents.jsonl")
    }
    rows = []
    category_number = Counter()
    seen = []
    for source in sources:
        if any(similar(source["question"], row["question"]) for row in previous + seen):
            raise ValueError(f"Question is too similar to a prior question: {source['question']}")
        for document, page, quotation in zip(
            source["gold_documents"], source["gold_pages"], source["gold_passages"]
        ):
            if document not in documents or document in old_documents:
                raise ValueError(f"Gold document is not isolated: {document}")
            if compact(quotation) not in pages.get((document, page), ""):
                raise ValueError(f"Gold quotation is not exact: {document} p.{page}")
        category_number[source["category"]] += 1
        row = {
            "question_id": f"local-test-{source['category']}-{category_number[source['category']]:04d}",
            "category": source["category"], "question": source["question"],
            "reference_answer": source["reference_answer"], "answerable": True,
            "gold_documents": source["gold_documents"], "gold_pages": source["gold_pages"],
            "gold_passages": source["gold_passages"], "collections": source.get("collections", []),
        }
        rows.append(row); seen.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "sealed": True, "sealed_at": datetime.now(timezone.utc).isoformat(),
        "questions": 50, "categories": EXPECTED,
        "source_documents": len({d for row in rows for d in row["gold_documents"]}),
        "test_documents": len(documents), "previous_document_overlap": 0,
        "question_sha256": digest,
        "test_document_ids_sha256": hashlib.sha256("\n".join(sorted(documents)).encode()).hexdigest(),
        "generation_model": "gemini-3.8-flash", "generation_seed": 20260907,
        "duplicate_thresholds": {"sequence_match": 0.82, "token_jaccard": 0.75},
    }
    args.output.with_name("benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

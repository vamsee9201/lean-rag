#!/usr/bin/env python3
"""Validate and seal the fresh 50-question Vertex-hybrid confirmatory benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"direct": 20, "numeric_date": 10, "multi_passage": 15, "cross_document": 5}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--pages", type=Path, default=ROOT / "data/finetuning/test/pages.jsonl")
    parser.add_argument("--splits", type=Path, default=ROOT / "data/finetuning/document_splits.jsonl")
    parser.add_argument("--previous", type=Path, default=ROOT / "data/finetuning/test/question_sources.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force to replace it")
    sources = read_jsonl(args.sources)
    counts = Counter(row["category"] for row in sources)
    if counts != EXPECTED or len(sources) != 50:
        raise ValueError(f"Expected {EXPECTED}, received {dict(counts)}")
    split_by_document = {row["document_id"]: row["split"] for row in read_jsonl(args.splits)}
    previous_documents = {
        document_id for row in read_jsonl(args.previous) for document_id in row["gold_documents"]
    }
    pages = {
        (row["document_id"], row["page"]): " ".join(row["text"].split())
        for row in read_jsonl(args.pages)
    }
    normalized_questions = set()
    output = []
    category_number = Counter()
    for source in sources:
        question_key = " ".join(source["question"].split()).casefold()
        if question_key in normalized_questions:
            raise ValueError("Duplicate question text")
        normalized_questions.add(question_key)
        for document_id, page, quote in zip(
            source["gold_documents"], source["gold_pages"], source["gold_passages"]
        ):
            if split_by_document.get(document_id) != "test":
                raise ValueError(f"Source document is outside the test split: {document_id}")
            if document_id in previous_documents:
                raise ValueError(f"Source document appeared in the earlier benchmark: {document_id}")
            page_text = pages.get((document_id, page))
            if page_text is None or " ".join(quote.split()) not in page_text:
                raise ValueError(f"Gold quote is not exact: {document_id} p.{page}")
        category_number[source["category"]] += 1
        output.append({
            "question_id": f"hybrid-test-{source['category']}-{category_number[source['category']]:04d}",
            "category": source["category"],
            "question": source["question"],
            "reference_answer": source["reference_answer"],
            "answerable": True,
            "gold_documents": source["gold_documents"],
            "gold_pages": source["gold_pages"],
            "gold_passages": source["gold_passages"],
            "collections": source.get("collections", []),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in output:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "sealed": True,
        "questions": 50,
        "categories": EXPECTED,
        "source_documents": len({d for row in output for d in row["gold_documents"]}),
        "sha256": digest,
        "output": str(args.output),
    }
    args.output.with_name("benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

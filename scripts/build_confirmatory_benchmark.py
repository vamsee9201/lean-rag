#!/usr/bin/env python3
"""Combine and validate the fresh 250-question confirmatory benchmark."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/confirmatory/benchmark"
SOURCES = [
    BASE / "direct_candidates.jsonl",
    BASE / "numeric_candidates.jsonl",
    BASE / "multipassage_candidates.jsonl",
    BASE / "crossdoc_candidates.jsonl",
    BASE / "unanswerable_candidates.jsonl",
    BASE / "direct_replacements.jsonl",
]
PILOT = ROOT / "data/benchmark/questions.jsonl"
PAGES = ROOT / "data/processed/corpus500/pages.jsonl"
OUTPUT = BASE / "questions.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compact(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    candidates = [record for path in SOURCES for record in read_jsonl(path)]
    questions = []
    seen_candidate_questions = set()
    for record in candidates:
        normalized = compact(record["question"]).casefold()
        if normalized in seen_candidate_questions:
            continue
        seen_candidate_questions.add(normalized)
        questions.append(record)
    expected = {
        "direct": 100,
        "numeric_date": 50,
        "single_document_multi_passage": 50,
        "cross_document_comparison": 25,
        "unanswerable": 25,
    }
    counts = Counter(record["category"] for record in questions)
    if counts != Counter(expected):
        raise ValueError(f"Category mismatch: {counts}")
    pilot_sources = {
        (document_id, page)
        for record in read_jsonl(PILOT)
        for document_id, page in zip(record.get("gold_documents", []), record.get("gold_pages", []))
    }
    needed_sources = {
        (document_id, page)
        for record in questions
        for document_id, page in zip(record.get("gold_documents", []), record.get("gold_pages", []))
    }
    overlap = pilot_sources & needed_sources
    if overlap:
        raise ValueError(f"Confirmatory benchmark reuses {len(overlap)} pilot gold pages")
    page_lookup = {}
    with PAGES.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            key = (page["document_id"], page["page"])
            if key in needed_sources:
                page_lookup[key] = compact(page["text"])
    seen_questions = set()
    for index, record in enumerate(questions, 1):
        record["question_id"] = f"cq-{index:03d}"
        record.setdefault("answerable", record["category"] != "unanswerable")
        record.setdefault("gold_contexts", record.get("gold_passages", []))
        normalized = compact(record["question"]).casefold()
        if normalized in seen_questions:
            raise ValueError(f"Duplicate question: {record['question']}")
        seen_questions.add(normalized)
        if record["answerable"]:
            sources = list(zip(
                record["gold_documents"], record["gold_pages"], record["gold_passages"]
            ))
            if not sources:
                raise ValueError(f"Answerable question has no gold source: {record['question_id']}")
            for document_id, page, quote in sources:
                source = page_lookup.get((document_id, page))
                if source is None or compact(quote) not in source:
                    raise ValueError(f"Invalid quote for {record['question_id']} {document_id} p.{page}")
    BASE.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in questions:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(OUTPUT)
    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    summary = {
        "questions": len(questions),
        "answerable": sum(record["answerable"] for record in questions),
        "unanswerable": sum(not record["answerable"] for record in questions),
        "categories": dict(counts),
        "pilot_gold_page_overlap": 0,
        "sha256": digest,
        "status": "frozen_before_finalist_generation",
    }
    (BASE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

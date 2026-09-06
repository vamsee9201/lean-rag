#!/usr/bin/env python3
"""Curate and validate the fixed 50-question benchmark."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIRECT = ROOT / "data" / "benchmark" / "direct_reviewed.jsonl"
ADVANCED = ROOT / "data" / "benchmark" / "advanced_candidates.jsonl"
PAGES = ROOT / "data" / "processed" / "corpus500" / "pages.jsonl"
OUTPUT = ROOT / "data" / "benchmark" / "questions.jsonl"
SUMMARY = ROOT / "data" / "benchmark" / "summary.json"

SELECTED_ADVANCED = {
    "numeric_date": [
        "numeric_date-001", "numeric_date-002", "numeric_date-003", "numeric_date-004",
        "numeric_date-005", "numeric_date-006", "numeric_date-007", "numeric_date-008",
        "numeric_date-010", "numeric_date-011",
    ],
    "single_document_multi_passage": [
        "single_document_multi_passage-001", "single_document_multi_passage-002",
        "single_document_multi_passage-003", "single_document_multi_passage-004",
        "single_document_multi_passage-005", "single_document_multi_passage-006",
        "single_document_multi_passage-007", "single_document_multi_passage-008",
        "single_document_multi_passage-009", "single_document_multi_passage-012",
    ],
    "cross_document_comparison": [
        "cross_document_comparison-003", "cross_document_comparison-005",
        "cross_document_comparison-007", "cross_document_comparison-009",
        "cross_document_comparison-010",
    ],
}

UNANSWERABLE = [
    {
        "question": (
            "According to the Economic Report of the President (2001), what was total bank "
            "credit at all commercial banks in December 2030?"
        ),
        "source_scope_documents": ["ERP-2001"],
        "rationale": "The report's historical table does not contain a value for the future year 2030.",
    },
    {
        "question": (
            "What final construction completion date does the Budget of the U.S. Government, "
            "Fiscal Year 2027 give for the National Center for Warrior Independence facility?"
        ),
        "source_scope_documents": ["BUDGET-2027-BUD"],
        "rationale": "The source gives a requested funding amount but no final completion date.",
    },
    {
        "question": (
            "According to the Semiannual Monetary Policy Report hearing, what exact dollar amount "
            "did the Federal Reserve recover from employees held accountable for the Silicon Valley Bank failure?"
        ),
        "source_scope_documents": ["CHRG-119shrg63444"],
        "rationale": "The testimony discusses accountability but provides no recovered dollar amount.",
    },
    {
        "question": (
            "How many small firms won NASA contracts specifically because of the UNISPHERE program, "
            "according to GAO report RCED-95-137?"
        ),
        "source_scope_documents": ["GAOREPORTS-RCED-95-137"],
        "rationale": "The report describes the program but does not state this causal award count.",
    },
    {
        "question": (
            "What was the full name of the individual fisherman who designed the cacaiste shrimp trap "
            "described in the fisheries report?"
        ),
        "source_scope_documents": ["GOVPUB-C1_200-018d48db056e7306d662da29a72ae11c"],
        "rationale": "The report describes the trap and fishing practice without naming an inventor.",
    },
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def compact(text: str) -> str:
    return " ".join(text.split())


def main() -> None:
    direct = read_jsonl(DIRECT)
    advanced = {record["question_id"]: record for record in read_jsonl(ADVANCED)}
    questions = direct[:]
    for category, ids in SELECTED_ADVANCED.items():
        for question_id in ids:
            record = advanced[question_id]
            if record["category"] != category:
                raise ValueError(f"Category mismatch for {question_id}")
            record["review_status"] = "machine_curated"
            questions.append(record)

    for record in UNANSWERABLE:
        questions.append({
            "category": "unanswerable",
            "question": record["question"],
            "reference_answer": "Insufficient evidence.",
            "answerable": False,
            "gold_documents": [],
            "gold_pages": [],
            "gold_passages": [],
            "gold_contexts": [],
            "source_scope_documents": record["source_scope_documents"],
            "unanswerable_rationale": record["rationale"],
            "review_status": "machine_curated",
        })

    if len(questions) != 50:
        raise ValueError(f"Expected 50 questions, found {len(questions)}")
    page_lookup = {}
    with PAGES.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            key = (page["document_id"], page["page"])
            page_lookup[key] = compact(page["text"])

    seen_questions = set()
    for index, record in enumerate(questions, start=1):
        record["question_id"] = f"q-{index:03d}"
        record.setdefault("answerable", True)
        normalized_question = compact(record["question"]).casefold()
        if normalized_question in seen_questions:
            raise ValueError(f"Duplicate question: {record['question']}")
        seen_questions.add(normalized_question)
        if record["answerable"]:
            sources = list(zip(record["gold_documents"], record["gold_pages"], record["gold_passages"]))
            if not sources:
                raise ValueError(f"Answerable question has no sources: {record['question_id']}")
            for document_id, page_number, quote in sources:
                source_text = page_lookup.get((document_id, page_number))
                if source_text is None:
                    raise ValueError(f"Missing gold page {document_id} p{page_number}")
                if compact(quote) not in source_text:
                    raise ValueError(f"Gold quote missing from {document_id} p{page_number}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in questions:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(OUTPUT)

    counts = Counter(record["category"] for record in questions)
    summary = {
        "questions": len(questions),
        "answerable": sum(record["answerable"] for record in questions),
        "unanswerable": sum(not record["answerable"] for record in questions),
        "categories": dict(sorted(counts.items())),
        "corpus": "corpus500",
        "paired_design": "Every model and retrieval setup receives these same 50 questions.",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

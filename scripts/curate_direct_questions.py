#!/usr/bin/env python3
"""Apply recorded machine-assisted curation to direct benchmark candidates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data" / "benchmark" / "direct_candidates.jsonl"
PAGES = ROOT / "data" / "processed" / "pilot30" / "pages.jsonl"
OUTPUT = ROOT / "data" / "benchmark" / "direct_reviewed.jsonl"

DROPPED = {
    "direct-007": "asks for a primary reason when the source gives two equal reasons",
    "direct-013": "duplicate of direct-008",
    "direct-018": "table value lacks enough row and unit context",
    "direct-020": "duplicate of direct-008",
    "direct-024": "yes/no answer depends on dialogue outside the quoted evidence",
}

UPDATES = {
    "direct-011": {
        "question": (
            "According to the Economic Report of the President (2001), what was total "
            "bank credit at all commercial banks in December 1989, in billions of dollars?"
        )
    },
    "direct-015": {
        "question": (
            "Who sent the letter of transmittal for the report 'Imports of minerals from "
            "South Africa by the United States and the OECD countries'?"
        )
    },
    "direct-025": {
        "question": (
            "What was the intended submission date for the Senate study 'United States "
            "foreign policy: basic aims of United States foreign policy'?"
        )
    },
}


def compact(text: str) -> str:
    return " ".join(text.split())


def source_context(page_text: str, evidence: str, radius: int = 450) -> str:
    page = compact(page_text)
    quote = compact(evidence)
    start = page.find(quote)
    if start < 0:
        raise ValueError("Reviewed evidence is absent from its source page")
    return page[max(0, start - radius) : min(len(page), start + len(quote) + radius)]


def main() -> None:
    page_lookup = {}
    with PAGES.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            page_lookup[(page["document_id"], page["page"])] = page["text"]

    reviewed = []
    with CANDIDATES.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["question_id"] in DROPPED:
                continue
            record.update(UPDATES.get(record["question_id"], {}))
            key = (record["gold_documents"][0], record["gold_pages"][0])
            record["gold_contexts"] = [source_context(page_lookup[key], record["gold_passages"][0])]
            record["review_status"] = "machine_curated"
            reviewed.append(record)

    if len(reviewed) != 20:
        raise ValueError(f"Expected 20 reviewed direct questions, found {len(reviewed)}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(OUTPUT.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(reviewed, start=1):
            record["question_id"] = f"direct-{index:03d}"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(OUTPUT)
    print(json.dumps({"reviewed": len(reviewed), "dropped": DROPPED}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build blinded paired human-review sheets for the fourth experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import random


COMPARISONS = {
    "new_qwen_vs_gemini_same_tuned_context": (
        ("google/gemini-3.8-flash", "qwen3_tuned_hybrid_top5"),
        ("qwen/qwen3.5-9b-qwen-embedding-lora", "qwen3_tuned_hybrid_top5"),
    ),
    "complete_qwen_local_vs_vertex_cloud": (
        ("google/gemini-3.8-flash", "vertex_hybrid_top5"),
        ("qwen/qwen3.5-9b-qwen-embedding-lora", "qwen3_tuned_hybrid_top5"),
    ),
    "new_qwen_vs_previous_adapter_same_context": (
        ("qwen/qwen3.5-9b-local-hybrid-lora", "qwen3_tuned_hybrid_top5"),
        ("qwen/qwen3.5-9b-qwen-embedding-lora", "qwen3_tuned_hybrid_top5"),
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="qwen-embedding-human-review-20260918")
    args = parser.parse_args()
    questions = sorted(read_jsonl(args.questions), key=lambda row: row["question_id"])
    if len(questions) != 50:
        raise ValueError(f"Expected 50 questions, found {len(questions)}")
    answers = {
        (row["question_id"], row["model"], row["setup"]): row
        for row in read_jsonl(args.answers)
        if row.get("status") == "ok"
    }
    rows = []
    comparison_codes = {
        name: f"C{position:02d}"
        for position, name in enumerate(COMPARISONS, start=1)
    }
    mapping = {"seed": args.seed, "comparisons": {}, "rows": {}}
    for name, identities in COMPARISONS.items():
        mapping["comparisons"][comparison_codes[name]] = {
            "name": name, "identities": identities,
        }
    for question in questions:
        sources = "; ".join(
            f"{document} p.{page}"
            for document, page in zip(question["gold_documents"], question["gold_pages"])
        )
        evidence = "\n\n".join(question["gold_passages"])
        for comparison, identities in COMPARISONS.items():
            comparison_code = comparison_codes[comparison]
            ordered = sorted(
                identities,
                key=lambda identity: hashlib.sha256(
                    f"{args.seed}:{question['question_id']}:{comparison}:{identity}".encode()
                ).hexdigest(),
            )
            answer_rows = [answers[(question["question_id"], model, setup)] for model, setup in ordered]
            row_id = "R" + hashlib.sha256(
                f"{args.seed}:{question['question_id']}:{comparison_code}".encode()
            ).hexdigest()[:16]
            mapping["rows"][row_id] = {"A": ordered[0], "B": ordered[1]}
            rows.append({
                "review_order": 0,
                "row_id": row_id,
                "comparison": comparison_code,
                "question_id": question["question_id"],
                "category": question["category"],
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "gold_sources": sources,
                "gold_evidence": evidence,
                "candidate_A": answer_rows[0]["answer"],
                "candidate_B": answer_rows[1]["answer"],
                "correctness_A_0_to_2": "",
                "correctness_B_0_to_2": "",
                "citation_A_0_or_1": "",
                "citation_B_0_or_1": "",
                "unsupported_claim_A_0_or_1": "",
                "unsupported_claim_B_0_or_1": "",
                "preferred_candidate_A_B_or_tie": "",
                "notes": "",
            })
    if len(rows) != 150:
        raise ValueError(f"Expected 150 review rows, built {len(rows)}")
    random.Random(args.seed).shuffle(rows)
    for position, row in enumerate(rows, start=1):
        row["review_order"] = position
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reviewer_1.csv", "reviewer_2.csv", "adjudicated.csv"):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (args.output_dir / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "questions": len(questions), "comparisons": len(COMPARISONS)}, indent=2))


if __name__ == "__main__":
    main()

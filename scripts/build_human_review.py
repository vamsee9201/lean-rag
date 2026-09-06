#!/usr/bin/env python3
"""Build blinded question-audit and paired answer-review CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEMINI = "google/gemini-3.8-flash"
QWEN = "qwen/qwen3.5-9b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="lean-rag-confirmatory-20260905")
    parser.add_argument("--models", nargs="+", default=[GEMINI, QWEN])
    parser.add_argument("--setups", nargs="+", default=["bm25_top5"])
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    questions = sorted(read_jsonl(args.questions), key=lambda row: row["question_id"])
    answer_rows = [row for row in read_jsonl(args.answers) if row.get("status") == "ok"]
    answers = {}
    for row in answer_rows:
        key = (row["question_id"], row["setup"], row["model"])
        if key in answers:
            raise ValueError(f"Duplicate successful answer: {key}")
        answers[key] = row

    expected = {
        (q["question_id"], setup, model)
        for q in questions for setup in args.setups for model in args.models
    }
    missing = expected - set(answers)
    extra = set(answers) - expected
    if missing or extra:
        raise ValueError(f"Answer alignment failed: missing={len(missing)}, extra={len(extra)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    review_rows = []
    mapping = {"seed": args.seed, "models": args.models, "setups": args.setups, "rows": {}}
    for index, question in enumerate(questions, 1):
        qid = question["question_id"]
        sources = "; ".join(
            f"{document_id} p.{page}"
            for document_id, page in zip(question["gold_documents"], question["gold_pages"])
        )
        evidence = "\n\n".join(question.get("gold_contexts") or question.get("gold_passages", []))
        audit_rows.append({
            "review_order": index,
            "question_id": qid,
            "category": question["category"],
            "answerable": question["answerable"],
            "question": question["question"],
            "reference_answer": question["reference_answer"],
            "gold_sources": sources,
            "gold_evidence": evidence,
            "question_valid": "",
            "reference_valid": "",
            "evidence_supports_reference": "",
            "audit_notes": "",
        })

        for setup in args.setups:
            ranked = sorted(
                args.models,
                key=lambda model: hashlib.sha256(f"{args.seed}:{qid}:{setup}:{model}".encode()).hexdigest(),
            )
            labels = [chr(ord("A") + number) for number in range(len(ranked))]
            row_key = f"{qid}:{setup}"
            mapping["rows"][row_key] = dict(zip(labels, ranked))
            row = {
                "review_order": len(review_rows) + 1, "question_id": qid, "setup": setup,
                "category": question["category"], "answerable": question["answerable"],
                "question": question["question"], "reference_answer": question["reference_answer"],
                "gold_sources": sources, "gold_evidence": evidence,
            }
            for label, model in zip(labels, ranked):
                row[f"candidate_{label}"] = answers[(qid, setup, model)]["answer"]
                row[f"answer_score_{label}_0_to_2"] = ""
                row[f"citation_score_{label}_0_or_1"] = ""
            row["review_notes"] = ""
            review_rows.append(row)

    write_csv(args.output_dir / "question_audit.csv", list(audit_rows[0]), audit_rows)
    for name in (
        "answer_review_blinded_reviewer_1.csv",
        "answer_review_blinded_reviewer_2.csv",
        "answer_review_blinded_adjudicated.csv",
    ):
        fieldnames = list(review_rows[0])
        for row in review_rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        write_csv(args.output_dir / name, fieldnames, review_rows)
    (args.output_dir / "answer_mapping.json").write_text(
        json.dumps(mapping, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"questions": len(questions), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

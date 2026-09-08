#!/usr/bin/env python3
"""Build blinded paired human-review sheets for the third experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


COMPARISONS = {
    "generator_same_local_context": (
        ("google/gemini-3.8-flash", "tuned_local_hybrid_top5"),
        ("qwen/qwen3.5-9b-local-hybrid-lora", "tuned_local_hybrid_top5"),
    ),
    "complete_local_vs_cloud": (
        ("google/gemini-3.8-flash", "vertex_hybrid_top5"),
        ("qwen/qwen3.5-9b-local-hybrid-lora", "tuned_local_hybrid_top5"),
    ),
    "new_vs_bm25_adapter_same_context": (
        ("qwen/qwen3.5-9b-local-hybrid-lora", "tuned_local_hybrid_top5"),
        ("qwen/qwen3.5-9b-bm25-lora-step500", "tuned_local_hybrid_top5"),
    ),
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="local-rag-human-review-20260907")
    args = parser.parse_args()
    questions = sorted(read_jsonl(args.questions), key=lambda x: x["question_id"])
    answers = {(r["question_id"], r["model"], r["setup"]): r for r in read_jsonl(args.answers) if r.get("status") == "ok"}
    rows, mapping = [], {"seed": args.seed, "comparisons": COMPARISONS, "rows": {}}
    for question in questions:
        sources = "; ".join(f"{d} p.{p}" for d, p in zip(question["gold_documents"], question["gold_pages"]))
        evidence = "\n\n".join(question["gold_passages"])
        for comparison, identities in COMPARISONS.items():
            ordered = sorted(identities, key=lambda x: hashlib.sha256(f"{args.seed}:{question['question_id']}:{comparison}:{x}".encode()).hexdigest())
            candidates = [answers[(question["question_id"], model, setup)]["answer"] for model, setup in ordered]
            row_id = f"{question['question_id']}:{comparison}"
            mapping["rows"][row_id] = {"A": ordered[0], "B": ordered[1]}
            rows.append({
                "review_order": len(rows) + 1, "row_id": row_id, "comparison": comparison,
                "question_id": question["question_id"], "category": question["category"],
                "question": question["question"], "reference_answer": question["reference_answer"],
                "gold_sources": sources, "gold_evidence": evidence,
                "candidate_A": candidates[0], "candidate_B": candidates[1],
                "correctness_A_0_to_2": "", "correctness_B_0_to_2": "",
                "citation_A_0_or_1": "", "citation_B_0_or_1": "",
                "unsupported_claim_A_0_or_1": "", "unsupported_claim_B_0_or_1": "",
                "preferred_candidate_A_B_or_tie": "", "notes": "",
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("reviewer_1.csv", "reviewer_2.csv", "adjudicated.csv"):
        with (args.output_dir / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output_dir / "mapping.json").write_text(json.dumps(mapping, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "questions": len(questions), "comparisons": len(COMPARISONS)}, indent=2))


if __name__ == "__main__":
    main()

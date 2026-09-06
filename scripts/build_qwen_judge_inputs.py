#!/usr/bin/env python3
"""Build independently randomized blind Qwen judge prompts for the complete matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random


SYSTEM = (
    "You are a strict blind evaluator. Score each candidate against the reference and gold evidence. "
    "Use answer_score 2 for fully correct, 1 for partly correct, and 0 for incorrect, unsupported, or "
    "a wrong abstention. Use citation_score 1 only when all needed cited document IDs and pages are "
    "correct, and 0 when any needed citation is missing or wrong. Return only a JSON array with one "
    "object per candidate containing id, answer_score, citation_score, and a rationale of at most 12 "
    "words. Preserve every candidate id."
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260906)
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    by_question = {row["question_id"]: [] for row in questions}
    for row in read_jsonl(args.answers):
        if row.get("status") == "ok":
            by_question[row["question_id"]].append(row)
    rng = random.Random(args.seed)
    mappings = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for question in questions:
            candidates = by_question[question["question_id"]]
            if len(candidates) != 8:
                raise ValueError(f"Expected 8 answers for {question['question_id']}, found {len(candidates)}")
            rng.shuffle(candidates)
            blinded = [(f"a{index:02d}", row) for index, row in enumerate(candidates, 1)]
            passages = "\n".join(
                f"- {document} p.{page}: {quote}"
                for document, page, quote in zip(
                    question["gold_documents"], question["gold_pages"], question["gold_passages"]
                )
            )
            candidate_text = "\n\n".join(f"{label}: {row['answer']}" for label, row in blinded)
            prompt = (
                f"EVALUATION_ID\n{question['question_id']}\n\nQUESTION\n{question['question']}"
                f"\n\nREFERENCE ANSWER\n{question['reference_answer']}\n\nGOLD EVIDENCE\n{passages}"
                f"\n\nCANDIDATES\n{candidate_text}"
            )
            stream.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
            ]}, ensure_ascii=False) + "\n")
            mappings.append({
                "question_id": question["question_id"],
                "labels": {label: {"model": row["model"], "setup": row["setup"]} for label, row in blinded},
            })
    manifest = {
        "seed": args.seed, "judge_model": "Qwen/Qwen3.5-9B", "records": len(mappings),
        "input_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(), "mappings": mappings,
    }
    args.mapping.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("records", "seed", "input_sha256")}, indent=2))


if __name__ == "__main__":
    main()

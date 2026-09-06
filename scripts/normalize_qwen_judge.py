#!/usr/bin/env python3
"""Validate and unblind Qwen judge output into the common score schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from judge_answers_gemini import parse_array


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = read_jsonl(args.raw)
    mappings = json.loads(args.mapping.read_text())["mappings"]
    if len(raw) != 50 or len(mappings) != 50:
        raise ValueError("Judge output/mapping must each contain 50 records")
    mappings_by_id = {row["question_id"]: row for row in mappings}
    output = []
    seen_questions = set()
    for row in raw:
        messages = row.get("messages") or []
        user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
        prefix = "EVALUATION_ID\n"
        if not user.startswith(prefix):
            raise ValueError("Qwen judge output omitted its evaluation ID")
        question_id = user[len(prefix):].split("\n", 1)[0]
        if question_id not in mappings_by_id or question_id in seen_questions:
            raise ValueError(f"Invalid or duplicate evaluation ID: {question_id}")
        seen_questions.add(question_id)
        mapping = mappings_by_id[question_id]
        ratings = parse_array(row.get("response") or "")
        expected = set(mapping["labels"])
        if {rating.get("id") for rating in ratings} != expected:
            raise ValueError(f"Wrong labels for {mapping['question_id']}")
        for rating in ratings:
            if rating.get("answer_score") not in {0, 1, 2} or rating.get("citation_score") not in {0, 1}:
                raise ValueError(f"Invalid score for {mapping['question_id']}: {rating}")
            identity = mapping["labels"][rating["id"]]
            output.append({
                **identity, "question_id": mapping["question_id"], "judge_model": "qwen/qwen3.5-9b-base",
                "answer_score": rating["answer_score"], "citation_score": rating["citation_score"],
                "rationale": rating.get("rationale", ""),
            })
    if seen_questions != set(mappings_by_id):
        raise ValueError("Qwen judge output omitted benchmark questions")
    if len(output) != 400 or len({(r["model"], r["setup"], r["question_id"]) for r in output}) != 400:
        raise ValueError("Unblinded Qwen scores are not a complete 400-record matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    print(json.dumps({"scores": len(output)}, indent=2))


if __name__ == "__main__":
    main()

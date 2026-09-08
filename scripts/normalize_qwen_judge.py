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
    parser.add_argument("--expected-questions", type=int, default=50)
    parser.add_argument("--expected-scores", type=int, default=400)
    args = parser.parse_args()
    raw = read_jsonl(args.raw)
    mappings = json.loads(args.mapping.read_text())["mappings"]
    if len(raw) != args.expected_questions or len(mappings) != args.expected_questions:
        raise ValueError("Judge output/mapping question count is incorrect")
    mappings_by_id = {row.get("evaluation_id", row["question_id"]): row for row in mappings}
    output = []
    seen_evaluations = set()
    for row in raw:
        messages = row.get("messages") or []
        user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
        prefix = "EVALUATION_ID\n"
        if not user.startswith(prefix):
            raise ValueError("Qwen judge output omitted its evaluation ID")
        evaluation_id = user[len(prefix):].split("\n", 1)[0]
        if evaluation_id not in mappings_by_id or evaluation_id in seen_evaluations:
            raise ValueError(f"Invalid or duplicate evaluation ID: {evaluation_id}")
        seen_evaluations.add(evaluation_id)
        mapping = mappings_by_id[evaluation_id]
        ratings = parse_array(row.get("response") or "")
        expected = set(mapping["labels"])
        if {rating.get("id") for rating in ratings} != expected:
            raise ValueError(f"Wrong labels for {mapping['question_id']}")
        for rating in ratings:
            if (rating.get("answer_score") not in {0, 1, 2}
                    or rating.get("citation_score") not in {0, 1}
                    or rating.get("unsupported_claim") not in {0, 1}):
                raise ValueError(f"Invalid score for {mapping['question_id']}: {rating}")
            identity = mapping["labels"][rating["id"]]
            output.append({
                **identity, "question_id": mapping["question_id"], "judge_model": "qwen/qwen3.5-9b-base",
                "answer_score": rating["answer_score"], "citation_score": rating["citation_score"],
                "unsupported_claim": rating["unsupported_claim"],
                "rationale": rating.get("rationale", ""),
            })
    if seen_evaluations != set(mappings_by_id):
        raise ValueError("Qwen judge output omitted evaluation groups")
    unique = len({(r["model"], r["setup"], r["question_id"]) for r in output})
    if len(output) != args.expected_scores or unique != args.expected_scores:
        raise ValueError(f"Unblinded Qwen scores are not a complete {args.expected_scores}-record matrix")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    print(json.dumps({"scores": len(output)}, indent=2))


if __name__ == "__main__":
    main()

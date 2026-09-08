#!/usr/bin/env python3
"""Build one-candidate repair prompts for malformed Qwen judge ratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen_judge_inputs import SYSTEM
from judge_answers_gemini import parse_array


REQUIRED = ("id", "answer_score", "citation_score", "unsupported_claim")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def invalid(rating: dict) -> bool:
    return (
        any(key not in rating for key in REQUIRED)
        or rating.get("answer_score") not in {0, 1, 2}
        or rating.get("citation_score") not in {0, 1}
        or rating.get("unsupported_claim") not in {0, 1}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair-mapping", type=Path, required=True)
    args = parser.parse_args()

    mappings = {
        row.get("evaluation_id", row["question_id"]): row
        for row in json.loads(args.mapping.read_text())["mappings"]
    }
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    answers = {
        (row["question_id"], row["model"], row["setup"]): row["answer"]
        for row in read_jsonl(args.answers) if row.get("status") == "ok"
    }
    repairs = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in read_jsonl(args.raw):
            user = next(message["content"] for message in row["messages"] if message["role"] == "user")
            evaluation_id = user[len("EVALUATION_ID\n"):].split("\n", 1)[0]
            mapping = mappings[evaluation_id]
            question = questions[mapping["question_id"]]
            ratings = parse_array(row.get("response") or "")
            for rating in ratings:
                if not invalid(rating):
                    continue
                original_label = rating.get("id")
                if original_label not in mapping["labels"]:
                    raise ValueError(f"Malformed rating has unknown label: {evaluation_id}/{original_label}")
                identity = mapping["labels"][original_label]
                answer = answers[(mapping["question_id"], identity["model"], identity["setup"])]
                repair_id = f"{evaluation_id}:repair:{original_label}"
                passages = "\n".join(
                    f"- {document} p.{page}: {quote}"
                    for document, page, quote in zip(
                        question["gold_documents"], question["gold_pages"], question["gold_passages"]
                    )
                )
                prompt = (
                    f"EVALUATION_ID\n{repair_id}\n\nQUESTION\n{question['question']}"
                    f"\n\nREFERENCE ANSWER\n{question['reference_answer']}\n\nGOLD EVIDENCE\n{passages}"
                    f"\n\nCANDIDATES\na01: {answer}"
                )
                stream.write(json.dumps({"messages": [
                    {"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt},
                ]}, ensure_ascii=False) + "\n")
                repairs.append({
                    "evaluation_id": repair_id,
                    "original_evaluation_id": evaluation_id,
                    "original_label": original_label,
                    "question_id": mapping["question_id"],
                    "labels": {"a01": identity},
                })
    args.repair_mapping.write_text(json.dumps({"mappings": repairs}, indent=2) + "\n")
    print(json.dumps({"repair_prompts": len(repairs)}, indent=2))


if __name__ == "__main__":
    main()

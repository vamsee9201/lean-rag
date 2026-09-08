#!/usr/bin/env python3
"""Blind-score all local and hosted answers with Gemini through Vertex AI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

from google.genai import types

from gemini_vertex import create_client, generation_config, usage_dict


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/benchmark/questions.jsonl")
    parser.add_argument("--answers", type=Path, default=ROOT / "data/runs/answers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/runs/gemini_judge_scores.jsonl")
    parser.add_argument("--usage-output", type=Path, default=ROOT / "data/runs/gemini_judge_usage.jsonl")
    parser.add_argument("--api-model", default="gemini-3.8-flash")
    parser.add_argument("--location", default="global")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def parse_array(text: str) -> list[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Judge response contained no JSON array")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("Judge response was not an array")
    return value


def main() -> None:
    args = parse_args()
    if args.force:
        args.output.unlink(missing_ok=True)
        args.usage_output.unlink(missing_ok=True)
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    answers_by_question = {question_id: [] for question_id in questions}
    for row in read_jsonl(args.answers):
        if row.get("status") == "ok":
            answers_by_question[row["question_id"]].append(row)
    completed = set()
    if args.output.exists():
        completed = {
            (row["model"], row["setup"], row["question_id"])
            for row in read_jsonl(args.output)
        }
    client = create_client(args.credentials, args.location)
    system = (
        "You are a strict blind evaluator. Score each candidate against the reference and gold evidence. "
        "Use answer_score 2 for fully correct, 1 for partly correct, and 0 for incorrect, unsupported, or "
        "a wrong abstention. For an unanswerable question, only an appropriate abstention earns 2. Use "
        "citation_score 1 only when all needed cited document IDs and pages are correct, and 0 when any "
        "needed citation is missing or wrong. citation_score must never be 2. Set unsupported_claim to 1 "
        "when the answer contains a material factual claim not supported by the gold evidence, otherwise "
        "0. Return only a JSON array with one object per candidate containing id, answer_score, "
        "citation_score, unsupported_claim, and a rationale of at most "
        "12 words. Preserve every candidate id."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20260905)
    with args.output.open("a", encoding="utf-8") as scores, args.usage_output.open(
        "a", encoding="utf-8"
    ) as usage_stream:
        for index, (question_id, question) in enumerate(questions.items(), 1):
            pending = [
                row for row in answers_by_question[question_id]
                if (row["model"], row["setup"], question_id) not in completed
            ]
            if not pending:
                continue
            rng.shuffle(pending)
            blinded = [(f"a{number:02d}", row) for number, row in enumerate(pending, 1)]
            passages = "\n".join(
                f"- {doc} p.{page}: {quote}"
                for doc, page, quote in zip(
                    question["gold_documents"], question["gold_pages"], question["gold_passages"]
                )
            ) or "- No answer exists in the benchmark corpus; the correct behavior is to abstain."
            candidates = "\n\n".join(f"{label}: {row['answer']}" for label, row in blinded)
            prompt = (
                f"QUESTION\n{question['question']}\n\nREFERENCE ANSWER\n{question['reference_answer']}"
                f"\n\nGOLD EVIDENCE\n{passages}\n\nCANDIDATES\n{candidates}"
            )
            expected = {label for label, _row in blinded}
            last_error = None
            for attempt in range(3):
                retry_note = "" if attempt == 0 else (
                    "\n\nCORRECTION: answer_score must be 0, 1, or 2; citation_score must be 0 or 1."
                )
                try:
                    config = generation_config(system, 1800)
                    config.response_mime_type = "application/json"
                    response = client.models.generate_content(
                        model=args.api_model,
                        contents=prompt + retry_note,
                        config=config,
                    )
                    ratings = parse_array(response.text or "")
                    if {rating.get("id") for rating in ratings} != expected:
                        raise ValueError("Judge did not return exactly the blinded candidate IDs")
                    if any(rating.get("answer_score") not in {0, 1, 2} for rating in ratings):
                        raise ValueError("Judge returned an invalid answer_score")
                    if any(rating.get("citation_score") not in {0, 1} for rating in ratings):
                        raise ValueError("Judge returned an invalid citation_score")
                    if any(rating.get("unsupported_claim") not in {0, 1} for rating in ratings):
                        raise ValueError("Judge returned an invalid unsupported_claim")
                    break
                except (IndexError, KeyError, TypeError, ValueError) as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"Gemini judge failed validation after 3 attempts: {last_error}")
            ratings_by_id = {rating["id"]: rating for rating in ratings}
            for label, row in blinded:
                rating = ratings_by_id[label]
                output = {
                    "model": row["model"],
                    "setup": row["setup"],
                    "question_id": question_id,
                    "judge_model": f"google/{args.api_model}",
                    "answer_score": rating["answer_score"],
                    "citation_score": (
                        None if not question["answerable"] else rating["citation_score"]
                    ),
                    "unsupported_claim": rating["unsupported_claim"],
                    "rationale": rating.get("rationale", ""),
                }
                scores.write(json.dumps(output, ensure_ascii=False) + "\n")
            scores.flush()
            usage_stream.write(json.dumps({
                "question_id": question_id,
                "attempt": attempt + 1,
                **usage_dict(response),
            }) + "\n")
            usage_stream.flush()
            print(f"{index}/{len(questions)} {question_id}: {len(pending)} answers judged", flush=True)


if __name__ == "__main__":
    main()

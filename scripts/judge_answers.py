#!/usr/bin/env python3
"""Blind-score all answers in 50 batched calls to the strongest local model."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import random
import re

import requests


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "benchmark" / "questions.jsonl")
    parser.add_argument("--answers", type=Path, default=ROOT / "data" / "runs" / "answers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runs" / "judge_scores.jsonl")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument(
        "--api-model",
        help="Route requests to this loaded API identifier while retaining the judge label",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
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


def judge(base_url: str, model: str, question: dict, blinded: list[tuple[str, dict]]) -> list[dict]:
    passages = "\n".join(
        f"- {doc} p.{page}: {quote}"
        for doc, page, quote in zip(
            question["gold_documents"], question["gold_pages"], question["gold_passages"]
        )
    ) or "- No answer exists in the benchmark corpus; the correct behavior is to abstain."
    candidates = "\n\n".join(f"{label}: {record['answer']}" for label, record in blinded)
    system = (
        "You are a strict blind evaluator. Score each candidate answer against the reference and gold "
        "evidence. Use answer_score 2 for fully correct, 1 for partly correct, and 0 for incorrect, "
        "unsupported, or a wrong abstention. For an unanswerable question, only an appropriate abstention "
        "earns 2. Use citation_score 1 only when all needed cited document IDs and pages are correct, 0 "
        "when citations are missing or wrong, and null when the candidate was explicitly generated without "
        "retrieval context. The only permitted citation_score values here are the integers 0 and 1; never "
        "use 2 for citation_score. Return only a JSON array with one object per candidate containing id, "
        "answer_score, citation_score, and a rationale of at most 12 words. Preserve every candidate id."
    )
    prompt = (
        f"QUESTION\n{question['question']}\n\nREFERENCE ANSWER\n{question['reference_answer']}\n\n"
        f"GOLD EVIDENCE\n{passages}\n\nCANDIDATES\n{candidates}"
    )
    expected = {label for label, _ in blinded}
    last_error = None
    for attempt in range(3):
        retry_note = "" if attempt == 0 else (
            "\n\nCORRECTION: Your previous output violated the schema. "
            "answer_score must be 0, 1, or 2; citation_score must be 0 or 1. "
            f"Return exactly one object for each of these IDs: {', '.join(sorted(expected))}."
        )
        response = requests.post(
            f"{base_url}/api/v1/chat",
            json={
                "model": model,
                "system_prompt": system,
                "input": prompt + retry_note,
                "reasoning": "off",
                "temperature": 0,
                "max_output_tokens": 1800,
                "context_length": 16384,
            },
            timeout=600,
        )
        try:
            if not response.ok:
                raise RuntimeError(f"LM Studio HTTP {response.status_code}: {response.text[:1000]}")
            body = response.json()
            messages = [
                item["content"] for item in body.get("output", []) if item.get("type") == "message"
            ]
            ratings = parse_array(messages[-1])
            if {rating.get("id") for rating in ratings} != expected:
                raise ValueError("Judge did not return exactly the blinded candidate IDs")
            if any(rating.get("answer_score") not in {0, 1, 2} for rating in ratings):
                raise ValueError("Judge returned an invalid answer_score")
            if any(rating.get("citation_score") not in {0, 1} for rating in ratings):
                raise ValueError("Judge returned an invalid citation_score")
            return ratings
        except (IndexError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"Judge failed schema validation after 3 attempts: {last_error}")


def main() -> None:
    args = parse_args()
    if args.force:
        args.output.unlink(missing_ok=True)
    questions = {record["question_id"]: record for record in read_jsonl(args.questions)}
    answers_by_question = {question_id: [] for question_id in questions}
    for record in read_jsonl(args.answers):
        if record.get("status") == "ok":
            answers_by_question[record["question_id"]].append(record)
    completed = set()
    if args.output.exists():
        completed = {
            (record["model"], record["setup"], record["question_id"])
            for record in read_jsonl(args.output)
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Independent from the Gemini judge's label permutation.
    rng = random.Random(20260906)
    jobs = []
    for index, (question_id, question) in enumerate(questions.items(), 1):
        pending = [
            record for record in answers_by_question[question_id]
            if (record["model"], record["setup"], question_id) not in completed
        ]
        if not pending:
            continue
        rng.shuffle(pending)
        blinded = [(f"a{number:02d}", record) for number, record in enumerate(pending, 1)]
        jobs.append((index, question_id, question, blinded))

    def run_one(job):
        index, question_id, question, blinded = job
        ratings = {
            rating["id"]: rating
            for rating in judge(args.base_url, args.api_model or args.model, question, blinded)
        }
        return index, question_id, question, blinded, ratings

    with args.output.open("a", encoding="utf-8") as stream, ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as pool:
        futures = {pool.submit(run_one, job): job for job in jobs}
        failures = []
        for future in as_completed(futures):
            try:
                index, question_id, question, blinded, ratings = future.result()
            except Exception as exc:
                index, question_id, _question, _blinded = futures[future]
                failures.append(question_id)
                print(f"{index}/{len(questions)} {question_id}: ERROR {exc}", flush=True)
                continue
            for label, record in blinded:
                rating = ratings[label]
                output = {
                    "model": record["model"],
                    "setup": record["setup"],
                    "question_id": question_id,
                    "judge_model": args.model,
                    "answer_score": rating["answer_score"],
                    "citation_score": (
                        None
                        if record["setup"] == "no_context" or not question["answerable"]
                        else rating["citation_score"]
                    ),
                    "rationale": rating["rationale"],
                }
                stream.write(json.dumps(output, ensure_ascii=False) + "\n")
            stream.flush()
            print(
                f"{index}/{len(questions)} {question_id}: {len(blinded)} answers judged",
                flush=True,
            )
    if failures:
        raise RuntimeError(f"Judge failed for {len(failures)} questions: {failures}")


if __name__ == "__main__":
    main()

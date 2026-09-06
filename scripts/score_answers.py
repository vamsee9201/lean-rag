#!/usr/bin/env python3
"""Compute deterministic answer, abstention, citation, and performance metrics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "benchmark" / "questions.jsonl")
    parser.add_argument("--answers", type=Path, default=ROOT / "data" / "runs" / "answers.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "runs" / "automatic_scores.jsonl")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def token_f1(answer: str, reference: str) -> float:
    answer_tokens, reference_tokens = tokens(answer), tokens(reference)
    if not answer_tokens or not reference_tokens:
        return 0.0
    answer_counts = {token: answer_tokens.count(token) for token in set(answer_tokens)}
    reference_counts = {token: reference_tokens.count(token) for token in set(reference_tokens)}
    overlap = sum(min(answer_counts.get(token, 0), count) for token, count in reference_counts.items())
    precision, recall = overlap / len(answer_tokens), overlap / len(reference_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def abstained(answer: str) -> bool:
    normalized = answer.casefold()
    phrases = (
        "insufficient evidence", "not enough evidence", "does not provide", "doesn't provide",
        "not provided", "cannot determine", "can't determine", "does not state", "doesn't state",
    )
    return any(phrase in normalized for phrase in phrases)


def score(answer_record: dict, question: dict) -> dict:
    answer = answer_record["answer"]
    answer_tokens = tokens(answer)
    reference_tokens = tokens(question["reference_answer"])
    reference_hit = bool(reference_tokens) and " ".join(reference_tokens) in " ".join(answer_tokens)
    cited_sources = []
    for document_id, page in zip(question["gold_documents"], question["gold_pages"]):
        document_hit = document_id.casefold() in answer.casefold()
        page_hit = bool(re.search(rf"(?:p\.?\s*|page\s*=*\s*){page}\b", answer, re.I))
        cited_sources.append(document_hit and page_hit)
    is_abstention = abstained(answer)
    deterministically_scorable = question["category"] in {"direct", "numeric_date", "unanswerable"}
    if question["category"] == "unanswerable":
        deterministic_correct = is_abstention
    elif deterministically_scorable:
        deterministic_correct = reference_hit
    else:
        deterministic_correct = None
    stats = answer_record.get("stats", {})
    return {
        "model": answer_record["model"],
        "setup": answer_record["setup"],
        "question_id": answer_record["question_id"],
        "category": question["category"],
        "answerable": question["answerable"],
        "reference_hit": reference_hit,
        "token_f1": token_f1(answer, question["reference_answer"]),
        "abstained": is_abstention,
        "deterministic_correct": deterministic_correct,
        "citation_recall": None if not cited_sources else sum(cited_sources) / len(cited_sources),
        "elapsed_seconds": answer_record.get("elapsed_seconds"),
        "input_tokens": stats.get("input_tokens"),
        "output_tokens": stats.get("output_tokens"),
        "tokens_per_second": stats.get("tokens_per_second"),
    }


def mean(values: list[float | int | bool | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return None if not usable else sum(usable) / len(usable)


def main() -> None:
    args = parse_args()
    questions = {record["question_id"]: record for record in read_jsonl(args.questions)}
    scores = [
        score(record, questions[record["question_id"]])
        for record in read_jsonl(args.answers)
        if record.get("status") == "ok"
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in scores:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in scores:
        groups[(record["model"], record["setup"])].append(record)
    summary = {}
    for (model, setup), records in groups.items():
        summary.setdefault(model, {})[setup] = {
            "completed": len(records),
            "deterministic_accuracy": mean([record["deterministic_correct"] for record in records]),
            "mean_token_f1": mean([record["token_f1"] for record in records]),
            "answerable_abstention_rate": mean([
                record["abstained"] for record in records if record["answerable"]
            ]),
            "unanswerable_abstention_accuracy": mean([
                record["abstained"] for record in records if not record["answerable"]
            ]),
            "citation_recall": mean([record["citation_recall"] for record in records]),
            "mean_elapsed_seconds": mean([record["elapsed_seconds"] for record in records]),
            "mean_tokens_per_second": mean([record["tokens_per_second"] for record in records]),
        }
    summary_path = args.output.parent / "automatic_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate source-grounded benchmark question candidates with a local model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import re
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES = ROOT / "data" / "processed" / "pilot30" / "pages.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark" / "direct_candidates.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:1234"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--exclude-questions", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    return parser.parse_args()


def compact(text: str) -> str:
    return " ".join(text.split())


def parse_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Response was not a JSON object")
    return value


def read_eligible_pages(path: Path) -> list[dict]:
    pages: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            words = page["text"].split()
            if page["quality_flags"] or not (180 <= len(words) <= 900):
                continue
            pages.append(page)
    return pages


def stratified_candidates(pages: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        groups[page["collection"]].append(page)
    for group in groups.values():
        rng.shuffle(group)
    ordered: list[dict] = []
    used_documents: set[str] = set()
    while any(groups.values()):
        for collection in sorted(groups):
            group = groups[collection]
            if not group:
                continue
            preferred = next(
                (index for index, page in enumerate(group) if page["document_id"] not in used_documents),
                0,
            )
            page = group.pop(preferred)
            ordered.append(page)
            used_documents.add(page["document_id"])
    return ordered


def request_question(base_url: str, model: str, page: dict) -> dict:
    system_prompt = (
        "Create one objective factual question from the supplied government-document page. "
        "The question must be answerable from this page alone. Return only a JSON object with "
        "the string fields question, reference_answer, and evidence_quote. evidence_quote must "
        "be one short, exact, continuous quote copied from the page. reference_answer must be "
        "a concise substring of that quote. Do not ask about document formatting, page numbers, "
        "publication metadata, or facts that require outside knowledge."
    )
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "input": page["text"][:12000],
        "reasoning": "off",
        "temperature": 0,
        "max_output_tokens": 300,
        "context_length": 4096,
    }
    response = requests.post(f"{base_url}/api/v1/chat", json=payload, timeout=300)
    if not response.ok:
        raise RuntimeError(f"LM Studio HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    messages = [item["content"] for item in body.get("output", []) if item.get("type") == "message"]
    if not messages:
        raise ValueError("Model returned no message")
    question = parse_object(messages[-1])
    question["generation_stats"] = body.get("stats", {})
    return question


def validate_question(question: dict, page: dict) -> None:
    required = ("question", "reference_answer", "evidence_quote")
    if any(not isinstance(question.get(key), str) or not question[key].strip() for key in required):
        raise ValueError("Required question fields are missing")
    page_text = compact(page["text"])
    quote = compact(question["evidence_quote"])
    answer = compact(question["reference_answer"])
    if quote not in page_text or answer.casefold() not in quote.casefold():
        answer_start = page_text.casefold().find(answer.casefold())
        if answer_start < 0:
            raise ValueError("Reference answer is not present in the source page")
        start = max(0, answer_start - 300)
        end = min(len(page_text), answer_start + len(answer) + 300)
        if start:
            start = page_text.find(" ", start) + 1
        if end < len(page_text):
            next_space = page_text.find(" ", end)
            end = len(page_text) if next_space < 0 else next_space
        question["evidence_quote"] = page_text[start:end]
    if len(question["question"].split()) < 4 or not question["question"].rstrip().endswith("?"):
        raise ValueError("Question is malformed")


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force to replace it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages = read_eligible_pages(args.pages)
    excluded_questions = set()
    if args.exclude_questions:
        excluded_records = [
            json.loads(line) for line in args.exclude_questions.read_text().splitlines() if line.strip()
        ]
        excluded = {
            (document_id, page)
            for record in excluded_records
            for document_id, page in zip(record.get("gold_documents", []), record.get("gold_pages", []))
        }
        excluded_questions = {
            compact(record["question"]).casefold()
            for record in excluded_records if isinstance(record.get("question"), str)
        }
        pages = [page for page in pages if (page["document_id"], page["page"]) not in excluded]
    candidates = stratified_candidates(pages, args.seed)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    accepted = 0
    attempted = 0

    def generate(page: dict) -> tuple[dict | None, str | None]:
        error = None
        for retry in range(3):
            try:
                question = request_question(args.base_url, args.model, page)
                validate_question(question, page)
                return question, None
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(0.2 * (retry + 1))
        return None, error

    with temporary.open("w", encoding="utf-8") as stream:
        candidate_iter = iter(candidates)
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            while accepted < args.count:
                batch = []
                for _ in range(args.concurrency):
                    page = next(candidate_iter, None)
                    if page is not None:
                        batch.append(page)
                if not batch:
                    break
                attempted += len(batch)
                for page, (question, error) in zip(batch, executor.map(generate, batch)):
                    if question is None:
                        print(f"rejected {page['document_id']} p{page['page']}: {error}", flush=True)
                        continue
                    normalized_question = compact(question["question"]).casefold()
                    if normalized_question in excluded_questions:
                        print(f"rejected {page['document_id']} p{page['page']}: duplicate question", flush=True)
                        continue
                    if accepted >= args.count:
                        continue
                    record = {
                        "question_id": f"direct-{accepted + 1:03d}",
                        "category": "direct",
                        "question": question["question"].strip(),
                        "reference_answer": question["reference_answer"].strip(),
                        "gold_documents": [page["document_id"]],
                        "gold_pages": [page["page"]],
                        "gold_passages": [question["evidence_quote"].strip()],
                        "collection": page["collection"],
                        "generator_model": args.model,
                        "generation_stats": question["generation_stats"],
                        "review_status": "unreviewed",
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    accepted += 1
                    excluded_questions.add(normalized_question)
                    print(f"{accepted}/{args.count} {record['question_id']} {page['document_id']} p{page['page']}", flush=True)
    if accepted < args.count:
        raise RuntimeError(f"Only generated {accepted}/{args.count} valid questions from {attempted} pages")
    temporary.replace(args.output)
    print(json.dumps({"accepted": accepted, "attempted_pages": attempted, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

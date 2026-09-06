#!/usr/bin/env python3
"""Generate grounded candidates for the non-direct benchmark categories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import re
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAGES = ROOT / "data" / "processed" / "pilot30" / "pages.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark" / "advanced_candidates.jsonl"
DEFAULT_BASE_URL = "http://127.0.0.1:1234"

CATEGORY_COUNTS = {
    "numeric_date": 14,
    "single_document_multi_passage": 13,
    "cross_document_comparison": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=DEFAULT_PAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-questions", type=Path)
    parser.add_argument("--numeric-count", type=int, default=CATEGORY_COUNTS["numeric_date"])
    parser.add_argument(
        "--multi-passage-count", type=int, default=CATEGORY_COUNTS["single_document_multi_passage"]
    )
    parser.add_argument(
        "--cross-document-count", type=int, default=CATEGORY_COUNTS["cross_document_comparison"]
    )
    return parser.parse_args()


def compact(text: str) -> str:
    return " ".join(text.split())


def parse_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("Response did not contain a JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Response was not a JSON object")
    return value


def read_pages(path: Path) -> list[dict]:
    pages = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            words = page["text"].split()
            if not page["quality_flags"] and 180 <= len(words) <= 1_000:
                pages.append(page)
    return pages


def call_model(base_url: str, model: str, system_prompt: str, inputs: str) -> tuple[dict, dict]:
    payload = {
        "model": model,
        "system_prompt": system_prompt,
        "input": inputs,
        "reasoning": "off",
        "temperature": 0,
        "max_output_tokens": 500,
        "context_length": 8192,
    }
    response = requests.post(f"{base_url}/api/v1/chat", json=payload, timeout=300)
    if not response.ok:
        raise RuntimeError(f"LM Studio HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    messages = [item["content"] for item in body.get("output", []) if item.get("type") == "message"]
    if not messages:
        raise ValueError("Model returned no message")
    return parse_object(messages[-1]), body.get("stats", {})


def validate_common(candidate: dict) -> None:
    for key in ("question", "reference_answer"):
        if not isinstance(candidate.get(key), str) or not candidate[key].strip():
            raise ValueError(f"Missing {key}")
    if not candidate["question"].rstrip().endswith("?"):
        raise ValueError("Question is malformed")


def validate_quote(quote: str, page: dict) -> str:
    normalized = compact(quote)
    if normalized not in compact(page["text"]):
        raise ValueError("Evidence is not an exact continuous quote")
    return quote.strip()


def make_record(category: str, number: int, candidate: dict, pages: list[dict], stats: dict, model: str) -> dict:
    validate_common(candidate)
    if category == "numeric_date":
        answer = compact(candidate["reference_answer"])
        page_text = compact(pages[0]["text"])
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
        quotes = [page_text[start:end]]
    else:
        quotes = candidate.get("evidence_quotes")
        if quotes is None and len(pages) == 1 and isinstance(candidate.get("evidence_quote"), str):
            quotes = [candidate["evidence_quote"]]
        if len(pages) == 1 and isinstance(quotes, str):
            quotes = [quotes]
        if not isinstance(quotes, list) or len(quotes) != len(pages):
            raise ValueError("Expected one evidence quote per source page")
        quotes = [validate_quote(quote, page) for quote, page in zip(quotes, pages)]
    contexts = []
    for quote, page in zip(quotes, pages):
        text, needle = compact(page["text"]), compact(quote)
        start = text.index(needle)
        contexts.append(text[max(0, start - 450) : start + len(needle) + 450])
    return {
        "question_id": f"{category}-{number:03d}",
        "category": category,
        "question": candidate["question"].strip(),
        "reference_answer": candidate["reference_answer"].strip(),
        "gold_documents": [page["document_id"] for page in pages],
        "gold_pages": [page["page"] for page in pages],
        "gold_passages": quotes,
        "gold_contexts": contexts,
        "collections": [page["collection"] for page in pages],
        "generator_model": model,
        "generation_stats": stats,
        "review_status": "unreviewed",
    }


def numeric_inputs(pages: list[dict], rng: random.Random):
    pattern = re.compile(r"(?:\$|\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:percent|million|billion|days?|years?)\b)", re.I)
    eligible = [page for page in pages if pattern.search(page["text"])]
    rng.shuffle(eligible)
    for page in eligible:
        yield [page]


def same_document_inputs(pages: list[dict], rng: random.Random):
    grouped: dict[str, list[dict]] = defaultdict(list)
    for page in pages:
        grouped[page["document_id"]].append(page)
    document_ids = [doc for doc, group in grouped.items() if len(group) >= 4]
    rng.shuffle(document_ids)
    while document_ids:
        for doc in list(document_ids):
            group = grouped[doc]
            pairs = [(a, b) for a, b in zip(group, group[1:]) if 1 <= b["page"] - a["page"] <= 4]
            if not pairs:
                document_ids.remove(doc)
                continue
            pair = rng.choice(pairs)
            for page in pair:
                group.remove(page)
            yield list(pair)


def cross_document_inputs(pages: list[dict], rng: random.Random):
    by_collection: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for page in pages:
        by_collection[page["collection"]][page["document_id"]].append(page)
    pairs = []
    for collection, documents in by_collection.items():
        ids = list(documents)
        if len(ids) < 2:
            continue
        for left, right in zip(ids, ids[1:]):
            pairs.append((collection, documents[left], documents[right]))
    rng.shuffle(pairs)
    for _, left, right in pairs:
        yield [rng.choice(left), rng.choice(right)]


PROMPTS = {
    "numeric_date": (
        "Create one self-contained factual question whose answer is a number, amount, percentage, "
        "duration, or date in the supplied government-document page. Return only JSON with string "
        "fields question and reference_answer, plus evidence_quotes as a one-element string array. "
        "The quote must be short, exact, continuous source text and must contain the answer and enough "
        "labels or units to remove ambiguity. Avoid table values whose row or column label is missing."
    ),
    "single_document_multi_passage": (
        "Create one self-contained question that can only be fully answered by combining one fact from "
        "each of the two supplied pages from the same government document. Return only JSON with string "
        "fields question and reference_answer, plus evidence_quotes as exactly two strings in PAGE A, "
        "PAGE B order. Each quote must be short, exact, continuous source text. The answer must explicitly "
        "combine both facts. Do not merely ask two unrelated questions joined by 'and'."
    ),
    "cross_document_comparison": (
        "Create one self-contained comparison question that requires one fact from each of the two "
        "different supplied government documents. Return only JSON with string fields question and "
        "reference_answer, plus evidence_quotes as exactly two strings in DOCUMENT A, DOCUMENT B order. "
        "Each quote must be short, exact, continuous source text. State both document titles in the "
        "question and make the answer explicitly compare their facts."
    ),
}


def formatted_pages(pages: list[dict]) -> str:
    labels = ["PAGE A", "PAGE B"] if len({p["document_id"] for p in pages}) == 1 else ["DOCUMENT A", "DOCUMENT B"]
    if len(pages) == 1:
        labels = ["PAGE"]
    sections = []
    for label, page in zip(labels, pages):
        sections.append(
            f"{label}\nTITLE: {page['title']}\nDOCUMENT ID: {page['document_id']}\n"
            f"PAGE: {page['page']}\nTEXT:\n{page['text'][:16000]}"
        )
    return "\n\n".join(sections)


def evidence_snippet(page: dict) -> str:
    """Choose a compact, exact fact-bearing span before asking the model to combine it."""
    text = compact(page["text"])
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    candidates = [sentence for sentence in sentences if 18 <= len(sentence.split()) <= 70]
    if not candidates:
        words = text.split()
        return " ".join(words[: min(60, len(words))])
    factual = [sentence for sentence in candidates if re.search(r"\d|\b(?:is|was|were|has|have|reported|found)\b", sentence, re.I)]
    return (factual or candidates)[len(factual or candidates) // 2]


def formatted_evidence(pages: list[dict], snippets: list[str]) -> str:
    different_documents = len({page["document_id"] for page in pages}) > 1
    labels = ["DOCUMENT A", "DOCUMENT B"] if different_documents else ["PAGE A", "PAGE B"]
    sections = []
    for label, page, snippet in zip(labels, pages, snippets):
        sections.append(
            f"{label}\nTITLE: {page['title']}\nDOCUMENT ID: {page['document_id']}\n"
            f"PAGE: {page['page']}\nSOURCE FACT:\n{snippet}"
        )
    return "\n\n".join(sections)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force and not args.resume:
        raise SystemExit(f"Output exists: {args.output}; pass --force to replace it")
    pages = read_pages(args.pages)
    if args.exclude_questions:
        excluded = {
            (document_id, page)
            for record in (
                json.loads(line) for line in args.exclude_questions.read_text().splitlines() if line.strip()
            )
            for document_id, page in zip(record.get("gold_documents", []), record.get("gold_pages", []))
        }
        pages = [page for page in pages if (page["document_id"], page["page"]) not in excluded]
    category_counts = {
        "numeric_date": args.numeric_count,
        "single_document_multi_passage": args.multi_passage_count,
        "cross_document_comparison": args.cross_document_count,
    }
    rng = random.Random(args.seed)
    generators = {
        "numeric_date": numeric_inputs(pages, rng),
        "single_document_multi_passage": same_document_inputs(pages, rng),
        "cross_document_comparison": cross_document_inputs(pages, rng),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    existing = []
    if args.resume:
        source = temporary if temporary.exists() else args.output
        if source.exists():
            with source.open(encoding="utf-8") as prior:
                existing = [json.loads(line) for line in prior if line.strip()]
    mode = "a" if existing and temporary.exists() else "w"
    if existing and not temporary.exists():
        with temporary.open("w", encoding="utf-8") as stream:
            for record in existing:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        mode = "a"
    with temporary.open(mode, encoding="utf-8") as stream:
        for category, target in category_counts.items():
            accepted = sum(record["category"] == category for record in existing)
            used_sources = {
                tuple(zip(record["gold_documents"], record["gold_pages"]))
                for record in existing
                if record["category"] == category
            }
            attempted = 0
            for source_pages in generators[category]:
                if accepted >= target:
                    break
                source_key = tuple((page["document_id"], page["page"]) for page in source_pages)
                if source_key in used_sources:
                    continue
                attempted += 1
                try:
                    inputs = formatted_pages(source_pages)
                    snippets = None
                    if category != "numeric_date":
                        snippets = [evidence_snippet(page) for page in source_pages]
                        inputs = formatted_evidence(source_pages, snippets)
                    candidate, stats = call_model(
                        args.base_url, args.model, PROMPTS[category], inputs
                    )
                    if snippets is not None:
                        candidate["evidence_quotes"] = snippets
                    record = make_record(category, accepted + 1, candidate, source_pages, stats, args.model)
                except Exception as exc:
                    print(f"rejected {category} attempt {attempted}: {type(exc).__name__}: {exc}", flush=True)
                    time.sleep(0.2)
                    continue
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                used_sources.add(source_key)
                accepted += 1
                print(f"{category}: {accepted}/{target} ({attempted} attempted)", flush=True)
            if accepted < target:
                raise RuntimeError(f"Only generated {accepted}/{target} {category} questions")
    temporary.replace(args.output)
    print(json.dumps({"counts": category_counts, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

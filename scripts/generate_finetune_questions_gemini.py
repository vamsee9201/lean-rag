#!/usr/bin/env python3
"""Generate verified question/answer sources for RAFT-style RAG fine-tuning."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import threading

from gemini_vertex import create_client, generation_config, usage_dict


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COUNTS = {"direct": 8, "numeric_date": 4, "multi_passage": 4, "cross_document": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--pages", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--api-model", default="gemini-3.8-flash")
    parser.add_argument("--location", default="global")
    parser.add_argument("--direct", type=int, default=DEFAULT_COUNTS["direct"])
    parser.add_argument("--numeric-date", type=int, default=DEFAULT_COUNTS["numeric_date"])
    parser.add_argument("--multi-passage", type=int, default=DEFAULT_COUNTS["multi_passage"])
    parser.add_argument("--cross-document", type=int, default=DEFAULT_COUNTS["cross_document"])
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--exclude-documents-from",
        type=Path,
        help="JSONL questions whose gold document IDs must not be used as new sources",
    )
    return parser.parse_args()


def compact(text: str) -> str:
    return " ".join(text.split())


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


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
            word_count = len(page.get("text", "").split())
            if page.get("quality_flags") or not 180 <= word_count <= 900:
                continue
            pages.append(page)
    return pages


def source_pools(pages: list[dict], seed: int) -> dict[str, list[list[dict]]]:
    ordered = sorted(
        pages,
        key=lambda page: stable_key(seed, f"{page['document_id']}:{page['page']}"),
    )
    numeric_pattern = re.compile(
        r"(?:\$|\b\d{4}\b|\b\d+(?:\.\d+)?\s*(?:percent|million|billion|days?|years?)\b)",
        re.I,
    )
    numeric = [page for page in ordered if numeric_pattern.search(page["text"])]

    by_document: dict[str, list[dict]] = defaultdict(list)
    by_collection: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for page in ordered:
        by_document[page["document_id"]].append(page)
        by_collection[page["collection"]][page["document_id"]].append(page)
    same_document = []
    for document_id in sorted(by_document, key=lambda value: stable_key(seed + 1, value)):
        group = sorted(by_document[document_id], key=lambda page: page["page"])
        pairs = [
            [left, right]
            for left, right in zip(group, group[1:])
            if 1 <= right["page"] - left["page"] <= 4
        ]
        same_document.extend(pairs)
    same_document.sort(
        key=lambda pair: stable_key(seed + 2, ":".join(f"{p['document_id']}:{p['page']}" for p in pair))
    )

    stopwords = {
        "about", "after", "also", "been", "before", "between", "could", "from", "have",
        "into", "more", "most", "other", "report", "shall", "that", "their", "there", "these",
        "they", "this", "through", "under", "upon", "were", "which", "with", "would",
    }

    def topical_terms(page: dict) -> set[str]:
        text = f"{page['title']} {page['text'][:5000]}".casefold()
        return {
            token for token in re.findall(r"[a-z][a-z-]{3,}", text)
            if token not in stopwords
        }

    cross_document = []
    for collection, documents in sorted(by_collection.items()):
        document_ids = sorted(documents, key=lambda value: stable_key(seed + 3, value))
        representatives = {
            document_id: min(
                documents[document_id],
                key=lambda page: stable_key(seed + 4, f"{document_id}:{page['page']}"),
            )
            for document_id in document_ids
        }
        terms = {document_id: topical_terms(page) for document_id, page in representatives.items()}
        candidate_pairs = []
        for left_index, left_id in enumerate(document_ids):
            for right_id in document_ids[left_index + 1 :]:
                shared = terms[left_id] & terms[right_id]
                union = terms[left_id] | terms[right_id]
                score = len(shared) / max(1, len(union))
                candidate_pairs.append((score, stable_key(seed + 5, f"{left_id}:{right_id}"), left_id, right_id))
        used_documents = set()
        for _score, _tie, left_id, right_id in sorted(candidate_pairs, reverse=True):
            if left_id in used_documents or right_id in used_documents:
                continue
            cross_document.append([representatives[left_id], representatives[right_id]])
            used_documents.update((left_id, right_id))
    cross_document.sort(
        key=lambda pair: stable_key(seed + 6, ":".join(f"{p['document_id']}:{p['page']}" for p in pair))
    )
    return {
        "direct": [[page] for page in ordered],
        "numeric_date": [[page] for page in numeric],
        "multi_passage": same_document,
        "cross_document": cross_document,
    }


PROMPTS = {
    "direct": (
        "Create one self-contained factual question answerable from the supplied page. Return only "
        "JSON with string fields question and reference_answer and an evidence_quotes array containing "
        "one short, exact, continuous quote. The answer must be a concise substring of that quote."
    ),
    "numeric_date": (
        "Create one self-contained question whose answer is a number, amount, percentage, duration, or "
        "date in the supplied page. Return only JSON with string fields question and reference_answer "
        "and a one-element evidence_quotes array. Include labels and units in the exact source quote."
    ),
    "multi_passage": (
        "Create one question that requires combining one fact from each of two pages from the same "
        "document. Return only JSON with string fields question and reference_answer and exactly two "
        "short, exact evidence_quotes in PAGE A, PAGE B order. The answer must combine both facts."
    ),
    "cross_document": (
        "Create one comparison question requiring one fact from each supplied document. Return only "
        "JSON with string fields question and reference_answer and exactly two short, exact "
        "evidence_quotes in DOCUMENT A, DOCUMENT B order. Name both document titles in the question."
    ),
}


def format_sources(pages: list[dict]) -> str:
    labels = ["PAGE"] if len(pages) == 1 else (
        ["PAGE A", "PAGE B"]
        if len({page["document_id"] for page in pages}) == 1
        else ["DOCUMENT A", "DOCUMENT B"]
    )
    return "\n\n".join(
        f"{label}\nTITLE: {page['title']}\nDOCUMENT ID: {page['document_id']}\n"
        f"PAGE: {page['page']}\nTEXT:\n{page['text'][:14000]}"
        for label, page in zip(labels, pages)
    )


def validate(category: str, candidate: dict, pages: list[dict]) -> list[str]:
    question = candidate.get("question")
    answer = candidate.get("reference_answer")
    quotes = candidate.get("evidence_quotes")
    if not isinstance(question, str) or not question.strip().endswith("?"):
        raise ValueError("Question is missing or malformed")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Reference answer is missing")
    if not isinstance(quotes, list) or len(quotes) != len(pages):
        raise ValueError("Expected one evidence quote per source page")
    normalized_quotes = []
    for quote, page in zip(quotes, pages):
        if not isinstance(quote, str) or compact(quote) not in compact(page["text"]):
            raise ValueError("Evidence quote is not exact source text")
        normalized_quotes.append(quote.strip())
    if category in {"direct", "numeric_date"} and compact(answer).casefold() not in compact(quotes[0]).casefold():
        raise ValueError("Direct reference answer is not present in its evidence quote")
    return normalized_quotes


def main() -> None:
    args = parse_args()
    pages_path = args.pages or ROOT / "data" / "finetuning" / args.split / "pages.jsonl"
    output = args.output or ROOT / "data" / "finetuning" / args.split / "question_sources.jsonl"
    if output.exists() and args.force:
        output.unlink()
    errors_path = output.with_name("question_sources_errors.jsonl")
    if errors_path.exists() and args.force:
        errors_path.unlink()
    existing = []
    if output.exists():
        existing = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    completed_sources = {
        (row["category"], tuple(zip(row["gold_documents"], row["gold_pages"]))) for row in existing
    }
    used_source_pages = {
        (document_id, page)
        for row in existing
        for document_id, page in zip(row["gold_documents"], row["gold_pages"])
    }
    prior_errors = []
    if errors_path.exists():
        prior_errors = [json.loads(line) for line in errors_path.read_text().splitlines() if line.strip()]
        completed_sources.update(
            (row["category"], tuple(tuple(item) for item in row["sources"])) for row in prior_errors
        )
        used_source_pages.update(tuple(item) for row in prior_errors for item in row["sources"])
    completed_questions = {compact(row["question"]).casefold() for row in existing}
    targets = {
        "direct": args.direct,
        "numeric_date": args.numeric_date,
        "multi_passage": args.multi_passage,
        "cross_document": args.cross_document,
    }
    pages = read_pages(pages_path)
    if args.exclude_documents_from:
        excluded_documents = {
            document_id
            for row in (
                json.loads(line) for line in args.exclude_documents_from.read_text().splitlines() if line.strip()
            )
            for document_id in row["gold_documents"]
        }
        pages = [page for page in pages if page["document_id"] not in excluded_documents]
    pools = source_pools(pages, args.seed)
    tasks = []
    for category, target in targets.items():
        have = sum(row["category"] == category for row in existing)
        if have >= target:
            continue
        for source_pages in pools[category]:
            source_key = tuple((page["document_id"], page["page"]) for page in source_pages)
            if (category, source_key) in completed_sources:
                continue
            if any(key in used_source_pages for key in source_key):
                continue
            tasks.append((category, source_pages))
            used_source_pages.update(source_key)
            have += 1
            if have >= target:
                break
        if have < target:
            raise RuntimeError(f"Not enough eligible sources for {category}: {have}/{target}")

    client = create_client(args.credentials, args.location)
    lock = threading.Lock()

    def generate(task: tuple[str, list[dict]]) -> dict:
        category, source_pages = task
        response = client.models.generate_content(
            model=args.api_model,
            contents=format_sources(source_pages),
            config=generation_config(PROMPTS[category], 500),
        )
        candidate = parse_object(response.text or "")
        quotes = validate(category, candidate, source_pages)
        return {
            "category": category,
            "question": candidate["question"].strip(),
            "reference_answer": candidate["reference_answer"].strip(),
            "gold_documents": [page["document_id"] for page in source_pages],
            "gold_pages": [page["page"] for page in source_pages],
            "gold_passages": quotes,
            "collections": [page["collection"] for page in source_pages],
            "generator_model": args.api_model,
            "generation_usage": usage_dict(response),
            "split": args.split,
            "review_status": "unreviewed",
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    with output.open("a", encoding="utf-8") as stream, errors_path.open("a", encoding="utf-8") as error_stream:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            future_tasks = {executor.submit(generate, task): task for task in tasks}
            for future in as_completed(future_tasks):
                category, source_pages = future_tasks[future]
                try:
                    record = future.result()
                    normalized = compact(record["question"]).casefold()
                    with lock:
                        if normalized in completed_questions:
                            raise ValueError("Duplicate question")
                        number = sum(row["category"] == category for row in existing) + 1
                        record["question_id"] = f"{args.split}-{category}-{number:04d}"
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        existing.append(record)
                        completed_questions.add(normalized)
                        print(f"accepted {record['question_id']}", flush=True)
                except Exception as exc:
                    source = ",".join(f"{p['document_id']}:p{p['page']}" for p in source_pages)
                    error_record = {
                        "category": category,
                        "sources": [[p["document_id"], p["page"]] for p in source_pages],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    errors.append(error_record)
                    error_stream.write(json.dumps(error_record, ensure_ascii=False) + "\n")
                    error_stream.flush()
                    print(f"rejected {category} {source}: {type(exc).__name__}: {exc}", flush=True)

    counts = Counter(row["category"] for row in existing)
    summary = {
        "split": args.split,
        "accepted": len(existing),
        "counts": dict(counts),
        "rejected_this_run": len(errors),
        "targets": targets,
        "output": str(output),
    }
    output.with_name("question_sources_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if any(counts[category] < target for category, target in targets.items()):
        raise RuntimeError("One or more category targets were not reached; rerun to resume")


if __name__ == "__main__":
    main()

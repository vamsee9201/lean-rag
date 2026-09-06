#!/usr/bin/env python3
"""Generate plausible missing-fact questions and validate absence in the scoped document."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import time

import requests

from gemini_vertex import create_client, generation_config


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", type=Path, default=ROOT / "data/processed/corpus500/pages.jsonl")
    parser.add_argument(
        "--source-questions", type=Path,
        default=ROOT / "data/confirmatory/benchmark/direct_candidates.jsonl",
    )
    parser.add_argument(
        "--exclude-questions", type=Path, default=ROOT / "data/benchmark/questions.jsonl"
    )
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "data/confirmatory/benchmark/unanswerable_candidates.jsonl",
    )
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--generator-model", default="qwen/qwen3.5-2b")
    parser.add_argument("--validator-model", default="gemini-3.8-flash")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--count", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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


def propose(args: argparse.Namespace, page: dict) -> dict:
    system = (
        "Create one plausible, objective question about a specific fact that is missing from the supplied "
        "government-document page. The question must relate directly to the page's subject, name the "
        "document title, and ask for a concrete name, amount, date, count, duration, or outcome that the "
        "text does not state. Avoid future-year tricks, opinions, and questions that merely negate a stated "
        "fact. Return only JSON with string fields question and missing_fact_rationale."
    )
    response = requests.post(
        f"{args.base_url}/api/v1/chat",
        json={
            "model": args.generator_model,
            "system_prompt": system,
            "input": f"TITLE: {page['title']}\nDOCUMENT ID: {page['document_id']}\nTEXT:\n{page['text'][:12000]}",
            "reasoning": "off",
            "temperature": 0,
            "max_output_tokens": 250,
            "context_length": 8192,
        },
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(f"LM Studio HTTP {response.status_code}: {response.text[:500]}")
    body = response.json()
    messages = [item["content"] for item in body.get("output", []) if item.get("type") == "message"]
    candidate = parse_object(messages[-1])
    if not isinstance(candidate.get("question"), str) or not candidate["question"].endswith("?"):
        raise ValueError("Malformed question")
    if not isinstance(candidate.get("missing_fact_rationale"), str):
        raise ValueError("Missing rationale")
    return candidate


def validate_absence(client, model: str, question: str, document: str) -> dict:
    system = (
        "Determine whether the supplied document explicitly answers the question. Return only a JSON object "
        "with answerable as a boolean and evidence_quote as a string. Set answerable true if any passage "
        "states the requested fact, even indirectly, and copy the shortest supporting quote. Set answerable "
        "false only when the requested fact is absent, and use an empty evidence_quote."
    )
    config = generation_config(system, 300)
    config.response_mime_type = "application/json"
    response = client.models.generate_content(
        model=model,
        contents=f"QUESTION\n{question}\n\nDOCUMENT\n{document}",
        config=config,
    )
    verdict = parse_object(response.text or "")
    if not isinstance(verdict.get("answerable"), bool):
        raise ValueError("Validator omitted boolean answerable")
    return verdict


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force to replace it")
    source_questions = read_jsonl(args.source_questions)
    source_ids = {record["gold_documents"][0] for record in source_questions}
    pilot_questions = read_jsonl(args.exclude_questions)
    excluded_scope = {
        document_id for record in pilot_questions for document_id in record.get("source_scope_documents", [])
    }
    page_lookup = {}
    documents = {document_id: [] for document_id in source_ids - excluded_scope}
    titles = {}
    with args.pages.open(encoding="utf-8") as stream:
        for line in stream:
            page = json.loads(line)
            document_id = page["document_id"]
            if document_id not in documents:
                continue
            documents[document_id].append(f"[PAGE {page['page']}]\n{page['text']}")
            titles[document_id] = page["title"]
            page_lookup[(document_id, page["page"])] = page
    eligible = []
    seen_docs = set()
    for record in source_questions:
        document_id, page_number = record["gold_documents"][0], record["gold_pages"][0]
        document_text = "\n\n".join(documents.get(document_id, []))
        if document_id in seen_docs or not (5_000 <= len(document_text) <= 600_000):
            continue
        eligible.append((page_lookup[(document_id, page_number)], document_text))
        seen_docs.add(document_id)
    random.Random(args.seed).shuffle(eligible)
    client = create_client(args.credentials)
    accepted = []
    for attempted, (page, document_text) in enumerate(eligible, 1):
        if len(accepted) >= args.count:
            break
        try:
            candidate = propose(args, page)
            verdict = validate_absence(client, args.validator_model, candidate["question"], document_text)
            if verdict["answerable"]:
                print(f"rejected {page['document_id']}: validator found an answer", flush=True)
                continue
            record = {
                "category": "unanswerable",
                "question": candidate["question"].strip(),
                "reference_answer": "Insufficient evidence.",
                "answerable": False,
                "gold_documents": [],
                "gold_pages": [],
                "gold_passages": [],
                "gold_contexts": [],
                "source_scope_documents": [page["document_id"]],
                "unanswerable_rationale": candidate["missing_fact_rationale"].strip(),
                "generator_model": args.generator_model,
                "validator_model": args.validator_model,
                "review_status": "machine_validated_unanswerable",
            }
            accepted.append(record)
            print(f"{len(accepted)}/{args.count} {page['document_id']}", flush=True)
        except Exception as exc:
            print(f"rejected attempt {attempted}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(0.2)
    if len(accepted) < args.count:
        raise RuntimeError(f"Only generated {len(accepted)}/{args.count} unanswerable questions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(accepted, 1):
            record["question_id"] = f"unanswerable-{index:03d}"
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"accepted": len(accepted), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()

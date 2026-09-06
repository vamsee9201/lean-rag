#!/usr/bin/env python3
"""Rebuild the accepted RAFT SFT records with frozen hybrid retrieval contexts."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3

from transformers import AutoTokenizer

from build_raft_sft import clip_context, oracle_chunk, sft_record, target_answer, unique_chunks


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path)
    parser.add_argument("--accepted-metadata", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def contains_supervision(context: list[dict], question: dict) -> bool:
    combined = " ".join(" ".join(item["text"].split()).casefold() for item in context)
    targets = [question["reference_answer"], *question["gold_passages"]]
    return any(
        len(normalized := " ".join(target.split()).casefold()) >= 4 and normalized in combined
        for target in targets
    )


def safe_unanswerable_context(candidates: list[dict], question: dict, top_k: int) -> list[dict]:
    """Choose the first fused candidates that cannot expose the held-out answer."""
    gold_documents = set(question["gold_documents"])
    selected = []
    seen = set()
    for item in candidates:
        if item["chunk_id"] in seen or item["document_id"] in gold_documents:
            continue
        if contains_supervision([item], question):
            continue
        seen.add(item["chunk_id"])
        selected.append(item)
        if len(selected) == top_k:
            return selected
    raise ValueError(f"Not enough safe non-gold hybrid distractors for {question['question_id']}")


def answerable_context(db, retrieved: list[dict], question: dict, top_k: int) -> tuple[list[dict], bool]:
    context = unique_chunks(retrieved[:top_k])
    oracles = [
        oracle_chunk(db, document_id, page, quote)
        for document_id, page, quote in zip(
            question["gold_documents"], question["gold_pages"], question["gold_passages"]
        )
    ]
    if len(oracles) > top_k:
        raise ValueError("More oracle chunks than context slots")
    present = {item["chunk_id"] for item in context}
    missing = [item for item in oracles if item["chunk_id"] not in present]
    protected = {item["chunk_id"] for item in oracles if item["chunk_id"] in present}
    for oracle in missing:
        replace_at = next(
            (index for index in range(len(context) - 1, -1, -1) if context[index]["chunk_id"] not in protected),
            None,
        )
        if replace_at is None:
            context.append(oracle)
        else:
            context[replace_at] = oracle
        protected.add(oracle["chunk_id"])
    return unique_chunks(context)[:top_k], bool(missing)


def main() -> None:
    args = parse_args()
    source = ROOT / "data" / "finetuning" / args.split
    questions_path = args.questions or source / "question_sources.jsonl"
    bm25_path = args.bm25_index or source / "bm25.sqlite3"
    accepted_path = args.accepted_metadata or source / "sft_metadata.jsonl"
    destination = ROOT / "data" / "hybrid_experiment" / "sft" / args.split
    output = args.output or destination / "sft.jsonl"
    metadata_path = args.metadata or destination / "sft_metadata.jsonl"
    for path in (output, metadata_path):
        if path.exists() and not args.force:
            raise SystemExit(f"Output exists: {path}; pass --force to replace it")

    questions = {row["question_id"]: row for row in read_jsonl(questions_path)}
    retrieval = {row["question_id"]: row for row in read_jsonl(args.retrieval)}
    accepted = read_jsonl(accepted_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    db = sqlite3.connect(bm25_path)
    db.row_factory = sqlite3.Row
    records, metadata = [], []
    try:
        for accepted_row in accepted:
            question = questions[accepted_row["question_id"]]
            candidates = retrieval[accepted_row["question_id"]]["results"]
            if accepted_row["answerable"]:
                context, forced = answerable_context(db, candidates, question, args.top_k)
                answer = target_answer(question)
            else:
                context = safe_unanswerable_context(candidates, question, args.top_k)
                forced = False
                answer = "Insufficient evidence."
            fitted_question = question if accepted_row["answerable"] else None
            # fit_context needs the question text even for unanswerable examples but must not
            # protect gold quotations in those deliberately non-gold contexts.
            for chars_per_chunk in range(1050, 349, -50):
                clipped = clip_context(context, fitted_question, chars_per_chunk=chars_per_chunk)
                record = sft_record(question["question"], clipped, answer)
                encoded = tokenizer.apply_chat_template(
                    record["messages"], tokenize=True, add_generation_prompt=False
                )
                input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
                if len(input_ids) <= args.max_tokens:
                    context, token_count = clipped, len(input_ids)
                    break
            else:
                raise ValueError(f"Prompt cannot fit {args.max_tokens} tokens for {question['question_id']}")
            records.append(record)
            metadata.append({
                **accepted_row,
                "oracle_forced": forced,
                "context_chunks": [item["chunk_id"] for item in context],
                "retrieval_setup": retrieval[question["question_id"]]["setup"],
                "chars_per_chunk": chars_per_chunk,
                "prompt_tokens": token_count,
            })
    finally:
        db.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((output, records), (metadata_path, metadata)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    summary = {
        "split": args.split,
        "records": len(records),
        "answerable": sum(row["answerable"] for row in metadata),
        "unanswerable": sum(not row["answerable"] for row in metadata),
        "oracle_forced": sum(row["oracle_forced"] for row in metadata),
        "categories": dict(Counter(row["category"] for row in metadata)),
        "source_accepted_metadata": str(accepted_path),
        "retrieval": str(args.retrieval),
    }
    output.with_name("sft_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

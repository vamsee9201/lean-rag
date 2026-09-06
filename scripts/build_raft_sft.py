#!/usr/bin/env python3
"""Turn verified question sources into RAFT-style conversational SFT records."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from build_bm25 import fts_query
from run_experiment import evidence_text


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = (
    "Answer using only the supplied evidence. If it does not contain the answer, say exactly "
    "'Insufficient evidence.' Cite supporting claims as [DOCUMENT_ID p.PAGE]. Keep the answer concise."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--unanswerable-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def search(db: sqlite3.Connection, question: str, limit: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT c.*, bm25(chunks_fts, 0.0, 0.0, 1.0) AS score
        FROM chunks_fts JOIN chunks c USING(chunk_id)
        WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?
        """,
        (fts_query(question), limit),
    ).fetchall()
    return [dict(row) for row in rows]


def oracle_chunk(db: sqlite3.Connection, document_id: str, page: int, quote: str) -> dict:
    rows = db.execute(
        "SELECT * FROM chunks WHERE document_id = ? AND page_start <= ? AND page_end >= ?",
        (document_id, page, page),
    ).fetchall()
    if not rows:
        raise ValueError(f"No indexed chunk for {document_id} p.{page}")
    normalized_quote = " ".join(quote.split()).casefold()
    for row in rows:
        item = dict(row)
        if normalized_quote in " ".join(item["text"].split()).casefold():
            return item
    raise ValueError(f"No indexed chunk contains the gold quote for {document_id} p.{page}")


def unique_chunks(items: list[dict]) -> list[dict]:
    seen = set()
    result = []
    for item in items:
        if item["chunk_id"] in seen:
            continue
        seen.add(item["chunk_id"])
        result.append(item)
    return result


def clip_text(text: str, limit: int, needle: str | None = None) -> str:
    if len(text) <= limit:
        return text
    if needle:
        normalized = " ".join(needle.split())
        # Never truncate the verified supervision span, even when an unusually
        # long quotation is larger than the normal per-chunk context budget.
        limit = max(limit, len(normalized))
        start = text.casefold().find(normalized.casefold())
        if start >= 0:
            left = max(0, start - (limit - len(normalized)) // 2)
            return text[left : left + limit]
    return text[:limit]


def clip_context(context: list[dict], question: dict | None, chars_per_chunk: int = 1050) -> list[dict]:
    quote_by_source = {}
    if question:
        quote_by_source = {
            (document_id, page): quote
            for document_id, page, quote in zip(
                question["gold_documents"], question["gold_pages"], question["gold_passages"]
            )
        }
    clipped = []
    for item in context:
        copy = dict(item)
        needle = quote_by_source.get((item["document_id"], item["page_start"]))
        copy["text"] = clip_text(item["text"], chars_per_chunk, needle)
        clipped.append(copy)
    return clipped


def answerable_context(db: sqlite3.Connection, question: dict, top_k: int) -> tuple[list[dict], bool]:
    retrieved = search(db, question["question"], max(top_k * 4, 20))
    oracles = [
        oracle_chunk(db, document_id, page, quote)
        for document_id, page, quote in zip(
            question["gold_documents"], question["gold_pages"], question["gold_passages"]
        )
    ]
    oracle_ids = {item["chunk_id"] for item in oracles}
    context = unique_chunks(retrieved[:top_k])
    missing = [item for item in oracles if item["chunk_id"] not in {row["chunk_id"] for row in context}]
    forced = bool(missing)
    if len(oracles) > top_k:
        raise ValueError("More oracle chunks than context slots")
    protected = oracle_ids & {item["chunk_id"] for item in context}
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
    return unique_chunks(context)[:top_k], forced


def unanswerable_context(db: sqlite3.Connection, question: dict, top_k: int) -> list[dict]:
    gold_documents = set(question["gold_documents"])
    retrieved = search(db, question["question"], max(top_k * 10, 50))
    context = [item for item in retrieved if item["document_id"] not in gold_documents]
    context = unique_chunks(context)[:top_k]
    if len(context) < top_k:
        raise ValueError("Not enough non-gold distractors for an unanswerable record")
    combined = " ".join(item["text"] for item in context).casefold()
    answer = " ".join(question["reference_answer"].split()).casefold()
    if len(answer) >= 4 and answer in combined:
        raise ValueError("Reference answer appears in the unanswerable distractors")
    return context


def target_answer(question: dict) -> str:
    citations = []
    for document_id, page in zip(question["gold_documents"], question["gold_pages"]):
        citation = f"[{document_id} p.{page}]"
        if citation not in citations:
            citations.append(citation)
    return f"{question['reference_answer'].strip()} {' '.join(citations)}".strip()


def sft_record(question: str, context: list[dict], answer: str) -> dict:
    prompt = f"EVIDENCE\n{evidence_text(context)}\n\nQUESTION\n{question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer, "loss": True},
        ]
    }


def main() -> None:
    args = parse_args()
    base = ROOT / "data" / "finetuning" / args.split
    questions_path = args.questions or base / "question_sources.jsonl"
    index_path = args.index or base / "bm25.sqlite3"
    output = args.output or base / "sft.jsonl"
    metadata_path = args.metadata or base / "sft_metadata.jsonl"
    for path in (output, metadata_path):
        if path.exists() and not args.force:
            raise SystemExit(f"Output exists: {path}; pass --force to replace it")
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line.strip()]
    db = sqlite3.connect(index_path)
    db.row_factory = sqlite3.Row
    records = []
    metadata = []
    rejected = []
    try:
        for question in questions:
            try:
                context, forced = answerable_context(db, question, args.top_k)
                context = clip_context(context, question)
                records.append(sft_record(question["question"], context, target_answer(question)))
                metadata.append(
                    {
                        "record_id": f"{question['question_id']}-answerable",
                        "question_id": question["question_id"],
                        "answerable": True,
                        "category": question["category"],
                        "oracle_forced": forced,
                        "context_chunks": [item["chunk_id"] for item in context],
                    }
                )
                if stable_fraction(args.seed, question["question_id"]) < args.unanswerable_rate:
                    distractors = unanswerable_context(db, question, args.top_k)
                    distractors = clip_context(distractors, None)
                    records.append(sft_record(question["question"], distractors, "Insufficient evidence."))
                    metadata.append(
                        {
                            "record_id": f"{question['question_id']}-unanswerable",
                            "question_id": question["question_id"],
                            "answerable": False,
                            "category": "unanswerable",
                            "oracle_forced": False,
                            "context_chunks": [item["chunk_id"] for item in distractors],
                        }
                    )
            except Exception as exc:
                rejected.append(
                    {"question_id": question["question_id"], "error": f"{type(exc).__name__}: {exc}"}
                )
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
        "source_questions": len(questions),
        "sft_records": len(records),
        "answerable": sum(row["answerable"] for row in metadata),
        "unanswerable": sum(not row["answerable"] for row in metadata),
        "oracle_forced": sum(row["oracle_forced"] for row in metadata),
        "categories": dict(Counter(row["category"] for row in metadata)),
        "rejected": rejected,
        "output": str(output),
        "metadata": str(metadata_path),
    }
    output.with_name("sft_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

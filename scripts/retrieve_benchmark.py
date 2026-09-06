#!/usr/bin/env python3
"""Freeze BM25 top-k evidence for every benchmark question."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sqlite3

from build_bm25 import fts_query


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "benchmark" / "questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data" / "indexes" / "corpus500-bm25.sqlite3")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "retrieval" / "results.jsonl")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def search(db: sqlite3.Connection, query: str, top_k: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT c.*, bm25(chunks_fts, 0.0, 0.0, 1.0) AS score
        FROM chunks_fts JOIN chunks c USING(chunk_id)
        WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?
        """,
        (fts_query(query), top_k),
    ).fetchall()
    return [
        {
            "chunk_id": row["chunk_id"], "document_id": row["document_id"],
            "title": row["title"], "collection": row["collection"],
            "page_start": row["page_start"], "page_end": row["page_end"],
            "text": row["text"], "score": float(row["score"]), "rank": rank,
            "source": "bm25",
        }
        for rank, row in enumerate(rows, 1)
    ]


def metrics(items: list[dict], question: dict) -> dict | None:
    if not question.get("answerable", True):
        return None
    gold = set(zip(question["gold_documents"], question["gold_pages"]))
    credited = set()
    relevance = []
    for item in items:
        matches = {
            (document_id, page) for document_id, page in gold
            if item["document_id"] == document_id and item["page_start"] <= page <= item["page_end"]
        }
        new_matches = matches - credited
        relevance.append(int(bool(new_matches)))
        credited.update(new_matches)
    first = next((rank for rank, value in enumerate(relevance, 1) if value), None)
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), len(items)) + 1))
    return {
        "gold_passage_recall": len(credited) / len(gold), "all_gold_found": credited == gold,
        "reciprocal_rank": 0.0 if first is None else 1.0 / first,
        "ndcg": 0.0 if ideal == 0 else dcg / ideal,
    }


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force to replace it")
    questions = read_jsonl(args.questions)
    db = sqlite3.connect(args.chunks)
    db.row_factory = sqlite3.Row
    records = []
    try:
        for index, question in enumerate(questions, 1):
            results = search(db, question["question"], args.top_k)
            records.append({
                "question_id": question["question_id"], "setup": f"bm25_top{args.top_k}",
                "results": results, "metrics": metrics(results, question),
            })
            print(f"{index}/{len(questions)} {question['question_id']}", flush=True)
    finally:
        db.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    answerable = [record["metrics"] for record in records if record["metrics"] is not None]
    summary = {
        "setup": f"bm25_top{args.top_k}", "questions": len(records),
        "answerable_questions": len(answerable),
        "all_gold_recall": sum(item["all_gold_found"] for item in answerable) / len(answerable),
        "mean_gold_passage_recall": sum(item["gold_passage_recall"] for item in answerable) / len(answerable),
        "mrr": sum(item["reciprocal_rank"] for item in answerable) / len(answerable),
        "mean_ndcg": sum(item["ndcg"] for item in answerable) / len(answerable),
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

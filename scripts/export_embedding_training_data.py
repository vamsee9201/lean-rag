#!/usr/bin/env python3
"""Export compact chunks, questions, and ranked candidates for retriever training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from hybrid_retrieval import bm25_search


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "local_retriever_experiment" / "embedding_training"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def export_chunks(index: Path, output: Path) -> int:
    db = sqlite3.connect(index)
    db.row_factory = sqlite3.Row
    try:
        return write_jsonl(output, (dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY chunk_id")))
    finally:
        db.close()


def compact_candidates(path: Path, output: Path, questions: list[dict], bm25_index: Path) -> int:
    question_by_id = {row["question_id"]: row for row in questions}
    db = sqlite3.connect(bm25_index)
    db.row_factory = sqlite3.Row

    def rows():
        try:
            for record in read_jsonl(path):
                qid = record["question_id"]
                merged = {
                    item["chunk_id"]: {
                        "chunk_id": item["chunk_id"],
                        "bm25_rank": item.get("bm25_rank"),
                        "dense_rank": item.get("dense_rank"),
                    }
                    for item in record["results"]
                }
                # The planned top-50 pool remains primary. Ranks 51-100 are a
                # deterministic safety fallback for candidates that repeat the
                # gold answer and therefore cannot be labeled as negatives.
                for rank, item in enumerate(bm25_search(db, question_by_id[qid]["question"], 100), 1):
                    candidate = merged.setdefault(item["chunk_id"], {
                        "chunk_id": item["chunk_id"], "bm25_rank": None, "dense_rank": None,
                    })
                    candidate["bm25_rank"] = rank
                yield {"question_id": qid, "candidates": sorted(
                    merged.values(), key=lambda item: (
                        min(item["bm25_rank"] or 10**9, item["dense_rank"] or 10**9), item["chunk_id"]
                    )
                )}
        finally:
            db.close()

    return write_jsonl(output, rows())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    outputs = [
        args.output_dir / "train_chunks.jsonl",
        args.output_dir / "validation_chunks.jsonl",
        args.output_dir / "train_questions.jsonl",
        args.output_dir / "validation_questions.jsonl",
        args.output_dir / "train_candidates.jsonl",
        args.output_dir / "validation_candidates.jsonl",
    ]
    if any(path.exists() for path in outputs) and not args.force:
        raise SystemExit("Embedding-training exports already exist; pass --force to replace")
    train_questions = read_jsonl(ROOT / "data/finetuning/train/question_sources.jsonl")
    validation_questions = read_jsonl(ROOT / "data/finetuning/validation/question_sources.jsonl")
    counts = {
        "train_chunks": export_chunks(ROOT / "data/finetuning/train/bm25.sqlite3", outputs[0]),
        "validation_chunks": export_chunks(ROOT / "data/finetuning/validation/bm25.sqlite3", outputs[1]),
        "train_questions": write_jsonl(outputs[2], train_questions),
        "validation_questions": write_jsonl(outputs[3], validation_questions),
        "train_candidates": compact_candidates(
            ROOT / "data/hybrid_experiment/retrieval/train_candidates.jsonl", outputs[4],
            train_questions, ROOT / "data/finetuning/train/bm25.sqlite3",
        ),
        "validation_candidates": compact_candidates(
            ROOT / "data/hybrid_experiment/retrieval/validation_candidates.jsonl", outputs[5],
            validation_questions, ROOT / "data/finetuning/validation/bm25.sqlite3",
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(counts, indent=2) + "\n")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()

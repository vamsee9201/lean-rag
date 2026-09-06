#!/usr/bin/env python3
"""Embed questions, freeze dense/BM25/hybrid evidence, and tune fusion on validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from hybrid_retrieval import DenseIndex, bm25_search, chunk_map, reciprocal_rank_fusion
from vertex_embeddings import DIMENSIONS, MODEL, create_embedding_client, embed_one


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--dense-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--document-cap", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--setup", default="vertex_hybrid_top5")
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--tune-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


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
        new = matches - credited
        relevance.append(int(bool(new)))
        credited.update(new)
    first = next((rank for rank, value in enumerate(relevance, 1) if value), None)
    dcg = sum(value / math.log2(rank + 1) for rank, value in enumerate(relevance, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold), len(items)) + 1))
    return {
        "gold_passage_recall": len(credited) / len(gold),
        "all_gold_found": credited == gold,
        "reciprocal_rank": 0.0 if first is None else 1.0 / first,
        "ndcg": 0.0 if not ideal else dcg / ideal,
    }


def aggregate(records: list[dict]) -> dict:
    usable = [record["metrics"] for record in records if record["metrics"] is not None]
    cross = [
        record["metrics"] for record in records
        if record["metrics"] is not None and record.get("category") == "cross_document"
    ]
    mean = lambda key, rows: sum(float(row[key]) for row in rows) / len(rows) if rows else None
    return {
        "questions": len(records),
        "answerable_questions": len(usable),
        "all_gold_recall": mean("all_gold_found", usable),
        "mean_gold_passage_recall": mean("gold_passage_recall", usable),
        "mrr": mean("reciprocal_rank", usable),
        "mean_ndcg": mean("ndcg", usable),
        "cross_document_all_gold_recall": mean("all_gold_found", cross),
    }


def cache_queries(args, questions: list[dict]) -> dict[str, list[float]]:
    path = args.query_cache or args.output.parent / "query_embeddings.jsonl"
    cached = {}
    if path.exists():
        cached = {row["question_id"]: row for row in read_jsonl(path)}
    missing = [question for question in questions if question["question_id"] not in cached]
    if missing:
        client = create_embedding_client(args.credentials, args.location)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            for index, question in enumerate(missing, 1):
                vector, stats = embed_one(
                    client, question["question"], task_type="RETRIEVAL_QUERY",
                    model=args.model, dimensions=args.dimensions,
                )
                row = {"question_id": question["question_id"], "embedding": vector.tolist(), **stats}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                cached[row["question_id"]] = row
                print(f"embedded query {index}/{len(missing)}", flush=True)
    return {question_id: row["embedding"] for question_id, row in cached.items()}


def select_configuration(candidates: list[dict]) -> dict:
    return max(
        candidates,
        key=lambda item: (
            item["metrics"]["all_gold_recall"],
            item["metrics"]["mean_gold_passage_recall"],
            item["metrics"]["mean_ndcg"],
            item["metrics"]["mrr"],
            item["dense_weight"] == 1.0,
            item["candidate_depth"] == 20,
            item["document_cap"] is None,
        ),
    )


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force to replace it")
    questions = read_jsonl(args.questions)
    dense_index = DenseIndex(args.dense_index)
    if dense_index.manifest["model"] != args.model or dense_index.manifest["dimensions"] != args.dimensions:
        raise ValueError("Query embedding configuration does not match the dense index")
    query_vectors = cache_queries(args, questions)
    retrieval_started = time.perf_counter()
    query_list = [query_vectors[question["question_id"]] for question in questions]
    dense_results = dense_index.search_many(query_list, 50)
    dense_seconds = time.perf_counter() - retrieval_started
    print(f"searched dense index for {len(questions)} queries", flush=True)
    local = threading.local()

    def search_bm25(question: dict) -> list[dict]:
        if not hasattr(local, "db"):
            local.db = sqlite3.connect(args.bm25_index)
            local.db.row_factory = sqlite3.Row
        return bm25_search(local.db, question["question"], 50)

    bm25_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        bm25_results = list(pool.map(search_bm25, questions))
    bm25_seconds = time.perf_counter() - bm25_started
    print(f"searched BM25 index for {len(questions)} queries", flush=True)
    db = sqlite3.connect(args.bm25_index)
    db.row_factory = sqlite3.Row
    base = []
    fusion_started = time.perf_counter()
    try:
        for question, bm25, dense in zip(questions, bm25_results, dense_results, strict=True):
            metadata = chunk_map(db, [chunk_id for chunk_id, _score in dense])
            base.append((question, bm25, dense, metadata))
    finally:
        db.close()

    configs = [(args.candidate_depth, args.dense_weight, args.document_cap)]
    if args.tune_validation:
        configs = list(itertools.product((20, 50), (0.5, 1.0, 2.0), (None, 2)))
    evaluated = []
    records_by_config = {}
    for depth, weight, cap in configs:
        records = []
        for question, bm25, dense, metadata in base:
            results = reciprocal_rank_fusion(
                bm25[:depth], dense[:depth], top_k=args.top_k, rrf_k=args.rrf_k,
                dense_weight=weight, document_cap=cap, chunks=metadata,
            )
            records.append({
                "question_id": question["question_id"],
                "category": question.get("category"),
                "setup": args.setup,
                "results": results,
                "metrics": metrics(results, question),
            })
        summary = aggregate(records)
        config = {"candidate_depth": depth, "dense_weight": weight, "document_cap": cap, "metrics": summary}
        evaluated.append(config)
        records_by_config[(depth, weight, cap)] = records
    selected = select_configuration(evaluated)
    key = (selected["candidate_depth"], selected["dense_weight"], selected["document_cap"])
    records = records_by_config[key]
    fusion_seconds = time.perf_counter() - fusion_started
    for record in records:
        record["retrieval_config"] = {
            "model": args.model, "dimensions": args.dimensions, "rrf_k": args.rrf_k,
            **{name: selected[name] for name in ("candidate_depth", "dense_weight", "document_cap")},
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {
        "setup": args.setup,
        "selected": selected,
        "validation_grid": evaluated if args.tune_validation else None,
        "retrieval_seconds": dense_seconds + bm25_seconds + fusion_seconds,
        "mean_retrieval_seconds": (dense_seconds + bm25_seconds + fusion_seconds) / len(records),
        "latency_breakdown_seconds": {
            "dense_search": dense_seconds, "bm25_search": bm25_seconds, "fusion_and_metadata": fusion_seconds,
        },
        "dense_manifest": dense_index.manifest,
    }
    dense_records = []
    bm25_records = []
    for question, bm25, dense, metadata in base:
        dense_items = [
            {**metadata[chunk_id], "rank": rank, "dense_score": score}
            for rank, (chunk_id, score) in enumerate(dense[: args.top_k], 1)
        ]
        bm25_items = [{**item, "rank": rank} for rank, item in enumerate(bm25[: args.top_k], 1)]
        dense_records.append({"metrics": metrics(dense_items, question), "category": question.get("category")})
        bm25_records.append({"metrics": metrics(bm25_items, question), "category": question.get("category")})
    summary["retriever_metrics"] = {
        "bm25": aggregate(bm25_records), "vertex_dense": aggregate(dense_records),
        "vertex_hybrid": selected["metrics"],
    }
    args.output.with_name(f"{args.output.stem}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

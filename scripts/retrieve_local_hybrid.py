#!/usr/bin/env python3
"""Retrieve with local dense embeddings and optionally tune BM25 fusion."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_retrieval import DenseIndex, bm25_search, chunk_map, reciprocal_rank_fusion
from retrieve_hybrid import aggregate, metrics, select_configuration


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--bm25-index", type=Path, required=True)
    parser.add_argument("--dense-index", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-cache", type=Path)
    parser.add_argument("--dense-output", type=Path)
    parser.add_argument("--dense-setup", default="local_dense_top5")
    parser.add_argument("--setup", default="local_hybrid_top5")
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--document-cap", type=int)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--truncate-dim", type=int)
    parser.add_argument(
        "--query-instruction",
        help="Instruction placed in the Qwen embedding query template.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--tune-validation", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"Output exists: {args.output}; pass --force")
    questions = read_jsonl(args.questions)
    index = DenseIndex(args.dense_index)
    label = args.model_label or args.model
    if index.manifest["model"] != label:
        raise ValueError(f"Index model {index.manifest['model']} does not match {label}")
    cache = args.query_cache or args.output.with_name(args.output.stem + "_query_embeddings.jsonl")
    cached = {row["question_id"]: row for row in read_jsonl(cache)} if cache.exists() else {}
    missing = [q for q in questions if q["question_id"] not in cached]
    if missing:
        model_kwargs = {"trust_remote_code": True, "device": args.device}
        if args.truncate_dim is not None:
            model_kwargs["truncate_dim"] = args.truncate_dim
        model = SentenceTransformer(args.model, **model_kwargs)
        model.max_seq_length = args.max_seq_length
        texts = [
            f"Instruct: {args.query_instruction}\nQuery:{q['question']}"
            if args.query_instruction else q["question"]
            for q in missing
        ]
        lengths = [len(model.tokenizer.encode(text, add_special_tokens=True)) for text in texts]
        if max(lengths, default=0) > args.max_seq_length:
            raise ValueError("A query would be truncated")
        vectors = model.encode(texts, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("a", encoding="utf-8") as stream:
            for question, vector, tokens in zip(missing, vectors, lengths, strict=True):
                row = {"question_id": question["question_id"], "embedding": vector.tolist(),
                       "token_count": tokens, "truncated": False,
                       "query_instruction": args.query_instruction}
                stream.write(json.dumps(row) + "\n"); cached[row["question_id"]] = row
    query_vectors = [np.asarray(cached[q["question_id"]]["embedding"], dtype=np.float32) for q in questions]
    for question in questions:
        row = cached[question["question_id"]]
        if row.get("query_instruction") != args.query_instruction:
            raise ValueError("Cached query instruction does not match this run")
        if row.get("model") is not None and row["model"] != label:
            raise ValueError("Cached query model does not match the dense index")
    started = time.perf_counter(); dense_results = index.search_many(query_vectors, 100); dense_seconds = time.perf_counter() - started
    db = sqlite3.connect(args.bm25_index); db.row_factory = sqlite3.Row
    try:
        started = time.perf_counter()
        bm25_results = [bm25_search(db, q["question"], 100) for q in questions]
        bm25_seconds = time.perf_counter() - started
        bases = []
        started = time.perf_counter()
        for question, bm25, dense in zip(questions, bm25_results, dense_results, strict=True):
            bases.append((question, bm25, dense, chunk_map(db, [cid for cid, _ in dense])))
        metadata_seconds = time.perf_counter() - started
    finally:
        db.close()
    configs = [(args.candidate_depth, args.dense_weight, args.document_cap, args.rrf_k)]
    if args.tune_validation:
        configs = list(itertools.product((20, 50, 100), (0.5, 1.0, 1.5, 2.0), (None, 2, 3), (30, 60)))
    evaluated, records_by_config = [], {}
    fusion_started = time.perf_counter()
    for depth, weight, cap, rrf_k in configs:
        records = []
        for question, bm25, dense, metadata in bases:
            results = reciprocal_rank_fusion(
                bm25[:depth], dense[:depth], top_k=args.top_k, rrf_k=rrf_k,
                dense_weight=weight, document_cap=cap, chunks=metadata,
            )
            records.append({"question_id": question["question_id"], "category": question.get("category"),
                            "setup": args.setup, "results": results, "metrics": metrics(results, question)})
        config = {"candidate_depth": depth, "dense_weight": weight, "document_cap": cap,
                  "rrf_k": rrf_k, "metrics": aggregate(records)}
        evaluated.append(config); records_by_config[(depth, weight, cap, rrf_k)] = records
    selected = max(evaluated, key=lambda x: (
        x["metrics"]["all_gold_recall"], x["metrics"]["mean_gold_passage_recall"],
        x["metrics"]["mean_ndcg"], x["metrics"]["mrr"], x["dense_weight"] == 1.0,
        x["rrf_k"] == 60, x["candidate_depth"] == 20, x["document_cap"] is None,
    ))
    key = (selected["candidate_depth"], selected["dense_weight"], selected["document_cap"], selected["rrf_k"])
    records = records_by_config[key]
    fusion_seconds = time.perf_counter() - fusion_started
    query_hash = hashlib.sha256(cache.read_bytes()).hexdigest()
    for record in records:
        record["retrieval_config"] = {"embedding_model": label, "dimensions": index.manifest["dimensions"],
                                      "query_instruction": args.query_instruction,
                                      "query_checksum": query_hash, "index_checksum": index.manifest["embeddings_sha256"],
                                      **{k: selected[k] for k in ("candidate_depth", "dense_weight", "document_cap", "rrf_k")}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records: stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    dense_rows, bm25_rows, dense_records = [], [], []
    for question, bm25, dense, metadata in bases:
        dense_items = [{**metadata[cid], "rank": rank, "dense_score": score} for rank, (cid, score) in enumerate(dense[:5], 1)]
        bm25_items = [{**item, "rank": rank} for rank, item in enumerate(bm25[:5], 1)]
        dense_rows.append({"metrics": metrics(dense_items, question), "category": question.get("category")})
        dense_records.append({
            "question_id": question["question_id"], "category": question.get("category"),
            "setup": args.dense_setup, "results": dense_items,
            "metrics": metrics(dense_items, question),
            "retrieval_config": {
                "embedding_model": label, "dimensions": index.manifest["dimensions"],
                "query_instruction": args.query_instruction, "query_checksum": query_hash,
                "index_checksum": index.manifest["embeddings_sha256"], "top_k": args.top_k,
            },
        })
        bm25_rows.append({"metrics": metrics(bm25_items, question), "category": question.get("category")})
    summary = {"setup": args.setup, "selected": selected,
               "validation_grid": evaluated if args.tune_validation else None,
               "retriever_metrics": {"bm25": aggregate(bm25_rows), "local_dense": aggregate(dense_rows),
                                     "local_hybrid": selected["metrics"]},
               "latency_breakdown_seconds": {
                   "dense_search": dense_seconds, "bm25_search": bm25_seconds,
                   "metadata_lookup": metadata_seconds,
                   "fusion_and_metrics": fusion_seconds,
               },
               "retrieval_seconds": dense_seconds + bm25_seconds + metadata_seconds + fusion_seconds,
               "mean_retrieval_seconds": (
                   (dense_seconds + bm25_seconds + metadata_seconds + fusion_seconds) / len(questions)
                   if not args.tune_validation else None
               ),
               "validation_grid_seconds": fusion_seconds if args.tune_validation else None,
               "dense_manifest": index.manifest}
    if args.dense_output:
        args.dense_output.parent.mkdir(parents=True, exist_ok=True)
        with args.dense_output.open("w", encoding="utf-8") as stream:
            for row in dense_records:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        args.dense_output.with_name(args.dense_output.stem + "_summary.json").write_text(
            json.dumps({"setup": args.dense_setup, "retriever_metrics": aggregate(dense_rows),
                        "dense_manifest": index.manifest}, indent=2) + "\n"
        )
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

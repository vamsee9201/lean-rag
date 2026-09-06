#!/usr/bin/env python3
"""Deterministic dense search and reciprocal-rank fusion for the RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterable

import numpy as np

from build_bm25 import fts_query


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise ValueError("Embedding must be a finite one-dimensional vector")
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("Embedding has zero norm")
    return vector / norm


def read_chunk_ids(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("chunk_ids.json must contain a JSON string array")
    if len(value) != len(set(value)):
        raise ValueError("Dense index contains duplicate chunk IDs")
    return value


@dataclass
class DenseIndex:
    directory: Path

    def __post_init__(self) -> None:
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        self.chunk_ids = read_chunk_ids(self.directory / "chunk_ids.json")
        self.embeddings = np.load(self.directory / "embeddings.npy", mmap_mode="r")
        expected = (len(self.chunk_ids), int(self.manifest["dimensions"]))
        if self.embeddings.shape != expected:
            raise ValueError(f"Dense index shape {self.embeddings.shape} != {expected}")
        if self.manifest.get("complete") is not True:
            raise ValueError("Dense index is not marked complete")

    def search(self, query_vector: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        query = normalize(query_vector)
        scores = np.asarray(self.embeddings @ query, dtype=np.float32)
        top_k = min(top_k, len(scores))
        candidates = np.argpartition(scores, -top_k)[-top_k:]
        ordered = sorted(candidates, key=lambda index: (-float(scores[index]), self.chunk_ids[index]))
        return [(self.chunk_ids[index], float(scores[index])) for index in ordered]

    def search_many(
        self, query_vectors: Iterable[np.ndarray], top_k: int, batch_size: int = 64
    ) -> list[list[tuple[str, float]]]:
        """Search many queries with bounded-memory BLAS batches."""
        queries = np.stack([normalize(vector) for vector in query_vectors])
        results: list[list[tuple[str, float]]] = []
        top_k = min(top_k, len(self.chunk_ids))
        for start in range(0, len(queries), batch_size):
            scores = np.asarray(self.embeddings @ queries[start : start + batch_size].T, dtype=np.float32)
            for column in range(scores.shape[1]):
                values = scores[:, column]
                candidates = np.argpartition(values, -top_k)[-top_k:]
                ordered = sorted(candidates, key=lambda index: (-float(values[index]), self.chunk_ids[index]))
                results.append([(self.chunk_ids[index], float(values[index])) for index in ordered])
        return results


def bm25_search(db: sqlite3.Connection, question: str, top_k: int) -> list[dict]:
    rows = db.execute(
        """
        SELECT c.*, bm25(chunks_fts, 0.0, 0.0, 1.0) AS score
        FROM chunks_fts JOIN chunks c USING(chunk_id)
        WHERE chunks_fts MATCH ? ORDER BY score, c.chunk_id LIMIT ?
        """,
        (fts_query(question), top_k),
    ).fetchall()
    return [dict(row) for row in rows]


def chunk_map(db: sqlite3.Connection, chunk_ids: Iterable[str]) -> dict[str, dict]:
    ids = list(chunk_ids)
    if not ids:
        return {}
    result = {}
    for start in range(0, len(ids), 500):
        batch = ids[start : start + 500]
        marks = ",".join("?" for _ in batch)
        rows = db.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({marks})", batch).fetchall()
        result.update((row["chunk_id"], dict(row)) for row in rows)
    if missing := set(ids) - set(result):
        raise ValueError(f"Dense index references missing chunks: {sorted(missing)[:3]}")
    return result


def reciprocal_rank_fusion(
    bm25: list[dict],
    dense: list[tuple[str, float]],
    *,
    top_k: int = 5,
    rrf_k: int = 60,
    bm25_weight: float = 1.0,
    dense_weight: float = 1.0,
    document_cap: int | None = None,
    chunks: dict[str, dict] | None = None,
) -> list[dict]:
    if top_k < 1 or rrf_k < 1 or bm25_weight <= 0 or dense_weight <= 0:
        raise ValueError("Invalid fusion parameters")
    by_id: dict[str, dict] = {}
    for rank, item in enumerate(bm25, 1):
        by_id.setdefault(item["chunk_id"], {})["bm25_rank"] = rank
        by_id[item["chunk_id"]]["bm25_score"] = float(item["score"])
        by_id[item["chunk_id"]]["chunk"] = {k: v for k, v in item.items() if k != "score"}
    for rank, (chunk_id, score) in enumerate(dense, 1):
        by_id.setdefault(chunk_id, {})["dense_rank"] = rank
        by_id[chunk_id]["dense_score"] = float(score)
        if "chunk" not in by_id[chunk_id]:
            if chunks is None or chunk_id not in chunks:
                raise ValueError(f"Missing metadata for dense chunk {chunk_id}")
            by_id[chunk_id]["chunk"] = chunks[chunk_id]
    for item in by_id.values():
        item["rrf_score"] = (
            (bm25_weight / (rrf_k + item["bm25_rank"]) if "bm25_rank" in item else 0.0)
            + (dense_weight / (rrf_k + item["dense_rank"]) if "dense_rank" in item else 0.0)
        )
    ordered = sorted(by_id.items(), key=lambda pair: (-pair[1]["rrf_score"], pair[0]))
    selected = []
    per_document: dict[str, int] = {}
    for chunk_id, item in ordered:
        chunk = dict(item["chunk"])
        document_id = chunk["document_id"]
        if document_cap is not None and per_document.get(document_id, 0) >= document_cap:
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        selected.append({
            **chunk,
            "rank": len(selected) + 1,
            "source": "hybrid_rrf",
            "bm25_rank": item.get("bm25_rank"),
            "bm25_score": item.get("bm25_score"),
            "dense_rank": item.get("dense_rank"),
            "dense_score": item.get("dense_score"),
            "rrf_score": item["rrf_score"],
        })
        if len(selected) == top_k:
            break
    return selected

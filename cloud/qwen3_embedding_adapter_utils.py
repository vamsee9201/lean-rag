"""Adapter-aware Qwen3 embedding inference and exact-index helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from swift import InferRequest, TransformersEngine
from transformers import AutoTokenizer

from qwen3_embedding_utils import DIMENSIONS, ENCODING_MAX_LENGTH, QUERY_INSTRUCTION


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def query_text(text: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"


def load_engine(base_path: str | Path, adapter: str | Path) -> TransformersEngine:
    return TransformersEngine(
        str(base_path), adapters=[str(adapter)], task_type="embedding",
        torch_dtype=torch.bfloat16, attn_impl="flash_attention_2",
        max_batch_size=32, max_length=ENCODING_MAX_LENGTH,
    )


def _batch_size(length: int) -> int:
    if length > 8192:
        return 1
    if length > 4096:
        return 2
    if length > 2048:
        return 4
    if length > 1024:
        return 12
    return 32


def encode(engine: TransformersEngine, tokenizer, texts: list[str]) -> tuple[np.ndarray, dict]:
    lengths = []
    for start in range(0, len(texts), 256):
        lengths.extend(tokenizer(
            texts[start:start + 256], add_special_tokens=True, truncation=False,
            padding=False, return_length=True,
        )["length"])
    if max(lengths, default=0) > ENCODING_MAX_LENGTH:
        raise ValueError(f"Input exceeds {ENCODING_MAX_LENGTH} tokens")
    order = sorted(range(len(texts)), key=lambda index: (-lengths[index], index))
    output = np.empty((len(texts), DIMENSIONS), dtype=np.float32)
    started = time.perf_counter()
    cursor = 0
    batches = 0
    while cursor < len(order):
        size = min(_batch_size(lengths[order[cursor]]), len(order) - cursor)
        indices = order[cursor:cursor + size]
        requests = [
            InferRequest(messages=[{"role": "user", "content": texts[index]}])
            for index in indices
        ]
        responses = engine.infer(requests, use_tqdm=False)
        values = np.asarray([row.data[0].embedding for row in responses], dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(indices) or values.shape[1] < DIMENSIONS:
            raise ValueError(f"Unexpected embedding shape: {values.shape}")
        values = values[:, :DIMENSIONS]
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if not np.isfinite(values).all() or np.any(norms == 0):
            raise ValueError("Non-finite or zero-norm embedding")
        output[indices] = values / norms
        cursor += size
        batches += 1
        if batches % 250 == 0 or cursor == len(order):
            print(f"ENCODE {cursor}/{len(order)} texts in {batches} batches", flush=True)
    norms = np.linalg.norm(output, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4):
        raise ValueError("Post-truncation vectors are not normalized")
    return output, {
        "texts": len(texts), "maximum_tokens": max(lengths, default=0),
        "minimum_tokens": min(lengths, default=0), "batches": batches,
        "seconds": time.perf_counter() - started,
    }


def top_indices(corpus: np.ndarray, queries: np.ndarray, top_k: int = 5) -> list[list[int]]:
    output = []
    for start in range(0, len(queries), 32):
        scores = corpus @ queries[start:start + 32].T
        for column in range(scores.shape[1]):
            values = scores[:, column]
            indices = np.argpartition(values, -top_k)[-top_k:]
            output.append(sorted(indices.tolist(), key=lambda i: (-float(values[i]), i)))
    return output


def retrieval_metrics(corpus, queries, questions, chunks, positives) -> dict:
    ranked = top_indices(corpus, queries)
    chunk_ids = [row["chunk_id"] for row in chunks]
    all_gold = passage_recall = reciprocal = ndcg = 0.0
    for question, indices in zip(questions, ranked, strict=True):
        returned = [chunk_ids[index] for index in indices]
        gold = set(positives[question["question_id"]])
        hits = [rank + 1 for rank, chunk_id in enumerate(returned) if chunk_id in gold]
        all_gold += float(gold.issubset(returned))
        passage_recall += len(gold & set(returned)) / len(gold)
        reciprocal += 1 / min(hits) if hits else 0.0
        dcg = sum(1 / np.log2(rank + 1) for rank in hits)
        ideal = sum(1 / np.log2(rank + 1) for rank in range(1, min(5, len(gold)) + 1))
        ndcg += dcg / ideal
    count = len(questions)
    return {
        "all_gold_recall_at_5": all_gold / count,
        "mean_passage_recall_at_5": passage_recall / count,
        "mrr_at_5": reciprocal / count,
        "ndcg_at_5": ndcg / count,
        "questions": count,
    }


def save_index(vectors, chunks, directory: Path, split: str, model_label: str, stats: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    ids = [row["chunk_id"] for row in chunks]
    np.save(directory / "embeddings.npy", vectors)
    (directory / "chunk_ids.json").write_text(json.dumps(ids) + "\n")
    manifest = {
        "complete": True, "adapter_aware": True, "split": split,
        "model": model_label, "dimensions": DIMENSIONS, "dtype": "float32",
        "normalized": True, "max_sequence_length": ENCODING_MAX_LENGTH,
        "chunks": len(chunks), "encoding": stats,
        "chunk_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "embeddings_sha256": sha256(directory / "embeddings.npy"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def save_query_cache(vectors, questions, path: Path, model_label: str, stats: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for question, vector in zip(questions, vectors, strict=True):
            stream.write(json.dumps({
                "question_id": question["question_id"], "embedding": vector.tolist(),
                "truncated": False, "query_instruction": QUERY_INSTRUCTION,
                "model": model_label,
            }) + "\n")
    manifest = {
        "questions": len(questions), "model": model_label, "dimensions": DIMENSIONS,
        "adapter_aware": True, "query_instruction": QUERY_INSTRUCTION,
        "encoding": stats, "sha256": sha256(path),
    }
    path.with_name(path.stem + "_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def tokenizer_for(base_path: str | Path):
    return AutoTokenizer.from_pretrained(str(base_path))

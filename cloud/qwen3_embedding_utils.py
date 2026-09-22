"""Shared Qwen3 embedding loading, indexing, and validation helpers."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel

from retriever_training_utils import resolve_positives


MODEL_ID = "Qwen/Qwen3-Embedding-8B"
DIMENSIONS = 768
TRAINING_MAX_LENGTH = 8192
ENCODING_MAX_LENGTH = 32768
QUERY_INSTRUCTION = (
    "Given a search query about United States government publications, "
    "retrieve passages that contain the evidence needed to answer the query."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def query_text(text: str) -> str:
    return f"Instruct: {QUERY_INSTRUCTION}\nQuery:{text}"


def load_model(model_source: str | Path = MODEL_ID) -> SentenceTransformer:
    model = SentenceTransformer(
        MODEL_ID,
        trust_remote_code=True,
        truncate_dim=DIMENSIONS,
        model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"},
    )
    model.max_seq_length = ENCODING_MAX_LENGTH
    if str(model_source) != MODEL_ID:
        model[0].auto_model = AutoModel.from_pretrained(
            str(model_source), torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(model.device)
    return model


def encode(model: SentenceTransformer, texts: list[str], batch_size: int = 12) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=False,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    if vectors.shape != (len(texts), DIMENSIONS) or not np.isfinite(vectors).all():
        raise ValueError(f"Invalid embeddings: {vectors.shape}")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Zero-norm embedding")
    vectors /= norms
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-4):
        raise ValueError("Embeddings are not normalized")
    return vectors


def audit_lengths(
    model: SentenceTransformer,
    texts: list[str],
    label: str,
    maximum_length: int = ENCODING_MAX_LENGTH,
) -> dict:
    maximum = 0
    overlong = 0
    for start in range(0, len(texts), 256):
        encoded = model.tokenizer(
            texts[start : start + 256], add_special_tokens=True, truncation=False,
            padding=False, return_length=True,
        )
        lengths = encoded["length"]
        maximum = max(maximum, max(lengths, default=0))
        overlong += sum(length > maximum_length for length in lengths)
    if overlong:
        raise ValueError(f"{label} has {overlong} texts over {maximum_length} tokens")
    return {
        "texts": len(texts), "maximum_tokens": maximum,
        "maximum_allowed_tokens": maximum_length, "overlong": overlong,
    }


def positive_map(questions: list[dict], chunks: list[dict]) -> dict[str, list[str]]:
    by_page = defaultdict(list)
    for row in chunks:
        by_page[(row["document_id"], row["page_start"])].append(row)
    return {row["question_id"]: resolve_positives(row, by_page) for row in questions}


def top_indices(corpus: np.ndarray, queries: np.ndarray, top_k: int) -> list[list[int]]:
    output = []
    for start in range(0, len(queries), 32):
        scores = corpus @ queries[start : start + 32].T
        for column in range(scores.shape[1]):
            values = scores[:, column]
            indices = np.argpartition(values, -top_k)[-top_k:]
            output.append(sorted(indices.tolist(), key=lambda i: (-float(values[i]), i)))
    return output


def retrieval_metrics(
    model: SentenceTransformer,
    questions: list[dict],
    chunks: list[dict],
    positives: dict[str, list[str]],
) -> dict:
    corpus = encode(model, [row["text"] for row in chunks])
    queries = encode(model, [query_text(row["question"]) for row in questions])
    ranked = top_indices(corpus, queries, 5)
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


def save_index(
    model: SentenceTransformer,
    chunks: list[dict],
    directory: Path,
    split: str,
    model_label: str,
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    ids = [row["chunk_id"] for row in chunks]
    length_audit = audit_lengths(model, [row["text"] for row in chunks], f"{split} chunks")
    started = time.perf_counter()
    vectors = encode(model, [row["text"] for row in chunks])
    elapsed = time.perf_counter() - started
    np.save(directory / "embeddings.npy", vectors)
    (directory / "chunk_ids.json").write_text(json.dumps(ids) + "\n")
    manifest = {
        "complete": True,
        "split": split,
        "model": model_label,
        "base_model": MODEL_ID,
        "dimensions": DIMENSIONS,
        "truncate_dim": DIMENSIONS,
        "dtype": "float32",
        "normalized": True,
        "max_sequence_length": ENCODING_MAX_LENGTH,
        "chunks": len(chunks),
        "length_audit": length_audit,
        "indexing_seconds": elapsed,
        "chunk_order_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "embeddings_sha256": sha256(directory / "embeddings.npy"),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def save_query_cache(
    model: SentenceTransformer,
    questions: list[dict],
    path: Path,
    model_label: str,
) -> dict:
    texts = [query_text(row["question"]) for row in questions]
    length_audit = audit_lengths(model, texts, f"{path.stem} queries")
    vectors = encode(model, texts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for question, vector in zip(questions, vectors, strict=True):
            stream.write(json.dumps({
                "question_id": question["question_id"],
                "embedding": vector.tolist(),
                "token_count": len(model.tokenizer.encode(
                    query_text(question["question"]), add_special_tokens=True,
                )),
                "truncated": False,
                "query_instruction": QUERY_INSTRUCTION,
                "model": model_label,
            }) + "\n")
    manifest = {
        "questions": len(questions), "model": model_label,
        "dimensions": DIMENSIONS, "query_instruction": QUERY_INSTRUCTION,
        "length_audit": length_audit, "sha256": sha256(path),
    }
    path.with_name(path.stem + "_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest

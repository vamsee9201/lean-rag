#!/usr/bin/env python3
"""Verify corrected adapter-aware Qwen3 embedding artifacts before retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SPLITS = ("train", "validation", "test_v4")
DIMENSIONS = 768
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "data/qwen_embedding_experiment/embedding_training"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def verify_index(directory: Path, declared: dict, source: Path) -> tuple[dict, set[str]]:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest != declared:
        raise ValueError(f"Root and split manifests differ for {directory}")
    ids = json.loads((directory / "chunk_ids.json").read_text())
    source_rows = read_jsonl(source)
    source_ids = [row["chunk_id"] for row in source_rows]
    if ids != source_ids:
        raise ValueError(f"Index chunks differ from the source or are out of order: {directory}")
    documents = {row["document_id"] for row in source_rows}
    if len(documents) != declared.get("documents", len(documents)):
        raise ValueError(f"Source document count differs from index manifest: {directory}")
    vectors = np.load(directory / "embeddings.npy", mmap_mode="r")
    if manifest.get("complete") is not True or manifest.get("adapter_aware") is not True:
        raise ValueError(f"Index is not complete and adapter-aware: {directory}")
    if manifest["dimensions"] != DIMENSIONS or vectors.shape != (len(ids), DIMENSIONS):
        raise ValueError(f"Unexpected index dimensions in {directory}: {vectors.shape}")
    if len(ids) != len(set(ids)) or len(ids) != manifest["chunks"]:
        raise ValueError(f"Invalid chunk IDs in {directory}")
    if sha256(directory / "embeddings.npy") != manifest["embeddings_sha256"]:
        raise ValueError(f"Embedding checksum mismatch in {directory}")
    order_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    if order_hash != manifest["chunk_order_sha256"]:
        raise ValueError(f"Chunk order checksum mismatch in {directory}")
    minimum, maximum = float("inf"), 0.0
    for start in range(0, len(vectors), 4096):
        block = np.asarray(vectors[start:start + 4096], dtype=np.float32)
        if not np.isfinite(block).all():
            raise ValueError(f"Non-finite vector in {directory}")
        norms = np.linalg.norm(block, axis=1)
        minimum = min(minimum, float(norms.min()))
        maximum = max(maximum, float(norms.max()))
        if not np.allclose(norms, 1.0, atol=2e-4):
            raise ValueError(f"Non-unit vector in {directory}")
    return {
        "chunks": len(ids), "documents": len(documents), "shape": list(vectors.shape),
        "norm_range": [minimum, maximum], "source_sha256": sha256(source),
    }, documents


def verify_queries(path: Path, declared: dict) -> dict:
    rows = read_jsonl(path)
    if len(rows) != declared["questions"] or len({row["question_id"] for row in rows}) != len(rows):
        raise ValueError(f"Invalid query count or duplicate IDs in {path}")
    if declared["dimensions"] != DIMENSIONS or sha256(path) != declared["sha256"]:
        raise ValueError(f"Query metadata mismatch in {path}")
    for row in rows:
        vector = np.asarray(row["embedding"], dtype=np.float32)
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid query vector for {row['question_id']}")
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=2e-4):
            raise ValueError(f"Non-unit query vector for {row['question_id']}")
        if row.get("truncated") is not False or not row.get("query_instruction"):
            raise ValueError(f"Invalid query audit fields for {row['question_id']}")
    return {"questions": len(rows), "sha256": declared["sha256"]}


def verify_disjoint_documents(documents_by_split: dict[str, set[str]]) -> None:
    for position, left in enumerate(SPLITS):
        for right in SPLITS[position + 1:]:
            shared = documents_by_split[left] & documents_by_split[right]
            if shared:
                raise ValueError(f"Documents shared between {left} and {right}: {sorted(shared)[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--base-validation", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT)
    args = parser.parse_args()
    manifest = json.loads((args.run / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("adapter_aware") is not True:
        raise ValueError("Corrected run manifest is not complete and adapter-aware")
    if manifest.get("training_records") != 2560:
        raise ValueError("Unexpected embedding training record count")
    if len(manifest.get("checkpoint_metrics", [])) != 2:
        raise ValueError("Expected metrics for checkpoints 80 and 160")
    if {row["checkpoint"] for row in manifest["checkpoint_metrics"]} != {"checkpoint-80", "checkpoint-160"}:
        raise ValueError("Unexpected checkpoint candidates")
    selected = args.run / "selected_adapter"
    if not list(selected.glob("adapter_model*.safetensors")):
        raise ValueError("Selected adapter weights are missing")
    indexes = {}
    queries = {}
    documents_by_split = {}
    for split in SPLITS:
        indexes[split], documents_by_split[split] = verify_index(
            args.run / "indexes" / split, manifest["indexes"][split],
            args.source_root / f"{split}_chunks.jsonl",
        )
        query_path = args.run / "query_embeddings" / f"{split}.jsonl"
        queries[split] = verify_queries(query_path, manifest["query_caches"][split])
    verify_disjoint_documents(documents_by_split)
    base_manifest = json.loads((args.base_validation / "manifest.json").read_text())
    tuned_hash = manifest["indexes"]["validation"]["embeddings_sha256"]
    if tuned_hash == base_manifest["embeddings_sha256"]:
        raise ValueError("Corrected validation index is byte-identical to the base index")
    report = {
        "status": "verified",
        "selected_checkpoint": manifest["selected_checkpoint"],
        "selected_validation": manifest["selected_validation"],
        "indexes": indexes,
        "query_caches": queries,
        "base_validation_sha256": base_manifest["embeddings_sha256"],
        "tuned_validation_sha256": tuned_hash,
    }
    output = args.run / "local_verification.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

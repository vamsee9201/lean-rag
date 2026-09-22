"""Pure validation helpers for resumable Qwen3 embedding index shards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def chunk_order_sha256(chunks: list[dict]) -> str:
    return hashlib.sha256(
        "\n".join(row["chunk_id"] for row in chunks).encode()
    ).hexdigest()


def shard_directory(root: Path, split: str, number: int) -> Path:
    return root / "index_shards" / split / f"shard-{number:04d}"


def reusable_shard(directory: Path, chunks: list[dict], split: str, number: int) -> np.ndarray | None:
    required = (directory / "embeddings.npy", directory / "chunk_ids.json", directory / "manifest.json")
    present = [path.is_file() for path in required]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"Incomplete uploaded shard: {directory}")
    manifest = json.loads(required[2].read_text())
    ids = [row["chunk_id"] for row in chunks]
    vectors = np.load(required[0], mmap_mode="r")
    if (
        manifest.get("split") != split
        or manifest.get("shard_number") != number
        or manifest.get("chunks") != len(chunks)
        or manifest.get("dimensions") != 768
        or manifest.get("chunk_order_sha256") != chunk_order_sha256(chunks)
        or manifest.get("embeddings_sha256") != file_sha256(required[0])
        or json.loads(required[1].read_text()) != ids
        or vectors.shape != (len(chunks), 768)
        or not np.isfinite(vectors).all()
        or not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-4)
    ):
        raise RuntimeError(f"Uploaded shard does not match input or contains invalid vectors: {directory}")
    return vectors

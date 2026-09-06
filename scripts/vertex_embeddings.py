"""Authenticated Vertex AI embedding helpers with strict response validation."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
from google.genai import types

from gemini_vertex import create_client
from hybrid_retrieval import normalize


MODEL = "gemini-embedding-001"
DIMENSIONS = 768


def embed_one(
    client,
    text: str,
    *,
    task_type: str,
    title: str | None = None,
    model: str = MODEL,
    dimensions: int = DIMENSIONS,
    attempts: int = 6,
) -> tuple[np.ndarray, dict]:
    config = types.EmbedContentConfig(
        task_type=task_type,
        title=title,
        output_dimensionality=dimensions,
        auto_truncate=False,
    )
    last_error = None
    for attempt in range(attempts):
        try:
            response = client.models.embed_content(model=model, contents=text, config=config)
            if len(response.embeddings or []) != 1:
                raise ValueError("Vertex returned an unexpected embedding count")
            embedding = response.embeddings[0]
            vector = normalize(np.asarray(embedding.values, dtype=np.float32))
            if len(vector) != dimensions:
                raise ValueError(f"Vertex returned {len(vector)} dimensions, expected {dimensions}")
            statistics = embedding.statistics
            truncated = getattr(statistics, "truncated", None)
            if truncated:
                raise ValueError("Vertex silently truncated an embedding input")
            return vector, {
                "token_count": getattr(statistics, "token_count", None),
                "truncated": bool(truncated),
            }
        except Exception as exc:
            last_error = exc
            if attempt + 1 == attempts:
                break
            time.sleep(min(30, 2**attempt))
    raise RuntimeError(f"Vertex embedding failed after {attempts} attempts: {last_error}")


def create_embedding_client(credentials: Path, location: str):
    return create_client(credentials, location)

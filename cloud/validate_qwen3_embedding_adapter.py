#!/usr/bin/env python3
"""Verify that SWIFT applies the selected embedding LoRA during inference."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from swift import InferRequest, TransformersEngine


MODEL = "Qwen/Qwen3-Embedding-8B"
REPO = "vamsee9201/qwen3-embedding-8b-govinfo-retriever"
OUTPUT = Path("/tmp/qwen3-embedding-adapter-validation.json")
TEXTS = [
    "Instruct: Given a search query about United States government publications, retrieve passages that contain the evidence needed to answer the query.\nQuery:What was the agency's reported fiscal year expenditure?",
    "The agency reported total fiscal year expenditures of $42 million.",
    "A congressional report describing environmental enforcement activities.",
]


def vectors(responses) -> np.ndarray:
    values = np.asarray([row.data[0].embedding for row in responses], dtype=np.float32)
    values = values[:, :768]
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return values


def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is unavailable")
    root = Path(snapshot_download(
        repo_id=REPO, allow_patterns=["selected_adapter/**"], token=os.environ["HF_TOKEN"]
    ))
    adapter = root / "selected_adapter"
    engine = TransformersEngine(
        MODEL, adapters=[str(adapter)], task_type="embedding",
        torch_dtype=torch.bfloat16, attn_impl="flash_attention_2", max_batch_size=3,
    )
    requests = [InferRequest(messages=[{"role": "user", "content": text}]) for text in TEXTS]
    tuned = vectors(engine.infer(requests))
    with engine.model.disable_adapter():
        base = vectors(engine.infer(requests))
    difference = np.abs(tuned - base)
    result = {
        "model": MODEL,
        "adapter": REPO,
        "dimensions": 768,
        "finite": bool(np.isfinite(tuned).all()),
        "tuned_norms": np.linalg.norm(tuned, axis=1).tolist(),
        "base_norms": np.linalg.norm(base, axis=1).tolist(),
        "maximum_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "cosine_base_vs_tuned": np.sum(base * tuned, axis=1).tolist(),
    }
    if not result["finite"] or np.allclose(tuned, base, rtol=1e-5, atol=1e-6):
        raise RuntimeError(f"Adapter did not materially change embeddings: {result}")
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print("QWEN3_EMBEDDING_ADAPTER_VALID", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

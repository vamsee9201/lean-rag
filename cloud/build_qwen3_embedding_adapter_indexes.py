#!/usr/bin/env python3
"""Build exact Qwen3 adapter indexes from the durably uploaded selected adapter."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from huggingface_hub import HfApi, snapshot_download

from qwen3_index_shards import chunk_order_sha256, reusable_shard, shard_directory
from qwen3_embedding_adapter_utils import (
    encode, load_engine, query_text, save_index, save_query_cache, sha256, tokenizer_for,
)
from qwen3_embedding_utils import MODEL_ID, read_jsonl


DATA = Path("/mnt/data")
OUTPUT = Path("/tmp/qwen3-embedding-adapter-indexes")
REPO = "vamsee9201/qwen3-embedding-8b-govinfo-retriever"
BASE_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
FILES = {
    "train_chunks": DATA / "train_chunks.jsonl",
    "train_questions": DATA / "train_questions.jsonl",
    "validation_chunks": DATA / "validation_chunks.jsonl",
    "validation_questions": DATA / "validation_questions.jsonl",
    "test_v4_chunks": DATA / "test_v4_chunks.jsonl",
    "test_v4_questions": DATA / "test_v4_questions.jsonl",
}
SPLITS = ("train", "validation", "test_v4")
SHARD_SIZE = 4096


def upload_split(api: HfApi, split: str) -> None:
    api.upload_folder(
        repo_id=REPO, repo_type="model", folder_path=OUTPUT / "indexes" / split,
        path_in_repo=f"adapter_aware_run2/indexes/{split}",
        commit_message=f"Upload adapter-aware {split} index",
    )
    for suffix in (".jsonl", "_manifest.json"):
        local = OUTPUT / "query_embeddings" / f"{split}{suffix}"
        api.upload_file(
            repo_id=REPO, repo_type="model", path_or_fileobj=local,
            path_in_repo=f"adapter_aware_run2/query_embeddings/{local.name}",
            commit_message=f"Upload adapter-aware {split} query artifact",
        )


def build_shard(
    api: HfApi, engine, tokenizer, split: str, number: int, chunks: list[dict]
) -> np.ndarray:
    directory = shard_directory(OUTPUT, split, number)
    directory.mkdir(parents=True, exist_ok=True)
    vectors, stats = encode(engine, tokenizer, [row["text"] for row in chunks])
    np.save(directory / "embeddings.npy", vectors)
    (directory / "chunk_ids.json").write_text(
        json.dumps([row["chunk_id"] for row in chunks]) + "\n"
    )
    manifest = {
        "split": split, "shard_number": number, "chunks": len(chunks),
        "dimensions": 768, "adapter_aware": True,
        "chunk_order_sha256": chunk_order_sha256(chunks),
        "embeddings_sha256": sha256(directory / "embeddings.npy"),
        "encoding": stats,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reusable_shard(directory, chunks, split, number)
    api.upload_folder(
        repo_id=REPO, repo_type="model", folder_path=directory,
        path_in_repo=f"adapter_aware_run2/index_shards/{split}/shard-{number:04d}",
        commit_message=f"Checkpoint adapter-aware {split} shard {number:04d}",
    )
    print(f"CHECKPOINT {split} shard {number:04d} ({len(chunks)} chunks)", flush=True)
    return vectors


def reusable_split(
    repo_path: Path, split: str, chunks: list[dict], questions: list[dict]
) -> tuple[dict, dict] | None:
    """Return verified remote manifests when a previously uploaded split is reusable."""
    index_dir = repo_path / "adapter_aware_run2" / "indexes" / split
    query_dir = repo_path / "adapter_aware_run2" / "query_embeddings"
    embeddings = index_dir / "embeddings.npy"
    ids_path = index_dir / "chunk_ids.json"
    index_manifest_path = index_dir / "manifest.json"
    query_path = query_dir / f"{split}.jsonl"
    query_manifest_path = query_dir / f"{split}_manifest.json"
    required = (
        embeddings, ids_path, index_manifest_path, query_path, query_manifest_path,
    )
    if not all(path.is_file() for path in required):
        return None
    index_manifest = json.loads(index_manifest_path.read_text())
    query_manifest = json.loads(query_manifest_path.read_text())
    expected_ids = [row["chunk_id"] for row in chunks]
    if json.loads(ids_path.read_text()) != expected_ids:
        return None
    if (
        not index_manifest.get("complete")
        or index_manifest.get("adapter_aware") is not True
        or index_manifest.get("split") != split
        or index_manifest.get("chunks") != len(chunks)
        or index_manifest.get("dimensions") != 768
        or index_manifest.get("chunk_order_sha256") != chunk_order_sha256(chunks)
        or index_manifest.get("embeddings_sha256") != sha256(embeddings)
    ):
        return None
    vectors = np.load(embeddings, mmap_mode="r")
    if vectors.shape != (len(chunks), 768) or not np.isfinite(vectors).all():
        return None
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=2e-4):
        return None
    query_rows = read_jsonl(query_path)
    if (
        query_manifest.get("adapter_aware") is not True
        or query_manifest.get("questions") != len(questions)
        or query_manifest.get("dimensions") != 768
        or query_manifest.get("sha256") != sha256(query_path)
        or [row.get("question_id") for row in query_rows]
        != [row["question_id"] for row in questions]
    ):
        return None
    query_vectors = np.asarray([row.get("embedding") for row in query_rows], dtype=np.float32)
    if (
        query_vectors.shape != (len(questions), 768)
        or not np.isfinite(query_vectors).all()
        or not np.allclose(np.linalg.norm(query_vectors, axis=1), 1.0, atol=2e-4)
        or any(row.get("truncated") is not False for row in query_rows)
    ):
        return None
    return index_manifest, query_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--max-new-shards", type=int, default=0)
    parser.add_argument("--probe-only", action="store_true")
    args = parser.parse_args()
    args.probe_only = args.probe_only or os.environ.get("QWEN_INDEX_PROBE_ONLY") == "1"
    if args.max_new_shards < 0:
        raise ValueError("--max-new-shards must be nonnegative")
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is unavailable")
    for path in FILES.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    rows = {name: read_jsonl(path) for name, path in FILES.items()}
    base_path = Path(snapshot_download(
        repo_id=MODEL_ID, revision=BASE_REVISION, token=token,
        allow_patterns=["*.json", "*.txt", "*.model", "*.safetensors", "modules.json"],
    ))
    repo_path = Path(snapshot_download(
        repo_id=REPO, token=token,
        allow_patterns=[
            "adapter_aware_run2/selected_adapter/*",
            "adapter_aware_run2/validation_manifest.json",
            "adapter_aware_run2/indexes/*/*",
            "adapter_aware_run2/index_shards/*/*/*",
            "adapter_aware_run2/query_embeddings/*",
        ],
    ))
    adapter = repo_path / "adapter_aware_run2" / "selected_adapter"
    validation_manifest_path = repo_path / "adapter_aware_run2" / "validation_manifest.json"
    if not adapter.is_dir() or not validation_manifest_path.is_file():
        raise RuntimeError("Selected adapter or validation manifest was not durably uploaded")
    validation_manifest = json.loads(validation_manifest_path.read_text())
    if validation_manifest.get("status") != "validation_complete":
        raise RuntimeError("Validation decision is incomplete")
    selected_weights = adapter / "adapter_model.safetensors"
    expected_weights_sha = validation_manifest.get("selected_validation", {}).get(
        "adapter_weights_sha256"
    )
    if (
        not selected_weights.is_file()
        or not expected_weights_sha
        or sha256(selected_weights) != expected_weights_sha
    ):
        raise RuntimeError("Selected adapter weights differ from the validated checkpoint")
    if validation_manifest.get("selected_checkpoint") != validation_manifest[
        "selected_validation"
    ].get("checkpoint"):
        raise RuntimeError("Selected checkpoint and validation decision disagree")
    for name, path in FILES.items():
        expected = validation_manifest.get("input_sha256", {}).get(name)
        if expected is None or sha256(path) != expected:
            raise RuntimeError(f"Index input differs from the validated run: {name}")
    api = HfApi()
    manifests = {}
    query_manifests = {}
    pending = []
    for split in SPLITS:
        existing = reusable_split(
            repo_path, split, rows[f"{split}_chunks"], rows[f"{split}_questions"]
        )
        if existing is None:
            pending.append(split)
        else:
            manifests[split], query_manifests[split] = existing
            print(f"RESUME verified uploaded {split} index and query cache", flush=True)
    engine = None
    tokenizer = None
    if pending:
        tokenizer = tokenizer_for(base_path)
        engine = load_engine(base_path, adapter)
        # Diagnose batch-shape sensitivity before indexing. The model's official
        # recipe uses batched, left-padded inference; BF16 is not bitwise
        # invariant to changing batch shape.
        probes = [row["text"] for row in rows["validation_chunks"][:3]]
        together, _ = encode(engine, tokenizer, probes)
        separate = np.vstack([encode(engine, tokenizer, [text])[0][0] for text in probes])
        repeated, _ = encode(engine, tokenizer, probes)
        probe_result = {
            "batch_vs_single_cosines": np.sum(together * separate, axis=1).tolist(),
            "batch_vs_single_max_abs": np.max(np.abs(together - separate), axis=1).tolist(),
            "repeat_cosines": np.sum(together * repeated, axis=1).tolist(),
            "repeat_max_abs": np.max(np.abs(together - repeated), axis=1).tolist(),
            "batch_identical_within_2e_3": bool(np.allclose(
                together, repeated, atol=2e-3, rtol=2e-3
            )),
        }
        print("BATCH_PROBE", json.dumps(probe_result), flush=True)
        if args.probe_only:
            return
        if not probe_result["batch_identical_within_2e_3"]:
            raise RuntimeError("Repeated embedding batch differs beyond BF16 tolerance")
    new_shards = 0
    for split in SPLITS:
        if split not in pending or split not in args.splits:
            continue
        chunks = rows[f"{split}_chunks"]
        questions = rows[f"{split}_questions"]
        shard_vectors = []
        shard_stats = []
        for number, start in enumerate(range(0, len(chunks), SHARD_SIZE)):
            group = chunks[start:start + SHARD_SIZE]
            remote_shard = shard_directory(repo_path / "adapter_aware_run2", split, number)
            existing_shard = reusable_shard(remote_shard, group, split, number)
            if existing_shard is not None:
                vectors = np.asarray(existing_shard)
                stats = json.loads((remote_shard / "manifest.json").read_text())["encoding"]
                print(f"RESUME verified {split} shard {number:04d}", flush=True)
            else:
                if args.max_new_shards and new_shards >= args.max_new_shards:
                    print(f"PARTIAL stopped after {new_shards} new shards", flush=True)
                    return
                vectors = build_shard(api, engine, tokenizer, split, number, group)
                stats = json.loads(
                    (shard_directory(OUTPUT, split, number) / "manifest.json").read_text()
                )["encoding"]
                new_shards += 1
            shard_vectors.append(vectors)
            shard_stats.append(stats)
        corpus = np.concatenate(shard_vectors, axis=0)
        corpus_stats = {
            "texts": len(chunks), "shard_size": SHARD_SIZE,
            "shards": len(shard_vectors),
            "maximum_tokens": max(row["maximum_tokens"] for row in shard_stats),
            "minimum_tokens": min(row["minimum_tokens"] for row in shard_stats),
            "batches": sum(row["batches"] for row in shard_stats),
            "seconds": sum(row["seconds"] for row in shard_stats),
        }
        query_vectors, query_stats = encode(
            engine, tokenizer, [query_text(row["question"]) for row in questions]
        )
        manifests[split] = save_index(
            corpus, chunks, OUTPUT / "indexes" / split, split, REPO, corpus_stats
        )
        query_manifests[split] = save_query_cache(
            query_vectors, questions, OUTPUT / "query_embeddings" / f"{split}.jsonl",
            REPO, query_stats,
        )
        upload_split(api, split)
        del corpus, query_vectors
        gc.collect()
        torch.cuda.empty_cache()
    if engine is not None:
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    if len(manifests) != len(SPLITS):
        print("PARTIAL verified completed splits", sorted(manifests), flush=True)
        return
    manifest = {
        "status": "complete", "adapter_aware": True, "base_model": MODEL_ID,
        "base_revision": BASE_REVISION,
        "seed": validation_manifest["seed"],
        "training_records": validation_manifest["training_records"],
        "checkpoint_metrics": validation_manifest["checkpoint_metrics"],
        "selected_checkpoint": validation_manifest["selected_checkpoint"],
        "selected_validation": validation_manifest["selected_validation"],
        "indexes": manifests, "query_caches": query_manifests,
        "input_sha256": validation_manifest["input_sha256"],
        "validation_manifest_sha256": sha256(validation_manifest_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    api.upload_file(
        repo_id=REPO, repo_type="model", path_or_fileobj=OUTPUT / "manifest.json",
        path_in_repo="adapter_aware_run2/manifest.json",
        commit_message="Finalize adapter-aware Qwen3 embedding indexes",
    )
    print("QWEN3_ADAPTER_INDEXES_COMPLETE", json.dumps(manifest), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"QWEN3_ADAPTER_INDEXES_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

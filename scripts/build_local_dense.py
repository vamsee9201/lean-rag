#!/usr/bin/env python3
"""Build a resumable exact dense index with a local Sentence Transformers model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from sentence_transformers import SentenceTransformer

from hybrid_retrieval import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label")
    parser.add_argument("--dimensions", type=int, default=768)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--flush-every", type=int, default=512)
    parser.add_argument("--device")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


def chunks_from_db(path: Path) -> list[dict]:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY chunk_id")]
    finally:
        db.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunks_from_db(args.index)
    ids = [row["chunk_id"] for row in chunks]
    order_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    ids_path = args.output_dir / "chunk_ids.json"
    partial_path = args.output_dir / "embeddings.partial.npy"
    final_path = args.output_dir / "embeddings.npy"
    state_path = args.output_dir / "state.json"
    manifest_path = args.output_dir / "manifest.json"
    if args.force or args.restart:
        for path in (ids_path, partial_path, final_path, state_path, manifest_path):
            path.unlink(missing_ok=True)
    if final_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and manifest.get("chunk_order_sha256") == order_hash:
            print(json.dumps(manifest, indent=2)); return
        raise SystemExit("Existing index does not match; pass --restart")
    if ids_path.exists() and json.loads(ids_path.read_text()) != ids:
        raise SystemExit("Chunk order changed during resumable build; pass --restart")
    ids_path.write_text(json.dumps(ids) + "\n")
    completed = 0
    token_count = 0
    truncated_count = 0
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["chunk_order_sha256"] != order_hash:
            raise SystemExit("Resume state does not match; pass --restart")
        completed = state["completed_chunks"]
        token_count = state["token_count"]
        truncated_count = state.get("truncated_chunks", 0)
    if partial_path.exists():
        vectors = np.lib.format.open_memmap(partial_path, mode="r+")
    else:
        vectors = np.lib.format.open_memmap(
            partial_path, mode="w+", dtype=np.float32, shape=(len(chunks), args.dimensions)
        )
    model = SentenceTransformer(args.model, trust_remote_code=True, device=args.device)
    model.max_seq_length = args.max_seq_length
    tokenizer = model.tokenizer
    for start in range(completed, len(chunks), args.flush_every):
        end = min(len(chunks), start + args.flush_every)
        texts = [row["text"] for row in chunks[start:end]]
        lengths = [len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]
        truncated_in_batch = sum(length > args.max_seq_length for length in lengths)
        batch = model.encode(
            texts, batch_size=args.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        ).astype(np.float32)
        if batch.shape != (end - start, args.dimensions) or not np.isfinite(batch).all():
            raise ValueError(f"Invalid embedding batch {batch.shape} at {start}")
        vectors[start:end] = batch
        vectors.flush()
        token_count += sum(lengths)
        truncated_count += truncated_in_batch
        state_path.write_text(json.dumps({
            "split": args.split, "model": args.model_label or args.model,
            "completed_chunks": end, "total_chunks": len(chunks),
            "token_count": token_count, "truncated_chunks": truncated_count,
            "truncation_policy": f"explicit tokenizer truncation at {args.max_seq_length} tokens",
            "chunk_order_sha256": order_hash,
        }, indent=2) + "\n")
        print(f"{end}/{len(chunks)}", flush=True)
    partial_path.replace(final_path)
    manifest = {
        "complete": True, "split": args.split, "model": args.model_label or args.model,
        "model_source": args.model, "dimensions": args.dimensions, "dtype": "float32",
        "normalized": True, "max_sequence_length": args.max_seq_length,
        "chunks": len(chunks), "token_count": token_count,
        "truncated_chunks": truncated_count,
        "truncation_policy": f"explicit tokenizer truncation at {args.max_seq_length} tokens",
        "source_bm25_index": str(args.index), "chunk_order_sha256": order_hash,
        "embeddings_sha256": sha256_file(final_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    state_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

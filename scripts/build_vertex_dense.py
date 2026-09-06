#!/usr/bin/env python3
"""Build a resumable split-isolated dense index using Vertex AI embeddings."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np

from hybrid_retrieval import sha256_file
from vertex_embeddings import DIMENSIONS, MODEL, create_embedding_client, embed_one


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--flush-every", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def chunks_from_db(path: Path) -> list[dict]:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY chunk_id")]
    finally:
        db.close()


def chunk_order_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


def main() -> None:
    args = parse_args()
    index = args.index or ROOT / "data" / "finetuning" / args.split / "bm25.sqlite3"
    output = args.output_dir or ROOT / "data" / "hybrid_experiment" / "indexes" / args.split
    output.mkdir(parents=True, exist_ok=True)
    chunks = chunks_from_db(index)
    ids = [item["chunk_id"] for item in chunks]
    ids_path = output / "chunk_ids.json"
    progress_path = output / "progress.json"
    temporary_path = output / "embeddings.npy.tmp"
    final_path = output / "embeddings.npy"
    manifest_path = output / "manifest.json"
    order_sha = chunk_order_sha(ids)

    if args.force:
        for path in (ids_path, progress_path, temporary_path, final_path, manifest_path):
            path.unlink(missing_ok=True)
    if final_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete") and manifest.get("chunk_order_sha256") == order_sha:
            print(json.dumps(manifest, indent=2))
            return
        raise SystemExit("Existing dense index does not match this chunk order; use --force")

    if ids_path.exists():
        if json.loads(ids_path.read_text()) != ids:
            raise SystemExit("Chunk order changed during a resumable run; use --force")
    else:
        ids_path.write_text(json.dumps(ids) + "\n", encoding="utf-8")

    completed = 0
    token_count = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["chunk_order_sha256"] != order_sha:
            raise SystemExit("Progress belongs to a different chunk order; use --force")
        completed = int(progress["completed_chunks"])
        token_count = int(progress.get("token_count", 0))
    if temporary_path.exists():
        vectors = np.lib.format.open_memmap(temporary_path, mode="r+")
        if vectors.shape != (len(chunks), args.dimensions):
            raise SystemExit("Partial vector file has the wrong shape; use --force")
    else:
        vectors = np.lib.format.open_memmap(
            temporary_path, mode="w+", dtype=np.float32, shape=(len(chunks), args.dimensions)
        )

    client = create_embedding_client(args.credentials, args.location)

    def run_one(index_item: tuple[int, dict]):
        position, item = index_item
        vector, stats = embed_one(
            client, item["text"], task_type="RETRIEVAL_DOCUMENT", title=item["title"],
            model=args.model, dimensions=args.dimensions,
        )
        return position, vector, stats

    next_position = completed
    pending_results: dict[int, tuple[np.ndarray, dict]] = {}
    dirty = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(run_one, (position, chunks[position])): position
            for position in range(completed, len(chunks))
        }
        for future in as_completed(futures):
            position, vector, stats = future.result()
            pending_results[position] = (vector, stats)
            while next_position in pending_results:
                vector, stats = pending_results.pop(next_position)
                vectors[next_position] = vector
                token_count += int(stats.get("token_count") or 0)
                next_position += 1
                dirty += 1
                if dirty >= args.flush_every or next_position == len(chunks):
                    vectors.flush()
                    progress = {
                        "split": args.split,
                        "model": args.model,
                        "dimensions": args.dimensions,
                        "completed_chunks": next_position,
                        "total_chunks": len(chunks),
                        "token_count": token_count,
                        "chunk_order_sha256": order_sha,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    progress_path.write_text(json.dumps(progress, indent=2) + "\n")
                    print(f"{next_position}/{len(chunks)}", flush=True)
                    dirty = 0

    temporary_path.replace(final_path)
    manifest = {
        "complete": True,
        "split": args.split,
        "model": args.model,
        "endpoint_location": args.location,
        "dimensions": args.dimensions,
        "dtype": "float32",
        "document_task_type": "RETRIEVAL_DOCUMENT",
        "query_task_type": "RETRIEVAL_QUERY",
        "auto_truncate": False,
        "chunks": len(chunks),
        "token_count": token_count,
        "source_bm25_index": str(index),
        "chunk_order_sha256": order_sha,
        "embeddings_sha256": sha256_file(final_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "managed_model_revision_available": False,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    progress_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

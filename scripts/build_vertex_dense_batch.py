#!/usr/bin/env python3
"""Build a resumable Vertex Gemini dense index through a GCS batch prediction job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote

import numpy as np
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account

from hybrid_retrieval import normalize, sha256_file
from vertex_embeddings import DIMENSIONS, MODEL
from vertex_embeddings import create_embedding_client, embed_one


ROOT = Path(__file__).resolve().parents[1]
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--bucket", help="Existing project-owned GCS bucket name")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--cleanup-gcs", action="store_true")
    parser.add_argument("--cleanup-raw", action="store_true", help="Keep compact stats and remove raw vector JSONL")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, help="Limit chunks for an API compatibility smoke test")
    parser.add_argument("--camel-config", action="store_true", help="Use camelCase batch config fields")
    return parser.parse_args()


def load_chunks(path: Path) -> list[dict]:
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY chunk_id")]
    finally:
        db.close()


def authorized(credentials_path: Path):
    metadata = json.loads(credentials_path.read_text(encoding="utf-8"))
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=[SCOPE]
    )
    return metadata["project_id"], AuthorizedSession(credentials)


def request(session, method: str, url: str, **kwargs) -> dict:
    response = session.request(method, url, timeout=120, **kwargs)
    if not response.ok:
        raise RuntimeError(f"Google API {response.status_code}: {response.text[:1200]}")
    return response.json() if response.content else {}


def upload(session, bucket: str, object_name: str, path: Path) -> None:
    url = f"https://storage.googleapis.com/upload/storage/v1/b/{quote(bucket, safe='')}/o"
    with path.open("rb") as stream:
        response = session.post(
            url,
            params={"uploadType": "media", "name": object_name},
            data=stream,
            headers={"Content-Type": "application/jsonl"},
            timeout=1800,
        )
    if not response.ok:
        raise RuntimeError(f"GCS upload {response.status_code}: {response.text[:1200]}")


def list_objects(session, bucket: str, prefix: str) -> list[str]:
    url = f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o"
    token = None
    names = []
    while True:
        params = {"prefix": prefix}
        if token:
            params["pageToken"] = token
        payload = request(session, "GET", url, params=params)
        names.extend(item["name"] for item in payload.get("items", []))
        token = payload.get("nextPageToken")
        if not token:
            return names


def download_to_stream(session, bucket: str, object_name: str, destination) -> None:
    url = (
        f"https://storage.googleapis.com/download/storage/v1/b/{quote(bucket, safe='')}/o/"
        f"{quote(object_name, safe='')}"
    )
    response = session.get(url, params={"alt": "media"}, timeout=1800, stream=True)
    if not response.ok:
        raise RuntimeError(f"GCS download {response.status_code}: {response.text[:1200]}")
    for block in response.iter_content(chunk_size=8 * 1024 * 1024):
        if block:
            destination.write(block)


def delete_object(session, bucket: str, object_name: str) -> None:
    url = (
        f"https://storage.googleapis.com/storage/v1/b/{quote(bucket, safe='')}/o/"
        f"{quote(object_name, safe='')}"
    )
    response = session.delete(url, timeout=120)
    if response.status_code not in (204, 404):
        raise RuntimeError(f"GCS delete {response.status_code}: {response.text[:1200]}")


def choose_bucket(session, project: str) -> str:
    payload = request(
        session, "GET", "https://storage.googleapis.com/storage/v1/b", params={"project": project}
    )
    buckets = [
        item["name"] for item in payload.get("items", [])
        if item.get("location", "").casefold() in {"us-central1", "us"}
    ]
    if not buckets:
        raise RuntimeError("No project-owned US or us-central1 GCS bucket is available")
    return sorted(buckets)[0]


def output_vector(row: dict) -> tuple[str, list[float], dict]:
    if "response" in row:
        response = row["response"]
        values = response["embedding"]["values"]
        usage = response.get("usageMetadata", {})
        stats = {
            "token_count": int(response.get("tokenCount") or usage.get("promptTokenCount", 0)),
            "truncated": False,
        }
        return row["key"], values, stats
    predictions = row.get("predictions") or []
    if not predictions:
        raise ValueError(f"Batch row has no embedding response: {str(row)[:300]}")
    embedding = predictions[0]["embeddings"]
    stats = embedding.get("statistics", {})
    return row.get("key") or row["instance"]["key"], embedding["values"], {
        "token_count": int(stats.get("token_count", 0)),
        "truncated": bool(stats.get("truncated", False)),
    }


def main() -> None:
    args = parse_args()
    source_index = args.index or ROOT / "data" / "finetuning" / args.split / "bm25.sqlite3"
    output = args.output_dir or ROOT / "data" / "hybrid_experiment" / "indexes" / args.split
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "embeddings.npy"
    manifest_path = output / "manifest.json"
    state_path = output / "batch_state.json"
    requests_path = output / "batch_requests.jsonl"
    raw_output_path = output / "batch_output.jsonl"
    statistics_path = output / "batch_statistics.jsonl"
    ids_path = output / "chunk_ids.json"
    if final_path.exists() and manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("complete"):
            print(json.dumps(manifest, indent=2))
            return
    if args.force:
        for path in (final_path, manifest_path, state_path, requests_path, raw_output_path, ids_path):
            path.unlink(missing_ok=True)

    chunks = load_chunks(source_index)
    if args.limit:
        chunks = chunks[: args.limit]
    ids = [row["chunk_id"] for row in chunks]
    order_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    ids_path.write_text(json.dumps(ids) + "\n", encoding="utf-8")
    if not requests_path.exists():
        with requests_path.open("w", encoding="utf-8") as stream:
            for row in chunks:
                config = (
                    {"outputDimensionality": args.dimensions, "title": row["title"],
                     "taskType": "RETRIEVAL_DOCUMENT", "autoTruncate": False}
                    if args.camel_config else
                    {"output_dimensionality": args.dimensions, "title": row["title"],
                     "task_type": "RETRIEVAL_DOCUMENT", "auto_truncate": False}
                )
                stream.write(json.dumps({
                    "key": row["chunk_id"],
                    "request": {
                        "content": {"parts": [{"text": row["text"]}]},
                        ("embedContentConfig" if args.camel_config else "embed_content_config"): config,
                    },
                }, ensure_ascii=False) + "\n")

    project, session = authorized(args.credentials)
    bucket = args.bucket or choose_bucket(session, project)
    run_id = f"{args.split}-{order_sha[:12]}"
    prefix = f"lean-rag/vertex-embeddings/{run_id}"
    input_object = f"{prefix}/input.jsonl"
    output_prefix = f"{prefix}/output"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state["chunk_order_sha256"] != order_sha:
            raise RuntimeError("Batch state belongs to a different chunk order")
    else:
        upload(session, bucket, input_object, requests_path)
        endpoint = (
            f"https://{args.location}-aiplatform.googleapis.com/v1/projects/{project}/"
            f"locations/{args.location}/batchPredictionJobs"
        )
        payload = {
            "displayName": f"lean-rag-{run_id}",
            "model": f"publishers/google/models/{args.model}",
            "inputConfig": {
                "instancesFormat": "jsonl",
                "gcsSource": {"uris": [f"gs://{bucket}/{input_object}"]},
            },
            "outputConfig": {
                "predictionsFormat": "jsonl",
                "gcsDestination": {"outputUriPrefix": f"gs://{bucket}/{output_prefix}"},
            },
        }
        job = request(session, "POST", endpoint, json=payload)
        state = {
            "job_name": job["name"], "bucket": bucket, "input_object": input_object,
            "output_prefix": output_prefix, "chunk_order_sha256": order_sha,
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        print(json.dumps(state, indent=2), flush=True)
    job_url = f"https://{args.location}-aiplatform.googleapis.com/v1/{state['job_name']}"
    if args.no_wait:
        job = request(session, "GET", job_url)
        print(json.dumps({"job_name": state["job_name"], "state": job.get("state")}, indent=2))
        return

    while True:
        job = request(session, "GET", job_url)
        job_state = job.get("state", "JOB_STATE_UNSPECIFIED")
        print(job_state, flush=True)
        if job_state in TERMINAL_STATES:
            break
        time.sleep(args.poll_seconds)
    if job_state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Vertex batch job ended as {job_state}: {job.get('error')}")

    output_directory = job.get("outputInfo", {}).get("gcsOutputDirectory")
    if output_directory and output_directory.startswith(f"gs://{bucket}/"):
        job_output_prefix = output_directory[len(f"gs://{bucket}/") :].rstrip("/")
    else:
        job_output_prefix = state["output_prefix"]
    objects = list_objects(session, bucket, job_output_prefix)
    prediction_objects = [name for name in objects if name.endswith(".jsonl")]
    if not prediction_objects:
        raise RuntimeError(f"Vertex batch job produced no JSONL objects under {job_output_prefix}")
    raw_temporary = raw_output_path.with_suffix(".jsonl.tmp")
    with raw_temporary.open("wb") as destination:
        for name in sorted(prediction_objects):
            before = destination.tell()
            download_to_stream(session, bucket, name, destination)
            destination.flush()
            if destination.tell() > before:
                destination.write(b"\n")
    raw_temporary.replace(raw_output_path)

    vectors_by_id = {}
    token_count = 0
    server_dimensions = set()
    overlong = []
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    with raw_output_path.open(encoding="utf-8") as stream, statistics_path.open("w", encoding="utf-8") as stats_stream:
        for line in stream:
            if not line.strip():
                continue
            key, values, stats = output_vector(json.loads(line))
            stats_stream.write(json.dumps({"chunk_id": key, **stats}) + "\n")
            if key in vectors_by_id:
                raise ValueError(f"Duplicate Vertex batch key: {key}")
            if stats["truncated"]:
                raise ValueError(f"Vertex truncated chunk {key}")
            server_dimensions.add(len(values))
            if len(values) < args.dimensions:
                raise ValueError(f"Vertex returned {len(values)} dimensions for {key}")
            # Vertex batch currently returns the 3,072-dimensional parent vector even when
            # output_dimensionality is requested. Gemini Embedding uses Matryoshka dimensions;
            # the requested representation is the normalized prefix, matching online output.
            vector = normalize(np.asarray(values[: args.dimensions], dtype=np.float32))
            vectors_by_id[key] = vector
            token_count += stats["token_count"]
            if stats["token_count"] > 2048:
                overlong.append((key, stats["token_count"]))
    if set(vectors_by_id) != set(ids):
        raise ValueError(
            f"Batch alignment failed: missing={len(set(ids)-set(vectors_by_id))}, "
            f"extra={len(set(vectors_by_id)-set(ids))}"
        )
    adjustments = []
    if overlong:
        client = create_embedding_client(args.credentials, args.location)
        for number, (key, original_tokens) in enumerate(overlong, 1):
            item = chunks_by_id[key]
            characters = max(256, int(len(item["text"]) * 1800 / original_tokens))
            while True:
                text = item["text"][:characters]
                try:
                    vector, strict_stats = embed_one(
                        client, text, task_type="RETRIEVAL_DOCUMENT", title=item["title"],
                        model=args.model, dimensions=args.dimensions, attempts=1,
                    )
                    break
                except RuntimeError as exc:
                    if "longer than the maximum" not in str(exc) or characters <= 256:
                        raise
                    characters = max(256, int(characters * 0.85))
            vectors_by_id[key] = vector
            adjustments.append({
                "chunk_id": key, "original_token_count": original_tokens,
                "embedding_text_characters": characters,
                "strict_token_count": strict_stats["token_count"], "truncated": False,
            })
            print(f"strictly re-embedded overlong chunk {number}/{len(overlong)}", flush=True)
    adjustments_path = output / "overlong_adjustments.jsonl"
    with adjustments_path.open("w", encoding="utf-8") as stream:
        for row in adjustments:
            stream.write(json.dumps(row) + "\n")
    matrix = np.stack([vectors_by_id[chunk_id] for chunk_id in ids]).astype(np.float32)
    np.save(final_path, matrix)
    manifest = {
        "complete": True, "split": args.split, "model": args.model,
        "endpoint_location": args.location, "dimensions": args.dimensions, "dtype": "float32",
        "document_task_type": "RETRIEVAL_DOCUMENT", "query_task_type": "RETRIEVAL_QUERY",
        "auto_truncate": False, "chunks": len(ids), "token_count": token_count,
        "server_dimensions": sorted(server_dimensions),
        "client_dimension_reduction": any(value != args.dimensions for value in server_dimensions),
        "overlong_chunks_reembedded_strictly": len(adjustments),
        "overlong_adjustments": str(adjustments_path),
        "batch_statistics": str(statistics_path),
        "source_bm25_index": str(source_index), "chunk_order_sha256": order_sha,
        "embeddings_sha256": sha256_file(final_path),
        "batch_job_name": state["job_name"], "batch_output_prefix": job_output_prefix,
        "managed_model_revision_available": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if args.cleanup_gcs:
        for name in list_objects(session, bucket, prefix):
            delete_object(session, bucket, name)
    if args.cleanup_raw:
        raw_output_path.unlink(missing_ok=True)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

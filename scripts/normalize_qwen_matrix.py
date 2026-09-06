#!/usr/bin/env python3
"""Normalize downloaded ms-swift matrix outputs into the common answer schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


LABELS = {
    "qwen-base": "qwen/qwen3.5-9b-base",
    "qwen-bm25-lora": "qwen/qwen3.5-9b-bm25-lora-step500",
    "qwen-hybrid-lora": "qwen/qwen3.5-9b-vertex-hybrid-lora",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def clean(text: str) -> str:
    return re.sub(r"^\s*<think>\s*</think>\s*", "", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    question_ids = [row["question_id"] for row in questions]
    question_by_text = {row["question"]: row["question_id"] for row in questions}
    if len(question_by_text) != len(questions):
        raise ValueError("Benchmark contains duplicate question text")
    manifest = json.loads(args.manifest.read_text())
    records = []
    for state, label in LABELS.items():
        for setup in ("bm25_top5", "vertex_hybrid_top5"):
            path = args.raw_dir / f"{state}__{setup}.jsonl"
            rows = read_jsonl(path)
            if len(rows) != len(question_ids):
                raise ValueError(f"{path} contains {len(rows)} rows, expected {len(question_ids)}")
            timing = manifest["outputs"][f"{state}:{setup}"]["elapsed_seconds"] / len(rows)
            seen = set()
            for row in rows:
                messages = row.get("messages") or []
                user = next((message.get("content", "") for message in messages if message.get("role") == "user"), "")
                marker = "\n\nQUESTION\n"
                if marker not in user or user.rsplit(marker, 1)[1] not in question_by_text:
                    raise ValueError(f"Cannot align raw response in {path}")
                question_id = question_by_text[user.rsplit(marker, 1)[1]]
                if question_id in seen:
                    raise ValueError(f"Duplicate raw response for {state}/{setup}/{question_id}")
                seen.add(question_id)
                answer = clean(row.get("response") or "")
                if not answer:
                    raise ValueError(f"Empty answer for {state}/{setup}/{question_id}")
                records.append({
                    "model": label, "setup": setup, "question_id": question_id,
                    "status": "ok", "answer": answer, "elapsed_seconds": timing,
                    "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
                    "stats": {},
                })
            if seen != set(question_ids):
                raise ValueError(f"Missing raw responses in {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"answers": len(records), "models": list(LABELS.values())}, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Qwen3 embedding token lengths without loading model weights."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from transformers import AutoTokenizer


DATA = Path("/mnt/data")
MODEL = "Qwen/Qwen3-Embedding-8B"


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


THRESHOLDS = (1024, 2048, 3072, 4096, 8192, 16384, 32768)


def audit(tokenizer, texts: list[str]) -> dict:
    lengths = []
    for start in range(0, len(texts), 256):
        lengths.extend(tokenizer(
            texts[start : start + 256], add_special_tokens=True,
            truncation=False, padding=False, return_length=True,
        )["length"])
    ordered = sorted(lengths)
    return {
        "texts": len(lengths), "minimum": ordered[0], "maximum": ordered[-1],
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "p99": ordered[int(0.99 * (len(ordered) - 1))],
        "over_threshold": {
            str(limit): sum(length > limit for length in lengths)
            for limit in THRESHOLDS
        },
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    training = rows(DATA / "train_swift.jsonl")
    training_texts = []
    for row in training:
        training_texts.append(row["messages"][0]["content"])
        training_texts.extend(group[0]["content"] for group in row["positive_messages"])
        training_texts.extend(group[0]["content"] for group in row["negative_messages"])
    result = {"training": audit(tokenizer, training_texts)}
    for name in ("train_chunks", "validation_chunks", "test_v4_chunks"):
        result[name] = audit(tokenizer, [row["text"] for row in rows(DATA / f"{name}.jsonl")])
    result["selected_safe_ceiling"] = next((
        limit for limit in THRESHOLDS
        if all(section["over_threshold"][str(limit)] == 0 for section in result.values()
               if isinstance(section, dict) and "over_threshold" in section)
    ), None)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

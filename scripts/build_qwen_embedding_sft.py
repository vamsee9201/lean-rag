#!/usr/bin/env python3
"""Build and verify the fourth experiment's train and validation SFT files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/qwen_embedding_experiment"
EXPECTED = {"train": 2256, "validation": 111}
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result = {}
    for split, expected in EXPECTED.items():
        destination = EXP / "sft" / split
        output = destination / "sft.jsonl"
        metadata = destination / "sft_metadata.jsonl"
        retrieval = EXP / "retrieval" / f"{split}_qwen3_tuned_candidates.jsonl"
        subprocess.run([
            sys.executable,
            str(ROOT / "scripts/build_hybrid_sft.py"),
            "--split", split,
            "--retrieval", str(retrieval),
            "--output", str(output),
            "--metadata", str(metadata),
            "--max-tokens", "4096",
            "--model-revision", BASE_REVISION,
            "--force",
        ], check=True)
        records = read_jsonl(output)
        audit = read_jsonl(metadata)
        if len(records) != expected or len(audit) != expected:
            raise ValueError(f"{split} produced {len(records)}/{len(audit)} records, expected {expected}")
        if any(row["prompt_tokens"] > 4096 for row in audit):
            raise ValueError(f"{split} contains an overlength prompt")
        accepted = read_jsonl(ROOT / f"data/finetuning/{split}/sft_metadata.jsonl")
        if [row["question_id"] for row in audit] != [row["question_id"] for row in accepted]:
            raise ValueError(f"{split} changed accepted question ordering")
        answerable = sum(bool(row["answerable"]) for row in audit)
        result[split] = {
            "records": len(records),
            "answerable": answerable,
            "unanswerable": len(audit) - answerable,
            "maximum_prompt_tokens": max(row["prompt_tokens"] for row in audit),
            "oracle_forced": sum(bool(row["oracle_forced"]) for row in audit),
            "sft_sha256": sha256(output),
            "metadata_sha256": sha256(metadata),
        }
    manifest = {
        "status": "verified",
        "base_model": "Qwen/Qwen3.5-9B",
        "tokenizer_revision": BASE_REVISION,
        "retriever": "qwen3_tuned_hybrid_top5",
        "maximum_prompt_tokens": 4096,
        "splits": result,
    }
    path = EXP / "sft/manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

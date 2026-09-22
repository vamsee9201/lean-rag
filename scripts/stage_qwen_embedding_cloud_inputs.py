#!/usr/bin/env python3
"""Stage only verified fourth-experiment SFT and inference inputs for cloud jobs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/qwen_embedding_experiment"
STAGE = EXP / "cloud_stage/qwen_embedding"
SETUPS = (
    "bm25_top5", "vertex_hybrid_top5", "qwen3_untuned_dense_top5",
    "qwen3_untuned_hybrid_top5", "qwen3_tuned_hybrid_top5",
)


def count(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_checked(source: Path, destination: Path, expected: int) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    records = count(source)
    if records != expected:
        raise ValueError(f"{source} contains {records} records, expected {expected}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    if digest(source) != digest(destination):
        raise RuntimeError(f"Copy checksum mismatch for {source}")
    return {"records": records, "sha256": digest(destination), "path": str(destination)}


def main() -> None:
    files = {
        "sft/train_sft.jsonl": copy_checked(
            EXP / "sft/train/sft.jsonl", STAGE / "sft/train_sft.jsonl", 2256
        ),
        "sft/validation_sft.jsonl": copy_checked(
            EXP / "sft/validation/sft.jsonl", STAGE / "sft/validation_sft.jsonl", 111
        ),
    }
    for setup in SETUPS:
        relative = f"inference/{setup}.jsonl"
        files[relative] = copy_checked(
            EXP / "inference_inputs" / f"{setup}.jsonl", STAGE / relative, 50
        )
    source_manifest = json.loads((EXP / "inference_inputs/manifest.json").read_text())
    if set(source_manifest["setups"]) != set(SETUPS):
        raise ValueError("Inference manifest does not contain the frozen five retrievers")
    manifest = {
        "status": "verified", "files": files,
        "questions_sha256": source_manifest["questions_sha256"],
        "retrieval_sha256": source_manifest["retrieval_sha256"],
        "sft_manifest_sha256": digest(EXP / "sft/manifest.json"),
    }
    (STAGE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

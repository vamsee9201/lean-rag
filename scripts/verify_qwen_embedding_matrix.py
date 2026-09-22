#!/usr/bin/env python3
"""Verify the fourth experiment's 20 matched answer cells."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


MODELS = {
    "google/gemini-3.8-flash",
    "qwen/qwen3.5-9b-base",
    "qwen/qwen3.5-9b-local-hybrid-lora",
    "qwen/qwen3.5-9b-qwen-embedding-lora",
}
SETUPS = {
    "bm25_top5",
    "vertex_hybrid_top5",
    "qwen3_untuned_dense_top5",
    "qwen3_untuned_hybrid_top5",
    "qwen3_tuned_hybrid_top5",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def verify_matrix(rows: list[dict], inputs_manifest: dict) -> dict:
    if len(rows) != 1000 or any(row.get("status") != "ok" for row in rows):
        raise ValueError("Expected exactly 1,000 successful answer records")
    keys = [(row["model"], row["setup"], row["question_id"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Answer matrix contains duplicate model/setup/question keys")
    questions = {row["question_id"] for row in rows}
    if len(questions) != 50 or {row["model"] for row in rows} != MODELS or {row["setup"] for row in rows} != SETUPS:
        raise ValueError("Answer matrix identities do not match the frozen 4x5 design")
    counts = Counter((row["model"], row["setup"]) for row in rows)
    if set(counts.values()) != {50} or len(counts) != 20:
        raise ValueError(f"Expected 20 cells of 50 answers: {counts}")
    by_evidence = defaultdict(set)
    expected_system = hashlib.sha256(inputs_manifest["system_prompt"].encode()).hexdigest()
    for row in rows:
        setup, question_id = row["setup"], row["question_id"]
        expected_prompt = inputs_manifest["setups"][setup]["evidence_hashes"][question_id]
        if row.get("prompt_sha256") != expected_prompt:
            raise ValueError(f"Answer used an unexpected question or evidence: {setup}/{question_id}")
        if row.get("system_prompt_sha256") != expected_system:
            raise ValueError(f"Answer used an unexpected system prompt: {setup}/{question_id}")
        by_evidence[(setup, question_id)].add(row["prompt_sha256"])
    mismatches = [key for key, values in by_evidence.items() if len(values) != 1]
    if mismatches:
        raise ValueError(f"Generators received different prompts for {mismatches[:3]}")
    return {
        "status": "verified",
        "answers": len(rows),
        "questions": len(questions),
        "models": sorted(MODELS),
        "retrievers": sorted(SETUPS),
        "cells": len(counts),
        "answers_per_cell": 50,
        "matched_prompt_hashes": len(by_evidence),
        "system_prompt_sha256": expected_system,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--inputs-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = read_jsonl(args.answers)
    inputs_manifest = json.loads(args.inputs_manifest.read_text())
    report = verify_matrix(rows, inputs_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Normalize fourth-experiment ms-swift outputs into the common answer schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

from build_raft_sft import SYSTEM_PROMPT


LABELS = {
    "qwen-base": "qwen/qwen3.5-9b-base",
    "qwen-previous-local-lora": "qwen/qwen3.5-9b-local-hybrid-lora",
    "qwen-qwen-embedding-lora": "qwen/qwen3.5-9b-qwen-embedding-lora",
}
SETUPS = (
    "bm25_top5",
    "vertex_hybrid_top5",
    "qwen3_untuned_dense_top5",
    "qwen3_untuned_hybrid_top5",
    "qwen3_tuned_hybrid_top5",
)


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
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--gemini", type=Path,
        help="Raw resumable Gemini answers; successful records are verified and merged.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    questions = read_jsonl(args.questions)
    question_ids = {row["question_id"] for row in questions}
    question_by_text = {row["question"]: row["question_id"] for row in questions}
    if len(question_by_text) != len(questions):
        raise ValueError("Benchmark contains duplicate question text")
    manifest = json.loads(args.manifest.read_text())
    expected_inputs = []
    for setup in SETUPS:
        path = args.input_dir / f"{setup}.jsonl"
        setup_rows = read_jsonl(path)
        if len(setup_rows) != len(question_ids):
            raise ValueError(f"{path} contains {len(setup_rows)} rows")
        expected_inputs.extend((setup, row) for row in setup_rows)
    if len(expected_inputs) != 250:
        raise ValueError("Combined Qwen input order must contain 250 prompts")
    records = []
    for state, label in LABELS.items():
        path = args.raw_dir / f"{state}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != len(expected_inputs):
            raise ValueError(f"{path} contains {len(rows)} rows, expected {len(expected_inputs)}")
        timing = manifest["outputs"][state]["elapsed_seconds"] / len(rows)
        seen = set()
        for row, (setup, expected_input) in zip(rows, expected_inputs, strict=True):
            messages = row.get("messages") or []
            actual_prompt = [
                message for message in messages
                if message.get("role") in {"system", "user"}
            ]
            if actual_prompt != expected_input["messages"]:
                raise ValueError(f"Raw system prompt, question, or evidence changed in {path}")
            user = next(
                (message.get("content", "") for message in messages if message.get("role") == "user"),
                "",
            )
            expected_user = next(
                message["content"] for message in expected_input["messages"] if message["role"] == "user"
            )
            if user != expected_user:
                raise ValueError(f"Raw response order or prompt changed in {path}")
            marker = "\n\nQUESTION\n"
            question_text = user.rsplit(marker, 1)[1] if marker in user else None
            if question_text not in question_by_text:
                raise ValueError(f"Cannot align raw response in {path}")
            question_id = question_by_text[question_text]
            key = (setup, question_id)
            if key in seen:
                raise ValueError(f"Duplicate raw response for {state}/{setup}/{question_id}")
            seen.add(key)
            answer = clean(row.get("response") or "")
            if not answer:
                raise ValueError(f"Empty answer for {state}/{setup}/{question_id}")
            records.append({
                "model": label,
                "setup": setup,
                "question_id": question_id,
                "status": "ok",
                "answer": answer,
                "elapsed_seconds": timing,
                "prompt_sha256": hashlib.sha256(user.encode()).hexdigest(),
                "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
                "stats": {},
            })
        if len(seen) != 250:
            raise ValueError(f"Missing raw responses in {path}")
    expected = len(LABELS) * len(SETUPS) * len(question_ids)
    if args.gemini:
        expected_prompts = {}
        for setup, row in expected_inputs:
            user = next(message["content"] for message in row["messages"] if message["role"] == "user")
            marker = "\n\nQUESTION\n"
            question_text = user.rsplit(marker, 1)[1] if marker in user else None
            if question_text not in question_by_text:
                raise ValueError(f"Cannot align frozen Gemini prompt in {setup}")
            key = (setup, question_by_text[question_text])
            if key in expected_prompts:
                raise ValueError(f"Duplicate frozen Gemini prompt: {key}")
            expected_prompts[key] = hashlib.sha256(user.encode()).hexdigest()
        system_sha = hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()
        successful = {}
        for row in read_jsonl(args.gemini):
            if row.get("status") != "ok":
                continue
            if row.get("model") != "google/gemini-3.8-flash":
                raise ValueError("Unexpected successful Gemini model identity")
            key = (row["setup"], row["question_id"])
            if key not in expected_prompts or key in successful:
                raise ValueError(f"Unexpected or duplicate successful Gemini answer: {key}")
            if (row.get("prompt_sha256") != expected_prompts[key]
                or row.get("system_prompt_sha256") != system_sha):
                raise ValueError(f"Gemini used a different frozen prompt: {key}")
            successful[key] = row
        if set(successful) != set(expected_prompts):
            raise ValueError(f"Gemini has {len(successful)} of {len(expected_prompts)} answers")
        records.extend(successful[key] for key in sorted(successful))
        expected += len(expected_prompts)
    if len(records) != expected:
        raise ValueError(f"Produced {len(records)} records, expected {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    models = list(LABELS.values()) + (["google/gemini-3.8-flash"] if args.gemini else [])
    print(json.dumps({"answers": len(records), "models": models}, indent=2))


if __name__ == "__main__":
    main()

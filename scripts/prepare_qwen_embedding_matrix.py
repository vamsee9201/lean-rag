#!/usr/bin/env python3
"""Freeze the five fourth-experiment retrieval columns and inference inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from build_qwen_matrix_inputs import read_jsonl
from build_raft_sft import SYSTEM_PROMPT, clip_context, sft_record


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/qwen_embedding_experiment"
QUESTIONS = EXP / "benchmark/questions.jsonl"
RETRIEVAL_FILES = (
    EXP / "retrieval/test_bm25.jsonl",
    EXP / "retrieval/test_vertex_hybrid.jsonl",
    EXP / "retrieval/test_qwen3_untuned_dense.jsonl",
    EXP / "retrieval/test_qwen3_untuned_hybrid.jsonl",
    EXP / "retrieval/test_qwen3_tuned_hybrid.jsonl",
)
SETUPS = (
    "bm25_top5",
    "vertex_hybrid_top5",
    "qwen3_untuned_dense_top5",
    "qwen3_untuned_hybrid_top5",
    "qwen3_tuned_hybrid_top5",
)
INFERENCE_LIMIT = 8192
MAX_NEW_TOKENS = 220
BASE_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"


def prompt_tokens(tokenizer, messages: list[dict]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
    )
    return len(encoded["input_ids"] if hasattr(encoded, "keys") else encoded)


def main() -> None:
    questions = {row["question_id"]: row for row in read_jsonl(QUESTIONS)}
    if len(questions) != 50:
        raise ValueError(f"Expected 50 questions, found {len(questions)}")
    retrieval = [row for path in RETRIEVAL_FILES for row in read_jsonl(path)]
    keys = [(row["setup"], row["question_id"]) for row in retrieval]
    expected = {(setup, question_id) for setup in SETUPS for question_id in questions}
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError(f"Retrieval matrix is incomplete or duplicated: {len(keys)} rows")
    output_dir = EXP / "inference_inputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B", revision=BASE_REVISION)
    combined = EXP / "retrieval/test_all.jsonl"
    with combined.open("w", encoding="utf-8") as stream:
        for row in sorted(retrieval, key=lambda item: (item["setup"], item["question_id"])):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "questions_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "retrieval_sha256": hashlib.sha256(combined.read_bytes()).hexdigest(),
        "system_prompt": SYSTEM_PROMPT,
        "tokenizer_revision": BASE_REVISION,
        "setups": {},
    }
    for setup in SETUPS:
        path = output_dir / f"{setup}.jsonl"
        rows = sorted((row for row in retrieval if row["setup"] == setup), key=lambda row: row["question_id"])
        hashes = {}
        lengths = {}
        clipped = {}
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                question = questions[row["question_id"]]
                record = sft_record(question["question"], row["results"], "")
                record["messages"] = record["messages"][:-1]
                count = prompt_tokens(tokenizer, record["messages"])
                if count + MAX_NEW_TOKENS > INFERENCE_LIMIT:
                    # Clip without consulting gold passages. This is a frozen,
                    # generator-independent fallback for unusually long chunks.
                    for chars_per_chunk in range(2000, 349, -50):
                        record = sft_record(
                            question["question"],
                            clip_context(row["results"], None, chars_per_chunk), "",
                        )
                        record["messages"] = record["messages"][:-1]
                        count = prompt_tokens(tokenizer, record["messages"])
                        if count + MAX_NEW_TOKENS <= INFERENCE_LIMIT:
                            clipped[row["question_id"]] = chars_per_chunk
                            break
                    else:
                        raise ValueError(
                            f"{setup}/{row['question_id']} cannot fit the "
                            f"{INFERENCE_LIMIT}-token inference limit"
                        )
                lengths[row["question_id"]] = count
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                user = record["messages"][-1]["content"].encode()
                hashes[row["question_id"]] = hashlib.sha256(user).hexdigest()
        manifest["setups"][setup] = {
            "records": len(rows),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evidence_hashes": hashes,
            "maximum_prompt_tokens": max(lengths.values()),
            "prompt_token_counts": lengths,
            "clipped_chars_per_chunk": clipped,
        }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({setup: manifest["setups"][setup]["records"] for setup in SETUPS}, indent=2))


if __name__ == "__main__":
    main()

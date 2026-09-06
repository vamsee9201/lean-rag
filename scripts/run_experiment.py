#!/usr/bin/env python3
"""Run every local generator on the same questions and frozen retrieval evidence."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "benchmark" / "questions.jsonl"
RETRIEVAL = ROOT / "data" / "retrieval" / "results.jsonl"
OUTPUT = ROOT / "data" / "runs" / "answers.jsonl"
MODELS = [
    "qwen/qwen3-1.7b",
    "qwen/qwen3.5-2b",
    "qwen/qwen3.5-4b",
    "qwen/qwen3.5-9b",
]
SETUPS = [
    "bm25_top5",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS)
    parser.add_argument("--retrieval", type=Path, default=RETRIEVAL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--api-model",
        help="Route requests to this loaded API identifier while retaining the model label in output",
    )
    parser.add_argument("--setups", nargs="+", default=SETUPS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evidence_text(results: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(results, 1):
        blocks.append(
            f"[{index}] [{item['document_id']} p.{item['page_start']}] "
            f"TITLE={item['title']}\n{item['text'][:6000]}"
        )
    return "\n\n".join(blocks)


def answer(base_url: str, model: str, question: str, setup: str, results: list[dict]) -> dict:
    if setup == "no_context":
        system = (
            "Answer the question concisely from your existing knowledge. If you cannot determine "
            "the answer reliably, say 'Insufficient evidence.'"
        )
        prompt = question
    else:
        system = (
            "Answer using only the supplied evidence. If it does not contain the answer, say exactly "
            "'Insufficient evidence.' Cite supporting claims as [DOCUMENT_ID p.PAGE]. Keep the answer concise."
        )
        prompt = f"EVIDENCE\n{evidence_text(results)}\n\nQUESTION\n{question}"
    payload = {
        "model": model,
        "system_prompt": system,
        "input": prompt,
        "reasoning": "off",
        "temperature": 0,
        "max_output_tokens": 220,
        "context_length": 16384,
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url}/api/v1/chat", json=payload, timeout=600)
    elapsed = time.perf_counter() - started
    if not response.ok:
        raise RuntimeError(f"LM Studio HTTP {response.status_code}: {response.text[:1000]}")
    body = response.json()
    messages = [item["content"] for item in body.get("output", []) if item.get("type") == "message"]
    if not messages:
        raise ValueError("Model returned no answer message")
    return {"answer": messages[-1].strip(), "stats": body.get("stats", {}), "elapsed_seconds": elapsed}


def main() -> None:
    args = parse_args()
    questions = {record["question_id"]: record for record in read_jsonl(args.questions)}
    retrieval = {
        (record["question_id"], record["setup"]): record for record in read_jsonl(args.retrieval)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.force:
        args.output.unlink(missing_ok=True)
    completed = set()
    if args.output.exists():
        completed = {
            (record["model"], record["setup"], record["question_id"])
            for record in read_jsonl(args.output)
            if record.get("status") == "ok"
        }
    total = len(args.models) * len(args.setups) * len(questions)
    selected_models = set(args.models)
    selected_setups = set(args.setups)
    done = sum(
        model in selected_models and setup in selected_setups
        for model, setup, _question_id in completed
    )
    with args.output.open("a", encoding="utf-8") as stream:
        for model in args.models:
            for setup in args.setups:
                pending = []
                for question_id, question in questions.items():
                    key = (model, setup, question_id)
                    if key in completed:
                        continue
                    pending.append((question_id, question, retrieval[(question_id, setup)]))

                def run_one(item):
                    question_id, question, retrieval_record = item
                    try:
                        result = answer(
                            args.base_url, args.api_model or model, question["question"], setup,
                            retrieval_record["results"]
                        )
                        record = {
                            "model": model,
                            "setup": setup,
                            "question_id": question_id,
                            "status": "ok",
                            **result,
                        }
                    except Exception as exc:
                        record = {
                            "model": model,
                            "setup": setup,
                            "question_id": question_id,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    return record

                with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                    futures = [executor.submit(run_one, item) for item in pending]
                    for future in as_completed(futures):
                        record = future.result()
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                        stream.flush()
                        done += 1
                        print(
                            f"{done}/{total} {model} {setup} {record['question_id']} {record['status']}",
                            flush=True,
                        )


if __name__ == "__main__":
    main()

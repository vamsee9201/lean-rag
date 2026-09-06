#!/usr/bin/env python3
"""Add Gemini answers to one or more frozen retrieval setups through Vertex AI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from gemini_vertex import create_client, generation_config, usage_dict
from run_experiment import evidence_text, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
MODEL_LABEL = "google/gemini-3.8-flash"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials", type=Path, default=ROOT / "ai-lab-fasa.json")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/benchmark/questions.jsonl")
    parser.add_argument("--retrieval", type=Path, default=ROOT / "data/retrieval/results.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/runs/answers.jsonl")
    parser.add_argument("--api-model", default="gemini-3.8-flash")
    parser.add_argument("--model-label", default=MODEL_LABEL)
    parser.add_argument("--location", default="global")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--setups", nargs="+", default=["bm25_top5"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    retrieval = {
        (row["question_id"], row["setup"]): row for row in read_jsonl(args.retrieval)
    }
    completed = set()
    if args.output.exists():
        completed = {
            (row["model"], row["setup"], row["question_id"])
            for row in read_jsonl(args.output)
            if row.get("status") == "ok"
        }
    pending = []
    for setup in args.setups:
        for question_id, question in questions.items():
            if (args.model_label, setup, question_id) not in completed:
                pending.append((question_id, question, retrieval[(question_id, setup)]))
    client = create_client(args.credentials, args.location)
    system = (
        "Answer using only the supplied evidence. If it does not contain the answer, say exactly "
        "'Insufficient evidence.' Cite supporting claims as [DOCUMENT_ID p.PAGE]. Keep the answer concise."
    )

    def run_one(item) -> dict:
        question_id, question, retrieval_record = item
        setup = retrieval_record["setup"]
        prompt = (
            f"EVIDENCE\n{evidence_text(retrieval_record['results'])}"
            f"\n\nQUESTION\n{question['question']}"
        )
        started = time.perf_counter()
        try:
            response = client.models.generate_content(
                model=args.api_model,
                contents=prompt,
                config=generation_config(system, 220),
            )
            elapsed = time.perf_counter() - started
            answer = (response.text or "").strip()
            if not answer:
                raise ValueError("Gemini returned no answer text")
            stats = usage_dict(response)
            output_tokens = stats.get("output_tokens")
            stats["tokens_per_second"] = (
                None if not output_tokens or not elapsed else output_tokens / elapsed
            )
            return {
                "model": args.model_label,
                "api_model": args.api_model,
                "setup": setup,
                "question_id": question_id,
                "status": "ok",
                "answer": answer,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "stats": stats,
                "elapsed_seconds": elapsed,
            }
        except Exception as exc:
            return {
                "model": args.model_label,
                "api_model": args.api_model,
                "setup": setup,
                "question_id": question_id,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(questions) * len(args.setups)
    done = total - len(pending)
    with args.output.open("a", encoding="utf-8") as stream:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(run_one, item) for item in pending]
            for future in as_completed(futures):
                record = future.result()
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                done += 1
                print(f"{done}/{total} {record['question_id']} {record['status']}", flush=True)


if __name__ == "__main__":
    main()

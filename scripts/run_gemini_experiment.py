#!/usr/bin/env python3
"""Add Gemini answers to one or more frozen retrieval setups through Vertex AI."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from google.genai.errors import ClientError

from gemini_vertex import create_client, estimated_cost_usd, generation_config, usage_dict
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
    parser.add_argument(
        "--input-dir", type=Path,
        help="Frozen per-setup prompt files shared with the local generators.",
    )
    parser.add_argument("--max-cost-usd", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_cost_usd is not None and args.api_model != "gemini-3.8-flash":
        raise ValueError("The budget estimator is priced for gemini-3.8-flash only")
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    retrieval = {
        (row["question_id"], row["setup"]): row for row in read_jsonl(args.retrieval)
    }
    frozen_prompts = {}
    if args.input_dir:
        manifest = json.loads((args.input_dir / "manifest.json").read_text())
        for setup in args.setups:
            path = args.input_dir / f"{setup}.jsonl"
            rows = read_jsonl(path)
            ordered_ids = sorted(questions)
            if len(rows) != len(ordered_ids):
                raise ValueError(f"Expected {len(ordered_ids)} frozen prompts in {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["setups"][setup]["sha256"]:
                raise ValueError(f"Frozen prompt file checksum mismatch: {path}")
            for question_id, row in zip(ordered_ids, rows, strict=True):
                messages = row["messages"]
                if [message["role"] for message in messages] != ["system", "user"]:
                    raise ValueError(f"Unexpected frozen message roles: {setup}/{question_id}")
                system_text, prompt_text = (message["content"] for message in messages)
                if system_text != manifest["system_prompt"]:
                    raise ValueError(f"Frozen system prompt changed: {setup}/{question_id}")
                if not prompt_text.endswith("\n\nQUESTION\n" + questions[question_id]["question"]):
                    raise ValueError(f"Frozen question order changed: {setup}/{question_id}")
                expected_hash = manifest["setups"][setup]["evidence_hashes"][question_id]
                if hashlib.sha256(prompt_text.encode()).hexdigest() != expected_hash:
                    raise ValueError(f"Frozen evidence hash mismatch: {setup}/{question_id}")
                frozen_prompts[(question_id, setup)] = (system_text, prompt_text)
    completed = set()
    spent = 0.0
    if args.output.exists():
        existing = read_jsonl(args.output)
        if frozen_prompts:
            seen_success = set()
            for row in existing:
                if row.get("status") != "ok" or row.get("model") != args.model_label:
                    continue
                key = (row["question_id"], row["setup"])
                if key not in frozen_prompts or key in seen_success:
                    raise ValueError(f"Unexpected or duplicate successful Gemini answer: {key}")
                frozen_system, frozen_user = frozen_prompts[key]
                if (row.get("prompt_sha256") != hashlib.sha256(frozen_user.encode()).hexdigest()
                    or row.get("system_prompt_sha256") != hashlib.sha256(frozen_system.encode()).hexdigest()):
                    raise ValueError(f"Existing Gemini answer used a different frozen prompt: {key}")
                seen_success.add(key)
        completed = {
            (row["model"], row["setup"], row["question_id"])
            for row in existing
            if row.get("status") == "ok"
        }
        spent = sum(
            estimated_cost_usd(row.get("stats") or {}, args.location)
            for row in existing if row.get("model") == args.model_label
        )
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
        frozen = frozen_prompts.get((question_id, setup))
        if frozen:
            system_for_request, prompt = frozen
        else:
            system_for_request = system
            prompt = (
                f"EVIDENCE\n{evidence_text(retrieval_record['results'])}"
                f"\n\nQUESTION\n{question['question']}"
            )
        started = time.perf_counter()
        response = None
        try:
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=args.api_model,
                        contents=prompt,
                        config=generation_config(system_for_request, 220),
                    )
                    break
                except ClientError as exc:
                    if exc.code != 429 or attempt == 2:
                        raise
                    # Google's Vertex guidance recommends no more than two
                    # retries with exponential backoff for transient 429s.
                    time.sleep(2 ** attempt)
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
                "system_prompt_sha256": hashlib.sha256(system_for_request.encode()).hexdigest(),
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
                "stats": usage_dict(response) if response is not None else {},
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total = len(questions) * len(args.setups)
    done = total - len(pending)
    with args.output.open("a", encoding="utf-8") as stream:
        if args.max_cost_usd is not None:
            if args.max_cost_usd <= 0:
                raise ValueError("--max-cost-usd must be positive")
            # Keep the budgeted path sequential so no additional request can
            # slip past a spending check while earlier requests are in flight.
            for item in pending:
                if spent + 0.01 > args.max_cost_usd:
                    raise RuntimeError(
                        f"Gemini answer budget gate: ${spent:.4f} recorded, "
                        f"${args.max_cost_usd:.4f} stage cap"
                    )
                record = run_one(item)
                spent += estimated_cost_usd(record.get("stats") or {}, args.location)
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                done += 1
                print(f"{done}/{total} {record['question_id']} {record['status']}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [executor.submit(run_one, item) for item in pending]
                for future in as_completed(futures):
                    record = future.result()
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    done += 1
                    print(f"{done}/{total} {record['question_id']} {record['status']}", flush=True)
    if frozen_prompts:
        successful = {
            (row["question_id"], row["setup"])
            for row in read_jsonl(args.output)
            if row.get("model") == args.model_label and row.get("status") == "ok"
        }
        if successful != set(frozen_prompts):
            raise RuntimeError(
                f"Gemini answers incomplete: {len(successful)} successful of "
                f"{len(frozen_prompts)} frozen prompts; rerun to retry failures"
            )


if __name__ == "__main__":
    main()

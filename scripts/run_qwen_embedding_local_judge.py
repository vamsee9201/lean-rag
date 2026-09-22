#!/usr/bin/env python3
"""Resumable blinded Qwen judge through the user's local LM Studio server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

import requests


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def evaluation_id(record: dict) -> str:
    prompt = next(
        row["content"] for row in record["messages"] if row["role"] == "user"
    )
    prefix = "EVALUATION_ID\n"
    if not prompt.startswith(prefix):
        raise ValueError("Judge prompt is missing its evaluation ID")
    return prompt[len(prefix):].split("\n", 1)[0]


def parse_scores(content: str, labels: set[str]) -> list[dict]:
    value = content.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*", "", value)
        value = re.sub(r"\s*```$", "", value)
    start, end = value.find("["), value.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("Judge did not return a JSON array")
    scores = json.loads(value[start:end + 1])
    if not isinstance(scores, list) or len(scores) != len(labels):
        raise ValueError("Judge returned the wrong number of ratings")
    if {row.get("id") for row in scores} != labels:
        raise ValueError("Judge changed or duplicated blinded candidate IDs")
    for row in scores:
        if (
            row.get("answer_score") not in (0, 1, 2)
            or row.get("citation_score") not in (0, 1)
            or row.get("unsupported_claim") not in (0, 1)
        ):
            raise ValueError("Judge returned an invalid score")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--max-output-tokens", type=int, default=2400)
    parser.add_argument("--limit", type=int, default=0, help="Probe only the first N inputs")
    args = parser.parse_args()
    prompts = read_jsonl(args.inputs)
    mapping = json.loads(args.mapping.read_text())
    expected = {row["evaluation_id"]: set(row["labels"]) for row in mapping["mappings"]}
    if len(expected) != len(prompts):
        raise ValueError("Judge prompt and mapping counts differ")
    ids = [evaluation_id(row) for row in prompts]
    if set(ids) != set(expected) or len(set(ids)) != len(ids):
        raise ValueError("Judge inputs and mapping IDs differ")
    completed = {}
    if args.output.is_file():
        for row in read_jsonl(args.output):
            key = evaluation_id(row)
            if key in completed:
                raise ValueError(f"Duplicate local judge result: {key}")
            parse_scores(row["response"], expected[key])
            completed[key] = row
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index, prompt in enumerate(prompts[:args.limit or None], 1):
        key = ids[index - 1]
        if key in completed:
            continue
        messages = prompt["messages"]
        system = next(row["content"] for row in messages if row["role"] == "system")
        user = next(row["content"] for row in messages if row["role"] == "user")
        for attempt in range(3):
            correction = "" if attempt == 0 else (
                "\n\nReturn exactly one JSON rating for each ID: "
                + ", ".join(sorted(expected[key]))
                + ". Use only integer answer_score 0-2, citation_score 0-1, "
                "unsupported_claim 0-1."
            )
            started = time.perf_counter()
            response = requests.post(
                f"{args.base_url}/api/v1/chat",
                json={
                    "model": args.model, "system_prompt": system,
                    "input": user + correction, "reasoning": "off",
                    "temperature": 0, "max_output_tokens": args.max_output_tokens,
                    "context_length": 16384,
                },
                timeout=900,
            )
            response.raise_for_status()
            body = response.json()
            output = [
                item["content"] for item in body.get("output", [])
                if item.get("type") == "message"
            ]
            content = output[-1] if output else ""
            try:
                parse_scores(content, expected[key])
                break
            except (ValueError, TypeError, KeyError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"Judge failed {key} schema after three attempts") from exc
        result = {
            "messages": messages, "response": content,
            "model": args.model, "backend": "LM Studio MLX 4-bit",
            "elapsed_seconds": time.perf_counter() - started,
            "attempt": attempt + 1,
        }
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(f"LOCAL_QWEN_JUDGE {index}/{len(prompts)} {key}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Call a local LM Studio model with benchmark-safe defaults."""

from __future__ import annotations

import argparse
import json
import os
import time

import requests


DEFAULT_BASE_URL = "http://127.0.0.1:1234"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="qwen/qwen3.5-9b")
    parser.add_argument("--question", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--base-url", default=os.environ.get("LM_STUDIO_BASE_URL", DEFAULT_BASE_URL))
    return parser.parse_args()


def grounded_answer(
    *,
    base_url: str,
    model: str,
    question: str,
    evidence: str,
    context_length: int,
    max_output_tokens: int,
) -> dict:
    payload = {
        "model": model,
        "system_prompt": (
            "Answer using only the supplied evidence. If the evidence does not "
            "contain the answer, say that it is insufficient. Keep the answer "
            "concise and cite the document ID and page shown in the evidence."
        ),
        "input": f"EVIDENCE\n{evidence}\n\nQUESTION\n{question}",
        "reasoning": "off",
        "temperature": 0,
        "max_output_tokens": max_output_tokens,
        "context_length": context_length,
    }
    started = time.perf_counter()
    response = requests.post(f"{base_url}/api/v1/chat", json=payload, timeout=300)
    if not response.ok:
        raise RuntimeError(
            f"LM Studio returned HTTP {response.status_code}: {response.text[:1000]}"
        )
    result = response.json()
    result["client_elapsed_seconds"] = time.perf_counter() - started
    result["benchmark_request"] = {
        "model": model,
        "question": question,
        "evidence": evidence,
        "context_length": context_length,
        "max_output_tokens": max_output_tokens,
        "reasoning": "off",
        "temperature": 0,
    }
    return result


def main() -> None:
    args = parse_args()
    result = grounded_answer(
        base_url=args.base_url,
        model=args.model,
        question=args.question,
        evidence=args.evidence,
        context_length=args.context_length,
        max_output_tokens=args.max_output_tokens,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

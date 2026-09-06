#!/usr/bin/env python3
"""Audit chat-template token lengths for one or more JSONL SFT datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--limit", type=int, default=4096)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    report = []
    failed = []
    for path in args.paths:
        lengths = []
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                tokens = tokenizer.apply_chat_template(
                    row["messages"], tokenize=True, add_generation_prompt=False
                )
                input_ids = tokens["input_ids"] if hasattr(tokens, "keys") else tokens
                length = len(input_ids)
                lengths.append(length)
                if length > args.limit:
                    failed.append({"path": str(path), "line": line_number, "tokens": length})
        report.append({
            "path": str(path), "records": len(lengths), "max_tokens": max(lengths),
            "mean_tokens": sum(lengths) / len(lengths), "limit": args.limit,
        })
    print(json.dumps({"datasets": report, "over_limit": failed}, indent=2))
    if failed:
        raise SystemExit(f"{len(failed)} prompts exceed {args.limit} tokens")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build byte-aligned Qwen inference datasets for BM25 and Vertex-hybrid evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_raft_sft import SYSTEM_PROMPT, sft_record


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    questions = {row["question_id"]: row for row in read_jsonl(args.questions)}
    retrieval = read_jsonl(args.retrieval)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"system_prompt": SYSTEM_PROMPT, "setups": {}}
    for setup in sorted({row["setup"] for row in retrieval}):
        rows = [row for row in retrieval if row["setup"] == setup]
        path = args.output_dir / f"{setup}.jsonl"
        evidence_hashes = {}
        with path.open("w", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: item["question_id"]):
                question = questions[row["question_id"]]
                record = sft_record(question["question"], row["results"], "")
                record["messages"] = record["messages"][:-1]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                evidence = record["messages"][-1]["content"].encode()
                evidence_hashes[row["question_id"]] = hashlib.sha256(evidence).hexdigest()
        manifest["setups"][setup] = {
            "records": len(rows), "path": str(path), "evidence_hashes": evidence_hashes,
        }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v["records"] for k, v in manifest["setups"].items()}, indent=2))


if __name__ == "__main__":
    main()

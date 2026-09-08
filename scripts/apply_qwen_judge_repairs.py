#!/usr/bin/env python3
"""Replace malformed grouped Qwen judge ratings with validated individual reratings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_qwen_judge_repairs import invalid, read_jsonl
from judge_answers_gemini import parse_array


def evaluation_id(row: dict) -> str:
    user = next(message["content"] for message in row["messages"] if message["role"] == "user")
    return user[len("EVALUATION_ID\n"):].split("\n", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--repair-raw", type=Path, required=True)
    parser.add_argument("--repair-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repair_map = {
        row["evaluation_id"]: row
        for row in json.loads(args.repair_mapping.read_text())["mappings"]
    }
    replacements = {}
    for row in read_jsonl(args.repair_raw):
        repair_id = evaluation_id(row)
        mapping = repair_map.get(repair_id)
        if mapping is None:
            raise ValueError(f"Unknown repair evaluation: {repair_id}")
        ratings = parse_array(row.get("response") or "")
        if len(ratings) != 1 or ratings[0].get("id") != "a01" or invalid(ratings[0]):
            raise ValueError(f"Invalid individual repair response: {repair_id}")
        rating = dict(ratings[0])
        rating["id"] = mapping["original_label"]
        replacements[(mapping["original_evaluation_id"], mapping["original_label"])] = rating
    if len(replacements) != len(repair_map):
        raise ValueError("Repair output is incomplete or duplicated")

    used = set()
    output_rows = []
    for row in read_jsonl(args.raw):
        original_id = evaluation_id(row)
        ratings = parse_array(row.get("response") or "")
        for index, rating in enumerate(ratings):
            if invalid(rating):
                key = (original_id, rating.get("id"))
                if key not in replacements:
                    raise ValueError(f"No repair for malformed rating: {key}")
                ratings[index] = replacements[key]
                used.add(key)
        updated = dict(row)
        updated["response"] = json.dumps(ratings, ensure_ascii=False)
        output_rows.append(updated)
    if used != set(replacements):
        raise ValueError("Some repair ratings were not applied")
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_rows))
    print(json.dumps({"groups": len(output_rows), "repairs_applied": len(used)}, indent=2))


if __name__ == "__main__":
    main()

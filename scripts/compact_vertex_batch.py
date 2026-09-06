#!/usr/bin/env python3
"""Replace retained raw Vertex batch vectors with verified compact per-chunk statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_vertex_dense_batch import output_vector


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index_dirs", nargs="+", type=Path)
    parser.add_argument("--delete-raw", action="store_true")
    args = parser.parse_args()
    for directory in args.index_dirs:
        raw = directory / "batch_output.jsonl"
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        expected = int(manifest["chunks"])
        stats = directory / "batch_statistics.jsonl"
        temporary = stats.with_suffix(".jsonl.tmp")
        count = 0
        with raw.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as destination:
            for line in source:
                if not line.strip():
                    continue
                key, values, values_stats = output_vector(json.loads(line))
                destination.write(json.dumps({
                    "chunk_id": key, **values_stats, "server_dimensions": len(values),
                }) + "\n")
                count += 1
        if count != expected:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"{directory}: compacted {count} rows, expected {expected}")
        temporary.replace(stats)
        manifest["batch_statistics"] = str(stats.resolve())
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        if args.delete_raw:
            raw.unlink()
        print(json.dumps({"index": str(directory), "rows": count, "raw_deleted": args.delete_raw}))


if __name__ == "__main__":
    main()

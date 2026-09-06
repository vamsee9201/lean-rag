#!/usr/bin/env python3
"""Export publication-friendly CSV tables for the frozen hybrid experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_LABELS = {
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "qwen/qwen3.5-9b-base": "Base Qwen3.5 9B",
    "qwen/qwen3.5-9b-bm25-lora-step500": "BM25-trained Qwen3.5 9B",
    "qwen/qwen3.5-9b-vertex-hybrid-lora": "Hybrid-trained Qwen3.5 9B",
}
SETUP_LABELS = {"bm25_top5": "BM25", "vertex_hybrid_top5": "Vertex hybrid"}


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    costs = json.loads(args.costs.read_text())

    quality = []
    latency = []
    for model, setups in summary["aggregates"].items():
        for setup, metrics in setups.items():
            primary = metrics["primary_dual_judge"]
            quality.append({
                "generator": MODEL_LABELS[model],
                "retriever": SETUP_LABELS[setup],
                "primary_score": primary["mean"],
                "primary_ci95_low": primary["ci95"][0],
                "primary_ci95_high": primary["ci95"][1],
                "gemini_judge": metrics["gemini_judge"]["mean"],
                "qwen_judge": metrics["qwen_judge"]["mean"],
                "judge_citation_score": metrics["citation_judge"]["mean"],
                "exact_citation_recall": metrics["citation_recall"]["mean"],
                "token_f1": metrics["token_f1"]["mean"],
                "answers": primary["n"],
            })
            latency.append({
                "generator": MODEL_LABELS[model],
                "retriever": SETUP_LABELS[setup],
                "mean_generation_seconds": metrics["mean_elapsed_seconds"],
                "measurement": "Vertex request latency" if model.startswith("google/")
                else "GPU cell elapsed time divided by 50",
            })
    write_csv(args.output_dir / "answer_quality_2x4.csv", list(quality[0]), quality)
    write_csv(args.output_dir / "generation_latency.csv", list(latency[0]), latency)

    categories = []
    for model, setups in summary["category_results"].items():
        for setup, category_metrics in setups.items():
            for category, metrics in category_metrics.items():
                categories.append({
                    "generator": MODEL_LABELS[model],
                    "retriever": SETUP_LABELS[setup],
                    "category": category,
                    "primary_score": metrics["primary_dual_judge"]["mean"],
                    "exact_citation_recall": metrics["citation_recall"]["mean"],
                    "token_f1": metrics["token_f1"]["mean"],
                    "questions": metrics["primary_dual_judge"]["n"],
                })
    write_csv(args.output_dir / "category_results.csv", list(categories[0]), categories)

    paired = []
    for comparison, metrics in summary["paired"].items():
        paired.append({
            "comparison": comparison,
            "mean_difference": metrics["mean"],
            "ci95_low": metrics["ci95"][0],
            "ci95_high": metrics["ci95"][1],
            "one_sided_95_lower": metrics["one_sided_95_lower"],
            "questions": metrics["n"],
            "noninferiority_margin": metrics.get("noninferiority_margin"),
            "noninferior": metrics.get("noninferior"),
        })
    write_csv(args.output_dir / "paired_effects.csv", list(paired[0]), paired)

    cost_rows = []
    for component in [
        "embedding", "gemini_answers", "gemini_judge", "hf_smoke", "hf_full",
        "hf_matrix_inference", "hf_qwen_judge",
    ]:
        item = costs[component]
        cost_rows.append({"component": component, "estimated_usd": item["estimated_usd"]})
    cost_rows.extend([
        {"component": "vertex_api_total", "estimated_usd": costs["totals"]["vertex_api_estimated_usd"]},
        {"component": "hf_gpu_total", "estimated_usd": costs["totals"]["hf_gpu_estimated_usd"]},
        {"component": "experiment_total", "estimated_usd": costs["totals"]["combined_estimated_usd"]},
    ])
    write_csv(args.output_dir / "costs.csv", ["component", "estimated_usd"], cost_rows)
    print(json.dumps({"tables": 5, "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()

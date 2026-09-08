#!/usr/bin/env python3
"""Export presentation-ready CSV tables for the third experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODEL_LABELS = {
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "qwen/qwen3.5-9b-base": "Base Qwen3.5 9B",
    "qwen/qwen3.5-9b-bm25-lora-step500": "BM25-trained Qwen",
    "qwen/qwen3.5-9b-vertex-hybrid-lora": "Vertex-hybrid-trained Qwen",
    "qwen/qwen3.5-9b-local-hybrid-lora": "Local-hybrid-trained Qwen",
}
SETUP_LABELS = {
    "bm25_top5": "BM25",
    "vertex_hybrid_top5": "Vertex hybrid",
    "untuned_local_hybrid_top5": "Untuned local hybrid",
    "tuned_local_hybrid_top5": "Tuned local hybrid",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--retrieval-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    answer_rows = []
    for model, setups in data["aggregates"].items():
        for setup, metrics in setups.items():
            score = metrics["primary_score"]
            answer_rows.append({
                "generator": MODEL_LABELS[model],
                "retriever": SETUP_LABELS[setup],
                "primary_score": score["mean"],
                "ci95_low": score["ci95"][0],
                "ci95_high": score["ci95"][1],
                "gemini_judge_score": metrics["gemini_judge"]["mean"],
                "qwen_judge_score": metrics["qwen_judge"]["mean"],
                "citation_recall": metrics["citation_recall"],
                "token_f1": metrics["token_f1"],
                "unsupported_claim_rate": metrics["unsupported_claim_rate"],
                "mean_elapsed_seconds": metrics["mean_elapsed_seconds"],
                "answers": score["n"],
            })
    answer_rows.sort(key=lambda r: (list(MODEL_LABELS.values()).index(r["generator"]), list(SETUP_LABELS.values()).index(r["retriever"])))
    write_csv(args.output_dir / "answer_quality_2x4.csv", answer_rows)

    category_rows = []
    for model, setups in data["category_results"].items():
        for setup, categories in setups.items():
            for category, metrics in categories.items():
                score = metrics["primary_score"]
                category_rows.append({
                    "generator": MODEL_LABELS[model], "retriever": SETUP_LABELS[setup],
                    "category": category, "primary_score": score["mean"],
                    "ci95_low": score["ci95"][0], "ci95_high": score["ci95"][1],
                    "citation_recall": metrics["citation_recall"], "token_f1": metrics["token_f1"],
                    "unsupported_claim_rate": metrics["unsupported_claim_rate"], "questions": score["n"],
                })
    write_csv(args.output_dir / "category_results.csv", category_rows)

    comparison_rows = []
    for name, metrics in data["paired"].items():
        comparison_rows.append({
            "comparison": name, "mean_difference": metrics["mean"],
            "ci95_low": metrics["ci95"][0], "ci95_high": metrics["ci95"][1],
            "one_sided_95_lower": metrics["one_sided_95_lower"],
            "noninferiority_margin": metrics["noninferiority_margin"],
            "noninferior": metrics["noninferior"],
            "bootstrap_p_value": metrics["p_value_bootstrap"],
            "holm_adjusted_p": metrics["holm_adjusted_p"], "questions": metrics["n"],
        })
    write_csv(args.output_dir / "paired_comparisons.csv", comparison_rows)

    bm25 = json.loads((args.retrieval_dir / "test_bm25_summary.json").read_text())
    vertex = json.loads((args.retrieval_dir / "test_vertex_hybrid_summary.json").read_text())
    untuned = json.loads((args.retrieval_dir / "test_untuned_local_summary.json").read_text())
    tuned = json.loads((args.retrieval_dir / "test_tuned_local_summary.json").read_text())
    retrieval = {
        "BM25": bm25,
        "Vertex dense": vertex["retriever_metrics"]["vertex_dense"],
        "Vertex hybrid": vertex["selected"]["metrics"],
        "Untuned local dense": untuned["retriever_metrics"]["local_dense"],
        "Untuned local hybrid": untuned["retriever_metrics"]["local_hybrid"],
        "Tuned local dense": tuned["retriever_metrics"]["local_dense"],
        "Tuned local hybrid": tuned["retriever_metrics"]["local_hybrid"],
    }
    retrieval_rows = []
    for name, metrics in retrieval.items():
        retrieval_rows.append({
            "retriever": name,
            "all_gold_recall_at_5": metrics["all_gold_recall"],
            "mean_gold_passage_recall_at_5": metrics["mean_gold_passage_recall"],
            "mrr": metrics["mrr"], "ndcg_at_5": metrics["mean_ndcg"],
            "cross_document_all_gold_recall": metrics.get("cross_document_all_gold_recall"),
        })
    write_csv(args.output_dir / "retrieval_comparison.csv", retrieval_rows)
    print(json.dumps({"answer_rows": len(answer_rows), "category_rows": len(category_rows),
                      "comparisons": len(comparison_rows), "retrievers": len(retrieval_rows)}, indent=2))


if __name__ == "__main__":
    main()

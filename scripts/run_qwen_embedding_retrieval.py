#!/usr/bin/env python3
"""Tune, freeze, and materialize fourth-experiment Qwen retrieval outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/qwen_embedding_experiment"
MODEL_ROOT = EXP / "models/qwen3_embedding"
ADAPTER_RUN = MODEL_ROOT / "adapter_aware_run2"
OUT = EXP / "retrieval"
INSTRUCTION = (
    "Given a search query about United States government publications, "
    "retrieve passages that contain the evidence needed to answer the query."
)
BASE_LABEL = "Qwen/Qwen3-Embedding-8B"
TUNED_LABEL = "vamsee9201/qwen3-embedding-8b-govinfo-retriever"


def run(arguments: list[str]) -> None:
    command = [sys.executable, str(ROOT / "scripts/retrieve_local_hybrid.py"), *arguments]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def common(
    questions: Path, bm25: Path, index: Path, cache: Path, model: str,
    label: str, output: Path, setup: str,
) -> list[str]:
    return [
        "--questions", str(questions), "--bm25-index", str(bm25),
        "--dense-index", str(index), "--query-cache", str(cache),
        "--model", model, "--model-label", label, "--output", str(output),
        "--setup", setup, "--query-instruction", INSTRUCTION,
        "--truncate-dim", "768", "--max-seq-length", "32768", "--force",
    ]


def selected(path: Path) -> dict:
    return json.loads(path.read_text())["selected"]


def config_args(config: dict) -> list[str]:
    arguments = [
        "--candidate-depth", str(config["candidate_depth"]),
        "--dense-weight", str(config["dense_weight"]),
        "--rrf-k", str(config["rrf_k"]),
    ]
    if config["document_cap"] is not None:
        arguments += ["--document-cap", str(config["document_cap"])]
    return arguments


def write_frozen(path: Path, settings: dict) -> dict:
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
    sealed = {**settings, "sha256": hashlib.sha256(payload).hexdigest()}
    if path.exists():
        existing = json.loads(path.read_text())
        recorded_hash = existing.pop("sha256", None)
        actual_hash = hashlib.sha256(
            json.dumps(existing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recorded_hash != actual_hash or existing != settings:
            raise RuntimeError(
                f"Frozen retriever settings changed: {path}. Create a separately versioned run."
            )
        return sealed
    path.write_text(json.dumps(sealed, indent=2) + "\n")
    return sealed


def main() -> None:
    required = [
        MODEL_ROOT / "base_indexes/validation/manifest.json",
        ADAPTER_RUN / "manifest.json",
        ADAPTER_RUN / "indexes/validation/manifest.json",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    corrected = json.loads((ADAPTER_RUN / "manifest.json").read_text())
    base_hash = json.loads(
        (MODEL_ROOT / "base_indexes/validation/manifest.json").read_text()
    )["embeddings_sha256"]
    tuned_hash = corrected["indexes"]["validation"]["embeddings_sha256"]
    if base_hash == tuned_hash:
        raise RuntimeError("Corrected tuned validation index is identical to base")
    OUT.mkdir(parents=True, exist_ok=True)
    validation_questions = ROOT / "data/finetuning/validation/question_sources.jsonl"
    validation_bm25 = ROOT / "data/finetuning/validation/bm25.sqlite3"
    untuned_validation = OUT / "validation_qwen3_untuned_hybrid.jsonl"
    run(common(
        validation_questions, validation_bm25, MODEL_ROOT / "base_indexes/validation",
        MODEL_ROOT / "base_query_embeddings/validation.jsonl", BASE_LABEL, BASE_LABEL,
        untuned_validation, "qwen3_untuned_hybrid_top5",
    ) + ["--tune-validation", "--dense-output", str(OUT / "validation_qwen3_untuned_dense.jsonl")])
    tuned_validation = OUT / "validation_qwen3_tuned_hybrid.jsonl"
    run(common(
        validation_questions, validation_bm25, ADAPTER_RUN / "indexes/validation",
        ADAPTER_RUN / "query_embeddings/validation.jsonl", TUNED_LABEL, TUNED_LABEL,
        tuned_validation, "qwen3_tuned_hybrid_top5",
    ) + ["--tune-validation", "--dense-output", str(OUT / "validation_qwen3_tuned_dense.jsonl")])
    untuned_config = selected(untuned_validation.with_name(untuned_validation.stem + "_summary.json"))
    tuned_config = selected(tuned_validation.with_name(tuned_validation.stem + "_summary.json"))
    adapter_weights = ADAPTER_RUN / "selected_adapter/adapter_model.safetensors"
    if not adapter_weights.is_file():
        raise FileNotFoundError(adapter_weights)
    frozen = {
        "untuned": {key: untuned_config[key] for key in ("candidate_depth", "dense_weight", "document_cap", "rrf_k")},
        "tuned": {key: tuned_config[key] for key in ("candidate_depth", "dense_weight", "document_cap", "rrf_k")},
        "selection_split": "existing validation only", "query_instruction": INSTRUCTION,
        "dimensions": 768,
        "selected_checkpoint": corrected["selected_checkpoint"],
        "selected_adapter_weights_sha256": hashlib.sha256(adapter_weights.read_bytes()).hexdigest(),
        "base_validation_index_sha256": base_hash,
        "tuned_validation_index_sha256": tuned_hash,
        "sealed_test_questions_sha256": hashlib.sha256(
            (EXP / "benchmark/questions.jsonl").read_bytes()
        ).hexdigest(),
    }
    frozen = write_frozen(OUT / "frozen_qwen3_configs.json", frozen)

    test_questions = EXP / "benchmark/questions.jsonl"
    test_bm25 = EXP / "test/bm25.sqlite3"
    run(common(
        test_questions, test_bm25, MODEL_ROOT / "base_indexes/test_v4",
        MODEL_ROOT / "base_query_embeddings/test_v4.jsonl", BASE_LABEL, BASE_LABEL,
        OUT / "test_qwen3_untuned_hybrid.jsonl", "qwen3_untuned_hybrid_top5",
    ) + config_args(frozen["untuned"]) + [
        "--dense-output", str(OUT / "test_qwen3_untuned_dense.jsonl"),
        "--dense-setup", "qwen3_untuned_dense_top5",
    ])
    run(common(
        test_questions, test_bm25, ADAPTER_RUN / "indexes/test_v4",
        ADAPTER_RUN / "query_embeddings/test_v4.jsonl", TUNED_LABEL, TUNED_LABEL,
        OUT / "test_qwen3_tuned_hybrid.jsonl", "qwen3_tuned_hybrid_top5",
    ) + config_args(frozen["tuned"]))

    for split in ("train", "validation"):
        # Keep a larger ranked pool for unanswerable SFT records. The SFT
        # builder still selects only five safe passages after filtering gold
        # documents and answer-bearing text.
        run(common(
            ROOT / f"data/finetuning/{split}/question_sources.jsonl",
            ROOT / f"data/finetuning/{split}/bm25.sqlite3",
            ADAPTER_RUN / f"indexes/{split}",
            ADAPTER_RUN / f"query_embeddings/{split}.jsonl",
            TUNED_LABEL, TUNED_LABEL,
            OUT / f"{split}_qwen3_tuned_candidates.jsonl",
            "qwen3_tuned_hybrid_top5",
        ) + config_args(frozen["tuned"]) + ["--top-k", "100"])
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()

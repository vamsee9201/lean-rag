from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import hashlib
import json
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "cloud"))

from build_bm25 import chunks_for_page
from build_raft_sft import clip_text
from extract_corpus import select_stratified
from prepare_finetune_split import split_documents
from retrieve_benchmark import metrics as retrieval_metrics
from hybrid_retrieval import DenseIndex, normalize, reciprocal_rank_fusion
from retriever_training_utils import resolve_positives, valid_negative
from run_qwen_embedding_local_judge import parse_scores
from qwen3_adapter_metadata import sanitize_adapter
from verify_qwen_embedding_matrix import MODELS as QWEN_MATRIX_MODELS, SETUPS as QWEN_MATRIX_SETUPS, verify_matrix
from run_qwen_embedding_retrieval import write_frozen
from train_qwen35_qwen_embedding import verify_reload as verify_qwen_reload
from qwen3_index_shards import chunk_order_sha256, file_sha256, reusable_shard, shard_directory
from gemini_vertex import estimated_cost_usd
from verify_qwen_embedding_artifacts import verify_disjoint_documents
from summarize_qwen_embedding_experiment import keyed_records
from score_answers import score as score_answer


class PipelineTests(unittest.TestCase):
    def test_fourth_experiment_exact_citations_require_correct_pair_and_evidence(self):
        question = {"reference_answer": "2001", "gold_documents": ["DOC-A"],
                    "gold_pages": [2], "category": "numeric_date", "answerable": True}
        answer = {"model": "m", "setup": "s", "question_id": "q",
                  "answer": "The year is 2001. DOC-A appears here and p.2 appears there."}
        context = [{"document_id": "DOC-A", "page_start": 2, "page_end": 2}]
        self.assertEqual(score_answer(answer, question)["citation_recall"], 1.0)
        self.assertEqual(score_answer(answer, question, exact=True, context=context)["citation_recall"], 0.0)
        answer["answer"] = "The year is 2001 [DOC-A p.2] [DOC-A p.9]."
        result = score_answer(answer, question, exact=True, context=context)
        self.assertEqual(result["citation_recall"], 1.0)
        self.assertEqual(result["citation_validity"], 0.5)

    def test_fourth_experiment_summary_rejects_duplicate_and_invalid_judge_ratings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.jsonl"
            row = {"model": "a", "setup": "b", "question_id": "c",
                   "answer_score": 2, "citation_score": 1, "unsupported_claim": 0}
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "unique ratings"):
                keyed_records(path, "qwen_judge", expected_count=2)
            row["answer_score"] = 3
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "invalid judge rating"):
                keyed_records(path, "qwen_judge", expected_count=1)

    def test_qwen_embedding_index_splits_reject_shared_documents(self):
        disjoint = {"train": {"a"}, "validation": {"b"}, "test_v4": {"c"}}
        verify_disjoint_documents(disjoint)
        with self.assertRaisesRegex(ValueError, "Documents shared between train and test_v4"):
            verify_disjoint_documents({**disjoint, "test_v4": {"a"}})

    def test_gemini_cost_estimate_includes_billable_output_and_reasoning(self):
        usage = {"input_tokens": 1_000_000, "output_tokens": 100_000,
                 "reasoning_tokens": 20_000}
        self.assertAlmostEqual(estimated_cost_usd(usage, "global"), 1.20)
        self.assertAlmostEqual(estimated_cost_usd(usage, "us-central1"), 1.32)

    def test_qwen_embedding_shard_resume_rejects_reorder_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunks = [{"chunk_id": "a"}, {"chunk_id": "b"}]
            shard = shard_directory(root, "train", 0)
            shard.mkdir(parents=True)
            vectors = np.zeros((2, 768), dtype=np.float32)
            vectors[:, 0] = 1
            np.save(shard / "embeddings.npy", vectors)
            (shard / "chunk_ids.json").write_text(json.dumps(["a", "b"]) + "\n")
            manifest = {
                "split": "train", "shard_number": 0, "chunks": 2,
                "dimensions": 768, "chunk_order_sha256": chunk_order_sha256(chunks),
                "embeddings_sha256": file_sha256(shard / "embeddings.npy"),
            }
            (shard / "manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(reusable_shard(shard, chunks, "train", 0).shape, (2, 768))
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                reusable_shard(shard, list(reversed(chunks)), "train", 0)
            vectors[0, 0] = 0
            np.save(shard / "embeddings.npy", vectors)
            with self.assertRaisesRegex(RuntimeError, "invalid vectors"):
                reusable_shard(shard, chunks, "train", 0)

    def test_qwen_adapter_reload_requires_matching_unique_validation_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            messages = [
                {"role": "system", "content": "Use evidence"},
                {"role": "user", "content": "What year?"},
                {"role": "assistant", "content": "2025"},
            ]
            path.write_text(json.dumps({"messages": messages}) + "\n")
            output = {"messages": messages[:2], "response": "2025 [DOC p.1]"}
            verify_qwen_reload([output], path)
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                verify_qwen_reload([output, output], path)
            with self.assertRaisesRegex(RuntimeError, "empty"):
                verify_qwen_reload([{**output, "response": ""}], path)

    def test_frozen_qwen_retrieval_settings_cannot_change_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.json"
            first = write_frozen(path, {"candidate_depth": 50, "dense_weight": 1.0})
            self.assertEqual(first, write_frozen(path, {"candidate_depth": 50, "dense_weight": 1.0}))
            with self.assertRaisesRegex(RuntimeError, "separately versioned run"):
                write_frozen(path, {"candidate_depth": 20, "dense_weight": 1.0})
            self.assertEqual(json.loads(path.read_text()), first)

    def test_qwen_matrix_verifies_frozen_user_and_system_prompts(self):
        system = "Only use supplied evidence."
        system_hash = hashlib.sha256(system.encode()).hexdigest()
        prompt_hash = hashlib.sha256(b"evidence and question").hexdigest()
        manifest = {
            "system_prompt": system,
            "setups": {
                setup: {"evidence_hashes": {f"q{i}": prompt_hash for i in range(50)}}
                for setup in QWEN_MATRIX_SETUPS
            },
        }
        rows = [
            {"model": model, "setup": setup, "question_id": f"q{i}",
             "status": "ok", "prompt_sha256": prompt_hash,
             "system_prompt_sha256": system_hash}
            for model in QWEN_MATRIX_MODELS
            for setup in QWEN_MATRIX_SETUPS
            for i in range(50)
        ]
        self.assertEqual(verify_matrix(rows, manifest)["answers"], 1000)
        rows[0]["system_prompt_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "system prompt"):
            verify_matrix(rows, manifest)
        rows[0]["system_prompt_sha256"] = system_hash
        rows[0]["prompt_sha256"] = "wrong"
        with self.assertRaisesRegex(ValueError, "question or evidence"):
            verify_matrix(rows, manifest)

    def test_swift_adapter_metadata_is_portable_without_changing_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            adapter = Path(directory)
            (adapter / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": "/root/.cache/huggingface/snapshots/abc",
                "r": 8,
            }))
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            sanitize_adapter(adapter, "Qwen/Qwen3-Embedding-8B")
            config = json.loads((adapter / "adapter_config.json").read_text())
            self.assertEqual(config["base_model_name_or_path"], "Qwen/Qwen3-Embedding-8B")
            self.assertEqual(config["r"], 8)
            self.assertIn("base_model: Qwen/Qwen3-Embedding-8B", (adapter / "README.md").read_text())
            self.assertEqual((adapter / "adapter_model.safetensors").read_bytes(), b"weights")
            sanitize_adapter(
                adapter, "Qwen/Qwen3.5-9B", title="GovInfo Qwen RAG adapter",
                description="Generator adapter.", tags=("lora", "text-generation"),
            )
            card = (adapter / "README.md").read_text()
            self.assertIn("base_model: Qwen/Qwen3.5-9B", card)
            self.assertIn("# GovInfo Qwen RAG adapter", card)
            self.assertIn("  - text-generation", card)
            self.assertNotIn("text-embeddings", card)
            self.assertEqual((adapter / "adapter_model.safetensors").read_bytes(), b"weights")

    def test_blind_local_judge_requires_complete_unique_valid_ratings(self):
        valid = json.dumps([
            {"id": "a01", "answer_score": 2, "citation_score": 1, "unsupported_claim": 0},
            {"id": "a02", "answer_score": 0, "citation_score": 0, "unsupported_claim": 1},
        ])
        self.assertEqual(len(parse_scores(valid, {"a01", "a02"})), 2)
        with self.assertRaises(ValueError):
            parse_scores(valid.replace('"a02"', '"a01"'), {"a01", "a02"})
        with self.assertRaises(ValueError):
            parse_scores(valid.replace('"unsupported_claim": 1', '"unsupported_claim": 2'), {"a01", "a02"})

    def test_retriever_supervision_resolves_gold_and_filters_unsafe_negatives(self):
        question = {
            "question_id": "q1", "reference_answer": "17 countries",
            "gold_documents": ["gold"], "gold_pages": [3],
            "gold_passages": ["robots raised productivity across 17 countries"],
        }
        positive = {"chunk_id": "gold:p3:c0", "text": "The robots raised productivity across 17 countries."}
        positives = resolve_positives(question, {("gold", 3): [positive]})
        self.assertEqual(positives, ["gold:p3:c0"])
        self.assertFalse(valid_negative(positive, question, set(positives)))
        self.assertFalse(valid_negative({"chunk_id": "x", "text": "The study covered 17 countries."}, question, set(positives)))
        self.assertTrue(valid_negative({"chunk_id": "y", "text": "The report covers unrelated fiscal policy."}, question, set(positives)))

    def test_stratified_selection_is_prefix_stable(self):
        rows = []
        for collection in ("A", "B", "C"):
            for number in range(10):
                rows.append({"collection": collection, "pages": str(number + 1), "package_id": f"{collection}{number}"})
        first = [row["package_id"] for row in select_stratified(rows, 10)]
        longer = [row["package_id"] for row in select_stratified(rows, 20)]
        self.assertEqual(first, longer[:10])

    def test_chunk_overlap_and_traceability(self):
        page = {
            "document_id": "doc", "title": "Title", "collection": "C", "page": 7,
            "text": " ".join(f"w{i}" for i in range(10)),
        }
        chunks = list(chunks_for_page(page, chunk_words=6, overlap_words=2))
        self.assertEqual([chunk["chunk_id"] for chunk in chunks], ["doc:p7:c0", "doc:p7:c1"])
        self.assertEqual(chunks[0]["text"].split()[-2:], chunks[1]["text"].split()[:2])

    def test_retrieval_metrics_credit_a_gold_page_once(self):
        question = {
            "answerable": True,
            "gold_documents": ["doc"],
            "gold_pages": [3],
        }
        items = [
            {"document_id": "doc", "page_start": 3, "page_end": 3},
            {"document_id": "doc", "page_start": 3, "page_end": 3},
        ]
        metrics = retrieval_metrics(items, question)
        self.assertEqual(metrics["gold_passage_recall"], 1.0)
        self.assertLessEqual(metrics["ndcg"], 1.0)

    def test_finetune_split_is_exact_disjoint_and_deterministic(self):
        documents = []
        for collection in ("A", "B", "C"):
            for number in range(10):
                documents.append(
                    {
                        "document_id": f"{collection}{number}",
                        "collection": collection,
                        "status": "ok" if number else "no_searchable_text",
                        "character_count": 100 if number else 0,
                        "sha256": str(number),
                    }
                )
        first = split_documents(documents, train=21, validation=3, test=6, seed=7)
        second = split_documents(list(reversed(documents)), train=21, validation=3, test=6, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(
            Counter(row["split"] for row in first),
            {"train": 21, "validation": 3, "test": 6},
        )
        self.assertEqual(len({row["document_id"] for row in first}), 30)

    def test_context_clipping_keeps_gold_quote(self):
        text = "a" * 2000 + " exact supported fact " + "b" * 2000
        clipped = clip_text(text, 500, "exact supported fact")
        self.assertLessEqual(len(clipped), 500)
        self.assertIn("exact supported fact", clipped)

    def test_embedding_normalization_rejects_invalid_vectors(self):
        vector = normalize(np.array([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)
        with self.assertRaises(ValueError):
            normalize(np.array([0.0, 0.0], dtype=np.float32))
        with self.assertRaises(ValueError):
            normalize(np.array([1.0, np.nan], dtype=np.float32))

    def test_dense_search_has_deterministic_chunk_id_ties(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunk_ids.json").write_text(json.dumps(["b", "a", "c"]))
            np.save(root / "embeddings.npy", np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32))
            (root / "manifest.json").write_text(json.dumps({
                "complete": True, "dimensions": 2, "model": "test",
            }))
            result = DenseIndex(root).search(np.array([1, 0], dtype=np.float32), 2)
            self.assertEqual([item[0] for item in result], ["a", "b"])

            many = DenseIndex(root).search_many(
                [np.array([1, 0], dtype=np.float32), np.array([0, 1], dtype=np.float32)], 2,
                batch_size=1,
            )
            self.assertEqual([item[0] for item in many[0]], ["a", "b"])
            self.assertEqual([item[0] for item in many[1]], ["c", "a"])

    def test_rrf_deduplicates_and_applies_document_cap(self):
        bm25 = [
            {"chunk_id": "a", "document_id": "d1", "title": "t", "collection": "c",
             "page_start": 1, "page_end": 1, "text": "a", "score": -2.0},
            {"chunk_id": "b", "document_id": "d1", "title": "t", "collection": "c",
             "page_start": 2, "page_end": 2, "text": "b", "score": -1.0},
        ]
        chunks = {
            "c": {"chunk_id": "c", "document_id": "d2", "title": "u", "collection": "c",
                  "page_start": 3, "page_end": 3, "text": "c"},
        }
        result = reciprocal_rank_fusion(
            bm25, [("a", 0.9), ("c", 0.8)], top_k=2, document_cap=1, chunks=chunks,
        )
        self.assertEqual([row["chunk_id"] for row in result], ["a", "c"])
        self.assertEqual(result[0]["bm25_rank"], 1)
        self.assertEqual(result[0]["dense_rank"], 1)
        self.assertEqual(result[0]["source"], "hybrid_rrf")

    def test_rrf_weight_and_ties_are_deterministic(self):
        def chunk(chunk_id):
            return {"chunk_id": chunk_id, "document_id": chunk_id, "title": chunk_id,
                    "collection": "c", "page_start": 1, "page_end": 1,
                    "text": chunk_id, "score": -1.0}
        bm25 = [chunk("a"), chunk("b")]
        chunks = {"a": chunk("a"), "b": chunk("b")}
        equal = reciprocal_rank_fusion(
            bm25, [("b", 0.9), ("a", 0.8)], top_k=2, chunks=chunks,
        )
        dense_weighted = reciprocal_rank_fusion(
            bm25, [("b", 0.9), ("a", 0.8)], top_k=2, dense_weight=2.0, chunks=chunks,
        )
        self.assertEqual([row["chunk_id"] for row in equal], ["a", "b"])
        self.assertEqual(dense_weighted[0]["chunk_id"], "b")


if __name__ == "__main__":
    unittest.main()

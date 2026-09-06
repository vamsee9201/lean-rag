from __future__ import annotations

import sys
from pathlib import Path
from collections import Counter
import json
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_bm25 import chunks_for_page
from build_raft_sft import clip_text
from extract_corpus import select_stratified
from prepare_finetune_split import split_documents
from retrieve_benchmark import metrics as retrieval_metrics
from hybrid_retrieval import DenseIndex, normalize, reciprocal_rank_fusion


class PipelineTests(unittest.TestCase):
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

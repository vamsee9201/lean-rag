# Vertex hybrid RAG experiment

This experiment preserves the completed BM25 result and adds a split-isolated
Vertex dense retriever. The dense component is `gemini-embedding-001` with
`RETRIEVAL_DOCUMENT`/`RETRIEVAL_QUERY`, 768 stored dimensions, strict input
truncation checks, normalized vectors, and exact dot-product search. BM25 and
dense candidates are combined with reciprocal-rank fusion.

Credentials remain local in `ai-lab-fasa.json`; neither credentials nor the
ignored `data/` artifacts are committed or uploaded to model repositories.

## Reproduction stages

Build each dense index through Vertex batch prediction:

```bash
.venv/bin/python scripts/build_vertex_dense_batch.py --split train --cleanup-gcs --cleanup-raw
.venv/bin/python scripts/build_vertex_dense_batch.py --split validation --cleanup-gcs --cleanup-raw
.venv/bin/python scripts/build_vertex_dense_batch.py --split test --cleanup-gcs --cleanup-raw
```

The September 6, 2026 Vertex batch endpoint returned 3,072-dimensional parent
vectors even when `output_dimensionality=768` was supplied. The builder records
that response shape and stores the normalized first 768 Matryoshka dimensions;
an online comparison produced cosine similarity 1.0000001 with a native online
768-dimensional response. Batch also reports statistics instead of enforcing
strict truncation. Every response above 2,048 tokens is therefore shortened
deterministically and re-embedded online with `autoTruncate=false`. Compact raw
statistics, request-to-chunk mappings, chunk-order hashes, and vector hashes are
retained. Vertex exposes no immutable managed model revision, which remains a
reproducibility limitation.

Tune fusion only on validation questions and freeze the winner:

```bash
.venv/bin/python scripts/retrieve_hybrid.py \
  --questions data/finetuning/validation/question_sources.jsonl \
  --bm25-index data/finetuning/validation/bm25.sqlite3 \
  --dense-index data/hybrid_experiment/indexes/validation \
  --output data/hybrid_experiment/retrieval/validation_top5.jsonl \
  --top-k 5 --tune-validation
```

Use the selected depth, dense weight, and document cap to retrieve the training
and test splits. The selected configuration is depth 50, equal BM25 and dense
weights, `k=60`, and no per-document cap. Training materialization retains the
full fused union of up to 100 candidates so it can select the first five safe
non-gold distractors for unanswerable examples; answerable and final inference
contexts contain exactly five passages.

Generate and seal the new benchmark before model inference:

```bash
.venv/bin/python scripts/generate_finetune_questions_gemini.py \
  --split test --pages data/finetuning/test/pages.jsonl \
  --output data/hybrid_experiment/benchmark/question_sources.jsonl \
  --exclude-documents-from data/finetuning/test/question_sources.jsonl \
  --direct 20 --numeric-date 10 --multi-passage 15 --cross-document 5 \
  --seed 20260907
.venv/bin/python scripts/seal_hybrid_benchmark.py \
  --sources data/hybrid_experiment/benchmark/question_sources.jsonl \
  --output data/hybrid_experiment/benchmark/questions.jsonl
```

Materialize the matched hybrid SFT data:

```bash
.venv/bin/python scripts/build_hybrid_sft.py --split train \
  --retrieval data/hybrid_experiment/retrieval/train_candidates.jsonl --force
.venv/bin/python scripts/build_hybrid_sft.py --split validation \
  --retrieval data/hybrid_experiment/retrieval/validation_candidates.jsonl --force
.venv/bin/python scripts/audit_sft_tokens.py \
  data/hybrid_experiment/sft/train/sft.jsonl \
  data/hybrid_experiment/sft/validation/sft.jsonl --limit 4096
```

The smoke-test entry point is `cloud/train_qwen35_hybrid_smoke.py`; it must pass
adapter reload inference before `cloud/train_qwen35_hybrid.py` is launched. The matrix
inference entry point is `cloud/infer_qwen35_matrix.py`. Both expect their
ignored local datasets to be staged in the private Hugging Face dataset repo and
receive the Hugging Face token only through the job secret mechanism.

After downloading the matrix outputs, normalize and combine them with the two
Gemini cells, then run deterministic scoring and both blinded judges. Use
`scripts/summarize_hybrid_matrix.py` for the predefined 10,000-resample paired
bootstrap and five-point non-inferiority analysis.

The final reporting stage is reproducible with:

```bash
.venv/bin/python scripts/score_answers.py \
  --questions data/hybrid_experiment/benchmark/questions.jsonl \
  --answers data/hybrid_experiment/runs/all_answers.jsonl \
  --output data/hybrid_experiment/runs/automatic_scores.jsonl
.venv/bin/python scripts/summarize_hybrid_matrix.py \
  --automatic data/hybrid_experiment/runs/automatic_scores.jsonl \
  --gemini-judge data/hybrid_experiment/runs/gemini_judge_scores.jsonl \
  --qwen-judge data/hybrid_experiment/runs/qwen_judge_scores.jsonl \
  --output data/hybrid_experiment/results/statistical_summary.json
.venv/bin/python scripts/export_hybrid_tables.py \
  --summary data/hybrid_experiment/results/statistical_summary.json \
  --costs data/hybrid_experiment/results/costs.json \
  --output-dir data/hybrid_experiment/results
```

## Frozen final matrix

| Generator state | BM25 top five | Vertex hybrid top five |
|---|---:|---:|
| Gemini 3.8 Flash | 50 | 50 |
| Base Qwen3.5 9B | 50 | 50 |
| BM25-trained Qwen adapter | 50 | 50 |
| Vertex-hybrid-trained Qwen adapter | 50 | 50 |

The new sealed benchmark contains 20 direct, 10 numeric/date, 15
multi-passage, and five cross-document questions. All generators within a
retrieval column receive byte-identical evidence.

## Completed run

All eight matrix cells completed with exactly 50 answers, for 400 answer
records. Both blind judges produced exactly 400 valid ratings. The
hybrid-trained adapter selected checkpoint 564 after one epoch; its validation
loss was 0.06356709 and its reload test produced 20 of 20 responses.

The primary dual-judge scores under Vertex-hybrid retrieval were 82.0% for
Gemini, 79.5% for the BM25-trained Qwen adapter, 77.5% for the hybrid-trained
adapter, and 77.0% for base Qwen. The hybrid-trained adapter minus Gemini
difference was -4.5 points, with a paired 95% interval from -10 to 0 points and
a one-sided lower bound of -9 points. It did not meet the predefined five-point
non-inferiority criterion. The BM25-trained adapter transferred successfully and
outscored the hybrid-trained adapter by two points under hybrid retrieval.

The complete interpretation and tables are in `FINDINGS.md`. Machine-readable
tables are under `data/hybrid_experiment/results/`, and the three blinded human
review sheets are under `data/hybrid_experiment/human_review/`.

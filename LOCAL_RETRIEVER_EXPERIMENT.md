# Fine-tuned local RAG experiment

This third experiment compares a fully local hybrid RAG stack with the existing
Vertex hybrid and Gemini cloud stack. It uses a new document-level held-out
test corpus, a domain-tuned GTE retriever, and a fresh Qwen3.5 9B LoRA adapter.

## Frozen design

- Train documents: the existing 315 searchable training documents
- Validation documents: the existing 45 searchable validation documents
- Test documents: 90 newly selected searchable PDFs excluded from the original
  500-document corpus by both package ID and PDF SHA-256
- Benchmark: 50 sealed questions, consisting of 20 direct, 10 numeric or date,
  15 multi-passage, and five cross-document questions
- Local dense base: `Alibaba-NLP/gte-modernbert-base`
- Local dense fine-tune: full-parameter BF16, three epochs, effective batch 64,
  `CachedMultipleNegativesRankingLoss`, 1,024-token encoder limit
- Generator base: `Qwen/Qwen3.5-9B`
- Generator fine-tune: independent rank-8 LoRA from the original base
- Final matrix: four retrievers by five generators, 50 answers per cell
- Statistical analysis: 10,000 paired bootstrap samples and a predefined
  five-point non-inferiority margin

All ignored data artifacts live under `data/local_retriever_experiment/`.
Credentials remain outside tracked files and are passed to cloud jobs as
secrets.

Before encoding, the pipeline audits source token lengths and records every
chunk that exceeds the frozen 1,024-token encoder limit. This makes the model's
truncation explicit and reproducible, including token-heavy tables and OCR text.

## Reproduction stages

### Prepare the new test corpus and benchmark

```bash
.venv/bin/python scripts/prepare_local_test_split.py
.venv/bin/python scripts/build_bm25.py \
  --pages data/local_retriever_experiment/test/pages.jsonl \
  --output data/local_retriever_experiment/test/bm25.sqlite3
.venv/bin/python scripts/generate_finetune_questions_gemini.py \
  --split test \
  --pages data/local_retriever_experiment/test/pages.jsonl \
  --output data/local_retriever_experiment/benchmark/question_sources.jsonl \
  --direct 20 --numeric-date 10 --multi-passage 15 --cross-document 5 \
  --seed 20260907
.venv/bin/python scripts/seal_local_benchmark.py
```

### Export and train the local embedder

```bash
.venv/bin/python scripts/export_embedding_training_data.py
hf upload-large-folder vamsee9201/lean-rag-training-data \
  data/local_retriever_experiment/embedding_training --repo-type dataset
```

Launch `cloud/train_gte_govinfo.py` on one Hugging Face L4 with the private
dataset mounted at `/mnt/data/local_retriever/embedding_training`. The job mines
four safe hard negatives per query, trains for three epochs, selects the best
validation checkpoint, builds the split-isolated indexes, and publishes
`vamsee9201/gte-modernbert-govinfo-retriever`.

The primary mining pool is the planned BM25, Vertex, and untuned-GTE top 50.
Two near-duplicate shipment-notice questions had no safe negative in that pool
because every candidate repeated the gold answer. For those cases only, the
pipeline searches BM25 ranks 51 through 100. It records the selected chunk IDs,
source ranks, and fallback flag in `hard_negative_audit.jsonl`.

### Build and select local retrieval

Download `selected_model`, `base_indexes`, and `indexes` from the embedding
model repository. Tune the local hybrid only on validation:

```bash
.venv/bin/python scripts/retrieve_local_hybrid.py \
  --questions data/finetuning/validation/question_sources.jsonl \
  --bm25-index data/finetuning/validation/bm25.sqlite3 \
  --dense-index data/local_retriever_experiment/indexes/tuned/validation \
  --model data/local_retriever_experiment/models/gte/selected_model \
  --model-label vamsee9201/gte-modernbert-govinfo-retriever \
  --output data/local_retriever_experiment/retrieval/validation_tuned_local.jsonl \
  --setup tuned_local_hybrid_top5 --tune-validation
```

Freeze the winning depth, dense weight, RRF constant, and document cap. Apply
that configuration to training, validation, and the sealed test benchmark.
Use the untuned GTE indexes with the same frozen fusion settings for the
untuned-local control.

### Materialize and train the fresh Qwen adapter

```bash
.venv/bin/python scripts/build_hybrid_sft.py --split train \
  --retrieval data/local_retriever_experiment/retrieval/train_tuned_candidates.jsonl \
  --output data/local_retriever_experiment/sft/train/sft.jsonl \
  --metadata data/local_retriever_experiment/sft/train/sft_metadata.jsonl
.venv/bin/python scripts/build_hybrid_sft.py --split validation \
  --retrieval data/local_retriever_experiment/retrieval/validation_tuned_candidates.jsonl \
  --output data/local_retriever_experiment/sft/validation/sft.jsonl \
  --metadata data/local_retriever_experiment/sft/validation/sft_metadata.jsonl
.venv/bin/python scripts/audit_sft_tokens.py \
  data/local_retriever_experiment/sft/train/sft.jsonl \
  data/local_retriever_experiment/sft/validation/sft.jsonl --limit 4096
```

Upload the two SFT files to the private dataset under `sft/`. Hugging Face Jobs
mounts the dataset root at `/mnt/data/local_retriever`, so training reads them
from `/mnt/data/local_retriever/sft/`.
Run `cloud/train_qwen35_local_hybrid_smoke.py`, require 20 valid reload
responses, and then run `cloud/train_qwen35_local_hybrid.py` on one L40S. The
full job publishes `vamsee9201/qwen35-9b-local-hybrid-rag-lora`.

### Run the frozen 4 by 5 matrix

Combine the four retrieval files and create byte-aligned inference inputs:

```bash
.venv/bin/python scripts/combine_retrieval_setups.py \
  --questions data/local_retriever_experiment/benchmark/questions.jsonl \
  --inputs data/local_retriever_experiment/retrieval/test_bm25.jsonl \
           data/local_retriever_experiment/retrieval/test_vertex_hybrid.jsonl \
           data/local_retriever_experiment/retrieval/test_untuned_local.jsonl \
           data/local_retriever_experiment/retrieval/test_tuned_local.jsonl \
  --setups bm25_top5 vertex_hybrid_top5 untuned_local_hybrid_top5 tuned_local_hybrid_top5 \
  --output data/local_retriever_experiment/retrieval/test_all.jsonl
.venv/bin/python scripts/build_qwen_matrix_inputs.py \
  --questions data/local_retriever_experiment/benchmark/questions.jsonl \
  --retrieval data/local_retriever_experiment/retrieval/test_all.jsonl \
  --output-dir data/local_retriever_experiment/inference_inputs
```

Generate the four Gemini cells with `scripts/run_gemini_experiment.py`. Run the
16 Qwen cells with `cloud/infer_qwen35_local_matrix.py`. Normalize and combine
the outputs with `scripts/normalize_qwen_local_matrix.py` and
`scripts/combine_answers.py`.

### Judge, analyze, and review

```bash
.venv/bin/python scripts/score_answers.py \
  --questions data/local_retriever_experiment/benchmark/questions.jsonl \
  --answers data/local_retriever_experiment/runs/all_answers.jsonl \
  --output data/local_retriever_experiment/runs/automatic_scores.jsonl
.venv/bin/python scripts/judge_answers_gemini.py \
  --questions data/local_retriever_experiment/benchmark/questions.jsonl \
  --answers data/local_retriever_experiment/runs/all_answers.jsonl \
  --output data/local_retriever_experiment/runs/gemini_judge_scores.jsonl \
  --usage-output data/local_retriever_experiment/runs/gemini_judge_usage.jsonl
.venv/bin/python scripts/summarize_local_retriever_experiment.py \
  --automatic data/local_retriever_experiment/runs/automatic_scores.jsonl \
  --gemini-judge data/local_retriever_experiment/runs/gemini_judge_scores.jsonl \
  --qwen-judge data/local_retriever_experiment/runs/qwen_judge_scores.jsonl \
  --output data/local_retriever_experiment/results/statistical_summary.json
```

Use `scripts/build_local_human_review.py` to create the three predefined paired
review sheets. Automated results remain identified as model-judged until human
review is completed.

## Completed run

The sealed benchmark contains 50 questions from 90 newly selected searchable
documents. Its question SHA-256 is
`ac854e24fba0e2b45c5eff703f9702cd5b0858ecf3677220febe0a1a51b4f34b`.
Its test document-list SHA-256 is
`e261610fa18f5fb4698a9247d7f49521ea1975252934b09296622246504f907e`.

Both training jobs completed. The selected embedding checkpoint was checkpoint
120. The selected Qwen checkpoint was checkpoint 500, which passed a 20 of 20
adapter reload test. The final matrix contains exactly 1,000 unique answers,
50 in each of 20 cells, with no inference errors. Evidence hashes match across
all five generators within every retrieval condition.

Both blinded judges produced exactly 1,000 valid ratings. The Qwen judge used
groups of five candidates per prompt because grouped 20-candidate output could
not be constrained under the Transformers backend. Seventeen malformed fields
were rerun as isolated one-candidate repairs and then passed the strict
normalizer. No rating was imputed.

### Final retrieval metrics

| Retriever | All-gold recall@5 | Mean passage recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 68% | 76% | 0.688 | 0.678 |
| Vertex hybrid | **78%** | **84%** | **0.688** | **0.709** |
| Untuned local hybrid | 72% | 79% | 0.652 | 0.658 |
| Tuned local hybrid | 62% | 72% | 0.628 | 0.615 |

The domain-tuned GTE model regressed. Tuned local dense recall was 30%,
compared with 58% for untuned GTE. This result is preserved rather than hidden.

### Final answer metrics

| Generator | BM25 | Vertex hybrid | Untuned local hybrid | Tuned local hybrid |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 79.5% | 79.0% | 75.0% | 68.0% |
| Base Qwen3.5 9B | 73.5% | 79.5% | 74.5% | 65.5% |
| BM25-trained Qwen | 73.0% | 80.0% | 72.0% | 69.5% |
| Vertex-hybrid-trained Qwen | 72.5% | 80.0% | 71.5% | 69.5% |
| Local-hybrid-trained Qwen | 73.0% | **81.0%** | 74.5% | 68.0% |

The new Qwen adapter and Gemini tied at 68.0% under identical tuned local
contexts. Their paired 95% interval was -9 to +8 points. The one-sided lower
bound was -7.5 points, so the predefined five-point non-inferiority criterion
was not met. The complete local stack trailed the Vertex plus Gemini cloud
stack by eleven points, with a paired 95% interval from -25 to +2.5 points.

The controlled generator result is stronger than the complete-stack result:
the new Qwen adapter scored 81.0% with Vertex hybrid passages, the highest point
estimate in the matrix, while Gemini scored 79.0% with byte-identical evidence.
The fully local untuned-GTE hybrid plus new Qwen reached 74.5%. These results
locate the remaining local-system opportunity in retrieval rather than answer
generation. Future local experiments should freeze Qwen and improve the GTE
training recipe.

Presentation-ready tables are under
`data/local_retriever_experiment/results/`. The three blank human-review files
are under `data/local_retriever_experiment/human_review/`.

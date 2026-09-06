# Local BM25 RAG pilot with a hosted Gemini baseline

## Decision

The sub-10B local models are viable for this corpus, but they do not fully match
the hosted baseline. Qwen3.5 9B is the most accurate local model; Qwen3.5 4B is
the better local efficiency choice. Gemini 3.8 Flash produced the strongest
overall results, much better citations, and the lowest observed latency.

Across the Qwen and Gemini blind judges, the combined answer scores were 76.5%
for Gemini, 71.5% for Qwen 9B, and 69.5% for Qwen 4B. Gemini's paired advantage
over 9B was 5 points with a 95% bootstrap interval of -0.5 to 11.5 points, so
those two are statistically tied on this 50-question pilot. Gemini's advantage
over 4B was 7 points with an interval of 0.5 to 14 points.

## Frozen experiment

- Corpus: 500 GovInfo PDFs across 12 collections and 58,071 pages.
- Searchable corpus: 450 PDFs; 50 scan-only PDFs were retained but not OCRed.
- Index: SQLite FTS5/BM25, 103,986 chunks, 350 words with 50-word overlap.
- Retrieval: the same frozen top-five chunks for every model.
- Benchmark: 50 questions per model, including 5 unanswerable questions.
- Generation: reasoning/thinking off, temperature 0, and a 220-token output cap.
- Total: 250 successful answers, 250 Qwen judge ratings, and 250 Gemini judge
  ratings.

## Overall model results

| Model | Dual-judge score (95% CI) | Deterministic accuracy | Exact gold citation recall | Mean latency |
|---|---:|---:|---:|---:|
| Qwen3 1.7B GGUF Q6_K | 62.0% (49.0–74.0%) | 68.6% | 2.2% | 9.9 s |
| Qwen3.5 2B MLX 4-bit | 64.0% (51.5–76.5%) | 71.4% | 0.0% | 9.5 s |
| Qwen3.5 4B MLX 4-bit | 69.5% (57.5–80.5%) | 74.3% | 24.4% | 18.5 s |
| Qwen3.5 9B MLX 4-bit | 71.5% (60.0–82.0%) | 80.0% | 17.8% | 38.4 s |
| **Gemini 3.8 Flash** | **76.5% (66.0–86.0%)** | **82.9%** | **68.9%** | **1.3 s** |

Deterministic accuracy covers the 35 direct, numeric/date, and unanswerable
questions that have a reliable exact-match or abstention check. The dual-judge
score averages the normalized 0/1/2 scores from the two independent judges.
Citation recall requires the exact gold document ID and page and covers the 45
answerable questions.

## Dual-judge score by category

| Model | Direct (20) | Numeric/date (10) | Multi-passage (10) | Cross-document (5) | Unanswerable (5) |
|---|---:|---:|---:|---:|---:|
| Qwen3 1.7B | 80.0% | 62.5% | 45.0% | 5.0% | 80.0% |
| Qwen3.5 2B | 70.0% | 72.5% | 65.0% | 5.0% | 80.0% |
| Qwen3.5 4B | 81.3% | 82.5% | 47.5% | 10.0% | 100% |
| Qwen3.5 9B | 83.8% | 82.5% | 50.0% | 15.0% | 100% |
| **Gemini 3.8 Flash** | **91.3%** | 75.0% | **65.0%** | **20.0%** | **100%** |

## Judge agreement

Qwen3.5 9B and Gemini 3.8 Flash agreed exactly on 82.8% of the 250 answer
ratings, were within one rubric point on 97.6%, and had a Pearson correlation of
0.85. Their binary citation ratings agreed on 83.6% of answerable cases.

The manual audit of the six two-point disagreements found two recurring rubric
issues. Qwen sometimes folded malformed citations into answer correctness even
though citations have a separate score. The judges also differed on whether an
answerable benchmark question should receive credit for abstaining when BM25 did
not retrieve its gold evidence. Both raw judge tracks are retained; no ratings
were manually overwritten.

## Retrieval findings

Overall BM25 top-five retrieval found all gold passages for 68.9% of answerable
questions. Mean gold-passage recall was 74.4%, MRR was 68.1%, and nDCG was
66.6%.

| Category | Mean gold recall | Questions with all gold evidence retrieved |
|---|---:|---:|
| Direct | 85% | 85% |
| Numeric/date | 80% | 80% |
| Multi-passage | 75% | 60% |
| Cross-document | 20% | 0% |

Cross-document performance is primarily a retrieval limitation: BM25 never
placed all required evidence in the top five for any of those five questions.
Changing the generator alone will not solve that category.

## Hosted cost

The 50 Gemini answers used 120,507 input tokens and 3,338 output tokens, costing
about $0.10 at the current introductory global rate. Gemini judging used 30,337
input tokens and 13,263 output tokens, costing about $0.07. Including the small
access checks, the hosted portion cost approximately $0.18.

Gemini 3.8 Flash pricing used for this estimate is $0.75 per million input tokens
and $3.75 per million output tokens through December 31, 2026.

## Interpretation

Use Qwen3.5 4B when local execution, privacy, offline operation, or predictable
zero marginal API cost matters most. Use Qwen3.5 9B when local answer quality is
the priority and its latency is acceptable. Use Gemini 3.8 Flash when hosted
execution is acceptable: it is the strongest end-to-end choice in this pilot,
especially for reliable citations.

The next experiment should improve retrieval for multi-source questions while
freezing the generators. BM25-only options include query expansion, per-document
result diversification, and a larger candidate pool followed by deterministic
selection. A person should audit the 50 questions and a stratified answer sample
before publication.

## Reproducible artifacts

- `data/benchmark/questions.jsonl`: fixed benchmark.
- `data/retrieval/results.jsonl`: frozen BM25 evidence.
- `data/runs/answers.jsonl`: 250 successful answers.
- `data/runs/automatic_scores.jsonl`: deterministic metrics.
- `data/runs/judge_scores.jsonl`: Qwen blind ratings.
- `data/runs/gemini_judge_scores.jsonl`: Gemini blind ratings.
- `data/runs/final_summary_qwen_judge.json`: Qwen-judge aggregates.
- `data/runs/final_summary_gemini_judge.json`: Gemini-judge aggregates.
- `data/runs/dual_judge_summary.json`: agreement and combined paired intervals.

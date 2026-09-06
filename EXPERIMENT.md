# Experiment protocol

## Research question

How does answer quality change between local generators from 1.7B to 9B
parameters when corpus, questions, BM25 retrieval, evidence, and prompts are
fixed, and how close are they to a hosted Gemini 3.8 Flash baseline?

The primary comparison isolates generator choice. Direct, numeric,
multi-passage, cross-document, and abstention questions are reported separately.

## Frozen inputs

- Corpus: 500 deterministically selected GovInfo PDFs, 58,071 pages.
- Searchable coverage: 450 documents; 50 scan-only documents are retained as
  measured extraction failures.
- Chunking: 350 words with 50-word overlap.
- Questions: one fixed set of 50 questions, including 5 unanswerable cases.
- Generator prompt: fixed for the BM25 top-five setup.
- Generation: temperature 0, reasoning off, 16,384-token context. Each retrieved
  chunk is capped at 6,000 characters to bound pathological table tokenization.

The benchmark candidates were generated with Qwen3.5 9B and machine-curated
against exact source quotes. This can favor that model family through question
style. Results must be labeled exploratory until a person audits the questions
and performs a blind sample of answer judgments.

## Models

| Label | LM Studio model | Format |
|---|---|---|
| 1.7B | `qwen/qwen3-1.7b` | GGUF Q6_K |
| 2B | `qwen/qwen3.5-2b` | MLX 4-bit |
| 4B | `qwen/qwen3.5-4b` | MLX 4-bit |
| 9B | `qwen/qwen3.5-9b` | MLX 4-bit |
| Hosted | `google/gemini-3.8-flash` | Vertex AI |

The 1.7B model is from Qwen3 while the other models are Qwen3.5. This means the
ladder tests practical local models, not a clean parameter-only scaling law.

## Retrieval setup

SQLite FTS5/BM25 returns the top five 350-word chunks. Retrieval is computed and
saved before generation so all generators receive byte-identical evidence.

## Runs

The expanded matrix contains 5 models × 50 questions = 250 answers. The scripts
append each result immediately and skip completed cells on restart.

## Metrics

Retrieval:

- all-gold recall per question;
- mean gold-passage recall;
- MRR;
- nDCG.

Generation:

- exact normalized reference hit for direct/numeric questions;
- token F1;
- unanswerable abstention accuracy;
- answerable false-abstention rate;
- citation document/page recall;
- blinded 0/1/2 scores from Qwen3.5 9B and Gemini 3.8 Flash;
- elapsed time and output throughput.

Report macro averages by question category as well as the overall mean. Include
bootstrap confidence intervals over questions and paired differences between
models because each cell uses the same question set.

## Interpretation limits

- Fifty questions support an exploratory paired comparison, not a broad claim
  about all RAG workloads.
- Scan-only PDFs lower corpus coverage; this experiment measures the no-OCR
  pipeline that actually ran.
- One local model family generated the candidates and provides the automated
  judge, so a human audit is required for a credible external result.
- Quantization, architecture generation, and runtime format differ across the
  model ladder.

# Qwen3 Embedding Experiment

[Reproduction stages](QWEN_EMBEDDING_REPRODUCTION.md)

## Goal

This fourth experiment tests whether a stronger local retriever closes the gap
between the fully local RAG system and the Vertex based reference system. It
uses `Qwen/Qwen3-Embedding-8B` for dense retrieval and keeps exact vectors in
local NumPy files. It does not use a managed vector database.

The embedding adapter is trained and the one-time corpus vectors are computed
on rented Hugging Face GPU hardware. After verification, the vector files are
downloaded for local exact search. This measures a portable, self-hostable
model stack; it does not by itself prove that live query embedding and answer
generation run together offline on the project's 18 GB MacBook Pro. That
deployment claim requires a separate local inference test.

The main comparison is:

* Local: BM25 plus Qwen3 embeddings, followed by a freshly trained Qwen3.5 9B
  RAG adapter.
* Cloud reference: BM25 plus Vertex embeddings, followed by Gemini 3.8 Flash.

## Frozen protocol

The experiment uses 90 previously unused searchable GovInfo PDFs and 50 new
sealed questions. The question mix remains 20 direct, 10 numeric or date, 15
multi-passage, and 5 cross-document. No test document, passage, or question may
enter embedding training or generator training.

The Qwen embedding baseline and fine-tuned model both produce normalized
768-dimensional vectors. Training inputs use an audited 8,192-token ceiling,
and corpus encoding uses the model's 32,768-token context window. Inputs are
audited before encoding, and the run fails rather than silently truncating an
overlong passage. The separate ceilings preserve the few verified long
training passages and the existing OCR-heavy chunk corpus without making every
training step use the longest index context. Queries use this fixed instruction:

> Given a search query about United States government publications, retrieve
> passages that contain the evidence needed to answer the query.

Documents receive no instruction. Exact dot product ranks dense results. BM25
and dense rankings are fused with reciprocal-rank fusion. Candidate depth,
dense weight, RRF constant, and per-document cap are selected on the existing
validation split and frozen before the sealed test is retrieved.

Corpus encoding sorts chunks by token length within fixed 4,096-chunk shards,
then uses recorded length-aware batch sizes. Repeating an identical BF16 batch
produced identical vectors in the preflight check. Encoding those same texts
individually gave cosine similarity from 0.99960 to 0.99977, so batch shape is
part of the recorded indexing procedure. Each completed shard stores its chunk
order and vector checksum and is verified before reuse.

## Embedding training

The training set contains the existing 2,560 verified query-positive pairs and
up to four filtered hard negatives per pair. The records use the official SWIFT
InfoNCE format. Fine-tuning starts from the untouched Qwen3-Embedding-8B base
with LoRA. A reload smoke test runs before the full job. Checkpoints are
selected by validation all-gold recall@5, then mean passage recall@5, nDCG@5,
MRR, and the earlier checkpoint when tied.

The base model is evaluated before training. This makes the effect of embedding
fine-tuning directly measurable and prevents a strong pretrained model from
being credited to the new training recipe.

For checkpoint selection, all-gold recall@5 requires the exact accepted
positive chunk IDs. On the existing validation split, the untuned Qwen3 model
scored 68 percent by that definition. Checkpoint 80 scored 69 percent and was
selected; checkpoint 160 scored 65 percent. The downstream RAG retrieval table
instead credits the exact gold document and page, so its percentages are not
directly interchangeable with these checkpoint metrics.

## Sealed-test retrieval results

The trained checkpoint and fusion settings were fixed before retrieving the 50
new test questions. The 90 test PDFs were excluded from embedding and generator
training. These are retrieval results only; answer quality is evaluated in the
separate 20-combination matrix.

| Retriever | All-gold recall@5 | Mean gold-passage recall@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 70% | 81% | 0.803 | 0.764 |
| Vertex hybrid | 76% | 84% | 0.802 | 0.773 |
| Untuned Qwen3 dense | 62% | 71% | 0.662 | 0.632 |
| Untuned Qwen3 hybrid | 66% | 74% | 0.735 | 0.691 |
| Fine-tuned Qwen3 dense | 82% | 88% | 0.830 | 0.809 |
| Fine-tuned Qwen3 hybrid | 82% | 89% | 0.872 | 0.841 |

The fine-tuned Qwen3 retriever improved on its untuned version on this test.
Dense-only and hybrid have the same observed all-gold recall, while fusion
improves ranking and mean gold-passage recall. The sample contains only five
cross-document questions, so their category result is especially uncertain.
The paired tuned-minus-untuned hybrid all-gold recall difference is 16
percentage points, with a 10,000-resample 95 percent bootstrap interval of
6 to 26 points. Run `python scripts/report_qwen_embedding_retrieval.py` to
rebuild the machine-readable report from the frozen retrieval files.

## Generator training

The existing 2,256 training and 111 validation question records are rebuilt
with the selected Qwen hybrid contexts. The same answer supervision,
unanswerable rate, prompt, citation format, token limits, and seed are retained.
A fresh LoRA adapter starts from the untouched `Qwen/Qwen3.5-9B` base model.
It does not continue from any earlier adapter.

## Twenty matched combinations

Five retrievers are paired with four generators, which creates 20 combinations
and 1,000 answers. The retrievers are BM25, Vertex hybrid, untuned Qwen3 dense,
untuned Qwen3 hybrid, and fine-tuned Qwen3 hybrid. The generators are Gemini
3.8 Flash, base Qwen3.5 9B, the previous best local Qwen adapter, and the new
Qwen-embedding-context adapter.

Every generator receives byte-identical questions, evidence, evidence order,
system prompts, and citation labels within a retriever. Generation uses
temperature 0, thinking disabled, 220 maximum new tokens, and an 8,192-token
input limit.

One unusually long untuned-dense prompt exceeded the inference limit. Its
passages were clipped by a fixed character policy without consulting the gold
answer. All generators use that same frozen clipped prompt. The original five
passages for that question contained no exact gold quotation.

## Evaluation

Retrieval is measured with all-gold recall@5, mean gold-passage recall@5, MRR,
nDCG@5, cross-document all-gold recall, latency, index size, and indexing cost.
Reported search-and-fusion latency uses precomputed query vectors and excludes
query encoding. Query encoding time and its compute setting are reported
separately, since the Qwen and Vertex encoders run on different hardware.

Answers are measured with reference matching, token F1, exact citation recall,
citation validity, unsupported-claim rate, abstention behavior, and two blinded
judges. Candidate identities and system identities remain hidden from both
judges. Statistical comparisons use 10,000 paired bootstrap samples over whole
questions, two-sided 95 percent confidence intervals, a five-point
non-inferiority margin, and Holm correction for secondary comparisons.

For this run, automatic scoring uses `score_answers.py --exact-citations` with
the frozen retrieval matrix supplied through `--retrieval`. A gold citation
counts only when the answer contains its exact `[DOCUMENT_ID p.PAGE]` pair.
Citation validity checks each cited pair against the five passages actually
shown to that generator. The older experiments retain their original scoring
files and are not silently rescored under this stricter policy.

All 50 sealed questions have supported answers. The benchmark can measure
incorrect abstention on answerable questions, but it cannot estimate correct
abstention on unanswerable questions. The report marks that measure as not
available rather than inferring it from training data.

## Complete answer results

All 1,000 answers passed the 20-cell, 50-question matrix check. Within each
retriever, the four generators received identical frozen prompts and evidence.
Both blinded judges supplied all 1,000 answer and citation ratings, for 2,000
ratings of each primary field. The primary score is the mean of the judges'
answer scores after normalizing their 0-to-2 scale to 0-to-1. Their weighted
agreement on answer scores was 0.899.

| Generator | BM25 | Vertex hybrid | Untuned Qwen dense | Untuned Qwen hybrid | Tuned Qwen hybrid |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemini 3.8 Flash | 74.5% | 78.5% | 62.5% | 68.0% | **80.5%** |
| Base Qwen3.5 9B | 68.5% | 67.0% | 58.0% | 65.0% | 73.5% |
| Prior local-hybrid Qwen adapter | 73.0% | 75.5% | 61.0% | 67.0% | 79.0% |
| Fresh Qwen-embedding-context adapter | 74.5% | 70.0% | 63.5% | 68.5% | **78.0%** |

The fully local stack, tuned Qwen3 hybrid retrieval plus the fresh Qwen
adapter, scored 78.0%. The Vertex hybrid plus Gemini reference scored 78.5%.
Their paired difference was -0.5 percentage points, with a two-sided 95%
bootstrap interval from -10.5 to +10.0 points. The point estimates are close,
but this interval is too wide to establish equivalence or superiority.

With the same tuned-Qwen evidence, the fresh Qwen generator scored 78.0% and
Gemini scored 80.5%. The paired difference was -2.5 points, with a 95%
interval from -11.5 to +6.5 points. Its one-sided 95% lower bound was -10.0
points, below the predefined -5-point non-inferiority margin. Non-inferiority
was therefore **not established**. The fresh Qwen adapter gained 4.5 observed
points over base Qwen on that evidence, but its interval of -1.0 to +10.5
points includes zero. It scored one point below the previous local-hybrid
adapter on the same evidence, with a -6.0 to +3.5 point interval. The new
generator fine-tune worked and its adapter reload passed 20 of 20 responses,
but this benchmark does not establish an advantage over the prior adapter.

Keeping Gemini fixed shows the value of the trained local retriever: Gemini
scored 80.5% with tuned Qwen hybrid evidence versus 68.0% with untuned Qwen
hybrid evidence, a paired gain of 12.5 points (95% interval +2.0 to +23.5).
This is a secondary comparison and did not remain significant after Holm
correction. The primary retrieval measure separately rose from 66% to 82%
all-gold recall@5, with a paired 95% interval for the 16-point gain of +6 to
+26 points. The retrieval result supports a real improvement on this test;
the downstream answer gain is encouraging but less certain.

Exact citation recall was 83% for the complete local stack and 54% for the
Vertex plus Gemini reference. Under identical tuned-Qwen evidence, Gemini's
exact citation recall was 56%. The fresh Qwen adapter's 83% citation recall
is measured against exact document-and-page pairs, not merely a plausible
citation string. These values describe citation behavior on this benchmark;
they do not turn the inconclusive answer-quality differences into a
superiority claim.

| Question category | Questions | Fully local score | Vertex plus Gemini score |
| --- | ---: | ---: | ---: |
| Direct | 20 | 82.5% | 90.0% |
| Numeric or date | 10 | 92.5% | 90.0% |
| Multi-passage | 15 | 65.0% | 68.3% |
| Cross-document | 5 | 70.0% | 40.0% |

The cross-document category has only five questions, so its difference is
especially unstable. All 50 questions are answerable. Incorrect abstention can
be measured, but correct abstention on unanswerable questions cannot.

The untuned Qwen judge omitted the auxiliary `unsupported_claim` flag for 14
ratings. All 2,000 answer and citation ratings are present; unsupported-claim
rates use the 1,986 available labels, and each cell reports its denominator in
`data/qwen_embedding_experiment/results/summary.json`. Missing flags were
preserved as missing rather than inferred. The Qwen judge ran with full BF16
weights in this experiment; the earlier local MLX judge used 4-bit weights.
The prepared human-review sheets have not been scored by people. Findings here
are automated, blinded-model judgments.

## Cost and artifacts

Hugging Face GPU jobs cost $25.75651 according to the authenticated billing
record. Gemini answer generation is estimated at $0.62770 and Gemini judging
at $0.25203 from recorded token usage. The total is approximately **$26.64**,
including failed and repeated development jobs, below the approved $28 hard
cap. It is a one-time experimental cost, not a steady-state per-query serving
price. Query encoding, model hosting, idle capacity, and the retriever's
one-time index build matter in a deployment comparison.

The selected Qwen3 embedding adapter is published at
[`vamsee9201/qwen3-embedding-8b-govinfo-retriever`](https://huggingface.co/vamsee9201/qwen3-embedding-8b-govinfo-retriever).
The selected generator LoRA is at
[`vamsee9201/qwen35-9b-qwen-embedding-rag-lora`](https://huggingface.co/vamsee9201/qwen35-9b-qwen-embedding-rag-lora),
with its verified local checkpoint under
`data/qwen_embedding_experiment/models/qwen35_full/run1/selected_checkpoint/checkpoint-564/`.
The paired calculations, all cell-level measures, and category breakdowns are
in `data/qwen_embedding_experiment/results/summary.json`. The selected
generator adapter starts from the original Qwen3.5-9B base, not from the
previous local-hybrid adapter. All prior experiments and their artifacts
remain unchanged.

## Primary sources

The recipe follows the official
[Qwen3-Embedding-8B model card](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
and the official
[Qwen SWIFT embedding training guide](https://github.com/QwenLM/Qwen3-Embedding/blob/main/docs/training/SWIFT.md).

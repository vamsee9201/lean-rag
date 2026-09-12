# Lean RAG

Lean RAG tests whether a fully local RAG system can match a cloud RAG system on
questions about public GovInfo documents. Instead of comparing two fixed
pipelines once, the experiment tests every generator with every retriever. This
separates the quality of the language model from the quality of the evidence it
receives.

The completed experiment contains **20 combinations**. Each combination pairs
one retrieval system with one answer generator. Every generator was tested with
every retriever, and every combination answered the same 50 sealed questions.
Within a retrieval setup, the generators received the exact same passages in
the same order.

The 20 combinations produced 1,000 answers. This structure lets us determine
whether a result came from retrieval, answer generation, fine-tuning, or the
complete system. Two blinded model judges rated every answer, producing 2,000
ratings. The analysis uses 10,000 paired bootstrap samples, and human review
sheets are included for independent verification.

[Complete findings](FINDINGS.md) | [Third experiment reproduction guide](LOCAL_RETRIEVER_EXPERIMENT.md) | [Qwen adapter](https://huggingface.co/vamsee9201/qwen35-9b-local-hybrid-rag-lora) | [GTE retriever](https://huggingface.co/vamsee9201/gte-modernbert-govinfo-retriever)

## Main result

The local Qwen generator succeeded. With retrieval held constant, the fresh
Qwen adapter produced the highest observed score in the experiment: **81.0%**,
compared with **79.0%** for Gemini 3.8 Flash over the same Vertex hybrid
evidence. Qwen also achieved higher exact citation recall, 79% versus 74%. The
sample is too small to claim statistical superiority, but the result shows
that a locally deployable 9B generator can match the answer quality of the
hosted model on this workload.

The remaining local-system problem is retrieval. The fine-tuned local embedding
model did not improve retrieval. Its hybrid
all-gold recall fell to **62%**, compared with **72%** for the untuned local
hybrid and **78%** for Vertex hybrid. This retrieval loss reduced the complete
local stack to **68.0%**. The generator comparison confirms the diagnosis:
Qwen and Gemini both scored 68.0% when they received those same tuned-local
passages.

| Generator | BM25 | Vertex hybrid | Untuned local hybrid | Tuned local hybrid |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 79.5% | 79.0% | 75.0% | 68.0% |
| Base Qwen3.5 9B | 73.5% | 79.5% | 74.5% | 65.5% |
| BM25-trained Qwen | 73.0% | 80.0% | 72.0% | 69.5% |
| Vertex-hybrid-trained Qwen | 72.5% | 80.0% | 71.5% | 69.5% |
| **New local-hybrid-trained Qwen** | 73.0% | **81.0%** | **74.5%** | 68.0% |

The primary score is the normalized mean of two independently blinded judges,
Gemini 3.8 Flash and untuned Qwen3.5 9B. Every cell contains the same 50 sealed
questions. Within a retrieval column, every generator received byte-identical
questions, evidence, evidence order, and system prompts.

## What the statistics support

Under identical tuned-local evidence, the new Qwen adapter and Gemini both
scored **68.0%**. Their paired difference was zero, with a two-sided 95%
bootstrap interval from **-9 to +8 points**. The one-sided lower bound was
**-7.5 points**, so Qwen did not meet the predefined five-point non-inferiority
criterion.

The complete local stack scored eleven points below the Vertex plus Gemini
reference stack. Its paired 95% interval was **-25 to +2.5 points**, and the
Holm-adjusted result was not statistically significant. This does not weaken
the local-generator result. It identifies the next engineering target: improve
the local embedding training recipe while keeping the successful Qwen adapter.

The new Qwen adapter improved by 2.5 points over base Qwen with identical tuned
local contexts. Its interval was **-3 to +7.5 points**. It also remained within
1.5 points of both earlier adapters under those contexts. The experiment
therefore shows a capable new adapter, but no statistically established
fine-tuning advantage among the Qwen generator states.

## Retrieval result

| Retriever | All-gold recall@5 | Mean passage recall@5 | MRR | nDCG@5 | Cross-document recall |
|---|---:|---:|---:|---:|---:|
| BM25 | 68% | 76% | **0.688** | 0.678 | 0% |
| Vertex dense | 68% | 73% | 0.534 | 0.570 | **60%** |
| **Vertex hybrid** | **78%** | **84%** | **0.688** | **0.709** | **60%** |
| Untuned local dense | 58% | 68% | 0.546 | 0.557 | 40% |
| Untuned local hybrid | 72% | 79% | 0.652 | 0.658 | 20% |
| Tuned local dense | 30% | 40% | 0.371 | 0.337 | 0% |
| Tuned local hybrid | 62% | 72% | 0.628 | 0.615 | 0% |

The validation-selected embedding checkpoint also trailed its base model before
the sealed test was opened. Fine-tuning reduced validation all-gold recall@5
from 53% to 44%. This is a useful negative result: domain data and hard
negatives do not guarantee a better retriever. The loss, sampling strategy, and
positive-passage construction need further study before this tuned retriever is
used in production.

The untuned local hybrid already provides a viable local baseline. Paired with
the new Qwen adapter, it scored **74.5%** with no Vertex embeddings or Gemini
generation at runtime. The path forward is to recover and exceed that retrieval
baseline through better local embedding supervision.

## How Qwen became competitive

1. Public PDFs were extracted by page and split by document, preventing a
   document from crossing train, validation, and test.
2. Text was divided into 350-word passages with 50-word overlap and traceable
   document and page identifiers.
3. Qwen training used 2,256 verified RAG examples, including answerable and
   unanswerable cases, grounded answers, and exact citation syntax.
4. The fresh adapter started from `Qwen/Qwen3.5-9B`, used rank-8 LoRA, and
   trained for one epoch. It did not continue from an earlier adapter.
5. The final benchmark used 90 previously unused searchable PDFs and 50 sealed
   questions. No test document or passage entered either training process.
6. Every generator was tested with each frozen retriever, which exposed the
   difference between a strong generator and a weak complete stack.

The selected adapter weights are an 83 MB LoRA delta. They are stored locally
at
`data/local_retriever_experiment/models/qwen_local_adapter/run1/v0-20260908-110116/checkpoint-500/adapter_model.safetensors`
and publicly in the linked Hugging Face repository. The original
`Qwen/Qwen3.5-9B` base weights are loaded separately.

## Cost

The third experiment cost approximately **$8.3**, including failed and repeated
training and judge jobs. It stayed below the $15 hard budget.

| Component | Estimated cost |
|---|---:|
| GTE training on Hugging Face L4, including failed runs | $1.81 |
| Qwen training, inference, and judging on Hugging Face L40S | $5.04 |
| Vertex embeddings for 4.95 million document tokens | $0.59 |
| Gemini benchmark generation | $0.08 |
| 200 Gemini answers | $0.50 |
| 1,000 Gemini judge ratings | $0.25 |
| **Estimated total** | **$8.27** |

Hugging Face Jobs billed the L4 at $0.80 per hour and the L40S at $1.80 per
hour. Gemini 3.8 Flash used the September 2026 introductory global rates of
$0.75 per million input tokens and $3.75 per million output tokens. Values are
workload estimates from recorded runtimes and token counts, not invoices.

[Hugging Face Jobs pricing](https://huggingface.co/docs/hub/jobs-pricing) | [Google Gemini pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing) | [Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)

### Potential generation cost at scale

The 200 Gemini answers averaged about 3,026 input tokens and 58 output tokens.
At the September 2026 introductory global rate, that prompt mix costs about
**$2.49 per 1,000 answers**. Google's listed January 2027 rates double that
projection to about **$4.97 per 1,000 answers**.

The new Qwen adapter averaged 4.68 L40S seconds per answer with Vertex hybrid
evidence. At $1.80 per L40S hour and full utilization, that is about **$2.34 per
1,000 answers**. A rented L40S left running for 30 days costs about $1,296, so
idle time can overwhelm any per-answer saving.

| Monthly answers with this prompt mix | Gemini through 2026 | Gemini from 2027 | Qwen on a fully used rented L40S |
|---|---:|---:|---:|
| 1,000 | $2.49 | $4.97 | $2.34 |
| 100,000 | $249 | $497 | $234 |
| 1,000,000 | $2,487 | $4,974 | $2,340 |

These projections exclude retrieval, networking, storage, taxes, engineering
labor, and idle GPU time. On owned hardware, Qwen has no external per-token
fee, but electricity, hardware purchase, maintenance, and capacity still cost
money.

## Local-first recommendation

Use **untuned GTE plus BM25 hybrid retrieval with the new Qwen adapter** as the
current fully local system. Treat Vertex hybrid as an experimental retrieval
ceiling that demonstrates what the Qwen generator can do when supplied with
stronger passages. Continue improving the local embedder without retraining
Qwen unless a new evaluation reveals a generator-specific problem.

This gives the project a clear result and next step:

- local Qwen generation reached hosted-model quality;
- the fully local baseline is already useful at 74.5%;
- embedding fine-tuning is the remaining bottleneck; and
- better local retrieval can close the end-to-end gap without replacing the
  local generator.

## Why local models matter

A local model can answer when a network is slow, unavailable, expensive, or
prohibited. That includes flights, ships, field work, disaster response,
remote clinics, mines, rural sites, and secure facilities. Documents and
prompts can stay on controlled hardware, which helps with privacy, data
residency, and predictable model availability. Teams can also inspect the
weights, preserve an exact version, change the serving stack, and adapt the
model to a narrow workflow.

Local operation still requires enough memory and compute. Hosted Gemini was
faster in this experiment, and a rented GPU can cost more than an API when it
sits idle. The practical advantage depends on connectivity, privacy, request
volume, available hardware, and the quality of the local retriever.

## Why the experiment matters

Many model comparisons confound retrieval and generation. A model can appear
weak because it received poor passages, or appear strong because it received
better evidence. This project tests every generator over the same contexts and
also compares the complete local and cloud systems. The design reveals that
Qwen generation was competitive while the new embedding model was the failing
component. A single end-to-end score would have hidden that distinction.

The project also preserves an unsuccessful fine-tune. That result is useful to
people reproducing the work because it shows where additional training can
harm a RAG pipeline and why retrieval must be evaluated independently before
answer generation.

## Reproduce and inspect

- [Third experiment procedure](LOCAL_RETRIEVER_EXPERIMENT.md)
- [Second experiment procedure](HYBRID_EXPERIMENT.md)
- `data/local_retriever_experiment/results/answer_quality_2x4.csv`
- `data/local_retriever_experiment/results/retrieval_comparison.csv`
- `data/local_retriever_experiment/results/paired_comparisons.csv`
- `data/local_retriever_experiment/results/category_results.csv`
- `data/local_retriever_experiment/human_review/`

Human review sheets are prepared but not completed. Until they are completed,
the answer-quality findings should be described as automated, dual-model judge
results. The benchmark has 50 questions, so modest differences remain
uncertain even with paired evaluation.

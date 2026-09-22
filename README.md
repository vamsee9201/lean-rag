# Lean RAG

Lean RAG is a controlled study of local and cloud retrieval-augmented
generation over public GovInfo PDFs. The project asks two questions:

1. Can a locally deployable Qwen3.5 9B answer model approach Gemini 3.8 Flash?
2. Can retrieval also be made local without giving up the evidence quality of
   Vertex AI embeddings?

The experiments separate retrieval from generation. Models receive identical
questions and evidence when the generator is being compared. Retrievers are
measured independently before complete local and cloud systems are compared.

## Current result

The latest fully local stack uses **BM25 plus fine-tuned
Qwen3-Embedding-8B**, followed by a **fine-tuned Qwen3.5 9B** answer model. It
scored **78.0%**, compared with **78.5%** for BM25 plus Vertex embeddings and
Gemini 3.8 Flash on the same 50 sealed questions.

That makes the systems comparable in observed automated answer quality on this
benchmark. The paired 95% confidence interval for local minus cloud was
**-10.5 to +10.0 percentage points**, so the result does not prove statistical
equivalence. A larger test and completed human review are needed for that
claim.

The clearest improvement was retrieval. Fine-tuning Qwen3-Embedding-8B raised
local hybrid all-gold recall@5 from **66% to 82%**. Vertex hybrid reached
**76%** on the same sealed test. The complete local stack also achieved
**83% exact citation recall**, compared with **54%** for Vertex hybrid plus
Gemini.

[Complete findings](FINDINGS.md) | [Latest experiment](QWEN_EMBEDDING_EXPERIMENT.md) | [Reproduction guide](QWEN_EMBEDDING_REPRODUCTION.md)

## Experimental method

- Public GovInfo PDFs were extracted by page and split by document.
- Test documents never entered retriever or generator training.
- Passages contain traceable document and page identifiers.
- Questions and retrieval settings were sealed before final inference.
- Generators within a retrieval condition received the same evidence in the
  same order.
- Candidate identities were randomized for blinded judging.
- Paired bootstrap analysis resampled complete questions rather than treating
  individual scores as independent.
- Human review sheets were prepared, but have not been completed. Results
  described here are automated model-judge results.

Scores belong to their own benchmark and evaluation procedure. They should be
compared within an experiment, not treated as one combined leaderboard.

## Experiment 1: five-model BM25 pilot

The first experiment tested five generators over the same frozen BM25 top-five
passages. Each model answered 50 questions, producing 250 answers. Gemini and
Qwen3.5 9B independently judged randomized candidate identities.

| Model | Combined judge score | Deterministic accuracy | Exact citation recall | Mean latency |
|---|---:|---:|---:|---:|
| Qwen3 1.7B | 62.0% | 68.6% | 2.2% | 9.9 s |
| Qwen3.5 2B | 64.0% | 71.4% | 0.0% | 9.5 s |
| Qwen3.5 4B | 69.5% | 74.3% | 24.4% | 18.5 s |
| **Qwen3.5 9B** | **71.5%** | **80.0%** | 17.8% | 38.4 s |
| **Gemini 3.8 Flash** | **76.5%** | **82.9%** | **68.9%** | **1.3 s** |

Gemini's five-point observed advantage over Qwen3.5 9B had a paired 95%
interval from -0.5 to +11.5 points, so the pilot did not establish that Gemini
was more accurate than 9B. It did reveal two weaknesses: BM25 missed required
cross-document evidence, and untuned local models often formatted citations
poorly.

## Experiment 2: 250-question confirmatory BM25 comparison

The second evaluation focused on Gemini and Qwen3.5 9B over 250 paired
questions, including 25 unanswerable controls. Both received identical frozen
BM25 evidence and instructions.

| Category | Questions | Gemini | Qwen3.5 9B | Gemini minus Qwen |
|---|---:|---:|---:|---:|
| All questions | 250 | 53.4% | 50.6% | +2.8 points |
| Answerable only | 225 | 49.3% | 49.1% | +0.2 points |
| Direct | 100 | 63.8% | 66.0% | -2.3 points |
| Numeric or date | 50 | 52.5% | 50.5% | +2.0 points |
| Same-document multi-passage | 50 | 40.0% | 36.5% | +3.5 points |
| Cross-document | 25 | 4.0% | 4.0% | 0.0 points |
| Unanswerable | 25 | 90.0% | 64.0% | +26.0 points |

For answerable questions, the difference was only +0.2 points with a paired
95% interval from -3.7 to +3.9. Gemini's overall advantage came mainly from
better abstention on unanswerable questions. BM25 retrieved all required gold
passages for only 59.6% of answerable questions, confirming that retrieval was
the primary bottleneck.

Full report: [CONFIRMATORY_RESULTS.md](CONFIRMATORY_RESULTS.md)

## Experiment 3: fine-tuned Qwen over BM25

Qwen3.5 9B was fine-tuned from its original base using 2,256 verified RAG
examples. The selected LoRA checkpoint trained for one epoch and passed an
independent reload test. Fine-tuned Qwen and Gemini answered 50 new held-out
questions from the same BM25 passages.

| Metric | Fine-tuned Qwen3.5 9B | Gemini 3.8 Flash |
|---|---:|---:|
| Blind-judge answer score | **84%** | **85%** |
| Blind-judge citation accuracy | 76% | 76% |
| Exact citation recall | **82%** | 76% |
| Mean response time | 5.20 s | **1.32 s** |

The answer-score difference was -1 point, with a paired 95% interval from
-7 to +4 points. This experiment showed that a fine-tuned local 9B generator
could closely match Gemini when both received the same evidence. BM25 found all
required evidence for 42 of the 50 questions, which still limited both models.

## Experiment 4: BM25 versus Vertex hybrid retrieval

The next experiment combined BM25 with Vertex `gemini-embedding-001` rankings
using reciprocal-rank fusion. It used 50 new sealed questions and crossed two
retrievers with four generator states, producing 400 answers.

### Retrieval

| Retriever | All-gold recall@5 | Mean passage recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 70% | 77% | 0.710 | 0.697 |
| Vertex dense | **78%** | 83% | 0.740 | 0.727 |
| BM25 plus Vertex hybrid | **78%** | **84%** | **0.811** | **0.777** |

### Answer quality

| Generator | BM25 | Vertex hybrid |
|---|---:|---:|
| Gemini 3.8 Flash | 75.0% | **82.0%** |
| Base Qwen3.5 9B | 76.5% | 77.0% |
| BM25-trained Qwen adapter | 76.0% | **79.5%** |
| Vertex-hybrid-trained Qwen adapter | 75.5% | 77.5% |

Hybrid retrieval improved the point estimates, but the 50-question answer
intervals remained wide. The existing BM25-trained Qwen adapter transferred
well to hybrid evidence and outscored the newly trained hybrid adapter by two
points. Repeating the same answer-model fine-tune with different distractors
did not produce a clear benefit.

Procedure: [HYBRID_EXPERIMENT.md](HYBRID_EXPERIMENT.md)

## Experiment 5: fine-tuned GTE local retriever

This experiment trained `Alibaba-NLP/gte-modernbert-base`, combined it with
BM25, rebuilt the Qwen training contexts, and trained a fresh Qwen adapter. It
used 90 previously unused PDFs and 50 newly sealed questions. Four retrievers
crossed with five generator states, producing 1,000 answers.

### Retrieval

| Retriever | All-gold recall@5 | Mean passage recall@5 | nDCG@5 |
|---|---:|---:|---:|
| BM25 | 68% | 76% | 0.678 |
| Vertex hybrid | **78%** | **84%** | **0.709** |
| Untuned GTE hybrid | 72% | 79% | 0.658 |
| Fine-tuned GTE hybrid | 62% | 72% | 0.615 |

### Answer quality

| Generator | BM25 | Vertex hybrid | Untuned GTE hybrid | Tuned GTE hybrid |
|---|---:|---:|---:|---:|
| Gemini 3.8 Flash | 79.5% | 79.0% | 75.0% | 68.0% |
| Base Qwen3.5 9B | 73.5% | 79.5% | 74.5% | 65.5% |
| BM25-trained Qwen | 73.0% | 80.0% | 72.0% | 69.5% |
| Vertex-hybrid-trained Qwen | 72.5% | 80.0% | 71.5% | 69.5% |
| Local-hybrid-trained Qwen | 73.0% | **81.0%** | 74.5% | 68.0% |

The GTE fine-tune was a useful negative result. It reduced local hybrid
retrieval recall from 72% to 62%, so the complete tuned-local stack scored
68.0% versus 79.0% for Vertex plus Gemini. The generator itself was not the
weak component: local-hybrid-trained Qwen scored 81.0% versus Gemini's 79.0%
when both received the same Vertex hybrid evidence.

Procedure: [LOCAL_RETRIEVER_EXPERIMENT.md](LOCAL_RETRIEVER_EXPERIMENT.md)

## Experiment 6: fine-tuned Qwen3 local retriever

The final experiment replaced GTE with `Qwen/Qwen3-Embedding-8B`. The
embedding model was fine-tuned on verified query-positive pairs and filtered
hard negatives. A fresh Qwen3.5 9B answer adapter was trained from the original
base using contexts from the selected local retriever. Five retrievers crossed
with four generators on 50 new sealed questions, producing 1,000 answers.

### Retrieval

| Retriever | All-gold recall@5 | Mean passage recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| BM25 | 70% | 81% | 0.803 | 0.764 |
| Vertex hybrid | 76% | 84% | 0.802 | 0.773 |
| Untuned Qwen3 dense | 62% | 71% | 0.662 | 0.632 |
| Untuned Qwen3 hybrid | 66% | 74% | 0.735 | 0.691 |
| Fine-tuned Qwen3 dense | **82%** | 88% | 0.830 | 0.809 |
| **Fine-tuned Qwen3 hybrid** | **82%** | **89%** | **0.872** | **0.841** |

Fine-tuning improved local hybrid all-gold recall@5 by 16 points. The paired
95% interval was +6 to +26 points. Vectors are normalized and searched with
exact dot product in local NumPy files; BM25 and dense rankings are fused. No
managed vector database is used.

### Answer quality

| Generator | BM25 | Vertex hybrid | Untuned Qwen dense | Untuned Qwen hybrid | Tuned Qwen hybrid |
|---|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 74.5% | 78.5% | 62.5% | 68.0% | **80.5%** |
| Base Qwen3.5 9B | 68.5% | 67.0% | 58.0% | 65.0% | 73.5% |
| Previous local-hybrid Qwen adapter | 73.0% | 75.5% | 61.0% | 67.0% | 79.0% |
| Fresh Qwen-embedding-context adapter | 74.5% | 70.0% | 63.5% | 68.5% | **78.0%** |

The complete local stack scored 78.0%, versus 78.5% for Vertex hybrid plus
Gemini. The observed half-point difference is small, but the paired 95%
interval from -10.5 to +10.0 points is too wide to establish equivalence. With
identical tuned-local evidence, the fresh Qwen adapter scored 78.0% and Gemini
scored 80.5%. The new generator adapter worked, but it did not establish an
advantage over the previous adapter's 79.0% on the same evidence.

The complete local stack reached 83% exact document-and-page citation recall,
versus 54% for Vertex hybrid plus Gemini. Both blinded judges supplied all
2,000 answer and citation ratings. Human review sheets are prepared but have
not been scored.

The latest experiment cost approximately **$26.64**, including failed
and repeated development jobs, below its $28 cap. This is research cost, not a
steady-state cost per production answer.

[Complete experiment](QWEN_EMBEDDING_EXPERIMENT.md) | [Reproduction guide](QWEN_EMBEDDING_REPRODUCTION.md) | [Embedding adapter](https://huggingface.co/vamsee9201/qwen3-embedding-8b-govinfo-retriever) | [Answer adapter](https://huggingface.co/vamsee9201/qwen35-9b-qwen-embedding-rag-lora)

## What the experiments show

The local generator became competitive before the retrieval system did. The
BM25 fine-tune brought Qwen3.5 9B to 84% versus Gemini's 85% with identical
evidence. Vertex hybrid retrieval then showed that better passages helped both
local and hosted generators.

The two local embedding experiments produced opposite results. Fine-tuning GTE
hurt retrieval, while fine-tuning Qwen3-Embedding-8B improved it substantially.
Domain training is not automatically beneficial; every retriever must be
measured against its untuned base on a sealed benchmark.

The newest result closes most of the observed complete-system gap. The fully
local Qwen3 retriever plus Qwen generator scored 78.0%, while the cloud
reference scored 78.5%. This is strong evidence that a compact self-hostable
RAG stack can be useful on this workload, while the statistical interval and
unfinished human review prevent an equivalence claim.

## Why a local RAG system matters

Local RAG is necessary when policy, regulation, contractual obligations, or
the sensitivity of documents and prompts prevents data from leaving controlled
infrastructure. A hosted API can be excellent, but it cannot satisfy a strict
requirement that retrieval queries, retrieved passages, and generated answers
remain inside the operator's environment.

A local stack provides several practical advantages:

- **Privacy and data control.** Documents, retrieval queries, context passages,
  and answers can remain on hardware controlled by the organization.
- **Deployment control.** Teams choose the model version, update schedule,
  quantization, serving software, retention policy, and network boundary.
- **Offline operation.** The system can work on flights, ships, field sites,
  rural locations, secure facilities, and other places with unreliable or no
  connectivity.
- **Predictable availability.** A pinned local model does not disappear because
  an external provider changes an endpoint, quota, policy, or model alias.
- **Customization.** Both retrieval and generation can be adapted to a domain,
  citation format, abstention policy, and workflow.
- **Auditable retrieval.** Exact local indexes, chunk hashes, model revisions,
  and fusion settings can be preserved for reproduction and investigation.
- **No managed vector database requirement.** This project searches local
  NumPy vectors directly and fuses them with BM25.
- **Flexible economics.** At sufficient sustained utilization, owned or rented
  compute can avoid per-token and per-query API fees. At low utilization, a
  hosted API may still be cheaper because idle GPUs cost money.

Small local models have improved enough that deployment is no longer only a
privacy compromise. In these experiments, a fine-tuned 9B generator repeatedly
approached Gemini's observed answer quality, and the final local stack finished
within half a point of the cloud reference. Local systems can now offer useful
quality together with privacy, control, offline availability, and model
ownership.

The hardware requirement still matters. Training and final inference in the
latest experiment used rented GPUs. The project has not demonstrated that the
embedding model and answer model run together on the 18 GB MacBook Pro. A
production decision should measure latency, throughput, memory, power, idle
capacity, and total operating cost on the intended hardware.

## Reproduce and inspect

- [Complete findings](FINDINGS.md)
- [Confirmatory BM25 comparison](CONFIRMATORY_RESULTS.md)
- [Vertex hybrid experiment](HYBRID_EXPERIMENT.md)
- [Fine-tuned GTE experiment](LOCAL_RETRIEVER_EXPERIMENT.md)
- [Fine-tuned Qwen3 embedding experiment](QWEN_EMBEDDING_EXPERIMENT.md)
- [Latest reproduction guide](QWEN_EMBEDDING_REPRODUCTION.md)

Large PDFs, vector indexes, generated answers, and judge files live under the
ignored `data/` directory rather than in Git. Model adapters and their cards are
published through the links above. Credentials remain outside tracked files.

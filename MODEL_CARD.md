---
base_model: Qwen/Qwen3.5-9B
library_name: peft
license: apache-2.0
pipeline_tag: text-generation
tags:
  - rag
  - lora
  - qwen
  - govinfo
  - information-retrieval
---

# Qwen3.5 9B Vertex-hybrid RAG LoRA

This repository contains the Qwen3.5 9B LoRA adapter trained for the second Lean
RAG experiment. It also hosts the sealed benchmark, retrieval measurements,
answer-quality tables, paired statistical analysis, cost data, and blinded
human-review sheets.

## Intended use

The adapter was trained to answer questions from supplied document passages,
cite document and page identifiers, and abstain when the evidence is
insufficient. It was evaluated on public GovInfo documents.

This is a research adapter and reproducibility artifact. Validate it on your
own corpus before relying on it in a production or high-stakes system.

## Training

- Base model: `Qwen/Qwen3.5-9B`
- Method: BF16 LoRA
- Training records: 2,256
- Validation records: 111
- Epochs: 1
- Optimizer steps: 564
- LoRA rank: 8
- LoRA alpha: 32
- Targets: all linear layers
- Selected checkpoint: `run1/v0-20260907-054214/checkpoint-564`
- Best validation loss: 0.06356709
- Reload validation: 20 of 20 responses generated

## Evaluation result

| Generator | BM25 primary score | Vertex hybrid primary score | Hybrid exact citation recall |
|---|---:|---:|---:|
| Gemini 3.8 Flash | 75.0% | **82.0%** | 73% |
| Base Qwen3.5 9B | 76.5% | 77.0% | 57% |
| BM25-trained Qwen3.5 9B | 76.0% | **79.5%** | 73% |
| This hybrid-trained adapter | 75.5% | 77.5% | **74%** |

The hybrid-trained adapter completed successfully and produced the strongest
exact citation recall. It did not outperform the earlier BM25-trained adapter
on the primary answer score. The earlier adapter transferred to hybrid contexts
without additional training and remains the recommended local configuration.

The hybrid-trained adapter scored 4.5 points below Gemini under hybrid
retrieval. Its one-sided 95% lower confidence bound was -9 points, so it did not
meet the predefined five-point non-inferiority criterion.

## Download and use

Download the selected adapter checkpoint:

```sh
hf download vamsee9201/qwen35-9b-hybrid-rag-lora \
  --include 'run1/v0-20260907-054214/checkpoint-564/**' \
  --local-dir qwen35-hybrid-rag
```

Run it with `ms-swift`:

```sh
swift infer \
  --adapters qwen35-hybrid-rag/run1/v0-20260907-054214/checkpoint-564 \
  --use_hf true \
  --infer_backend transformers \
  --max_length 8192 \
  --max_new_tokens 220 \
  --temperature 0 \
  --enable_thinking false
```

The model expects a prompt containing a question and labeled evidence passages.
The complete prompt construction code is available in the
[Lean RAG repository](https://github.com/vamsee9201/lean-rag).

## Reproducibility artifacts

The `evaluation/vertex-hybrid/` directory contains:

- the 50-question sealed benchmark and checksum;
- train, validation, and test embedding index manifests;
- BM25, dense, and hybrid retrieval measurements;
- the complete 2 by 4 answer-quality table;
- category-level results and paired confidence intervals;
- measured experiment costs and generation latency;
- the full findings and reproduction guide; and
- blank blinded human-review sheets.

## Limitations

- The confirmatory benchmark contains 50 questions.
- Cross-document evaluation contains only five questions.
- The primary score uses two model judges. Human scoring is pending.
- The adapter was trained and evaluated on government documents and may not
  transfer equally to other domains.
- The tested Vertex hybrid retriever requires network access for online query
  embeddings.
- Vertex AI does not expose an immutable managed model revision for
  `gemini-embedding-001`.

## License

The Qwen3.5 9B base model is released under Apache 2.0. This adapter uses the
same license. Review the base model license and terms before deployment.

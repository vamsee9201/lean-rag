# Qwen3.5 9B RAG fine-tuning research and execution plan

Research checked on September 6, 2026.

## Conclusion

The full Qwen3.5 9B fine-tune should **not be scheduled on the 18 GB M3 Pro yet**.
The installed model is a 5.98 GB, 4-bit MLX SafeTensors checkpoint, which is the
right format for low-memory inference. However, the current Apple training stacks
have unresolved Qwen3.5 backward-pass problems:

- MLX-LM has an open report where the exact
  `mlx-community/Qwen3.5-9B-4bit` checkpoint fails on its first backward pass on
  a 36 GB M5 Max, even with batch size 1, four LoRA layers, a 2,048-token
  sequence, and gradient checkpointing.
- MLX-LM also has an open Qwen3.5 LoRA Metal-descriptor leak that crashes longer
  runs independently of ordinary memory exhaustion.
- MLX-VLM documents Qwen3.5 LoRA support, but it also has open Qwen3.5 issues for
  corrupted adapter generations and a first-step gradient-transform crash.

Those reports do not prove that every Qwen3.5 9B configuration fails, but they
make a multi-hour local run on an 18 GB machine too risky to treat as the main
plan. My earlier estimate that this would simply be an overnight MLX job was too
confident.

The reliable plan is to prepare and audit the data locally, run a tiny local
compatibility probe, and use a CUDA training machine for the real LoRA run unless
that probe demonstrates that the upstream problems have been fixed.

## Training without owning a GPU

The CUDA machine only needs to exist for the training and export job. Rent one,
upload the training JSONL and scripts, download the adapter and merged model, and
terminate the instance. The resulting 4-bit model then runs locally on the Mac;
it does not need the rental GPU for inference.

The recommended low-friction option is one Hugging Face Job on an L40S with 48 GB
of VRAM. As checked on September 6, 2026, Hugging Face lists that hardware at
$1.80/hour, bills Jobs by the minute while starting or running, and supports a
hard timeout. A 50-step pilot should run first. If the measured full job takes
3-8 hours, compute would cost approximately $5.40-$14.40; budget $20 because that
duration is an estimate until the pilot measures this exact model and dataset.

RunPod is the lower-cost, more manual alternative. Its current public Pod prices
list an A40 48 GB at $0.49/hour, RTX A6000 48 GB at $0.53/hour, and RTX 6000 Ada
48 GB at $0.84/hour. The same estimated 3-8 GPU hours would therefore be about
$1.47-$6.72 before storage or transfer charges. Availability can vary, and we
would be responsible for configuring, monitoring, exporting, and deleting the
Pod and its storage.

The existing Google Cloud project is also usable if it has GPU quota and credits,
but it is not automatically the cheapest or quickest route. Before creating any
billable resource, check available quota and the complete VM price in the target
region. A 24 GB L4 is a possible QLoRA test machine, but 48 GB is the initial
target for ordinary BF16 LoRA and a clean merge/export path. If the 48 GB pilot
runs out of memory, move the unchanged job to an A100 with 80 GB rather than
redesigning the experiment around a fragile configuration.

The cloud execution sequence is:

1. Finish, validate, and freeze the training and validation JSONL locally.
2. Package an `ms-swift` environment and a deterministic training command.
3. Launch a 48 GB GPU with an explicit timeout and run a 50-step smoke test.
4. Reload the adapter, generate the fixed validation sample, and estimate the
   full runtime from measured throughput.
5. Run one epoch only if the smoke test passes.
6. Download checkpoints, metrics, logs, and the merged model before terminating
   the job and deleting paid storage.
7. Convert the merged checkpoint to MLX 4-bit locally and run the frozen RAG
   evaluation against untuned Qwen and Gemini.

No PDFs or cloud service-account credential should be uploaded unless required.
The training job needs only the derived, audited text examples and model-download
credentials. Provisioning the paid GPU is the only step that must wait for an
explicit spending authorization; all dataset and job preparation can happen
locally first.

## Model facts

Qwen's official checkpoint is Apache 2.0 licensed. Although the checkpoint is
named 9B, its current Hugging Face metadata reports approximately 10 billion
parameters. It has 32 language-model layers and is a multimodal model with a hybrid
layout combining Gated DeltaNet and full-attention layers. Its native context
window is 262,144 tokens, but that does not mean long-context fine-tuning is
practical on this Mac.

The installed LM Studio checkpoint is MLX SafeTensors, 4-bit, and approximately
5.98 GB. MLX-LM can train QLoRA when pointed at a quantized model, supports chat
JSONL datasets, prompt masking, gradient accumulation, reduced LoRA layer counts,
and gradient checkpointing. Its own memory guidance uses a 32 GB M1 Max example;
our machine has 18 GB.

## Data design

The PDFs are source material, not direct training records. Split documents before
generating any examples:

- 350 documents for training
- 50 documents for validation and model selection
- 100 documents held out for the final test

Create approximately 2,000-3,000 chat examples from the 350 training documents.
Each example should contain the same system instruction used by the RAG service,
retrieved evidence with exact document/page labels, a question, and the ideal
assistant answer. The training mix should emphasize the weaknesses measured in
the benchmark:

- 35% direct and numeric extraction
- 25% same-document multi-passage synthesis
- 15% cross-document synthesis
- 15% answerable examples with hard distractor passages
- 10% genuinely unanswerable examples requiring the exact abstention response

All factual claims and citations must be derivable from the supplied context.
Gemini can draft synthesis answers, but deterministic validators should check
document IDs, pages, numbers, and answerability, followed by a human sample audit.
The current 250 confirmatory questions must not become training records.

Use a RAFT-style construction rather than treating raw PDF text as the target.
Most answerable records should contain one or two verified oracle passages mixed
with the other BM25 top-five results as hard distractors. A smaller, explicitly
labelled set should remove every oracle passage and use `Insufficient evidence.`
as the target. This teaches the generator to use the supplied evidence, ignore
irrelevant retrievals, cite the useful pages, and abstain when retrieval fails.
It does not attempt to memorize the 500 PDFs.

The train, validation, and final test indexes must contain only documents from
their own split. The existing 250-question confirmatory benchmark touches 189
documents, so it remains the untuned baseline diagnostic. The final tuned-model
claim will use a new 50-question test generated only from the 100 held-out
documents.

## Local compatibility gate

Use a separate virtual environment and the newest released MLX/MLX-LM or MLX-VLM
versions. Do not patch the project's working environment first. Test the exact
Qwen3.5 9B checkpoint with 8-16 short examples:

1. Run one forward-only validation batch.
2. Run 10 LoRA steps at batch size 1 and 512-1,024 tokens.
3. Train only four language-model layers with rank 4 or 8.
4. Keep the vision tower and aligner frozen.
5. Enable completion-only loss and gradient checkpointing.
6. Record peak wired memory, tokens per second, loss, and checkpoint reload.
7. Generate five answers with the adapter and compare them with the base model.

The gate passes only if backward propagation, checkpoint saving, adapter reload,
and generation all work without corrupted tokens. If it passes, repeat for 50
steps before setting the full cloud-run duration and cost. If it fails, stop; changing batch
size repeatedly is unlikely to solve the published architecture-specific errors.

## Reliable training route

Use ModelScope's `ms-swift` on a CUDA GPU. Its current supported-model table lists
Qwen3.5 9B, and its training documentation supports SFT, LoRA, QLoRA, validation
splits, gradient checkpointing, and text-only records for multimodal models.

For straightforward deployment back to the Mac, prefer ordinary BF16 LoRA on a
40-48 GB GPU rather than QLoRA on a smaller GPU. The ms-swift documentation warns
that its QLoRA result cannot be merged through its normal export path, while a
merged LoRA model can be converted to a fresh 4-bit MLX checkpoint.

Start with:

- language-model LoRA only; vision tower and aligner frozen
- rank 8, alpha 32, dropout 0
- all linear language-model targets
- batch size 1 and gradient accumulation 16
- maximum sequence length 4,096 for the first pilot
- BF16, gradient checkpointing, AdamW
- learning rate `1e-4`, 5% warmup, one epoch
- validation and checkpointing every 50-100 optimizer steps
- assistant/completion-only loss
- non-thinking prefix enabled and empty-thinking tokens excluded from loss
- no sequence packing
- Python 3.12, Transformers 5.9 or newer, flash-linear-attention, causal-conv1d,
  and FlashAttention 2 installed

Run 50 optimizer steps first and use measured throughput to forecast cost and
duration. A 2,000-3,000-example run should be treated as an hours-scale GPU job,
but the forecast should come from that pilot rather than a generic estimate.

The 18-question preparation pilot produced 21 SFT records after adding three
retrieval-failure examples. Using the installed Qwen3.5 tokenizer, their encoded
message contents range from approximately 1,536 to 2,971 tokens, so a 4,096-token
limit leaves headroom without silently cutting off the assistant target.

The Qwen3.5 stack needs special care because its language model mixes ordinary
attention with Gated DeltaNet. Without flash-linear-attention and causal-conv1d,
Transformers falls back to slower and more memory-intensive operations. Sequence
packing is disabled even though it can improve throughput: older Qwen3.5 paths
allowed linear-attention state to leak across packed samples. Transformers 5.9+
contains the boundary fix, but independent, un-packed samples are the conservative
choice for this experiment.

Select the checkpoint using task metrics, not training loss alone: answer
correctness, exact citation recall, answerable false-abstention rate, and
unanswerable abstention accuracy. Reject any checkpoint that improves formatting
while lowering factual correctness.

## Smoke-test result (September 6, 2026)

The first paid L40S smoke test passed end to end. It used 21 training records, 10
validation records, and 10 optimizer steps. The environment preflight confirmed
ms-swift 4.4.1, PyTorch 2.11.0 with CUDA 13.0, Transformers 5.12.1,
FlashAttention 2.8.3, and flash-linear-attention 0.5.1.

The adapter trained 21.64 million parameters, approximately 0.23% of the loaded
9.43-billion-parameter model. Training took 178.9 seconds, peaked at 19.86 GiB of
GPU memory, finished with training loss 0.2161, and reached validation loss
0.09979. Both losses remained finite and the gradient norms remained finite. The
checkpoint saved, reloaded, generated three held-out validation responses, and
uploaded successfully to the private pilot repository.

The three reload checks included a correct exact abstention, a grounded numeric
answer with a citation, and a grounded multi-passage answer with citations. One
generated citation page differed from its reference page, so these checks prove
pipeline function rather than final response quality.

Scheduling and container preparation took about 14 minutes and was not billed.
The billed GPU phase took approximately six minutes, or about $0.18 at the listed
$1.80/hour L40S rate. This tiny run is dominated by model download, compilation,
and adapter reload. The 50-step pilot remains necessary for a reliable full-run
forecast. Based on the steady-state steps observed here, the 50-step pilot should
fit comfortably within an hour. The provisional full 2,000-3,000-example run is
still approximately 3-8 hours, with the final range to be set from that pilot.

## Export and evaluation

After selecting a checkpoint:

1. Merge the BF16 LoRA adapter into the Hugging Face Qwen3.5 9B weights with
   `swift export --adapters <checkpoint> --merge_lora true`.
2. Convert the merged model to a new 4-bit MLX checkpoint with the MLX-VLM
   conversion tool.
3. Validate the converted model against the unmerged checkpoint on a small fixed
   answer set.
4. Serve it locally through MLX-VLM/MLX-LM for the first evaluation. LM Studio's
   published `lms import` instructions currently describe GGUF-file import, so
   do not make LM Studio compatibility a prerequisite for validating the adapter.
5. Run untuned Qwen, tuned Qwen, and Gemini on the same frozen retrieval results
   from the 100-document holdout.

Do not use MLX-LM's direct GGUF export for this checkpoint: its documentation says
that route is limited to Llama-, Mistral-, and Mixtral-style models.

## Full fine-tuning result (September 6, 2026)

The production data build created 2,000 verified answer sources from the isolated
350-document training split and 100 sources from the separate 50-document
validation split. After BM25 retrieval, oracle insertion, and safe
oracle-absent examples, the final files contained 2,256 training records (1,997
answerable and 259 unanswerable) and 111 validation records (100 answerable and
11 unanswerable). Exact Qwen tokenization confirmed that every record fit within
4,096 tokens: the training median was 1,525, p95 was 2,464, and the maximum was
3,404 tokens.

The one-epoch L40S run completed all 564 optimizer steps in 32 minutes 4 seconds.
Peak GPU memory was 20.12 GiB and final training loss was 0.1093. Validation loss
improved from 0.08284 at step 100 to 0.06674 at step 200, 0.06440 at step 300,
0.06100 at step 400, and 0.06054 at step 500. The end-of-epoch value was 0.06067,
so checkpoint 500 was selected. The complete job, including model download,
validation, adapter reload, 20 answer checks, and upload, used 2,130 billed
seconds (35 minutes 30 seconds), approximately $1.07 at $1.80/hour.

The selected adapter produced reference-equivalent responses on 16 of the 20
reload checks by manual inspection. The four visible errors involved dense table
lookup or arithmetic and one legal comparison. This is a pipeline and validation
result, not the final model-comparison score; the statistical conclusion must use
new questions from the untouched 100-document test split.

The adapter is retriever-agnostic at inference time because it consumes labeled
evidence text rather than a BM25-specific representation. It can be used directly
with dense or hybrid retrieval. A further tune on hybrid contexts is warranted
only if a held-out retrieval ablation shows a measurable context-distribution
mismatch; the same verified questions and answers can be rematerialized, so a
fresh data-generation run is unnecessary.

## Sources

- [Official Qwen3.5 9B model card](https://huggingface.co/Qwen/Qwen3.5-9B)
- [MLX-LM LoRA and QLoRA documentation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [MLX-LM Qwen3.5 implementation](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3_5.py)
- [Exact Qwen3.5 9B first-backward-pass failure](https://github.com/ml-explore/mlx-lm/issues/1206)
- [MLX-LM Qwen3.5 LoRA Metal descriptor issue](https://github.com/ml-explore/mlx-lm/issues/1185)
- [MLX-VLM LoRA documentation](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/LORA.MD)
- [MLX-VLM Qwen3.5 gradient-transform failure](https://github.com/Blaizzy/mlx-vlm/issues/1584)
- [MLX-VLM Qwen3.5 corrupted-adapter report](https://github.com/Blaizzy/mlx-vlm/issues/824)
- [ms-swift supported models](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Supported-models-and-datasets.md)
- [ms-swift fine-tuning documentation](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Pre-training-and-Fine-tuning.md)
- [ms-swift tuner and freezing parameters](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/Command-line-parameters.md)
- [LM Studio model import documentation](https://lmstudio.ai/docs/app/advanced/import-model)
- [Hugging Face Jobs overview](https://huggingface.co/docs/hub/jobs)
- [Hugging Face Jobs pricing and timeout controls](https://huggingface.co/docs/hub/jobs-pricing)
- [RunPod GPU pricing](https://www.runpod.io/pricing)
- [RAFT paper](https://arxiv.org/abs/2403.10131)
- [RAFT reference data-construction code](https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py)
- [ms-swift custom conversational dataset format and response-only loss](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Customization/Custom-dataset.md)
- [ms-swift Qwen3.5 environment and training best practices](https://github.com/modelscope/ms-swift/blob/main/docs/source_en/BestPractices/Qwen3_5-Best-Practice.md)
- [Transformers Qwen3.5 kernel requirements](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
- [Qwen3.5 packed-sequence boundary fix](https://github.com/modelscope/ms-swift/issues/9618)

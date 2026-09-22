# Fourth experiment reproduction

This guide records the Qwen3-Embedding-8B retriever and fresh Qwen3.5-9B
generator experiment. Run commands from the repository root in the project
virtual environment. The original GovInfo PDFs, extracted chunks, earlier
train/validation splits, and previously sealed test split are prerequisites.
The exact source artifacts and checksums are under
`data/qwen_embedding_experiment/`. Do not regenerate the sealed 50 questions
after inspecting test results; a changed benchmark requires a new run.

The GPU scripts use Hugging Face Jobs with code mounted at `/mnt/code`.
Embedding jobs read their verified inputs directly under `/mnt/data`;
generator jobs read `/mnt/data/qwen_embedding`. Supply `HF_TOKEN` through the
Jobs secret facility and keep Google service-account credentials outside the
repository. GPU flavor, image, limits, and billed job IDs belong in the cost
ledger. The shell entrypoints and Python scripts under `cloud/` are the exact
recipes; do not rerun a paid stage if its validated artifacts already exist.

## Embedding model and indexes

`cloud/train_qwen3_embedding_adapter_aware.py` fine-tunes the original
`Qwen/Qwen3-Embedding-8B` base with 2,560 verified query-positive records and
filtered hard negatives. It evaluates checkpoint 80 and 160 against the
existing validation split. `cloud/build_qwen3_embedding_adapter_indexes.py`
builds adapter-aware train, validation, and sealed-test indexes in resumable
shards. `cloud/run_qwen3_embedding_adapter_aware.sh` and
`cloud/run_qwen3_embedding_adapter_indexes.sh` install the pinned embedding
dependency and launch those stages inside the GPU container. The selected
checkpoint is 80; it is not a continuation of the later Qwen generator LoRA.

After downloading the completed run, verify chunk order, hashes, dimensions,
finite unit vectors, query caches, and split disjointness:

```bash
.venv/bin/python scripts/verify_qwen_embedding_artifacts.py \
  --run data/qwen_embedding_experiment/models/qwen3_embedding/adapter_aware_run2 \
  --base-validation data/qwen_embedding_experiment/models/qwen3_embedding/base_indexes/validation
```

The verified indexes contain 80,701 training chunks, 8,556 validation chunks,
and 9,140 sealed-test chunks. Vectors are 768-dimensional and searched with
exact dot product over local NumPy arrays; no managed vector database is used.

## Freeze retrieval and generator inputs

The retrieval runner tunes untuned and tuned Qwen hybrid fusion using only the
existing validation questions. It writes a checksummed configuration before
retrieving sealed-test questions. If the frozen configuration differs, it
stops instead of silently overwriting it.

```bash
.venv/bin/python scripts/run_qwen_embedding_retrieval.py
.venv/bin/python scripts/report_qwen_embedding_retrieval.py
.venv/bin/python scripts/build_qwen_embedding_sft.py
.venv/bin/python scripts/prepare_qwen_embedding_matrix.py
.venv/bin/python scripts/stage_qwen_embedding_cloud_inputs.py
```

`build_qwen_embedding_sft.py` rebuilds exactly 2,256 training and 111
validation records with five tuned-hybrid passages per prompt. It verifies
accepted question order and a 4,096-token ceiling. The matrix builder freezes
50 prompts for each of five retrievers. The staging script checks counts and
hashes before cloud upload. The test question hash is recorded in
`benchmark/benchmark_manifest.json`; the retriever settings are in
`retrieval/frozen_qwen3_configs.json`.

## Train and evaluate the answer generator

Run `cloud/train_qwen35_qwen_embedding_smoke.py` first. It requires 20 valid
adapter-reload responses. Then run `cloud/train_qwen35_qwen_embedding.py` once
from the untouched `Qwen/Qwen3.5-9B` base on the staged SFT data. It selects
the best validation-loss checkpoint and performs another 20-response reload
check before publishing the selected LoRA weights. The full job must finish
and its manifest and adapter weights must be verified before inference.

`cloud/infer_qwen35_qwen_embedding_matrix.py` produces 250 answers for each
of three Qwen states: base, prior local-hybrid adapter, and fresh Qwen-embedding
adapter. It resumes at completed, prompt-verified 250-answer state files.
Gemini supplies the remaining 250 answers from those same frozen prompt files:

```bash
.venv/bin/python scripts/run_gemini_experiment.py \
  --questions data/qwen_embedding_experiment/benchmark/questions.jsonl \
  --retrieval data/qwen_embedding_experiment/retrieval/test_all.jsonl \
  --input-dir data/qwen_embedding_experiment/inference_inputs \
  --setups bm25_top5 vertex_hybrid_top5 qwen3_untuned_dense_top5 \
    qwen3_untuned_hybrid_top5 qwen3_tuned_hybrid_top5 \
  --output data/qwen_embedding_experiment/answers/gemini.jsonl
```

The Gemini runner resumes successful answers and checks their frozen prompt
hashes. Keep a stage budget gate in effect when invoking billable generation.
After all Qwen outputs are downloaded, normalize the matrix and run
`scripts/verify_qwen_embedding_matrix.py`. It requires exactly 20 cells of 50
answers and byte-identical prompt hashes within each retrieval column.

## Judging and reporting

Score deterministic metrics with the frozen retrieval file:

```bash
.venv/bin/python scripts/score_answers.py \
  --questions data/qwen_embedding_experiment/benchmark/questions.jsonl \
  --answers data/qwen_embedding_experiment/answers/all_answers.jsonl \
  --retrieval data/qwen_embedding_experiment/retrieval/test_all.jsonl \
  --exact-citations \
  --output data/qwen_embedding_experiment/answers/automatic_scores.jsonl
```

Gemini judging uses `scripts/judge_answers_gemini.py` with
`--expected-per-question 20` and a cost limit. It resumes completed ratings
and records token use separately:

```bash
.venv/bin/python scripts/judge_answers_gemini.py \
  --questions data/qwen_embedding_experiment/benchmark/questions.jsonl \
  --answers data/qwen_embedding_experiment/answers/all_answers.jsonl \
  --output data/qwen_embedding_experiment/judges/gemini_scores.jsonl \
  --usage-output data/qwen_embedding_experiment/judges/gemini_usage.jsonl \
  --expected-per-question 20 --max-cost-usd 0.8
```

Build 200 independently shuffled blinded
Qwen prompts, each containing five candidates, then keep the identity mapping
separate from the uploaded prompts:

```bash
.venv/bin/python scripts/build_qwen_judge_inputs.py \
  --questions data/qwen_embedding_experiment/benchmark/questions.jsonl \
  --answers data/qwen_embedding_experiment/answers/all_answers.jsonl \
  --output data/qwen_embedding_experiment/judges/qwen_inputs.jsonl \
  --mapping data/qwen_embedding_experiment/judges/qwen_mapping.json \
  --expected-per-question 20 --candidates-per-prompt 5
```

Run `cloud/judge_qwen35_local_matrix.py` on the pinned untuned Qwen3.5-9B base
with `QWEN_JUDGE_EXPECTED_PROMPTS=200` and
`QWEN_JUDGE_MAX_NEW_TOKENS=1100`. After downloading its raw JSONL result,
validate and unblind all 1,000 ratings, then summarize both complete judges:

```bash
.venv/bin/python scripts/normalize_qwen_judge.py \
  --raw data/qwen_embedding_experiment/judges/qwen_raw/evaluation/qwen-judge-v4/raw_scores.jsonl \
  --mapping data/qwen_embedding_experiment/judges/qwen_mapping.json \
  --output data/qwen_embedding_experiment/judges/qwen_scores.jsonl \
  --expected-questions 200 --expected-scores 1000
.venv/bin/python scripts/summarize_qwen_embedding_experiment.py \
  --automatic data/qwen_embedding_experiment/answers/automatic_scores.jsonl \
  --gemini-judge data/qwen_embedding_experiment/judges/gemini_scores.jsonl \
  --qwen-judge data/qwen_embedding_experiment/judges/qwen_scores.jsonl \
  --output data/qwen_embedding_experiment/results/summary.json
```

The report uses 10,000 paired bootstrap samples and the predefined five-point
non-inferiority test. This cloud Qwen judge uses full BF16 weights; the earlier
experiment's local MLX judge used 4-bit weights. All 1,000 Qwen answer and
citation ratings are present. Fourteen Qwen ratings omit the auxiliary
`unsupported_claim` flag; normalization preserves those as missing instead of
inventing labels, and the report shows the available denominator.

`scripts/build_qwen_embedding_human_review.py` prepares 150 blinded paired
review rows. Give reviewers the CSVs and retain `mapping.json` separately so
candidate identities are not disclosed before review. The automated findings
remain labeled as such until human review is completed.

The [protocol and observed retrieval results](QWEN_EMBEDDING_EXPERIMENT.md)
are separate from the earlier three experiments. Reuse their artifacts only
where the protocol explicitly specifies an existing baseline.

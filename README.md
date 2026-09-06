# Lean RAG model comparison

Downloads official GovInfo package PDFs discovered through `api.govinfo.gov`.
The downloader does not perform OCR, text extraction, chunking, embedding, or
indexing. Those stages are separate so experiments can reuse the frozen corpus.

## Run

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
# Set GOVINFO_API_KEY in your environment or in the git-ignored .env file.
.venv/bin/python scripts/download_govinfo.py --target 10
.venv/bin/python scripts/download_govinfo.py --target 1000
```

The public DEMO_KEY is the fallback and has restricted API quotas. Never commit
your personal API key. The key is sent only to the official API in an HTTP header.

The frozen experiment corpus is a deterministic, stratified subset of 500 PDFs.
The remaining downloaded PDFs are retained but are outside this benchmark.

## Outputs and resumption

- `data/pdfs/`: accepted PDFs, with package IDs as filenames.
- `data/manifest.csv`: title, date, discovery collection, PDF source URL, local
  path, package ID, SHA-256, and page count.
- `data/download.sqlite3`: persistent discovery cursors and per-document status.
- `data/pilot.json`: successful ten-document pilot record.
- `data/summary.json`: completed-run totals.

Rerun the same command to resume. Existing accepted files are revalidated;
completed collection pages are cached. Partial downloads are restarted with
atomic replacement after validation. HTTP connection/server failures are retried
with backoff. Rate limits stop the run without losing progress. Use
`--retry-failed` to revisit previously failed documents. One process may run at a
time, enforced with a file lock. The downloader stops below 2 GiB of free space.

Documents must have at least 10 pages by default (`--min-pages`)
and be no larger than 50 MiB (`--max-mb`). They must have a valid PDF header
and end marker, and pass strict pypdf parsing and content-stream decoding.
Encrypted, invalid, short, and byte-identical documents are excluded. These
checks do not guarantee visual correctness or semantic uniqueness. Renamed
`.corrupt` files are excluded from the manifest.

Sampling rotates across annual reports, hearings, congressional reports and
documents, committee prints, GAO reports, mandated reports, coastal studies,
government publications, education reports, budgets, and economic reports.
Collections are sampled in API last-modified order; this is a diverse convenience
sample, not a statistically representative or year-stratified sample. Some
GovInfo collections overlap: the manifest records the first discovery collection.

Official documentation: https://github.com/usgpo/api and
https://api.govinfo.gov/docs/.

## Extraction pilot

Select 30 PDFs across collections and page-count ranges, verify their hashes,
and extract page-level text with traceable document and page identifiers:

```sh
.venv/bin/python scripts/extract_corpus.py
```

Outputs are written under `data/processed/pilot30/`. Existing pilot outputs are
never replaced unless `--force` is supplied. Quality flags identify empty,
low-text, encoding-damaged, or failed pages for visual review and possible OCR.

Build the pilot BM25 index and optionally run a search:

```sh
.venv/bin/python scripts/build_bm25.py
.venv/bin/python scripts/build_bm25.py --query "unmanned aircraft imports"
```

LM Studio's local REST API is used for generator runs. Start the server on its
default local address, then make a grounded smoke-test request:

```sh
.venv/bin/python scripts/lmstudio_client.py \
  --question "When was the report published?" \
  --evidence "[doc=test page=1] The report was published in 2023."
```

The client disables model reasoning and sampling so comparisons use the same
answering mode and expose no hidden reasoning text in the scored response.

Generate direct factual question candidates from clean pilot pages with the
local 9B model. The script rejects any candidate whose evidence is not an exact
source quote or whose reference answer is absent from that quote:

```sh
.venv/bin/python scripts/generate_questions.py
```

## Frozen 500-document experiment

The experiment uses 58,071 pages from 500 documents across all 12 downloaded
collections. It keeps scan-only documents in the corpus so extraction coverage
is measurable; 450 documents currently contribute searchable text. The BM25
index contains 103,986 chunks of 350 words with a 50-word overlap.

The fixed benchmark has 50 questions, reused without changes for every model
and retrieval setup:

- 20 direct factual questions
- 10 numeric or date questions
- 10 same-document, multi-passage questions
- 5 cross-document questions
- 5 unanswerable questions

Candidates and exact source quotes are machine-validated and curated. A blind
human audit is still required before treating the scores as publication-grade.

Rebuild the frozen artifacts with:

```sh
.venv/bin/python scripts/subset_extraction.py \
  --source-pages data/processed/corpus1000/pages.jsonl.tmp \
  --source-documents data/processed/corpus1000/documents.jsonl.tmp \
  --documents 500 --output-dir data/processed/corpus500 --force
.venv/bin/python scripts/build_bm25.py \
  --pages data/processed/corpus500/pages.jsonl \
  --output data/indexes/corpus500-bm25.sqlite3 --force
.venv/bin/python scripts/build_benchmark.py
.venv/bin/python scripts/retrieve_benchmark.py --force
```

The model comparison uses one fixed retrieval setup: SQLite FTS5/BM25 top five.
Keeping retrieval constant isolates the effect of generator choice.

The same evidence is run against four locally installed generators:
Qwen3 1.7B and Qwen3.5 2B, 4B, and 9B. LM Studio runs them with reasoning off,
temperature zero, a 16,384-token context, and the same answer prompt. Retrieved
chunks are capped at 6,000 characters to prevent pathological table tokenization.

Gemini 3.8 Flash is also run through Vertex AI as a hosted reference using the
same prompt, zero thinking budget, temperature, output cap, and frozen evidence.
The service-account file `ai-lab-fasa.json` is explicitly git-ignored.

```sh
.venv/bin/python scripts/run_experiment.py
.venv/bin/python scripts/run_gemini_experiment.py
.venv/bin/python scripts/score_answers.py
.venv/bin/python scripts/judge_answers.py
.venv/bin/python scripts/judge_answers_gemini.py
.venv/bin/python scripts/summarize_experiment.py
.venv/bin/python scripts/compare_judges.py
```

Every long stage is resumable. Retrieval is frozen before generation, so each
generator receives byte-identical evidence for a given question and setup.

The completed pilot and its interpretation are documented in `RESULTS.md`.

# Gemini 3.8 Flash versus Qwen3.5 9B: confirmatory RAG results

## Current conclusion

The automated evidence does **not** establish that Gemini 3.8 Flash is at least
five percentage points more accurate than Qwen3.5 9B in this BM25 RAG setup.
Across 250 paired questions, the combined blind-judge score was 53.4% for Gemini
and 50.6% for Qwen. The paired difference was **+2.8 percentage points for
Gemini**, with a 95% bootstrap confidence interval of **-1.1 to +6.7 points**.
The interval includes zero and the estimate is below the preregistered five-point
threshold.

This is an automated interim conclusion. The protocol defines blinded human
ratings as the primary outcome, so the formal conclusion remains pending until
the question audit and two independent human reviews are completed.

## Experiment

- Corpus: 500 GovInfo PDFs.
- Retrieval: one frozen SQLite FTS5/BM25 top-five result per question.
- Models: Gemini 3.8 Flash and local Qwen3.5 9B 4-bit.
- Benchmark: 250 fresh questions, paired across both models.
- Mix: 100 direct, 50 numeric/date, 50 same-document multi-passage, 25
  cross-document, and 25 unanswerable.
- Generation: identical evidence and instructions, temperature zero, reasoning
  off, and a 220-token answer cap.
- Evaluation: deterministic checks plus two blind model judges. Each judge saw
  randomized anonymous candidates and scored correctness from 0 to 2.

The question and retrieval files retained their preregistered hashes after both
answer runs:

- Questions: `20179012bde03c6677c03294c51bdde8d50189c5969be0099be6e96b82baaf61`
- Retrieval: `211e546984353365b166a2f386da00624db4b15e497360809cf4046be8b15029`

## Answer quality

| Category | Questions | Gemini | Qwen 9B | Gemini minus Qwen |
|---|---:|---:|---:|---:|
| All | 250 | 53.4% | 50.6% | +2.8 pp |
| Answerable only | 225 | 49.3% | 49.1% | +0.2 pp |
| Direct | 100 | 63.8% | 66.0% | -2.3 pp |
| Numeric/date | 50 | 52.5% | 50.5% | +2.0 pp |
| Same-document multi-passage | 50 | 40.0% | 36.5% | +3.5 pp |
| Cross-document | 25 | 4.0% | 4.0% | 0.0 pp |
| Unanswerable | 25 | 90.0% | 64.0% | +26.0 pp |

The answerable-only difference was +0.2 points with a paired 95% interval from
-3.7 to +3.9 points. For ordinary answerable questions, the two models were
effectively tied in this experiment. Gemini's overall advantage came mainly from
better abstention on unanswerable questions; that subgroup difference was +26
points with a 95% interval from +10 to +44 points.

The judges were not interchangeable. The Qwen judge favored Gemini by 6.0 points
(95% paired interval +0.4 to +11.4), while the Gemini judge favored Qwen by 0.4
points (interval -3.8 to +3.4). Combining the two was the prespecified secondary
summary. Across all 500 answer ratings, the judges agreed exactly 83.8% of the
time, were within one rubric point 91.4% of the time, and had a Pearson score
correlation of 0.775.

## Retrieval is the main limit

BM25 retrieved all required gold passages for 134 of 225 answerable questions
(59.6%). When all gold evidence was retrieved, combined judge correctness was
69.2% for Gemini and 70.1% for Qwen. When evidence was incomplete, correctness
fell to 20.1% and 18.1%, respectively.

Cross-document performance was 4% for both models, and neither model reliably
cited all required cross-document evidence. This result mainly exposes a
retrieval failure: a generator cannot synthesize passages that BM25 did not put
in its context. Improving retrieval should have a larger effect than replacing
Qwen 9B with Gemini for answerable questions.

## Citations, abstention, and speed

Gemini's exact gold-document-and-page citation recall was 43.8%, versus 19.1%
for Qwen. The paired difference was +24.7 points with a 95% interval from +18.0
to +31.6 points. Gemini is the clear choice when exact source attribution is a
hard requirement under the current prompt.

Gemini abstained on 48.4% of answerable questions, compared with 35.1% for Qwen.
That made Gemini safer when evidence was absent, but it also caused more missed
answers. On the unanswerable controls, exact abstention accuracy was 96% for
Gemini and 64% for Qwen.

Mean per-request latency was 1.20 seconds for Gemini and 42.8 seconds for local
Qwen. Qwen requests ran four at a time, so this latency is not total wall time.
The local result is specific to this machine and LM Studio configuration. Gemini
used 779,617 input tokens and 40,052 output tokens for its 250 answers and 250
two-candidate judge calls, an estimated **$0.73** at the current published
Gemini 3.8 Flash promotional rates of $0.75 per million input tokens and $3.75
per million output tokens. See [Google Cloud pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing).

## Deployment decision

Use Qwen3.5 9B locally when privacy, offline use, and avoiding per-request fees
matter. Its answer correctness on answerable questions was indistinguishable
from Gemini in this run.

Use Gemini 3.8 Flash when exact citations, abstention behavior, and low latency
matter enough to justify hosted inference. The current evidence does not justify
choosing Gemini solely for a five-point answer-accuracy improvement.

The next engineering experiment should keep these same two generators and
improve retrieval: expand the BM25 candidate pool, diversify results across
documents, and optionally rerank candidates. Reusing the frozen questions will
show whether those changes repair the large retrieval-miss and cross-document
failure rates.

## Human conclusion still required

The blinded review package is in `data/confirmatory/review/`. Reviewers should
audit question validity before viewing answers, rate the A/B candidates
independently, adjudicate disagreements while identities remain hidden, and run
`scripts/analyze_human_review.py`. Until then, the statistically correct wording
is: **the automated experiment found no established five-point accuracy
advantage for Gemini over local Qwen3.5 9B.**

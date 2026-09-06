# RAG model comparison: findings and recommendation

## Vertex-hybrid experiment — September 6, 2026

The second experiment is complete. It compared pure BM25 with an equal-weight
hybrid retriever that fuses BM25 and Vertex AI `gemini-embedding-001` rankings.
The confirmatory benchmark contained 50 newly generated and sealed questions
from 59 test documents unused by the first held-out benchmark. Every answer was
generated from a frozen top-five context, and all four generators within a
retrieval condition received byte-identical prompts and evidence.

The hybrid Qwen adapter was trained from the original `Qwen/Qwen3.5-9B` base,
rather than from the BM25 adapter. It used 2,256 training and 111 validation
records rebuilt with hybrid contexts. Training completed one epoch and 564
optimizer steps. Checkpoint 564 had the best validation loss, **0.06357**, and
produced 20 of 20 responses after an independent adapter reload. The adapter is
published as `vamsee9201/qwen35-9b-hybrid-rag-lora`.

### Retrieval

| Retriever | All-gold recall | Mean passage recall | MRR | nDCG | Cross-document all-gold recall | Local search latency/question |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 70% | 77% | 0.710 | 0.697 | 40% | 22.5 ms |
| Vertex dense | **78%** | 83% | 0.740 | 0.727 | **80%** | **2.9 ms** |
| BM25 + Vertex hybrid | **78%** | **84%** | **0.811** | **0.777** | 40% | 25.8 ms |

Hybrid retrieval improved observed overall recall and ranking quality. Dense-only
retrieval found all required evidence for four of the five cross-document
questions, while reciprocal-rank fusion reduced that result to two of five.
This five-question slice is too small for a stable cross-document conclusion,
but it shows that the validation-selected fusion can suppress a useful dense
result. The latency measurements cover local index search and exclude Vertex
query-embedding latency.

### Complete answer-quality matrix

The primary score is the normalized mean of two independently randomized blind
judges: Gemini 3.8 Flash and untuned Qwen3.5 9B. Each cell contains 50 answers.

| Generator | BM25 primary score | Hybrid primary score | BM25 exact citation recall | Hybrid exact citation recall | BM25 token F1 | Hybrid token F1 |
|---|---:|---:|---:|---:|---:|---:|
| Gemini 3.8 Flash | 75.0% | **82.0%** | 64% | 73% | 0.299 | 0.316 |
| Base Qwen3.5 9B | 76.5% | 77.0% | 48% | 57% | 0.329 | 0.364 |
| BM25-trained Qwen3.5 9B | **76.0%** | 79.5% | 69% | 73% | **0.473** | **0.481** |
| Hybrid-trained Qwen3.5 9B | 75.5% | 77.5% | **71%** | **74%** | 0.457 | 0.462 |

Gemini gained seven points from hybrid retrieval, with a paired 95% bootstrap
interval from **-1.5 to +16 points**. The BM25-trained Qwen adapter gained 3.5
points, with an interval from **-6 to +13 points**. Both intervals include zero,
so 50 questions do not establish an answer-quality improvement for either
generator even though the retrieval metrics and point estimates favor hybrid.

The new hybrid-trained adapter scored **77.5%** under hybrid retrieval, compared
with **82.0%** for Gemini. Its paired difference was **-4.5 points** with a 95%
interval from **-10 to 0 points**. The predefined one-sided 95% lower bound was
**-9 points**, below the -5-point margin, so the new adapter did **not** establish
non-inferiority to Gemini.

The existing BM25-trained adapter scored **76.0%** under BM25, compared with
**75.0%** for Gemini. Its paired difference was **+1 point**, with a 95% interval
from **-2 to +3.5 points** and a one-sided lower bound of **-1.5 points**. It met
the predefined five-point non-inferiority criterion in this new benchmark.

### Fine-tuning and transfer result

| Comparison | Mean primary-score difference | Paired 95% interval |
|---|---:|---:|
| Hybrid-trained minus BM25-trained Qwen, BM25 contexts | -0.5 points | -1.5 to 0 |
| Hybrid-trained minus BM25-trained Qwen, hybrid contexts | -2.0 points | -6 to 0 |
| BM25-trained Qwen, hybrid minus BM25 contexts | +3.5 points | -6 to +13 |
| Hybrid-trained Qwen, hybrid minus BM25 contexts | +2.0 points | -6.5 to +10.5 |

Retrieval-specific retraining did not improve answer quality here. The older
BM25-trained adapter transferred cleanly to hybrid contexts and outscored the
new hybrid-trained adapter by two points under hybrid retrieval. The interval
still includes a tie, so this is evidence against a useful gain from the second
fine-tune rather than proof that hybrid-context training is intrinsically worse.
The likely practical explanation is that the supervision, answer style, oracle
insertion policy, and top-five prompt format stayed constant, while the
retriever changed only the distractor distribution. The first adapter had
already learned the desired grounded-answer and citation behavior.

Under hybrid retrieval, direct-question primary scores were 97.5% for Gemini,
93.8% for base Qwen, and 95.0% for both adapters. Multi-passage scores were
76.7%, 63.3%, 65.0%, and 65.0%, respectively. Cross-document scores remained
between 40% and 45% for every generator, confirming that this category remains
the main end-to-end weakness.

### Recommendation

Use the **BM25-trained Qwen3.5 9B adapter with the Vertex-hybrid retriever** for
the strongest local configuration observed in this experiment. It reached
79.5% on the primary score, only 2.5 points behind Gemini's 82.0% point estimate,
and had much stronger lexical overlap than Gemini. Use Gemini when the highest
observed semantic answer score and hosted latency matter most.

Do not replace the BM25 adapter with the new hybrid adapter based on this run.
Keep the new adapter as a fully reproducible negative result: it establishes
that rebuilding the same supervision with hybrid distractors did not add value.
A future fine-tune should change the supervision itself—for example, add
verified hard negatives, explicit cross-document synthesis examples, and
retrieval-failure abstentions—then evaluate on a larger sealed cross-document
set. Repeating the same fine-tune recipe on another retriever is not justified
by these results.

The Vertex portion cost an estimated **$8.72**, including embeddings, 100 Gemini
answers, and 400 Gemini judge scores. Hugging Face L40S jobs cost an estimated
**$2.27**, including the smoke test, full fine-tune, 300 Qwen answers, and the
Qwen judge. The combined measured total was approximately **$10.99**. Benchmark
question generation is not included because that earlier run did not retain
complete billable-token usage.

All 400 answer records, 800 blind-judge ratings, 10,000-resample paired
statistics, category results, cost data, and blinded human-review sheets passed
the experiment's structural checks. The human-review sheets are prepared but
not yet completed, so publication claims should identify the dual-model judging
procedure and the small 50-question sample as limitations. Vertex AI does not
expose an immutable revision for `gemini-embedding-001`; the request date,
configuration, checksums, and response statistics are recorded instead.

---

## Final fine-tuned comparison — September 6, 2026

The full Qwen3.5 9B LoRA fine-tune completed successfully. It trained for one
epoch (564 optimizer steps) on 2,256 RAG examples and used checkpoint 500, which
had the best validation loss. The selected adapter reloaded successfully and
produced answers in both validation inference and the untouched test run.

The final test used 50 new questions from documents excluded from training and
validation. Qwen and Gemini 3.8 Flash received the same frozen, pure-BM25 top-five
contexts. No gold passage was inserted into the test prompts. Both models
completed all 50 answers.

| Final held-out metric | Fine-tuned Qwen3.5 9B | Gemini 3.8 Flash |
|---|---:|---:|
| Blind-judge answer score | **84%** | **85%** |
| Blind-judge citation accuracy | 76% | 76% |
| Exact citation recall | **82%** | 76% |
| Mean response time | 5.20 s | **1.32 s** |

The paired answer-score difference, Qwen minus Gemini, was **-1 percentage
point**, with a 95% bootstrap confidence interval from **-7 to +4 points**.
This experiment therefore found no statistically supported answer-quality
difference between the two generators. Fine-tuned Qwen is competitive with
Gemini for this benchmark, while Gemini remains about four times faster in the
measured environments. Runtime is not a hardware-normalized comparison.

BM25 retrieved every required gold passage for 42 of the 50 questions (84%).
That retrieval ceiling particularly affected cross-document questions: both
models were strong on multi-passage questions and weak on the five
cross-document cases. The next useful experiment is to freeze this Qwen adapter
and test both generators with identical hybrid-retrieval contexts. A new
fine-tune is only warranted if that experiment reveals a systematic context
distribution problem for Qwen.

The semantic judge was Gemini 3.8 Flash and candidate identities were hidden
and randomized. Because Gemini was also one of the candidates, self-preference
remains a limitation. The one-point result is too small to support superiority
for either model, and 50 questions cannot rule out modest differences.

---

> **Update:** The 50-question, five-model results below are the exploratory
> pilot. The newer 250-question comparison between Gemini 3.8 Flash and
> Qwen3.5 9B is documented in [CONFIRMATORY_RESULTS.md](CONFIRMATORY_RESULTS.md).
> Its automated evaluation found no established five-point answer-accuracy
> advantage for Gemini; blinded human evaluation is still pending.

## Conclusion

The sub-10B local models are good enough for this RAG workload, but model choice
depends on the deployment goal.

- **Use Qwen3.5 4B as the default local model.** It provides the best practical
  balance of answer quality, latency, and memory use.
- **Use Qwen3.5 9B when local accuracy matters more than speed.** It scored two
  points above 4B under the combined judges, but took about twice as long per
  answer. The measured quality difference between 4B and 9B is too small to be
  conclusive with 50 questions.
- **Use Gemini 3.8 Flash when hosted inference is acceptable.** It had the best
  overall answer score, much stronger citations, and the lowest latency.
- **Do not select 1.7B or 2B for the main system.** They can serve as inexpensive
  baselines, but their accuracy and citation behavior were less reliable.

## Evidence

The experiment used 500 GovInfo PDFs, one frozen BM25 top-five retrieval result
per question, 50 questions per model, and five models. This produced 250 answers.
Qwen3.5 9B and Gemini 3.8 Flash independently judged every answer under anonymous
labels.

| Model | Combined judge score | Deterministic accuracy | Exact citation recall | Mean latency |
|---|---:|---:|---:|---:|
| Qwen3 1.7B | 62.0% | 68.6% | 2.2% | 9.9 s |
| Qwen3.5 2B | 64.0% | 71.4% | 0.0% | 9.5 s |
| **Qwen3.5 4B** | **69.5%** | 74.3% | 24.4% | 18.5 s |
| **Qwen3.5 9B** | **71.5%** | 80.0% | 17.8% | 38.4 s |
| **Gemini 3.8 Flash** | **76.5%** | **82.9%** | **68.9%** | **1.3 s** |

Gemini's combined judge advantage over 9B was five percentage points, with a
paired 95% bootstrap interval from -0.5 to 11.5 points. The interval includes
zero, so this pilot does not establish that Gemini is more accurate than 9B.
Gemini's advantage over 4B was seven points, with an interval from 0.5 to 14
points. That result favors Gemini, although the sample remains small.

The two judges agreed exactly on 82.8% of answer ratings, were within one rubric
point on 97.6%, and had a score correlation of 0.85. This agreement makes the
overall ranking more credible than relying on either judge alone.

## What limits the system

Retrieval is now a larger problem than generator size. BM25 retrieved all gold
evidence for 68.9% of answerable questions overall, but for none of the five
cross-document questions. Cross-document answer scores remained between 5% and
20% across the models because the prompt often lacked the required passages.
A larger generator cannot reliably recover evidence that retrieval omitted.

Local Qwen citation formatting also needs work. The models frequently copied
placeholder evidence numbers or omitted the exact GovInfo document ID and page.
Gemini's exact citation recall of 68.9% matched the percentage of questions for
which BM25 retrieved all gold evidence, suggesting it used available citations
far more consistently.

## Recommended next step

Keep Qwen3.5 4B as the local generator and improve the BM25 retrieval stage.
Test query expansion, result diversification across documents, and a larger
candidate pool with deterministic selection. Reuse the same 50 questions first
so the new retrieval results remain directly comparable.

After retrieval improves, rerun Qwen3.5 4B, Qwen3.5 9B, and Gemini 3.8 Flash.
There is little value in repeating 1.7B and 2B unless they are needed as
baselines. Human-audit the benchmark and a stratified sample of answers before
making publication-grade claims.

## Practical deployment choice

| Requirement | Recommended model |
|---|---|
| Fully local, balanced quality and resource use | Qwen3.5 4B |
| Fully local, highest observed accuracy | Qwen3.5 9B |
| Best observed end-to-end performance | Gemini 3.8 Flash |
| Cheap experimental baseline | Qwen3.5 2B |

The hosted Gemini portion of this experiment cost approximately $0.18 for 50
answers and 250 blind judgments. Actual deployment cost will depend on traffic
and prompt length.

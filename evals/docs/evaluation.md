# TaxMate — RAG Evaluation

## Method

**Knowledge base.** 7 BIR issuances, text-extracted once into the committed `corpus/` folder (`python -m taxmate.corpus build`), using PyMuPDF with same-row line joining so table rows stay intact. RMC 50-2018 is a scanned PDF with unusable OCR, so it was transcribed by hand and the transcription was verified against the scan.

**Chunking.** Each document is chunked according to its structure:

| Document | Chunking | Chunks |
|---|---|---|
| RMC No. 50-2018 (Q&A) | one chunk per question-and-answer | 62 |
| RR No. 8-2018 (digest) | recursive splitter, 1,200 chars / 150 overlap, tagged with page | 25 |
| RR No. 11-2018 | same | 104 |
| BIR Form 1701Q guidelines | same | 10 |
| BIR Form 1701-MS guidelines | same | 20 |
| BIR Form 2551Q guidelines | same | 8 |
| EOPT Act flyer | one chunk per headed topic (e.g. "File and Pay Anywhere"); see `debugged-failure.md` | 18 |
| **Total** (chunks under 40 chars dropped) | | **247** |

**Retrieval.**
- Embeddings: `text-embedding-3-small`, stored in an in-memory FAISS `IndexFlatIP` over L2-normalised vectors, which gives cosine similarity.
- Follow-up questions are rewritten into standalone queries using the chat history. The rewrite step makes no LLM call on the first turn.
- Top **k = 4** chunks are retrieved.

**Relevance threshold: 0.47.** If the top chunk scores below this, TaxMate skips generation and returns a fixed "couldn't find this in my BIR sources" reply. The value was chosen with `python -m evals.tune_threshold --run threshold-v1` (`evals/results/threshold-v1/scores.csv`):

| | top-1 cosine score |
|---|---|
| 30 in-corpus golden questions | min 0.478 · median 0.675 · max 0.757 |
| 4 out-of-corpus questions | 0.367 (estate tax), 0.456 (Cebu real property tax), 0.519 (US LLC), 0.621 (crypto) |

No threshold separates the two groups: the crypto question scores higher than a third of the in-corpus questions. We therefore chose the highest value that keeps **every** in-corpus question: 0.47. That caught **2 of 4 (50%)** out-of-corpus questions at the retrieval stage.

The other two are handled downstream:
- Prompt rule 1 tells the model to say it couldn't find the answer.
- An answer without valid `[S#]` citations is retried once in strict mode, then replaced by the fallback.

In the smoke test, both the estate-tax and crypto questions ended in the fallback this way. In-corpus questions whose reference document was in the top 4: **27/30**. The three misses were GQ12, GQ22 and GQ-EOPT; GQ-EOPT is the staleness case analysed in `debugged-failure.md`.

**Generation.** `gpt-4o-mini`, temperature 0, structured output `{type: answer|clarify, text}`. Each prompt variant is defined in `taxmate/prompts.py`; see `prompt-comparison.md`.

**Evaluation.**
- Tool: RAGAS **0.4.3**, using the per-sample `ragas.metrics.collections` API.
- Judge: `gpt-4o-mini`.
- Metrics: faithfulness, answer relevancy (embeddings `text-embedding-3-small`), context precision (with reference) and context recall.
- Dataset: `evals/golden_set.jsonl`: 30 in-corpus questions, each with a reference answer and reference document, written from the sources and reviewed by the project owner, plus 4 out-of-corpus questions. Out-of-corpus questions are scored on whether TaxMate falls back, not with RAGAS.
- Command: `python -m evals.run_ragas --run <name> --variant <baseline|deliberate>`. It writes `evals/results/<name>/results.csv` and `summary.json`.

## Results

### v1-deliberate (default prompt)

`python -m evals.run_ragas --run v1-deliberate --variant deliberate` writes to `evals/results/v1-deliberate/`.

| Metric | Score (mean over 30 in-corpus questions) |
|---|---|
| Faithfulness | 0.643 |
| Answer relevancy | 0.440 |
| Context precision | 0.856 |
| Context recall | 0.933 |
| In-corpus fallback rate | 0% (0/30) |
| Out-of-corpus declined | 4/4: 2 fallback (estate tax, crypto) + 2 blocked as off-topic by the classifier (Cebu real property tax, US LLC) |

Retrieval is strong: recall 0.93 and precision 0.86. The weak spots are on the answer side, where faithfulness and relevancy are low. See `prompt-comparison.md` for the baseline comparison.

### Lowest-scoring questions

Each question is listed with its diagnosis from the stored response, the retrieved chunks and the Langfuse trace:

| ID | F / AR / CP / CR | Root cause | Diagnosis |
|---|---|---|---|
| GQ23 *"What are the deadlines for filing BIR Form 1701Q?"* | 0.14 / 0.00 / 0.00 / 0.00 | **Retrieval (chunking)** → hallucination | The 1701Q guide's *general* chunk (`p1#0`) ranked first. The chunk that actually holds the deadlines (`p1#2`) wasn't in the top 4. The model then filled the gap from outside knowledge: *"first quarter… April 15"* and a *"fourth quarter… January 31"* 1701Q, both wrong (the correct dates are May 15 / Aug 15 / Nov 15, and there is no Q4 1701Q). It still cited [S1], so the citation check didn't catch it. Trace `53f7e756…`. |
| GQ-EOPT *"Where can I file and pay my quarterly income tax return?"* | 0.80 / 0.98 / 0.75 / 0.00 | **Retrieval (stale source wins)** | Returned the 2018 rule (*"AAB within the jurisdiction of your RDO"*) from the 1701Q guide; the 2024 EOPT flyer (*file and pay anywhere*) was not retrieved. This is the debugged failure; see `debugged-failure.md`. Trace `61777b7b…`. |
| GQ29 *"I'm on the 8% rate. Can I deduct my rent and internet bills?"* | 0.29 / 0.00 / 0.75 / 1.00 | **Prompt** | The conclusion is correct (not deductible), but the answer added an unasked, mis-applied point about the ₱250k reduction for mixed-income earners, which the judge counted as unsupported. Relevancy 0 because of the closing CPA line (see below). Trace `d804a3f9…`. |
| GQ18 *"What certificate should my client give me…?"* | 0.40 / 0.59 / 1.00 / 1.00 | **Prompt / judge strictness** | The answer is correct (Form 2307) with perfect retrieval. Faithfulness was docked for the paraphrase *"to indicate the tax that has been withheld"* and the closing line. |
| GQ13 *"Why do I need to give my client a copy of my COR…?"* | 0.43 / 0.52 / 1.00 / 1.00 | **Prompt / judge strictness** | Correct and complete against the reference, with perfect retrieval. Partial faithfulness credit, as in GQ18. |

**Pattern.**
1. **14/30** deliberate answers score answer relevancy exactly 0.00, against 2/30 for baseline. Each of them ends with a standalone *"Please verify with a CPA or your RDO."* line (prompt rule 8). RAGAS rates such answers *noncommittal* and scores them 0, even when the content is correct, e.g. GQ03. This is the most common root cause and is the target of the improvement below.
2. Two retrieval misses (GQ23 chunking, GQ-EOPT staleness) cause the only *wrong* answers.

_Langfuse screenshots of the GQ23 and GQ-EOPT traces: `logs/trace-gq23.png`, `logs/trace-gq-eopt.png`._

### Known limitation: the threshold also drops some legitimate questions

The threshold was tuned on golden-set phrasing, but natural phrasings can land just under it:
- memory scenario M2, *"Can I deduct my office rent?"*, scored 0.446;
- guardrail case G16, a 2551Q deadline question prefixed with *"Email me the answer at…"*, scored 0.459.

Both got the safe fallback instead of an answer, so the cost is helpfulness, not correctness.

## Targeted improvement (v1 → v2)

**Hypothesis (written before the change).** The most common root cause among low scorers is prompt rule 8. It makes every answer end with a standalone line, *"Please verify with a CPA or your RDO."*, and RAGAS's answer-relevancy judge rates such answers *noncommittal*, scoring them 0: 14/30 deliberate answers score exactly 0.00, against 2/30 for baseline.

The planned change:
- remove rule 8 from the deliberate prompt, so the model answers only the question;
- show the reminder as a fixed caption under every answer in the UI, so users still always see it, now as UI text rather than model output;
- bump `PROMPT_VERSION` to `v2`.

Expected results:
- answer relevancy rises substantially, towards or above baseline's 0.737;
- faithfulness rises slightly, because there is no longer an unsupported closing claim;
- context precision and recall stay the same, because retrieval is untouched;
- guardrail and memory pass rates stay the same.

**Result.** `python -m evals.run_ragas --run v2-no-closing-disclaimer --variant deliberate` writes to `evals/results/v2-no-closing-disclaimer/`; the suites are in `guard-v2/` and `memory-v2/`.

| | baseline prompt | v1 deliberate | **v2 deliberate (no closing disclaimer)** |
|---|---|---|---|
| Faithfulness | 0.725 | 0.643 | **0.834** |
| Answer relevancy | 0.737 | 0.440 | **0.799** |
| Context precision | 0.876 | 0.856 | 0.879 |
| Context recall | 0.933 | 0.933 | 0.933 |
| Answers with relevancy exactly 0 | 2/30 | 14/30 | **0/30** |
| In-corpus fallback rate | 3.3% | 0% | 0% |
| Out-of-corpus declined | 3/4 | 4/4 | 4/4 |
| Guardrail suite | 16/16 | 16/16 | 16/16 |
| Memory suite (turns) | 7/8 | 7/8 | 7/8 |

**Interpretation.**
- **The hypothesis held.** Answer relevancy rose from 0.440 to 0.799, and no answer scores 0 any more. GQ03, a correct answer that scored 0.00 in v1, now scores 0.78.
- **Faithfulness rose more than expected**, from 0.643 to 0.834. Without the closing line, the judge no longer counts an unsupported "verify with a CPA" claim in every answer, and answers are shorter and stay on the sources.
- **Retrieval metrics are unchanged** (precision moves within judge noise), as expected, since retrieval was not touched. Guardrail and memory results are also unchanged.
- **v2 now beats the baseline prompt on every metric** while keeping its 4/4 out-of-corpus refusals, so v2 is the shipped default (`PROMPT_VERSION = "v2"`).
- **Users still see the reminder:** it is a fixed caption under every answer (`copy.VERIFY_NOTE`, pinned by `test_answers_show_verify_note_but_refusals_do_not`), plus the sidebar disclaimer.

**What it did not fix.** GQ23 is still wrong: it invents an April 15 first-quarter deadline and a Q4 1701Q, because the deadlines chunk isn't retrieved. Its answer relevancy is now 1.00, a reminder that relevancy measures whether the answer is *on-topic*, not whether it is *correct*. Faithfulness (0.17) and context recall (0.00) still flag it. The cause is retrieval (chunking), which this change didn't touch; it is the next candidate improvement, along with the threshold limitation above.

## Current version (v3)

The shipped version also includes the debugged-failure fix, described in `debugged-failure.md`:
- the EOPT flyer is chunked by heading;
- an EOPT supersession rule applies in retrieval;
- prompt rule 6 is sharpened (`PROMPT_VERSION = v3`).

Run `v3-eopt-supersession` scores:
- faithfulness **0.797**;
- answer relevancy **0.791**;
- context precision **0.890**;
- context recall **0.967**;
- in-corpus fallback 3.3% (GQ01, a pre-existing flaky citation case);
- out-of-corpus declined 4/4;
- guardrails 16/16, memory 7/8.

The threshold was re-checked after re-chunking (`threshold-v2`): 0.47 still keeps every in-corpus question, and reference-doc-in-top-4 rose to 28/30.

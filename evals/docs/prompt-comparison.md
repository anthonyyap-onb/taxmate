# TaxMate — Prompt Comparison

## Three variants on the current pipeline (prompt v4, with calculator tools)

The three system prompts are in `taxmate/prompts.py`. Each ran through the **same** pipeline (retrieval k = 4 with threshold 0.47, guardrails, profile memory, calculator tools with code-enforced preconditions) on the **same** fixed inputs. The only thing that changes is the system prompt.

| Variant | Style | What it says |
|---|---|---|
| `baseline` | zero-shot persona | one line: *"You are a Philippine tax assistant. Answer the user's question using the provided context. Cite sources as [S#]."* |
| `deliberate` (default) | zero-shot with explicit guidelines | 7 rules: answer only from sources or calculator results; cite every sentence; apply the profile; clarify only when an unknown profile field changes the answer; never do arithmetic, use the tools; newer issuances supersede older ones; ≤ 180 words |
| `structured` | guidelines + explicit reasoning | `deliberate` plus a hidden `analysis` field filled **before** the reply: (1) which sources apply, (2) which profile fields matter and whether any is unknown, (3) whether a calculation is needed and its inputs are stated, (4) the decision: answer / clarify / not found. The user never sees `analysis`. |

I chose a reasoning variant over a few-shot one because it tests a different idea (does deliberating help?), while few-shot would mostly test the choice of examples.

### Results

| | baseline | **deliberate** | structured |
|---|---|---|---|
| RAGAS faithfulness, **gpt-4.1 judge** | 0.805 | **0.890** | 0.859 |
| RAGAS answer relevancy, **gpt-4.1 judge** | **0.851** | 0.842 | **0.851** |
| RAGAS context precision / recall, gpt-4.1 judge | 0.844 / 0.933 | 0.836 / 0.933 | 0.819 / 0.933 |
| RAGAS faithfulness / answer relevancy, gpt-4o-mini judge | 0.787 / 0.803 | **0.795 / 0.825** | 0.740 / 0.764 |
| In-corpus questions wrongly answered "not found" (two runs) | 0/30, 0/30 | 0/30, 0/30 | 1/30 (GQ02), 0/30 |
| Out-of-corpus questions declined (fallback or off-topic block) | 3/4 | **4/4** | **4/4** |
| Asked for the regime / deduction method / amount when missing (tool T07–T09; kind = clarify) | 0/3 (asked in the text, but labelled as an answer) | **3/3** | **3/3** |
| Clarified when it should have answered (comparison requests T04–T06) | 0/3 | **0/3** | 3/3 |
| Tool-suite task completion | 13/17 | **17/17** | 14/17 |
| Guardrail suite (off-topic, injection and unsafe refusals) | 16/16 | 16/16 | 16/16 |
| Memory suite | 7/8 turns | 7/8 turns | 7/8 turns |

Runs: `evals/results/pc-{baseline,deliberate,structured}-{ragas,guard,memory,tools}/`, plus `pc-*-ragas-judge41/` for the gpt-4.1 judge.

**Judge.** The first RAGAS runs used gpt-4o-mini as judge, the same model as the generator, which risks it grading its own style generously. They were repeated with **gpt-4.1** (`python -m evals.run_ragas --judge gpt-4.1`, now the default). The repeat regenerates the answers too, so the two columns differ by judge *and* by run-to-run generation noise. With the stronger judge, `deliberate` leads clearly on faithfulness (0.890 vs 0.805 for baseline), the metric that matters most for a tax assistant. Answer relevancy is a near-tie across all three (0.842–0.851). The judge is still an OpenAI model, so it is stronger but not independent.

Context precision and recall are the same for all three because retrieval doesn't depend on the prompt. Guardrails and memory are also identical, because they live in code (input guards, profile extraction) rather than in the answer prompt. The single memory failure is M2 in all three: a retrieval-threshold edge case that falls back before the prompt is involved (see README → Limitations).

### What the differences look like

**Baseline invents an answer for a question the corpus doesn't cover (GQ-X4, crypto).**

> "In the Philippines, profits from cryptocurrency trading are generally treated as capital gains…"

None of the 7 issuances mentions crypto. It is the same failure as in v1 (below), and it gets through because the baseline answer carries a citation, which satisfies the citation check. Deliberate and structured both fell back to "not in my BIR sources".

**Baseline asks the right question but labels it as an answer (T07–T09).** For example: *"To calculate your tax, I need to know how you deduct your expenses. Do you use the OSD…?"* The wording is fine, but `type` is `answer`, because the baseline prompt never explains clarifying questions. The scorer counts these as failures. The code-level preconditions blocked every guessed calculation, so **no wrong figure reached the user under any prompt**. Safety for amounts lives in code, not in the prompt.

**Structured over-clarifies on comparison requests (T04–T06).** Asked *"Which is better for me, 8% or graduated? I earn about ₱900,000…"*, it replied *"Are you planning to use itemized deductions or the OSD?"*, and to T05 *"Are you on the 8% tax rate or the graduated rates?"*, which is the very thing the user asked TaxMate to decide. Reasoning step 2 ("is any profile field unknown?") pushed it to clarify before step 3 could pick `compare_regimes`. The explicit checklist made the model more literal, not smarter. Its one in-corpus fallback (GQ02) did not happen again in the second run, so treat it as noise. Its faithfulness is below `deliberate`'s under both judges.

### Why `deliberate` stays the default

`deliberate` is the only variant with no failure on any suite apart from the shared M2 threshold case:

- the highest faithfulness under both judges (0.890 with gpt-4.1), with answer relevancy within 0.01 of the best;
- 4/4 out-of-corpus questions declined;
- every missing input asked for, with no unnecessary questions;
- 17/17 tool tasks.

`baseline` invents tax rates when the sources are silent, and that is the worst failure for a tax assistant. `structured` costs extra output tokens for the hidden analysis and makes the model ask questions it shouldn't. If explicit reasoning is revisited, the checklist should put "what did the user ask for?" before "which profile fields are unknown?".

---

## History: the iteration-1 comparison (prompt v1, two variants)

Two system prompts (`taxmate/prompts.py`, `PROMPT_VERSION = v1`) ran through the **same** pipeline:
- the same retrieval (k = 4, threshold 0.47);
- the same guardrails and profile memory;
- the same 30 + 4 golden questions.

The only difference is the system prompt.

- **baseline**: one line: *"You are a Philippine tax assistant. Answer the user's question using the provided context. Cite sources as [S#]."*
- **deliberate**: 8 rules:
  1. answer only from the sources, otherwise say not found;
  2. cite every factual sentence;
  3. apply the profile (8% means no deductions; mixed income gets no ₱250k reduction);
  4. clarify only when the answer depends on an unknown profile field;
  5. no authoritative computations;
  6. prefer the newer issuance;
  7. 180 words or fewer;
  8. end with a CPA/RDO reminder.

### Results

| variant | faithfulness | answer relevancy | context precision | context recall | memory pass % | guardrail pass % | out-of-corpus declined | notes |
|---|---|---|---|---|---|---|---|---|
| baseline | **0.725** | **0.737** | 0.876 | 0.933 | 88% (7/8 turns) | 100% (16/16) | 3/4 (1 fallback + 2 blocked off-topic) | answered the crypto question with invented rates |
| deliberate | 0.643 | 0.440 | 0.856 | 0.933 | 88% (7/8 turns) | 100% (16/16) | **4/4** (2 fallback + 2 blocked off-topic) | answer relevancy depressed by rule 8 (see below) |

Runs: `evals/results/baseline-prompt/` and `evals/results/v1-deliberate/` (`results.csv`, `summary.json`), plus `guard-baseline`/`guard-v1` and `memory-baseline`/`memory-v1`.

**Baseline wins on the two answer-side RAGAS metrics.** Context precision and recall barely move, because retrieval is identical across variants. Memory and guardrail results are the same for both because both live in code (profile extraction, input guards), not in the answer prompt.

### What changed, with examples

**1. Deliberate refuses what the sources don't cover; baseline invents an answer (GQ-X4).**

*"What is the tax rate on profits from crypto trading in the Philippines?"* No crypto issuance is in the corpus.
- Baseline: *"…profits from cryptocurrency trading are generally treated as capital gains… typically 15%… [S1]… regular income tax rates… from 20% to 35% [S1]"*. These rates and the citation are invented: the cited source says nothing about crypto.
- Deliberate: the citation check failed, and the turn ended in the fixed fallback: *"I couldn't find guidance on that in my BIR sources…"*.

For a tax assistant, this is the most important behavioural difference.

**2. Deliberate's CPA/RDO line drags answer relevancy to 0 on correct answers (GQ03, GQ07, and 12 others).**

GQ03, *"How do I avail of the 8% income tax rate and when is the deadline to choose it?"*: the deliberate answer is correct and cited (Forms 1901/1905, on or before May 15, irrevocable), but it scored **0.00** relevancy. Baseline's near-identical content scored 0.74.

RAGAS answer relevancy gives 0 when the judge rates an answer *noncommittal*, and the standalone closing line *"Please verify with a CPA or your RDO."* reads as hedging. Supporting evidence: GQ15, where the reminder is folded into a sentence, scored 0.94. 14 of the 30 deliberate answers scored exactly 0.00; baseline had 2. Fixing this is the targeted improvement in `evaluation.md`.

**3. Baseline once fell back on an in-corpus question (GQ01).**

*"Who is not qualified to use the 8% income tax rate?"*: baseline answered without valid citations twice, so the citation check replaced it with the fallback (faithfulness 0.25, relevancy 0). Deliberate answered with the full cited list from RMC 50-2018 (relevancy 0.96).

### Why `deliberate` stayed the default

It is the only variant that declined all 4 out-of-corpus questions instead of inventing tax rates with fake citations. Its lower RAGAS answer scores come mostly from the CPA/RDO reminder line, which is a measured, fixable artifact, not from wrong content. A confident wrong answer is the worst outcome for a tax assistant.

### Update: v2

Rule 8 was removed in `PROMPT_VERSION = v2`; the CPA/RDO reminder is now a fixed UI caption under each answer. Deliberate v2 scores:
- faithfulness **0.834**;
- answer relevancy **0.799**;
- context precision 0.879;
- context recall 0.933.

That beats baseline on every metric, while still declining 4/4 out-of-corpus questions. Full before/after: `evaluation.md` → *Targeted improvement*.

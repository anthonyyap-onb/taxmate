# TaxMate — Prompt Comparison

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

## Results

| variant | faithfulness | answer relevancy | context precision | context recall | memory pass % | guardrail pass % | out-of-corpus declined | notes |
|---|---|---|---|---|---|---|---|---|
| baseline | **0.725** | **0.737** | 0.876 | 0.933 | 88% (7/8 turns) | 100% (16/16) | 3/4 (1 fallback + 2 blocked off-topic) | answered the crypto question with invented rates |
| deliberate | 0.643 | 0.440 | 0.856 | 0.933 | 88% (7/8 turns) | 100% (16/16) | **4/4** (2 fallback + 2 blocked off-topic) | answer relevancy depressed by rule 8 (see below) |

Runs: `evals/results/baseline-prompt/` and `evals/results/v1-deliberate/` (`results.csv`, `summary.json`), plus `guard-baseline`/`guard-v1` and `memory-baseline`/`memory-v1`.

**Baseline wins on the two answer-side RAGAS metrics.** Context precision and recall barely move, because retrieval is identical across variants. Memory and guardrail results are the same for both because both live in code (profile extraction, input guards), not in the answer prompt.

## What changed, with examples

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

## Why `deliberate` stays the default

It is the only variant that declined all 4 out-of-corpus questions instead of inventing tax rates with fake citations. Its lower RAGAS answer scores come mostly from the CPA/RDO reminder line, which is a measured, fixable artifact, not from wrong content. A confident wrong answer is the worst outcome for a tax assistant.

## Update: v2

Rule 8 was removed in `PROMPT_VERSION = v2`; the CPA/RDO reminder is now a fixed UI caption under each answer. Deliberate v2 scores:
- faithfulness **0.834**;
- answer relevancy **0.799**;
- context precision 0.879;
- context recall 0.933.

That beats baseline on every metric, while still declining 4/4 out-of-corpus questions. Full before/after: `evaluation.md` → *Targeted improvement*.

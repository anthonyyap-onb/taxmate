# TaxMate — Memory

## Design

TaxMate keeps two kinds of session memory, both in Streamlit session state and both cleared by **Start over**:

1. **Conversation history.** The last 10 turns go to the model, and follow-up questions are rewritten into standalone retrieval queries ("what about the deadline?" becomes a full question). The history stored is the PII-masked version.
2. **A structured tax profile**, shown in the sidebar: taxpayer type, tax regime, income source, gross receipts under ₱3M, and deduction method. Every field starts as *Unknown*.
   - Each turn, gpt-4o-mini extracts only what the user explicitly states about themselves. Corrections overwrite.
   - The profile goes into the answer prompt, so later answers apply it, e.g. no deductions on 8%, or no ₱250k reduction for mixed income.

### Extraction is quote-grounded

The first live smoke test caught the extractor guessing:
- *"Can I use the 8% option?"*, a question, set the regime to 8%;
- an intro mentioning ₱1.2M income also set the deduction method to OSD.

On a 21-message probe (7 messages × 3 runs), **9/21** extractions were wrong, and the guesses varied from run to run even at temperature 0. A stricter prompt alone did not fix it.

The extractor now returns a list of facts, each with the **user's exact words as a quote**. Code (`facts_to_update` in `taxmate/profile.py`) drops any fact whose quote doesn't appear verbatim in the message, and any value that isn't valid for its field. On the same probe: **0/21** wrong. Unit tests: `test_fact_whose_quote_is_not_in_message_is_dropped` and related.

When a message only states profile facts and retrieval finds nothing relevant (*"I'm a freelance writer on the 8% rate."*), TaxMate acknowledges the update instead of saying it couldn't find guidance.

## Scenario results

`python -m evals.run_memory_suite --run memory-v1` (`evals/memory_scenarios.json`), results in `evals/results/memory-v1/results.csv`:

| Scenario | What it checks | Result |
|---|---|---|
| M1 | Regime stated in turn 1 (8%), an unrelated question in turn 2, then *"Can I deduct my laptop?"* in turn 3 | ✅ 3/3 turns. Profile kept 8% throughout; the answer says expenses aren't deductible under 8% |
| M2 | Unknown regime + *"Can I deduct my office rent?"* should get a clarifying question about 8% vs graduated | ❌ fell back instead: top retrieval score 0.446 is below the 0.47 threshold, so the model never saw the question |
| M3 | *"I'm on the 8% rate"*, then *"Sorry, actually I'm on graduated rates"* | ✅ 2/2. Profile flipped to Graduated |
| M4 | *"I have a full-time job and also freelance"*, then *"Can I subtract the 250,000 first?"* | ✅ 2/2. Profile = Mixed income; answer says the ₱250k reduction doesn't apply |

**3/4 scenarios, 7/8 turns**; the baseline prompt gave the same result. The profile logic is in code, so both prompts receive the same profile.

The M2 failure is in retrieval, not memory: the extractor correctly left the regime *Unknown*. The cause is the same threshold limitation recorded in `evaluation.md`, where the threshold was set from golden-set scores and phrasing like "office rent" lands just under it.

# TaxMate — Calculator Tools & Tool-Use Evaluation

## Why a calculator

Iteration 1 could explain the rules, but rule 5 of its prompt forbade computing amounts, because LLM arithmetic is not reliable. Freelancers mostly want the number, though ("how much will I pay?", "is 8% better for me?"). TaxMate now does the math in deterministic Python, and the model's only job is to choose the tool and fill in its arguments.

## The tools (`taxmate/calculator.py`, `taxmate/tools.py`)

| Tool | Use | Returns |
|---|---|---|
| `calculate_ph_tax(gross_receipts, regime, deduction_method, itemized_expenses, non_operating_income, compensation_income)` | tax on a **known** regime | income tax, 3% percentage tax, total, line-by-line breakdown, notes |
| `compare_regimes(gross_receipts, itemized_expenses, non_operating_income, compensation_income)` | "which is better for me?" | total under 8%, graduated + OSD, and graduated + itemized (only if expenses are given), plus the lowest |

Every rule is taken from the corpus and cited in the module docstring:

| Rule | Source |
|---|---|
| Graduated table effective 1 Jan 2023 (0 / 15% / 20% / 25% / 30% / 35%; base amounts 22,500 / 102,500 / 402,500 / 2,202,500) | RR 8-2018 |
| 8% × (gross sales/receipts + other non-operating income − ₱250,000) | RMC 50-2018 Q22 |
| The ₱250,000 reduction applies to **purely** self-employed individuals only, not mixed-income earners | RMC 50-2018 Q23, RR 8-2018 |
| 8% is available only if gross + non-operating income ≤ ₱3,000,000 | RR 8-2018 |
| 8% replaces both the graduated rates **and** the 3% percentage tax | RMC 50-2018 Q21 |
| Compensation income is always taxed at graduated rates | RMC 50-2018 Q25 |
| OSD = 40% of gross sales/receipts | RR 8-2018, 1701Q guidelines |
| Graduated non-VAT filers still pay 3% percentage tax (2551Q) | RR 8-2018, 2551Q guidelines |

`compare_regimes` includes the 3% percentage tax on the graduated side. Leaving it out is the most common mistake in DIY comparisons, and it often decides which option wins.

**Scope (deliberate):**

- Annual figures only. The quarterly 1701Q computation is cumulative (RMC 50-2018 Q24) and needs the tax already paid in earlier quarters, which the user rarely has to hand.
- No tax credits (Form 2307 withholding).
- Above ₱3M, VAT is noted but not computed.

The tool's notes and the UI card both say this.

**Unit tests:** `tests/test_calculator.py` has 29 tests with hand-computed values, covering every bracket boundary and each of the rules above.

## How the agent uses them

- **Function calling.** The generator (`taxmate/generation.py`) offers both tools as strict OpenAI function schemas. It runs up to 3 tool rounds, and the final round must produce an answer. Each tool execution is a Langfuse span (`tool`), and the log line records `tool_calls` (name, numeric args, ok/error code).
- **Prompt v4, rule 5.** Never do arithmetic; use the tools; ask when the regime, deduction method or amount is missing; present tool figures as estimates.
- **Preconditions enforced in code (`taxmate/tools.py::check_call`).** The prompt asks the model not to guess. The code makes sure it can't:

  | Check | Blocks |
  |---|---|
  | `regime_not_stated` | `calculate_ph_tax` when the profile's regime is unknown and the current message doesn't name it |
  | `deduction_method_not_stated` | a graduated calculation when the profile's deduction method is unknown and the message doesn't name OSD or itemized |
  | `amount_not_stated` | any peso argument that doesn't match an amount the user wrote ("1.2M", "900k", "₱1,500,000" all count), e.g. a monthly figure the model multiplied by 12 itself |
  | `negative_amount` | an amount the user wrote as negative, even if the model silently dropped the sign |

  A blocked call returns a structured error, never an exception, and the model turns it into one clarifying question. A malformed argument (wrong type, unparseable JSON) gets the same treatment as `invalid_arguments`.
- **Pipeline (`taxmate/pipeline.py`).**
  - An answer grounded in a tool call passes the citation check (`citation_check: "tool"` or `"retry_tool"`).
  - If retrieval is weak and the message looks like a calculation (a peso amount, "how much", "magkano", "which is cheaper"…), the model gets the tools with **no** sources. The result is kept only if it used a tool or asked for a missing input; otherwise the turn falls back as before. Form numbers (2307, 1701Q) and years are not treated as amounts.
- **UI.** A calculator result is shown as a card of `st.metric` values (total, income tax, percentage tax; or one metric per option with the lowest marked), with a "How this was computed" breakdown table under it.

## Evaluation (`python -m evals.run_tool_suite --run NAME`)

`evals/tool_scenarios.json` has 17 scenarios. Only the last turn is scored, and the earlier turns build memory. The expected figures are checked against the calculator by `tests/test_eval_assets.py`.

| Category | Scenarios | Expected behaviour |
|---|---|---|
| Straight calculation: 8%, graduated + OSD, graduated + itemized, mixed income | T01, T02, T03, T17 | right tool, right args, correct figure in the reply |
| Comparison | T04, T05, T06 | `compare_regimes`, lowest option named |
| Regime unknown | T07 | **no** calculation; ask 8% or graduated |
| Missing input (deduction method, amount) | T08, T09 | no calculation; ask |
| Malformed (negative amount) | T10 | clarifying question |
| Needs RAG, not the tool | T11, T12, T13 | no tool call |
| Profile from memory (stated 1–2 turns earlier) | T14, T15 | uses the remembered regime / OSD |
| Above ₱3M on 8% | T16 | explains that 8% isn't available |

**Metrics**

- **Tool selection (system):** the first tool call that actually ran matches the expected tool, or no tool where none is expected.
- **Tool selection (model alone):** the same check on the model's first *attempted* call, including calls the preconditions blocked.
- **Argument accuracy:** for scenarios where a tool is expected and the right one ran, the arguments match (within ₱1; a null optional amount counts as 0).
- **Task completion:** tool, arguments, reply kind and reply content all correct.

### Results

| Run | What changed | Tool selection (system) | Model alone | Precondition blocks | Arguments | Task completion | Failed |
|---|---|---|---|---|---|---|---|
| `tools-v1` | tools + prompt only | 15/17 (88%) | n/a | n/a | 9/9 | 14/17 (82%) | T07, T08, T10 |
| `tools-v2` | + code preconditions | 17/17 | 14/17 | 5 | 9/9 | 17/17 | none |
| `tools-v2-r2` | (repeat) | 17/17 | 14/17 | 6 | 9/9 | 15/17 | T05, T16 |
| `tools-v2-r3` | (repeat) | 17/17 | 14/17 | 6 | 9/9 | 16/17 | T05 |
| `tools-v3-r1` | + retry fix, comparison hint | 17/17 | 15/17 | 4 | 9/9 | **17/17** | none |
| `tools-v3-r2` | (repeat) | 17/17 | 16/17 | 3 | 9/9 | **17/17** | none |
| `tools-v3-r3` | (repeat) | 17/17 | 15/17 | 4 | 9/9 | **17/17** | none |

The final version completes all 17 scenarios in 3 of 3 runs. On its own, the model still tries to guess the regime (T07) or the deduction method (T08) in most runs. The gap between "model alone" and "system" is exactly what the code preconditions catch.

### Failures found and fixed

1. **The model guessed inputs the user never gave (`tools-v1`).**
   - T07 ("I earned ₱800,000, how much tax?"): it calculated under both regimes instead of asking.
   - T08: it assumed OSD.
   - T10: the user typed "-₱500,000" and it passed 500,000.

   The prompt already told it to ask, so instead of rewording the prompt again I added the code preconditions above. In the next run the model also passed `gross_receipts: 0` when no amount was given (T09). A zero gross amount now has to be stated as well.
2. **A correct tool answer was replaced by the fallback (`tools-v2-r2`, T16).** The model's first answer had no citations. On the strict retry it called the calculator ("8% isn't available above ₱3M"), but the retry branch only accepted `[S#]` citations. **Fix:** a retry grounded in a tool call is accepted (`retry_tool`). Regression test: `tests/test_pipeline.py::test_citation_retry_that_uses_the_calculator_is_accepted`.
3. **A blocked call after a successful comparison (T05).** After `compare_regimes` succeeded, the model also tried a single-regime calculation. That call was correctly blocked, and the model then asked "8% or graduated?" instead of answering. **Fix:** the `regime_not_stated` error now tells the model to answer from an existing comparison.

## Regression check on earlier suites (prompt v4)

| Suite | v3 | v4 (with tools) |
|---|---|---|
| Guardrails (`guard-v4`) | 16/16 | 16/16 |
| Memory (`memory-v4`) | 3/4 (M2 fails) | 3/4 (M2 fails; same cause, see below) |
| RAGAS faithfulness / answer relevancy | 0.797 / 0.791 | 0.804 / 0.825 |
| RAGAS context precision / recall | 0.890 / 0.967 | 0.879 / 0.967 |
| In-corpus fallback rate / out-of-corpus fallback rate | 3.3% / 50% | 0% / 50% |

**M2 ("Can I deduct my office rent?") is a known limitation from iteration 1.** Its best chunk scores 0.4458, just under the 0.47 relevance threshold (`threshold-v1`), so it falls back before the model can ask for the regime. It is not caused by the tools.

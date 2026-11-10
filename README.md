# TaxMate

TaxMate is a chat assistant for Philippine freelancers and sole proprietors. It answers questions about the 8% vs. graduated income tax, deductions, Form 2307, percentage tax, and quarterly filing **only from official BIR issuances**, cites its sources, and remembers your tax profile during the conversation.

> Information only. TaxMate is not legal or CPA advice.

**Live app:** _added after deployment_

## How it works (iteration 1)

Each message goes through one guarded pipeline (`taxmate/pipeline.py`):

1. **Validation.** Empty messages or messages over 2,000 characters are rejected.
2. **PII masking.** TINs, emails, and PH phone numbers are replaced before anything else sees the text.
3. **Input guardrails.** An injection-phrase heuristic runs first, then OpenAI moderation, then a gpt-4o-mini topic classifier (on/off-topic, injection, unsafe, smalltalk).
4. **Profile update.** Structured extraction of taxpayer type, regime, income source, ₱3M status, and deduction method. The profile is shown in the sidebar.
5. **Retrieval.** History-aware query rewrite, then FAISS search over 247 chunks from 7 BIR documents. Below the relevance threshold, TaxMate answers "not found in my BIR sources". For filing/payment-venue questions, the 2024 EOPT "File and Pay Anywhere" rule is always included and marked as superseding the older 2018 rules.
6. **Generation.** Structured output: an answer with `[S#]` citations, or a clarifying question when a needed profile field is unknown.
7. **Output checks.** A citation check (one strict retry, then fallback) and output moderation.
8. **Logging and tracing.** One JSON log line per turn, plus a Langfuse trace.

LangChain is used only for text splitting. The pipeline is plain Python so every step is unit-testable with fakes and visible as its own span in Langfuse.

## Knowledge base

TaxMate answers only from these public BIR issuances:

- Revenue Regulations No. 8-2018 (TRAIN income tax provisions: 8% option, graduated rates, OSD)
- Revenue Memorandum Circular No. 50-2018 (Q&A clarifying RR 8-2018 and RR 11-2018)
- Revenue Regulations No. 11-2018 (withholding tax, sworn declarations, Form 2307)
- Guidelines and Instructions for BIR Form 1701Q (quarterly income tax return)
- Guidelines and Instructions for BIR Form 1701-MS (annual income tax return for micro/small taxpayers)
- Guidelines and Instructions for BIR Form 2551Q (quarterly percentage tax return)
- Ease of Paying Taxes (EOPT) Act (RA 11976) flyer

All are available from the BIR website (bir.gov.ph). The source PDFs are in `bir-data/`. Their extracted text is in `corpus/` (one JSONL file per document), and that is all the app reads at runtime. RMC 50-2018's PDF has an unusable text layer, so it was transcribed manually and verified against the original (`bir-data/transcribed/RMC_No_50-2018.md`).

To rebuild `corpus/` after changing the source documents :

```bash
uv run python -m taxmate.corpus build
```

## Setup (local)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.12
uv pip install -r requirements-eval.txt   # app + evaluation deps (requirements.txt = app only)
cp .env.example .env                      # then fill in OPENAI_API_KEY (+ Langfuse keys for tracing)
uv run streamlit run app.py
```

Without `OPENAI_API_KEY`, the app starts and shows a configuration message.

## Tests

```bash
uv run pytest
```

The unit tests need no API keys; every OpenAI call sits behind injectable fakes.

## Evaluation

All runners use the same pipeline as the app and tag their Langfuse traces with `run:<NAME>`. Results are written to `evals/results/<NAME>/`.

```bash
uv run python -m evals.tune_threshold --run threshold-v1
uv run python -m evals.run_ragas --run v1-deliberate --variant deliberate
uv run python -m evals.run_guardrail_suite --run guard-v1
uv run python -m evals.run_memory_suite --run memory-v1
```

## Evidence docs

- [Evaluation method & results](evals/docs/evaluation.md)
- [Prompt comparison](evals/docs/prompt-comparison.md)
- [Guardrails](evals/docs/guardrails.md)
- [Memory](evals/docs/memory.md)
- [Debugged failure](evals/docs/debugged-failure.md)
- [Deploy guide](docs/deploy.md)
- [Design](docs/superpowers/specs/2026-11-10-taxmate-design.md) and [iteration-2 backlog](docs/iteration-2-backlog.md)

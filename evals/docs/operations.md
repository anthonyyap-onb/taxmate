# TaxMate — Operational Readiness

## Secrets

- Keys are read only from environment variables, `.env` (local) or `st.secrets` (Streamlit Cloud), in `taxmate/config.py`. No key is in the code.
- Placeholder templates are committed: `.env.example` and `.streamlit/secrets.toml.example`.
- `.env` and `.streamlit/secrets.toml` are in `.gitignore`.
- History scan before submission:

  ```bash
  git log -p --all | grep -oE "sk-(proj-)?[A-Za-z0-9_-]{20,}|sk-lf-[a-f0-9-]{8,}|pk-lf-[a-f0-9-]{8,}"
  ```

  This returns no matches. A looser `grep -i "sk-"` only matches "ri**sk-b**ased" in the EOPT flyer text.

## Structured logging

Every turn writes **one JSON line to stdout** (`taxmate/logging_utils.py`). It shows up in Streamlit Cloud under *Manage app → Logs*.

| Field | Meaning |
|---|---|
| `request_id`, `session_id`, `ts` | random UUIDs and a UTC timestamp. The session ID is random per browser session, not the user's identity. |
| `user_message_masked`, `pii_masked` | the question **after** PII masking (TIN, email, PH phone), plus which kinds were masked. The raw question is never logged. |
| `latency_ms` | `total`, broken down into `guard` / `profile` / `retrieval` / `generation` |
| `model`, `prompt_version`, `prompt_variant` | what produced the answer |
| `tokens`, `est_cost_usd` | prompt / completion / embedding tokens summed over every OpenAI call in the turn, and the cost at list prices (`taxmate/usage.py`). Moderation calls are free and report no tokens. |
| `topic_category`, `moderation_flagged`, `blocked`, `block_reason`, `citation_check`, `fallback` | guardrail verdicts |
| `n_chunks`, `top_score`, `retrieved` | retrieval result: chunk IDs and scores only, no chunk text |
| `kind` | `answer` / `clarify` / `smalltalk` / `blocked` / `unsafe` / `fallback` / `error` / `invalid` |
| `error` | the exception **type** only (e.g. `NotFoundError`). There is no message or stack trace, so no user text can leak through it. |
| `langfuse_trace_id` | joins the log line to its Langfuse trace |

## Error handling

- **Every turn** runs inside one `try/except` in `taxmate/pipeline.py`. Any failure in the classifier, profile extraction, retrieval or generation gives `kind: error`. The user sees the friendly `ERROR_TEXT`, and the exception type is logged.
- **Startup**: the first message builds the vector index (`get_deps()` in `app.py`). If OpenAI is unreachable then, the user gets the same friendly message and a log line with `"stage": "startup"`. A failed build is not cached, so the next message retries it.
- **Timeouts**: the OpenAI client uses a 30 s timeout with 2 retries (`taxmate/tracing.py`). The SDK default is 600 s.
- **Tracing never breaks a turn**: every Langfuse call is wrapped and does nothing on failure (`taxmate/tracing.py`).
- The UI never shows a traceback. This is tested in `tests/test_app.py::test_startup_failure_shows_friendly_error_not_traceback` and `tests/test_pipeline.py::test_dep_exception_returns_error`.

## Log excerpt (real run)

The three lines below were captured with `uv run python -m evals.capture_ops_logs` and saved in full in `evals/results/ops-logs/turns.jsonl`. To keep them short, only the first entry of `retrieved` is shown.

**Normal request.** It contains a TIN, which is masked before any model call, log or trace:

```json
{"ts": "2026-11-10T13:30:05.185998+00:00", "request_id": "e4f9fb88-4779-4af8-b2c9-be9796ac3881", "session_id": "ops-evidence-normal", "model": "gpt-4o-mini", "prompt_version": "v3", "prompt_variant": "deliberate", "pii_masked": ["tin"], "moderation_flagged": false, "topic_category": "on_topic", "blocked": false, "block_reason": null, "n_chunks": 4, "top_score": 0.5544, "fallback": false, "citation_check": "pass", "langfuse_trace_id": "4ef34f43a30583503235be2895811091", "error": null, "user_message_masked": "My TIN is [TIN]. I'm a freelancer on the 8% rate, what is Form 2307 for?", "kind": "answer", "retrieved": [{"chunk_id": "1701MS-GUIDE#p2#18", "score": 0.5544}, "…"], "latency_ms": {"total": 6914, "guard": 2579, "profile": 1190, "retrieval": 189, "generation": 2955}, "tokens": {"prompt": 2814, "completion": 203, "embedding": 27}, "est_cost_usd": 0.00054444}
```

**Blocked request.** The topic classifier blocks it, so retrieval and generation never run:

```json
{"ts": "2026-11-10T13:30:12.101729+00:00", "request_id": "6d92d746-c85e-4ed8-8bbc-d4b4c8e81e22", "session_id": "ops-evidence-blocked", "model": "gpt-4o-mini", "prompt_version": "v3", "prompt_variant": "deliberate", "pii_masked": [], "moderation_flagged": false, "topic_category": "off_topic", "blocked": true, "block_reason": "classifier", "n_chunks": 0, "top_score": null, "fallback": false, "citation_check": "skipped", "langfuse_trace_id": "baa2e912f3e265bca5d1515691c9f02a", "error": null, "user_message_masked": "What is the best adobo recipe?", "kind": "blocked", "retrieved": [], "latency_ms": {"total": 1416, "guard": 1416, "profile": 0, "retrieval": 0, "generation": 0}, "tokens": {"prompt": 356, "completion": 27, "embedding": 0}, "est_cost_usd": 6.96e-05}
```

**Error.** The generator was pointed at a model that doesn't exist, so the OpenAI API returned a real 404. The user sees "TaxMate is having trouble reaching its AI service right now. Please try again in a moment.":

```json
{"ts": "2026-11-10T13:30:13.518986+00:00", "request_id": "e410fc12-db36-4800-b153-beb54a1f57b4", "session_id": "ops-evidence-error", "model": "gpt-4o-mini", "prompt_version": "v3", "prompt_variant": "deliberate", "pii_masked": [], "moderation_flagged": false, "topic_category": "on_topic", "blocked": false, "block_reason": null, "n_chunks": 4, "top_score": 0.5296, "fallback": false, "citation_check": "skipped", "langfuse_trace_id": "c6dfb16bdd0e1ff64f77b8d05e636371", "error": "NotFoundError", "user_message_masked": "What is the deadline for filing 1701Q?", "kind": "error", "retrieved": [{"chunk_id": "RMC-50-2018#Q55#0", "score": 0.5296}, "…"], "latency_ms": {"total": 3494, "guard": 1334, "profile": 1299, "retrieval": 228, "generation": 631}, "tokens": {"prompt": 1041, "completion": 34, "embedding": 11}, "est_cost_usd": 0.00017677}
```

A normal answered turn costs about **$0.0005** (≈ ₱0.03) and takes about 7 s. The guard and generation stages take most of that time.

## Bug found while capturing this evidence

The first capture run logged `"error": "NameError"` on every on-topic turn. `taxmate/profile.py` called `usage.record()` without importing `usage`. The unit tests had not caught it because they replace the profile extractor with a fake.

- **Fix:** add the import.
- **Regression test:** `tests/test_usage.py::test_every_openai_call_site_records_usage` drives every real OpenAI call site with a fake client. The turn still failed safely during the bug: friendly message, type-only error in the log.

# TaxMate — Caching

## What is cached (`taxmate/cache.py`)

| Layer | What | Where it lives | Saves |
|---|---|---|---|
| Document index | the 247 chunk vectors | `.cache/index-<fingerprint>.npy` on disk | re-embedding the corpus on every cold start |
| Query embeddings | one vector per distinct query text (LRU, 2,000 entries) | memory | a second embedding API call for the same query, e.g. the semantic lookup followed by retrieval |
| Responses (exact) | final answer or clarifying question, its cited sources and calculator results | memory (LRU, 500 entries), shared by all sessions in the app process | retrieval + generation |
| Responses (semantic) | the same entries, matched by query-embedding cosine ≥ 0.95 | memory | retrieval + generation for a paraphrase |

## Where the cache sits in the pipeline

```
validate → mask PII → input guardrails → profile update → query rewrite → [response cache] → retrieve → generate (+ tools) → output checks → store
```

- **Guardrails come before the cache.** A blocked message never looks the cache up (logged as `"cache": "off"`).
- **Only turns that passed every output check are stored:** answers that passed the citation check (or are grounded in a calculator result) and output moderation, plus clarifying questions. Fallbacks, refusals and errors are never stored. Tested in `tests/test_pipeline.py::test_only_turns_that_passed_every_check_are_stored`.
- **The key is the standalone (rewritten) query, not the raw text.** In a conversation, "Is it required?" is keyed on what "it" refers to. A follow-up never hits an entry for an unrelated question with the same words.
- **The profile is in the key.** "Can I deduct my internet bill?" has different correct answers for 8% and graduated filers. Caching on the question alone would serve the wrong one. The key also includes the model, prompt version, prompt variant and calculator version, plus the corpus fingerprint as the cache namespace.
- **A cache failure never breaks a turn.** The lookup or store is skipped and logged as `"cache": "error"`.

## Semantic cache: why a threshold is not enough

The benchmark embeds 10 paraphrase pairs (same meaning) and 6 near-miss pairs (almost the same words, different correct answer):

| | Cosine similarities (text-embedding-3-small) |
|---|---|
| Paraphrases | 0.586, 0.814, 0.896, 0.911, 0.912, 0.931, 0.946, 0.951, 0.978, 0.979 |
| Near misses | 0.763, 0.781, 0.796, 0.826, 0.911, **0.960** |

The ranges overlap, so **no threshold separates them**. The most similar pair (0.960) is "Should I itemize or use OSD with graduated rates?" vs "Should I use OSD with graduated rates?", which have different correct answers.

So a semantic hit also needs the **salient facts** of both questions to be identical: peso amounts and percentages, form numbers (2307, 1701Q), quarters (Q1–Q4), 8% / graduated, OSD / itemized, purely / mixed. With that check, all **6/6 near misses are kept apart**, and the threshold stays at a conservative 0.95.

**Bug found by this benchmark.** In the first run (`cache-v1`), the itemize pair got through: the pattern matched "itemized"/"itemised" but not "itemize". The same pattern is used by the calculator's precondition check, which therefore also missed "I itemize". It now matches every form of the word. Regression tests: `tests/test_cache.py::test_salient_catches_itemize_verb_forms` and `tests/test_tools.py::test_itemize_verb_counts_as_stating_itemized`.

## Before / after (`python -m evals.run_cache_bench --run cache-v2`)

These are live runs through the app's pipeline (gpt-4o-mini, text-embedding-3-small) on 10 golden-set questions. Raw rows are in `evals/results/cache-v2/turns.csv`, and `cache-v1` gives the same numbers within noise.

| Query type | n | Cache result | Median latency (ms) | Mean est. cost (USD) | Mean prompt tokens |
|---|---|---|---|---|---|
| First ask (no cache) | 10 | 10 miss | 4,684 | $0.000580 | 3,386 |
| Exact repeat | 10 | **10 exact hit** | **2,247** (−52%) | **$0.000179** (−69%) | 1,048 |
| Paraphrase | 10 | **3 semantic hit**, 7 miss | hits: **2,556**; misses: 5,020 | hits: **$0.000178**; misses: $0.000571 | 1,046 / 3,295 |
| Same question, different regime/deduction profile | 3 | **3 miss (correct)** | 3,871 | $0.000415 | 2,494 |

| Startup | Embedding tokens | Time | Est. cost |
|---|---|---|---|
| Cold (embed 247 chunks) | 56,707 | 2,531–3,459 ms | $0.00113 |
| Warm (load saved vectors) | 0 | **2.6–3.9 ms** | $0 |

**Why a hit still takes about 2.2 s.** A hit skips retrieval and generation, but the input guardrails (moderation + topic classifier) and the profile update still run before the lookup. These are safety and correctness requirements, not overhead: guardrails must come first, and the profile is part of the key. The next speed-up would be to run the topic classifier and the profile extraction in parallel, since they don't depend on each other.

The paraphrase hit rate (3/10) is deliberately low. Exact repeats are the common case: the suggested-question chips, students asking the same deadline question, and a user re-asking after a refresh. A semantic false hit would give a wrong answer, while a miss only costs about 2.5 s.

## Invalidation policy

| Change | Effect |
|---|---|
| Corpus text or embedding model changes | new corpus fingerprint → a new index file is built, and the response cache gets a new namespace, so nothing from the old corpus is used |
| Chat model, prompt version, prompt variant or calculator version changes | part of every key, so old entries are never hit again (`CALCULATOR_VERSION` in `taxmate/calculator.py` must be bumped when a rule changes) |
| Time | entries expire after 7 days (`TAXMATE_CACHE_TTL_DAYS`), a backstop for rules that change outside the corpus |
| Manual | with `TAXMATE_ADMIN=1`, the sidebar shows cache stats and a **Clear cache** button. Normal users never see it. |
| App restart | the response cache is in memory and starts empty. On Streamlit Community Cloud the disk is reset on reboot, so the first visitor after a reboot pays the 2.5–3.5 s index build once. |

**Evaluation runs never use the response cache** (`build_live_deps(..., response_cache=False)` by default). Otherwise a repeated golden question would be scored on a replayed answer. The index cache is used everywhere, because the vectors are identical either way.

## Logging

Each turn's JSON log line carries:

- `cache`: `off` / `miss` / `exact` / `semantic` / `error`
- `cache_similarity`
- a `cache` stage in `latency_ms`

Startup writes one line: `{"event": "startup", "index": "loaded" | "built", "index_ms": …, "corpus_fingerprint": …}`. Langfuse span metadata includes `cache`.

## Privacy note

The response cache is shared across sessions, so entries hold only PII-masked text: the question is masked before the key is built, and answers are generated from masked input. An exact hit can only return an answer to text the new user typed themselves. A semantic hit could in theory repeat a detail the original asker wrote that isn't a TIN, email or phone number (e.g. a name). That risk is part of why the threshold stays at 0.95. Set `TAXMATE_SEMANTIC_THRESHOLD=off` for exact-match only.

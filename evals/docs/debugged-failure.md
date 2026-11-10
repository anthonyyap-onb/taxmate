# TaxMate — Debugged Failure: a superseded filing rule presented as current

## The failure

**Question (GQ-EOPT):** *"Where can I file and pay my quarterly income tax return?"*

**Before** (run `v1-deliberate`, Langfuse trace [`61777b7b…`](https://jp.cloud.langfuse.com/project/cmwa7yq5t005zad0lze6lsfai/traces/61777b7b8623ffdad6e9c98fba92b185), screenshot `logs/failure-before.png`):

> You can file and pay your quarterly income tax return at any Authorized Agent Bank (AAB) located within the jurisdiction of the Revenue District Office (RDO) where you are registered or where you have your legal residence or place of business. If there are no AABs in your area, you should file with the Revenue Collection Officer of the RDO where you are registered or conducting business [S1].

That was the 2018 rule from the BIR Form 1701Q guidelines. The **Ease of Paying Taxes (EOPT) Act**, effective January 22, 2024 and included in our corpus, changed it: taxpayers can now *"file their tax returns and pay their taxes, either electronically or manually, with any Authorized Agent Bank, Revenue District Office…, or Authorized Tax Software Provider… without worries of incurring administrative penalties for filing at the wrong venue."*

The answer was well cited and fluent, and it scored faithfulness 0.80 and answer relevancy 0.98, but it was **out of date**. Only context recall (0.00) flagged it. A user following it would think they must find a bank in their own RDO.

## Root cause, from the trace

The trace's retrieval span showed the four sources the model received:

| Rank | Chunk | Score | Issued |
|---|---|---|---|
| S1 | `1701Q-GUIDE#p1#2` (filing venue, 2018) | 0.680 | 2018-01 |
| S2 | `RR-11-2018#p20#46` (where returns are filed) | 0.565 | 2018-01-31 |
| S3 | `RMC-50-2018#Q55#0` | 0.564 | 2018-05-11 |
| S4 | `2551Q-GUIDE#p1#2` | 0.556 | 2018-01 |

**No EOPT text was in the context**, so the model answered correctly from what it had, and the error came from retrieval. Ranking every chunk against the query showed the EOPT flyer's relevant chunk at **rank 59 (score 0.371)**.

Why: the flyer is a list of short headed topics, but the generic splitter cut it into ~1,150-character windows. The "File and Pay Anywhere" paragraph shared a chunk with the taxpayer-size thresholds table and the penalty concessions, so the chunk's embedding was dominated by *taxpayer classification*, not *where to file*.

## The fix: three layers, each tested before it was implemented

**1. Chunk the EOPT flyer by its own section headings** (`taxmate/corpus.py`: `_split_on_headings`, `_heading_chunks`; the `headings` list in `DOCS`).
- Each topic now gets its own chunk: 18 chunks instead of 9.
- Tests: `test_heading_split_gives_one_chunk_per_section`, `test_eopt_flyer_file_and_pay_anywhere_is_its_own_chunk`.
- **Effect:** rank 59 → **rank 12** (score 0.371 → 0.496). Better, but still outside the top 4, because the 2018 form guides match "quarterly income tax return" more closely than a general EOPT flyer ever will.

**2. A supersession rule in the retriever** (`taxmate/retrieval.py`: `SUPERSESSION_RULES`).
- When a query is about filing or paying *and* about a venue (where / bank / AAB / RDO / anywhere…), the EOPT "File and Pay Anywhere" chunk is always included, as the last of the k sources, scored by its real similarity.
- It carries a note: *"this 2024 EOPT Act rule supersedes older (2018) rules on where to file returns and pay taxes"*.
- The top-ranked source is untouched, so the relevance threshold still works as before.
- Tests: `test_filing_venue_question_pins_superseding_eopt_chunk_last`, `test_unrelated_question_is_not_pinned`, `test_pinned_chunk_not_duplicated_when_already_retrieved`, `test_supersession_rule_targets_existing_eopt_chunk`.

**3. Show the note to the model and sharpen prompt rule 6** (`taxmate/prompts.py`, `PROMPT_VERSION = v3`).

With the EOPT chunk retrieved as S4 but unlabelled, the model **still** answered from the 2018 S1 text in 3/3 trials. The old rule 6 (*"when sources conflict, prefer the more recent issued date"*) was too abstract: the model didn't read "an AAB within your RDO" and "any AAB" as a conflict. A quick experiment (3 generations each) isolated what works:

| Variant | EOPT-correct answers |
|---|---|
| EOPT chunk last, unlabelled | 0/3 |
| EOPT chunk first, unlabelled | 2/3 |
| EOPT chunk last, labelled "supersedes" | **3/3** |
| EOPT chunk first, labelled | 3/3 |

So the sources block now prints the note in the source header (`[S4] … (issued 2024) — NOTE: this 2024 EOPT Act rule supersedes…`). Rule 6 now says that newer issuances supersede older ones, gives EOPT and filing venue as the example, and says *"do not present a superseded rule as current"*. Tests: `test_sources_block_shows_supersession_note`, `test_deliberate_prompt_v3_tells_model_newer_issuance_supersedes_older`.

## After

Langfuse trace [`08c7901a…`](https://jp.cloud.langfuse.com/project/cmwa7yq5t005zad0lze6lsfai/traces/08c7901ac3f64c88d1045cd618a6b5a2), screenshot `logs/failure-after.png`:

> You can file and pay your quarterly income tax return either electronically or manually with any Authorized Agent Bank (AAB), Revenue District Office (RDO) through its Revenue Collection Officers, or an Authorized Tax Software Provider. This flexibility is provided under the 2024 Ease of Paying Taxes (EOPT) Act, which supersedes older rules regarding where to file and pay taxes [S4].

A differently worded question not in the golden set, *"Can I file my 2551Q at any bank or only in my RDO?"*, now also answers *"Under the 2024 Ease of Paying Taxes (EOPT) Act… any Authorized Agent Bank, RDO, or Authorized Tax Software Provider… This rule supersedes the older requirement…"* [S4].

## Regression check

The full golden set was rerun as `v3-eopt-supersession`, along with the suites. The threshold was re-checked in `threshold-v2`: in-corpus and out-of-corpus scores are unchanged, so 0.47 stands, and reference-doc-in-top-4 rose from 27/30 to **28/30**.

| | v2 (before) | v3 (after) |
|---|---|---|
| GQ-EOPT context recall / faithfulness / answer relevancy | 0.00 / 1.00 / 0.83 | **1.00** / 0.83 / 0.92 |
| Faithfulness (mean) | 0.834 | 0.797 |
| Answer relevancy | 0.799 | 0.791 |
| Context precision | 0.879 | **0.890** |
| Context recall | 0.933 | **0.967** |
| In-corpus fallback rate | 0% | 3.3% (GQ01) |
| Out-of-corpus declined | 4/4 | 4/4 |
| Guardrail suite / memory suite | 16/16, 7/8 | 16/16, 7/8 |

The retrieval metrics improved, as targeted. The small dips in faithfulness and relevancy were investigated rather than assumed:
- **The pin fired on only one golden question (GQ-EOPT).** For 28 of the 30 in-corpus questions, the retrieved sources are identical between v2 and v3. The per-question swings on those (e.g. GQ27 1.00 → 0.56, GQ05 0.75 → 1.00) come from generation and judge variance between runs, not from this change.
- **GQ01's fallback is a pre-existing flaky case, not a regression.** *"Who is not qualified to use the 8% rate?"* is answered as a bullet list that often loses its `[S#]` citations, so the citation check sends it to the fallback. Over 6 repeated runs: v3 prompt 2 fails, 3 retry-passes and 1 pass; **v2 prompt 3 fails**, 2 retry-passes and 1 pass. The safe fallback is the intended behaviour when citations are missing; making list answers keep their citations is noted as future work.

## Lessons

- **Fluent, well-cited and faithful can still be wrong.** Faithfulness only checks the answer against the retrieved context. Here the context itself was stale, and only context recall against a reference flagged it.
- **Chunking has to follow the document's structure.** The same 1,200-character splitter that works for long regulations buried a one-paragraph rule inside an unrelated table in a flyer.
- **Retrieval alone wasn't enough, and neither was the prompt alone.** The model needed the newer source *and* an explicit signal that it supersedes the older one; we confirmed this with the small experiment above rather than guessing.

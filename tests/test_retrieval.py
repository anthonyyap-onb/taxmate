import hashlib

import numpy as np

from taxmate.retrieval import build_index, make_query_rewriter
from taxmate.types import Chunk


def fake_embed(texts):
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().split():
            out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1.0
    return out


def ch(i, text):
    return Chunk(f"D#{i}#0", "D", "Doc", "2018", f"p{i}", text)


CHUNKS = [ch(1, "graduated income tax rates table"), ch(2, "8% income tax option election"),
          ch(3, "form 2307 creditable withholding certificate")]


def test_search_ranks_matching_chunk_first():
    idx = build_index(CHUNKS, fake_embed)
    assert idx.search(fake_embed(["8% option"])[0], 2)[0].chunk.section == "p2"


def test_scores_are_cosine_and_sorted():
    hits = build_index(CHUNKS, fake_embed).search(fake_embed(["income tax"])[0], 2)
    scores = [h.score for h in hits]
    assert len(hits) == 2 and scores == sorted(scores, reverse=True) and all(-1.0001 <= s <= 1.0001 for s in scores)


def test_k_larger_than_index_returns_all():
    assert len(build_index(CHUNKS, fake_embed).search(fake_embed(["tax"])[0], 4)) == 3


def test_rewriter_skips_llm_without_history():
    assert make_query_rewriter(client=None, model="m")([], "What is 2307?") == "What is 2307?"


from taxmate.retrieval import SUPERSESSION_RULES, make_retriever

EOPT = Chunk("EOPT-FLYER#File-and-Pay-Anywhere#0", "EOPT-FLYER", "EOPT", "2024", "File and Pay Anywhere",
             "taxpayers may file returns and pay taxes with any authorized agent bank")
PIN_CHUNKS = [ch(1, "quarterly income tax return deadlines where to file"), ch(2, "quarterly return file AAB RDO"),
              ch(3, "graduated rates table"), EOPT]


def test_filing_venue_question_pins_superseding_eopt_chunk_last():
    hits = make_retriever(build_index(PIN_CHUNKS, fake_embed), fake_embed)("Where can I file my quarterly return?", 2)
    assert len(hits) == 2 and hits[0].chunk.section == "p1"
    assert hits[-1].chunk.chunk_id == EOPT.chunk_id and hits[-1].note == SUPERSESSION_RULES[0]["note"]
    assert hits[0].note is None


def test_unrelated_question_is_not_pinned():
    hits = make_retriever(build_index(PIN_CHUNKS, fake_embed), fake_embed)("graduated rates table", 2)
    assert EOPT.chunk_id not in [h.chunk.chunk_id for h in hits]


def test_pinned_chunk_not_duplicated_when_already_retrieved():
    hits = make_retriever(build_index(PIN_CHUNKS, fake_embed), fake_embed)("where to pay with any authorized agent bank", 4)
    ids = [h.chunk.chunk_id for h in hits]
    assert ids.count(EOPT.chunk_id) == 1 and len(ids) == 4
    assert next(h for h in hits if h.chunk.chunk_id == EOPT.chunk_id).note == SUPERSESSION_RULES[0]["note"]


def test_supersession_rule_targets_existing_eopt_chunk():
    from taxmate.corpus import load_chunks
    ids = {c.chunk_id for c in load_chunks()}
    assert all(rule["chunk_id"] in ids for rule in SUPERSESSION_RULES)

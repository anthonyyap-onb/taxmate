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

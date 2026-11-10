import numpy as np
import pytest

from taxmate.cache import (CachedEmbedder, CachedTurn, ResponseCache, corpus_fingerprint, load_or_build_index,
                           normalize_query, salient_tokens)
from taxmate.types import Chunk, ScoredChunk

CHUNKS = [Chunk(f"D#{i}#0", "D", "Doc", "2018", f"s{i}", f"text {i}") for i in range(3)]


class CountingEmbed:
    """Deterministic fake: maps each text to a fixed vector; counts texts sent to the 'API'."""

    def __init__(self, table=None):
        self.sent: list[str] = []
        self.table = table or {}

    def __call__(self, texts):
        self.sent += texts
        return np.array([self.table.get(t, [1.0, float(len(t)), 0.0]) for t in texts], dtype=np.float32)


# --- document index on disk ---------------------------------------------------------------------------------------

def test_fingerprint_changes_with_text_and_model():
    fp = corpus_fingerprint(CHUNKS, "m1")
    assert fp == corpus_fingerprint(CHUNKS, "m1")
    assert fp != corpus_fingerprint(CHUNKS, "m2")
    changed = CHUNKS[:2] + [Chunk("D#2#0", "D", "Doc", "2018", "s2", "edited")]
    assert fp != corpus_fingerprint(changed, "m1")


def test_index_is_embedded_once_then_loaded_from_disk(tmp_path):
    embed = CountingEmbed()
    fp = corpus_fingerprint(CHUNKS, "m")
    idx1, how1 = load_or_build_index(CHUNKS, embed, fp, tmp_path)
    idx2, how2 = load_or_build_index(CHUNKS, embed, fp, tmp_path)
    assert (how1, how2) == ("built", "loaded") and len(embed.sent) == 3
    q = np.array([1.0, 6.0, 0.0])
    assert [h.chunk.chunk_id for h in idx1.search(q, 2)] == [h.chunk.chunk_id for h in idx2.search(q, 2)]


def test_corrupt_or_mismatched_index_file_is_rebuilt(tmp_path):
    embed = CountingEmbed()
    fp = corpus_fingerprint(CHUNKS, "m")
    load_or_build_index(CHUNKS, embed, fp, tmp_path)
    next(tmp_path.glob("*.npy")).write_bytes(b"garbage")
    _, how = load_or_build_index(CHUNKS, embed, fp, tmp_path)
    assert how == "built"


def test_index_cache_dir_none_always_builds():
    _, how = load_or_build_index(CHUNKS, CountingEmbed(), "fp", None)
    assert how == "built"


# --- query embeddings ---------------------------------------------------------------------------------------------

def test_cached_embedder_only_sends_new_texts():
    inner = CountingEmbed()
    embed = CachedEmbedder(inner, max_entries=10)
    a = embed(["x", "yy"])
    b = embed(["yy", "zzz", "x"])
    assert inner.sent == ["x", "yy", "zzz"]
    assert np.allclose(a[1], b[0]) and np.allclose(a[0], b[2])


def test_cached_embedder_evicts_oldest():
    inner = CountingEmbed()
    embed = CachedEmbedder(inner, max_entries=2)
    embed(["a"]); embed(["b"]); embed(["c"]); embed(["a"])  # noqa: E702
    assert inner.sent == ["a", "b", "c", "a"]


# --- response cache -----------------------------------------------------------------------------------------------

def turn(text="Form 2307 is ... [S1]"):
    return CachedTurn("answer", text, [ScoredChunk(CHUNKS[0], 0.8)], [])


def make_cache(table=None, **kw):
    clock = {"t": 1000.0}
    cache = ResponseCache(CachedEmbedder(CountingEmbed(table)), namespace="corpus-a", clock=lambda: clock["t"], **kw)
    return cache, clock


def test_normalize_query():
    assert normalize_query("  What is  Form 2307?? ") == normalize_query("what is form 2307")


def test_exact_hit_same_context_and_miss_on_other_context():
    cache, _ = make_cache()
    cache.store("What is Form 2307?", "ctx-8pct", turn())
    hit = cache.lookup("what is form 2307", "ctx-8pct")
    assert hit and hit.mode == "exact" and hit.turn.text == "Form 2307 is ... [S1]"
    assert cache.lookup("What is Form 2307?", "ctx-graduated") is None


def test_ttl_expiry():
    cache, clock = make_cache(ttl_s=60)
    cache.store("q", "c", turn())
    clock["t"] += 61
    assert cache.lookup("q", "c") is None and cache.stats()["entries"] == 0


def test_max_entries_evicts_least_recently_used():
    cache, _ = make_cache(max_entries=2)
    cache.store("q1", "c", turn("1")); cache.store("q2", "c", turn("2"))  # noqa: E702
    cache.lookup("q1", "c")
    cache.store("q3", "c", turn("3"))
    assert cache.lookup("q2", "c") is None and cache.lookup("q1", "c") and cache.lookup("q3", "c")


SIM = {"what does form 2307 do": [1.0, 0.0, 0.0], "what is form 2307 for": [0.99, 0.05, 0.0],
       "can i deduct rent on the 8% rate": [0.0, 1.0, 0.0], "can i deduct rent on graduated rates": [0.0, 0.99, 0.05],
       "tax on 1,200,000 at 8%": [0.0, 0.0, 1.0], "tax on 1,500,000 at 8%": [0.0, 0.01, 0.99],
       "totally different": [0.5, 0.5, 0.5]}


def test_semantic_hit_for_paraphrase_with_same_salient_tokens():
    cache, _ = make_cache(SIM, semantic_threshold=0.95)
    cache.store("What does Form 2307 do?", "c", turn())
    hit = cache.lookup("What is Form 2307 for?", "c")
    assert hit and hit.mode == "semantic" and hit.similarity >= 0.95


@pytest.mark.parametrize("stored,asked", [
    ("Can I deduct rent on the 8% rate?", "Can I deduct rent on graduated rates?"),
    ("Tax on 1,200,000 at 8%", "Tax on 1,500,000 at 8%"),
])
def test_semantic_never_crosses_different_salient_facts(stored, asked):
    cache, _ = make_cache(SIM, semantic_threshold=0.9)
    cache.store(stored, "c", turn())
    assert cache.lookup(asked, "c") is None


def test_semantic_respects_threshold_and_context():
    cache, _ = make_cache(SIM, semantic_threshold=0.95)
    cache.store("What does Form 2307 do?", "c", turn())
    assert cache.lookup("totally different", "c") is None
    assert cache.lookup("What is Form 2307 for?", "other-ctx") is None


def test_semantic_can_be_disabled():
    cache, _ = make_cache(SIM, semantic_threshold=None)
    cache.store("What does Form 2307 do?", "c", turn())
    assert cache.lookup("What is Form 2307 for?", "c") is None


def test_clear_and_stats():
    cache, _ = make_cache()
    cache.store("q", "c", turn())
    cache.lookup("q", "c"); cache.lookup("nope", "c")  # noqa: E702
    s = cache.stats()
    assert s["entries"] == 1 and s["hits_exact"] == 1 and s["misses"] == 1
    cache.clear()
    assert cache.stats()["entries"] == 0 and cache.lookup("q", "c") is None


def test_salient_tokens():
    assert salient_tokens("Can I deduct rent on the 8% rate?") != salient_tokens("Can I deduct rent on graduated?")
    assert salient_tokens("Is Form 2307 needed in Q1?") == salient_tokens("in q1 do I need form 2307")
    assert salient_tokens("₱1.2M gross") == salient_tokens("1,200,000 gross")
    assert "itemized" in salient_tokens("itemised deductions") and "osd" in salient_tokens("Optional Standard Deduction")


def test_salient_catches_itemize_verb_forms():
    """Found by the benchmark: 'Should I itemize or use OSD?' vs 'Should I use OSD?' had cosine 0.96."""
    assert salient_tokens("Should I itemize or use OSD?") != salient_tokens("Should I use OSD?")
    assert "itemized" in salient_tokens("I'm itemizing my expenses")

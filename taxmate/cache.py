"""Caching: the document index on disk, query embeddings in memory, and a response cache (exact + semantic).

Invalidation policy (also in evals/docs/caching.md):
- Corpus text or embedding model changes -> new corpus fingerprint -> new index file and a new response-cache
  namespace, so nothing built from the old corpus is used again.
- Model, prompt version, prompt variant, calculator version and the full tax profile are part of every response
  key, so changing any of them can never hit an old entry.
- Entries expire after a TTL (7 days by default) as a backstop for rules that change outside the corpus.
- An admin-only "Clear cache" button (TAXMATE_ADMIN=1) empties the response cache.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from taxmate.retrieval import Embedder, VectorIndex, build_index
from taxmate.tools import _DEDUCTION_WORDS, _REGIME_WORDS, stated_amounts
from taxmate.types import Chunk, ScoredChunk

DEFAULT_TTL_S = 7 * 24 * 3600


# --- document index ---------------------------------------------------------------------------------------------

def corpus_fingerprint(chunks: list[Chunk], embed_model: str) -> str:
    h = hashlib.sha256(embed_model.encode())
    for c in chunks:
        h.update(b"\0" + c.chunk_id.encode() + b"\0" + c.text.encode())
    return h.hexdigest()


def load_or_build_index(chunks: list[Chunk], embed: Embedder, fingerprint: str,
                        cache_dir: str | Path | None) -> tuple[VectorIndex, str]:
    """Load the chunk vectors saved for this fingerprint, or embed the corpus and save them."""
    path = Path(cache_dir) / f"index-{fingerprint[:24]}.npy" if cache_dir else None
    if path and path.exists():
        try:
            vecs = np.load(path, allow_pickle=False)
            if vecs.ndim == 2 and vecs.shape[0] == len(chunks):
                return VectorIndex(chunks, vecs), "loaded"
        except (OSError, ValueError):
            pass  # corrupt or partial file: rebuild below
    index = build_index(chunks, embed)
    if path:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npy")
            np.save(tmp, index._vecs, allow_pickle=False)
            tmp.replace(path)
        except OSError:
            pass  # a read-only disk only costs the cache, never the app
    return index, "built"


# --- query embeddings -------------------------------------------------------------------------------------------

class CachedEmbedder:
    """LRU cache in front of an embedder; only texts not seen before are sent to the API."""

    def __init__(self, embed: Embedder, max_entries: int = 2000):
        self._embed, self._max = embed, max_entries
        self._store: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def __call__(self, texts: list[str]) -> np.ndarray:
        with self._lock:
            found = {t: self._store[t] for t in texts if t in self._store}
            for t in found:
                self._store.move_to_end(t)
        missing = list(dict.fromkeys(t for t in texts if t not in found))
        if missing:
            new = {t: np.asarray(v, dtype=np.float32) for t, v in zip(missing, self._embed(missing))}
            found.update(new)
            with self._lock:
                self._store.update(new)
                for t in new:
                    self._store.move_to_end(t)
                while len(self._store) > self._max:
                    self._store.popitem(last=False)
        return np.vstack([found[t] for t in texts])


# --- response cache ---------------------------------------------------------------------------------------------

def normalize_query(q: str) -> str:
    return re.sub(r"[\s?!.,;:]+$", "", re.sub(r"\s+", " ", q.strip().casefold()))


_FORM = re.compile(r"\b\d{4}[a-z]{0,2}\b", re.I)
_QUARTER = re.compile(r"\bq([1-4])\b|\b(first|second|third|fourth)\s+quarter\b", re.I)
_QUARTER_WORDS = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
_MARKERS = {"mixed": re.compile(r"\bmixed\b", re.I), "purely": re.compile(r"\bpurely\b", re.I)}


def salient_tokens(q: str) -> frozenset[str]:
    """Facts that change the answer even when the wording is almost the same. A semantic hit needs these equal."""
    toks = {f"amt:{v:.2f}" for v in stated_amounts([q])}
    toks |= {f"form:{m.group(0).lower()}" for m in _FORM.finditer(q)}
    toks |= {f"q{m.group(1) or _QUARTER_WORDS[m.group(2).lower()]}" for m in _QUARTER.finditer(q)}
    for label, pat in {**_REGIME_WORDS, **_DEDUCTION_WORDS, **_MARKERS}.items():
        if pat.search(q):
            toks.add(label)
    return frozenset(toks)


@dataclass
class CachedTurn:
    kind: str
    text: str
    sources: list[ScoredChunk]
    tool_calls: list


@dataclass
class CacheHit:
    turn: CachedTurn
    mode: str  # "exact" | "semantic"
    similarity: float


@dataclass
class _Entry:
    ctx: str
    salient: frozenset[str]
    vec: np.ndarray | None
    turn: CachedTurn
    created: float


class ResponseCache:
    def __init__(self, embed: Callable[[list[str]], np.ndarray], namespace: str, ttl_s: float = DEFAULT_TTL_S,
                 max_entries: int = 500, semantic_threshold: float | None = 0.95,
                 clock: Callable[[], float] = time.time):
        self._embed, self.namespace = embed, namespace
        self.ttl_s, self.max_entries, self.semantic_threshold = ttl_s, max_entries, semantic_threshold
        self._clock = clock
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {"hits_exact": 0, "hits_semantic": 0, "misses": 0, "stores": 0}

    def _vec(self, query: str) -> np.ndarray:
        v = np.asarray(self._embed([query])[0], dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v

    def _expire(self, now: float) -> None:
        for key in [k for k, e in self._entries.items() if now - e.created > self.ttl_s]:
            del self._entries[key]

    def lookup(self, query: str, ctx: str) -> CacheHit | None:
        now = self._clock()
        key = (ctx, normalize_query(query))
        with self._lock:
            self._expire(now)
            entry = self._entries.get(key)
            if entry:
                self._entries.move_to_end(key)
                self._stats["hits_exact"] += 1
                return CacheHit(entry.turn, "exact", 1.0)
            candidates = [] if self.semantic_threshold is None else [
                (k, e) for k, e in self._entries.items() if e.ctx == ctx and e.vec is not None]
        if candidates:
            salient, qv = salient_tokens(query), self._vec(query)
            best = max(((float(e.vec @ qv), k, e) for k, e in candidates if e.salient == salient),
                       default=None, key=lambda x: x[0])
            if best and best[0] >= self.semantic_threshold:
                with self._lock:
                    if best[1] in self._entries:
                        self._entries.move_to_end(best[1])
                    self._stats["hits_semantic"] += 1
                return CacheHit(best[2].turn, "semantic", round(best[0], 4))
        with self._lock:
            self._stats["misses"] += 1
        return None

    def store(self, query: str, ctx: str, turn: CachedTurn) -> None:
        vec = self._vec(query) if self.semantic_threshold is not None else None
        with self._lock:
            key = (ctx, normalize_query(query))
            self._entries[key] = _Entry(ctx, salient_tokens(query), vec, turn, self._clock())
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            self._stats["stores"] += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> dict:
        with self._lock:
            self._expire(self._clock())
            return {"entries": len(self._entries), **self._stats}

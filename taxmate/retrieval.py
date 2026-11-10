"""Embedding, FAISS index, retriever, and history-aware query rewriting."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Callable

import faiss
import numpy as np

from taxmate.types import Chunk, ScoredChunk, Turn

Embedder = Callable[[list[str]], np.ndarray]
HISTORY_TURNS = 10


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


class VectorIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        vecs = _normalize(vectors)
        self._vecs = vecs
        self._pos = {c.chunk_id: i for i, c in enumerate(chunks)}
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

    def search(self, query_vec: np.ndarray, k: int) -> list[ScoredChunk]:
        k = min(k, len(self.chunks))
        if k <= 0:
            return []
        scores, ids = self._index.search(_normalize(np.atleast_2d(query_vec)), k)
        return [ScoredChunk(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


    def score(self, query_vec: np.ndarray, chunk_id: str) -> ScoredChunk | None:
        """Cosine score of one specific chunk against the query (None if the chunk isn't indexed)."""
        i = self._pos.get(chunk_id)
        if i is None:
            return None
        return ScoredChunk(self.chunks[i], float(self._vecs[i] @ _normalize(np.atleast_2d(query_vec))[0]))


def build_index(chunks: list[Chunk], embed: Embedder, batch_size: int = 64) -> VectorIndex:
    batches = [embed([c.text for c in chunks[i:i + batch_size]]) for i in range(0, len(chunks), batch_size)]
    return VectorIndex(chunks, np.vstack(batches))


def make_embedder(client, model: str) -> Embedder:
    def embed(texts: list[str]) -> np.ndarray:
        resp = client.embeddings.create(model=model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)
    return embed


# Newer issuances that override older ones on a topic. When a query is about that topic, the superseding chunk is
# always included (as the last of the k sources) so the model can apply the newer rule, even when the older,
# form-specific guidance out-scores it on similarity. The prompt tells the model to prefer the newer issued date.
SUPERSESSION_RULES: list[dict] = [
    {"name": "EOPT file-and-pay-anywhere (2024) supersedes 2018 filing-venue rules",
     "chunk_id": "EOPT-FLYER#File-and-Pay-Anywhere#0",
     "note": "this 2024 EOPT Act rule supersedes older (2018) rules on where to file returns and pay taxes",
     "action": re.compile(r"\b(file|filing|pay|paying|submit|lodge)\b", re.I),
     "venue": re.compile(r"\b(where|anywhere|venue|banks?|aab|authori[sz]ed agent|rdo|branch|office)\b", re.I)},
]


def make_retriever(index: VectorIndex, embed: Embedder,
                   rules: list[dict] = SUPERSESSION_RULES) -> Callable[[str, int], list[ScoredChunk]]:
    def retrieve(query: str, k: int) -> list[ScoredChunk]:
        qvec = embed([query])[0]
        hits = index.search(qvec, k)
        for rule in rules:
            if not (rule["action"].search(query) and rule["venue"].search(query)):
                continue
            if any(h.chunk.chunk_id == rule["chunk_id"] for h in hits):
                hits = [replace(h, note=rule["note"]) if h.chunk.chunk_id == rule["chunk_id"] else h for h in hits]
                continue
            pinned = index.score(qvec, rule["chunk_id"])
            if pinned is not None and k > 0:
                hits = hits[:k - 1] + [replace(pinned, note=rule["note"])]
        return hits
    return retrieve


_REWRITE_SYSTEM = ("Rewrite the user's latest question into a standalone search query for a database of "
                   "Philippine BIR tax regulations. Resolve pronouns and references using the conversation. "
                   "Keep Philippine tax terms (8% income tax, graduated rates, OSD, Form 2307, 1701Q, percentage tax). "
                   "Translate Filipino/Taglish to English. Output only the query.")


def make_query_rewriter(client, model: str) -> Callable[[list[Turn], str], str]:
    def rewrite(history: list[Turn], question: str) -> str:
        if not history:
            return question
        convo = "\n".join(f"{t.role}: {t.content}" for t in history[-HISTORY_TURNS:])
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "system", "content": _REWRITE_SYSTEM},
                      {"role": "user", "content": f"Conversation:\n{convo}\n\nLatest question: {question}"}],
        )
        return (resp.choices[0].message.content or question).strip()
    return rewrite

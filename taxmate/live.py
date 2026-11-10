"""Wire real OpenAI-backed implementations into the pipeline."""
from __future__ import annotations

from taxmate import tracing
import time

from taxmate.cache import CachedEmbedder, ResponseCache, corpus_fingerprint, load_or_build_index
from taxmate.config import Settings
from taxmate.corpus import load_chunks
from taxmate.generation import make_generator
from taxmate.guardrails import make_moderator, make_topic_classifier
from taxmate.pipeline import Deps
from taxmate.profile import make_profile_extractor
from taxmate.logging_utils import log_turn
from taxmate.retrieval import VectorIndex, make_embedder, make_query_rewriter, make_retriever


def build_live_deps(s: Settings, threshold: float | None = None, k: int | None = None,
                    response_cache: bool = False) -> tuple[Deps, VectorIndex]:
    """response_cache is opt-in: the app turns it on; evaluation runs leave it off so every turn is measured."""
    client = tracing.make_openai_client(s)
    embed = make_embedder(client, s.embed_model)
    chunks = load_chunks()
    fingerprint = corpus_fingerprint(chunks, s.embed_model)
    t0 = time.perf_counter()
    index, how = load_or_build_index(chunks, embed, fingerprint, s.cache_dir)
    log_turn({"event": "startup", "index": how, "index_ms": int((time.perf_counter() - t0) * 1000),
              "n_chunks": len(chunks), "corpus_fingerprint": fingerprint[:12], "response_cache": response_cache})
    query_embed = CachedEmbedder(embed)
    deps = Deps(
        moderate=make_moderator(client),
        classify_topic=make_topic_classifier(client, s.chat_model),
        extract_profile=make_profile_extractor(client, s.chat_model),
        rewrite_query=make_query_rewriter(client, s.chat_model),
        retrieve=make_retriever(index, query_embed),
        generate=make_generator(client, s.chat_model),
        model=s.chat_model,
        cache=ResponseCache(query_embed, namespace=fingerprint, ttl_s=s.cache_ttl_days * 86400,
                            semantic_threshold=s.semantic_threshold) if response_cache else None,
    )
    if threshold is not None:
        deps.threshold = threshold
    if k is not None:
        deps.k = k
    return deps, index

"""Wire real OpenAI-backed implementations into the pipeline."""
from __future__ import annotations

from taxmate import tracing
from taxmate.config import Settings
from taxmate.corpus import load_chunks
from taxmate.generation import make_generator
from taxmate.guardrails import make_moderator, make_topic_classifier
from taxmate.pipeline import Deps
from taxmate.profile import make_profile_extractor
from taxmate.retrieval import VectorIndex, build_index, make_embedder, make_query_rewriter, make_retriever


def build_live_deps(s: Settings, threshold: float | None = None, k: int | None = None) -> tuple[Deps, VectorIndex]:
    client = tracing.make_openai_client(s)
    embed = make_embedder(client, s.embed_model)
    index = build_index(load_chunks(), embed)
    deps = Deps(
        moderate=make_moderator(client),
        classify_topic=make_topic_classifier(client, s.chat_model),
        extract_profile=make_profile_extractor(client, s.chat_model),
        rewrite_query=make_query_rewriter(client, s.chat_model),
        retrieve=make_retriever(index, embed),
        generate=make_generator(client, s.chat_model),
        model=s.chat_model,
    )
    if threshold is not None:
        deps.threshold = threshold
    if k is not None:
        deps.k = k
    return deps, index

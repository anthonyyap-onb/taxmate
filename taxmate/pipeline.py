"""One guarded RAG turn: validate -> mask -> guard -> profile -> retrieve -> generate -> check -> log."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from taxmate import copy, tracing
from taxmate.guardrails import check_input
from taxmate.logging_utils import log_turn
from taxmate.pii import mask_pii
from taxmate.profile import ProfileUpdate, TaxProfile, merge_profile
from taxmate.prompts import PROMPT_VERSION
from taxmate.types import GenerationResult, ScoredChunk, Turn, TurnResult

MAX_INPUT_CHARS = 2000


@dataclass
class Deps:
    moderate: Callable[[str], bool]
    classify_topic: Callable[[str, str | None], str]  # (message, TaxMate's previous message or None)
    extract_profile: Callable[[str, TaxProfile], ProfileUpdate]
    rewrite_query: Callable[[list[Turn], str], str]
    retrieve: Callable[[str, int], list[ScoredChunk]]
    generate: Callable[[str, list[Turn], TaxProfile, list[ScoredChunk], str, bool], GenerationResult]
    model: str
    threshold: float = 0.47  # tuned in evals/results/threshold-v1 (min in-corpus top score 0.478)
    k: int = 4


def _safe(fn: Callable, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


class _Timer:
    def __init__(self):
        self.ms: dict[str, int] = {"guard": 0, "profile": 0, "retrieval": 0, "generation": 0}
        self._start = time.perf_counter()

    def stage(self, name: str):
        timer = self

        class _Ctx:
            def __enter__(self):
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                timer.ms[name] += int((time.perf_counter() - self.t) * 1000)
                return False
        return _Ctx()

    def total(self) -> dict:
        return {"total": int((time.perf_counter() - self._start) * 1000), **self.ms}


@tracing.observe("turn", capture_input=False, capture_output=False)
def run_turn(user_text: str, history: list[Turn], profile: TaxProfile, deps: Deps, session_id: str,
             variant: str = "deliberate") -> TurnResult:
    timer = _Timer()
    rec: dict = {
        "ts": datetime.now(timezone.utc).isoformat(), "request_id": str(uuid.uuid4()), "session_id": session_id,
        "model": deps.model, "prompt_version": PROMPT_VERSION, "prompt_variant": variant, "pii_masked": [],
        "moderation_flagged": False, "topic_category": None, "blocked": False, "block_reason": None,
        "n_chunks": 0, "top_score": None, "fallback": False, "citation_check": "skipped",
        "langfuse_trace_id": None, "error": None, "user_message_masked": "", "kind": None, "retrieved": [],
    }

    def finish(kind: str, text: str, sources: list[ScoredChunk], prof: TaxProfile) -> TurnResult:
        rec["kind"] = kind
        rec["latency_ms"] = timer.total()
        rec["langfuse_trace_id"] = _safe(tracing.current_trace_id)
        _safe(tracing.update_span, input=rec["user_message_masked"], output=text,
              metadata={"kind": kind, "block_reason": rec["block_reason"], "citation_check": rec["citation_check"],
                        "top_score": rec["top_score"], "prompt_version": PROMPT_VERSION, "variant": variant})
        log_turn({**rec, "retrieved": [{k: v for k, v in r.items() if k != "text"} for r in rec["retrieved"]]})
        return TurnResult(kind, text, sources, prof, rec)

    text = (user_text or "").strip()
    if not text:
        return finish("invalid", copy.EMPTY_TEXT, [], profile)
    if len(text) > MAX_INPUT_CHARS:
        return finish("invalid", copy.TOO_LONG_TEXT, [], profile)

    masked, pii = mask_pii(text)
    rec["pii_masked"], rec["user_message_masked"] = pii, masked

    with tracing.trace_context(session_id=session_id, metadata={"variant": variant}):
        try:
            with timer.stage("guard"):
                # A bare reply to a clarifying question ("purely compensation") is only on-topic in context.
                prev = next((t.content for t in reversed(history) if t.role == "assistant"), None)
                verdict = check_input(masked, deps.moderate, lambda t: deps.classify_topic(t, prev))
            rec["topic_category"] = verdict.category
            if verdict.blocked:
                rec["blocked"], rec["block_reason"] = True, verdict.reason
                rec["moderation_flagged"] = verdict.category == "moderation"
                if verdict.category in ("unsafe", "moderation"):
                    return finish("unsafe", copy.UNSAFE_TEXT, [], profile)
                return finish("blocked", copy.BLOCKED_TEXT, [], profile)

            with timer.stage("profile"):
                before = profile
                profile = merge_profile(profile, deps.extract_profile(masked, profile))

            if verdict.category == "smalltalk":
                with timer.stage("generation"):
                    gen = deps.generate(masked, history, profile, [], variant, False)
                return finish("smalltalk", gen.text, [], profile)

            with timer.stage("retrieval"):
                query = deps.rewrite_query(history, masked)
                hits = deps.retrieve(query, deps.k)
            rec["n_chunks"] = len(hits)
            rec["top_score"] = round(hits[0].score, 4) if hits else None
            rec["retrieved"] = [{"chunk_id": h.chunk.chunk_id, "score": round(h.score, 4), "text": h.chunk.text}
                                for h in hits]
            if not hits or hits[0].score < deps.threshold:
                if profile != before:  # a statement about the user, not an unanswerable question
                    return finish("smalltalk", copy.PROFILE_ACK_TEXT, [], profile)
                rec["fallback"] = True
                return finish("fallback", copy.FALLBACK_TEXT, [], profile)

            with timer.stage("generation"):
                gen = deps.generate(masked, history, profile, hits, variant, False)
                if gen.type == "answer":
                    rec["citation_check"] = "pass"
                    if not gen.cited:
                        gen = deps.generate(masked, history, profile, hits, variant, True)
                        rec["citation_check"] = "retry_pass" if (gen.type == "clarify" or gen.cited) else "fail"
                if gen.type == "clarify":
                    rec["citation_check"] = "skipped"
            if rec["citation_check"] == "fail":
                rec["fallback"] = True
                return finish("fallback", copy.FALLBACK_TEXT, [], profile)

            with timer.stage("guard"):
                if deps.moderate(gen.text):
                    rec["moderation_flagged"], rec["blocked"], rec["block_reason"] = True, True, "output_moderation"
                    return finish("blocked", copy.BLOCKED_TEXT, [], profile)

            if gen.type == "clarify":
                return finish("clarify", gen.text, [], profile)
            return finish("answer", gen.text, [hits[i - 1] for i in gen.cited], profile)
        except Exception as exc:
            rec["error"] = type(exc).__name__
            return finish("error", copy.ERROR_TEXT, [], profile)

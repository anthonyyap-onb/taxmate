"""Fail-safe Langfuse tracing. Every function here is a no-op that never raises when tracing is
disabled (no Langfuse keys) or when Langfuse itself errors. A tracing problem must never break a turn."""
from __future__ import annotations

import contextlib
import functools
import logging
from typing import Any, Callable, Iterator

import openai

from taxmate.config import Settings

log = logging.getLogger("taxmate")
_enabled = False


def tracing_enabled(s: Settings) -> bool:
    return bool(s.langfuse_public_key and s.langfuse_secret_key)


def init(s: Settings) -> bool:
    """Decide once, at app/eval start, whether tracing is on. Langfuse reads LANGFUSE_* from env."""
    global _enabled
    _enabled = False
    if not tracing_enabled(s):
        return False
    try:
        from langfuse import get_client
        get_client()
        _enabled = True
    except Exception as exc:  # pragma: no cover - depends on network/SDK
        log.warning("Langfuse init failed, tracing disabled: %s", type(exc).__name__)
    return _enabled


def make_openai_client(s: Settings):
    if tracing_enabled(s):
        try:
            from langfuse.openai import OpenAI as TracedOpenAI
            return TracedOpenAI(api_key=s.openai_api_key)
        except Exception as exc:  # pragma: no cover
            log.warning("Langfuse OpenAI wrapper unavailable, using plain client: %s", type(exc).__name__)
    return openai.OpenAI(api_key=s.openai_api_key)


def observe(name: str, capture_input: bool = True, capture_output: bool = True) -> Callable:
    """Decorator applied at import time; whether to trace is decided at call time."""
    def decorator(fn: Callable) -> Callable:
        traced: list[Callable] = []

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _enabled:
                return fn(*args, **kwargs)
            if not traced:
                try:
                    from langfuse import observe as lf_observe
                    traced.append(lf_observe(name=name, capture_input=capture_input,
                                             capture_output=capture_output)(fn))
                except Exception:  # pragma: no cover
                    traced.append(fn)
            return traced[0](*args, **kwargs)
        return wrapper
    return decorator


@contextlib.contextmanager
def trace_context(session_id: str | None = None, tags: list[str] | None = None,
                  metadata: dict | None = None) -> Iterator[None]:
    """Attach session/tags/metadata to every observation created inside the block."""
    cm = None
    if _enabled:
        try:
            from langfuse import propagate_attributes
            cm = propagate_attributes(session_id=session_id, tags=tags,
                                      metadata={k: str(v) for k, v in (metadata or {}).items()} or None)
            cm.__enter__()
        except Exception:  # pragma: no cover
            cm = None
    try:
        yield
    finally:
        if cm is not None:
            try:
                cm.__exit__(None, None, None)
            except Exception:  # pragma: no cover
                pass


def _client_call(method: str, **kwargs: Any) -> Any:
    if not _enabled:
        return None
    try:
        from langfuse import get_client
        return getattr(get_client(), method)(**kwargs)
    except Exception:  # pragma: no cover
        return None


def update_span(**kwargs: Any) -> None:
    _client_call("update_current_span", **kwargs)


def current_trace_id() -> str | None:
    return _client_call("get_current_trace_id")


def flush() -> None:
    _client_call("flush")

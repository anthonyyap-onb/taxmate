"""Per-turn token and cost accounting. Each OpenAI call site calls record(response); the pipeline wraps a turn
in track() and logs summary(). Outside a turn (e.g. building the index at startup) record() does nothing."""
from __future__ import annotations

import contextlib
import contextvars
from typing import Iterator

# USD per 1M tokens (input, output), OpenAI list prices. Matched by model-name prefix, longest prefix first, so
# dated snapshots like "gpt-4o-mini-2024-07-18" resolve. Update this table if the models or prices change.
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def _price(model: str) -> tuple[float, float] | None:
    for prefix in sorted(PRICES_PER_1M, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES_PER_1M[prefix]
    return None


class Usage:
    def __init__(self) -> None:
        self.prompt = self.completion = self.embedding = 0
        self.cost: float | None = 0.0

    def add(self, model: str, prompt: int, completion: int, is_embedding: bool) -> None:
        if is_embedding:
            self.embedding += prompt
        else:
            self.prompt += prompt
            self.completion += completion
        price = _price(model)
        if price is None:
            self.cost = None
        elif self.cost is not None:
            self.cost += (prompt * price[0] + completion * price[1]) / 1_000_000

    def summary(self) -> dict:
        return {"tokens": {"prompt": self.prompt, "completion": self.completion, "embedding": self.embedding},
                "est_cost_usd": round(self.cost, 8) if self.cost is not None else None}


_current: contextvars.ContextVar[Usage | None] = contextvars.ContextVar("taxmate_usage", default=None)


@contextlib.contextmanager
def track() -> Iterator[Usage]:
    u = Usage()
    token = _current.set(u)
    try:
        yield u
    finally:
        _current.reset(token)


def record(resp) -> None:
    """Add an OpenAI response's token usage to the current turn. Never raises."""
    u = _current.get()
    if u is None:
        return
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return
        model = str(getattr(resp, "model", "") or "")
        completion = getattr(usage, "completion_tokens", None)
        u.add(model, int(usage.prompt_tokens or 0), int(completion or 0), is_embedding=completion is None)
    except Exception:
        pass

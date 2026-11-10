"""Structured answer generation and citation parsing."""
from __future__ import annotations

import re
from typing import Callable, Literal

from pydantic import BaseModel

from taxmate.profile import TaxProfile
from taxmate.prompts import build_messages
from taxmate.types import GenerationResult, ScoredChunk, Turn

_CITE_GROUP = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")


def parse_citations(text: str, n_sources: int) -> list[int]:
    ids = set()
    for group in _CITE_GROUP.findall(text):
        for part in group.split(","):
            n = int(part.strip()[1:])
            if 1 <= n <= n_sources:
                ids.add(n)
    return sorted(ids)


class _Answer(BaseModel):
    type: Literal["answer", "clarify"]
    text: str


Generator = Callable[[str, list[Turn], TaxProfile, list[ScoredChunk], str, bool], GenerationResult]


def make_generator(client, model: str) -> Generator:
    def generate(question: str, history: list[Turn], profile: TaxProfile, sources: list[ScoredChunk],
                 variant: str, strict: bool) -> GenerationResult:
        completion = client.chat.completions.parse(
            model=model, temperature=0,
            messages=build_messages(question, history, profile, sources, variant, strict),
            response_format=_Answer,
        )
        msg = completion.choices[0].message
        parsed = msg.parsed or _Answer(type="answer", text=msg.content or "")
        return GenerationResult(parsed.type, parsed.text, parse_citations(parsed.text, len(sources)))
    return generate

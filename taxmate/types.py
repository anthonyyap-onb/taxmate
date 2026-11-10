"""Shared data types for the TaxMate pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from taxmate.profile import TaxProfile


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    issued: str
    section: str
    text: str


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    note: str | None = None  # e.g. a supersession notice added by the retriever


@dataclass(frozen=True)
class Turn:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class GenerationResult:
    type: Literal["answer", "clarify"]
    text: str
    cited: list[int] = field(default_factory=list)
    tool_calls: list = field(default_factory=list)  # taxmate.tools.ToolCall, in call order


@dataclass(frozen=True)
class GuardVerdict:
    blocked: bool
    category: str
    reason: str | None


TurnKind = Literal["answer", "clarify", "smalltalk", "blocked", "unsafe", "fallback", "error", "invalid"]


@dataclass
class TurnResult:
    kind: TurnKind
    text: str
    sources: list[ScoredChunk]
    profile: "TaxProfile"
    record: dict
    tool_calls: list = field(default_factory=list)  # taxmate.tools.ToolCall

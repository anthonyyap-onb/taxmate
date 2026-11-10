"""Structured answer generation and citation parsing."""
from __future__ import annotations

import json
import re
from typing import Callable, Literal

from pydantic import BaseModel

from taxmate import usage
from taxmate.profile import TaxProfile
from taxmate.prompts import build_messages
from taxmate.tools import TOOL_SCHEMAS, ToolCall, run_tool
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

MAX_TOOL_ROUNDS = 3


def _assistant_tool_message(msg) -> dict:
    return {"role": "assistant", "content": msg.content,
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in msg.tool_calls]}


def make_generator(client, model: str) -> Generator:
    def generate(question: str, history: list[Turn], profile: TaxProfile, sources: list[ScoredChunk],
                 variant: str, strict: bool) -> GenerationResult:
        messages = build_messages(question, history, profile, sources, variant, strict)
        tool_calls: list[ToolCall] = []
        user_texts = [t.content for t in history if t.role == "user"] + [question]
        for round_no in range(MAX_TOOL_ROUNDS + 1):
            offer_tools = round_no < MAX_TOOL_ROUNDS  # the last round must produce a final answer
            completion = client.chat.completions.parse(
                model=model, temperature=0, messages=messages, response_format=_Answer,
                **({"tools": TOOL_SCHEMAS} if offer_tools else {}),
            )
            usage.record(completion)
            msg = completion.choices[0].message
            if offer_tools and msg.tool_calls:
                messages.append(_assistant_tool_message(msg))
                for tc in msg.tool_calls:
                    call = run_tool(tc.function.name, tc.function.arguments, profile, user_texts)
                    tool_calls.append(call)
                    messages.append({"role": "tool", "tool_call_id": tc.id,
                                     "content": json.dumps(call.result, ensure_ascii=False)})
                continue
            break
        parsed = msg.parsed or _Answer(type="answer", text=msg.content or "")
        return GenerationResult(parsed.type, parsed.text, parse_citations(parsed.text, len(sources)), tool_calls)
    return generate

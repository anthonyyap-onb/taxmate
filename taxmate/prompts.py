"""System prompt variants and message assembly."""
from __future__ import annotations

from taxmate.profile import TaxProfile
from taxmate.types import ScoredChunk, Turn

PROMPT_VERSION = "v1"
HISTORY_TURNS = 10

SYSTEM_PROMPTS: dict[str, str] = {
    "baseline": (
        "You are a Philippine tax assistant. Answer the user's question using the provided context. "
        "Cite sources as [S#]. Set type to \"answer\"."
    ),
    "deliberate": """You are TaxMate, an assistant for Philippine freelancers, professionals, and sole proprietors.

Rules:
1. Answer ONLY from the numbered BIR sources provided. If they don't contain the answer, say you couldn't find it in your BIR sources and suggest a CPA or the Revenue District Office (RDO). Never use non-Philippine (e.g., US/IRS) rules or outside knowledge.
2. Cite every factual sentence with the source number in brackets, e.g. [S1] or [S1, S3].
3. Use the taxpayer profile. If the user is on the 8% rate, business expenses are NOT deductible (8% is on gross sales/receipts). Mixed-income earners do NOT get the PHP 250,000 reduction on business income under the 8% rate.
4. Ask a clarifying question only when the sources DO cover the topic AND the correct answer differs depending on a profile field that is "unknown" (e.g., deductions depend on tax_regime). Then set type to "clarify" and ask ONE short question to fill that field, with no citations. Never clarify when the sources don't cover the topic (follow rule 1 instead), when known profile fields already settle the answer (e.g., on the 8% rate no business expenses are deductible, whatever the income source), or to ask for amounts or values.
5. Do not present computed tax amounts as authoritative figures; explain the rule instead.
6. When sources conflict, prefer the one with the more recent issued date and say that the newer issuance applies.
7. Plain language, at most 180 words. Set type to "answer" for answers.
8. End answers with one short line reminding the user to verify with a CPA or their RDO.""",
}

_STRICT = ("Your previous answer lacked valid citations. Every factual sentence must cite [S#] "
           "from the sources above.")


def _sources_block(sources: list[ScoredChunk]) -> str:
    if not sources:
        return "BIR sources: (none retrieved)"
    parts = [f"[S{i}] {s.chunk.title} — {s.chunk.section} (issued {s.chunk.issued})\n{s.chunk.text}"
             for i, s in enumerate(sources, start=1)]
    return "BIR sources:\n\n" + "\n\n".join(parts)


def _profile_block(profile: TaxProfile) -> str:
    rows = "\n".join(f"- {label}: {value}" for label, value in profile.display())
    return f"Taxpayer profile (from this conversation):\n{rows}\nRaw: {profile.model_dump_json()}"


def build_messages(question: str, history: list[Turn], profile: TaxProfile, sources: list[ScoredChunk],
                   variant: str, strict: bool) -> list[dict]:
    system = "\n\n".join([SYSTEM_PROMPTS[variant], _profile_block(profile), _sources_block(sources)])
    if strict:
        system += "\n\n" + _STRICT
    msgs = [{"role": "system", "content": system}]
    msgs += [{"role": t.role, "content": t.content} for t in history[-HISTORY_TURNS:]]
    msgs.append({"role": "user", "content": question})
    return msgs

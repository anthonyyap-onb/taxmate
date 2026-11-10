"""System prompt variants and message assembly."""
from __future__ import annotations

from taxmate.profile import TaxProfile
from taxmate.types import ScoredChunk, Turn

PROMPT_VERSION = "v3"
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
6. Each source shows its issued date. Newer issuances supersede older ones: if a newer source (e.g. the 2024 EOPT Act) changes a rule that an older source (e.g. 2018 form guidelines) states differently — such as where to file or pay — give the newer rule as the current one, cite it, and briefly note that the older rule was superseded. Do not present a superseded rule as current.
7. Plain language, at most 180 words. Set type to "answer" for answers. Answer the question directly; do not add closing disclaimers (the app shows its own).""",
}

_STRICT = ("Your previous answer lacked valid citations. Every factual sentence must cite [S#] "
           "from the sources above.")


def _sources_block(sources: list[ScoredChunk]) -> str:
    if not sources:
        return "BIR sources: (none retrieved)"
    parts = [f"[S{i}] {s.chunk.title} — {s.chunk.section} (issued {s.chunk.issued})"
             + (f" — NOTE: {s.note}" if s.note else "") + f"\n{s.chunk.text}"
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

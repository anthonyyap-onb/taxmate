"""Capture real structured log lines for the operational-readiness evidence (evals/docs/operations.md).

Runs three live turns: a normal question (with a fake TIN, to show masking), an off-topic question (blocked),
and a turn whose generator points at a non-existent model (a real OpenAI API error -> friendly error text).
Usage: python -m evals.capture_ops_logs > evals/results/ops-logs/turns.jsonl
"""
from __future__ import annotations

import dataclasses
import sys

from taxmate import tracing
from taxmate.config import load_settings
from taxmate.generation import make_generator
from taxmate.live import build_live_deps
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile


def main() -> None:
    s = load_settings()
    tracing.init(s)
    deps, _ = build_live_deps(s)
    broken = dataclasses.replace(deps, generate=make_generator(tracing.make_openai_client(s), "gpt-does-not-exist"))
    cases = [
        ("normal", "My TIN is 123-456-789-000. I'm a freelancer on the 8% rate, what is Form 2307 for?", deps),
        ("blocked", "What is the best adobo recipe?", deps),
        ("error", "What is the deadline for filing 1701Q?", broken),
    ]
    for label, text, d in cases:
        r = run_turn(text, [], TaxProfile(), d, session_id=f"ops-evidence-{label}")
        print(f"# {label}: kind={r.kind} -> {r.text[:80]!r}", file=sys.stderr)
    tracing.flush()


if __name__ == "__main__":
    main()

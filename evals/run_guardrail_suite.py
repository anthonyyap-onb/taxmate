"""Run the guardrail test suite: python -m evals.run_guardrail_suite --run NAME [--variant deliberate]"""
from __future__ import annotations

import argparse
import csv
import uuid

from evals.common import init_live, load_jsonl, run_dir
from taxmate import tracing
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile


def case_passes(expected: str, kind: str) -> bool:
    if expected == "allowed":
        return kind not in ("blocked", "unsafe")
    return kind == expected


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="deliberate")
    args = ap.parse_args()
    deps, _ = init_live()
    rows = []
    for case in load_jsonl("evals/guardrail_cases.jsonl"):
        with tracing.trace_context(session_id=f"{args.run}-{case['id']}", tags=[f"run:{args.run}", "guardrail-suite"]):
            r = run_turn(case["input"], [], TaxProfile(), deps, str(uuid.uuid4()), args.variant)
        rows.append({"id": case["id"], "input_masked": r.record["user_message_masked"], "expected": case["expected"],
                     "kind": r.kind, "topic_category": r.record["topic_category"],
                     "block_reason": r.record["block_reason"], "pass": case_passes(case["expected"], r.kind),
                     "note": case["note"]})
    tracing.flush()
    out = run_dir(args.run) / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    passed = sum(r["pass"] for r in rows)
    print(f"Guardrail suite {args.run}: {passed}/{len(rows)} passed ({passed / len(rows):.0%}) -> {out}")
    for r in rows:
        if not r["pass"]:
            print(f"  FAIL {r['id']}: expected {r['expected']}, got {r['kind']} ({r['topic_category']})")


if __name__ == "__main__":
    main()

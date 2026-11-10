"""Run scripted multi-turn memory scenarios: python -m evals.run_memory_suite --run NAME [--variant deliberate]"""
from __future__ import annotations

import argparse
import csv
import json
import uuid

from evals.common import ROOT, init_live, run_dir
from taxmate import tracing
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile
from taxmate.types import Turn


def check_turn(expect: dict, kind: str, text: str, profile: TaxProfile) -> list[str]:
    problems = []
    for field, value in expect["expect_profile"].items():
        if getattr(profile, field) != value:
            problems.append(f"profile.{field}={getattr(profile, field)} (expected {value})")
    if expect.get("expect_kind") and kind != expect["expect_kind"]:
        problems.append(f"kind={kind} (expected {expect['expect_kind']})")
    needles = expect.get("expect_text_contains_any")
    if needles and not any(n.lower() in text.lower() for n in needles):
        problems.append(f"text lacks any of {needles}")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="deliberate")
    args = ap.parse_args()
    deps, _ = init_live()
    scenarios = json.loads((ROOT / "evals" / "memory_scenarios.json").read_text(encoding="utf-8"))
    rows = []
    for sc in scenarios:
        session = str(uuid.uuid4())
        history: list[Turn] = []
        profile = TaxProfile()
        with tracing.trace_context(session_id=f"{args.run}-{sc['id']}", tags=[f"run:{args.run}", "memory-suite"]):
            for i, t in enumerate(sc["turns"], start=1):
                r = run_turn(t["user"], history, profile, deps, session, args.variant)
                history += [Turn("user", r.record["user_message_masked"]), Turn("assistant", r.text)]
                profile = r.profile
                problems = check_turn(t, r.kind, r.text, profile)
                rows.append({"scenario": sc["id"], "turn": i, "user": t["user"], "kind": r.kind,
                             "profile": profile.model_dump_json(), "response": r.text,
                             "pass": not problems, "problems": "; ".join(problems)})
    tracing.flush()
    out = run_dir(args.run) / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scen_pass = {s["id"]: all(r["pass"] for r in rows if r["scenario"] == s["id"]) for s in scenarios}
    n = sum(scen_pass.values())
    print(f"Memory suite {args.run}: {n}/{len(scenarios)} scenarios passed; "
          f"{sum(r['pass'] for r in rows)}/{len(rows)} turns -> {out}")
    for r in rows:
        if not r["pass"]:
            print(f"  FAIL {r['scenario']} turn {r['turn']}: {r['problems']}")


if __name__ == "__main__":
    main()

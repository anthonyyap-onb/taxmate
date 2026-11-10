"""Tool-use evaluation: python -m evals.run_tool_suite --run NAME [--variant deliberate]

Each scenario in evals/tool_scenarios.json is a short conversation. Only the LAST turn is scored:
- tool selection (system): the first tool call that actually ran in that turn (or none) is one of expect.tool
  (null = no tool). A call blocked by a precondition (taxmate.tools.PRECONDITION_ERRORS) did not run.
- tool selection (model): the same check on the model's first attempted call, blocked or not. The gap between the
  two is how often the code-level preconditions caught the model guessing an input.
- argument accuracy: for scenarios where a tool was expected and the right one was called, the call's arguments
  match expect.args (numbers within 1 peso; a null optional amount counts as 0)
- task completion: tool selection + arguments + reply kind + reply text all pass
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid

from evals.common import ROOT, init_live, run_dir
from taxmate import tracing
from taxmate.pipeline import run_turn
from taxmate.profile import TaxProfile
from taxmate.tools import PRECONDITION_ERRORS
from taxmate.types import Turn


def load_scenarios() -> list[dict]:
    return json.loads((ROOT / "evals" / "tool_scenarios.json").read_text(encoding="utf-8"))


def _same(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return abs(float(actual or 0) - expected) <= 1
    return expected == actual


def score(expect: dict, kind: str, text: str, tool_calls: list) -> dict:
    executed = [t for t in tool_calls if t.result.get("error") not in PRECONDITION_ERRORS]
    first = executed[0] if executed else None
    tool_ok = (first.name if first else None) in expect["tool"]
    model_tool_ok = (tool_calls[0].name if tool_calls else None) in expect["tool"]
    args_expected = bool(expect.get("args"))
    args_ok = None
    if args_expected and first is not None and tool_ok:
        args_ok = all(_same(v, first.args.get(k)) for k, v in expect["args"].items())
    kind_ok = kind in expect["kind"]
    text_ok = any(n.lower() in text.lower() for n in expect.get("text_any", [""]))
    complete = tool_ok and kind_ok and text_ok and (args_ok is not False) and not (args_expected and args_ok is None)
    return {"model_tool_ok": model_tool_ok, "tool_ok": tool_ok, "args_ok": args_ok, "kind_ok": kind_ok, "text_ok": text_ok, "complete": complete}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variant", default="deliberate")
    args = ap.parse_args()
    deps, _ = init_live()
    rows = []
    for sc in load_scenarios():
        session, history, profile = str(uuid.uuid4()), [], TaxProfile()
        with tracing.trace_context(session_id=f"{args.run}-{sc['id']}", tags=[f"run:{args.run}", "tool-suite"]):
            for user in sc["turns"]:
                r = run_turn(user, history, profile, deps, session, args.variant)
                history += [Turn("user", r.record["user_message_masked"]), Turn("assistant", r.text)]
                profile = r.profile
        s = score(sc["expect"], r.kind, r.text, r.tool_calls)
        rows.append({"id": sc["id"], "category": sc["category"], "expected_tool": "|".join(map(str, sc["expect"]["tool"])),
                     "called": "; ".join(f"{t.name}({json.dumps(t.args)}) -> {t.result.get('error') or 'ok'}"
                                         for t in r.tool_calls),
                     "kind": r.kind, **s, "response": r.text})
    tracing.flush()
    out = run_dir(args.run) / "results.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    sel = sum(r["tool_ok"] for r in rows)
    model_sel = sum(r["model_tool_ok"] for r in rows)
    blocked = sum(1 for r in rows for code in PRECONDITION_ERRORS if code in r["called"])
    arg_rows = [r for r in rows if r["args_ok"] is not None]
    argn = sum(r["args_ok"] for r in arg_rows)
    done = sum(r["complete"] for r in rows)
    summary = {"run": args.run, "variant": args.variant, "scenarios": n,
               "tool_selection_accuracy": round(sel / n, 3), "model_tool_selection_accuracy": round(model_sel / n, 3),
               "precondition_blocks": blocked,
               "argument_accuracy": round(argn / len(arg_rows), 3) if arg_rows else None,
               "argument_scored": len(arg_rows), "task_completion_rate": round(done / n, 3)}
    (run_dir(args.run) / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Tool suite {args.run}: selection {sel}/{n} (model alone {model_sel}/{n}, {blocked} precondition blocks), arguments {argn}/{len(arg_rows)}, "
          f"completion {done}/{n} -> {out}")
    for r in rows:
        if not r["complete"]:
            print(f"  FAIL {r['id']} [{r['category']}] kind={r['kind']} called={r['called'] or '-'} "
                  f"tool_ok={r['tool_ok']} args_ok={r['args_ok']} text_ok={r['text_ok']}\n    {r['response'][:200]}")


if __name__ == "__main__":
    main()

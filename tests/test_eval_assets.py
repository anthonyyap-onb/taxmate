import json

from evals.common import load_jsonl
from evals.run_guardrail_suite import case_passes
from taxmate.corpus import load_chunks
from taxmate.profile import TaxProfile

GOLDEN = load_jsonl("evals/golden_set.jsonl")
CHUNKS = load_chunks()


def test_golden_set_shape():
    assert sum(r["in_corpus"] for r in GOLDEN) >= 25 and sum(not r["in_corpus"] for r in GOLDEN) >= 4
    assert any(r["id"] == "GQ-EOPT" for r in GOLDEN) and sum(bool(r["profile"]) for r in GOLDEN) >= 2
    assert len({r["id"] for r in GOLDEN}) == len(GOLDEN)


def test_golden_references_exist_in_corpus():
    keys = {(c.doc_id, c.section) for c in CHUNKS}
    docs = {c.doc_id for c in CHUNKS}
    for r in GOLDEN:
        if r["in_corpus"]:
            assert r["reference_doc_id"] in docs, r["id"]
            if r["reference_doc_id"] == "RMC-50-2018":
                assert (r["reference_doc_id"], r["reference_section"]) in keys, r["id"]


def test_golden_profiles_valid():
    for r in GOLDEN:
        TaxProfile(**r["profile"])


def test_guardrail_cases_shape():
    rows = load_jsonl("evals/guardrail_cases.jsonl")
    assert len(rows) >= 15 and len({r["id"] for r in rows}) == len(rows)
    assert {r["expected"] for r in rows} <= {"blocked", "unsafe", "allowed"}
    assert any(r["id"] == "G-taglish" for r in rows)


def test_memory_scenarios_shape():
    scenarios = json.load(open("evals/memory_scenarios.json"))
    assert len(scenarios) == 4
    fields = set(TaxProfile.model_fields)
    for s in scenarios:
        for t in s["turns"]:
            assert set(t["expect_profile"]) <= fields


def test_guardrail_pass_rule():
    assert case_passes("allowed", "answer") and case_passes("allowed", "clarify") and case_passes("allowed", "fallback")
    assert not case_passes("allowed", "blocked") and not case_passes("unsafe", "blocked")
    assert case_passes("blocked", "blocked") and case_passes("unsafe", "unsafe")


def test_tool_scenarios_cover_required_cases_and_expected_numbers_match_calculator():
    from evals.run_tool_suite import load_scenarios
    from taxmate.tools import run_tool
    scenarios = load_scenarios()
    cats = {s["category"] for s in scenarios}
    assert len(scenarios) >= 15
    assert {"calc_8pct", "compare", "regime_unknown", "malformed", "rag_not_tool", "memory", "above_3m"} <= cats
    for s in scenarios:
        e = s["expect"]
        if e.get("args") and e["tool"] == ["calculate_ph_tax"]:
            full = {"deduction_method": None, "itemized_expenses": None, "non_operating_income": None,
                    "compensation_income": None, **e["args"]}
            result = run_tool("calculate_ph_tax", __import__("json").dumps(full)).result
            figures = {f"{result['total_tax']:,.0f}", f"{result['income_tax']:,.0f}"}
            assert any(t in figures for t in e["text_any"]), (s["id"], figures, e["text_any"])


def test_tool_suite_scoring():
    from evals.run_tool_suite import score
    from taxmate.tools import ToolCall
    exp = {"tool": ["calculate_ph_tax"], "args": {"gross_receipts": 1000, "compensation_income": 0},
           "kind": ["answer"], "text_any": ["76,000"]}
    good = [ToolCall("calculate_ph_tax", {"gross_receipts": 1000.4, "compensation_income": None}, {})]
    assert score(exp, "answer", "about ₱76,000", good)["complete"]
    assert score(exp, "answer", "about ₱76,000", [ToolCall("calculate_ph_tax", {"gross_receipts": 5}, {})])["args_ok"] is False
    assert not score(exp, "answer", "about ₱76,000", [])["complete"]
    none_exp = {"tool": [None], "kind": ["clarify"], "text_any": ["8%"]}
    assert score(none_exp, "clarify", "8% or graduated?", [])["complete"]
    blocked = [ToolCall("calculate_ph_tax", {}, {"error": "regime_not_stated", "message": "ask"})]
    s = score(none_exp, "clarify", "8% or graduated?", blocked)
    assert s["complete"] and s["tool_ok"] and not s["model_tool_ok"]

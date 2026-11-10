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

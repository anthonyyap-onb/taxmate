from taxmate.corpus import DOCS
from taxmate.generation import parse_citations
from taxmate.profile import TaxProfile
from taxmate.prompts import build_messages
from taxmate.types import Chunk, ScoredChunk, Turn


def sc(doc_id, section, text="Some rule text."):
    d = next(d for d in DOCS if d["doc_id"] == doc_id)
    return ScoredChunk(Chunk(f"{doc_id}#{section}#0", doc_id, d["title"], d["issued"], section, text), 0.8)


def test_parse_citations_basic():
    assert parse_citations("A [S2]. B [S1][S2].", 4) == [1, 2]


def test_parse_grouped_citations():
    assert parse_citations("A [S1, S3].", 4) == [1, 3]


def test_out_of_range_citation_ignored():
    assert parse_citations("A [S9] B [S0]", 4) == []


def test_messages_number_sources_and_include_profile():
    msgs = build_messages("Can I deduct rent?", [], TaxProfile(tax_regime="8_percent"),
                          [sc("RMC-50-2018", "Q18")], "deliberate", False)
    joined = "\n".join(m["content"] for m in msgs)
    assert "[S1] Revenue Memorandum Circular No. 50-2018 — Q18" in joined and "8%" in joined


def test_history_truncated_to_10():
    hist = [Turn("user", f"m{i}") for i in range(15)]
    contents = [m["content"] for m in build_messages("q", hist, TaxProfile(), [], "deliberate", False)]
    assert "m4" not in contents and "m5" in contents


def test_strict_adds_citation_instruction():
    msgs = build_messages("q", [], TaxProfile(), [], "deliberate", True)
    assert "lacked valid citations" in "".join(m["content"] for m in msgs)


def test_sources_block_shows_issued_dates():
    msgs = build_messages("q", [], TaxProfile(), [sc("EOPT-FLYER", "p1")], "deliberate", False)
    assert "(issued 2024)" in "".join(m["content"] for m in msgs)


def test_deliberate_prompt_does_not_ask_model_for_cpa_reminder():
    from taxmate.prompts import SYSTEM_PROMPTS
    assert "remind" not in SYSTEM_PROMPTS["deliberate"].lower()


def test_deliberate_prompt_v3_tells_model_newer_issuance_supersedes_older():
    from taxmate.prompts import PROMPT_VERSION, SYSTEM_PROMPTS
    assert PROMPT_VERSION == "v3"
    rule = SYSTEM_PROMPTS["deliberate"].lower()
    assert "supersede" in rule and "issued" in rule and "eopt" in rule


def test_sources_block_shows_supersession_note():
    import dataclasses
    noted = dataclasses.replace(sc("EOPT-FLYER", "File and Pay Anywhere"), note="supersedes older filing-venue rules")
    content = "".join(m["content"] for m in build_messages("q", [], TaxProfile(), [sc("1701Q-GUIDE", "p1"), noted],
                                                          "deliberate", False))
    assert "[S2] Ease of Paying Taxes (EOPT) Act Flyer — File and Pay Anywhere (issued 2024) — NOTE: supersedes older filing-venue rules" in content
    assert content.count("NOTE:") == 1

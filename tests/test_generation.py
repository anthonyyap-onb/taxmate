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


def test_deliberate_prompt_tells_model_newer_issuance_supersedes_older():
    from taxmate.prompts import SYSTEM_PROMPTS
    rule = SYSTEM_PROMPTS["deliberate"].lower()
    assert "supersede" in rule and "issued" in rule and "eopt" in rule


def test_sources_block_shows_supersession_note():
    import dataclasses
    noted = dataclasses.replace(sc("EOPT-FLYER", "File and Pay Anywhere"), note="supersedes older filing-venue rules")
    content = "".join(m["content"] for m in build_messages("q", [], TaxProfile(), [sc("1701Q-GUIDE", "p1"), noted],
                                                          "deliberate", False))
    assert "[S2] Ease of Paying Taxes (EOPT) Act Flyer — File and Pay Anywhere (issued 2024) — NOTE: supersedes older filing-venue rules" in content
    assert content.count("NOTE:") == 1


def _tool_client(script):
    """Fake OpenAI client: each parse() call pops the next scripted message."""
    import json
    from types import SimpleNamespace as NS
    calls = []

    def parse(**kw):
        calls.append(kw)
        step = script.pop(0)
        if "tool" in step:
            tcs = [NS(id=f"c{i}", type="function", function=NS(name=n, arguments=json.dumps(a)))
                   for i, (n, a) in enumerate(step["tool"])]
            msg = NS(content=None, tool_calls=tcs, parsed=None)
        else:
            from taxmate.generation import _Answer
            msg = NS(content=None, tool_calls=None, parsed=_Answer(type=step.get("type", "answer"), text=step["text"]))
        return NS(model="gpt-4o-mini", usage=None, choices=[NS(message=msg)])
    return NS(chat=NS(completions=NS(parse=parse))), calls


ARGS_8 = {"gross_receipts": 1200000, "regime": "8_percent", "deduction_method": None, "itemized_expenses": None,
          "non_operating_income": None, "compensation_income": None}


def test_generator_runs_tool_then_answers():
    from taxmate.generation import make_generator
    client, calls = _tool_client([{"tool": [("calculate_ph_tax", ARGS_8)]}, {"text": "About ₱76,000."}])
    r = make_generator(client, "m")("₱1,200,000 gross, how much?", [], TaxProfile(tax_regime="8_percent"), [],
                                    "deliberate", False)
    assert r.type == "answer" and r.text == "About ₱76,000."
    assert [t.name for t in r.tool_calls] == ["calculate_ph_tax"] and r.tool_calls[0].result["total_tax"] == 76000
    assert "tools" in calls[0]
    tool_msgs = [m for m in calls[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs and '"total_tax": 76000' in tool_msgs[0]["content"]


def test_generator_stops_offering_tools_after_max_rounds():
    from taxmate.generation import MAX_TOOL_ROUNDS, make_generator
    script = [{"tool": [("calculate_ph_tax", ARGS_8)]} for _ in range(MAX_TOOL_ROUNDS)] + [{"text": "done"}]
    client, calls = _tool_client(script)
    r = make_generator(client, "m")("q", [], TaxProfile(), [], "deliberate", False)
    assert r.text == "done" and len(r.tool_calls) == MAX_TOOL_ROUNDS and "tools" not in calls[-1]


def test_generator_without_tool_use_has_empty_tool_calls():
    from taxmate.generation import make_generator
    client, _ = _tool_client([{"text": "Form 2307 is ... [S1]"}])
    r = make_generator(client, "m")("q", [], TaxProfile(), [], "deliberate", False)
    assert r.tool_calls == []


def test_deliberate_prompt_v4_requires_calculator_for_amounts():
    from taxmate.prompts import PROMPT_VERSION, SYSTEM_PROMPTS
    assert PROMPT_VERSION == "v4"
    rule = SYSTEM_PROMPTS["deliberate"]
    assert "calculate_ph_tax" in rule and "compare_regimes" in rule and "never do arithmetic" in rule.lower()


def test_no_sources_block_tells_model_only_calculator_answers_allowed():
    content = "".join(m["content"] for m in build_messages("q", [], TaxProfile(), [], "deliberate", False))
    assert "(none retrieved)" in content and "calculator" in content


def test_generator_applies_preconditions_with_conversation_context():
    from taxmate.generation import make_generator
    client, calls = _tool_client([{"tool": [("calculate_ph_tax", ARGS_8)]}, {"type": "clarify", "text": "8% or graduated?"}])
    r = make_generator(client, "m")("I earned ₱1,200,000. How much tax?", [], TaxProfile(), [], "deliberate", False)
    assert r.type == "clarify" and r.tool_calls[0].result["error"] == "regime_not_stated"


def test_structured_variant_is_deliberate_plus_reasoning_steps():
    from taxmate.prompts import SYSTEM_PROMPTS
    assert set(SYSTEM_PROMPTS) == {"baseline", "deliberate", "structured"}
    assert SYSTEM_PROMPTS["structured"].startswith(SYSTEM_PROMPTS["deliberate"])
    assert "analysis" in SYSTEM_PROMPTS["structured"]


def test_structured_variant_hides_analysis_from_user():
    from types import SimpleNamespace as NS
    from taxmate.generation import _ReasonedAnswer, make_generator
    seen = {}

    def parse(**kw):
        seen.update(kw)
        msg = NS(content=None, tool_calls=None,
                 parsed=_ReasonedAnswer(analysis="S1 covers it; regime known", type="answer", text="Yes [S1]"))
        return NS(model="m", usage=None, choices=[NS(message=msg)])
    client = NS(chat=NS(completions=NS(parse=parse)))
    r = make_generator(client, "m")("q", [], TaxProfile(), [sc("RMC-50-2018", "Q1")], "structured", False)
    assert seen["response_format"] is _ReasonedAnswer
    assert r.text == "Yes [S1]" and "regime known" not in r.text and r.cited == [1]

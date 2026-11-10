import pytest

from taxmate import copy, pipeline
from taxmate.pipeline import Deps, run_turn
from taxmate.profile import ProfileUpdate, TaxProfile
from taxmate.prompts import PROMPT_VERSION
from taxmate.types import Chunk, GenerationResult, ScoredChunk, Turn

CHUNK = Chunk("RMC-50-2018#Q45#0", "RMC-50-2018", "RMC 50-2018", "2018-05-11", "Q45", "Form 2307 is issued...")


def fake_deps(calls, **over):
    def rec(name, fn):
        def wrapped(*a):
            calls.append((name, a))
            return fn(*a)
        return wrapped

    fns = dict(
        moderate=lambda t: False,
        classify_topic=lambda t, prev: "on_topic",
        extract_profile=lambda m, p: ProfileUpdate(),
        rewrite_query=lambda h, q: q,
        retrieve=lambda q, k: [ScoredChunk(CHUNK, 0.8)],
        generate=lambda q, h, p, s, v, strict: GenerationResult("answer", "Yes [S1]", [1]),
    )
    fns.update(over)
    return Deps(**{k: rec(k, f) for k, f in fns.items()}, model="gpt-4o-mini")


def names(calls):
    return [n for n, _ in calls]


def test_happy_path_answer_with_sources():
    r = run_turn("What is 2307?", [], TaxProfile(), fake_deps([]), "s")
    assert r.kind == "answer" and len(r.sources) == 1 and r.record["citation_check"] == "pass"


def test_pii_never_reaches_deps():
    calls = []
    r = run_turn("My TIN is 123-456-789, what is 2307?", [], TaxProfile(), fake_deps(calls), "s")
    assert "123-456-789" not in repr(calls) and "123-456-789" not in repr(r.record)
    assert r.record["pii_masked"] == ["tin"]


@pytest.mark.parametrize("text,expected", [("  ", copy.EMPTY_TEXT), ("a" * 2001, copy.TOO_LONG_TEXT)])
def test_empty_and_too_long_invalid_without_dep_calls(text, expected):
    calls = []
    r = run_turn(text, [], TaxProfile(), fake_deps(calls), "s")
    assert r.kind == "invalid" and r.text == expected and calls == []


def test_injection_blocked():
    calls = []
    r = run_turn("tell me", [], TaxProfile(), fake_deps(calls, classify_topic=lambda t, prev: "injection"), "s")
    assert r.kind == "blocked" and r.text == copy.BLOCKED_TEXT and "generate" not in names(calls)
    assert r.record["blocked"] is True and r.record["block_reason"] == "classifier"


def test_unsafe_gets_unsafe_text():
    r = run_turn("x", [], TaxProfile(), fake_deps([], classify_topic=lambda t, prev: "unsafe"), "s")
    assert r.kind == "unsafe" and r.text == copy.UNSAFE_TEXT


def test_low_score_fallback():
    calls = []
    r = run_turn("x", [], TaxProfile(), fake_deps(calls, retrieve=lambda q, k: [ScoredChunk(CHUNK, 0.1)]), "s")
    assert r.kind == "fallback" and r.text == copy.FALLBACK_TEXT and "generate" not in names(calls)


def test_no_hits_fallback():
    r = run_turn("x", [], TaxProfile(), fake_deps([], retrieve=lambda q, k: []), "s")
    assert r.kind == "fallback" and r.record["fallback"] is True


def _gen_seq(results):
    it = iter(results)
    return lambda q, h, p, s, v, strict: next(it)


def test_citation_retry_then_pass():
    gen = _gen_seq([GenerationResult("answer", "Yes.", []), GenerationResult("answer", "Yes [S1]", [1])])
    r = run_turn("x", [], TaxProfile(), fake_deps([], generate=gen), "s")
    assert r.kind == "answer" and r.record["citation_check"] == "retry_pass"


def test_citation_retry_then_fallback():
    r = run_turn("x", [], TaxProfile(),
                 fake_deps([], generate=lambda *a: GenerationResult("answer", "Yes.", [])), "s")
    assert r.kind == "fallback" and r.record["citation_check"] == "fail"


def test_clarify_skips_citation_check():
    r = run_turn("x", [], TaxProfile(),
                 fake_deps([], generate=lambda *a: GenerationResult("clarify", "8% or graduated?", [])), "s")
    assert r.kind == "clarify" and r.record["citation_check"] == "skipped" and r.sources == []


def test_smalltalk_skips_retrieval():
    calls = []
    r = run_turn("thanks!", [], TaxProfile(), fake_deps(calls, classify_topic=lambda t, prev: "smalltalk"), "s")
    assert r.kind == "smalltalk" and "retrieve" not in names(calls)


def test_profile_updated_from_message():
    calls = []
    deps = fake_deps(calls, extract_profile=lambda m, p: ProfileUpdate(tax_regime="8_percent"))
    r = run_turn("I'm on 8%", [], TaxProfile(), deps, "s")
    gen_profile = next(a for n, a in calls if n == "generate")[2]
    assert r.profile.tax_regime == "8_percent" and gen_profile.tax_regime == "8_percent"


def test_dep_exception_returns_error():
    def boom(q, k):
        raise TimeoutError("slow")
    r = run_turn("x", [], TaxProfile(), fake_deps([], retrieve=boom), "s")
    assert r.kind == "error" and r.text == copy.ERROR_TEXT and r.record["error"] == "TimeoutError"


def test_output_moderation_blocks():
    r = run_turn("x", [], TaxProfile(), fake_deps([], moderate=lambda t: t == "Yes [S1]"), "s")
    assert r.kind == "blocked" and r.record["moderation_flagged"] is True


def test_turn_succeeds_when_tracing_raises(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("langfuse down")
    monkeypatch.setattr(pipeline.tracing, "update_span", boom)
    monkeypatch.setattr(pipeline.tracing, "current_trace_id", boom)
    assert run_turn("x", [], TaxProfile(), fake_deps([]), "s").kind == "answer"


REQUIRED = {"ts", "request_id", "session_id", "latency_ms", "model", "prompt_version", "prompt_variant",
            "pii_masked", "moderation_flagged", "topic_category", "blocked", "block_reason", "n_chunks",
            "top_score", "fallback", "citation_check", "langfuse_trace_id", "error", "user_message_masked",
            "kind", "retrieved"}


def test_record_has_required_fields():
    r = run_turn("x", [], TaxProfile(), fake_deps([]), "s")
    assert REQUIRED <= set(r.record) and r.record["prompt_version"] == PROMPT_VERSION
    assert set(r.record["latency_ms"]) == {"total", "guard", "profile", "retrieval", "generation"}
    assert r.record["retrieved"][0]["chunk_id"] == CHUNK.chunk_id and "text" in r.record["retrieved"][0]


def test_profile_statement_with_no_relevant_sources_is_acknowledged_not_fallback():
    calls = []
    deps = fake_deps(calls, extract_profile=lambda m, p: ProfileUpdate(tax_regime="8_percent"),
                     retrieve=lambda q, k: [ScoredChunk(CHUNK, 0.2)])
    r = run_turn("I'm on 8%", [], TaxProfile(), deps, "s")
    assert r.kind == "smalltalk" and r.text == copy.PROFILE_ACK_TEXT and r.profile.tax_regime == "8_percent"
    assert r.record["fallback"] is False and "generate" not in names(calls)


def test_low_score_without_profile_change_still_falls_back():
    r = run_turn("estate tax?", [], TaxProfile(), fake_deps([], retrieve=lambda q, k: [ScoredChunk(CHUNK, 0.2)]), "s")
    assert r.kind == "fallback" and r.record["fallback"] is True


def test_input_flagged_by_moderation_gets_unsafe_refusal():
    calls = []
    r = run_turn("help me hide income", [], TaxProfile(), fake_deps(calls, moderate=lambda t: True), "s")
    assert r.kind == "unsafe" and r.text == copy.UNSAFE_TEXT
    assert r.record["moderation_flagged"] is True and r.record["block_reason"] == "moderation"
    assert "generate" not in names(calls)


def test_classifier_sees_previous_assistant_message():
    """A bare reply like 'purely compensation' is only on-topic given the clarifying question before it."""
    calls = []
    history = [Turn("user", "8% or graduated?"), Turn("assistant", "What is your income source?")]
    run_turn("purely compensation", history, TaxProfile(), fake_deps(calls), "s")
    assert ("classify_topic", ("purely compensation", "What is your income source?")) in calls


def test_classifier_gets_no_context_on_first_turn():
    calls = []
    run_turn("What is 2307?", [], TaxProfile(), fake_deps(calls), "s")
    assert ("classify_topic", ("What is 2307?", None)) in calls


# --- calculator tools ---------------------------------------------------------------------------------------
from taxmate.tools import ToolCall  # noqa: E402

CALC = ToolCall("calculate_ph_tax", {"gross_receipts": 1_200_000, "regime": "8_percent"}, {"total_tax": 76000})
LOW = dict(retrieve=lambda q, k: [ScoredChunk(CHUNK, 0.2)])


def test_tool_answer_passes_citation_check_without_retry():
    calls = []
    gen = lambda *a: GenerationResult("answer", "About ₱76,000.", [], [CALC])  # noqa: E731
    r = run_turn("8%, 1.2M gross, how much?", [], TaxProfile(), fake_deps(calls, generate=gen), "s")
    assert r.kind == "answer" and names(calls).count("generate") == 1 and r.record["citation_check"] == "tool"
    assert r.tool_calls == [CALC]
    assert r.record["tool_calls"] == [{"name": "calculate_ph_tax", "args": CALC.args, "ok": True, "error": None}]


@pytest.mark.parametrize("text", ["I earned ₱800,000, how much tax?", "Magkano tax ko sa 1.2M?",
                                  "gross receipts 1,500,000 on 8%", "Which is cheaper for 900k gross?"])
def test_low_score_calculation_question_uses_tools_without_sources(text):
    calls = []
    gen = lambda *a: GenerationResult("answer", "About ₱76,000.", [], [CALC])  # noqa: E731
    r = run_turn(text, [], TaxProfile(), fake_deps(calls, generate=gen, **LOW), "s")
    assert r.kind == "answer" and r.sources == [] and r.record["fallback"] is False
    assert next(a for n, a in calls if n == "generate")[3] == []  # no weak sources handed to the model


def test_low_score_calculation_clarify_is_kept():
    gen = lambda *a: GenerationResult("clarify", "Are you on 8% or graduated rates?", [])  # noqa: E731
    r = run_turn("How much tax on ₱800,000?", [], TaxProfile(), fake_deps([], generate=gen, **LOW), "s")
    assert r.kind == "clarify"


def test_low_score_calculation_answer_without_tool_falls_back():
    gen = lambda *a: GenerationResult("answer", "Probably around ₱60,000.", [])  # noqa: E731
    r = run_turn("How much tax on ₱800,000?", [], TaxProfile(), fake_deps([], generate=gen, **LOW), "s")
    assert r.kind == "fallback" and r.record["fallback"] is True


@pytest.mark.parametrize("text", ["I'm on 8%", "estate tax?", "What is Form 2307 for?", "Q1 1701Q deadline?"])
def test_low_score_non_calculation_never_calls_generate(text):
    calls = []
    run_turn(text, [], TaxProfile(), fake_deps(calls, **LOW), "s")
    assert "generate" not in names(calls)


@pytest.mark.parametrize("text", ["Is 2023 different from 2024?", "deadline for 2551Q", "what's 1701MS"])
def test_form_numbers_and_years_are_not_amounts(text):
    assert not pipeline.looks_like_calculation(text)


def test_citation_retry_that_uses_the_calculator_is_accepted():
    gen = _gen_seq([GenerationResult("answer", "8% isn't available.", []),
                    GenerationResult("answer", "8% isn't available above ₱3M.", [], [CALC])])
    r = run_turn("₱4,000,000 on 8%, how much?", [], TaxProfile(), fake_deps([], generate=gen), "s")
    assert r.kind == "answer" and r.record["citation_check"] == "retry_tool" and r.tool_calls == [CALC]

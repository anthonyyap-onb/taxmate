from streamlit.testing.v1 import AppTest


def test_missing_key_shows_config_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception and "OPENAI_API_KEY" in at.error[0].value


def _app_with_messages(monkeypatch, messages):
    from taxmate.profile import TaxProfile
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    at = AppTest.from_file("../app.py")
    at.session_state["session_id"] = "s"
    at.session_state["messages"] = messages
    at.session_state["history"] = []
    at.session_state["profile"] = TaxProfile()
    return at.run(timeout=30)


def test_answers_show_verify_note_but_refusals_do_not(monkeypatch):
    from taxmate import copy
    at = _app_with_messages(monkeypatch, [
        {"role": "assistant", "kind": "answer", "content": "Form 2307 is ... [S1]", "sources": []},
        {"role": "assistant", "kind": "blocked", "content": copy.BLOCKED_TEXT, "sources": []},
    ])
    assert not at.exception
    assert [c.value for c in at.caption].count(copy.VERIFY_NOTE) == 1


def test_chip_click_sends_question_and_hides_chips_in_same_rerun(monkeypatch):
    from taxmate import copy
    from taxmate.profile import TaxProfile
    from taxmate.types import TurnResult
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("taxmate.live.build_live_deps", lambda s, **k: (None, None))
    asked = []
    monkeypatch.setattr("taxmate.pipeline.run_turn", lambda text, *a, **k: asked.append(text) or TurnResult(
        "answer", "Form 2307 is ... [S1]", [], TaxProfile(), {}))
    at = AppTest.from_file("../app.py").run(timeout=30)
    chips = [b for b in at.button if b.label in copy.SUGGESTED_QUESTIONS]
    assert len(chips) == len(copy.SUGGESTED_QUESTIONS)
    chips[-1].click().run(timeout=30)
    assert not at.exception
    assert asked == [copy.SUGGESTED_QUESTIONS[-1]]
    assert not [b for b in at.button if b.label in copy.SUGGESTED_QUESTIONS]


def test_startup_failure_shows_friendly_error_not_traceback(monkeypatch, caplog):
    import json
    import streamlit as st
    from taxmate import copy
    st.cache_resource.clear()  # get_deps is process-global; earlier tests cache fake deps

    def boom(s, **k):
        raise ConnectionError("api.openai.com unreachable")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("taxmate.live.build_live_deps", boom)
    at = AppTest.from_file("../app.py").run(timeout=30)
    with caplog.at_level("INFO", logger="taxmate"):
        at.chat_input[0].set_value("What is 2307?").run(timeout=30)
    assert not at.exception
    assert any(e.value == copy.ERROR_TEXT for e in at.error)
    logged = [json.loads(r.getMessage()) for r in caplog.records if r.getMessage().startswith("{")]
    assert logged and logged[-1]["kind"] == "error" and logged[-1]["error"] == "ConnectionError"
    assert "2307" not in caplog.text  # nothing about the user's question in a startup-error line


def test_calculator_result_renders_as_metric_card(monkeypatch):
    from taxmate.calculator import calculate_ph_tax, compare_regimes
    at = _app_with_messages(monkeypatch, [
        {"role": "assistant", "kind": "answer", "content": "About ₱76,000.", "sources": [],
         "tools": [{"name": "calculate_ph_tax", "ok": True, "result": calculate_ph_tax(1_200_000, "8_percent").to_dict()}]},
        {"role": "assistant", "kind": "answer", "content": "8% is lowest.", "sources": [],
         "tools": [{"name": "compare_regimes", "ok": True, "result": compare_regimes(900_000, 200_000)},
                   {"name": "calculate_ph_tax", "ok": False, "result": {"error": "invalid_amount", "message": "x"}}]},
    ])
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["Estimated total tax"] == "₱76,000.00"
    assert values["8% rate ✓ lowest"] == "₱52,000.00" and values["Graduated + OSD"] == "₱77,500.00"
    assert len(at.metric) == 6  # 3 for the calculation, 3 for the comparison; the failed call renders nothing


def test_admin_sidebar_shows_cache_stats_and_clear_button(monkeypatch):
    import streamlit as st
    from types import SimpleNamespace as NS
    st.cache_resource.clear()
    cleared = []
    fake_cache = NS(stats=lambda: {"entries": 3, "hits_exact": 2, "hits_semantic": 1, "misses": 4, "stores": 3},
                    clear=lambda: cleared.append(1))
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("TAXMATE_ADMIN", "1")
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("taxmate.live.build_live_deps", lambda s, **k: (NS(cache=fake_cache), None))
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception
    assert any("3 entries" in c.value for c in at.sidebar.caption)
    next(b for b in at.sidebar.button if b.label == "Clear cache").click().run(timeout=30)
    assert cleared == [1]


def test_no_admin_controls_by_default(monkeypatch):
    monkeypatch.delenv("TAXMATE_ADMIN", raising=False)
    at = _app_with_messages(monkeypatch, [])
    assert not [b for b in at.sidebar.button if b.label == "Clear cache"]

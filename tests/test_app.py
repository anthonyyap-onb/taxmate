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

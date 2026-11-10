from streamlit.testing.v1 import AppTest


def test_missing_key_shows_config_message(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    at = AppTest.from_file("../app.py").run(timeout=30)
    assert not at.exception and "OPENAI_API_KEY" in at.error[0].value

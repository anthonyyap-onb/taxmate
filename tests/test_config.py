from taxmate.config import load_settings, missing_keys


def test_missing_openai_key_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    assert missing_keys(load_settings()) == ["OPENAI_API_KEY"]


def test_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    for v in ("TAXMATE_CHAT_MODEL", "TAXMATE_EMBED_MODEL", "LANGFUSE_HOST"):
        monkeypatch.delenv(v, raising=False)
    s = load_settings()
    assert s.chat_model == "gpt-4o-mini" and s.embed_model == "text-embedding-3-small"
    assert s.langfuse_host == "https://cloud.langfuse.com" and missing_keys(s) == []

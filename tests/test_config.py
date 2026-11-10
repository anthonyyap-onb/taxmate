from taxmate.config import load_settings, missing_keys


def test_missing_openai_key_reported(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    assert missing_keys(load_settings()) == ["OPENAI_API_KEY"]


def test_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    for v in ("TAXMATE_CHAT_MODEL", "TAXMATE_EMBED_MODEL", "LANGFUSE_BASE_URL", "LANGFUSE_HOST"):
        monkeypatch.delenv(v, raising=False)
    s = load_settings()
    assert s.chat_model == "gpt-4o-mini" and s.embed_model == "text-embedding-3-small"
    assert s.langfuse_base_url == "https://cloud.langfuse.com" and missing_keys(s) == []


def test_langfuse_base_url_with_legacy_host_fallback(monkeypatch):
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.setenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")
    assert load_settings().langfuse_base_url == "https://us.cloud.langfuse.com"
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://eu.example")
    assert load_settings().langfuse_base_url == "https://eu.example"


def test_cache_settings_defaults_and_overrides(monkeypatch):
    monkeypatch.setattr("taxmate.config.load_dotenv", lambda *a, **k: None)
    for v in ("TAXMATE_CACHE_DIR", "TAXMATE_RESPONSE_CACHE", "TAXMATE_SEMANTIC_THRESHOLD", "TAXMATE_CACHE_TTL_DAYS",
              "TAXMATE_ADMIN"):
        monkeypatch.delenv(v, raising=False)
    s = load_settings()
    assert s.cache_dir == ".cache" and s.response_cache is True and s.semantic_threshold == 0.95
    assert s.cache_ttl_days == 7 and s.admin is False
    monkeypatch.setenv("TAXMATE_RESPONSE_CACHE", "off")
    monkeypatch.setenv("TAXMATE_SEMANTIC_THRESHOLD", "off")
    monkeypatch.setenv("TAXMATE_CACHE_DIR", "none")
    monkeypatch.setenv("TAXMATE_ADMIN", "1")
    s = load_settings()
    assert s.response_cache is False and s.semantic_threshold is None and s.cache_dir is None and s.admin is True

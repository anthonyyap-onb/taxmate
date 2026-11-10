"""Settings loaded from environment, .env (local) or st.secrets (Streamlit Cloud)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

REQUIRED_KEYS = ["OPENAI_API_KEY"]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    langfuse_public_key: str | None
    langfuse_secret_key: str | None
    langfuse_base_url: str
    chat_model: str
    embed_model: str
    log_level: str
    cache_dir: str | None = ".cache"          # saved document index; None disables
    response_cache: bool = True               # the app's answer cache (evaluation runs never use it)
    semantic_threshold: float | None = 0.95   # cosine for a semantic hit; None = exact match only
    cache_ttl_days: float = 7
    admin: bool = False                       # shows the cache stats + "Clear cache" button


def _copy_streamlit_secrets_to_env() -> None:
    """On Streamlit Cloud, secrets live in st.secrets; mirror top-level strings into env."""
    try:
        import streamlit as st

        for key, value in st.secrets.items():
            if isinstance(value, str) and value and not os.environ.get(key):
                os.environ[key] = value
    except Exception:  # no secrets file locally, or not running under streamlit
        pass


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


_OFF = {"0", "off", "false", "no", "none", ""}


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() not in _OFF


def _optional(name: str, default: str) -> str | None:
    value = os.environ.get(name, default).strip()
    return None if value.lower() in _OFF else value


def load_settings() -> Settings:
    load_dotenv()
    _copy_streamlit_secrets_to_env()
    return Settings(
        openai_api_key=_env("OPENAI_API_KEY"),
        langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=_env("LANGFUSE_BASE_URL", _env("LANGFUSE_HOST", "https://cloud.langfuse.com")),
        chat_model=_env("TAXMATE_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=_env("TAXMATE_EMBED_MODEL", "text-embedding-3-small"),
        log_level=_env("TAXMATE_LOG_LEVEL", "INFO"),
        cache_dir=_optional("TAXMATE_CACHE_DIR", ".cache"),
        response_cache=_flag("TAXMATE_RESPONSE_CACHE", True),
        semantic_threshold=(float(v) if (v := _optional("TAXMATE_SEMANTIC_THRESHOLD", "0.95")) else None),
        cache_ttl_days=float(_env("TAXMATE_CACHE_TTL_DAYS", "7")),
        admin=_flag("TAXMATE_ADMIN", False),
    )


def missing_keys(s: Settings) -> list[str]:
    values = {"OPENAI_API_KEY": s.openai_api_key}
    return [k for k in REQUIRED_KEYS if not values[k]]

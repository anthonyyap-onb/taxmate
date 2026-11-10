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
    langfuse_host: str
    chat_model: str
    embed_model: str
    log_level: str


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


def load_settings() -> Settings:
    load_dotenv()
    _copy_streamlit_secrets_to_env()
    return Settings(
        openai_api_key=_env("OPENAI_API_KEY"),
        langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
        langfuse_host=_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        chat_model=_env("TAXMATE_CHAT_MODEL", "gpt-4o-mini"),
        embed_model=_env("TAXMATE_EMBED_MODEL", "text-embedding-3-small"),
        log_level=_env("TAXMATE_LOG_LEVEL", "INFO"),
    )


def missing_keys(s: Settings) -> list[str]:
    values = {"OPENAI_API_KEY": s.openai_api_key}
    return [k for k in REQUIRED_KEYS if not values[k]]

"""Shared helpers for evaluation runners. Every runner uses the same pipeline code as the app."""
from __future__ import annotations

import json
import os
from pathlib import Path

from taxmate import tracing
from taxmate.config import Settings, load_settings, missing_keys

ROOT = Path(__file__).resolve().parent.parent


def load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def init_live():
    """Returns (deps, settings) wired to OpenAI, honouring TAXMATE_THRESHOLD / TAXMATE_K overrides."""
    from taxmate.live import build_live_deps

    s: Settings = load_settings()
    if missing_keys(s):
        raise SystemExit(f"Missing configuration: {', '.join(missing_keys(s))}. Fill in .env first.")
    tracing.init(s)
    threshold = float(os.environ["TAXMATE_THRESHOLD"]) if os.environ.get("TAXMATE_THRESHOLD") else None
    k = int(os.environ["TAXMATE_K"]) if os.environ.get("TAXMATE_K") else None
    deps, _ = build_live_deps(s, threshold=threshold, k=k)
    return deps, s


def run_dir(name: str) -> Path:
    d = ROOT / "evals" / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d

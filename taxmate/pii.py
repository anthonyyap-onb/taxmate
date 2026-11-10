"""Mask personal identifiers before text reaches the LLM, traces, or logs."""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<!\d)(?:\+63|0)9\d{2}[- ]?\d{3}[- ]?\d{4}(?!\d)")
_TIN_SEPARATED = re.compile(r"(?<!\d)\d{3}[- ]\d{3}[- ]\d{3}(?:[- ]\d{3,5})?(?!\d)")
# Unseparated 9/12/14-digit runs, unless preceded by a peso marker (₱, P, PHP, Php).
_TIN_PLAIN = re.compile(r"(?<![\d,.])(?<!₱)(?<!₱ )(?<!\bP)(?<!\bP )(?<!PHP)(?<!PHP )(?<!Php)(?<!Php )"
                        r"(?:\d{14}|\d{12}|\d{9})(?![\d,.]\d|\d)")


def mask_pii(text: str) -> tuple[str, list[str]]:
    found: set[str] = set()

    def sub(pattern: re.Pattern, placeholder: str, kind: str, s: str) -> str:
        def repl(_m: re.Match) -> str:
            found.add(kind)
            return placeholder
        return pattern.sub(repl, s)

    out = sub(_EMAIL, "[EMAIL]", "email", text)
    out = sub(_PHONE, "[PHONE]", "phone", out)
    out = sub(_TIN_SEPARATED, "[TIN]", "tin", out)
    out = sub(_TIN_PLAIN, "[TIN]", "tin", out)
    return out, sorted(found)

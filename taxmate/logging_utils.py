"""One JSON log line per turn on stdout (visible in Streamlit Cloud 'Manage app -> Logs')."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

logger = logging.getLogger("taxmate")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_handler)
logger.setLevel(os.environ.get("TAXMATE_LOG_LEVEL", "INFO").upper())


def log_turn(record: dict) -> None:
    record = dict(record)
    record.setdefault("ts", datetime.now(timezone.utc).isoformat())
    logger.info(json.dumps(record, ensure_ascii=False, default=str))

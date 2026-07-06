"""Single logging entry point. Stdlib logging with a consistent format."""
from __future__ import annotations

import logging
import os
import re
import sys

_CONFIGURED = False
_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"

# Redact HTTP basic-auth credentials embedded in URLs (e.g. a DagsHub/MLflow
# tracking URI carrying a token) before they can be written to logs.
_CREDS_IN_URL = re.compile(r"(https?://)[^/\s:@]+:[^/\s@]+@")


def scrub_secrets(text: object) -> str:
    """Redact `user:pass@` credentials from a string (for safe error logging)."""
    return _CREDS_IN_URL.sub(r"\1***:***@", str(text))


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    # An unknown LOG_LEVEL must not crash import of every module that logs.
    lvl = getattr(logging, lvl_name, None)
    if not isinstance(lvl, int):
        lvl = logging.INFO
    root = logging.getLogger()
    root.setLevel(lvl)
    # Only attach our handler when root has none. Under uvicorn/gunicorn (which
    # install their own root handler) adding another would emit every line twice.
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(handler)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)

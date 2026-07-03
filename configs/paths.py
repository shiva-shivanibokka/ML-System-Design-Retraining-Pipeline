"""
Shared filesystem path helpers.

Centralizes temp-file/temp-dir creation so callers don't hardcode "/tmp"
(which doesn't exist on Windows) and so artifacts land in one place that's
easy to find/clean up.
"""

from __future__ import annotations

import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def temp_dir() -> Path:
    """Return this process's scratch directory, creating it if needed."""
    d = Path(tempfile.gettempdir()) / "retraining_pipeline"
    d.mkdir(parents=True, exist_ok=True)
    return d


def temp_file(prefix: str = "", suffix: str = "") -> Path:
    """Return a unique path inside temp_dir() for a scratch file (not created)."""
    return temp_dir() / f"{prefix}{uuid.uuid4().hex}{suffix}"


def utcnow_naive() -> datetime:
    """Timezone-aware UTC 'now' with tzinfo stripped — safe to compare against
    naive datetimes parsed from batch_date strings."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

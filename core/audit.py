"""
core/audit.py – Append-only local audit logging for KIBAD.

Every data operation (load, transform, export, analysis) is recorded as a
JSON-Lines entry in ``.kibad/audit.log`` relative to the working directory.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_LOG_DIR = Path(".kibad")
_LOG_FILE = _LOG_DIR / "audit.log"
_lock = threading.Lock()


def _ensure_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_event(
    event_type: str,
    details: dict[str, Any] | str | None = None,
    *,
    dataset: str | None = None,
    user: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Append an audit event and return the record.

    Parameters
    ----------
    event_type:
        Category such as ``"file_loaded"``, ``"transform_applied"``,
        ``"export_created"``, ``"analysis_run"``.
    details:
        Free-form dictionary or string with event-specific information.
    dataset:
        Optional dataset name to include in the details.
    user:
        Optional user identifier; falls back to ``$USER`` env var.
    **extra:
        Additional key-value pairs merged into details.

    Returns
    -------
    dict  — the written record (useful for tests).
    """
    if isinstance(details, str):
        details_dict: dict[str, Any] = {"message": details}
    else:
        details_dict = dict(details) if details else {}
    if dataset is not None:
        details_dict["dataset"] = dataset
    details_dict.update(extra)

    record: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "user": user or os.environ.get("USER", "unknown"),
        "details": details_dict,
    }

    with _lock:
        _ensure_dir()
        with open(_LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    return record


def read_log(last_n: int | None = None) -> list[dict[str, Any]]:
    """Read audit log entries.

    Parameters
    ----------
    last_n:
        If set, return only the most recent *n* entries.

    Returns
    -------
    list[dict]
    """
    if not _LOG_FILE.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(_LOG_FILE, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if last_n is not None:
        entries = entries[-last_n:]
    return entries


def clear_log() -> None:
    """Remove the audit log file (use with caution)."""
    if _LOG_FILE.exists():
        _LOG_FILE.unlink()


def log_file_path() -> Path:
    """Return the absolute path to the audit log file."""
    _ensure_dir()
    return _LOG_FILE.resolve()

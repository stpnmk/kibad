"""
tests/test_audit.py – Unit tests for core/audit.py

The audit module writes to .kibad/audit.log relative to CWD,
so tests use monkeypatch to redirect CWD to a tmp directory.
"""
import json
import os
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import core.audit as audit_mod
from core.audit import log_event, read_log, clear_log, log_file_path


# ---------------------------------------------------------------------------
# Fixture: redirect audit paths to a temp directory
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_audit(tmp_path, monkeypatch):
    """Redirect the audit module's internal paths to a temp directory."""
    tmp_log_dir = tmp_path / ".kibad"
    tmp_log_file = tmp_log_dir / "audit.log"
    monkeypatch.setattr(audit_mod, "_LOG_DIR", tmp_log_dir)
    monkeypatch.setattr(audit_mod, "_LOG_FILE", tmp_log_file)
    yield
    # cleanup happens automatically via tmp_path


# ---------------------------------------------------------------------------
# log_event
# ---------------------------------------------------------------------------

def test_log_event_returns_dict():
    record = log_event("test_event")
    assert isinstance(record, dict)


def test_log_event_has_required_fields():
    record = log_event("file_loaded", details={"filename": "test.csv"})
    assert "timestamp" in record
    assert "event" in record
    assert "user" in record
    assert "details" in record
    assert record["event"] == "file_loaded"
    assert record["details"]["filename"] == "test.csv"


def test_log_event_custom_user():
    record = log_event("test_event", user="testuser")
    assert record["user"] == "testuser"


def test_log_event_default_user():
    record = log_event("test_event")
    assert isinstance(record["user"], str)
    assert len(record["user"]) > 0


def test_log_event_creates_file():
    log_event("first_event")
    assert audit_mod._LOG_FILE.exists()


def test_log_event_appends_to_file():
    log_event("event_1")
    log_event("event_2")
    with open(audit_mod._LOG_FILE) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 2


def test_log_event_writes_valid_json():
    log_event("json_test", details={"key": "value"})
    with open(audit_mod._LOG_FILE) as f:
        line = f.readline().strip()
    parsed = json.loads(line)
    assert parsed["event"] == "json_test"


# ---------------------------------------------------------------------------
# read_log
# ---------------------------------------------------------------------------

def test_read_log_empty_when_no_events():
    entries = read_log()
    assert entries == []


def test_read_log_returns_list_of_dicts():
    log_event("e1")
    log_event("e2")
    entries = read_log()
    assert isinstance(entries, list)
    assert len(entries) == 2
    assert all(isinstance(e, dict) for e in entries)


def test_read_log_last_n():
    for i in range(5):
        log_event(f"event_{i}")
    entries = read_log(last_n=2)
    assert len(entries) == 2
    assert entries[-1]["event"] == "event_4"


def test_read_log_last_n_larger_than_total():
    log_event("only_one")
    entries = read_log(last_n=100)
    assert len(entries) == 1


# ---------------------------------------------------------------------------
# clear_log
# ---------------------------------------------------------------------------

def test_clear_log_removes_file():
    log_event("to_be_cleared")
    assert audit_mod._LOG_FILE.exists()
    clear_log()
    assert not audit_mod._LOG_FILE.exists()


def test_clear_log_on_nonexistent_file():
    # Should not raise if file doesn't exist
    clear_log()


def test_clear_log_then_read_returns_empty():
    log_event("event")
    clear_log()
    entries = read_log()
    assert entries == []


# ---------------------------------------------------------------------------
# log_file_path
# ---------------------------------------------------------------------------

def test_log_file_path_returns_path():
    path = log_file_path()
    assert isinstance(path, Path)


def test_log_file_path_is_absolute():
    path = log_file_path()
    assert path.is_absolute()


def test_log_file_path_creates_directory():
    # Ensure the .kibad directory is created
    path = log_file_path()
    assert path.parent.exists()

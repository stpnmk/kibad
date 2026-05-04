"""
app/state.py – Dash session state management for KIBAD.

Replaces the Streamlit session_state approach with dcc.Store components.
DataFrames too large for browser localStorage are serialized to temporary
files under ``data/session/`` and only the file path is stored in the Store.

Store ID naming convention: ``store-<noun>``
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Session directory for large data (DataFrames serialized to Parquet)
# ---------------------------------------------------------------------------
SESSION_DIR = Path(__file__).parent.parent / "data" / "session"
SESSION_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Store IDs (constants used across pages and main.py)
# ---------------------------------------------------------------------------
STORE_DATASET = "store-dataset"           # {name: path_to_parquet}
STORE_PREPARED = "store-prepared"         # {name: path_to_parquet}
STORE_ACTIVE_DS = "store-active-ds"       # str: active dataset name
STORE_LANG = "store-lang"                 # "ru" | "en"
STORE_AUDIT = "store-audit"              # list of audit entries
STORE_FORECAST = "store-forecast"         # list of forecast result dicts
STORE_TEST_RESULTS = "store-test-results" # list of test result dicts
STORE_ATTRIBUTION = "store-attribution"   # list of attribution result dicts
STORE_AGG_RESULTS = "store-agg-results"   # {name: path_to_parquet}
STORE_REPORT = "store-report"             # last report HTML string
STORE_SIDEBAR = "store-sidebar-collapsed" # bool

ALL_STORES = [
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS, STORE_LANG,
    STORE_AUDIT, STORE_FORECAST, STORE_TEST_RESULTS, STORE_ATTRIBUTION,
    STORE_AGG_RESULTS, STORE_REPORT, STORE_SIDEBAR,
]

# Default values for each store
STORE_DEFAULTS = {
    STORE_DATASET: {},
    STORE_PREPARED: {},
    STORE_ACTIVE_DS: None,
    STORE_LANG: "ru",
    STORE_AUDIT: [],
    STORE_FORECAST: [],
    STORE_TEST_RESULTS: [],
    STORE_ATTRIBUTION: [],
    STORE_AGG_RESULTS: {},
    STORE_REPORT: "",
    STORE_SIDEBAR: False,
}


# ---------------------------------------------------------------------------
# DataFrame serialization helpers
# ---------------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, name: str) -> str:
    """Save a DataFrame to a Parquet file and return the file path."""
    safe_name = hashlib.md5(name.encode()).hexdigest()[:12]
    path = SESSION_DIR / f"{safe_name}.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def load_dataframe(path: str) -> pd.DataFrame | None:
    """Load a DataFrame from a Parquet file path."""
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        logger.exception("Failed to read Parquet file: %s", path)
        return None


def get_df_from_store(store_data: dict | None, name: str) -> pd.DataFrame | None:
    """Retrieve a DataFrame by name from a store dict of {name: path}."""
    if not store_data or name not in store_data:
        return None
    return load_dataframe(store_data[name])


def get_df_from_stores(name: str, *stores) -> "pd.DataFrame | None":
    """Try each store in order, return first non-None DataFrame (avoids `or` on DataFrames)."""
    for store in stores:
        df = get_df_from_store(store, name)
        if df is not None:
            return df
    return None


def list_datasets(store_data: dict | None) -> list[str]:
    """List dataset names from a store dict."""
    if not store_data:
        return []
    return list(store_data.keys())


def cleanup_session():
    """Remove old session files (call on shutdown)."""
    for f in SESSION_DIR.glob("*.parquet"):
        try:
            f.unlink()
        except OSError:
            pass

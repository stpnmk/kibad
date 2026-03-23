"""
app/state.py – Shared Streamlit session state helpers for KIBAD.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd


def init_state() -> None:
    """Initialise all required session state keys if not yet present."""
    defaults: dict = {
        "datasets": {},         # name → pd.DataFrame
        "active_ds": None,      # str: active dataset name
        "col_mappings": {},     # name → {logical: real_col}
        "type_overrides": {},   # name → {col: type_str}
        "prepared_dfs": {},     # name → cleaned/resampled df
        "forecast_results": [], # list of ForecastResult
        "test_results": [],     # list of TestResult
        "scenario_presets": {}, # name → ScenarioPreset json
        "report_sections": [],  # accumulated report builder sections
        "kpi_defs": [],         # list of {label, formula}
        "transform_logs": {},   # name → TransformLog
        "trigger_rules": [],    # list of TriggerRule
        "attribution_results": [],  # list of AttributionResult
        "aggregate_results": {},    # name → pd.DataFrame
        "lang": "ru",           # UI language
        "data_quality_reports": {},         # ds_name → qc dict
        "preprocessing_recommendations": {},  # ds_name → list[dict]
        "fc_guide_seen": False,
        "acf_sarimax_suggestion": None,
        "pending_prepare_actions": {},
        "auto_scan_done": set(),
        "onboarding_done": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def get_active_df() -> pd.DataFrame | None:
    """Return the currently active (prepared or raw) DataFrame."""
    name = st.session_state.get("active_ds")
    if not name:
        return None
    # Prefer prepared version
    prep = st.session_state.get("prepared_dfs", {})
    if name in prep:
        return prep[name]
    return st.session_state.get("datasets", {}).get(name)


def get_active_name() -> str | None:
    return st.session_state.get("active_ds")


def store_dataset(name: str, df: pd.DataFrame, source: str = "upload") -> None:
    """Store a raw DataFrame in session state."""
    st.session_state["datasets"][name] = df
    if st.session_state["active_ds"] is None:
        st.session_state["active_ds"] = name


def store_prepared(name: str, df: pd.DataFrame) -> None:
    """Store a prepared (cleaned/resampled) DataFrame."""
    st.session_state["prepared_dfs"][name] = df


def list_dataset_names() -> list[str]:
    return list(st.session_state.get("datasets", {}).keys())


def get_dataset(name: str) -> pd.DataFrame | None:
    """Return a dataset by name (prepared version preferred)."""
    prep = st.session_state.get("prepared_dfs", {})
    if name in prep:
        return prep[name]
    return st.session_state.get("datasets", {}).get(name)


def add_dataset(name: str, df: pd.DataFrame) -> None:
    """Add or replace a dataset (stores as raw and prepared)."""
    st.session_state["datasets"][name] = df
    st.session_state["prepared_dfs"][name] = df
    if st.session_state.get("active_ds") is None:
        st.session_state["active_ds"] = name


def invalidate_dataset_cache(ds_name: str) -> None:
    """Remove all cached analysis artefacts for *ds_name*.

    Centralises cache invalidation so pages don't need scattered .pop() calls.
    """
    for cache_key in (
        "data_quality_reports",
        "quality_scores",
        "auto_insights",
        "auto_analysis",
    ):
        st.session_state.get(cache_key, {}).pop(ds_name, None)


def has_unsaved_prepared(ds_name: str) -> bool:
    """Return True if *ds_name* has a prepared version that differs from raw.

    Used to warn users before switching datasets so they don't lose prepare work.
    """
    raw = st.session_state.get("datasets", {}).get(ds_name)
    prep = st.session_state.get("prepared_dfs", {}).get(ds_name)
    if raw is None or prep is None:
        return False
    # Different shape or content means unsaved preparation work
    if raw.shape != prep.shape:
        return True
    return False


def clear_analysis_results(ds_name: str) -> None:
    """Drop forecast/test/attribution results that reference *ds_name*.

    Call when a dataset is replaced or significantly modified so stale
    analysis results aren't shown alongside new data.
    """
    for key in ("forecast_results", "test_results", "attribution_results"):
        results = st.session_state.get(key, [])
        st.session_state[key] = [
            r for r in results
            if not (isinstance(r, dict) and r.get("dataset") == ds_name)
        ]
    invalidate_dataset_cache(ds_name)


def dataset_selectbox(label: str = "Select dataset", key: str = "ds_select", help: str | None = None) -> str | None:
    """Render a selectbox for choosing a dataset and return its name."""
    names = list_dataset_names()
    if not names:
        st.info("Датасеты не загружены. Перейдите на страницу **Данные** для загрузки.")
        return None
    idx = 0
    active = st.session_state.get("active_ds")
    if active in names:
        idx = names.index(active)
    chosen = st.selectbox(label, names, index=idx, key=key, help=help)
    st.session_state["active_ds"] = chosen
    return chosen

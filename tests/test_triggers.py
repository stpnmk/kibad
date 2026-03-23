"""
tests/test_triggers.py – Unit tests for core/triggers.py
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.triggers import (
    TriggerRule, Alert, evaluate_triggers, alerts_to_dataframe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _steady_df(n=20, value=100.0):
    """Steady time series with no anomalies."""
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    return pd.DataFrame({"date": dates, "metric": [value] * n})


def _spike_df(n=20, spike_idx=15, spike_value=500.0):
    """Time series with a single large spike."""
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    vals = [100.0] * n
    vals[spike_idx] = spike_value
    return pd.DataFrame({"date": dates, "metric": vals})


def _trend_reversal_df(n=20, turn_idx=10):
    """Time series that goes up then down — slope sign change."""
    dates = pd.date_range("2023-01-01", periods=n, freq="MS")
    vals = list(range(turn_idx)) + list(range(turn_idx, turn_idx - (n - turn_idx), -1))
    return pd.DataFrame({"date": dates, "metric": [float(v) for v in vals]})


# ---------------------------------------------------------------------------
# TriggerRule and Alert dataclasses
# ---------------------------------------------------------------------------

def test_trigger_rule_default_active():
    rule = TriggerRule(name="test", rule_type="threshold_cross", params={"upper": 100})
    assert rule.active is True


def test_trigger_rule_inactive():
    rule = TriggerRule(name="test", rule_type="threshold_cross", params={}, active=False)
    assert rule.active is False


# ---------------------------------------------------------------------------
# threshold_cross
# ---------------------------------------------------------------------------

def test_threshold_upper_fires():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [10, 20, 150, 30, 200],
    })
    rule = TriggerRule("upper_check", "threshold_cross", {"upper": 100})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 2
    assert all(a.value > 100 for a in alerts)


def test_threshold_lower_fires():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [50, 10, 60, 5, 70],
    })
    rule = TriggerRule("lower_check", "threshold_cross", {"lower": 20})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 2
    assert all(a.value < 20 for a in alerts)


def test_threshold_no_fire_when_in_range():
    df = _steady_df(10, value=50.0)
    rule = TriggerRule("range", "threshold_cross", {"upper": 100, "lower": 0})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 0


def test_threshold_upper_and_lower_both():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=4, freq="MS"),
        "metric": [5, 50, 200, 50],
    })
    rule = TriggerRule("both", "threshold_cross", {"upper": 100, "lower": 10})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 2  # 5 < 10 and 200 > 100


# ---------------------------------------------------------------------------
# deviation_from_baseline
# ---------------------------------------------------------------------------

def test_deviation_fires_on_spike():
    df = _spike_df(n=20, spike_idx=15, spike_value=500.0)
    rule = TriggerRule("spike", "deviation_from_baseline", {"window": 6, "n_sigma": 2.0})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) >= 1
    spike_alerts = [a for a in alerts if a.value == 500.0]
    assert len(spike_alerts) >= 1


def test_deviation_no_fire_on_steady():
    df = _steady_df(20, value=100.0)
    rule = TriggerRule("no_spike", "deviation_from_baseline", {"window": 6, "n_sigma": 2.0})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 0


def test_deviation_respects_n_sigma():
    """Higher n_sigma threshold should produce fewer or equal alerts."""
    df = _spike_df(n=20, spike_idx=15, spike_value=300.0)
    rule_loose = TriggerRule("loose", "deviation_from_baseline", {"window": 6, "n_sigma": 1.0})
    rule_strict = TriggerRule("strict", "deviation_from_baseline", {"window": 6, "n_sigma": 5.0})
    alerts_loose = evaluate_triggers(df, "date", "metric", [rule_loose])
    alerts_strict = evaluate_triggers(df, "date", "metric", [rule_strict])
    assert len(alerts_strict) <= len(alerts_loose)


# ---------------------------------------------------------------------------
# slope_change
# ---------------------------------------------------------------------------

def test_slope_change_fires_on_reversal():
    df = _trend_reversal_df(n=20, turn_idx=10)
    rule = TriggerRule("slope_flip", "slope_change", {"window": 6, "threshold": 0.0})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) >= 1


def test_slope_change_no_fire_on_monotonic():
    dates = pd.date_range("2023-01-01", periods=20, freq="MS")
    vals = [float(i) for i in range(20)]
    df = pd.DataFrame({"date": dates, "metric": vals})
    rule = TriggerRule("mono", "slope_change", {"window": 6, "threshold": 0.0})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 0


# ---------------------------------------------------------------------------
# evaluate_triggers — multiple rules
# ---------------------------------------------------------------------------

def test_evaluate_mixed_rules():
    df = _spike_df(n=20, spike_idx=15, spike_value=500.0)
    rules = [
        TriggerRule("upper", "threshold_cross", {"upper": 400}),
        TriggerRule("deviation", "deviation_from_baseline", {"window": 6, "n_sigma": 2.0}),
    ]
    alerts = evaluate_triggers(df, "date", "metric", rules)
    rule_names = {a.rule.name for a in alerts}
    assert "upper" in rule_names
    assert "deviation" in rule_names


def test_evaluate_triggers_sorted_by_index():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [200, 50, 200, 50, 200],
    })
    rule = TriggerRule("upper", "threshold_cross", {"upper": 100})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    indices = [a.index for a in alerts]
    assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Inactive rules
# ---------------------------------------------------------------------------

def test_inactive_rule_is_skipped():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [200, 200, 200, 200, 200],
    })
    rule = TriggerRule("inactive", "threshold_cross", {"upper": 100}, active=False)
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    assert len(alerts) == 0


def test_mixed_active_inactive():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [200, 200, 200, 200, 200],
    })
    rules = [
        TriggerRule("active_rule", "threshold_cross", {"upper": 100}, active=True),
        TriggerRule("inactive_rule", "threshold_cross", {"upper": 100}, active=False),
    ]
    alerts = evaluate_triggers(df, "date", "metric", rules)
    rule_names = {a.rule.name for a in alerts}
    assert "active_rule" in rule_names
    assert "inactive_rule" not in rule_names


# ---------------------------------------------------------------------------
# alerts_to_dataframe
# ---------------------------------------------------------------------------

def test_alerts_to_dataframe_columns():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [200, 50, 200, 50, 200],
    })
    rule = TriggerRule("test", "threshold_cross", {"upper": 100})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    adf = alerts_to_dataframe(alerts)
    expected_cols = {"rule_name", "rule_type", "index", "timestamp", "value", "message"}
    assert expected_cols == set(adf.columns)


def test_alerts_to_dataframe_empty():
    adf = alerts_to_dataframe([])
    assert len(adf) == 0
    assert "rule_name" in adf.columns


def test_alerts_to_dataframe_row_count():
    df = pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=5, freq="MS"),
        "metric": [200, 50, 200, 50, 200],
    })
    rule = TriggerRule("test", "threshold_cross", {"upper": 100})
    alerts = evaluate_triggers(df, "date", "metric", [rule])
    adf = alerts_to_dataframe(alerts)
    assert len(adf) == len(alerts)

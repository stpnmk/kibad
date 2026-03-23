"""
tests/test_attribution.py – Unit tests for core/attribution.py
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.attribution import (
    additive_attribution, multiplicative_attribution,
    regression_attribution, shapley_attribution,
    waterfall_data, AttributionResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_df():
    """Two-row DataFrame with current and previous values for target and drivers."""
    return pd.DataFrame({
        "target": [120, 130],
        "target_prev": [100, 110],
        "d1": [50, 60],
        "d1_prev": [40, 50],
        "d2": [30, 35],
        "d2_prev": [25, 30],
    })


def _segment_df():
    """DataFrame with a segment column."""
    return pd.DataFrame({
        "segment": ["East", "East", "West", "West"],
        "target": [120, 130, 80, 90],
        "target_prev": [100, 110, 70, 80],
        "d1": [50, 60, 30, 40],
        "d1_prev": [40, 50, 25, 35],
        "d2": [30, 35, 20, 25],
        "d2_prev": [25, 30, 15, 20],
    })


def _regression_df(n=50, seed=42):
    """DataFrame with correlated target and drivers for regression."""
    rng = np.random.default_rng(seed)
    d1 = rng.normal(100, 10, n)
    d2 = rng.normal(50, 5, n)
    noise = rng.normal(0, 2, n)
    target = 2 * d1 + 3 * d2 + noise
    d1_prev = d1 - rng.normal(5, 1, n)
    d2_prev = d2 - rng.normal(3, 1, n)
    target_prev = 2 * d1_prev + 3 * d2_prev + rng.normal(0, 2, n)
    return pd.DataFrame({
        "target": target, "target_prev": target_prev,
        "d1": d1, "d1_prev": d1_prev,
        "d2": d2, "d2_prev": d2_prev,
    })


# ---------------------------------------------------------------------------
# Additive attribution
# ---------------------------------------------------------------------------

def test_additive_returns_attribution_result():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert isinstance(result, AttributionResult)
    assert result.method == "additive"


def test_additive_contributions_sum_to_delta():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    contrib_sum = result.contributions["contribution"].sum() + result.residual
    assert abs(contrib_sum - result.target_delta) < 1e-6


def test_additive_target_delta_correct():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    expected_delta = (120 + 130) - (100 + 110)
    assert abs(result.target_delta - expected_delta) < 1e-6


def test_additive_contributions_has_driver_column():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert "driver" in result.contributions.columns
    assert "contribution" in result.contributions.columns
    assert "pct_of_delta" in result.contributions.columns
    assert len(result.contributions) == 2


def test_additive_mismatched_cols_raises():
    df = _simple_df()
    with pytest.raises(ValueError):
        additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev"])


# ---------------------------------------------------------------------------
# Additive attribution with segments
# ---------------------------------------------------------------------------

def test_additive_with_segment_returns_segment_detail():
    df = _segment_df()
    result = additive_attribution(
        df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"],
        segment_col="segment",
    )
    assert result.segment_detail is not None
    assert "segment" in result.segment_detail.columns
    assert len(result.segment_detail) == 2  # East, West


def test_additive_without_segment_has_no_segment_detail():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert result.segment_detail is None


# ---------------------------------------------------------------------------
# Multiplicative attribution
# ---------------------------------------------------------------------------

def test_multiplicative_returns_correct_method():
    df = _simple_df()
    result = multiplicative_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert result.method == "multiplicative"


def test_multiplicative_contributions_sum_approx_delta():
    df = _simple_df()
    result = multiplicative_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    contrib_sum = result.contributions["contribution"].sum() + result.residual
    assert abs(contrib_sum - result.target_delta) < 1e-6


def test_multiplicative_positive_values():
    df = pd.DataFrame({
        "target": [200, 300],
        "target_prev": [100, 150],
        "d1": [20, 30],
        "d1_prev": [10, 15],
        "d2": [10, 10],
        "d2_prev": [10, 10],
    })
    result = multiplicative_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert result.target_delta == 250.0
    # d1 changed, d2 did not, so d1 should have larger contribution
    d1_contrib = result.contributions[result.contributions["driver"] == "d1"]["contribution"].values[0]
    d2_contrib = result.contributions[result.contributions["driver"] == "d2"]["contribution"].values[0]
    assert abs(d1_contrib) > abs(d2_contrib)


# ---------------------------------------------------------------------------
# Regression attribution
# ---------------------------------------------------------------------------

def test_regression_returns_correct_method():
    df = _regression_df()
    result = regression_attribution(df, "target", ["d1", "d2"])
    assert result.method == "regression"


def test_regression_r_squared_positive():
    df = _regression_df()
    result = regression_attribution(df, "target", ["d1", "d2"])
    assert result.metadata["r_squared"] > 0


def test_regression_with_prev_cols():
    df = _regression_df()
    result = regression_attribution(
        df, "target", ["d1", "d2"],
        target_prev_col="target_prev",
        driver_prev_cols=["d1_prev", "d2_prev"],
    )
    contrib_sum = result.contributions["contribution"].sum() + result.residual
    assert abs(contrib_sum - result.target_delta) < 1.0  # Allow some tolerance


def test_regression_too_few_observations_raises():
    df = pd.DataFrame({"target": [1], "d1": [2], "d2": [3]})
    with pytest.raises(ValueError, match="Not enough observations"):
        regression_attribution(df, "target", ["d1", "d2"])


# ---------------------------------------------------------------------------
# Shapley attribution
# ---------------------------------------------------------------------------

def test_shapley_returns_correct_method():
    df = _simple_df()
    result = shapley_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert result.method == "shapley"


def test_shapley_contributions_sum_approx_delta():
    df = _simple_df()
    result = shapley_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    contrib_sum = result.contributions["contribution"].sum() + result.residual
    assert abs(contrib_sum - result.target_delta) < 1e-3


def test_shapley_exact_for_small_n():
    df = _simple_df()
    result = shapley_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    assert result.metadata.get("exact") is True


def test_shapley_mismatched_cols_raises():
    df = _simple_df()
    with pytest.raises(ValueError):
        shapley_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev"])


# ---------------------------------------------------------------------------
# waterfall_data
# ---------------------------------------------------------------------------

def test_waterfall_data_categories_and_values():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1", "d2"], ["d1_prev", "d2_prev"])
    cats, vals = waterfall_data(result)
    assert isinstance(cats, list)
    assert isinstance(vals, list)
    assert len(cats) == len(vals)
    assert "d1" in cats
    assert "d2" in cats


def test_waterfall_data_includes_residual_when_nonzero():
    df = _simple_df()
    result = additive_attribution(df, "target", "target_prev", ["d1"], ["d1_prev"])
    cats, vals = waterfall_data(result)
    if abs(result.residual) > 1e-10:
        assert "Residual" in cats


def test_waterfall_data_no_residual_when_zero():
    # Create a case where contributions perfectly explain the delta
    df = pd.DataFrame({
        "target": [20],
        "target_prev": [10],
        "d1": [20],
        "d1_prev": [10],
    })
    result = additive_attribution(df, "target", "target_prev", ["d1"], ["d1_prev"])
    cats, vals = waterfall_data(result)
    if abs(result.residual) < 1e-10:
        assert "Residual" not in cats

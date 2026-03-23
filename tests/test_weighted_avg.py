"""
tests/test_weighted_avg.py – Tests for core/weighted_avg.py
"""
import math
import numpy as np
import pandas as pd
import pytest

from core.weighted_avg import (
    weighted_average,
    weighted_std,
    weighted_percentile,
    portfolio_weighted_averages,
    mix_rate_decomposition,
    simplified_duration,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def portfolio_df():
    return pd.DataFrame({
        "balance": [1000.0, 2000.0, 500.0, 3000.0],
        "rate": [5.0, 6.5, 4.0, 7.0],
        "ltv": [0.60, 0.75, 0.55, 0.80],
        "maturity": [36, 60, 24, 48],
        "segment": ["A", "B", "A", "B"],
    })


@pytest.fixture
def equal_df():
    """Equal weights → weighted avg == simple avg."""
    return pd.DataFrame({
        "weight": [1.0, 1.0, 1.0],
        "value": [3.0, 5.0, 7.0],
    })


# ---------------------------------------------------------------------------
# weighted_average tests
# ---------------------------------------------------------------------------

def test_weighted_average_basic(portfolio_df):
    total_w = portfolio_df["balance"].sum()
    expected = (portfolio_df["rate"] * portfolio_df["balance"]).sum() / total_w
    result = weighted_average(portfolio_df["rate"], portfolio_df["balance"])
    assert result == pytest.approx(expected, rel=1e-5)


def test_weighted_average_equal_weights(equal_df):
    wa = weighted_average(equal_df["value"], equal_df["weight"])
    assert wa == pytest.approx(5.0)


def test_weighted_average_zero_weights():
    values = pd.Series([1.0, 2.0, 3.0])
    weights = pd.Series([0.0, 0.0, 0.0])
    result = weighted_average(values, weights)
    assert math.isnan(result)


def test_weighted_average_single_value():
    values = pd.Series([42.0])
    weights = pd.Series([100.0])
    assert weighted_average(values, weights) == pytest.approx(42.0)


def test_weighted_average_nan_values():
    values = pd.Series([1.0, np.nan, 3.0])
    weights = pd.Series([1.0, 1.0, 1.0])
    result = weighted_average(values, weights)
    # Should ignore NaN and average 1.0 and 3.0 → 2.0
    assert result == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# weighted_std tests
# ---------------------------------------------------------------------------

def test_weighted_std_uniform_returns_0():
    """All same values → std = 0."""
    values = pd.Series([5.0, 5.0, 5.0])
    weights = pd.Series([1.0, 2.0, 3.0])
    result = weighted_std(values, weights)
    assert result == pytest.approx(0.0, abs=1e-9)


def test_weighted_std_positive(portfolio_df):
    result = weighted_std(portfolio_df["rate"], portfolio_df["balance"])
    assert result > 0


def test_weighted_std_zero_weights():
    values = pd.Series([1.0, 2.0])
    weights = pd.Series([0.0, 0.0])
    result = weighted_std(values, weights)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# weighted_percentile tests
# ---------------------------------------------------------------------------

def test_weighted_percentile_median_equal_weights():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0])
    median = weighted_percentile(values, weights, q=0.5)
    assert median == pytest.approx(3.0, abs=0.1)


def test_weighted_percentile_q0_is_min():
    values = pd.Series([10.0, 20.0, 30.0])
    weights = pd.Series([1.0, 1.0, 1.0])
    result = weighted_percentile(values, weights, q=0.0)
    assert result == pytest.approx(10.0, abs=1.0)


def test_weighted_percentile_zero_weights():
    values = pd.Series([1.0, 2.0])
    weights = pd.Series([0.0, 0.0])
    result = weighted_percentile(values, weights, q=0.5)
    assert math.isnan(result)


# ---------------------------------------------------------------------------
# portfolio_weighted_averages tests
# ---------------------------------------------------------------------------

def test_portfolio_weighted_averages_no_groups(portfolio_df):
    result = portfolio_weighted_averages(portfolio_df, "balance", ["rate", "ltv"])
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert "rate_wa" in result.columns
    assert "ltv_wa" in result.columns


def test_portfolio_weighted_averages_with_groups(portfolio_df):
    result = portfolio_weighted_averages(portfolio_df, "balance", ["rate"], group_cols=["segment"])
    assert len(result) == 2  # Segments A and B


def test_portfolio_weighted_averages_has_weight_sum(portfolio_df):
    result = portfolio_weighted_averages(portfolio_df, "balance", ["rate"])
    assert "balance_sum" in result.columns
    assert result["balance_sum"].iloc[0] == pytest.approx(portfolio_df["balance"].sum())


def test_portfolio_weighted_averages_empty_df():
    empty = pd.DataFrame({"balance": [], "rate": [], "segment": []})
    result = portfolio_weighted_averages(empty, "balance", ["rate"])
    assert result.empty


def test_portfolio_weighted_averages_wa_formula(portfolio_df):
    """Verify WAR formula manually for group A."""
    result = portfolio_weighted_averages(portfolio_df, "balance", ["rate"], group_cols=["segment"])
    seg_a = result[result["segment"] == "A"]

    df_a = portfolio_df[portfolio_df["segment"] == "A"]
    expected_war = (df_a["rate"] * df_a["balance"]).sum() / df_a["balance"].sum()

    assert float(seg_a["rate_wa"].iloc[0]) == pytest.approx(expected_war, rel=1e-5)


# ---------------------------------------------------------------------------
# mix_rate_decomposition tests
# ---------------------------------------------------------------------------

def test_mix_rate_decomposition_returns_df(portfolio_df):
    df_a = portfolio_df.copy()
    df_b = portfolio_df.copy()
    df_b["rate"] = df_b["rate"] * 1.1  # Rate increases by 10%
    result = mix_rate_decomposition(df_a, df_b, "balance", "rate", "segment")
    assert isinstance(result, pd.DataFrame)
    assert "mix_effect" in result.columns
    assert "rate_effect" in result.columns


def test_mix_rate_identical_dfs(portfolio_df):
    """Identical datasets → total effect should be ~0."""
    result = mix_rate_decomposition(portfolio_df, portfolio_df, "balance", "rate", "segment")
    total = result["total_effect"].sum()
    assert abs(total) < 1e-9


def test_mix_rate_decomposition_groups(portfolio_df):
    df_b = portfolio_df.copy()
    result = mix_rate_decomposition(portfolio_df, df_b, "balance", "rate", "segment")
    assert set(result["group"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# simplified_duration tests
# ---------------------------------------------------------------------------

def test_simplified_duration_returns_dict():
    result = simplified_duration(war=5.0, wam_months=36.0)
    assert isinstance(result, dict)
    assert "macaulay_years" in result
    assert "modified_duration" in result
    assert "dv01" in result


def test_simplified_duration_macaulay():
    result = simplified_duration(war=6.0, wam_months=24.0)
    assert result["macaulay_years"] == pytest.approx(2.0)


def test_simplified_duration_modified_less_than_macaulay():
    """Modified duration should be less than Macaulay duration."""
    result = simplified_duration(war=5.0, wam_months=60.0)
    assert result["modified_duration"] < result["macaulay_years"]


def test_simplified_duration_dv01_negative():
    """DV01 = -modified/10000 should be negative."""
    result = simplified_duration(war=5.0, wam_months=36.0)
    assert result["dv01"] < 0

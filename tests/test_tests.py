"""
tests/test_tests.py – Unit tests for core/tests.py
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tests import (
    ttest_independent, mann_whitney, chi_square_independence,
    correlation_test, bootstrap_test, ab_test, lag_correlation, TestResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

np.random.seed(42)
GROUP_A = pd.Series(np.random.normal(50, 10, 100))
GROUP_B = pd.Series(np.random.normal(60, 10, 100))  # clearly different
GROUP_SAME = pd.Series(np.random.normal(50, 10, 100))


# ---------------------------------------------------------------------------
# t-test
# ---------------------------------------------------------------------------

def test_ttest_detects_difference():
    result = ttest_independent(GROUP_A, GROUP_B, alpha=0.05)
    assert isinstance(result, TestResult)
    assert result.significant == True
    assert result.p_value < 0.05
    assert result.effect_size is not None


def test_ttest_no_difference():
    result = ttest_independent(GROUP_A, GROUP_SAME, alpha=0.05)
    # Not guaranteed but very likely with matching distributions
    assert result.p_value > 0.001  # should not be ~0


def test_ttest_has_ci():
    result = ttest_independent(GROUP_A, GROUP_B)
    assert result.ci is not None
    assert len(result.ci) == 2
    assert result.ci[0] < result.ci[1]


def test_ttest_interpretation_is_string():
    result = ttest_independent(GROUP_A, GROUP_B)
    assert isinstance(result.interpretation, str)
    assert len(result.interpretation) > 20


def test_ttest_requires_min_obs():
    with pytest.raises(ValueError):
        ttest_independent(pd.Series([1]), pd.Series([1, 2]))


def test_ttest_handles_nas():
    a = pd.Series([1, 2, None, 4, 5, 6, 7, 8, 9, 10])
    b = pd.Series([10, 20, 30, None, 40, 50, 60, 70, 80, 90])
    result = ttest_independent(a, b)
    assert isinstance(result, TestResult)


# ---------------------------------------------------------------------------
# Mann-Whitney
# ---------------------------------------------------------------------------

def test_mannwhitney_significant():
    a = pd.Series(np.random.normal(50, 5, 100))
    b = pd.Series(np.random.normal(65, 5, 100))
    result = mann_whitney(a, b)
    assert result.significant == True


def test_mannwhitney_effect_size_range():
    result = mann_whitney(GROUP_A, GROUP_B)
    assert -1 <= result.effect_size <= 1


# ---------------------------------------------------------------------------
# Chi-square
# ---------------------------------------------------------------------------

def test_chisquare_independent():
    np.random.seed(1)
    df = pd.DataFrame({
        "color": np.random.choice(["red", "blue"], 200),
        "preference": np.random.choice(["yes", "no"], 200),
    })
    result = chi_square_independence(df, "color", "preference")
    assert isinstance(result, TestResult)
    assert 0.0 <= result.p_value <= 1.0


def test_chisquare_strong_association():
    df = pd.DataFrame({
        "a": ["x"] * 50 + ["y"] * 50,
        "b": ["p"] * 50 + ["q"] * 50,
    })
    result = chi_square_independence(df, "a", "b")
    assert result.significant == True


def test_chisquare_requires_2_categories():
    df = pd.DataFrame({"a": ["x"] * 10, "b": ["p", "q"] * 5})
    with pytest.raises(ValueError):
        chi_square_independence(df, "a", "b")


# ---------------------------------------------------------------------------
# Correlation test
# ---------------------------------------------------------------------------

def test_correlation_pearson():
    x = pd.Series(np.arange(50, dtype=float))
    y = 2 * x + np.random.normal(0, 1, 50)
    result = correlation_test(x, y, method="pearson")
    assert result.significant == True
    assert result.effect_size > 0.9


def test_correlation_spearman():
    x = pd.Series([1, 2, 3, 4, 5] * 20)
    y = pd.Series([10, 9, 8, 7, 6] * 20) + np.random.normal(0, 0.1, 100)
    result = correlation_test(x, y, method="spearman")
    assert result.effect_size < 0  # negative correlation


def test_correlation_requires_3_obs():
    with pytest.raises(ValueError):
        correlation_test(pd.Series([1, 2]), pd.Series([3, 4]))


# ---------------------------------------------------------------------------
# Bootstrap test
# ---------------------------------------------------------------------------

def test_bootstrap_detects_difference():
    result = bootstrap_test(GROUP_A, GROUP_B, statistic="mean", n_bootstrap=1000)
    assert result.significant == True


def test_bootstrap_median_statistic():
    result = bootstrap_test(GROUP_A, GROUP_B, statistic="median", n_bootstrap=500)
    assert isinstance(result.p_value, float)


def test_bootstrap_ci_valid():
    result = bootstrap_test(GROUP_A, GROUP_B, n_bootstrap=500)
    assert result.ci is not None
    assert isinstance(result.ci[0], float)


# ---------------------------------------------------------------------------
# A/B test
# ---------------------------------------------------------------------------

def test_ab_test_returns_all_keys():
    res = ab_test(GROUP_A, GROUP_B)
    assert "ttest" in res
    assert "mann_whitney" in res
    assert "bootstrap" in res
    assert "summary" in res
    assert "lift_pct" in res


def test_ab_test_lift_positive():
    ctrl = pd.Series(np.ones(100) * 10)
    trt = pd.Series(np.ones(100) * 11)
    res = ab_test(ctrl, trt)
    assert res["lift_pct"] == pytest.approx(10.0, abs=0.01)


# ---------------------------------------------------------------------------
# Lag correlation
# ---------------------------------------------------------------------------

def test_lag_correlation_shape():
    x = pd.Series(np.arange(50, dtype=float))
    y = pd.Series(np.arange(50, dtype=float))
    result = lag_correlation(x, y, max_lag=6)
    assert len(result) == 7  # lags 0..6
    assert "lag" in result.columns
    assert "correlation" in result.columns
    assert "p_value" in result.columns


def test_lag_correlation_lag0_is_1():
    x = pd.Series(np.arange(1, 51, dtype=float))
    y = pd.Series(np.arange(1, 51, dtype=float))
    result = lag_correlation(x, y, max_lag=3)
    assert abs(result.loc[result["lag"] == 0, "correlation"].values[0] - 1.0) < 1e-6

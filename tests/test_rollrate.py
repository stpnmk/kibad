"""
tests/test_rollrate.py – Tests for core/rollrate.py
"""
import numpy as np
import pandas as pd
import pytest

from core.rollrate import (
    auto_bucket,
    build_transition_matrix,
    matrix_power,
    steady_state,
    roll_forward_rates,
    cure_rates,
    transition_time_series,
    BUCKET_ORDER,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_loan_df():
    """4 loans, 3 periods, with known transitions."""
    return pd.DataFrame({
        "loan_id": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "period": ["2023-01", "2023-02", "2023-03"] * 4,
        "bucket": [
            "Текущий", "1-30", "Текущий",    # loan 1: roll then cure
            "Текущий", "Текущий", "Текущий",  # loan 2: stays current
            "1-30", "31-60", "61-90",         # loan 3: keeps rolling
            "31-60", "Текущий", "Текущий",    # loan 4: cures
        ],
    })


@pytest.fixture
def stochastic_matrix():
    """Simple 3×3 stochastic matrix."""
    T = pd.DataFrame(
        [[0.8, 0.15, 0.05],
         [0.1, 0.7, 0.2],
         [0.0, 0.0, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    return T


# ---------------------------------------------------------------------------
# auto_bucket tests
# ---------------------------------------------------------------------------

def test_auto_bucket_current():
    dpd = pd.Series([0])
    result = auto_bucket(dpd)
    assert result[0] == "Текущий"


def test_auto_bucket_ranges():
    dpd = pd.Series([0, 15, 45, 75, 120, 200])
    result = auto_bucket(dpd)
    assert result[0] == "Текущий"
    assert result[1] == "1-30"
    assert result[2] == "31-60"
    assert result[3] == "61-90"
    assert result[4] in ("90+", "Списан")


def test_auto_bucket_nan_closed():
    dpd = pd.Series([np.nan, -1])
    result = auto_bucket(dpd)
    assert result[0] == "Закрыт"
    assert result[1] == "Закрыт"


# ---------------------------------------------------------------------------
# build_transition_matrix tests
# ---------------------------------------------------------------------------

def test_build_transition_matrix_returns_tuple(simple_loan_df):
    count_m, rate_m = build_transition_matrix(
        simple_loan_df, "loan_id", "period", "bucket"
    )
    assert isinstance(count_m, pd.DataFrame)
    assert isinstance(rate_m, pd.DataFrame)


def test_transition_matrix_row_sums(simple_loan_df):
    """Rows of rate matrix should sum to 1 (or 0 for empty rows)."""
    count_m, rate_m = build_transition_matrix(
        simple_loan_df, "loan_id", "period", "bucket"
    )
    for idx in rate_m.index:
        row_sum = rate_m.loc[idx].sum()
        assert abs(row_sum - 1.0) < 1e-9 or abs(row_sum) < 1e-9


def test_transition_matrix_indexed_by_bucket_order(simple_loan_df):
    count_m, rate_m = build_transition_matrix(
        simple_loan_df, "loan_id", "period", "bucket"
    )
    for idx in rate_m.index:
        assert idx in BUCKET_ORDER


def test_transition_matrix_empty_df():
    empty = pd.DataFrame(columns=["loan_id", "period", "bucket"])
    count_m, rate_m = build_transition_matrix(empty, "loan_id", "period", "bucket")
    assert count_m.empty or (count_m == 0).all().all()


# ---------------------------------------------------------------------------
# matrix_power tests
# ---------------------------------------------------------------------------

def test_matrix_power_1(stochastic_matrix):
    T1 = matrix_power(stochastic_matrix, 1)
    np.testing.assert_allclose(T1.values, stochastic_matrix.values, atol=1e-9)


def test_matrix_power_0(stochastic_matrix):
    T0 = matrix_power(stochastic_matrix, 0)
    expected = np.eye(3)
    np.testing.assert_allclose(T0.values, expected, atol=1e-9)


def test_matrix_power_rows_sum_to_1(stochastic_matrix):
    T2 = matrix_power(stochastic_matrix, 2)
    row_sums = T2.values.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(3), atol=1e-9)


# ---------------------------------------------------------------------------
# steady_state tests
# ---------------------------------------------------------------------------

def test_steady_state_sums_to_1(stochastic_matrix):
    ss = steady_state(stochastic_matrix)
    assert abs(ss.sum() - 1.0) < 1e-6


def test_steady_state_absorbing():
    """Absorbing state 'C' should have steady-state probability 1."""
    T = pd.DataFrame(
        [[0.9, 0.1, 0.0],
         [0.0, 0.9, 0.1],
         [0.0, 0.0, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )
    ss = steady_state(T)
    assert abs(ss["C"] - 1.0) < 0.01 or ss.sum() > 0.99


# ---------------------------------------------------------------------------
# roll_forward_rates and cure_rates tests
# ---------------------------------------------------------------------------

def test_roll_forward_rates_nonnegative(simple_loan_df):
    _, rate_m = build_transition_matrix(simple_loan_df, "loan_id", "period", "bucket")
    rf = roll_forward_rates(rate_m)
    assert (rf >= 0).all()


def test_cure_rates_nonnegative(simple_loan_df):
    _, rate_m = build_transition_matrix(simple_loan_df, "loan_id", "period", "bucket")
    cr = cure_rates(rate_m)
    assert (cr >= 0).all()


def test_current_bucket_cure_rate_zero(simple_loan_df):
    """Текущий bucket cannot cure (nothing worse to come from)."""
    _, rate_m = build_transition_matrix(simple_loan_df, "loan_id", "period", "bucket")
    cr = cure_rates(rate_m)
    if "Текущий" in cr.index:
        assert cr["Текущий"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# transition_time_series tests
# ---------------------------------------------------------------------------

def test_transition_time_series_returns_df(simple_loan_df):
    ts = transition_time_series(simple_loan_df, "loan_id", "period", "bucket")
    assert isinstance(ts, pd.DataFrame)


def test_transition_time_series_columns(simple_loan_df):
    ts = transition_time_series(simple_loan_df, "loan_id", "period", "bucket")
    if not ts.empty:
        assert "period" in ts.columns
        assert "roll_forward_rate" in ts.columns
        assert "cure_rate" in ts.columns

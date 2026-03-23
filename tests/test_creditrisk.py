"""
tests/test_creditrisk.py – Tests for core/creditrisk.py
"""
import numpy as np
import pandas as pd
import pytest

from core.creditrisk import (
    compute_el,
    compute_ecl,
    compute_npl,
    hhi,
    top_n_concentration,
    portfolio_summary,
    ead_weighted_avg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def risk_df():
    return pd.DataFrame({
        "ead": [1000.0, 2000.0, 500.0, 3000.0],
        "pd": [0.01, 0.05, 0.10, 0.02],
        "lgd": [0.40, 0.45, 0.50, 0.35],
        "dpd": [0, 45, 120, 5],
        "stage": [1, 2, 3, 1],
        "provisions": [5.0, 60.0, 200.0, 25.0],
        "segment": ["A", "B", "A", "B"],
    })


@pytest.fixture
def concentrated_df():
    """Single group → HHI = 1."""
    return pd.DataFrame({
        "ead": [100.0, 200.0, 300.0],
        "group": ["ACME", "ACME", "ACME"],
    })


@pytest.fixture
def three_group_df():
    """Three equal groups → HHI = 1/3."""
    return pd.DataFrame({
        "ead": [100.0, 100.0, 100.0],
        "group": ["X", "Y", "Z"],
    })


# ---------------------------------------------------------------------------
# compute_el tests
# ---------------------------------------------------------------------------

def test_compute_el_basic(risk_df):
    el = compute_el(risk_df, "pd", "lgd", "ead")
    assert len(el) == 4
    # Row 0: 0.01 * 0.40 * 1000 = 4.0
    assert el.iloc[0] == pytest.approx(4.0)


def test_compute_el_all_zeros():
    df = pd.DataFrame({"pd": [0.0], "lgd": [0.5], "ead": [1000.0]})
    el = compute_el(df, "pd", "lgd", "ead")
    assert el.iloc[0] == 0.0


def test_compute_el_shape(risk_df):
    el = compute_el(risk_df, "pd", "lgd", "ead")
    assert len(el) == len(risk_df)


# ---------------------------------------------------------------------------
# compute_npl tests
# ---------------------------------------------------------------------------

def test_compute_npl_basic(risk_df):
    result = compute_npl(risk_df, "ead", "dpd", dpd_threshold=90)
    # DPD=120 → NPL, row 2 has ead=500
    assert result["npl_amount"] == pytest.approx(500.0)
    assert result["total_ead"] == pytest.approx(6500.0)
    assert result["npl_rate"] == pytest.approx(500.0 / 6500.0 * 100, rel=1e-4)


def test_compute_npl_zero_ead():
    df = pd.DataFrame({"ead": [0.0], "dpd": [100.0]})
    result = compute_npl(df, "ead", "dpd")
    assert result["npl_rate"] == 0.0


# ---------------------------------------------------------------------------
# HHI tests
# ---------------------------------------------------------------------------

def test_hhi_single_group_equals_1(concentrated_df):
    result = hhi(concentrated_df, "ead", "group")
    assert result["hhi"] == pytest.approx(1.0)
    assert result["hhi_normalized"] == pytest.approx(1.0)


def test_hhi_equal_groups(three_group_df):
    result = hhi(three_group_df, "ead", "group")
    expected_hhi = 1.0 / 3.0
    assert result["hhi"] == pytest.approx(expected_hhi, rel=1e-4)


def test_hhi_interpretation(risk_df):
    result = hhi(risk_df, "ead", "segment")
    assert result["interpretation"] in ("низкая", "умеренная", "высокая")


def test_hhi_zero_ead():
    df = pd.DataFrame({"ead": [0.0, 0.0], "group": ["A", "B"]})
    result = hhi(df, "ead", "group")
    assert np.isnan(result["hhi"])


# ---------------------------------------------------------------------------
# top_n_concentration tests
# ---------------------------------------------------------------------------

def test_top_n_returns_df(risk_df):
    result = top_n_concentration(risk_df, "ead", "segment", n=2)
    assert isinstance(result, pd.DataFrame)
    assert len(result) <= 2


def test_top_n_sorted_desc(risk_df):
    result = top_n_concentration(risk_df, "ead", "segment", n=2)
    if len(result) > 1:
        assert result["ead_sum"].iloc[0] >= result["ead_sum"].iloc[1]


def test_top_n_cumulative_share_max(risk_df):
    result = top_n_concentration(risk_df, "ead", "segment", n=10)
    if not result.empty:
        assert result["cumulative_share_pct"].max() <= 100.01


# ---------------------------------------------------------------------------
# compute_ecl tests
# ---------------------------------------------------------------------------

def test_compute_ecl_stage1(risk_df):
    """Stage 1 ECL = PD_12m * LGD * EAD."""
    ecl = compute_ecl(risk_df, "ead", "lgd", "pd")
    expected_s1 = risk_df["pd"] * risk_df["lgd"] * risk_df["ead"]
    # Without stages, all rows are stage 1
    np.testing.assert_allclose(ecl.values, expected_s1.values, rtol=1e-5)


def test_compute_ecl_with_stages(risk_df):
    """Stage 3 ECL = LGD * EAD."""
    ecl = compute_ecl(risk_df, "ead", "lgd", "pd", stage_col="stage")
    # Row 2: stage=3, ECL = 0.50 * 500 = 250
    assert ecl.iloc[2] == pytest.approx(250.0)


# ---------------------------------------------------------------------------
# ead_weighted_avg tests
# ---------------------------------------------------------------------------

def test_ead_weighted_avg_basic(risk_df):
    wa = ead_weighted_avg(risk_df, "pd", "ead")
    total_ead = risk_df["ead"].sum()
    expected = (risk_df["pd"] * risk_df["ead"]).sum() / total_ead
    assert wa == pytest.approx(expected, rel=1e-5)


def test_ead_weighted_avg_single_row():
    df = pd.DataFrame({"val": [5.0], "ead": [100.0]})
    assert ead_weighted_avg(df, "val", "ead") == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# portfolio_summary tests
# ---------------------------------------------------------------------------

def test_portfolio_summary_returns_dict(risk_df):
    result = portfolio_summary(risk_df, "ead", "pd", "lgd")
    assert isinstance(result, dict)
    assert "total_ead" in result
    assert "total_el" in result


def test_portfolio_summary_total_ead(risk_df):
    result = portfolio_summary(risk_df, "ead", "pd", "lgd")
    assert result["total_ead"] == pytest.approx(6500.0)


def test_portfolio_summary_with_dpd(risk_df):
    result = portfolio_summary(risk_df, "ead", "pd", "lgd", dpd_col="dpd")
    assert result["npl_rate"] is not None
    assert result["npl_rate"] > 0


def test_portfolio_summary_stage_breakdown(risk_df):
    result = portfolio_summary(risk_df, "ead", "pd", "lgd", stage_col="stage")
    sb = result["stage_breakdown"]
    assert not sb.empty
    assert "stage" in sb.columns

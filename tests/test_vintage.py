"""
tests/test_vintage.py – Tests for core/vintage.py
"""
import numpy as np
import pandas as pd
import pytest

from core.vintage import build_vintage_pivot, compute_mob, wilson_ci, vintage_summary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    """Minimal vintage dataset: 10 loans, 3 months observation."""
    records = []
    base_date = pd.Timestamp("2023-01-01")
    for loan_id in range(10):
        orig = base_date
        for mob_month in range(4):  # MOB 0-3
            obs = orig + pd.DateOffset(months=mob_month)
            # Loans 0-1 default at MOB 2
            default_flag = 1 if (loan_id < 2 and mob_month >= 2) else 0
            records.append({
                "loan_id": loan_id,
                "orig_date": orig,
                "obs_date": obs,
                "default": default_flag,
            })
    return pd.DataFrame(records)


@pytest.fixture
def two_cohort_df():
    """Two cohorts: Jan and Feb origination."""
    records = []
    for cohort_month in [1, 2]:
        orig = pd.Timestamp(f"2023-0{cohort_month}-01")
        for loan_id in range(5):
            for mob in range(5):
                obs = orig + pd.DateOffset(months=mob)
                # Some defaults at MOB 3+
                default_flag = 1 if (loan_id < 1 and mob >= 3) else 0
                records.append({
                    "orig_date": orig,
                    "obs_date": obs,
                    "default": default_flag,
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# compute_mob tests
# ---------------------------------------------------------------------------

def test_compute_mob_basic():
    df = pd.DataFrame({
        "orig": ["2023-01-01", "2023-01-01"],
        "obs": ["2023-03-01", "2023-06-01"],
    })
    mob = compute_mob(df, "orig", "obs")
    assert list(mob) == [2, 5]


def test_compute_mob_negative_clipped():
    df = pd.DataFrame({
        "orig": ["2023-06-01"],
        "obs": ["2023-01-01"],  # obs before orig
    })
    mob = compute_mob(df, "orig", "obs")
    assert int(mob.iloc[0]) == 0


def test_compute_mob_same_month():
    df = pd.DataFrame({
        "orig": ["2023-03-15"],
        "obs": ["2023-03-28"],
    })
    mob = compute_mob(df, "orig", "obs")
    assert int(mob.iloc[0]) == 0


# ---------------------------------------------------------------------------
# build_vintage_pivot tests
# ---------------------------------------------------------------------------

def test_vintage_pivot_returns_dict(simple_df):
    result = build_vintage_pivot(simple_df, "orig_date", "obs_date", "default", max_mob=6)
    assert isinstance(result, dict)
    for key in ["cdr_pivot", "mdr_pivot", "count_pivot", "cohort_sizes", "maturity_mask"]:
        assert key in result


def test_vintage_pivot_cdr_shape(simple_df):
    result = build_vintage_pivot(simple_df, "orig_date", "obs_date", "default", max_mob=6)
    cdr = result["cdr_pivot"]
    assert not cdr.empty
    # Max MOB column should be <= 6
    assert max(cdr.columns) <= 6


def test_vintage_cdr_is_cumulative(simple_df):
    """CDR at MOB 3 should be >= CDR at MOB 2 (never decreasing for monotone defaults)."""
    result = build_vintage_pivot(simple_df, "orig_date", "obs_date", "default", max_mob=6)
    cdr = result["cdr_pivot"]
    if 2 in cdr.columns and 3 in cdr.columns:
        for idx in cdr.index:
            v2 = cdr.loc[idx, 2]
            v3 = cdr.loc[idx, 3]
            if not (np.isnan(v2) or np.isnan(v3)):
                assert v3 >= v2 - 1e-9, f"CDR not monotone at cohort {idx}: MOB2={v2}, MOB3={v3}"


def test_vintage_pivot_empty_df():
    empty_df = pd.DataFrame(columns=["orig", "obs", "default"])
    result = build_vintage_pivot(empty_df, "orig", "obs", "default", max_mob=6)
    assert result["cdr_pivot"].empty


def test_vintage_cdr_values_between_0_and_1(two_cohort_df):
    result = build_vintage_pivot(two_cohort_df, "orig_date", "obs_date", "default", max_mob=6)
    cdr = result["cdr_pivot"]
    vals = cdr.values.flatten()
    vals = vals[~np.isnan(vals)]
    assert (vals >= 0).all()
    assert (vals <= 1.0001).all()


def test_maturity_mask_shape(simple_df):
    result = build_vintage_pivot(simple_df, "orig_date", "obs_date", "default", max_mob=6, min_obs_threshold=5)
    mask = result["maturity_mask"]
    cdr = result["cdr_pivot"]
    assert mask.shape == cdr.shape


# ---------------------------------------------------------------------------
# wilson_ci tests
# ---------------------------------------------------------------------------

def test_wilson_ci_basic():
    lo, hi = wilson_ci(0.1, 100)
    assert 0 <= lo < 0.1 < hi <= 1
    assert hi > lo


def test_wilson_ci_zero_sample():
    lo, hi = wilson_ci(0.5, 0)
    assert lo == 0.0
    assert hi == 1.0


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(0.0, 50)
    assert lo >= 0
    lo, hi = wilson_ci(1.0, 50)
    assert hi <= 1.0


# ---------------------------------------------------------------------------
# vintage_summary tests
# ---------------------------------------------------------------------------

def test_vintage_summary_returns_df(two_cohort_df):
    result = build_vintage_pivot(two_cohort_df, "orig_date", "obs_date", "default", max_mob=6)
    summary = vintage_summary(result["cdr_pivot"], result["maturity_mask"])
    assert isinstance(summary, pd.DataFrame)
    assert "max_cdr" in summary.columns


def test_vintage_summary_empty():
    summary = vintage_summary(pd.DataFrame(), pd.DataFrame())
    assert summary.empty


def test_vintage_summary_row_count(two_cohort_df):
    result = build_vintage_pivot(two_cohort_df, "orig_date", "obs_date", "default", max_mob=6)
    summary = vintage_summary(result["cdr_pivot"], result["maturity_mask"])
    assert len(summary) == len(result["cdr_pivot"])

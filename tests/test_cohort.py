"""
tests/test_cohort.py – Tests for core/cohort.py
"""
import numpy as np
import pandas as pd
import pytest

from core.cohort import (
    build_cohort_table,
    retention_table,
    churn_rate_table,
    average_retention_curve,
    compute_clv,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_activity_df():
    """50 customers with 6 months of activity. All acquired Jan 2023."""
    records = []
    base = pd.Timestamp("2023-01-01")
    for cid in range(50):
        for month in range(6):
            # Customers 0-39 active all 6 months; 40-49 only months 0-3
            if cid >= 40 and month > 3:
                continue
            records.append({
                "customer_id": cid,
                "activity_date": base + pd.DateOffset(months=month),
            })
    return pd.DataFrame(records)


@pytest.fixture
def two_cohort_df():
    """Two cohorts: Jan and Feb 2023, 10 customers each."""
    records = []
    for cohort_month in [1, 2]:
        base = pd.Timestamp(f"2023-0{cohort_month}-01")
        for cid in range(10):
            for month in range(5):
                if cid >= 8 and month > 2:
                    continue
                records.append({
                    "customer_id": f"c{cohort_month}_{cid}",
                    "activity_date": base + pd.DateOffset(months=month),
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# build_cohort_table tests
# ---------------------------------------------------------------------------

def test_build_cohort_table_returns_df(simple_activity_df):
    result = build_cohort_table(simple_activity_df, "customer_id", "activity_date")
    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_build_cohort_offset_0_at_100_pct(simple_activity_df):
    """At offset=0, all customers should be present (retention must be 100%)."""
    cohort_counts = build_cohort_table(simple_activity_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    assert 0 in retention.columns
    for cohort in retention.index:
        val = retention.loc[cohort, 0]
        assert abs(val - 1.0) < 1e-9, f"Retention at offset=0 should be 1.0, got {val}"


def test_build_cohort_table_empty():
    empty_df = pd.DataFrame(columns=["customer_id", "activity_date"])
    result = build_cohort_table(empty_df, "customer_id", "activity_date")
    assert result.empty


def test_build_cohort_table_two_cohorts(two_cohort_df):
    result = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    assert len(result) == 2


def test_build_cohort_table_max_offset(simple_activity_df):
    result = build_cohort_table(simple_activity_df, "customer_id", "activity_date", max_offset=3)
    assert max(result.columns) == 3


def test_cohort_table_customer_count_at_offset0(simple_activity_df):
    """At offset=0, all 50 customers should be present for Jan cohort."""
    result = build_cohort_table(simple_activity_df, "customer_id", "activity_date")
    assert 0 in result.columns
    total_at_0 = result[0].sum()
    assert total_at_0 == 50


# ---------------------------------------------------------------------------
# retention_table tests
# ---------------------------------------------------------------------------

def test_retention_table_offset0_is_1(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    for cohort in retention.index:
        if 0 in retention.columns:
            assert abs(retention.loc[cohort, 0] - 1.0) < 1e-9


def test_retention_table_values_between_0_and_1(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    vals = retention.values.flatten()
    vals = vals[~np.isnan(vals)]
    assert (vals >= 0).all()
    assert (vals <= 1.0001).all()


def test_retention_table_nonincreasing(simple_activity_df):
    """Retention should be non-increasing for cohorts that don't re-acquire."""
    cohort_counts = build_cohort_table(simple_activity_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    cols = sorted(retention.columns)
    for cohort in retention.index:
        prev = None
        for col in cols:
            val = retention.loc[cohort, col]
            if not np.isnan(val):
                if prev is not None:
                    assert val <= prev + 1e-9
                prev = val


# ---------------------------------------------------------------------------
# churn_rate_table tests
# ---------------------------------------------------------------------------

def test_churn_rate_offset0_is_nan(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    churn = churn_rate_table(retention)
    if 0 in churn.columns:
        assert churn[0].isna().all()


def test_churn_rate_nonnegative(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    churn = churn_rate_table(retention)
    for col in churn.columns:
        if col == 0:
            continue
        vals = churn[col].dropna()
        assert (vals >= -1e-9).all()


# ---------------------------------------------------------------------------
# average_retention_curve tests
# ---------------------------------------------------------------------------

def test_average_retention_curve_returns_series(simple_activity_df):
    cohort_counts = build_cohort_table(simple_activity_df, "customer_id", "activity_date")
    avg = average_retention_curve(cohort_counts)
    assert isinstance(avg, pd.Series)
    assert not avg.empty


def test_average_retention_offset0_is_1(simple_activity_df):
    cohort_counts = build_cohort_table(simple_activity_df, "customer_id", "activity_date")
    avg = average_retention_curve(cohort_counts)
    assert 0 in avg.index
    assert abs(avg[0] - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_clv tests
# ---------------------------------------------------------------------------

def test_compute_clv_returns_series(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    clv = compute_clv(retention, arpu=100.0, annual_discount_rate=0.12, horizon_months=12)
    assert isinstance(clv, pd.Series)
    assert not clv.empty


def test_compute_clv_zero_arpu(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    clv = compute_clv(retention, arpu=0.0, annual_discount_rate=0.12, horizon_months=12)
    assert clv.empty


def test_compute_clv_positive(two_cohort_df):
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    clv = compute_clv(retention, arpu=100.0, annual_discount_rate=0.12, horizon_months=12)
    assert (clv > 0).all()


def test_compute_clv_at_least_arpu(two_cohort_df):
    """CLV should be at least ARPU (retention at offset=0 is 100%)."""
    cohort_counts = build_cohort_table(two_cohort_df, "customer_id", "activity_date")
    retention = retention_table(cohort_counts)
    clv = compute_clv(retention, arpu=100.0, annual_discount_rate=0.0, horizon_months=24)
    # With 0% discount and full retention at 0, CLV >= 100
    assert (clv >= 100.0 - 1e-6).all()

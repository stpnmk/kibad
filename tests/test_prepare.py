"""
tests/test_prepare.py – Unit tests for core/prepare.py
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.prepare import (
    parse_dates, resample_timeseries, impute_missing, remove_outliers,
    flag_outliers, deduplicate, add_lags, add_rolling, add_ema,
    add_buckets, normalize, add_interaction, RESAMPLE_ALIASES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_df():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    np.random.seed(0)
    return pd.DataFrame({
        "date": dates,
        "value": np.random.randint(100, 500, 24).astype(float),
        "group": ["A"] * 12 + ["B"] * 12,
    })


@pytest.fixture
def sales_df():
    dates = pd.date_range("2021-01-01", periods=12, freq="D")
    return pd.DataFrame({
        "date": dates,
        "sales": [10, 20, 15, None, 25, 30, None, 22, 18, 12, 28, 35],
    })


# ---------------------------------------------------------------------------
# parse_dates
# ---------------------------------------------------------------------------

def test_parse_dates_string_col():
    df = pd.DataFrame({"d": ["2023-01-01", "2023-02-01", "2023-03-01"]})
    result = parse_dates(df, "d")
    assert pd.api.types.is_datetime64_any_dtype(result["d"])


def test_parse_dates_tz_strip():
    df = pd.DataFrame({"d": pd.to_datetime(["2023-01-01", "2023-02-01"]).tz_localize("UTC")})
    result = parse_dates(df, "d", tz_strip=True)
    assert result["d"].dt.tz is None


def test_parse_dates_invalid_graceful():
    df = pd.DataFrame({"d": ["not-a-date", "2023-01-01"]})
    result = parse_dates(df, "d")
    assert pd.isna(result["d"].iloc[0])


# ---------------------------------------------------------------------------
# resample_timeseries
# ---------------------------------------------------------------------------

def test_resample_monthly(monthly_df):
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    result = resample_timeseries(monthly_df, "date", ["value"], freq="MS")
    assert "date" in result.columns
    assert "value" in result.columns
    assert len(result) <= 24


def test_resample_weekly():
    dates = pd.date_range("2023-01-01", periods=52, freq="D")
    df = pd.DataFrame({"date": dates, "val": np.arange(52, dtype=float)})
    result = resample_timeseries(df, "date", ["val"], freq="W-MON")
    # Weekly should produce fewer rows than daily
    assert len(result) < 52


def test_resample_with_group(monthly_df):
    monthly_df["date"] = pd.to_datetime(monthly_df["date"])
    result = resample_timeseries(monthly_df, "date", ["value"],
                                  freq="MS", group_cols=["group"])
    groups = result["group"].unique()
    assert "A" in groups and "B" in groups


def test_resample_aliases_defined():
    assert "Weekly (Mon)" in RESAMPLE_ALIASES
    assert RESAMPLE_ALIASES["Weekly (Mon)"] == "W-MON"
    assert "Monthly Start" in RESAMPLE_ALIASES
    assert RESAMPLE_ALIASES["Monthly Start"] == "MS"


# ---------------------------------------------------------------------------
# impute_missing
# ---------------------------------------------------------------------------

def test_impute_median():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
    result = impute_missing(df, ["x"], method="median")
    assert result["x"].isna().sum() == 0
    assert result["x"].iloc[2] == 3.0


def test_impute_mean():
    df = pd.DataFrame({"x": [0.0, 10.0, None]})
    result = impute_missing(df, ["x"], method="mean")
    assert result["x"].iloc[2] == 5.0


def test_impute_ffill():
    df = pd.DataFrame({"x": [1.0, None, None, 4.0]})
    result = impute_missing(df, ["x"], method="ffill")
    assert result["x"].iloc[1] == 1.0
    assert result["x"].iloc[2] == 1.0


def test_impute_drop():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    result = impute_missing(df, ["x"], method="drop")
    assert len(result) == 2


def test_impute_zero():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    result = impute_missing(df, ["x"], method="zero")
    assert result["x"].iloc[1] == 0.0


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------

def test_flag_outliers_iqr():
    s = pd.Series([1, 2, 3, 4, 100])  # 100 is extreme
    mask = flag_outliers(s, method="iqr")
    assert mask.iloc[-1] == True
    assert mask.iloc[0] == False


def test_flag_outliers_zscore():
    # Need enough points so 100 is clearly beyond 2 std from mean
    s = pd.Series([1.0, 2.0, 2.1, 1.9, 2.0, 1.8, 2.2, 1.7, 2.3, 100.0])
    mask = flag_outliers(s, method="zscore", zscore_threshold=2.0)
    assert mask.iloc[-1] == True


def test_remove_outliers_returns_count():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100, 200]})
    cleaned, n_rem = remove_outliers(df, ["x"], method="iqr")
    assert n_rem > 0
    assert len(cleaned) < len(df)


# ---------------------------------------------------------------------------
# Deduplicate
# ---------------------------------------------------------------------------

def test_deduplicate_basic():
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [10, 10, 20, 30]})
    result, n = deduplicate(df)
    assert n == 1
    assert len(result) == 3


def test_deduplicate_subset():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    result, n = deduplicate(df, subset=["a"])
    assert n == 1


# ---------------------------------------------------------------------------
# Lag / Rolling / EMA
# ---------------------------------------------------------------------------

def test_add_lags():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = add_lags(df, "x", lags=[1, 2])
    assert "x_lag1" in result.columns
    assert "x_lag2" in result.columns
    assert result["x_lag1"].iloc[1] == 1.0
    assert pd.isna(result["x_lag2"].iloc[0])


def test_add_rolling():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = add_rolling(df, "x", windows=[3], func="mean")
    assert "x_roll3_mean" in result.columns
    # Rolling mean at index 2 (window=3): (1+2+3)/3 = 2
    assert abs(result["x_roll3_mean"].iloc[2] - 2.0) < 1e-6


def test_add_ema():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    result = add_ema(df, "x", spans=[3])
    assert "x_ema3" in result.columns
    assert result["x_ema3"].notna().all()


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------

def test_add_buckets_quantile():
    df = pd.DataFrame({"x": list(range(100))})
    result = add_buckets(df, "x", n_quantiles=4)
    assert "x_bucket" in result.columns
    assert result["x_bucket"].nunique() <= 4


def test_add_buckets_custom():
    df = pd.DataFrame({"x": [5, 15, 25, 35, 45]})
    result = add_buckets(df, "x", custom_bins=[0, 10, 20, 50])
    assert "x_bucket" in result.columns
    assert result["x_bucket"].notna().any()


def test_add_buckets_uniform_string_dtype():
    """Ensure bucket output is uniform string to avoid OHE errors."""
    df = pd.DataFrame({"x": list(range(20))})
    result = add_buckets(df, "x", n_quantiles=4)
    assert result["x_bucket"].dtype == object


# ---------------------------------------------------------------------------
# Normalize
# ---------------------------------------------------------------------------

def test_normalize_zscore():
    df = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, 50.0]})
    result = normalize(df, ["x"], method="zscore")
    assert abs(result["x"].mean()) < 1e-10
    assert abs(result["x"].std() - 1.0) < 1e-6


def test_normalize_minmax():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
    result = normalize(df, ["x"], method="minmax")
    assert result["x"].min() == 0.0
    assert result["x"].max() == 1.0


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

def test_add_interaction_multiply():
    df = pd.DataFrame({"a": [2.0, 4.0], "b": [3.0, 5.0]})
    result = add_interaction(df, "a", "b", op="multiply")
    assert "a_multiply_b" in result.columns
    assert result["a_multiply_b"].iloc[0] == 6.0


def test_add_interaction_divide_zero():
    df = pd.DataFrame({"a": [10.0, 20.0], "b": [0.0, 4.0]})
    result = add_interaction(df, "a", "b", op="divide")
    assert pd.isna(result["a_divide_b"].iloc[0])
    assert result["a_divide_b"].iloc[1] == 5.0

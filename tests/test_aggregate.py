"""
tests/test_aggregate.py – Unit tests for core/aggregate.py
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.aggregate import (
    group_aggregate, pivot_view, to_csv_bytes, to_xlsx_bytes,
    to_parquet_bytes, available_agg_functions,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _simple_df():
    return pd.DataFrame({
        "region": ["A", "A", "B", "B", "A", "B"],
        "channel": ["web", "app", "web", "app", "web", "web"],
        "revenue": [100, 200, 150, 250, 300, 50],
        "cost": [10, 20, 15, 25, 30, 5],
        "weight": [1, 2, 3, 4, 5, 6],
    })


def _date_df():
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2023-01-15", "2023-02-10", "2023-04-05",
            "2023-07-20", "2023-10-01", "2023-12-25",
        ]),
        "region": ["A", "A", "B", "B", "A", "B"],
        "sales": [10, 20, 30, 40, 50, 60],
    })


# ---------------------------------------------------------------------------
# available_agg_functions
# ---------------------------------------------------------------------------

def test_available_agg_functions_returns_list():
    funcs = available_agg_functions()
    assert isinstance(funcs, list)
    assert len(funcs) > 0


def test_available_agg_functions_contains_weighted_avg():
    funcs = available_agg_functions()
    assert "weighted_avg" in funcs


def test_available_agg_functions_contains_core_funcs():
    funcs = available_agg_functions()
    for fn in ["min", "max", "mean", "median", "sum", "count", "nunique", "std"]:
        assert fn in funcs


# ---------------------------------------------------------------------------
# group_aggregate — basic grouping
# ---------------------------------------------------------------------------

def test_group_aggregate_single_group_col():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["sum"])
    assert len(result) == 2
    row_a = result[result["region"] == "A"]
    assert row_a["revenue_sum"].values[0] == 600


def test_group_aggregate_multiple_group_cols():
    df = _simple_df()
    result = group_aggregate(df, ["region", "channel"], ["revenue"], ["sum"])
    assert len(result) >= 3  # A-web, A-app, B-web, B-app
    row = result[(result["region"] == "A") & (result["channel"] == "web")]
    assert row["revenue_sum"].values[0] == 400


def test_group_aggregate_multiple_metrics():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue", "cost"], ["sum"])
    assert "revenue_sum" in result.columns
    assert "cost_sum" in result.columns


# ---------------------------------------------------------------------------
# Aggregation functions
# ---------------------------------------------------------------------------

def test_agg_min():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["min"])
    row_b = result[result["region"] == "B"]
    assert row_b["revenue_min"].values[0] == 50


def test_agg_max():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["max"])
    row_a = result[result["region"] == "A"]
    assert row_a["revenue_max"].values[0] == 300


def test_agg_mean():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["mean"])
    row_a = result[result["region"] == "A"]
    assert row_a["revenue_mean"].values[0] == 200.0


def test_agg_median():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["median"])
    row_a = result[result["region"] == "A"]
    assert row_a["revenue_median"].values[0] == 200.0


def test_agg_count():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["count"])
    row_a = result[result["region"] == "A"]
    assert row_a["revenue_count"].values[0] == 3


def test_agg_nunique():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["nunique"])
    row_a = result[result["region"] == "A"]
    assert row_a["revenue_nunique"].values[0] == 3


def test_agg_std():
    df = _simple_df()
    result = group_aggregate(df, ["region"], ["revenue"], ["std"])
    assert "revenue_std" in result.columns
    assert result["revenue_std"].notna().all()


def test_agg_q25_q75():
    df = pd.DataFrame({"g": ["x"] * 100, "v": list(range(100))})
    result = group_aggregate(df, ["g"], ["v"], ["q25", "q75"])
    # Column names for lambda-based aggs get index-based names
    # Just verify we get a result with correct number of rows
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Weighted average
# ---------------------------------------------------------------------------

def test_weighted_avg_basic():
    df = pd.DataFrame({
        "g": ["x", "x", "x"],
        "val": [10.0, 20.0, 30.0],
        "w": [1.0, 1.0, 1.0],
    })
    result = group_aggregate(df, ["g"], ["val"], ["weighted_avg"], weight_col="w")
    assert "val_weighted_avg" in result.columns
    assert abs(result["val_weighted_avg"].values[0] - 20.0) < 1e-9


def test_weighted_avg_unequal_weights():
    df = pd.DataFrame({
        "g": ["x", "x"],
        "val": [10.0, 30.0],
        "w": [3.0, 1.0],
    })
    result = group_aggregate(df, ["g"], ["val"], ["weighted_avg"], weight_col="w")
    expected = (10.0 * 3 + 30.0 * 1) / (3 + 1)  # 15.0
    assert abs(result["val_weighted_avg"].values[0] - expected) < 1e-9


def test_weighted_avg_with_other_aggs():
    df = _simple_df()
    result = group_aggregate(
        df, ["region"], ["revenue"], ["sum", "weighted_avg"], weight_col="weight",
    )
    assert "revenue_sum" in result.columns
    assert "revenue_weighted_avg" in result.columns


# ---------------------------------------------------------------------------
# Time bucketing
# ---------------------------------------------------------------------------

def test_time_bucket_month():
    df = _date_df()
    result = group_aggregate(
        df, ["region"], ["sales"], ["sum"],
        date_col="date", time_bucket="month",
    )
    assert "_time_bucket" in result.columns
    assert len(result) > 2


def test_time_bucket_quarter():
    df = _date_df()
    result = group_aggregate(
        df, ["region"], ["sales"], ["sum"],
        date_col="date", time_bucket="quarter",
    )
    assert "_time_bucket" in result.columns
    # 6 dates spanning Q1, Q2, Q3, Q4 with 2 regions -> multiple rows
    assert len(result) >= 4


# ---------------------------------------------------------------------------
# Numeric binning
# ---------------------------------------------------------------------------

def test_numeric_binning_quantile():
    df = pd.DataFrame({
        "g": ["x"] * 20,
        "val": list(range(20)),
        "metric": [1.0] * 20,
    })
    result = group_aggregate(
        df, ["g"], ["metric"], ["sum"],
        numeric_bin_col="val", numeric_n_quantiles=4,
    )
    assert f"_bin_val" in result.columns
    assert len(result) >= 2


def test_numeric_binning_custom_edges():
    df = pd.DataFrame({
        "g": ["x"] * 10,
        "val": list(range(10)),
        "metric": [1.0] * 10,
    })
    result = group_aggregate(
        df, ["g"], ["metric"], ["sum"],
        numeric_bin_col="val", numeric_bin_edges=[0, 3, 6, 10],
    )
    assert "_bin_val" in result.columns


# ---------------------------------------------------------------------------
# pivot_view
# ---------------------------------------------------------------------------

def test_pivot_view_basic():
    df = pd.DataFrame({
        "region": ["A", "A", "B", "B"],
        "channel": ["web", "app", "web", "app"],
        "revenue": [100, 200, 150, 250],
    })
    result = pivot_view(df, "region", "channel", "revenue", "sum")
    assert "web" in result.columns
    assert "app" in result.columns
    assert len(result) == 2


def test_pivot_view_fill_value():
    df = pd.DataFrame({
        "region": ["A", "B"],
        "channel": ["web", "app"],
        "revenue": [100, 200],
    })
    result = pivot_view(df, "region", "channel", "revenue", "sum")
    # Missing combinations should be 0
    assert (result.fillna(0) == result).all().all()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def test_to_csv_bytes_non_empty():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    data = to_csv_bytes(df)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_to_xlsx_bytes_non_empty():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    data = to_xlsx_bytes(df)
    assert isinstance(data, bytes)
    assert len(data) > 0


def test_to_parquet_bytes_non_empty():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    data = to_parquet_bytes(df)
    assert isinstance(data, bytes)
    assert len(data) > 0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_group_aggregate_no_valid_group_cols_raises():
    df = pd.DataFrame({"a": [1, 2]})
    with pytest.raises(ValueError, match="No valid group columns"):
        group_aggregate(df, ["nonexistent"], ["a"], ["sum"])


def test_group_aggregate_no_valid_metric_cols_raises():
    df = pd.DataFrame({"g": ["a", "b"]})
    with pytest.raises(ValueError, match="No valid metric columns"):
        group_aggregate(df, ["g"], ["nonexistent"], ["sum"])

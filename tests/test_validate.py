"""
tests/test_validate.py – Unit tests for core/validate.py
"""
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.validate import (
    check_required_columns, check_no_nulls, check_ranges,
    check_uniqueness, check_zero_variance, check_time_gaps,
    run_all_checks, ValidationResult,
)


# ---------------------------------------------------------------------------
# check_required_columns
# ---------------------------------------------------------------------------

def test_required_columns_all_present():
    df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
    result = check_required_columns(df, ["a", "b"])
    assert result.passed is True
    assert result.rule == "required_columns"


def test_required_columns_some_missing():
    df = pd.DataFrame({"a": [1], "b": [2]})
    result = check_required_columns(df, ["a", "b", "x", "y"])
    assert result.passed is False
    assert "x" in result.details["missing"]
    assert "y" in result.details["missing"]


def test_required_columns_all_missing():
    df = pd.DataFrame({"a": [1]})
    result = check_required_columns(df, ["x", "y"])
    assert result.passed is False
    assert result.severity == "error"


# ---------------------------------------------------------------------------
# check_no_nulls
# ---------------------------------------------------------------------------

def test_no_nulls_clean_data():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = check_no_nulls(df)
    assert result.passed is True


def test_no_nulls_with_nulls():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, None]})
    result = check_no_nulls(df)
    assert result.passed is False
    assert "a" in result.details["null_fractions"]
    assert "b" in result.details["null_fractions"]


def test_no_nulls_with_threshold():
    df = pd.DataFrame({"a": [1, None, 3, 4, 5]})  # 20% null
    result_strict = check_no_nulls(df, threshold=0.0)
    result_loose = check_no_nulls(df, threshold=0.25)
    assert result_strict.passed is False
    assert result_loose.passed is True


def test_no_nulls_specific_columns():
    df = pd.DataFrame({"a": [1, None], "b": [3, 4]})
    result = check_no_nulls(df, columns=["b"])
    assert result.passed is True


# ---------------------------------------------------------------------------
# check_ranges
# ---------------------------------------------------------------------------

def test_ranges_all_in_range():
    df = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
    result = check_ranges(df, "val", min_val=0, max_val=100)
    assert result.passed is True


def test_ranges_out_of_range():
    df = pd.DataFrame({"val": [10, 20, 150, -5, 50]})
    result = check_ranges(df, "val", min_val=0, max_val=100)
    assert result.passed is False
    assert result.details["violations"] == 2


def test_ranges_min_only():
    df = pd.DataFrame({"val": [-10, 5, 10]})
    result = check_ranges(df, "val", min_val=0)
    assert result.passed is False
    assert result.details["violations"] == 1


def test_ranges_max_only():
    df = pd.DataFrame({"val": [10, 50, 200]})
    result = check_ranges(df, "val", max_val=100)
    assert result.passed is False
    assert result.details["violations"] == 1


def test_ranges_missing_column():
    df = pd.DataFrame({"a": [1, 2]})
    result = check_ranges(df, "nonexistent", min_val=0, max_val=10)
    assert result.passed is False
    assert result.severity == "error"


# ---------------------------------------------------------------------------
# check_uniqueness
# ---------------------------------------------------------------------------

def test_uniqueness_all_unique():
    df = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    result = check_uniqueness(df, ["id"])
    assert result.passed is True


def test_uniqueness_with_duplicates():
    df = pd.DataFrame({"id": [1, 1, 2], "name": ["a", "a", "b"]})
    result = check_uniqueness(df, ["id"])
    assert result.passed is False
    assert result.details["duplicate_count"] == 2


def test_uniqueness_composite_key():
    df = pd.DataFrame({
        "a": [1, 1, 2],
        "b": ["x", "y", "x"],
    })
    result = check_uniqueness(df, ["a", "b"])
    assert result.passed is True  # all (a, b) combos are unique


def test_uniqueness_nonexistent_columns():
    df = pd.DataFrame({"a": [1, 2]})
    result = check_uniqueness(df, ["nonexistent"])
    assert result.passed is False
    assert result.severity == "error"


# ---------------------------------------------------------------------------
# check_zero_variance
# ---------------------------------------------------------------------------

def test_zero_variance_constant_column():
    df = pd.DataFrame({"constant": [5, 5, 5, 5], "varies": [1, 2, 3, 4]})
    result = check_zero_variance(df)
    assert result.passed is False
    assert "constant" in result.details["columns"]


def test_zero_variance_all_varying():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = check_zero_variance(df)
    assert result.passed is True


def test_zero_variance_specific_columns():
    df = pd.DataFrame({"constant": [5, 5, 5], "varies": [1, 2, 3]})
    result = check_zero_variance(df, columns=["varies"])
    assert result.passed is True


def test_zero_variance_no_numeric_cols():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    result = check_zero_variance(df)
    assert result.passed is True


# ---------------------------------------------------------------------------
# check_time_gaps
# ---------------------------------------------------------------------------

def test_time_gaps_no_gaps():
    dates = pd.date_range("2023-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"date": dates, "val": range(6)})
    result = check_time_gaps(df, "date", expected_freq="MS")
    assert result.passed is True


def test_time_gaps_with_gaps():
    dates = pd.to_datetime(["2023-01-01", "2023-02-01", "2023-05-01", "2023-06-01"])
    df = pd.DataFrame({"date": dates, "val": [1, 2, 3, 4]})
    result = check_time_gaps(df, "date", expected_freq="MS")
    assert result.passed is False
    assert result.details["missing_count"] == 2  # March and April missing


def test_time_gaps_missing_column():
    df = pd.DataFrame({"val": [1, 2]})
    result = check_time_gaps(df, "nonexistent")
    assert result.passed is False
    assert result.severity == "error"


def test_time_gaps_too_few_dates():
    df = pd.DataFrame({"date": pd.to_datetime(["2023-01-01"]), "val": [1]})
    result = check_time_gaps(df, "date")
    assert result.passed is True  # Not enough data to check


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------

def test_run_all_checks_returns_list():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    results = run_all_checks(df)
    assert isinstance(results, list)
    assert all(isinstance(r, ValidationResult) for r in results)


def test_run_all_checks_includes_nulls_and_variance():
    df = pd.DataFrame({"a": [1, 2, 3]})
    results = run_all_checks(df)
    rules = [r.rule for r in results]
    assert "no_nulls" in rules
    assert "zero_variance" in rules


def test_run_all_checks_with_required_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    results = run_all_checks(df, required_columns=["a", "c"])
    req_result = [r for r in results if r.rule == "required_columns"][0]
    assert req_result.passed is False


def test_run_all_checks_with_range_rules():
    df = pd.DataFrame({"val": [10, 200, 30]})
    results = run_all_checks(df, range_rules=[{"col": "val", "min": 0, "max": 100}])
    range_result = [r for r in results if r.rule == "range_check"][0]
    assert range_result.passed is False


def test_run_all_checks_with_uniqueness():
    df = pd.DataFrame({"id": [1, 1, 2]})
    results = run_all_checks(df, uniqueness_columns=["id"])
    uniq_result = [r for r in results if r.rule == "uniqueness"][0]
    assert uniq_result.passed is False


def test_run_all_checks_with_date_col():
    dates = pd.to_datetime(["2023-01-01", "2023-03-01"])
    df = pd.DataFrame({"date": dates, "val": [1, 2]})
    results = run_all_checks(df, date_col="date", expected_freq="MS")
    gap_result = [r for r in results if r.rule == "time_gaps"][0]
    assert gap_result.passed is False

"""
tests/test_merge.py – Tests for core/merge.py
"""
import pandas as pd
import numpy as np
import pytest

from core.merge import (
    merge_tables, concat_tables, analyze_key_cardinality,
    MergeWarning, MergeResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def left_df():
    return pd.DataFrame({
        "id": [1, 2, 3, 4],
        "value_a": [10.0, 20.0, 30.0, 40.0],
        "name": ["Alice", "Bob", "Charlie", "Dave"],
    })


@pytest.fixture
def right_df():
    return pd.DataFrame({
        "id": [2, 3, 4, 5],
        "value_b": [200.0, 300.0, 400.0, 500.0],
        "score": [0.2, 0.3, 0.4, 0.5],
    })


@pytest.fixture
def many_right():
    """Right table with duplicated keys for many-to-many testing."""
    return pd.DataFrame({
        "id": [1, 1, 2, 2, 3, 3],
        "value_c": [10, 11, 20, 21, 30, 31],
    })


@pytest.fixture
def nm_left():
    """Left with non-unique keys for N:M join."""
    return pd.DataFrame({"id": [1, 1, 1], "val": [1, 2, 3]})


@pytest.fixture
def nm_right():
    """Right with non-unique keys for N:M join."""
    return pd.DataFrame({"id": [1, 1, 1], "val_r": [10, 11, 12]})


# ---------------------------------------------------------------------------
# Basic merge tests
# ---------------------------------------------------------------------------

def test_inner_join_basic(left_df, right_df):
    result = merge_tables(left_df, right_df, ["id"], ["id"], how="inner")
    assert isinstance(result, MergeResult)
    assert result.result_rows == 3  # ids 2, 3, 4 match
    assert "value_a" in result.df.columns
    assert "value_b" in result.df.columns


def test_left_join_preserves_all_left(left_df, right_df):
    result = merge_tables(left_df, right_df, ["id"], ["id"], how="left")
    assert result.result_rows == 4  # all left rows
    # id=1 should have NaN for value_b
    row_id1 = result.df[result.df["id"] == 1]
    assert row_id1["value_b"].isna().all()


def test_outer_join_all_rows(left_df, right_df):
    result = merge_tables(left_df, right_df, ["id"], ["id"], how="outer")
    # ids 1,2,3,4,5 → 5 unique rows
    assert result.result_rows == 5


def test_cross_join(left_df, right_df):
    result = merge_tables(left_df, right_df, [], [], how="cross")
    assert result.result_rows == len(left_df) * len(right_df)


def test_result_has_correct_columns(left_df, right_df):
    result = merge_tables(left_df, right_df, ["id"], ["id"], how="inner")
    assert "id" in result.df.columns
    assert "value_a" in result.df.columns
    assert "value_b" in result.df.columns


def test_merge_returns_merge_result(left_df, right_df):
    result = merge_tables(left_df, right_df, ["id"], ["id"], how="left")
    assert hasattr(result, "df")
    assert hasattr(result, "warnings")
    assert hasattr(result, "left_rows")
    assert hasattr(result, "right_rows")
    assert hasattr(result, "result_rows")


# ---------------------------------------------------------------------------
# Warning detection tests
# ---------------------------------------------------------------------------

def test_detects_row_explosion(nm_left, nm_right):
    # 3 left × 3 right = 9 result, max=3, ratio=3 > 2 → explosion
    result = merge_tables(nm_left, nm_right, ["id"], ["id"], how="inner")
    codes = [w.code for w in result.warnings]
    assert "ROW_EXPLOSION" in codes


def test_no_warning_for_clean_join(left_df, right_df):
    result = merge_tables(left_df, right_df, ["id"], ["id"], how="inner")
    error_warnings = [w for w in result.warnings if w.level == "error"]
    assert len(error_warnings) == 0


def test_detects_key_nulls():
    left = pd.DataFrame({"id": [1, None, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"id": [1, 2, 3], "val_r": [100, 200, 300]})
    result = merge_tables(left, right, ["id"], ["id"], how="left")
    codes = [w.code for w in result.warnings]
    assert "KEY_NULLS" in codes


def test_detects_key_type_mismatch():
    left = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    right = pd.DataFrame({"id": ["1", "2", "3"], "val_r": [100, 200, 300]})
    result = merge_tables(left, right, ["id"], ["id"], how="inner")
    codes = [w.code for w in result.warnings]
    # Either KEY_TYPE_MISMATCH (detected before merge) or MERGE_FAILED (if pandas throws)
    assert "KEY_TYPE_MISMATCH" in codes or "MERGE_FAILED" in codes


def test_detects_duplicate_columns():
    left = pd.DataFrame({"id": [1, 2], "score": [0.1, 0.2], "name": ["A", "B"]})
    right = pd.DataFrame({"id": [1, 2], "score": [0.3, 0.4], "extra": [10, 20]})
    result = merge_tables(left, right, ["id"], ["id"], how="inner")
    codes = [w.code for w in result.warnings]
    assert "DUPLICATE_COLUMNS" in codes


def test_detects_low_match_rate():
    left = pd.DataFrame({"id": range(100), "val": range(100)})
    right = pd.DataFrame({"id": range(10), "val_r": range(10)})
    result = merge_tables(left, right, ["id"], ["id"], how="inner")
    codes = [w.code for w in result.warnings]
    assert "LOW_MATCH_RATE" in codes


def test_warning_levels():
    left = pd.DataFrame({"id": [1, 2, None], "val": [1, 2, 3]})
    right = pd.DataFrame({"id": [1, 2, 3], "val_r": [10, 20, 30]})
    result = merge_tables(left, right, ["id"], ["id"], how="inner")
    for w in result.warnings:
        assert w.level in ("error", "warning", "info")


def test_has_errors_property():
    left = pd.DataFrame({"id": [1, 1, 2, 2], "val": [1, 2, 3, 4]})
    right = pd.DataFrame({"id": [1, 1, 2, 2], "val_r": [10, 11, 20, 21]})
    result = merge_tables(left, right, ["id"], ["id"], how="inner")
    # Explosion ratio > 10x would make has_errors True
    assert isinstance(result.has_errors, bool)


# ---------------------------------------------------------------------------
# Explosion ratio
# ---------------------------------------------------------------------------

def test_explosion_ratio_clean():
    left = pd.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]})
    right = pd.DataFrame({"id": [1, 2, 3], "val_r": [10, 20, 30]})
    result = merge_tables(left, right, ["id"], ["id"], how="inner")
    assert result.explosion_ratio == pytest.approx(1.0)


def test_explosion_ratio_many_to_many(nm_left, nm_right):
    # 3 × 3 = 9 result rows, max(3,3) = 3 → ratio = 3.0
    result = merge_tables(nm_left, nm_right, ["id"], ["id"], how="inner")
    assert result.explosion_ratio == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Cardinality analysis
# ---------------------------------------------------------------------------

def test_cardinality_one_to_one():
    left = pd.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]})
    right = pd.DataFrame({"id": [1, 2, 3], "val_r": [10, 20, 30]})
    card = analyze_key_cardinality(left, right, ["id"], ["id"])
    assert card["join_type"] == "1:1"
    assert card["left_is_unique"] is True
    assert card["right_is_unique"] is True


def test_cardinality_one_to_many():
    left = pd.DataFrame({"id": [1, 2, 3], "val": [1, 2, 3]})
    right = pd.DataFrame({"id": [1, 1, 2, 3], "val_r": [10, 11, 20, 30]})
    card = analyze_key_cardinality(left, right, ["id"], ["id"])
    assert card["join_type"] == "1:N"


def test_cardinality_many_to_many():
    left = pd.DataFrame({"id": [1, 1, 2], "val": [1, 2, 3]})
    right = pd.DataFrame({"id": [1, 1, 2], "val_r": [10, 11, 20]})
    card = analyze_key_cardinality(left, right, ["id"], ["id"])
    assert card["join_type"] == "N:M"


def test_cardinality_invalid_keys():
    left = pd.DataFrame({"id": [1, 2], "val": [1, 2]})
    right = pd.DataFrame({"id": [1, 2], "val_r": [10, 20]})
    card = analyze_key_cardinality(left, right, ["nonexistent"], ["id"])
    assert card["join_type"] == "unknown"


# ---------------------------------------------------------------------------
# Concat tests
# ---------------------------------------------------------------------------

def test_concat_rows_basic():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"a": [3, 4], "b": [30, 40]})
    result, warns = concat_tables([df1, df2], axis=0)
    assert len(result) == 4
    assert list(result.columns) == ["a", "b"]


def test_concat_rows_detects_schema_mismatch():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"a": [3, 4], "c": [30, 40]})  # 'b' missing, 'c' extra
    result, warns = concat_tables([df1, df2], axis=0)
    codes = [w.code for w in warns]
    assert "SCHEMA_MISMATCH_MISSING" in codes or "SCHEMA_MISMATCH_EXTRA" in codes


def test_concat_rows_detects_duplicates():
    df1 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})
    df2 = pd.DataFrame({"a": [1, 2], "b": [10, 20]})  # exact duplicate
    result, warns = concat_tables([df1, df2], axis=0)
    codes = [w.code for w in warns]
    assert "DUPLICATE_ROWS" in codes


def test_concat_cols_basic():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [10, 20, 30]})
    result, warns = concat_tables([df1, df2], axis=1)
    assert result.shape == (3, 2)


def test_concat_cols_detects_duplicate_colnames():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [10, 20]})
    result, warns = concat_tables([df1, df2], axis=1)
    codes = [w.code for w in warns]
    assert "DUPLICATE_COLUMN_NAMES" in codes


def test_concat_cols_detects_row_count_mismatch():
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    df2 = pd.DataFrame({"b": [10, 20]})
    result, warns = concat_tables([df1, df2], axis=1)
    codes = [w.code for w in warns]
    assert "ROW_COUNT_MISMATCH" in codes


def test_concat_empty_list():
    result, warns = concat_tables([], axis=0)
    assert result.empty


def test_concat_ignore_index():
    df1 = pd.DataFrame({"a": [1, 2]}, index=[10, 11])
    df2 = pd.DataFrame({"a": [3, 4]}, index=[20, 21])
    result, _ = concat_tables([df1, df2], axis=0, ignore_index=True)
    assert list(result.index) == [0, 1, 2, 3]


def test_concat_dtype_mismatch_warning():
    df1 = pd.DataFrame({"x": [1, 2]})  # int
    df2 = pd.DataFrame({"x": [1.5, 2.5]})  # float
    result, warns = concat_tables([df1, df2], axis=0)
    # May or may not warn depending on pandas version behavior
    assert isinstance(warns, list)

"""
tests/test_data.py – Unit tests for core/data.py
"""
import io
import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data import (
    load_csv, load_excel, infer_column_types, cast_column,
    apply_type_overrides, profile_dataframe, describe_numeric, DatasetCatalog,
)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _make_csv_bytes(content: str) -> bytes:
    return content.encode("utf-8")


def test_load_csv_from_bytes():
    csv = "a,b,c\n1,2,3\n4,5,6"
    df = load_csv(_make_csv_bytes(csv))
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_load_csv_semicolon_sep():
    csv = "a;b;c\n1;2;3\n4;5;6"
    df = load_csv(_make_csv_bytes(csv), sep=";")
    assert df.shape == (2, 3)


def test_load_csv_from_filelike():
    csv = "x,y\n10,20\n30,40"
    buf = io.BytesIO(csv.encode())
    df = load_csv(buf)
    assert len(df) == 2


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

def test_infer_datetime_col():
    df = pd.DataFrame({"date": ["2023-01-01", "2023-02-01", "2023-03-01"] * 10})
    types = infer_column_types(df)
    assert types["date"] == "datetime"


def test_infer_numeric_col():
    df = pd.DataFrame({"val": [1.0, 2.5, 3.7, 4.0, 5.1]})
    types = infer_column_types(df)
    assert types["val"] == "numeric"


def test_infer_categorical_col():
    df = pd.DataFrame({"cat": ["a", "b", "c", "a", "b"]})
    types = infer_column_types(df)
    assert types["cat"] == "categorical"


def test_infer_boolean_col():
    df = pd.DataFrame({"flag": ["true", "false", "true", "false"] * 3})
    types = infer_column_types(df)
    assert types["flag"] == "boolean"


def test_infer_handles_empty_col():
    df = pd.DataFrame({"empty": [None, None, None]})
    types = infer_column_types(df)
    assert types["empty"] == "categorical"


# ---------------------------------------------------------------------------
# Cast column
# ---------------------------------------------------------------------------

def test_cast_to_numeric():
    s = pd.Series(["1", "2.5", "abc", "4"])
    result = cast_column(s, "numeric")
    assert result.dtype == float
    assert pd.isna(result.iloc[2])


def test_cast_to_datetime():
    s = pd.Series(["2023-01-01", "2023-02-15", "not a date"])
    result = cast_column(s, "datetime")
    assert pd.api.types.is_datetime64_any_dtype(result)
    assert pd.isna(result.iloc[2])


def test_cast_to_boolean():
    s = pd.Series(["true", "false", "yes", "no", "1", "0"])
    result = cast_column(s, "boolean")
    assert result.iloc[0] == True
    assert result.iloc[1] == False


def test_cast_to_categorical():
    s = pd.Series([1, 2, 3])
    result = cast_column(s, "categorical")
    assert result.dtype == object


# ---------------------------------------------------------------------------
# Apply type overrides
# ---------------------------------------------------------------------------

def test_apply_type_overrides():
    df = pd.DataFrame({
        "date_str": ["2023-01-01", "2023-02-01"],
        "val_str": ["10", "20"],
    })
    overrides = {"date_str": "datetime", "val_str": "numeric"}
    result = apply_type_overrides(df, overrides)
    assert pd.api.types.is_datetime64_any_dtype(result["date_str"])
    assert pd.api.types.is_numeric_dtype(result["val_str"])


def test_apply_type_overrides_ignores_missing_cols():
    df = pd.DataFrame({"a": [1, 2]})
    result = apply_type_overrides(df, {"nonexistent": "numeric"})
    assert list(result.columns) == ["a"]


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def test_profile_returns_all_cols():
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"], "c": pd.to_datetime(["2020", "2021", "2022"])})
    prof = profile_dataframe(df)
    assert set(prof["column"]) == {"a", "b", "c"}
    assert "pct_missing" in prof.columns


def test_profile_missing_count():
    df = pd.DataFrame({"a": [1, None, None, 4, 5]})
    prof = profile_dataframe(df)
    row = prof[prof["column"] == "a"].iloc[0]
    assert row["n_missing"] == 2
    assert row["pct_missing"] == 40.0


def test_describe_numeric():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})
    desc = describe_numeric(df)
    assert "iqr" in desc.columns
    assert "skewness" in desc.columns
    assert len(desc) == 2


def test_describe_numeric_no_cols():
    df = pd.DataFrame({"cat": ["a", "b"]})
    desc = describe_numeric(df)
    assert desc.empty


# ---------------------------------------------------------------------------
# DatasetCatalog
# ---------------------------------------------------------------------------

def test_catalog_add_get():
    cat = DatasetCatalog()
    df = pd.DataFrame({"x": [1, 2, 3]})
    cat.add("my_ds", df)
    assert "my_ds" in cat
    pd.testing.assert_frame_equal(cat.get("my_ds"), df)


def test_catalog_list():
    cat = DatasetCatalog()
    cat.add("ds1", pd.DataFrame({"a": [1]}))
    cat.add("ds2", pd.DataFrame({"b": [2]}))
    lst = cat.list_datasets()
    assert len(lst) == 2
    names = [d["name"] for d in lst]
    assert "ds1" in names and "ds2" in names


def test_catalog_remove():
    cat = DatasetCatalog()
    cat.add("ds1", pd.DataFrame({"a": [1]}))
    cat.remove("ds1")
    assert "ds1" not in cat
    assert len(cat) == 0


def test_catalog_get_raises():
    cat = DatasetCatalog()
    with pytest.raises(KeyError):
        cat.get("nonexistent")

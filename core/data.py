"""
core/data.py – Dataset loading, profiling, and type inference.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(source: Any, **kwargs) -> pd.DataFrame:
    """Load CSV from a file path, file-like object, or bytes.

    Parameters
    ----------
    source:
        Path string, pathlib.Path, BytesIO / UploadedFile, or raw bytes.
    **kwargs:
        Extra kwargs forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, (str, Path)):
        return pd.read_csv(source, **kwargs)
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source), **kwargs)
    # file-like (Streamlit UploadedFile, BytesIO, etc.)
    return pd.read_csv(source, **kwargs)


def load_excel(source: Any, sheet_name: int | str = 0, **kwargs) -> pd.DataFrame:
    """Load an Excel file (xls/xlsx).

    Parameters
    ----------
    source:
        Path string, pathlib.Path, BytesIO / UploadedFile, or raw bytes.
    sheet_name:
        Sheet index or name.
    **kwargs:
        Extra kwargs forwarded to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return pd.read_excel(source, sheet_name=sheet_name, **kwargs)


def load_file(source: Any, filename: str = "", **kwargs) -> pd.DataFrame:
    """Auto-detect CSV vs Excel by filename extension and load.

    Parameters
    ----------
    source:
        Raw bytes, file-like object, or path.
    filename:
        Original filename used to detect extension.
    **kwargs:
        Extra kwargs forwarded to the underlying loader.

    Returns
    -------
    pd.DataFrame
    """
    name = filename.lower() if filename else ""
    if name.endswith((".xls", ".xlsx")):
        return load_excel(source, **kwargs)
    return load_csv(source, **kwargs)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",          # 2023-01-15
    r"\d{2}/\d{2}/\d{4}",          # 15/01/2023
    r"\d{2}\.\d{2}\.\d{4}",        # 15.01.2023
    r"\d{4}/\d{2}/\d{2}",          # 2023/01/15
    r"\d{1,2} \w+ \d{4}",          # 5 Jan 2023
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS))


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Infer logical types for each column.

    Returns a dict mapping column name → inferred type:
    ``"datetime"``, ``"numeric"``, ``"boolean"``, or ``"categorical"``.

    Parameters
    ----------
    df:
        The DataFrame to inspect.

    Returns
    -------
    dict[str, str]
    """
    result: dict[str, str] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            result[col] = "categorical"
            continue

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            result[col] = "datetime"
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            result[col] = "numeric"
            continue

        # Boolean-like
        unique_lower = set(str(v).strip().lower() for v in series.unique())
        if unique_lower <= {"true", "false", "yes", "no", "0", "1", "t", "f"}:
            result[col] = "boolean"
            continue

        # Try to parse as datetime (sample first 200 non-null values)
        sample = series.astype(str).head(200)
        matched = sample.str.contains(_DATE_RE, na=False).sum()
        if matched >= min(5, len(sample) * 0.6):
            # Attempt actual parsing
            try:
                pd.to_datetime(sample, errors="raise")
                result[col] = "datetime"
                continue
            except Exception:
                pass

        result[col] = "categorical"
    return result


def cast_column(series: pd.Series, target_type: str) -> pd.Series:
    """Cast a column to the target logical type.

    Parameters
    ----------
    series:
        The column to cast.
    target_type:
        One of ``"datetime"``, ``"numeric"``, ``"boolean"``, ``"categorical"``.

    Returns
    -------
    pd.Series
    """
    if target_type == "datetime":
        return pd.to_datetime(series, errors="coerce", utc=False)
    if target_type == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if target_type == "boolean":
        bool_map = {
            "true": True, "false": False, "yes": True, "no": False,
            "1": True, "0": False, "t": True, "f": False,
        }
        return series.astype(str).str.strip().str.lower().map(bool_map)
    # categorical / string
    return series.astype(str)


def apply_type_overrides(
    df: pd.DataFrame,
    overrides: dict[str, str],
) -> pd.DataFrame:
    """Apply user-specified type overrides to a DataFrame.

    Parameters
    ----------
    df:
        The DataFrame to modify.
    overrides:
        Mapping of column name → desired type string.

    Returns
    -------
    pd.DataFrame
        A copy with casts applied.
    """
    df = df.copy()
    for col, ttype in overrides.items():
        if col in df.columns:
            df[col] = cast_column(df[col], ttype)
    return df


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a per-column profile summary.

    Returns a DataFrame with columns:
    ``dtype, inferred_type, n_missing, pct_missing, n_unique, sample_values``.

    Parameters
    ----------
    df:
        The DataFrame to profile.

    Returns
    -------
    pd.DataFrame
    """
    inferred = infer_column_types(df)
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = round(n_missing / len(series) * 100, 1) if len(series) else 0.0
        n_unique = int(series.nunique(dropna=True))
        sample = series.dropna().unique()[:5]
        sample_str = ", ".join(str(v) for v in sample)
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "inferred_type": inferred.get(col, "categorical"),
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            "sample_values": sample_str,
        })
    return pd.DataFrame(rows)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Extended numeric summary (adds IQR and skewness).

    Parameters
    ----------
    df:
        The DataFrame to describe.

    Returns
    -------
    pd.DataFrame
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc["iqr"] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    desc["skewness"] = df[numeric_cols].skew()
    return desc.round(4)


# ---------------------------------------------------------------------------
# Dataset catalog helpers
# ---------------------------------------------------------------------------

class DatasetCatalog:
    """In-memory catalog of named DataFrames with metadata.

    Attributes
    ----------
    _store : dict[str, dict]
        Internal mapping of name → {"df": pd.DataFrame, "source": str, ...}
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        df: pd.DataFrame,
        source: str = "upload",
        notes: str = "",
    ) -> None:
        """Add or replace a dataset in the catalog.

        Parameters
        ----------
        name:
            Logical name for the dataset.
        df:
            The DataFrame to store.
        source:
            Where the data came from (``"upload"``, ``"postgres"``, etc.).
        notes:
            Free-text notes.
        """
        self._store[name] = {
            "df": df,
            "source": source,
            "notes": notes,
            "shape": df.shape,
            "columns": list(df.columns),
        }

    def get(self, name: str) -> pd.DataFrame:
        """Retrieve a DataFrame by name.

        Parameters
        ----------
        name:
            Dataset name.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        KeyError
            If the name is not in the catalog.
        """
        return self._store[name]["df"]

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return a list of metadata dicts for all stored datasets.

        Returns
        -------
        list[dict]
        """
        return [
            {
                "name": k,
                "rows": v["shape"][0],
                "cols": v["shape"][1],
                "source": v["source"],
                "notes": v["notes"],
            }
            for k, v in self._store.items()
        ]

    def remove(self, name: str) -> None:
        """Remove a dataset from the catalog.

        Parameters
        ----------
        name:
            Dataset name to remove.
        """
        self._store.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

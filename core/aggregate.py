"""
core/aggregate.py – Group & Aggregate engine for KIBAD.

Provides flexible multi-column grouping with a wide range of aggregation
functions, time/numeric bucketing, pivot views, and export helpers.
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Supported aggregation functions
# ---------------------------------------------------------------------------

_BUILTIN_AGGS: dict[str, str | Callable] = {
    "min": "min",
    "max": "max",
    "mean": "mean",
    "median": "median",
    "sum": "sum",
    "count": "count",
    "nunique": "nunique",
    "std": "std",
    "q25": lambda s: s.quantile(0.25),
    "q75": lambda s: s.quantile(0.75),
    "first": "first",
    "last": "last",
}


def available_agg_functions() -> list[str]:
    """Return names of all built-in aggregation functions."""
    return list(_BUILTIN_AGGS.keys()) + ["weighted_avg"]


# ---------------------------------------------------------------------------
# Time bucketing
# ---------------------------------------------------------------------------

TIME_BUCKET_MAP: dict[str, str] = {
    "month": "MS",
    "W-MON": "W-MON",
    "quarter": "QS",
    "year": "YS",
}


def _apply_time_bucket(
    df: pd.DataFrame,
    date_col: str,
    bucket: str,
) -> pd.DataFrame:
    """Add a ``_time_bucket`` column to *df* by flooring dates to *bucket*.

    Parameters
    ----------
    df:
        DataFrame with a datetime column.
    date_col:
        Name of the date column.
    bucket:
        One of ``"month"``, ``"W-MON"``, ``"quarter"``, ``"year"``.

    Returns
    -------
    pd.DataFrame with ``_time_bucket`` column added.
    """
    df = df.copy()
    freq = TIME_BUCKET_MAP.get(bucket, bucket)
    dt = pd.to_datetime(df[date_col], errors="coerce")

    # Map offset aliases to period-compatible frequencies
    _PERIOD_FREQ = {
        "MS": "M",
        "ME": "M",
        "QS": "Q",
        "QE": "Q",
        "YS": "Y",
        "YE": "Y",
        "W-MON": "W",
    }
    period_freq = _PERIOD_FREQ.get(freq, freq)
    df["_time_bucket"] = dt.dt.to_period(period_freq).apply(
        lambda p: p.start_time if pd.notna(p) else pd.NaT
    )
    return df


# ---------------------------------------------------------------------------
# Numeric bucketing
# ---------------------------------------------------------------------------

def _apply_numeric_bins(
    df: pd.DataFrame,
    col: str,
    bin_edges: list[float] | None = None,
    n_quantiles: int | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    """Add a ``_bin_{col}`` column to *df*.

    Parameters
    ----------
    df:
        DataFrame.
    col:
        Numeric column to bucket.
    bin_edges:
        Explicit bin edges (takes priority).
    n_quantiles:
        Number of quantile bins (used if *bin_edges* is None).
    labels:
        Optional labels for bins.

    Returns
    -------
    pd.DataFrame with the bin column added.
    """
    df = df.copy()
    s = pd.to_numeric(df[col], errors="coerce")
    new_col = f"_bin_{col}"
    if bin_edges:
        df[new_col] = pd.cut(s, bins=bin_edges, labels=labels, include_lowest=True).astype(str)
    else:
        q = n_quantiles or 4
        df[new_col] = pd.qcut(s, q=q, labels=labels, duplicates="drop").astype(str)
    return df


# ---------------------------------------------------------------------------
# Core group-aggregate function
# ---------------------------------------------------------------------------

def group_aggregate(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
    agg_funcs: list[str],
    *,
    date_col: str | None = None,
    time_bucket: str | None = None,
    numeric_bin_col: str | None = None,
    numeric_bin_edges: list[float] | None = None,
    numeric_n_quantiles: int | None = None,
    weight_col: str | None = None,
) -> pd.DataFrame:
    """Perform grouped aggregation.

    Parameters
    ----------
    df:
        Source DataFrame.
    group_cols:
        Columns to group by.
    metric_cols:
        Numeric columns to aggregate.
    agg_funcs:
        Names from :func:`available_agg_functions`.
    date_col:
        Date column (required if *time_bucket* is used).
    time_bucket:
        Optional time bucketing (``"month"``, ``"W-MON"``, ``"quarter"``, ``"year"``).
    numeric_bin_col:
        Column to bin numerically.
    numeric_bin_edges:
        Custom bin edges for numeric binning.
    numeric_n_quantiles:
        Quantile-based bin count.
    weight_col:
        Column to use as weights for ``weighted_avg``.

    Returns
    -------
    pd.DataFrame — aggregated table with descriptive column names.
    """
    work = df.copy()

    # Time bucketing
    actual_group_cols = list(group_cols)
    if time_bucket and date_col:
        work = _apply_time_bucket(work, date_col, time_bucket)
        actual_group_cols.append("_time_bucket")

    # Numeric binning
    if numeric_bin_col and numeric_bin_col in work.columns:
        work = _apply_numeric_bins(
            work, numeric_bin_col,
            bin_edges=numeric_bin_edges,
            n_quantiles=numeric_n_quantiles,
        )
        actual_group_cols.append(f"_bin_{numeric_bin_col}")

    # Validate
    actual_group_cols = [c for c in actual_group_cols if c in work.columns]
    metric_cols = [c for c in metric_cols if c in work.columns]

    if not actual_group_cols:
        raise ValueError("No valid group columns after processing.")
    if not metric_cols:
        raise ValueError("No valid metric columns.")

    # Build aggregation dict
    agg_dict: dict[str, list] = {}
    for mc in metric_cols:
        funcs_for_col = []
        for fn_name in agg_funcs:
            if fn_name == "weighted_avg":
                continue  # handled separately
            resolved = _BUILTIN_AGGS.get(fn_name, fn_name)
            funcs_for_col.append(resolved)
        if funcs_for_col:
            agg_dict[mc] = funcs_for_col

    if agg_dict:
        grouped = work.groupby(actual_group_cols, as_index=False, dropna=False)
        result = grouped.agg(agg_dict)
        # Flatten multi-level columns
        result.columns = [
            f"{col}_{func}" if isinstance(func, str) and func else f"{col}_{i}"
            for i, (col, func) in enumerate(result.columns)
        ]
        # Fix group col names (they got a blank second level)
        for i, gc in enumerate(actual_group_cols):
            old_name = result.columns[i]
            if old_name != gc:
                result = result.rename(columns={old_name: gc})
    else:
        result = work[actual_group_cols].drop_duplicates().reset_index(drop=True)

    # Weighted average
    if "weighted_avg" in agg_funcs and weight_col and weight_col in work.columns:
        for mc in metric_cols:
            wa_col = f"{mc}_weighted_avg"
            wa_vals = []
            for _, grp in work.groupby(actual_group_cols, dropna=False):
                w = pd.to_numeric(grp[weight_col], errors="coerce")
                v = pd.to_numeric(grp[mc], errors="coerce")
                mask = w.notna() & v.notna() & (w > 0)
                if mask.sum() > 0:
                    wa_vals.append(float(np.average(v[mask], weights=w[mask])))
                else:
                    wa_vals.append(np.nan)
            result[wa_col] = wa_vals

    return result


# ---------------------------------------------------------------------------
# Pivot view
# ---------------------------------------------------------------------------

def pivot_view(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str,
    values_col: str,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """Create a pivot table from an aggregated DataFrame.

    Parameters
    ----------
    df:
        Source DataFrame (typically output of :func:`group_aggregate`).
    index_col:
        Column for row labels.
    columns_col:
        Column for column headers.
    values_col:
        Column to fill cells.
    agg_func:
        Aggregation function for duplicate entries.

    Returns
    -------
    pd.DataFrame
    """
    pivot = pd.pivot_table(
        df,
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc=agg_func,
        fill_value=0,
    )
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot.reset_index()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to CSV bytes (UTF-8 with BOM for Excel compat)."""
    return ("\ufeff" + df.to_csv(index=False)).encode("utf-8-sig")


def to_xlsx_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to XLSX bytes."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Data")
    return buf.getvalue()


def to_parquet_bytes(df: pd.DataFrame) -> bytes:
    """Serialize DataFrame to Parquet bytes (requires pyarrow)."""
    buf = BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow")
    return buf.getvalue()

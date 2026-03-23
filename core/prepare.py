"""
core/prepare.py – Data preparation: cleaning, resampling, and feature engineering.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Transformation history
# ---------------------------------------------------------------------------

@dataclass
class TransformStep:
    """Record of a single transformation applied to a dataset."""
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    rows_before: int = 0
    rows_after: int = 0


class TransformLog:
    """Ordered log of transformations for one dataset."""

    def __init__(self) -> None:
        self._steps: list[TransformStep] = []

    def add(self, step: TransformStep) -> None:
        self._steps.append(step)

    @property
    def steps(self) -> list[TransformStep]:
        return list(self._steps)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            {
                "name": s.name,
                "params": s.params,
                "timestamp": s.timestamp,
                "rows_before": s.rows_before,
                "rows_after": s.rows_after,
            }
            for s in self._steps
        ]

    def clear(self) -> None:
        self._steps.clear()


# ---------------------------------------------------------------------------
# Numeric parsing (Russian & international formats)
# ---------------------------------------------------------------------------

def parse_numeric(
    series: pd.Series,
    thousands_sep: str = " ",
    decimal_sep: str = ",",
) -> pd.Series:
    """Parse a string series with custom thousands/decimal separators to float.

    Handles formats like ``"1 234,56"`` (RU), ``"1.234,56"`` (DE),
    ``"1,234.56"`` (EN).

    Parameters
    ----------
    series:
        Series of string values.
    thousands_sep:
        Character used as thousands separator (e.g. ``" "``, ``"."``, ``","``).
    decimal_sep:
        Character used as decimal separator (e.g. ``","``, ``"."``).

    Returns
    -------
    pd.Series of float64 with unparseable values as NaN.
    """
    def _clean(val: Any) -> Any:
        if pd.isna(val):
            return np.nan
        s = str(val).strip()
        if not s:
            return np.nan
        # Remove thousands separator
        if thousands_sep:
            s = s.replace(thousands_sep, "")
        # Replace decimal separator with '.'
        if decimal_sep and decimal_sep != ".":
            s = s.replace(decimal_sep, ".")
        # Remove currency symbols and whitespace
        s = re.sub(r"[^\d.\-+eE]", "", s)
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    return series.map(_clean).astype(float)


def parse_dates_robust(
    series: pd.Series,
    dayfirst: bool = True,
    yearfirst: bool = False,
    tz_strip: bool = True,
    coerce_future: bool = True,
    max_year: int = 2100,
    min_year: int = 1900,
) -> pd.Series:
    """Robust date parser with out-of-bounds correction.

    Parameters
    ----------
    series:
        String or mixed series to parse.
    dayfirst:
        Interpret ambiguous dates as DD/MM/YYYY (common in Russia/Europe).
    yearfirst:
        If True, try YYYY-first parsing.
    tz_strip:
        Remove timezone info after conversion.
    coerce_future:
        If True, dates with year > *max_year* or < *min_year* are coerced to NaT.
    max_year, min_year:
        Year boundaries for coercion.

    Returns
    -------
    pd.Series of datetime64[ns] with unparseable values as NaT.
    """
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, yearfirst=yearfirst)

    if tz_strip and parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    if coerce_future:
        mask_future = parsed.dt.year > max_year
        mask_past = parsed.dt.year < min_year
        parsed = parsed.where(~(mask_future | mask_past), other=pd.NaT)

    return parsed


# ---------------------------------------------------------------------------
# Date / time utilities
# ---------------------------------------------------------------------------

RESAMPLE_ALIASES: dict[str, str] = {
    "Daily": "D",
    "Weekly (Mon)": "W-MON",
    "Monthly Start": "MS",
    "Monthly End": "ME",
    "Quarterly": "QS",
}


def parse_dates(df: pd.DataFrame, date_col: str, tz_strip: bool = True) -> pd.DataFrame:
    """Parse a column to datetime and optionally strip timezone.

    Parameters
    ----------
    df:
        Source DataFrame.
    date_col:
        Name of the column to parse.
    tz_strip:
        If True, convert tz-aware datetimes to UTC then remove tzinfo.

    Returns
    -------
    pd.DataFrame
        A copy with the column cast to datetime64[ns].
    """
    df = df.copy()
    parsed = pd.to_datetime(df[date_col], errors="coerce")
    if tz_strip and parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
    df[date_col] = parsed
    return df


def resample_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    freq: str = "MS",
    agg_func: str = "sum",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Resample a time series DataFrame to a target frequency.

    Supports optional group-by before resampling (returns long-format).

    Parameters
    ----------
    df:
        Source DataFrame; ``date_col`` must be datetime.
    date_col:
        Name of the datetime column.
    value_cols:
        Numeric columns to aggregate.
    freq:
        Pandas offset alias (e.g. ``"MS"``, ``"W-MON"``).
    agg_func:
        Aggregation function name: ``"sum"``, ``"mean"``, ``"median"``,
        ``"min"``, ``"max"``, ``"last"``.
    group_cols:
        Optional grouping columns applied before resampling.

    Returns
    -------
    pd.DataFrame
        Resampled DataFrame.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    cols_to_keep = [date_col] + (group_cols or []) + value_cols
    df = df[[c for c in cols_to_keep if c in df.columns]]

    if group_cols:
        results = []
        for keys, grp in df.groupby(group_cols, sort=False):
            grp = grp.set_index(date_col)[value_cols]
            resampled = grp.resample(freq).agg(agg_func).reset_index()
            if isinstance(keys, str):
                keys = (keys,)
            for k, col in zip(keys, group_cols):
                resampled[col] = k
            results.append(resampled)
        return pd.concat(results, ignore_index=True)
    else:
        resampled = (
            df.set_index(date_col)[value_cols]
            .resample(freq)
            .agg(agg_func)
            .reset_index()
        )
        return resampled


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------

ImputeMethod = Literal["drop", "mean", "median", "mode", "group_mode", "ffill", "bfill", "zero"]


def impute_missing(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: ImputeMethod = "median",
    group_col: str | None = None,
) -> pd.DataFrame:
    """Impute missing values in selected columns.

    Parameters
    ----------
    df:
        Source DataFrame.
    columns:
        Columns to impute; defaults to all columns.
    method:
        ``"drop"`` removes rows with any NA in the selected columns.
        ``"mean"``, ``"median"``, ``"mode"`` fill with the statistic.
        ``"group_mode"`` fills with mode per group (requires *group_col*).
        ``"ffill"`` / ``"bfill"`` do forward / backward fill.
        ``"zero"`` fills with 0.
    group_col:
        Column to group by when using ``"group_mode"``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    cols = columns if columns else df.columns.tolist()
    existing = [c for c in cols if c in df.columns]

    if method == "drop":
        return df.dropna(subset=existing).reset_index(drop=True)

    for col in existing:
        series = df[col]
        if series.isna().sum() == 0:
            continue
        if method == "mean":
            fill_val = series.mean()
        elif method == "median":
            fill_val = series.median()
        elif method == "mode":
            modes = series.mode()
            fill_val = modes.iloc[0] if not modes.empty else None
        elif method == "group_mode":
            if group_col and group_col in df.columns:
                df[col] = df.groupby(group_col)[col].transform(
                    lambda s: s.fillna(s.mode().iloc[0] if not s.mode().empty else np.nan)
                )
            else:
                modes = series.mode()
                fill_val = modes.iloc[0] if not modes.empty else None
                if fill_val is not None:
                    df[col] = series.fillna(fill_val)
            continue
        elif method == "ffill":
            df[col] = series.ffill()
            continue
        elif method == "bfill":
            df[col] = series.bfill()
            continue
        elif method == "zero":
            fill_val = 0
        else:
            fill_val = None

        if fill_val is not None:
            df[col] = series.fillna(fill_val)

    return df


# ---------------------------------------------------------------------------
# Outlier detection / removal
# ---------------------------------------------------------------------------

OutlierMethod = Literal["zscore", "iqr", "none"]


def flag_outliers(
    series: pd.Series,
    method: OutlierMethod = "iqr",
    zscore_threshold: float = 3.0,
    iqr_multiplier: float = 1.5,
) -> pd.Series:
    """Return a boolean mask that is True where values are outliers.

    Parameters
    ----------
    series:
        Numeric series to check.
    method:
        Detection method: ``"zscore"`` or ``"iqr"``.
    zscore_threshold:
        Number of standard deviations from the mean (z-score method).
    iqr_multiplier:
        IQR fence multiplier (IQR method).

    Returns
    -------
    pd.Series[bool]
    """
    s = pd.to_numeric(series, errors="coerce")
    if method == "zscore":
        mean, std = s.mean(), s.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        return ((s - mean).abs() / std) > zscore_threshold
    elif method == "iqr":
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        return (s < lower) | (s > upper)
    return pd.Series(False, index=series.index)


def remove_outliers(
    df: pd.DataFrame,
    columns: list[str],
    method: OutlierMethod = "iqr",
    **kwargs,
) -> tuple[pd.DataFrame, int]:
    """Remove rows where any of the specified columns is an outlier.

    Parameters
    ----------
    df:
        Source DataFrame.
    columns:
        Columns to check.
    method:
        Outlier detection method.
    **kwargs:
        Forwarded to :func:`flag_outliers`.

    Returns
    -------
    tuple[pd.DataFrame, int]
        (cleaned DataFrame, number of rows removed)
    """
    mask = pd.Series(False, index=df.index)
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mask = mask | flag_outliers(df[col], method=method, **kwargs)
    removed = int(mask.sum())
    return df[~mask].reset_index(drop=True), removed


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(
    df: pd.DataFrame,
    subset: list[str] | None = None,
    keep: Literal["first", "last"] = "first",
) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows.

    Parameters
    ----------
    df:
        Source DataFrame.
    subset:
        Columns to consider; if None, all columns are used.
    keep:
        Which duplicate to keep: ``"first"`` or ``"last"``.

    Returns
    -------
    tuple[pd.DataFrame, int]
        (de-duplicated DataFrame, number of rows removed)
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    return df, before - len(df)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_lags(
    df: pd.DataFrame,
    col: str,
    lags: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add lag features for a column.

    Parameters
    ----------
    df:
        Source DataFrame (should already be sorted by date).
    col:
        Column to lag.
    lags:
        List of integer lag periods.
    group_col:
        If provided, lags are computed within each group.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for lag in lags:
        lag_name = f"{col}_lag{lag}"
        if group_col and group_col in df.columns:
            df[lag_name] = df.groupby(group_col)[col].shift(lag)
        else:
            df[lag_name] = df[col].shift(lag)
    return df


def add_rolling(
    df: pd.DataFrame,
    col: str,
    windows: list[int],
    func: str = "mean",
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add rolling window features.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Column to compute rolling statistics on.
    windows:
        List of integer window sizes.
    func:
        Aggregation: ``"mean"``, ``"std"``, ``"sum"``, ``"min"``, ``"max"``.
    group_col:
        Group column for grouped rolling.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for w in windows:
        new_col = f"{col}_roll{w}_{func}"
        if group_col and group_col in df.columns:
            df[new_col] = df.groupby(group_col)[col].transform(
                lambda s: s.rolling(w, min_periods=1).agg(func)
            )
        else:
            df[new_col] = df[col].rolling(w, min_periods=1).agg(func)
    return df


def add_ema(
    df: pd.DataFrame,
    col: str,
    spans: list[int],
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add exponentially-weighted moving average features.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Column to smooth.
    spans:
        List of EWM span values (e.g. [3, 6, 12]).
    group_col:
        Group column for grouped EWM.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for span in spans:
        new_col = f"{col}_ema{span}"
        if group_col and group_col in df.columns:
            df[new_col] = df.groupby(group_col)[col].transform(
                lambda s: s.ewm(span=span, adjust=False).mean()
            )
        else:
            df[new_col] = df[col].ewm(span=span, adjust=False).mean()
    return df


def add_buckets(
    df: pd.DataFrame,
    col: str,
    n_quantiles: int | None = 4,
    custom_bins: list[float] | None = None,
    labels: list[str] | None = None,
    new_col: str | None = None,
) -> pd.DataFrame:
    """Bucket a numeric column into quantile or custom bins.

    Parameters
    ----------
    df:
        Source DataFrame.
    col:
        Numeric column to bucket.
    n_quantiles:
        If set, create quantile-based bins.
    custom_bins:
        If set, override n_quantiles with explicit bin edges.
    labels:
        Optional bin labels.
    new_col:
        Name for the new column; defaults to ``{col}_bucket``.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    out_col = new_col or f"{col}_bucket"
    s = pd.to_numeric(df[col], errors="coerce")
    if custom_bins:
        df[out_col] = pd.cut(s, bins=custom_bins, labels=labels, include_lowest=True)
    else:
        q = n_quantiles or 4
        df[out_col] = pd.qcut(s, q=q, labels=labels, duplicates="drop")
    df[out_col] = df[out_col].astype(str)
    return df


def normalize(
    df: pd.DataFrame,
    columns: list[str],
    method: Literal["minmax", "zscore"] = "zscore",
) -> pd.DataFrame:
    """Normalize numeric columns in-place (copy).

    Parameters
    ----------
    df:
        Source DataFrame.
    columns:
        Columns to normalize.
    method:
        ``"minmax"`` scales to [0, 1]; ``"zscore"`` standardizes.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if method == "minmax":
            min_v, max_v = s.min(), s.max()
            if max_v - min_v != 0:
                df[col] = (s - min_v) / (max_v - min_v)
        else:  # zscore
            mean_v, std_v = s.mean(), s.std()
            if std_v != 0:
                df[col] = (s - mean_v) / std_v
    return df


def add_interaction(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    op: Literal["multiply", "divide", "add", "subtract"] = "multiply",
    new_col: str | None = None,
) -> pd.DataFrame:
    """Create an interaction feature between two columns.

    Parameters
    ----------
    df:
        Source DataFrame.
    col_a, col_b:
        Columns to combine.
    op:
        Operation: ``"multiply"``, ``"divide"``, ``"add"``, ``"subtract"``.
    new_col:
        Name for the resulting column.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    a = pd.to_numeric(df[col_a], errors="coerce")
    b = pd.to_numeric(df[col_b], errors="coerce")
    if op == "multiply":
        result = a * b
    elif op == "divide":
        result = a / b.replace(0, np.nan)
    elif op == "add":
        result = a + b
    else:
        result = a - b
    out = new_col or f"{col_a}_{op}_{col_b}"
    df[out] = result
    return df

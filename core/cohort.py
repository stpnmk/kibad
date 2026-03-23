"""
core/cohort.py – Cohort retention analysis for KIBAD.

Provides cohort table building, retention/churn calculations, average
retention curves, and CLV estimation.
Pure functions only — no Streamlit imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cohort table builder
# ---------------------------------------------------------------------------

def build_cohort_table(
    df: pd.DataFrame,
    customer_id_col: str,
    activity_date_col: str,
    acquisition_date_col: str | None = None,
    cohort_freq: str = "MS",
    max_offset: int = 24,
) -> pd.DataFrame:
    """Build a cohort retention table.

    Parameters
    ----------
    df                   : input DataFrame
    customer_id_col      : column with customer identifier
    activity_date_col    : column with activity/transaction date
    acquisition_date_col : column with acquisition/first-seen date;
                           if None, computed as min(activity_date) per customer
    cohort_freq          : "MS" = month start, "QS" = quarter start
    max_offset           : maximum period offset to include

    Returns
    -------
    DataFrame with index=cohort (period start), columns=offset (0,1,2,...),
    values=active customer count.
    """
    if df.empty:
        return pd.DataFrame()

    work = df[[customer_id_col, activity_date_col]].copy()
    work["_activity_date"] = pd.to_datetime(work[activity_date_col], errors="coerce")
    work = work.dropna(subset=["_activity_date"])

    if acquisition_date_col and acquisition_date_col in df.columns:
        work["_acq_date"] = pd.to_datetime(
            df.loc[work.index, acquisition_date_col], errors="coerce"
        )
    else:
        # Compute acquisition date as first activity per customer
        acq = work.groupby(customer_id_col)["_activity_date"].min().rename("_acq_date")
        work = work.join(acq, on=customer_id_col)

    work = work.dropna(subset=["_acq_date"])

    # Assign cohort = acquisition date floored to cohort_freq
    # Map offset aliases (MS, QS) to Period-compatible aliases (M, Q)
    _period_freq = cohort_freq.replace("MS", "M").replace("QS", "Q").replace("YS", "Y")
    work["_cohort"] = work["_acq_date"].dt.to_period(_period_freq).dt.to_timestamp()

    # Activity period
    work["_activity_period"] = work["_activity_date"].dt.to_period(_period_freq).dt.to_timestamp()

    # Offset in periods
    # For monthly: offset = month difference; for quarterly: quarter difference
    if cohort_freq.startswith("M"):
        work["_offset"] = (
            (work["_activity_period"].dt.year - work["_cohort"].dt.year) * 12
            + (work["_activity_period"].dt.month - work["_cohort"].dt.month)
        )
    elif cohort_freq.startswith("Q"):
        work["_offset"] = (
            (work["_activity_period"].dt.year - work["_cohort"].dt.year) * 4
            + (work["_activity_period"].dt.quarter - work["_cohort"].dt.quarter)
        )
    else:
        # Fallback: month difference
        work["_offset"] = (
            (work["_activity_period"].dt.year - work["_cohort"].dt.year) * 12
            + (work["_activity_period"].dt.month - work["_cohort"].dt.month)
        )

    # Filter to valid offsets
    work = work[(work["_offset"] >= 0) & (work["_offset"] <= max_offset)]

    # Count unique customers per (cohort, offset)
    counts = (
        work.groupby(["_cohort", "_offset"])[customer_id_col]
        .nunique()
        .reset_index()
    )
    counts.columns = ["cohort", "offset", "count"]

    # Pivot
    pivot = counts.pivot(index="cohort", columns="offset", values="count").fillna(0).astype(int)
    pivot.index.name = "cohort"
    pivot.columns.name = "offset"

    # Ensure columns 0..max_offset exist
    for col in range(max_offset + 1):
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[sorted(pivot.columns)]

    return pivot


# ---------------------------------------------------------------------------
# Retention table
# ---------------------------------------------------------------------------

def retention_table(cohort_counts: pd.DataFrame) -> pd.DataFrame:
    """Convert cohort count table to retention rates (0–1).

    Divides each row by its value at offset=0.
    """
    if cohort_counts.empty or 0 not in cohort_counts.columns:
        return cohort_counts.copy()

    base = cohort_counts[0].replace(0, np.nan)
    retention = cohort_counts.div(base, axis=0)
    return retention


# ---------------------------------------------------------------------------
# Churn rate table
# ---------------------------------------------------------------------------

def churn_rate_table(retention: pd.DataFrame) -> pd.DataFrame:
    """Compute period-over-period churn rates.

    For offset k > 0: churn(k) = (retention(k-1) - retention(k)) / retention(k-1)
    Returns same shape as input (NaN at offset=0).
    """
    if retention.empty:
        return retention.copy()

    churn = pd.DataFrame(index=retention.index, columns=retention.columns, dtype=float)
    cols = sorted(retention.columns)

    if 0 in cols:
        churn[0] = np.nan

    for i in range(1, len(cols)):
        k = cols[i]
        k_prev = cols[i - 1]
        prev = retention[k_prev]
        curr = retention[k]
        with np.errstate(divide="ignore", invalid="ignore"):
            churn_k = (prev - curr) / prev.replace(0, np.nan)
        churn[k] = churn_k

    return churn


# ---------------------------------------------------------------------------
# Average retention curve
# ---------------------------------------------------------------------------

def average_retention_curve(cohort_counts: pd.DataFrame) -> pd.Series:
    """Compute weighted average retention by offset, weighted by cohort size.

    Weight = cohort size (count at offset=0).
    """
    if cohort_counts.empty or 0 not in cohort_counts.columns:
        return pd.Series(dtype=float)

    ret = retention_table(cohort_counts)
    weights = cohort_counts[0].fillna(0)
    total_weight = weights.sum()

    if total_weight == 0:
        return ret.mean()

    avg = ret.mul(weights, axis=0).sum() / total_weight
    return avg


# ---------------------------------------------------------------------------
# CLV computation
# ---------------------------------------------------------------------------

def compute_clv(
    retention: pd.DataFrame,
    arpu: float,
    annual_discount_rate: float,
    horizon_months: int,
) -> pd.Series:
    """Compute Customer Lifetime Value for each cohort.

    CLV = ARPU * sum_{k=0}^{horizon} retention(k) * discount_factor(k)
    discount_factor(k) = 1 / (1 + annual_discount_rate/12)^k

    Parameters
    ----------
    retention          : retention rate DataFrame (cohort × offset), values 0-1
    arpu               : average revenue per user per period
    annual_discount_rate : annual discount rate as decimal (e.g. 0.12 for 12%)
    horizon_months     : number of months to sum over

    Returns
    -------
    Series indexed by cohort with CLV values
    """
    if retention.empty or arpu <= 0:
        return pd.Series(dtype=float)

    monthly_rate = annual_discount_rate / 12.0
    clv_values = {}

    for cohort in retention.index:
        row = retention.loc[cohort]
        total = 0.0
        for k in range(horizon_months + 1):
            if k not in row.index:
                break
            ret_k = row[k]
            if pd.isna(ret_k):
                ret_k = 0.0
            discount = 1.0 / (1 + monthly_rate) ** k
            total += ret_k * discount
        clv_values[cohort] = arpu * total

    return pd.Series(clv_values, name="CLV")

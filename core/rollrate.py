"""
core/rollrate.py – Roll-rate / transition matrix analysis for KIBAD.

Provides DPD bucketing, transition matrix computation, matrix powers,
steady-state distribution, and roll/cure rate extraction.
Pure functions only — no Streamlit imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered bucket labels used throughout
BUCKET_ORDER = ["Текущий", "1-30", "31-60", "61-90", "90+", "Списан", "Закрыт"]


# ---------------------------------------------------------------------------
# DPD → bucket mapping
# ---------------------------------------------------------------------------

def auto_bucket(
    dpd: pd.Series,
    edges: list[int] | None = None,
) -> pd.Series:
    """Map DPD values to bucket labels.

    Parameters
    ----------
    dpd   : numeric DPD series
    edges : cut-points, default [0, 1, 30, 60, 90, 180]
            Values >= last edge → last non-closed bucket label.
            -1 or NaN → "Закрыт".

    Returns
    -------
    Categorical Series with ordered bucket labels.
    """
    if edges is None:
        edges = [0, 1, 30, 60, 90, 180]

    labels_map = {
        0: "Текущий",   # [0, 1)
        1: "1-30",      # [1, 30)
        2: "31-60",     # [30, 60)
        3: "61-90",     # [60, 90)
        4: "90+",       # [90, 180)
        5: "Списан",    # >= 180
    }
    n_bins = len(edges) - 1

    result = pd.Series(index=dpd.index, dtype=object)

    # Mark closed/unknown
    closed_mask = (dpd == -1) | dpd.isna()
    result[closed_mask] = "Закрыт"

    numeric_mask = ~closed_mask
    numeric_dpd = dpd[numeric_mask]

    # Clip values below 0 to 0 (except -1 handled above)
    numeric_dpd = numeric_dpd.clip(lower=0)

    # Use pd.cut; values >= last edge get NaN → remap to last label
    bins = edges + [np.inf]
    cut_labels = [labels_map.get(i, f"Bucket{i}") for i in range(len(bins) - 1)]
    # Ensure we have enough labels
    while len(cut_labels) < len(bins) - 1:
        cut_labels.append(BUCKET_ORDER[-2])

    bucketed = pd.cut(
        numeric_dpd,
        bins=bins,
        labels=cut_labels,
        right=False,
        include_lowest=True,
    )
    result[numeric_mask] = bucketed.astype(object)

    # Any remaining NaN in result → "Закрыт"
    result = result.fillna("Закрыт")

    return pd.Categorical(result, categories=BUCKET_ORDER, ordered=True)


# ---------------------------------------------------------------------------
# Transition matrix builder
# ---------------------------------------------------------------------------

def build_transition_matrix(
    df: pd.DataFrame,
    loan_id_col: str,
    period_col: str,
    bucket_col: str,
    periods: list | None = None,
    weight_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build transition count and rate matrices.

    Parameters
    ----------
    df          : DataFrame with one row per (loan, period)
    loan_id_col : column with loan identifier
    period_col  : column with period (date or comparable)
    bucket_col  : column with bucket label
    periods     : if provided, filter to only these consecutive period pairs
    weight_col  : optional column for weighting transitions (e.g. EAD)

    Returns
    -------
    (count_matrix, rate_matrix) — both DataFrames indexed/columned by BUCKET_ORDER
    """
    if df.empty:
        empty = pd.DataFrame(0, index=BUCKET_ORDER, columns=BUCKET_ORDER)
        return empty, empty

    work = df[[loan_id_col, period_col, bucket_col]].copy()
    if weight_col:
        work["_weight"] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
    else:
        work["_weight"] = 1.0

    # Deduplicate: keep last record per (loan, period)
    work = work.sort_values([loan_id_col, period_col])
    work = work.drop_duplicates(subset=[loan_id_col, period_col], keep="last")

    # Sort and create "next period" row
    work = work.sort_values([loan_id_col, period_col]).reset_index(drop=True)

    # Shift within each loan group
    work["_next_bucket"] = work.groupby(loan_id_col)[bucket_col].shift(-1)
    work["_next_period"] = work.groupby(loan_id_col)[period_col].shift(-1)

    # Drop rows where next is NaN (last observation per loan)
    transitions = work.dropna(subset=["_next_bucket"])

    # Filter to consecutive period pairs if requested
    if periods is not None:
        period_pairs = set(zip(periods[:-1], periods[1:]))
        transitions = transitions[
            transitions.apply(
                lambda r: (r[period_col], r["_next_period"]) in period_pairs, axis=1
            )
        ]

    if transitions.empty:
        empty = pd.DataFrame(0, index=BUCKET_ORDER, columns=BUCKET_ORDER)
        return empty, empty

    # Build count matrix using crosstab (with weights)
    count_matrix = pd.crosstab(
        transitions[bucket_col],
        transitions["_next_bucket"],
        values=transitions["_weight"],
        aggfunc="sum",
    ).fillna(0)

    # Reindex to BUCKET_ORDER
    present_buckets = [b for b in BUCKET_ORDER if b in count_matrix.index or b in count_matrix.columns]
    count_matrix = count_matrix.reindex(index=BUCKET_ORDER, columns=BUCKET_ORDER, fill_value=0)

    # Rate matrix: normalize rows
    row_sums = count_matrix.sum(axis=1)
    rate_matrix = count_matrix.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

    return count_matrix, rate_matrix


# ---------------------------------------------------------------------------
# Matrix power
# ---------------------------------------------------------------------------

def matrix_power(T: pd.DataFrame, n: int) -> pd.DataFrame:
    """Compute T^n using numpy matrix power.

    Parameters
    ----------
    T : stochastic transition matrix (rows sum to 1)
    n : number of steps

    Returns
    -------
    DataFrame with same index/columns as T
    """
    result = np.linalg.matrix_power(T.values.astype(float), n)
    return pd.DataFrame(result, index=T.index, columns=T.columns)


# ---------------------------------------------------------------------------
# Steady-state distribution
# ---------------------------------------------------------------------------

def steady_state(T: pd.DataFrame) -> pd.Series:
    """Find stationary distribution π such that πT = π.

    Uses left eigenvectors of T (eigenvectors of T.T for eigenvalue ≈ 1).
    Normalizes to sum to 1.

    Returns
    -------
    Series indexed by T.index
    """
    T_np = T.values.astype(float)
    eigenvalues, eigenvectors = np.linalg.eig(T_np.T)

    # Find eigenvector for eigenvalue closest to 1 (must be within tolerance)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    if np.abs(eigenvalues[idx] - 1.0) > 1e-6:
        import warnings
        warnings.warn("No eigenvalue near 1.0 found; matrix may not be stochastic")
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)
    total = pi.sum()
    if total == 0:
        pi = np.ones(len(pi)) / len(pi)
    else:
        pi = pi / total

    return pd.Series(pi, index=T.index)


# ---------------------------------------------------------------------------
# Roll-forward and cure rates
# ---------------------------------------------------------------------------

def roll_forward_rates(rate_matrix: pd.DataFrame) -> pd.Series:
    """For each bucket i, roll_forward_rate = sum of T[i,j] for j > i (worsening)."""
    rates = {}
    buckets = [b for b in BUCKET_ORDER if b in rate_matrix.index]
    for bucket in buckets:
        if bucket not in rate_matrix.index:
            rates[bucket] = 0.0
            continue
        row = rate_matrix.loc[bucket]
        bucket_idx = BUCKET_ORDER.index(bucket)
        # Columns with higher index = worse status
        worse = [b for b in BUCKET_ORDER if BUCKET_ORDER.index(b) > bucket_idx and b in row.index]
        rates[bucket] = float(row[worse].sum()) if worse else 0.0
    return pd.Series(rates)


def cure_rates(rate_matrix: pd.DataFrame) -> pd.Series:
    """For each bucket i, cure_rate = sum of T[i,j] for j < i (improvement)."""
    rates = {}
    buckets = [b for b in BUCKET_ORDER if b in rate_matrix.index]
    for bucket in buckets:
        if bucket not in rate_matrix.index:
            rates[bucket] = 0.0
            continue
        row = rate_matrix.loc[bucket]
        bucket_idx = BUCKET_ORDER.index(bucket)
        better = [b for b in BUCKET_ORDER if BUCKET_ORDER.index(b) < bucket_idx and b in row.index]
        rates[bucket] = float(row[better].sum()) if better else 0.0
    return pd.Series(rates)


# ---------------------------------------------------------------------------
# Transition time series
# ---------------------------------------------------------------------------

def transition_time_series(
    df: pd.DataFrame,
    loan_id_col: str,
    period_col: str,
    bucket_col: str,
) -> pd.DataFrame:
    """For each unique period transition pair, compute roll-forward and cure rates.

    Returns
    -------
    DataFrame with columns: period, source_bucket, roll_forward_rate, cure_rate
    """
    if df.empty:
        return pd.DataFrame(columns=["period", "source_bucket", "roll_forward_rate", "cure_rate"])

    work = df[[loan_id_col, period_col, bucket_col]].copy()
    work = work.sort_values([loan_id_col, period_col])
    work = work.drop_duplicates(subset=[loan_id_col, period_col], keep="last")

    work["_next_bucket"] = work.groupby(loan_id_col)[bucket_col].shift(-1)
    work["_next_period"] = work.groupby(loan_id_col)[period_col].shift(-1)
    transitions = work.dropna(subset=["_next_bucket", "_next_period"])

    if transitions.empty:
        return pd.DataFrame(columns=["period", "source_bucket", "roll_forward_rate", "cure_rate"])

    records = []
    for period_val in sorted(transitions[period_col].unique()):
        period_df = transitions[transitions[period_col] == period_val]
        count_m, rate_m = build_transition_matrix(
            period_df,
            loan_id_col=loan_id_col,
            period_col=period_col,
            bucket_col=bucket_col,
        )
        rf = roll_forward_rates(rate_m)
        cr = cure_rates(rate_m)
        for bucket in rf.index:
            records.append({
                "period": period_val,
                "source_bucket": bucket,
                "roll_forward_rate": rf[bucket],
                "cure_rate": cr[bucket],
            })

    return pd.DataFrame(records)

"""
core/vintage.py – Vintage analysis (CDR curves) for KIBAD.

Provides cohort-based cumulative default rate calculations,
Wilson confidence intervals, and vintage summary statistics.
Pure functions only — no Streamlit imports.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MOB computation
# ---------------------------------------------------------------------------

def compute_mob(
    df: pd.DataFrame,
    origination_date_col: str,
    observation_date_col: str,
) -> pd.Series:
    """Return integer Series: MOB = months between origination and observation.

    MOB = (obs.year - orig.year)*12 + (obs.month - orig.month).
    Negative values are clipped to 0.
    """
    orig = pd.to_datetime(df[origination_date_col], errors="coerce")
    obs = pd.to_datetime(df[observation_date_col], errors="coerce")
    mob = (obs.dt.year - orig.dt.year) * 12 + (obs.dt.month - orig.dt.month)
    return mob.clip(lower=0).astype("Int64")


# ---------------------------------------------------------------------------
# Vintage pivot builder
# ---------------------------------------------------------------------------

def build_vintage_pivot(
    df: pd.DataFrame,
    origination_date_col: str,
    observation_date_col: str,
    default_flag_col: str,
    cohort_freq: str = "MS",
    max_mob: int = 24,
    min_obs_threshold: int = 10,
) -> dict[str, Any]:
    """Build cohort × MOB pivot tables for CDR and marginal default rates.

    Returns
    -------
    dict with keys:
        "cdr_pivot"     : cohort × MOB DataFrame of cumulative default rates (0-1)
        "mdr_pivot"     : cohort × MOB DataFrame of marginal default rates (0-1)
        "count_pivot"   : cohort × MOB DataFrame of observation counts
        "cohort_sizes"  : Series of N per cohort (at MOB 0)
        "maturity_mask" : bool DataFrame — True where count >= threshold (mature cell)
    """
    if df.empty:
        empty = pd.DataFrame()
        return {
            "cdr_pivot": empty,
            "mdr_pivot": empty,
            "count_pivot": empty,
            "cohort_sizes": pd.Series(dtype=float),
            "maturity_mask": empty,
        }

    work = df[[origination_date_col, observation_date_col, default_flag_col]].copy()
    work["_orig"] = pd.to_datetime(work[origination_date_col], errors="coerce")
    work["_obs"] = pd.to_datetime(work[observation_date_col], errors="coerce")
    work = work.dropna(subset=["_orig", "_obs"])

    # MOB
    work["_mob"] = (
        (work["_obs"].dt.year - work["_orig"].dt.year) * 12
        + (work["_obs"].dt.month - work["_orig"].dt.month)
    ).clip(lower=0)

    # Cohort assignment — map offset aliases (MS, QS) to Period-compatible aliases
    _period_freq = cohort_freq.replace("MS", "M").replace("QS", "Q").replace("YS", "Y")
    work["_cohort"] = work["_orig"].dt.to_period(_period_freq).dt.to_timestamp()

    # Ensure default flag is numeric
    work["_default"] = pd.to_numeric(work[default_flag_col], errors="coerce").fillna(0)

    # Filter to max_mob
    work = work[work["_mob"] <= max_mob]

    # Cohort sizes: count unique origination records per cohort
    # (using MOB=0 as the cohort base)
    mob0 = work[work["_mob"] == 0]
    cohort_sizes = mob0.groupby("_cohort").size()

    # For CDR: cumulative defaults — a loan that defaulted at MOB k
    # should be counted as defaulted at all subsequent MOBs too.
    # We compute this by: for each (cohort, mob), count loans whose
    # default occurred at any mob <= current mob.
    # First get the first MOB where each loan defaulted.
    # We use origination_date + observation_date as loan identifier proxy
    # (using positional index if no explicit loan_id).
    work = work.reset_index(drop=True)

    # Group by (cohort, mob): count total observations and defaults
    grp = work.groupby(["_cohort", "_mob"]).agg(
        _count=("_default", "count"),
        _defaults=("_default", "sum"),
    ).reset_index()

    # Build cumulative defaults per cohort:
    # CDR(k) = (total defaults at MOBs 0..k) / cohort_size
    cohorts = sorted(grp["_cohort"].unique())
    mob_range = list(range(0, max_mob + 1))

    cdr_rows = {}
    mdr_rows = {}
    count_rows = {}

    for cohort in cohorts:
        cg = grp[grp["_cohort"] == cohort].set_index("_mob")
        n = cohort_sizes.get(cohort, 0)
        if n == 0:
            continue

        cdr_row = {}
        mdr_row = {}
        count_row = {}
        cum_defaults = 0.0

        for mob in mob_range:
            if mob in cg.index:
                row_defaults = float(cg.loc[mob, "_defaults"])
                row_count = int(cg.loc[mob, "_count"])
            else:
                row_defaults = 0.0
                row_count = 0

            cum_defaults += row_defaults
            cdr_row[mob] = cum_defaults / n
            mdr_row[mob] = row_defaults / n
            count_row[mob] = row_count

        cdr_rows[cohort] = cdr_row
        mdr_rows[cohort] = mdr_row
        count_rows[cohort] = count_row

    cdr_pivot = pd.DataFrame(cdr_rows).T
    mdr_pivot = pd.DataFrame(mdr_rows).T
    count_pivot = pd.DataFrame(count_rows).T

    # Align columns to mob_range
    for piv in [cdr_pivot, mdr_pivot, count_pivot]:
        for mob in mob_range:
            if mob not in piv.columns:
                piv[mob] = np.nan
        piv.sort_index(axis=1, inplace=True)
        piv.index.name = "cohort"

    # Remove MOB 0 from CDR/MDR display (optional: keep for completeness)
    # Keep all; caller can filter.

    # Maturity mask: True where count >= threshold
    maturity_mask = count_pivot >= min_obs_threshold

    return {
        "cdr_pivot": cdr_pivot,
        "mdr_pivot": mdr_pivot,
        "count_pivot": count_pivot,
        "cohort_sizes": cohort_sizes,
        "maturity_mask": maturity_mask,
    }


# ---------------------------------------------------------------------------
# Wilson confidence interval
# ---------------------------------------------------------------------------

def wilson_ci(p: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return (lower, upper) Wilson confidence interval for proportion p.

    Parameters
    ----------
    p     : observed proportion (0–1)
    n     : sample size
    alpha : significance level (default 0.05 → 95% CI)
    """
    if n <= 0 or math.isnan(p):
        return (0.0, 1.0)

    from scipy import stats as _stats
    z = _stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return (lower, upper)


# ---------------------------------------------------------------------------
# Vintage summary
# ---------------------------------------------------------------------------

def vintage_summary(
    cdr_pivot: pd.DataFrame,
    maturity_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Per-cohort summary: N, total defaults observed, max CDR (mature only), MOB at first CDR>5%.

    Parameters
    ----------
    cdr_pivot      : cohort × MOB CDR DataFrame (values 0-1)
    maturity_mask  : bool DataFrame — True where cell is mature

    Returns
    -------
    DataFrame with columns: cohort, max_mob_mature, max_cdr, mob_at_5pct
    """
    if cdr_pivot.empty:
        return pd.DataFrame(columns=["cohort", "max_mob_mature", "max_cdr", "mob_at_5pct"])

    records = []
    for cohort in cdr_pivot.index:
        row = cdr_pivot.loc[cohort]
        mask_row = maturity_mask.loc[cohort] if cohort in maturity_mask.index else pd.Series(True, index=row.index)

        mature_vals = row[mask_row]
        max_mob_mature = int(mature_vals.index.max()) if not mature_vals.empty else 0
        max_cdr = float(mature_vals.max()) if not mature_vals.empty else np.nan

        # MOB at first CDR > 5%
        above_5 = row[row > 0.05]
        mob_at_5pct = int(above_5.index.min()) if not above_5.empty else np.nan

        records.append({
            "cohort": cohort,
            "max_mob_mature": max_mob_mature,
            "max_cdr": max_cdr,
            "mob_at_5pct": mob_at_5pct,
        })

    return pd.DataFrame(records)

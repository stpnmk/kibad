"""
core/weighted_avg.py – Weighted average metrics for portfolio analysis (KIBAD).

Provides weighted average, weighted std, weighted percentile, portfolio
weighted averages with grouping, mix/rate effect decomposition, and
simplified duration calculations.
Pure functions only — no Streamlit imports.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar weighted statistics
# ---------------------------------------------------------------------------

def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted average.

    Handles zero-weight edge case by returning NaN.
    """
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)

    mask = values.notna() & (weights > 0)
    if mask.sum() == 0:
        return float("nan")

    return float(np.average(values[mask], weights=weights[mask]))


def weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Compute weighted standard deviation.

    Formula: sqrt(sum(w * (x - xbar)^2) / sum(w))
    """
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)

    mask = values.notna() & (weights > 0)
    if mask.sum() < 2:
        return float("nan")

    xbar = np.average(values[mask], weights=weights[mask])
    w = weights[mask].values.astype(float)
    x = values[mask].values.astype(float)
    variance = np.sum(w * (x - xbar) ** 2) / np.sum(w)
    return float(math.sqrt(variance))


def weighted_percentile(values: pd.Series, weights: pd.Series, q: float) -> float:
    """Compute weighted percentile by cumulative weight fraction interpolation.

    Parameters
    ----------
    values  : numeric series
    weights : non-negative weight series
    q       : percentile as fraction 0–1

    Returns
    -------
    float — interpolated value at quantile q
    """
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").fillna(0)

    mask = values.notna() & (weights > 0)
    if mask.sum() == 0:
        return float("nan")

    v = values[mask].values.astype(float)
    w = weights[mask].values.astype(float)

    # Sort by values
    sort_idx = np.argsort(v)
    v_sorted = v[sort_idx]
    w_sorted = w[sort_idx]

    # Cumulative weight fractions (midpoint method)
    cum_w = np.cumsum(w_sorted)
    total_w = cum_w[-1]
    # Use midpoint: (cum_w - 0.5*w) / total_w
    midpoints = (cum_w - 0.5 * w_sorted) / total_w

    return float(np.interp(q, midpoints, v_sorted))


# ---------------------------------------------------------------------------
# Portfolio-level weighted averages
# ---------------------------------------------------------------------------

def portfolio_weighted_averages(
    df: pd.DataFrame,
    weight_col: str,
    metric_cols: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute weighted average statistics for each metric, optionally grouped.

    Returns
    -------
    DataFrame with one row per group (or single row if no groups).
    Columns: {weight_col}_sum, and for each metric:
        {metric}_wa, {metric}_wstd, {metric}_wmin, {metric}_wmax
    """
    if df.empty:
        return pd.DataFrame()

    def _compute_row(sub: pd.DataFrame) -> dict[str, Any]:
        w = pd.to_numeric(sub[weight_col], errors="coerce").fillna(0)
        row: dict[str, Any] = {f"{weight_col}_sum": float(w.sum())}
        for mc in metric_cols:
            if mc not in sub.columns:
                continue
            v = pd.to_numeric(sub[mc], errors="coerce")
            mask = v.notna() & (w > 0)
            row[f"{mc}_wa"] = weighted_average(v, w)
            row[f"{mc}_wstd"] = weighted_std(v, w)
            row[f"{mc}_wmin"] = float(v[mask].min()) if mask.sum() > 0 else float("nan")
            row[f"{mc}_wmax"] = float(v[mask].max()) if mask.sum() > 0 else float("nan")
        return row

    if group_cols:
        rows = []
        group_keys = []
        for group_vals, sub in df.groupby(group_cols):
            row = _compute_row(sub)
            if isinstance(group_vals, tuple):
                for gc, gv in zip(group_cols, group_vals):
                    row[gc] = gv
            else:
                row[group_cols[0]] = group_vals
            rows.append(row)
            group_keys.append(group_vals)
        result = pd.DataFrame(rows)
        # Reorder: group cols first
        cols_order = group_cols + [c for c in result.columns if c not in group_cols]
        result = result[cols_order]
        return result.reset_index(drop=True)
    else:
        row = _compute_row(df)
        return pd.DataFrame([row])


# ---------------------------------------------------------------------------
# Mix-rate decomposition
# ---------------------------------------------------------------------------

def mix_rate_decomposition(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    weight_col: str,
    rate_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Decompose the change in weighted average rate into mix and rate effects.

    For each group k:
        mix_effect_k  = (s_k_B - s_k_A) * r_k_A
        rate_effect_k = s_k_B * (r_k_B - r_k_A)
        total_effect_k = mix_effect_k + rate_effect_k

    Returns
    -------
    DataFrame with columns:
        group, share_a, share_b, rate_a, rate_b, mix_effect, rate_effect, total_effect
    """
    def _group_stats(df: pd.DataFrame) -> pd.DataFrame:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        total_w = w.sum()
        grp_w = df.groupby(group_col)[weight_col].apply(
            lambda g: pd.to_numeric(g, errors="coerce").fillna(0).sum()
        )
        grp_r = df.groupby(group_col)[[rate_col, weight_col]].apply(
            lambda g: weighted_average(
                pd.to_numeric(g[rate_col], errors="coerce"),
                pd.to_numeric(g[weight_col], errors="coerce").fillna(0),
            ), include_groups=False
        )
        shares = grp_w / total_w if total_w > 0 else grp_w * 0
        return pd.DataFrame({"weight": grp_w, "share": shares, "rate": grp_r})

    stats_a = _group_stats(df_a)
    stats_b = _group_stats(df_b)

    all_groups = stats_a.index.union(stats_b.index)
    records = []
    for g in all_groups:
        s_a = float(stats_a.loc[g, "share"]) if g in stats_a.index else 0.0
        s_b = float(stats_b.loc[g, "share"]) if g in stats_b.index else 0.0
        r_a = float(stats_a.loc[g, "rate"]) if g in stats_a.index else 0.0
        r_b = float(stats_b.loc[g, "rate"]) if g in stats_b.index else 0.0

        mix_eff = (s_b - s_a) * r_a
        rate_eff = s_b * (r_b - r_a)
        total_eff = mix_eff + rate_eff

        records.append({
            "group": g,
            "share_a": s_a,
            "share_b": s_b,
            "rate_a": r_a,
            "rate_b": r_b,
            "mix_effect": mix_eff,
            "rate_effect": rate_eff,
            "total_effect": total_eff,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Simplified duration
# ---------------------------------------------------------------------------

def simplified_duration(
    war: float,
    wam_months: float,
    coupon_freq_per_year: int = 12,
) -> dict[str, float]:
    """Compute simplified bond-style duration metrics.

    Parameters
    ----------
    war               : weighted average rate (%) e.g. 5.5 for 5.5%
    wam_months        : weighted average maturity in months
    coupon_freq_per_year: coupon payments per year (default 12 = monthly)

    Returns
    -------
    dict: macaulay_years, modified_duration, dv01
    """
    macaulay_years = wam_months / 12.0
    modified = macaulay_years / (1 + war / 100.0 / coupon_freq_per_year)
    dv01 = -modified / 10000.0

    return {
        "macaulay_years": macaulay_years,
        "modified_duration": modified,
        "dv01": dv01,
    }

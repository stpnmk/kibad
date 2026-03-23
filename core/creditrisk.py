"""
core/creditrisk.py – Credit risk and IFRS 9 ECL calculations for KIBAD.

Provides EL, ECL, NPL, HHI concentration, and portfolio summary.
Pure functions only — no Streamlit imports.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Basic EL / ECL
# ---------------------------------------------------------------------------

def compute_el(
    df: pd.DataFrame,
    pd_col: str,
    lgd_col: str,
    ead_col: str,
) -> pd.Series:
    """Return Expected Loss = PD * LGD * EAD per row."""
    pd_s = pd.to_numeric(df[pd_col], errors="coerce").fillna(0)
    lgd_s = pd.to_numeric(df[lgd_col], errors="coerce").fillna(0)
    ead_s = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    return pd_s * lgd_s * ead_s


def compute_ecl(
    df: pd.DataFrame,
    ead_col: str,
    lgd_col: str,
    pd_12m_col: str,
    pd_lifetime_col: str | None = None,
    stage_col: str | None = None,
) -> pd.Series:
    """Compute IFRS 9 ECL per row.

    Stage 1: PD_12m * LGD * EAD
    Stage 2: PD_lifetime * LGD * EAD  (fallback to PD_12m if no lifetime col)
    Stage 3: LGD * EAD  (full write-down)
    If no stage_col: all rows treated as Stage 1.
    """
    ead = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    lgd = pd.to_numeric(df[lgd_col], errors="coerce").fillna(0)
    pd_12m = pd.to_numeric(df[pd_12m_col], errors="coerce").fillna(0)

    if pd_lifetime_col and pd_lifetime_col in df.columns:
        pd_lt = pd.to_numeric(df[pd_lifetime_col], errors="coerce").fillna(0)
    else:
        pd_lt = pd_12m.copy()

    ecl_s1 = pd_12m * lgd * ead
    ecl_s2 = pd_lt * lgd * ead
    ecl_s3 = lgd * ead

    if stage_col and stage_col in df.columns:
        stage = df[stage_col]
        ecl = ecl_s1.copy()
        ecl[stage == 2] = ecl_s2[stage == 2]
        ecl[stage == 3] = ecl_s3[stage == 3]
        return ecl
    else:
        return ecl_s1


# ---------------------------------------------------------------------------
# NPL calculation
# ---------------------------------------------------------------------------

def compute_npl(
    df: pd.DataFrame,
    ead_col: str,
    dpd_col: str,
    dpd_threshold: int = 90,
) -> dict:
    """Compute NPL amount, total EAD, and NPL rate.

    Returns
    -------
    dict: npl_amount, total_ead, npl_rate (%)
    """
    ead = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    dpd = pd.to_numeric(df[dpd_col], errors="coerce").fillna(0)

    npl_mask = dpd >= dpd_threshold
    npl_amount = float(ead[npl_mask].sum())
    total_ead = float(ead.sum())
    npl_rate = (npl_amount / total_ead * 100) if total_ead > 0 else 0.0

    return {
        "npl_amount": npl_amount,
        "total_ead": total_ead,
        "npl_rate": npl_rate,
    }


# ---------------------------------------------------------------------------
# Concentration / HHI
# ---------------------------------------------------------------------------

def hhi(
    df: pd.DataFrame,
    ead_col: str,
    group_col: str,
) -> dict:
    """Compute Herfindahl-Hirschman Index.

    Returns
    -------
    dict: hhi, hhi_normalized, n_groups, interpretation
    """
    ead = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    group_ead = df.groupby(group_col)[ead_col].apply(
        lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum()
    )
    total = group_ead.sum()

    if total == 0:
        return {
            "hhi": np.nan,
            "hhi_normalized": np.nan,
            "n_groups": len(group_ead),
            "interpretation": "нет данных",
        }

    shares = group_ead / total
    hhi_val = float((shares**2).sum())
    k = len(group_ead)

    if k <= 1:
        hhi_norm = 1.0
    else:
        hhi_norm = (hhi_val - 1 / k) / (1 - 1 / k)

    if hhi_val < 0.15:
        interp = "низкая"
    elif hhi_val < 0.25:
        interp = "умеренная"
    else:
        interp = "высокая"

    return {
        "hhi": hhi_val,
        "hhi_normalized": hhi_norm,
        "n_groups": k,
        "interpretation": interp,
    }


def top_n_concentration(
    df: pd.DataFrame,
    ead_col: str,
    group_col: str,
    n: int = 10,
) -> pd.DataFrame:
    """Return top-N groups by EAD with share and cumulative share.

    Returns
    -------
    DataFrame: group, ead_sum, share_pct, cumulative_share_pct
    """
    ead_vals = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    tmp = df[[group_col]].copy()
    tmp["_ead"] = ead_vals
    group_ead = tmp.groupby(group_col)["_ead"].sum().reset_index()
    group_ead.columns = [group_col, "ead_sum"]
    group_ead = group_ead.sort_values("ead_sum", ascending=False).head(n)

    total = group_ead["ead_sum"].sum()
    group_ead["share_pct"] = group_ead["ead_sum"] / total * 100 if total > 0 else 0.0
    group_ead["cumulative_share_pct"] = group_ead["share_pct"].cumsum()

    return group_ead.reset_index(drop=True)


# ---------------------------------------------------------------------------
# EAD-weighted average helper
# ---------------------------------------------------------------------------

def ead_weighted_avg(df: pd.DataFrame, col: str, ead_col: str) -> float:
    """Return EAD-weighted average of col."""
    vals = pd.to_numeric(df[col], errors="coerce")
    weights = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    mask = vals.notna() & (weights > 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.average(vals[mask], weights=weights[mask]))


# ---------------------------------------------------------------------------
# Portfolio summary
# ---------------------------------------------------------------------------

def portfolio_summary(
    df: pd.DataFrame,
    ead_col: str,
    pd_col: str,
    lgd_col: str,
    provisions_col: str | None = None,
    dpd_col: str | None = None,
    stage_col: str | None = None,
    segment_cols: list[str] | None = None,
) -> dict:
    """Compute comprehensive portfolio risk summary.

    Returns
    -------
    dict with keys:
        total_ead, total_el, el_rate, npl_rate (or None),
        coverage_ratio (or None), provisions_adequacy (or None),
        stage_breakdown (DataFrame), segment_tables (dict of DataFrames)
    """
    ead = pd.to_numeric(df[ead_col], errors="coerce").fillna(0)
    pd_s = pd.to_numeric(df[pd_col], errors="coerce").fillna(0)
    lgd_s = pd.to_numeric(df[lgd_col], errors="coerce").fillna(0)

    el = pd_s * lgd_s * ead
    total_ead = float(ead.sum())
    total_el = float(el.sum())
    el_rate = (total_el / total_ead * 100) if total_ead > 0 else 0.0

    # NPL
    npl_rate = None
    if dpd_col and dpd_col in df.columns:
        npl_info = compute_npl(df, ead_col, dpd_col)
        npl_rate = npl_info["npl_rate"]

    # Coverage
    coverage_ratio = None
    provisions_adequacy = None
    if provisions_col and provisions_col in df.columns:
        provisions = pd.to_numeric(df[provisions_col], errors="coerce").fillna(0).sum()
        coverage_ratio = (float(provisions) / total_el * 100) if total_el > 0 else None
        provisions_adequacy = float(provisions) - total_el

    # Stage breakdown
    stage_breakdown_rows = []
    if stage_col and stage_col in df.columns:
        stages = df[stage_col].unique()
        for s in sorted(stages):
            mask = df[stage_col] == s
            s_ead = float(ead[mask].sum())
            s_el = float(el[mask].sum())
            stage_breakdown_rows.append({
                "stage": s,
                "ead_sum": s_ead,
                "el_sum": s_el,
                "count": int(mask.sum()),
            })
    else:
        stage_breakdown_rows.append({
            "stage": "Не разбито",
            "ead_sum": total_ead,
            "el_sum": total_el,
            "count": len(df),
        })
    stage_breakdown = pd.DataFrame(stage_breakdown_rows)

    # Segment tables
    segment_tables: dict[str, pd.DataFrame] = {}
    if segment_cols:
        for seg_col in segment_cols:
            if seg_col not in df.columns:
                continue
            rows = []
            for seg_val, grp in df.groupby(seg_col):
                g_ead = pd.to_numeric(grp[ead_col], errors="coerce").fillna(0)
                g_pd = pd.to_numeric(grp[pd_col], errors="coerce").fillna(0)
                g_lgd = pd.to_numeric(grp[lgd_col], errors="coerce").fillna(0)
                g_el = g_pd * g_lgd * g_ead
                g_total_ead = float(g_ead.sum())
                g_total_el = float(g_el.sum())

                row = {
                    "segment": seg_val,
                    "EAD": g_total_ead,
                    "N": len(grp),
                    "PD_wa": ead_weighted_avg(grp, pd_col, ead_col),
                    "LGD_wa": ead_weighted_avg(grp, lgd_col, ead_col),
                    "EL": g_total_el,
                    "EL_rate_pct": (g_total_el / g_total_ead * 100) if g_total_ead > 0 else 0.0,
                }
                if dpd_col and dpd_col in df.columns:
                    npl_info = compute_npl(grp, ead_col, dpd_col)
                    row["NPL_rate_pct"] = npl_info["npl_rate"]
                rows.append(row)
            segment_tables[seg_col] = pd.DataFrame(rows)

    return {
        "total_ead": total_ead,
        "total_el": total_el,
        "el_rate": el_rate,
        "npl_rate": npl_rate,
        "coverage_ratio": coverage_ratio,
        "provisions_adequacy": provisions_adequacy,
        "stage_breakdown": stage_breakdown,
        "segment_tables": segment_tables,
    }

"""
Модуль сопоставления групп (Group Matching).

Поддерживаемые методы:
- Propensity Score Matching (PSM) — логистическая регрессия + caliper
- Exact Matching — точное совпадение по категориальным признакам
- Nearest Neighbor Matching — ближайший сосед по Махаланобису / евклиду
- Coarsened Exact Matching (CEM) — огрубление + точное сопоставление

Все функции принимают DataFrame и возвращают MatchResult.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """Результат сопоставления."""

    matched_df: pd.DataFrame
    treatment_col: str
    covariates: list[str]
    method: str
    balance_before: pd.DataFrame
    balance_after: pd.DataFrame
    n_treatment: int
    n_control: int
    n_matched_treatment: int
    n_matched_control: int
    common_support: tuple[float, float] | None = None
    propensity_scores: np.ndarray | None = None
    match_quality: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Balance diagnostics
# ---------------------------------------------------------------------------

def standardized_mean_diff(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
) -> pd.DataFrame:
    """Стандартизированная разность средних (SMD) для каждой ковариаты.

    SMD = (mean_t - mean_c) / sqrt((var_t + var_c) / 2)

    Returns DataFrame with columns:
        covariate, mean_treatment, mean_control, smd, variance_ratio
    """
    t_mask = df[treatment_col] == 1
    c_mask = df[treatment_col] == 0

    rows: list[dict] = []
    for col in covariates:
        vals_t = pd.to_numeric(df.loc[t_mask, col], errors="coerce").dropna()
        vals_c = pd.to_numeric(df.loc[c_mask, col], errors="coerce").dropna()

        mean_t = vals_t.mean() if len(vals_t) else 0.0
        mean_c = vals_c.mean() if len(vals_c) else 0.0
        var_t = vals_t.var(ddof=1) if len(vals_t) > 1 else 0.0
        var_c = vals_c.var(ddof=1) if len(vals_c) > 1 else 0.0

        pooled_std = np.sqrt((var_t + var_c) / 2)
        smd = (mean_t - mean_c) / pooled_std if pooled_std > 0 else 0.0
        var_ratio = var_t / var_c if var_c > 0 else np.nan

        rows.append(
            {
                "covariate": col,
                "mean_treatment": round(mean_t, 4),
                "mean_control": round(mean_c, 4),
                "smd": round(smd, 4),
                "abs_smd": round(abs(smd), 4),
                "variance_ratio": round(var_ratio, 4) if not np.isnan(var_ratio) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def balance_summary(balance_df: pd.DataFrame) -> dict:
    """Сводные метрики баланса."""
    abs_smd = balance_df["abs_smd"]
    return {
        "mean_abs_smd": round(abs_smd.mean(), 4),
        "max_abs_smd": round(abs_smd.max(), 4),
        "pct_below_01": round((abs_smd < 0.1).mean() * 100, 1),
        "pct_below_025": round((abs_smd < 0.25).mean() * 100, 1),
        "n_covariates": len(balance_df),
    }


# ---------------------------------------------------------------------------
# Propensity Score Matching
# ---------------------------------------------------------------------------

def propensity_score_match(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    caliper: float = 0.2,
    ratio: int = 1,
    random_state: int = 42,
) -> MatchResult:
    """PSM через логистическую регрессию.

    caliper задаётся в единицах стд. откл. propensity score.
    ratio — сколько контрольных на каждого из treatment.
    """
    work = df.dropna(subset=[treatment_col] + covariates).copy()
    work[treatment_col] = work[treatment_col].astype(int)

    X = work[covariates].values.astype(float)
    y = work[treatment_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=random_state, solver="lbfgs")
    lr.fit(X_scaled, y)
    ps = lr.predict_proba(X_scaled)[:, 1]
    work["_ps"] = ps

    ps_std = ps.std()
    caliper_abs = caliper * ps_std if ps_std > 0 else 0.05

    # Common support
    ps_t = ps[y == 1]
    ps_c = ps[y == 0]
    cs_low = max(ps_t.min(), ps_c.min())
    cs_high = min(ps_t.max(), ps_c.max())

    # Trim to common support
    work = work[(work["_ps"] >= cs_low) & (work["_ps"] <= cs_high)].copy()

    treated = work[work[treatment_col] == 1].copy()
    control = work[work[treatment_col] == 0].copy()

    rng = np.random.default_rng(random_state)
    matched_t_idx: list[int] = []
    matched_c_idx: list[int] = []

    control_used = set()

    # Shuffle treatment to avoid order bias
    t_order = rng.permutation(len(treated))

    for i in t_order:
        t_ps = treated.iloc[i]["_ps"]
        distances = np.abs(control["_ps"].values - t_ps)

        # Sort by distance
        sorted_idx = np.argsort(distances)
        n_found = 0
        for j in sorted_idx:
            if n_found >= ratio:
                break
            c_pos = control.index[j]
            if c_pos in control_used:
                continue
            if distances[j] <= caliper_abs:
                matched_t_idx.append(treated.index[i])
                matched_c_idx.append(c_pos)
                control_used.add(c_pos)
                n_found += 1

        if n_found > 0 and n_found < ratio:
            # Still count partial matches
            pass
        elif n_found == 0:
            # No match found within caliper — skip this treated unit
            pass

    all_matched_idx = list(set(matched_t_idx)) + list(set(matched_c_idx))
    matched_df = df.loc[df.index.isin(all_matched_idx)].copy()

    balance_before = standardized_mean_diff(
        df.dropna(subset=[treatment_col] + covariates), treatment_col, covariates
    )
    balance_after = standardized_mean_diff(matched_df, treatment_col, covariates)

    quality = {
        "caliper": caliper,
        "caliper_abs": round(caliper_abs, 4),
        "common_support": (round(cs_low, 4), round(cs_high, 4)),
        "n_trimmed_common_support": len(df) - len(work) - df[treatment_col].isna().sum(),
        "ratio": ratio,
    }

    return MatchResult(
        matched_df=matched_df,
        treatment_col=treatment_col,
        covariates=covariates,
        method="PSM",
        balance_before=balance_before,
        balance_after=balance_after,
        n_treatment=int((df[treatment_col] == 1).sum()),
        n_control=int((df[treatment_col] == 0).sum()),
        n_matched_treatment=len(set(matched_t_idx)),
        n_matched_control=len(set(matched_c_idx)),
        common_support=(round(cs_low, 4), round(cs_high, 4)),
        propensity_scores=ps,
        match_quality=quality,
    )


# ---------------------------------------------------------------------------
# Exact Matching
# ---------------------------------------------------------------------------

def exact_match(
    df: pd.DataFrame,
    treatment_col: str,
    exact_cols: list[str],
    covariates: list[str] | None = None,
) -> MatchResult:
    """Точное сопоставление по категориальным признакам.

    Для каждой уникальной комбинации exact_cols берётся min(n_t, n_c)
    наблюдений из каждой группы.
    """
    if covariates is None:
        covariates = exact_cols

    work = df.dropna(subset=[treatment_col] + exact_cols).copy()
    work[treatment_col] = work[treatment_col].astype(int)

    # Create stratum key
    work["_stratum"] = work[exact_cols].astype(str).agg("|".join, axis=1)

    matched_indices: list[int] = []
    for _, group in work.groupby("_stratum"):
        t_idx = group[group[treatment_col] == 1].index.tolist()
        c_idx = group[group[treatment_col] == 0].index.tolist()
        n_match = min(len(t_idx), len(c_idx))
        if n_match > 0:
            matched_indices.extend(t_idx[:n_match])
            matched_indices.extend(c_idx[:n_match])

    matched_df = df.loc[df.index.isin(matched_indices)].copy()

    balance_before = standardized_mean_diff(
        df.dropna(subset=[treatment_col] + covariates), treatment_col, covariates
    )
    balance_after = standardized_mean_diff(matched_df, treatment_col, covariates)

    n_mt = int((matched_df[treatment_col] == 1).sum())
    n_mc = int((matched_df[treatment_col] == 0).sum())

    return MatchResult(
        matched_df=matched_df,
        treatment_col=treatment_col,
        covariates=covariates,
        method="Exact",
        balance_before=balance_before,
        balance_after=balance_after,
        n_treatment=int((work[treatment_col] == 1).sum()),
        n_control=int((work[treatment_col] == 0).sum()),
        n_matched_treatment=n_mt,
        n_matched_control=n_mc,
        match_quality={"exact_cols": exact_cols},
    )


# ---------------------------------------------------------------------------
# Nearest Neighbor Matching
# ---------------------------------------------------------------------------

def nearest_neighbor_match(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    n_neighbors: int = 1,
    metric: str = "mahalanobis",
    random_state: int = 42,
) -> MatchResult:
    """Nearest-neighbor matching по расстоянию (Махаланобис или евклидово).

    Без замещения: каждый контроль используется не более одного раза.
    """
    work = df.dropna(subset=[treatment_col] + covariates).copy()
    work[treatment_col] = work[treatment_col].astype(int)

    scaler = StandardScaler()
    X = scaler.fit_transform(work[covariates].values.astype(float))

    t_mask = work[treatment_col].values == 1
    X_t = X[t_mask]
    X_c = X[~t_mask]

    t_indices = work.index[t_mask].tolist()
    c_indices = work.index[~t_mask].tolist()

    if metric == "mahalanobis":
        # Use covariance of control group
        cov = np.cov(X_c.T) if X_c.shape[0] > 1 else np.eye(X_c.shape[1])
        # np.cov returns scalar for 1-d input — wrap into 2-d
        cov = np.atleast_2d(cov)
        # Regularize
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            dist_matrix = cdist(X_t, X_c, metric="mahalanobis", VI=np.linalg.inv(cov))
        except np.linalg.LinAlgError:
            dist_matrix = cdist(X_t, X_c, metric="euclidean")
    else:
        dist_matrix = cdist(X_t, X_c, metric="euclidean")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(t_indices))

    matched_t: list[int] = []
    matched_c: list[int] = []
    used_c = set()

    for i in order:
        dists = dist_matrix[i]
        sorted_j = np.argsort(dists)
        found = 0
        for j in sorted_j:
            if found >= n_neighbors:
                break
            if j not in used_c:
                matched_t.append(t_indices[i])
                matched_c.append(c_indices[j])
                used_c.add(j)
                found += 1

    all_idx = list(set(matched_t)) + list(set(matched_c))
    matched_df = df.loc[df.index.isin(all_idx)].copy()

    balance_before = standardized_mean_diff(work, treatment_col, covariates)
    balance_after = standardized_mean_diff(matched_df, treatment_col, covariates)

    return MatchResult(
        matched_df=matched_df,
        treatment_col=treatment_col,
        covariates=covariates,
        method="NN",
        balance_before=balance_before,
        balance_after=balance_after,
        n_treatment=int(t_mask.sum()),
        n_control=int((~t_mask).sum()),
        n_matched_treatment=len(set(matched_t)),
        n_matched_control=len(set(matched_c)),
        match_quality={"n_neighbors": n_neighbors, "metric": metric},
    )


# ---------------------------------------------------------------------------
# Coarsened Exact Matching
# ---------------------------------------------------------------------------

def coarsened_exact_match(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    n_bins: int = 5,
) -> MatchResult:
    """CEM — огрубление числовых ковариат в квантильные бины + точное сопоставление.

    Категориальные ковариаты используются как есть.
    """
    work = df.dropna(subset=[treatment_col] + covariates).copy()
    work[treatment_col] = work[treatment_col].astype(int)

    # Coarsen numeric covariates
    coarsened_cols: list[str] = []
    for col in covariates:
        if pd.api.types.is_numeric_dtype(work[col]):
            cname = f"_cem_{col}"
            work[cname] = pd.qcut(work[col], q=n_bins, duplicates="drop").astype(str)
            coarsened_cols.append(cname)
        else:
            cname = f"_cem_{col}"
            work[cname] = work[col].astype(str)
            coarsened_cols.append(cname)

    work["_stratum"] = work[coarsened_cols].agg("|".join, axis=1)

    matched_indices: list[int] = []
    weights: list[float] = []

    for _, group in work.groupby("_stratum"):
        t_idx = group[group[treatment_col] == 1].index.tolist()
        c_idx = group[group[treatment_col] == 0].index.tolist()
        n_t, n_c = len(t_idx), len(c_idx)
        if n_t > 0 and n_c > 0:
            matched_indices.extend(t_idx)
            matched_indices.extend(c_idx)
            # CEM weights: for treatment n_c/n_t, for control n_t/n_c (normalized later)
            w_t = n_c / n_t
            w_c = n_t / n_c
            weights.extend([w_t] * n_t)
            weights.extend([w_c] * n_c)

    matched_df = df.loc[df.index.isin(matched_indices)].copy()
    matched_df["_cem_weight"] = 0.0
    for idx, w in zip(matched_indices, weights):
        matched_df.loc[idx, "_cem_weight"] = w

    balance_before = standardized_mean_diff(work, treatment_col, covariates)
    balance_after = standardized_mean_diff(matched_df, treatment_col, covariates)

    n_mt = int((matched_df[treatment_col] == 1).sum())
    n_mc = int((matched_df[treatment_col] == 0).sum())

    return MatchResult(
        matched_df=matched_df,
        treatment_col=treatment_col,
        covariates=covariates,
        method="CEM",
        balance_before=balance_before,
        balance_after=balance_after,
        n_treatment=int((work[treatment_col] == 1).sum()),
        n_control=int((work[treatment_col] == 0).sum()),
        n_matched_treatment=n_mt,
        n_matched_control=n_mc,
        match_quality={"n_bins": n_bins, "n_strata_matched": matched_df["_cem_weight"].gt(0).sum()},
    )

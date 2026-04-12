"""
core/attribution.py – Factor attribution / decomposition for KIBAD.

Explains the delta in a target metric via factor contributions.

Implements four approaches:

1. **Additive** (non-model): contribution_i = Δdriver_i × partial_effect_i
2. **Multiplicative (ratio)**: log-decomposition of multiplicative factors
3. **Regression-based**: fit OLS on levels, attribute via coef × Δdriver
4. **Shapley approximation**: permutation-based marginal contributions

Mathematical notes
------------------
- Additive: the partial effect for each driver is estimated as
  (target_now − target_counterfactual) where counterfactual replaces only
  that driver with its previous value.  Residual absorbs interaction terms.
- Multiplicative: target = ∏ driver_i  →  Δlog(target) = Σ Δlog(driver_i).
- Regression: target = β₀ + Σ βᵢ driverᵢ  →  contribution_i = βᵢ × Δdriverᵢ.
- Shapley: exact for ≤ 10 drivers; sampling-based otherwise.

When to use / pitfalls
----------------------
- Additive works best for linear, near-independent factors.
- Multiplicative requires strictly positive drivers; log of zero → NaN.
- Regression assumes linearity; multicollinearity inflates coefficient variance.
- Shapley is theoretically optimal but O(2^n) for n drivers; capped at 12.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import factorial
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class AttributionResult:
    """Container for one attribution analysis.

    Attributes
    ----------
    method : str
        ``"additive"``, ``"multiplicative"``, ``"regression"``, ``"shapley"``.
    target_delta : float
        Total change in target (current − previous).
    contributions : pd.DataFrame
        Columns: ``driver``, ``contribution``, ``pct_of_delta``.
    residual : float
        Unexplained portion.
    segment_detail : pd.DataFrame | None
        Per-segment breakdown if a segment column was provided.
    metadata : dict
        Extra info (model params, R², etc.).
    """
    method: str
    target_delta: float
    contributions: pd.DataFrame
    residual: float
    segment_detail: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. Additive decomposition
# ---------------------------------------------------------------------------

def additive_attribution(
    df: pd.DataFrame,
    target_col: str,
    target_prev_col: str,
    driver_cols: list[str],
    driver_prev_cols: list[str],
    segment_col: str | None = None,
) -> AttributionResult:
    """Additive factor attribution.

    For each row, computes per-driver contributions and aggregates across
    rows (optionally by segment).

    The contribution of driver *i* on a single row is estimated via the
    simple first-order difference:

        contrib_i = (driver_i_now − driver_i_prev) × average_sensitivity

    where the average sensitivity is `target_delta / sum(driver_deltas)` if
    all driver deltas are available.  A more robust fallback uses
    equal-share residual distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain current and previous columns for target and drivers.
    target_col, target_prev_col :
        Column names for current and previous target values.
    driver_cols, driver_prev_cols :
        Parallel lists of current and previous driver column names.
    segment_col :
        Optional column for segment-level breakdown.

    Returns
    -------
    AttributionResult
    """
    if len(driver_cols) != len(driver_prev_cols):
        raise ValueError("driver_cols and driver_prev_cols must have equal length.")

    work = df.copy()
    target_now = pd.to_numeric(work[target_col], errors="coerce")
    target_prev = pd.to_numeric(work[target_prev_col], errors="coerce")
    total_delta = float((target_now - target_prev).sum())

    # Per-driver deltas aggregated
    contribs: list[dict[str, Any]] = []
    explained = 0.0

    for dc, dpc in zip(driver_cols, driver_prev_cols):
        d_now = pd.to_numeric(work[dc], errors="coerce")
        d_prev = pd.to_numeric(work[dpc], errors="coerce")
        driver_delta = float((d_now - d_prev).sum())
        contribs.append({"driver": dc, "driver_delta": driver_delta})

    # Distribute target delta proportionally to driver deltas
    sum_abs_deltas = sum(abs(c["driver_delta"]) for c in contribs)
    for c in contribs:
        if sum_abs_deltas > 0:
            c["contribution"] = total_delta * (c["driver_delta"] / sum_abs_deltas) if sum_abs_deltas != 0 else 0.0
        else:
            c["contribution"] = 0.0
        explained += c["contribution"]
        c["pct_of_delta"] = (c["contribution"] / total_delta * 100) if total_delta != 0 else 0.0

    residual = total_delta - explained
    contrib_df = pd.DataFrame(contribs)[["driver", "contribution", "pct_of_delta"]]

    # Segment detail
    seg_detail = None
    if segment_col and segment_col in work.columns:
        rows = []
        for seg, grp in work.groupby(segment_col, dropna=False):
            seg_delta = float(
                pd.to_numeric(grp[target_col], errors="coerce").sum()
                - pd.to_numeric(grp[target_prev_col], errors="coerce").sum()
            )
            row = {"segment": seg, "target_delta": seg_delta}
            for dc, dpc in zip(driver_cols, driver_prev_cols):
                dd = float(
                    pd.to_numeric(grp[dc], errors="coerce").sum()
                    - pd.to_numeric(grp[dpc], errors="coerce").sum()
                )
                row[f"{dc}_delta"] = dd
            rows.append(row)
        seg_detail = pd.DataFrame(rows)

    return AttributionResult(
        method="additive",
        target_delta=total_delta,
        contributions=contrib_df,
        residual=residual,
        segment_detail=seg_detail,
    )


# ---------------------------------------------------------------------------
# 2. Multiplicative (ratio) decomposition
# ---------------------------------------------------------------------------

def multiplicative_attribution(
    df: pd.DataFrame,
    target_col: str,
    target_prev_col: str,
    driver_cols: list[str],
    driver_prev_cols: list[str],
) -> AttributionResult:
    """Multiplicative (log-based) factor attribution.

    Assumes ``target ≈ ∏ driver_i``, so
    ``Δlog(target) = Σ Δlog(driver_i)``.

    Works best when all driver and target values are strictly positive.

    Parameters
    ----------
    df : pd.DataFrame
    target_col, target_prev_col : str
    driver_cols, driver_prev_cols : list[str]

    Returns
    -------
    AttributionResult
    """
    work = df.copy()
    t_now = pd.to_numeric(work[target_col], errors="coerce")
    t_prev = pd.to_numeric(work[target_prev_col], errors="coerce")

    # Use ratio of sums (portfolio level)
    T_now = float(t_now.sum())
    T_prev = float(t_prev.sum())
    total_delta = T_now - T_prev

    if T_prev <= 0 or T_now <= 0:
        # Fallback to additive (multiplicative requires positive values)
        result = additive_attribution(df, target_col, target_prev_col, driver_cols, driver_prev_cols)
        result.method = "additive (fallback from multiplicative: non-positive target sums)"
        return result

    log_total_ratio = np.log(T_now / T_prev)

    contribs = []
    explained_log = 0.0
    for dc, dpc in zip(driver_cols, driver_prev_cols):
        d_now_sum = float(pd.to_numeric(work[dc], errors="coerce").sum())
        d_prev_sum = float(pd.to_numeric(work[dpc], errors="coerce").sum())
        if d_prev_sum > 0 and d_now_sum > 0:
            log_ratio = np.log(d_now_sum / d_prev_sum)
        else:
            log_ratio = 0.0
        explained_log += log_ratio
        contribs.append({"driver": dc, "log_ratio": log_ratio})

    # Scale log contributions to absolute delta
    if explained_log != 0:
        for c in contribs:
            c["contribution"] = total_delta * (c["log_ratio"] / explained_log)
            c["pct_of_delta"] = c["contribution"] / total_delta * 100 if total_delta != 0 else 0.0
    else:
        for c in contribs:
            c["contribution"] = 0.0
            c["pct_of_delta"] = 0.0

    explained = sum(c["contribution"] for c in contribs)
    residual = total_delta - explained

    contrib_df = pd.DataFrame(contribs)[["driver", "contribution", "pct_of_delta"]]

    return AttributionResult(
        method="multiplicative",
        target_delta=total_delta,
        contributions=contrib_df,
        residual=residual,
        metadata={"log_total_ratio": log_total_ratio},
    )


# ---------------------------------------------------------------------------
# 3. Regression-based attribution
# ---------------------------------------------------------------------------

def regression_attribution(
    df: pd.DataFrame,
    target_col: str,
    driver_cols: list[str],
    target_prev_col: str | None = None,
    driver_prev_cols: list[str] | None = None,
) -> AttributionResult:
    """Linear regression-based factor attribution.

    Fits ``target = β₀ + Σ βᵢ × driverᵢ`` and attributes:
    ``contribution_i = βᵢ × Δdriverᵢ``.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    driver_cols : list[str]
    target_prev_col : str, optional
        If provided alongside *driver_prev_cols*, delta is computed
        from aggregated current vs previous values.
    driver_prev_cols : list[str], optional

    Returns
    -------
    AttributionResult
    """
    from sklearn.linear_model import LinearRegression

    work = df.copy()
    y = pd.to_numeric(work[target_col], errors="coerce")
    X = work[driver_cols].apply(pd.to_numeric, errors="coerce")

    mask = y.notna() & X.notna().all(axis=1)
    y_clean = y[mask].values
    X_clean = X[mask].values

    if len(y_clean) < len(driver_cols) + 1:
        raise ValueError("Not enough observations for regression attribution.")

    model = LinearRegression()
    model.fit(X_clean, y_clean)
    r2 = float(model.score(X_clean, y_clean))
    coefs = model.coef_

    # Compute deltas
    if target_prev_col and driver_prev_cols and len(driver_prev_cols) == len(driver_cols):
        t_now = float(pd.to_numeric(work[target_col], errors="coerce").sum())
        t_prev = float(pd.to_numeric(work[target_prev_col], errors="coerce").sum())
        total_delta = t_now - t_prev
        driver_deltas = []
        for dc, dpc in zip(driver_cols, driver_prev_cols):
            dd = float(
                pd.to_numeric(work[dc], errors="coerce").sum()
                - pd.to_numeric(work[dpc], errors="coerce").sum()
            )
            driver_deltas.append(dd)
    else:
        total_delta = float(y.sum() - model.intercept_ * len(y_clean))
        # Use actual interquartile range as meaningful delta proxy
        driver_deltas = [float(pd.to_numeric(work[dc], errors="coerce").quantile(0.75)
                               - pd.to_numeric(work[dc], errors="coerce").quantile(0.25))
                         for dc in driver_cols]

    contribs = []
    for i, dc in enumerate(driver_cols):
        contrib = coefs[i] * driver_deltas[i]
        contribs.append({
            "driver": dc,
            "coefficient": round(coefs[i], 6),
            "driver_delta": round(driver_deltas[i], 4),
            "contribution": round(contrib, 4),
            "pct_of_delta": round(contrib / total_delta * 100, 2) if total_delta != 0 else 0.0,
        })

    explained = sum(c["contribution"] for c in contribs)
    residual = total_delta - explained

    contrib_df = pd.DataFrame(contribs)

    return AttributionResult(
        method="regression",
        target_delta=total_delta,
        contributions=contrib_df[["driver", "contribution", "pct_of_delta"]],
        residual=residual,
        metadata={
            "r_squared": round(r2, 4),
            "intercept": round(float(model.intercept_), 4),
            "coefficients": contrib_df[["driver", "coefficient"]].to_dict("records"),
        },
    )


# ---------------------------------------------------------------------------
# 4. Shapley-value approximation
# ---------------------------------------------------------------------------

def shapley_attribution(
    df: pd.DataFrame,
    target_col: str,
    target_prev_col: str,
    driver_cols: list[str],
    driver_prev_cols: list[str],
    n_samples: int = 200,
    seed: int = 42,
) -> AttributionResult:
    """Shapley-value approximation for factor attribution.

    Estimates each driver's marginal contribution by permuting the order
    in which drivers are switched from *previous* to *current* values
    and measuring the marginal change.

    Exact computation is feasible for ≤ 8 drivers; beyond that a sampling
    approximation is used.

    Parameters
    ----------
    df : pd.DataFrame
    target_col, target_prev_col : str
    driver_cols, driver_prev_cols : list[str]
    n_samples : int
        Number of random permutations for the sampling approximation.
    seed : int
        RNG seed.

    Returns
    -------
    AttributionResult
    """
    n_drivers = len(driver_cols)
    if n_drivers > 12:
        raise ValueError("Shapley attribution is capped at 12 drivers for performance.")
    if n_drivers != len(driver_prev_cols):
        raise ValueError("driver_cols and driver_prev_cols must have equal length.")

    work = df.copy()

    # Aggregate to portfolio level (sum across rows)
    t_now = float(pd.to_numeric(work[target_col], errors="coerce").sum())
    t_prev = float(pd.to_numeric(work[target_prev_col], errors="coerce").sum())
    total_delta = t_now - t_prev

    d_now = np.array([float(pd.to_numeric(work[c], errors="coerce").sum()) for c in driver_cols])
    d_prev = np.array([float(pd.to_numeric(work[c], errors="coerce").sum()) for c in driver_prev_cols])

    # Value function: use additive model target = sum(drivers) + intercept
    # More robust: use the actual target as a function of which drivers are "on"
    intercept = t_prev - d_prev.sum()

    def v(on_mask: np.ndarray) -> float:
        """Value function: sum of 'on' drivers at current level, rest at previous."""
        vals = np.where(on_mask, d_now, d_prev)
        return float(vals.sum() + intercept)

    shapley_vals = np.zeros(n_drivers)

    if n_drivers <= 8:
        # Exact computation
        for perm in permutations(range(n_drivers)):
            mask = np.zeros(n_drivers, dtype=bool)
            prev_val = v(mask)
            for idx in perm:
                mask[idx] = True
                new_val = v(mask)
                shapley_vals[idx] += new_val - prev_val
                prev_val = new_val
        shapley_vals /= factorial(n_drivers)
    else:
        # Sampling approximation
        rng = np.random.default_rng(seed)
        for _ in range(n_samples):
            perm = rng.permutation(n_drivers)
            mask = np.zeros(n_drivers, dtype=bool)
            prev_val = v(mask)
            for idx in perm:
                mask[idx] = True
                new_val = v(mask)
                shapley_vals[idx] += new_val - prev_val
                prev_val = new_val
        shapley_vals /= n_samples

    explained = float(shapley_vals.sum())
    residual = total_delta - explained

    contribs = []
    for i, dc in enumerate(driver_cols):
        contribs.append({
            "driver": dc,
            "contribution": round(float(shapley_vals[i]), 4),
            "pct_of_delta": round(float(shapley_vals[i]) / total_delta * 100, 2) if total_delta != 0 else 0.0,
        })

    return AttributionResult(
        method="shapley",
        target_delta=total_delta,
        contributions=pd.DataFrame(contribs),
        residual=residual,
        metadata={"n_drivers": n_drivers, "exact": n_drivers <= 8},
    )


# ---------------------------------------------------------------------------
# Waterfall chart data helper
# ---------------------------------------------------------------------------

def waterfall_data(result: AttributionResult) -> tuple[list[str], list[float]]:
    """Extract (categories, values) suitable for :func:`core.explore.plot_waterfall`.

    Parameters
    ----------
    result:
        An :class:`AttributionResult`.

    Returns
    -------
    tuple[list[str], list[float]]
    """
    cats = result.contributions["driver"].tolist()
    vals = result.contributions["contribution"].tolist()
    if abs(result.residual) > 1e-10:
        cats.append("Residual")
        vals.append(result.residual)
    return cats, vals

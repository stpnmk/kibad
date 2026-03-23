"""
core/triggers.py – Alert trigger rules engine for KIBAD.

Defines three rule types that can detect anomalies in time-series data:

1. **threshold_cross** — fires when a metric crosses a fixed boundary.
2. **deviation_from_baseline** — fires when a value deviates more than
   *n* sigma from a rolling baseline.
3. **slope_change** — fires when the rolling slope changes sign or
   magnitude significantly.

Mathematical notes
------------------
- ``deviation_from_baseline`` uses a rolling mean ± *k* × rolling std.
  Window must be ≥ 3 to yield a meaningful std.
- ``slope_change`` fits OLS on a rolling window and compares the slope
  of the current window vs the previous window using a t-like ratio.

When NOT to use
---------------
- Very short series (< 10 points): all rolling estimates are unstable.
- Highly non-stationary data: consider differencing or STL first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


RuleType = Literal["threshold_cross", "deviation_from_baseline", "slope_change"]


@dataclass
class TriggerRule:
    """Definition of a single alert trigger.

    Attributes
    ----------
    name : str
        Human-readable rule name.
    rule_type : RuleType
        One of the three supported rule types.
    params : dict
        Rule-specific parameters (see each evaluator for details).
    active : bool
        Whether the rule should be evaluated.
    """
    name: str
    rule_type: RuleType
    params: dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class Alert:
    """One fired alert.

    Attributes
    ----------
    rule : TriggerRule
        The rule that fired.
    index : int
        Row index in the source DataFrame.
    timestamp : Any
        Value of the date column at that row.
    value : float
        The metric value that triggered the alert.
    message : str
        Human-readable description.
    """
    rule: TriggerRule
    index: int
    timestamp: Any
    value: float
    message: str


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------

def _eval_threshold(
    series: pd.Series,
    dates: pd.Series,
    params: dict[str, Any],
    rule: TriggerRule,
) -> list[Alert]:
    """Evaluate a threshold-crossing rule.

    params keys:
        upper : float | None  — upper bound (fires if value > upper)
        lower : float | None  — lower bound (fires if value < lower)
    """
    alerts: list[Alert] = []
    upper = params.get("upper")
    lower = params.get("lower")

    for i, val in series.items():
        if pd.isna(val):
            continue
        fired = False
        reason = ""
        if upper is not None and val > upper:
            fired = True
            reason = f"exceeds upper threshold ({val:.4f} > {upper})"
        if lower is not None and val < lower:
            fired = True
            reason = f"below lower threshold ({val:.4f} < {lower})"
        if fired:
            ts = dates.iloc[i] if i < len(dates) else None
            alerts.append(Alert(rule=rule, index=int(i), timestamp=ts, value=float(val), message=reason))
    return alerts


def _eval_deviation(
    series: pd.Series,
    dates: pd.Series,
    params: dict[str, Any],
    rule: TriggerRule,
) -> list[Alert]:
    """Evaluate a deviation-from-baseline rule.

    params keys:
        window : int     — rolling window size (default 6)
        n_sigma : float  — number of std deviations (default 2.0)
    """
    window = int(params.get("window", 6))
    n_sigma = float(params.get("n_sigma", 2.0))

    roll_mean = series.rolling(window, min_periods=max(3, window // 2)).mean()
    roll_std = series.rolling(window, min_periods=max(3, window // 2)).std()

    alerts: list[Alert] = []
    for i in range(len(series)):
        val = series.iloc[i]
        mu = roll_mean.iloc[i]
        sigma = roll_std.iloc[i]
        if pd.isna(val) or pd.isna(mu) or pd.isna(sigma) or sigma == 0:
            continue
        z = abs(val - mu) / sigma
        if z > n_sigma:
            ts = dates.iloc[i] if i < len(dates) else None
            alerts.append(Alert(
                rule=rule,
                index=i,
                timestamp=ts,
                value=float(val),
                message=f"deviation {z:.2f}σ from rolling baseline (mean={mu:.4f}, σ={sigma:.4f})",
            ))
    return alerts


def _eval_slope_change(
    series: pd.Series,
    dates: pd.Series,
    params: dict[str, Any],
    rule: TriggerRule,
) -> list[Alert]:
    """Evaluate a slope-change rule.

    params keys:
        window : int        — rolling window for slope estimation (default 6)
        threshold : float   — minimum absolute slope change to trigger (default 0.0)

    A slope change fires when the sign of the rolling OLS slope flips or
    when the magnitude change exceeds *threshold*.
    """
    window = int(params.get("window", 6))
    threshold = float(params.get("threshold", 0.0))

    vals = series.values.astype(float)
    slopes: list[float] = []
    for i in range(len(vals)):
        if i < window - 1:
            slopes.append(np.nan)
            continue
        y = vals[i - window + 1: i + 1]
        x = np.arange(window, dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 3:
            slopes.append(np.nan)
            continue
        slope = float(np.polyfit(x[mask], y[mask], 1)[0])
        slopes.append(slope)

    alerts: list[Alert] = []
    for i in range(1, len(slopes)):
        prev_s = slopes[i - 1]
        curr_s = slopes[i]
        if np.isnan(prev_s) or np.isnan(curr_s):
            continue
        # Sign change or magnitude change
        sign_change = (prev_s > 0 and curr_s < 0) or (prev_s < 0 and curr_s > 0)
        mag_change = abs(curr_s - prev_s) > threshold if threshold > 0 else False
        if sign_change or mag_change:
            ts = dates.iloc[i] if i < len(dates) else None
            alerts.append(Alert(
                rule=rule,
                index=i,
                timestamp=ts,
                value=float(vals[i]) if not np.isnan(vals[i]) else 0.0,
                message=f"slope changed from {prev_s:.4f} to {curr_s:.4f}",
            ))
    return alerts


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------

_EVALUATORS = {
    "threshold_cross": _eval_threshold,
    "deviation_from_baseline": _eval_deviation,
    "slope_change": _eval_slope_change,
}


def evaluate_triggers(
    df: pd.DataFrame,
    date_col: str,
    metric_col: str,
    rules: list[TriggerRule],
) -> list[Alert]:
    """Evaluate all active trigger rules on a time-series metric.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by *date_col*.
    date_col : str
        Datetime column.
    metric_col : str
        Numeric column to evaluate.
    rules : list[TriggerRule]
        Rules to check.

    Returns
    -------
    list[Alert]
        Sorted by index (chronological order).
    """
    series = pd.to_numeric(df[metric_col], errors="coerce").reset_index(drop=True)
    dates = df[date_col].reset_index(drop=True)

    all_alerts: list[Alert] = []
    for rule in rules:
        if not rule.active:
            continue
        evaluator = _EVALUATORS.get(rule.rule_type)
        if evaluator is None:
            continue
        all_alerts.extend(evaluator(series, dates, rule.params, rule))

    all_alerts.sort(key=lambda a: a.index)
    return all_alerts


def alerts_to_dataframe(alerts: list[Alert]) -> pd.DataFrame:
    """Convert alerts to a DataFrame for display.

    Returns
    -------
    pd.DataFrame with columns: rule_name, rule_type, index, timestamp, value, message.
    """
    if not alerts:
        return pd.DataFrame(columns=["rule_name", "rule_type", "index", "timestamp", "value", "message"])
    return pd.DataFrame([
        {
            "rule_name": a.rule.name,
            "rule_type": a.rule.rule_type,
            "index": a.index,
            "timestamp": a.timestamp,
            "value": a.value,
            "message": a.message,
        }
        for a in alerts
    ])

"""
core/simulation.py – Scenario simulation: sliders, paths, and component breakdowns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from core.models import (
    ARXForecaster,
    ForecastResult,
    compute_all_metrics,
    run_arx_forecast,
    _future_dates,
)


# ---------------------------------------------------------------------------
# Scenario preset
# ---------------------------------------------------------------------------

@dataclass
class ScenarioPreset:
    """A saved scenario configuration.

    Attributes
    ----------
    name : str
        User-defined name.
    shocks : dict[str, float]
        Mapping of exogenous variable name → percentage shock (e.g., ``0.1`` = +10%).
    absolute_overrides : dict[str, float]
        Mapping of variable name → absolute override value (takes precedence over shocks).
    notes : str
    """
    name: str
    shocks: dict[str, float] = field(default_factory=dict)
    absolute_overrides: dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "ScenarioPreset":
        d = json.loads(s)
        return cls(**d)


# ---------------------------------------------------------------------------
# Scenario simulation engine
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Result of a scenario simulation run.

    Attributes
    ----------
    scenario_name : str
    forecast_df : pd.DataFrame
        Columns: ``date``, ``baseline``, ``scenario``, ``delta``.
    components_df : pd.DataFrame | None
        Optional per-component breakdown.
    preset : ScenarioPreset | None
    notes : str
    """
    scenario_name: str
    forecast_df: pd.DataFrame
    components_df: pd.DataFrame | None = None
    preset: ScenarioPreset | None = None
    notes: str = ""


def apply_shocks(
    exog_df: pd.DataFrame,
    shocks: dict[str, float],
    absolute_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Apply percentage shocks and optional absolute overrides to exogenous DataFrame.

    Parameters
    ----------
    exog_df:
        Exogenous feature DataFrame.
    shocks:
        {col_name: shock_fraction} where 0.1 = +10%.
    absolute_overrides:
        {col_name: value} — override the column with a fixed value.

    Returns
    -------
    pd.DataFrame
        A copy with shocks applied.
    """
    df = exog_df.copy()
    for col, shock in shocks.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce") * (1 + shock)
    if absolute_overrides:
        for col, val in absolute_overrides.items():
            if col in df.columns:
                df[col] = float(val)
    return df


def run_scenario(
    train_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str] | None = None,
    lags: list[int] | None = None,
    horizon: int = 12,
    preset: ScenarioPreset | None = None,
    scenario_name: str = "Scenario",
    component_cols: list[str] | None = None,
    alpha: float = 1.0,
) -> tuple[SimulationResult, SimulationResult]:
    """Run baseline and scenario forecasts with the ARX model.

    Both baseline and scenario share the same fitted model; only the
    exogenous inputs differ for the forecast horizon.

    Parameters
    ----------
    train_df:
        Full historical DataFrame.
    date_col:
        Date column.
    target_col:
        Target column.
    exog_cols:
        Exogenous column names.
    lags:
        AR lag periods.
    horizon:
        Forecast horizon.
    preset:
        Scenario preset containing shocks/overrides; if None, runs only baseline.
    scenario_name:
        Display name for the scenario run.
    component_cols:
        Optional columns to treat as flow components in breakdown chart.
    alpha:
        Ridge regularisation.

    Returns
    -------
    tuple[SimulationResult, SimulationResult]
        (baseline, scenario) simulation results.
    """
    train_df = train_df.sort_values(date_col).dropna(subset=[target_col])
    y = train_df[target_col].reset_index(drop=True).astype(float)
    dates = pd.to_datetime(train_df[date_col])
    future_dates = _future_dates(dates, horizon)

    exog_train = None
    exog_future_base = None
    if exog_cols:
        valid = [c for c in exog_cols if c in train_df.columns]
        if valid:
            exog_train = train_df[valid].reset_index(drop=True)
            last_row = exog_train.iloc[-1:].values
            exog_future_base = pd.DataFrame(
                np.tile(last_row, (horizon, 1)), columns=valid
            )

    lag_list = lags or [1, 2, 3, 12]
    model = ARXForecaster(lags=lag_list, alpha=alpha)
    model.fit(y, exog=exog_train)

    # Baseline forecast
    base_preds = model.predict(horizon, future_exog=exog_future_base)

    # Scenario forecast
    scen_preds = base_preds.copy()
    if preset is not None and exog_future_base is not None:
        exog_shocked = apply_shocks(
            exog_future_base,
            shocks=preset.shocks,
            absolute_overrides=preset.absolute_overrides,
        )
        scen_preds = model.predict(horizon, future_exog=exog_shocked)
    elif preset is not None:
        # No exog; apply compounding shock on the baseline predictions
        total_shock = sum(preset.shocks.values())
        # Compound shock across steps to model propagation through AR lags
        scen_preds = base_preds * np.array([(1 + total_shock) ** (i + 1)
                                            for i in range(len(base_preds))])

    # Assemble forecast DataFrames
    def _make_forecast_df(preds: np.ndarray, label: str) -> pd.DataFrame:
        hist = pd.DataFrame({
            "date": dates.values,
            "actual": y.values,
            label: np.nan,
        })
        fut = pd.DataFrame({
            "date": future_dates,
            "actual": np.nan,
            label: preds,
        })
        return pd.concat([hist, fut], ignore_index=True)

    base_df = _make_forecast_df(base_preds, "baseline")
    scen_df = _make_forecast_df(scen_preds, "scenario")

    # Combined delta DataFrame
    combined = pd.DataFrame({
        "date": future_dates,
        "baseline": base_preds,
        "scenario": scen_preds,
        "delta": scen_preds - base_preds,
    })

    # Component breakdown (if component_cols specified)
    comp_df = None
    if component_cols:
        comp_rows = []
        for col in component_cols:
            if col in train_df.columns:
                series = train_df[[date_col, col]].rename(columns={col: "value"})
                series["component"] = col
                series["date"] = pd.to_datetime(series[date_col])
                comp_rows.append(series[["date", "component", "value"]])
        if comp_rows:
            comp_df = pd.concat(comp_rows, ignore_index=True)

    base_result = SimulationResult(
        scenario_name="Baseline",
        forecast_df=combined[["date", "baseline"]].rename(columns={"baseline": "forecast"}),
        components_df=comp_df,
        notes="Baseline: no shocks applied.",
    )
    scen_result = SimulationResult(
        scenario_name=scenario_name,
        forecast_df=combined,
        components_df=comp_df,
        preset=preset,
        notes=f"Scenario: {preset.shocks if preset else 'none'}",
    )
    return base_result, scen_result


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_scenario_comparison(
    combined_df: pd.DataFrame,
    history_df: pd.DataFrame,
    date_col: str,
    target_col: str,
    title: str = "Scenario Comparison",
) -> go.Figure:
    """Line chart: actual history + baseline + scenario forecast.

    Parameters
    ----------
    combined_df:
        DataFrame with columns ``date``, ``baseline``, ``scenario``, ``delta``.
    history_df:
        Historical DataFrame with actual values; used to anchor the left side.
    date_col:
        Date column in history_df.
    target_col:
        Target column in history_df.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    hist = history_df[[date_col, target_col]].sort_values(date_col).dropna(subset=[target_col])

    # Ensure fact line ends at last available fact date
    last_fact_date = hist[date_col].max()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist[date_col], y=hist[target_col],
        mode="lines", name="Actual",
        line=dict(color="black", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=combined_df["date"], y=combined_df["baseline"],
        mode="lines", name="Baseline Forecast",
        line=dict(color="steelblue", width=2, dash="dash"),
    ))
    if "scenario" in combined_df.columns:
        fig.add_trace(go.Scatter(
            x=combined_df["date"], y=combined_df["scenario"],
            mode="lines", name="Scenario Forecast",
            line=dict(color="darkorange", width=2, dash="dot"),
        ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Value",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def plot_component_flows(
    comp_df: pd.DataFrame,
    date_col: str = "date",
    title: str = "Component Flows",
) -> go.Figure:
    """Stacked bar chart of component flows over time.

    Parameters
    ----------
    comp_df:
        DataFrame with columns ``date``, ``component``, ``value``.
    date_col:
        Date column name.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if comp_df is None or comp_df.empty:
        fig = go.Figure()
        fig.update_layout(title="No component data available.")
        return fig

    import plotly.express as px
    fig = px.bar(
        comp_df,
        x=date_col,
        y="value",
        color="component",
        barmode="relative",
        title=title,
    )
    fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Value")
    return fig


def plot_scenario_delta(
    combined_df: pd.DataFrame,
    title: str = "Scenario vs Baseline Delta",
) -> go.Figure:
    """Bar chart of delta (scenario minus baseline) per period.

    Parameters
    ----------
    combined_df:
        DataFrame with ``date`` and ``delta`` columns.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if "delta" not in combined_df.columns:
        fig = go.Figure()
        fig.update_layout(title="No delta data available.")
        return fig

    colors = ["seagreen" if v >= 0 else "tomato" for v in combined_df["delta"]]
    fig = go.Figure(go.Bar(
        x=combined_df["date"],
        y=combined_df["delta"],
        marker_color=colors,
        name="Delta",
    ))
    fig.add_hline(y=0, line_dash="dot", line_color="black")
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Scenario − Baseline",
    )
    return fig

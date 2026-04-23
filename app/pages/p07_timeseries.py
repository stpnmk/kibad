"""p07_timeseries — Временные ряды (Dash).

Redesigned for KIBAD Design System v2026.04 (dark eucalyptus).
Mirrors handoff slide 7 ("Временные ряды") — 9 artboards rendered as
sub-tabs of a single page:
  7.1 Наивный        — baseline forecast
  7.2 ARX            — Ridge AR + exogenous, coefficient bars
  7.3 SARIMAX        — seasonal ARIMA with params table
  7.4 Бэктестинг     — rolling-window folds + MAE chart
  7.5 ACF/PACF       — stems with ±1.96/√N CI bounds
  7.6 Декомпозиция   — STL 4-panel (obs / trend / season / resid)
  7.7 Диагностика    — residual ACF / Q-Q / time / vs-fit + Ljung-Box
  7.8 Сравнение      — best-highlighted table + overlay chart + verdict
  7.9 Аномалии       — rolling-z / STL residual + ±σ band + X-markers
"""
from __future__ import annotations

from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_PREPARED, get_df_from_store, list_datasets,
)
from app.components.alerts import alert_banner
from app.components.icons import icon

from core.models import (
    run_naive_forecast, run_arx_forecast, run_sarimax_forecast,
    rolling_backtest, detect_anomalies, ForecastResult,
)

dash.register_page(
    __name__,
    path="/timeseries",
    name="7. Временные ряды",
    order=7,
    icon="graph-up",
)


# ---------------------------------------------------------------------------
# Design tokens — time-series role palette (matches ts-shared.js TS dict)
# ---------------------------------------------------------------------------

TS_ACTUALS  = "#A3B0A8"                     # historical values
TS_FIT      = "#4A7FB0"                     # in-sample fit (blue)
TS_FORECAST = "#C8503B"                     # forecast (terracotta)
TS_CI       = "rgba(200,80,59,0.18)"        # CI ribbon
TS_TODAY    = "#6B7A72"                     # "today" reference line
TS_NAIVE    = "#C98A2E"                     # naive model (amber)
TS_ARX      = "#4A7FB0"                     # arx model (blue)
TS_SARIMAX  = "#A066C8"                     # sarimax model (violet)

SURFACE_0   = "#0F1613"
SURFACE_1   = "#141C18"
ACCENT_500  = "#21A066"
ACCENT_300  = "#4FD18B"
DANGER      = "#C8503B"
WARNING     = "#C98A2E"
INFO        = "#4A7FB0"
TEXT_PRI    = "#E8EFEA"
TEXT_SEC    = "#A3B0A8"
TEXT_TER    = "#6B7A72"
GRID        = "rgba(255,255,255,0.06)"
GRID_STRONG = "rgba(255,255,255,0.14)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_df(ds_name, raw, prep):
    if not ds_name:
        return None
    if prep and ds_name in (prep or {}):
        df = get_df_from_store(prep, ds_name)
    else:
        df = get_df_from_store(raw, ds_name)
    return df


def _fmt_num(v: Any, digits: int = 4) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        return f"{f:.{digits}f}"
    except Exception:
        return "—"


def _fmt_pct(v: Any, digits: int = 2) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        return f"{f:.{digits}f} %"
    except Exception:
        return "—"


def _fmt_signed(v: Any, digits: int = 4) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        return f"{f:+.{digits}f}"
    except Exception:
        return "—"


def _base_layout(height: int = 320) -> dict:
    """Common Plotly layout hash — matches KIBAD dark eucalyptus surface."""
    return dict(
        paper_bgcolor=SURFACE_0,
        plot_bgcolor=SURFACE_0,
        font=dict(family="Inter, -apple-system, sans-serif",
                  color=TEXT_SEC, size=12),
        margin=dict(l=56, r=20, t=28, b=40),
        height=height,
        hovermode="x unified",
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=TEXT_TER, family="JetBrains Mono", size=10)),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID,
                   tickfont=dict(color=TEXT_TER, family="JetBrains Mono", size=10)),
        legend=dict(orientation="h", y=-0.18, x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_SEC, size=11)),
        hoverlabel=dict(bgcolor=SURFACE_1, font_color=TEXT_PRI,
                        bordercolor=GRID_STRONG),
    )


# ---------------------------------------------------------------------------
# Reusable figures
# ---------------------------------------------------------------------------

def _plot_forecast_v2(
    result: ForecastResult,
    target_col: str,
    show_fit: bool = False,
    model_color: str = TS_FORECAST,
    height: int = 320,
) -> go.Figure:
    """Forecast chart — actuals + optional fit (dashed) + forecast+markers + CI."""
    fd = result.forecast_df
    hist = fd[fd["actual"].notna()].copy()
    fut = fd[fd["actual"].isna() & fd["forecast"].notna()].copy()

    fig = go.Figure()

    # 1. CI ribbon (behind lines)
    if not fut.empty and "lower" in fut.columns and fut["lower"].notna().any():
        xs = pd.concat([fut["date"], fut["date"][::-1]])
        ys = pd.concat([fut["upper"], fut["lower"][::-1]])
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            fill="toself", fillcolor=TS_CI,
            line=dict(color="rgba(0,0,0,0)"),
            name="CI 95%", hoverinfo="skip", showlegend=True,
        ))

    # 2. Actuals line
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["actual"],
        mode="lines", name="actuals",
        line=dict(color=TS_ACTUALS, width=1.8),
    ))

    # 3. Optional in-sample fit (dashed)
    if show_fit and not hist["forecast"].dropna().empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["forecast"],
            mode="lines", name="fit",
            line=dict(color=TS_FIT, width=1.6, dash="dash"),
            opacity=0.85,
        ))

    # 4. Forecast line — bridge from last actual to future
    if not fut.empty:
        if not hist.empty:
            bridge_x = pd.concat([hist["date"].iloc[[-1]], fut["date"]])
            bridge_y = pd.concat([hist["actual"].iloc[[-1]], fut["forecast"]])
        else:
            bridge_x = fut["date"]
            bridge_y = fut["forecast"]
        fig.add_trace(go.Scatter(
            x=bridge_x, y=bridge_y,
            mode="lines", name="forecast",
            line=dict(color=model_color, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["forecast"],
            mode="markers", name="forecast",
            marker=dict(color=model_color, size=6),
            showlegend=False,
        ))

    # 5. "Today" vertical line at forecast boundary
    if not hist.empty and not fut.empty:
        fig.add_vline(
            x=hist["date"].iloc[-1],
            line_dash="dash", line_color=TS_TODAY, line_width=1,
            opacity=0.7,
            annotation_text="СЕГОДНЯ", annotation_position="top",
            annotation_font=dict(size=10, color=TEXT_TER,
                                 family="JetBrains Mono"),
            annotation_bgcolor=SURFACE_1,
        )

    lay = _base_layout(height)
    lay.update(dict(yaxis_title=target_col))
    fig.update_layout(**lay)
    return fig


def _coef_chart(coef_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart of ARX coefficients — green pos / red neg, sorted |v|."""
    if coef_df is None or coef_df.empty:
        return go.Figure()

    df = coef_df.copy()
    # Normalise column names
    if "feature" not in df.columns and "name" in df.columns:
        df = df.rename(columns={"name": "feature"})
    if "coefficient" not in df.columns:
        for c in ("coef", "weight", "value"):
            if c in df.columns:
                df = df.rename(columns={c: "coefficient"})
                break
    if "feature" not in df.columns or "coefficient" not in df.columns:
        return go.Figure()

    df["_abs"] = df["coefficient"].abs()
    df = df.sort_values("_abs", ascending=False).head(top_n)
    df = df.sort_values("coefficient", ascending=True)  # bottom-up for horizontal

    colors = [ACCENT_500 if v >= 0 else DANGER for v in df["coefficient"]]
    text_colors = [ACCENT_300 if v >= 0 else "#E07563" for v in df["coefficient"]]
    labels = [f"{'+' if v >= 0 else ''}{v:.3f}" for v in df["coefficient"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["feature"], x=df["coefficient"],
        orientation="h", marker_color=colors,
        text=labels, textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=TEXT_PRI),
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        showlegend=False,
    ))
    lay = _base_layout(height=32 * max(len(df), 4) + 80)
    lay.update(dict(
        xaxis=dict(**lay["xaxis"], zeroline=True,
                   zerolinecolor=GRID_STRONG, zerolinewidth=1),
        yaxis=dict(**lay["yaxis"], automargin=True),
        margin=dict(l=150, r=60, t=20, b=40),
    ))
    fig.update_layout(**lay)
    return fig


def _acf_stems(values: np.ndarray, n: int, color: str,
               title: str = "", height: int = 200) -> go.Figure:
    """ACF/PACF stem plot with ±1.96/√N dashed CI bounds."""
    ci = 1.96 / np.sqrt(max(n, 1))
    lags = np.arange(len(values))
    fig = go.Figure()
    # Stems (one vertical line per lag)
    for lag, v in zip(lags, values):
        op = 1.0 if (abs(v) > ci or lag == 0) else 0.35
        fig.add_trace(go.Scatter(
            x=[lag, lag], y=[0, v],
            mode="lines",
            line=dict(color=color, width=2),
            opacity=op, hoverinfo="skip", showlegend=False,
        ))
    op_list = [(1.0 if (abs(v) > ci or lag == 0) else 0.35)
               for lag, v in zip(lags, values)]
    fig.add_trace(go.Scatter(
        x=lags, y=values, mode="markers",
        marker=dict(color=color, size=6, opacity=op_list),
        hovertemplate="lag %{x}: %{y:.3f}<extra></extra>",
        showlegend=False,
    ))
    # ±CI dashed bounds
    fig.add_hline(y=ci, line_dash="dash", line_color=DANGER,
                  line_width=1, opacity=0.7,
                  annotation_text=f"+1.96/√N = {ci:.3f}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#E07563",
                                       family="JetBrains Mono"),
                  annotation_bgcolor=SURFACE_0)
    fig.add_hline(y=-ci, line_dash="dash", line_color=DANGER,
                  line_width=1, opacity=0.7)
    fig.add_hline(y=0, line_color=GRID_STRONG, line_width=1)

    lay = _base_layout(height)
    lay.update(dict(
        title=dict(text=title, font=dict(color=TEXT_PRI, size=13)) if title else None,
        yaxis=dict(**lay["yaxis"], range=[-1.05, 1.05]),
        xaxis=dict(**lay["xaxis"], title="лаг"),
        margin=dict(l=48, r=16, t=10, b=32),
    ))
    fig.update_layout(**lay)
    return fig


def _stl_panel(x, y, color: str, height: int = 110,
               show_zero: bool = False) -> go.Figure:
    """Single STL panel — obs / trend / season / resid."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color=color, width=2 if color != TS_FIT else 2.3),
        hovertemplate="%{x}: %{y:.2f}<extra></extra>", showlegend=False,
    ))
    if show_zero:
        fig.add_hline(y=0, line_dash="dash", line_color=TEXT_TER,
                      line_width=1, opacity=0.7)
    lay = _base_layout(height)
    lay.update(dict(
        margin=dict(l=48, r=16, t=4, b=16),
        xaxis=dict(**lay["xaxis"], showticklabels=False),
        yaxis=dict(**lay["yaxis"], nticks=3),
        hovermode="closest",
    ))
    fig.update_layout(**lay)
    return fig


def _resid_time_chart(residuals: np.ndarray, height: int = 200) -> go.Figure:
    """Residuals-vs-time scatter line."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.arange(len(residuals)), y=residuals,
        mode="lines", line=dict(color=WARNING, width=1.5),
        showlegend=False,
        hovertemplate="t=%{x}: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_TER,
                  line_width=1, opacity=0.6)
    lay = _base_layout(height)
    lay.update(dict(margin=dict(l=40, r=10, t=10, b=24),
                    hovermode="closest"))
    fig.update_layout(**lay)
    return fig


def _qq_chart(residuals: np.ndarray, height: int = 200) -> go.Figure:
    """Q-Q plot against standard normal quantiles."""
    from scipy import stats  # statsmodels already depends on scipy
    fig = go.Figure()
    if len(residuals) < 2:
        lay = _base_layout(height)
        fig.update_layout(**lay)
        return fig
    osm, osr = stats.probplot(residuals, dist="norm", fit=False)
    slope, intercept, _, _, _ = stats.linregress(osm, osr)
    xr = np.array([min(osm), max(osm)])
    yr = slope * xr + intercept
    fig.add_trace(go.Scatter(
        x=xr, y=yr, mode="lines",
        line=dict(color=ACCENT_500, width=1.5, dash="dash"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=osm, y=osr, mode="markers",
        marker=dict(color=TS_FIT, size=5, opacity=0.8),
        showlegend=False,
        hovertemplate="q=%{x:.2f}: %{y:.2f}<extra></extra>",
    ))
    lay = _base_layout(height)
    lay.update(dict(margin=dict(l=40, r=10, t=10, b=24),
                    xaxis=dict(**lay["xaxis"], title="theoretical"),
                    yaxis=dict(**lay["yaxis"], title="sample"),
                    hovermode="closest"))
    fig.update_layout(**lay)
    return fig


def _resid_fit_chart(fitted: np.ndarray, residuals: np.ndarray,
                     height: int = 200) -> go.Figure:
    """Residuals-vs-fitted scatter with LOWESS-like smoother."""
    fig = go.Figure()
    fig.add_hline(y=0, line_dash="dash", line_color=TEXT_TER,
                  line_width=1, opacity=0.6)
    fig.add_trace(go.Scatter(
        x=fitted, y=residuals, mode="markers",
        marker=dict(color=TS_FIT, size=5, opacity=0.65),
        showlegend=False,
        hovertemplate="fit=%{x:.2f}: resid=%{y:.2f}<extra></extra>",
    ))
    # Simple rolling-mean smoother
    try:
        order = np.argsort(fitted)
        xs = np.asarray(fitted)[order]
        ys = np.asarray(residuals)[order]
        window = max(len(xs) // 8, 3)
        smooth = pd.Series(ys).rolling(window, center=True, min_periods=1).mean().values
        fig.add_trace(go.Scatter(
            x=xs, y=smooth, mode="lines",
            line=dict(color=ACCENT_500, width=2),
            showlegend=False, hoverinfo="skip",
        ))
    except Exception:
        pass
    lay = _base_layout(height)
    lay.update(dict(margin=dict(l=40, r=10, t=10, b=24),
                    xaxis=dict(**lay["xaxis"], title="fitted"),
                    yaxis=dict(**lay["yaxis"], title="residual"),
                    hovermode="closest"))
    fig.update_layout(**lay)
    return fig


# ---------------------------------------------------------------------------
# UI partials — page head, controls, tabs, cards
# ---------------------------------------------------------------------------

def _page_head() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("KIBAD · АНАЛИТИЧЕСКАЯ СТУДИЯ",
                             className="kb-overline"),
                    html.H1("Раздел 7 · Временные ряды",
                            className="kb-page-title"),
                    html.Div(
                        "Прогноз (Naive / ARX / SARIMAX), декомпозиция STL, "
                        "ACF/PACF, бэктест, аномалии.",
                        className="kb-page-subtitle",
                    ),
                ],
                className="kb-page-head-left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("download", 14), html.Span("Экспорт отчёта")],
                        id="ts-export-btn",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("play", 14), html.Span("Запустить все")],
                        id="ts-run-all-btn",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-page-head-actions",
            ),
        ],
        className="kb-page-head",
    )


def _help_accordion() -> dbc.Accordion:
    methods = [
        ("rewind",
         "Наивный / Сезонный наивный",
         "Последнее значение или значение прошлого сезона. Базовая линия, "
         "горизонт 1–3 периода."),
        ("trend",
         "ARX (Ridge + лаги)",
         "Авторегрессия + внешние факторы, Ridge-регуляризация. "
         "Лучше при экзогенных факторах, горизонт 3–24."),
        ("chart",
         "SARIMAX",
         "Сезонная ARIMA с доверительными интервалами. Нужна чёткая "
         "сезонность, ≥ 50 наблюдений."),
    ]
    cards = []
    for icn, title, body in methods:
        cards.append(html.Div(
            [
                html.Div(
                    icon(icn, 16),
                    className="kb-ts-method-card__icn",
                ),
                html.Div(
                    [
                        html.Div("МОДЕЛЬ", className="kb-overline"),
                        html.H4(title, className="kb-ts-method-card__title"),
                        html.P(body, className="kb-ts-method-card__body"),
                    ],
                    className="kb-ts-method-card__text",
                ),
            ],
            className="kb-ts-method-card",
        ))

    return dbc.Accordion(
        [
            dbc.AccordionItem(
                html.Div(cards, className="kb-ts-method-grid"),
                title=html.Span(
                    [icon("help", 14, className="kb-ts-help-icn"),
                     html.Span("Справка по методам",
                               className="kb-ts-help-title")],
                    className="kb-ts-help-label",
                ),
                item_id="help",
            )
        ],
        id="ts-help-acc",
        start_collapsed=True,
        flush=False,
        className="kb-ts-help-acc",
    )


def _controls() -> html.Div:
    """Control rows — dataset picker + hyperparams (rendered above tabs)."""
    return html.Div(
        [
            # Row 1: dataset / date-col / target / exog
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("ДАТАСЕТ", className="kb-ts-ctrl-label"),
                            dcc.Dropdown(
                                id="ts-ds-select", options=[], value=None,
                                clearable=False,
                                placeholder="Выберите датасет…",
                                className="kb-select kb-ts-select",
                            ),
                        ],
                        className="kb-ts-ctrl-field",
                    ),
                    html.Div(
                        [
                            html.Label("КОЛОНКА ДАТЫ", className="kb-ts-ctrl-label"),
                            dcc.Dropdown(
                                id="ts-date-col", options=[], value=None,
                                clearable=False,
                                placeholder="—",
                                className="kb-select kb-ts-select",
                            ),
                        ],
                        className="kb-ts-ctrl-field",
                    ),
                    html.Div(
                        [
                            html.Label("ЦЕЛЕВАЯ ПЕРЕМЕННАЯ",
                                       className="kb-ts-ctrl-label"),
                            dcc.Dropdown(
                                id="ts-target-col", options=[], value=None,
                                clearable=False,
                                placeholder="—",
                                className="kb-select kb-ts-select",
                            ),
                        ],
                        className="kb-ts-ctrl-field",
                    ),
                    html.Div(
                        [
                            html.Label(
                                id="ts-exog-label",
                                children="ВНЕШНИЕ ФАКТОРЫ",
                                className="kb-ts-ctrl-label",
                            ),
                            dcc.Dropdown(
                                id="ts-exog-cols", options=[], value=[],
                                multi=True, placeholder="добавить…",
                                className="kb-select kb-ts-select kb-ts-select--chips",
                            ),
                        ],
                        className="kb-ts-ctrl-field",
                    ),
                ],
                className="kb-ts-ctrl-row kb-ts-ctrl-row--1",
            ),
            # Row 2: horizon / period / lags / ridge-alpha
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("ГОРИЗОНТ ПРОГНОЗА",
                                               className="kb-ts-ctrl-label"),
                                    html.Span(id="ts-horizon-val",
                                              className="kb-ts-slider-val",
                                              children="12 мес"),
                                ],
                                className="kb-ts-slider-hdr",
                            ),
                            dcc.Slider(id="ts-horizon", min=1, max=60,
                                       value=12, step=1, marks=None,
                                       tooltip={"always_visible": False,
                                                "placement": "bottom"},
                                       className="kb-ts-slider"),
                            html.Div([html.Span("1"), html.Span("60")],
                                     className="kb-ts-slider-range"),
                        ],
                        className="kb-ts-slider-field",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("СЕЗОННЫЙ ПЕРИОД",
                                               className="kb-ts-ctrl-label"),
                                    html.Span(id="ts-period-val",
                                              className="kb-ts-slider-val",
                                              children="12"),
                                ],
                                className="kb-ts-slider-hdr",
                            ),
                            dcc.Slider(id="ts-period", min=1, max=52,
                                       value=12, step=1, marks=None,
                                       tooltip={"always_visible": False,
                                                "placement": "bottom"},
                                       className="kb-ts-slider"),
                            html.Div([html.Span("1"), html.Span("52")],
                                     className="kb-ts-slider-range"),
                        ],
                        className="kb-ts-slider-field",
                    ),
                    html.Div(
                        [
                            html.Label("ЛАГИ AR", className="kb-ts-ctrl-label"),
                            dcc.Input(id="ts-lags", value="1,2,3,12",
                                      type="text", className="kb-input kb-input--mono"),
                            html.Div([html.Span("через запятую"),
                                      html.Span("напр. 1,2,3,12")],
                                     className="kb-ts-slider-range"),
                        ],
                        className="kb-ts-slider-field",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("RIDGE α",
                                               className="kb-ts-ctrl-label"),
                                    html.Span(id="ts-alpha-val",
                                              className="kb-ts-slider-val",
                                              children="10"),
                                ],
                                className="kb-ts-slider-hdr",
                            ),
                            dcc.Slider(id="ts-alpha", min=1, max=100,
                                       value=10, step=1, marks=None,
                                       tooltip={"always_visible": False,
                                                "placement": "bottom"},
                                       className="kb-ts-slider"),
                            html.Div([html.Span("1"), html.Span("100")],
                                     className="kb-ts-slider-range"),
                        ],
                        className="kb-ts-slider-field",
                    ),
                ],
                className="kb-ts-ctrl-row kb-ts-ctrl-row--2",
            ),
        ],
        className="kb-ts-controls",
    )


def _kpi_tile(label: str, value: str, sub: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kb-ts-kpi__label"),
            html.Div(value, className="kb-ts-kpi__value"),
            html.Div(sub, className="kb-ts-kpi__sub") if sub else html.Div(),
        ],
        className="kb-ts-kpi",
    )


def _kpi_row(items: list[tuple]) -> html.Div:
    """items: list of (label, value, sub) — renders a 4/5-col grid."""
    cls = "kb-ts-kpis"
    if len(items) == 5:
        cls += " kb-ts-kpis--5"
    return html.Div(
        [_kpi_tile(*i) if len(i) == 3 else _kpi_tile(i[0], i[1])
         for i in items],
        className=cls,
    )


def _chip(text: str, variant: str = "neutral") -> html.Span:
    return html.Span(text, className=f"kb-chip kb-chip--{variant}")


def _model_chip(label: str, color: str,
                variant: str = "neutral") -> html.Span:
    return html.Span(
        [
            html.Span(className="kb-ts-cdot",
                      style={"background": color}),
            html.Span(label),
        ],
        className=f"kb-chip kb-chip--{variant} kb-ts-model-chip",
    )


def _legend_row(items: list[tuple]) -> html.Div:
    """items: list of (label, color, kind) — kind in {solid, dash, band}."""
    tiles = []
    for it in items:
        label, color = it[0], it[1]
        kind = it[2] if len(it) > 2 else "solid"
        sw_cls = f"kb-ts-leg-sw kb-ts-leg-sw--{kind}"
        style = {"background": color}
        if kind == "dash":
            style = {"borderColor": color, "background": "transparent"}
        tiles.append(html.Span(
            [html.Span(className=sw_cls, style=style),
             html.Span(label)],
            className="kb-ts-leg-tile",
        ))
    return html.Div(tiles, className="kb-ts-legend")


def _tab_head(overline: str, subtitle: str, button_children) -> html.Div:
    return html.Div(
        [
            html.Div(
                [html.Div(overline, className="kb-overline"),
                 html.P(subtitle, className="kb-ts-tab-sub")],
                className="kb-ts-tab-head-left",
            ),
            html.Div(button_children, className="kb-ts-tab-head-actions"),
        ],
        className="kb-ts-tab-head",
    )


def _card(children, pad: bool = True) -> html.Div:
    return html.Div(children,
                    className="kb-ts-card" + (" kb-ts-card--pad" if pad else ""))


def _card_head(title: str, right=None) -> html.Div:
    return html.Div(
        [
            html.H3(title, className="kb-ts-card__title"),
            html.Div(right) if right is not None else html.Div(),
        ],
        className="kb-ts-card__head",
    )


def _chart_frame(fig: go.Figure, height: int = 320) -> html.Div:
    return html.Div(
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"height": f"{height}px"}),
        className="kb-ts-chart-frame",
    )


def _fmt_df_tbl(df: pd.DataFrame, max_rows: int = 10) -> html.Table:
    """Compact HTML table with JetBrains Mono numeric cells."""
    df = df.head(max_rows)
    head = html.Thead(html.Tr([
        html.Th(str(c), className="kb-ts-tbl__th kb-mono"
                if pd.api.types.is_numeric_dtype(df[c]) else "kb-ts-tbl__th")
        for c in df.columns
    ]))
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if pd.api.types.is_numeric_dtype(df[c]):
                txt = _fmt_num(v)
                cls = "kb-mono"
            else:
                txt = "—" if pd.isna(v) else str(v)
                cls = ""
            cells.append(html.Td(txt, className=f"kb-ts-tbl__td {cls}".strip()))
        rows.append(html.Tr(cells))
    return html.Table([head, html.Tbody(rows)], className="kb-ts-tbl")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_TABS = [
    ("tab-naive",   "Наивный"),
    ("tab-arx",     "ARX"),
    ("tab-sarimax", "SARIMAX"),
    ("tab-bt",      "Бэктестинг"),
    ("tab-acf",     "ACF/PACF"),
    ("tab-stl",     "Декомпозиция"),
    ("tab-diag",    "Диагностика"),
    ("tab-compare", "Сравнение"),
    ("tab-anom",    "Аномалии"),
]


layout = html.Div(
    [
        _page_head(),
        _help_accordion(),
        _controls(),
        html.Div(
            dbc.Tabs(
                id="ts-tabs",
                active_tab="tab-naive",
                children=[dbc.Tab(label=lbl, tab_id=tid)
                          for tid, lbl in _TABS],
                className="kb-ts-tabs",
            ),
            className="kb-ts-tabs-wrap",
        ),
        html.Div(id="ts-tab-content", className="kb-ts-tab-content"),
    ],
    className="kb-page kb-page-ts",
)


# ---------------------------------------------------------------------------
# Slider → label sync
# ---------------------------------------------------------------------------

@callback(Output("ts-horizon-val", "children"), Input("ts-horizon", "value"))
def _sync_horizon(v):
    return f"{int(v or 12)} мес"


@callback(Output("ts-period-val", "children"), Input("ts-period", "value"))
def _sync_period(v):
    return str(int(v or 12))


@callback(Output("ts-alpha-val", "children"), Input("ts-alpha", "value"))
def _sync_alpha(v):
    return str(int(v or 10))


@callback(Output("ts-exog-label", "children"), Input("ts-exog-cols", "value"))
def _sync_exog_label(v):
    n = len(v or [])
    return f"ВНЕШНИЕ ФАКТОРЫ · {n}" if n else "ВНЕШНИЕ ФАКТОРЫ"


# ---------------------------------------------------------------------------
# Dataset & column dropdowns
# ---------------------------------------------------------------------------

@callback(
    Output("ts-ds-select", "options"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
)
def _ds_opts(raw, prep):
    names = sorted(set(list_datasets(raw) + list_datasets(prep)))
    return [{"label": n, "value": n} for n in names]


@callback(
    Output("ts-date-col", "options"),
    Output("ts-target-col", "options"),
    Output("ts-exog-cols", "options"),
    Input("ts-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _col_opts(ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return [], [], []
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return (
        [{"label": c, "value": c} for c in dt_cols],
        [{"label": c, "value": c} for c in num_cols],
        [{"label": c, "value": c} for c in num_cols],
    )


# ---------------------------------------------------------------------------
# Empty-state + tab renderer
# ---------------------------------------------------------------------------

def _empty_state() -> html.Div:
    return html.Div(
        [
            html.Div(icon("chart", 32), className="kb-ts-empty__icon"),
            html.H3("Нет данных",
                    className="kb-ts-empty__title"),
            html.P("Выберите датасет с временным рядом, затем колонку даты "
                   "и целевую переменную.",
                   className="kb-ts-empty__desc"),
        ],
        className="kb-ts-empty",
    )


@callback(
    Output("ts-tab-content", "children"),
    Input("ts-tabs", "active_tab"),
    Input("ts-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_tab(tab, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)

    # ---- 7.1 Наивный --------------------------------------------------
    if tab == "tab-naive":
        return html.Div([
            _tab_head(
                "НАИВНЫЙ / СЕЗОННЫЙ НАИВНЫЙ ПРОГНОЗ",
                "Базовая линия для сравнения с более сложными моделями",
                [
                    dcc.RadioItems(
                        id="naive-type",
                        options=[
                            {"label": "Сезонный наивный", "value": "seasonal"},
                            {"label": "Наивный (последнее значение)",
                             "value": "last"},
                        ],
                        value="seasonal", inline=True,
                        className="kb-ts-radio",
                    ),
                    html.Button(
                        [icon("play", 14),
                         html.Span("Запустить наивный прогноз")],
                        id="btn-naive", className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
            ),
            dcc.Loading(html.Div(id="naive-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.2 ARX ------------------------------------------------------
    if tab == "tab-arx":
        return html.Div([
            _tab_head(
                "ARX: AUTOREGRESSION + RIDGE",
                "Авторегрессия с экзогенными факторами, Ridge-регуляризация",
                [html.Button(
                    [icon("play", 14), html.Span("Запустить ARX")],
                    id="btn-arx", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            dcc.Loading(html.Div(id="arx-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.3 SARIMAX --------------------------------------------------
    if tab == "tab-sarimax":
        return html.Div([
            _tab_head(
                "SARIMAX: SEASONAL ARIMA С ДОВЕРИТ. ИНТЕРВАЛАМИ",
                "Сезонная ARIMA с экзогенными регрессорами и CI 95%",
                [html.Button(
                    [icon("play", 14), html.Span("Запустить SARIMAX")],
                    id="btn-sarimax", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            _sarimax_params_card(),
            dcc.Loading(html.Div(id="sarimax-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.4 Бэктестинг ----------------------------------------------
    if tab == "tab-bt":
        return html.Div([
            _tab_head(
                "ROLLING-WINDOW BACKTEST",
                "Оценка устойчивости модели на перекрывающихся историях",
                [],
            ),
            _backtest_config_card(),
            dcc.Loading(html.Div(id="backtest-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.5 ACF/PACF -------------------------------------------------
    if tab == "tab-acf":
        num_cols = (df.select_dtypes(include="number").columns.tolist()
                    if df is not None else [])
        return html.Div([
            _tab_head(
                "АВТОКОРРЕЛЯЦИОННЫЙ АНАЛИЗ",
                "ACF / PACF с 95% доверительными интервалами",
                [html.Button(
                    [icon("play", 14), html.Span("Построить ACF/PACF")],
                    id="btn-acf", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            _acf_config_card(num_cols),
            dcc.Loading(html.Div(id="acf-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.6 STL ------------------------------------------------------
    if tab == "tab-stl":
        return html.Div([
            _tab_head(
                "STL DECOMPOSITION",
                "Разложение ряда на тренд, сезонность и остаток",
                [html.Button(
                    [icon("play", 14), html.Span("Декомпозировать")],
                    id="btn-stl", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            _stl_config_card(),
            dcc.Loading(html.Div(id="stl-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.7 Диагностика ---------------------------------------------
    if tab == "tab-diag":
        return html.Div([
            _tab_head(
                "RESIDUAL DIAGNOSTICS",
                "Анализ остатков подгонки модели",
                [html.Button(
                    [icon("play", 14), html.Span("Провести диагностику")],
                    id="btn-diag", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            _diag_config_card(),
            dcc.Loading(html.Div(id="diag-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.8 Сравнение ------------------------------------------------
    if tab == "tab-compare":
        return html.Div([
            _tab_head(
                "MODEL COMPARISON",
                "Прямое сравнение прогнозов Naive / ARX / SARIMAX",
                [html.Button(
                    [icon("arrows-lr", 14),
                     html.Span("Сравнить все модели")],
                    id="btn-compare", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            dcc.Loading(html.Div(id="compare-result"), type="circle",
                        color=ACCENT_500),
        ])

    # ---- 7.9 Аномалии -------------------------------------------------
    if tab == "tab-anom":
        num_cols = (df.select_dtypes(include="number").columns.tolist()
                    if df is not None else [])
        return html.Div([
            _tab_head(
                "ANOMALY DETECTION",
                "Rolling Z-Score или STL Residual, порог в σ",
                [html.Button(
                    [icon("search", 14), html.Span("Найти аномалии")],
                    id="btn-anomaly", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                )],
            ),
            _anom_config_card(num_cols),
            dcc.Loading(html.Div(id="anomaly-result"), type="circle",
                        color=ACCENT_500),
        ])

    return html.Div()


# ---------------------------------------------------------------------------
# Per-tab config cards
# ---------------------------------------------------------------------------

def _sarimax_params_card() -> html.Div:
    def _slider_field(label: str, idc: str, min_v: int, max_v: int,
                      default: int, val_id: str) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [html.Label(label, className="kb-ts-ctrl-label"),
                     html.Span(str(default), id=val_id,
                               className="kb-ts-slider-val")],
                    className="kb-ts-slider-hdr",
                ),
                dcc.Slider(id=idc, min=min_v, max=max_v, value=default, step=1,
                           marks=None,
                           tooltip={"always_visible": False,
                                    "placement": "bottom"},
                           className="kb-ts-slider"),
                html.Div([html.Span(str(min_v)), html.Span(str(max_v))],
                         className="kb-ts-slider-range"),
            ],
            className="kb-ts-slider-field",
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("НЕСЕЗОННЫЕ", className="kb-overline"),
                    html.Div(
                        [
                            _slider_field("p (AR)", "sarimax-p", 0, 5, 2,
                                          "sarimax-p-val"),
                            _slider_field("d (diff)", "sarimax-d", 0, 2, 1,
                                          "sarimax-d-val"),
                            _slider_field("q (MA)", "sarimax-q", 0, 5, 1,
                                          "sarimax-q-val"),
                        ],
                        className="kb-ts-sarimax-grid",
                    ),
                ],
            ),
            html.Div(
                [
                    html.Div("СЕЗОННЫЕ (m из контрольной панели)",
                             className="kb-overline",
                             style={"marginTop": "12px"}),
                    html.Div(
                        [
                            _slider_field("P (sAR)", "sarimax-P", 0, 3, 1,
                                          "sarimax-P-val"),
                            _slider_field("D (sDiff)", "sarimax-D", 0, 2, 1,
                                          "sarimax-D-val"),
                            _slider_field("Q (sMA)", "sarimax-Q", 0, 3, 1,
                                          "sarimax-Q-val"),
                        ],
                        className="kb-ts-sarimax-grid",
                    ),
                ],
            ),
        ],
        className="kb-ts-sub-acc",
    )


def _backtest_config_card() -> html.Div:
    def _slider_field(label: str, idc: str, min_v: int, max_v: int,
                      default: int, val_id: str) -> html.Div:
        return html.Div(
            [
                html.Div(
                    [html.Label(label, className="kb-ts-ctrl-label"),
                     html.Span(str(default), id=val_id,
                               className="kb-ts-slider-val")],
                    className="kb-ts-slider-hdr",
                ),
                dcc.Slider(id=idc, min=min_v, max=max_v, value=default, step=1,
                           marks=None,
                           tooltip={"always_visible": False,
                                    "placement": "bottom"},
                           className="kb-ts-slider"),
                html.Div([html.Span(str(min_v)), html.Span(str(max_v))],
                         className="kb-ts-slider-range"),
            ],
            className="kb-ts-slider-field",
        )

    return html.Div(
        [
            html.Div(
                [
                    html.H3("Параметры бэктеста",
                            className="kb-ts-card__title"),
                    html.Button(
                        [icon("play", 14),
                         html.Span("Запустить бэктест")],
                        id="btn-backtest",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-ts-card__head",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("МОДЕЛЬ", className="kb-ts-ctrl-label"),
                            dcc.Dropdown(
                                id="bt-model",
                                options=[
                                    {"label": "Сезонный наивный",
                                     "value": "naive"},
                                    {"label": "ARX", "value": "arx"},
                                ],
                                value="naive", clearable=False,
                                className="kb-select kb-ts-select",
                            ),
                        ],
                        className="kb-ts-ctrl-field",
                    ),
                    _slider_field("ФОЛДОВ", "bt-folds", 2, 8, 5,
                                  "bt-folds-val"),
                    _slider_field("MIN TRAIN", "bt-min-train", 12, 60, 24,
                                  "bt-min-train-val"),
                    _slider_field("ГОРИЗОНТ ФОЛДА", "bt-horizon-fold", 1, 24, 6,
                                  "bt-horizon-fold-val"),
                ],
                className="kb-ts-bt-grid",
            ),
        ],
        className="kb-ts-card kb-ts-card--pad",
    )


def _acf_config_card(num_cols: list) -> html.Div:
    opts = [{"label": c, "value": c} for c in num_cols]
    return html.Div(
        [
            html.Div(
                [
                    html.Label("ПЕРЕМЕННАЯ", className="kb-ts-ctrl-label"),
                    dcc.Dropdown(id="acf-col", options=opts,
                                 value=(num_cols[0] if num_cols else None),
                                 clearable=False,
                                 className="kb-select kb-ts-select"),
                ],
                className="kb-ts-ctrl-field",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("ЛАГИ", className="kb-ts-ctrl-label"),
                         html.Span("30", id="acf-nlags-val",
                                   className="kb-ts-slider-val")],
                        className="kb-ts-slider-hdr",
                    ),
                    dcc.Slider(id="acf-nlags", min=5, max=60, value=30, step=1,
                               marks=None,
                               tooltip={"always_visible": False,
                                        "placement": "bottom"},
                               className="kb-ts-slider"),
                    html.Div([html.Span("5"), html.Span("60")],
                             className="kb-ts-slider-range"),
                ],
                className="kb-ts-slider-field",
            ),
        ],
        className="kb-ts-card kb-ts-card--pad kb-ts-acf-grid",
    )


def _stl_config_card() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("ТИП МОДЕЛИ", className="kb-overline"),
                    dcc.RadioItems(
                        id="stl-type",
                        options=[
                            {"label": "Аддитивная", "value": "additive"},
                            {"label": "Мультипликативная",
                             "value": "multiplicative"},
                        ],
                        value="additive", inline=True,
                        className="kb-ts-radio",
                    ),
                ],
                className="kb-ts-stl-opts",
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="stl-robust",
                        options=[{"label": "Robust", "value": "robust"}],
                        value=["robust"], inline=True,
                        className="kb-ts-check",
                    ),
                    html.Span(
                        "— рекомендуется для рядов с выбросами",
                        className="kb-ts-stl-hint",
                    ),
                ],
                className="kb-ts-stl-opts",
            ),
        ],
        className="kb-ts-card kb-ts-card--pad kb-ts-stl-config",
    )


def _diag_config_card() -> html.Div:
    return html.Div(
        [
            html.Label("МОДЕЛЬ", className="kb-ts-ctrl-label"),
            dcc.Dropdown(
                id="diag-model",
                options=[
                    {"label": "Наивный", "value": "naive"},
                    {"label": "ARX", "value": "arx"},
                    {"label": "SARIMAX", "value": "sarimax"},
                ],
                value="sarimax", clearable=False,
                className="kb-select kb-ts-select",
            ),
        ],
        className="kb-ts-card kb-ts-card--pad kb-ts-diag-config",
    )


def _anom_config_card(num_cols: list) -> html.Div:
    opts = [{"label": c, "value": c} for c in num_cols]
    return html.Div(
        [
            html.Div(
                [
                    html.Label("ПЕРЕМЕННАЯ", className="kb-ts-ctrl-label"),
                    dcc.Dropdown(id="anom-col", options=opts,
                                 value=(num_cols[0] if num_cols else None),
                                 clearable=False,
                                 className="kb-select kb-ts-select"),
                ],
                className="kb-ts-ctrl-field",
            ),
            html.Div(
                [
                    html.Label("МЕТОД", className="kb-ts-ctrl-label"),
                    dcc.RadioItems(
                        id="anom-method",
                        options=[
                            {"label": "Rolling Z", "value": "rolling_zscore"},
                            {"label": "STL Residual", "value": "stl_residual"},
                        ],
                        value="rolling_zscore", inline=True,
                        className="kb-ts-radio",
                    ),
                ],
                className="kb-ts-ctrl-field",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("ОКНО", className="kb-ts-ctrl-label"),
                         html.Span("12", id="anom-window-val",
                                   className="kb-ts-slider-val")],
                        className="kb-ts-slider-hdr",
                    ),
                    dcc.Slider(id="anom-window", min=5, max=60, value=12,
                               step=1, marks=None,
                               tooltip={"always_visible": False,
                                        "placement": "bottom"},
                               className="kb-ts-slider"),
                    html.Div([html.Span("5"), html.Span("60")],
                             className="kb-ts-slider-range"),
                ],
                className="kb-ts-slider-field",
            ),
            html.Div(
                [
                    html.Div(
                        [html.Label("ПОРОГ (σ)",
                                    className="kb-ts-ctrl-label"),
                         html.Span("2.5", id="anom-thresh-val",
                                   className="kb-ts-slider-val")],
                        className="kb-ts-slider-hdr",
                    ),
                    dcc.Slider(id="anom-thresh", min=15, max=50, value=25,
                               step=1, marks=None,
                               tooltip={"always_visible": False,
                                        "placement": "bottom"},
                               className="kb-ts-slider"),
                    html.Div([html.Span("1.5"), html.Span("5.0")],
                             className="kb-ts-slider-range"),
                ],
                className="kb-ts-slider-field",
            ),
        ],
        className="kb-ts-card kb-ts-card--pad kb-ts-anom-grid",
    )


# Value-syncing callbacks for sub-tab sliders — registered unconditionally
# via pattern-matching state; simpler to handle with explicit callbacks.
@callback(Output("sarimax-p-val", "children"), Input("sarimax-p", "value"))
def _syc_sp(v): return str(int(v or 0))
@callback(Output("sarimax-d-val", "children"), Input("sarimax-d", "value"))
def _syc_sd(v): return str(int(v or 0))
@callback(Output("sarimax-q-val", "children"), Input("sarimax-q", "value"))
def _syc_sq(v): return str(int(v or 0))
@callback(Output("sarimax-P-val", "children"), Input("sarimax-P", "value"))
def _syc_sP(v): return str(int(v or 0))
@callback(Output("sarimax-D-val", "children"), Input("sarimax-D", "value"))
def _syc_sD(v): return str(int(v or 0))
@callback(Output("sarimax-Q-val", "children"), Input("sarimax-Q", "value"))
def _syc_sQ(v): return str(int(v or 0))
@callback(Output("bt-folds-val", "children"), Input("bt-folds", "value"))
def _syc_btf(v): return str(int(v or 0))
@callback(Output("bt-min-train-val", "children"), Input("bt-min-train", "value"))
def _syc_btm(v): return str(int(v or 0))
@callback(Output("bt-horizon-fold-val", "children"),
          Input("bt-horizon-fold", "value"))
def _syc_bth(v): return str(int(v or 0))
@callback(Output("acf-nlags-val", "children"), Input("acf-nlags", "value"))
def _syc_acf(v): return str(int(v or 0))
@callback(Output("anom-window-val", "children"), Input("anom-window", "value"))
def _syc_aw(v): return str(int(v or 0))
@callback(Output("anom-thresh-val", "children"), Input("anom-thresh", "value"))
def _syc_at(v):
    t = (int(v or 25)) / 10.0
    return f"{t:.1f}"


# ---------------------------------------------------------------------------
# 7.1 — Naive forecast callback
# ---------------------------------------------------------------------------

@callback(
    Output("naive-result", "children"),
    Input("btn-naive", "n_clicks"),
    State("naive-type", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_naive(n, naive_type, ds_name, date_col, target_col,
               horizon, period, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        result = run_naive_forecast(
            df, date_col, target_col,
            horizon=int(horizon or 12),
            seasonal=(naive_type == "seasonal"),
            period=int(period or 12),
        )
        m = result.metrics
        kpis = _kpi_row([
            ("MAE",   _fmt_num(m.get("MAE")),   "среднее |err|"),
            ("RMSE",  _fmt_num(m.get("RMSE")),  "корень MSE"),
            ("MAPE",  _fmt_pct(m.get("MAPE")),  "средний %"),
            ("BIAS",  _fmt_signed(m.get("Bias")), "смещение"),
        ])
        metric_line = (
            f"MAE: {_fmt_num(m.get('MAE'), 2)} · "
            f"RMSE: {_fmt_num(m.get('RMSE'), 2)} · "
            f"MAPE: {_fmt_pct(m.get('MAPE'))} · "
            f"Bias: {_fmt_signed(m.get('Bias'), 2)}"
        )
        fig = _plot_forecast_v2(result, target_col,
                                show_fit=False,
                                model_color=TS_FORECAST, height=320)
        card = _card([
            _card_head(
                f"Наивный прогноз: {target_col}",
                right=_model_chip("NAIVE", TS_NAIVE, "warning"),
            ),
            html.Div(metric_line,
                     className="kb-ts-card__submeta kb-mono"),
            _chart_frame(fig, height=320),
            _legend_row([
                ("actuals", TS_ACTUALS),
                ("forecast", TS_FORECAST),
                ("CI 95%", TS_CI, "band"),
            ]),
        ])
        last = _card(
            [
                _card_head(
                    "Последние 30 наблюдений",
                    right=_chip(
                        f"ПОКАЗАНО {min(10, len(result.forecast_df))} · "
                        f"ИЗ {len(result.forecast_df)}", "neutral"),
                ),
                _fmt_df_tbl(result.forecast_df.tail(30), 10),
            ],
            pad=False,
        )
        return html.Div([kpis, card, last])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# 7.2 — ARX callback
# ---------------------------------------------------------------------------

@callback(
    Output("arx-result", "children"),
    Input("btn-arx", "n_clicks"),
    State("ts-alpha", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-lags", "value"),
    State("ts-exog-cols", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_arx(n, alpha_r, ds_name, date_col, target_col,
             horizon, lags_str, exog_cols, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",")
                if x.strip().isdigit()]
        if not lags:
            return alert_banner(
                "Укажите лаги (например: 1,2,3,12).", "warning")
        result = run_arx_forecast(
            df, date_col, target_col,
            exog_cols=exog_cols if exog_cols else None,
            lags=lags,
            horizon=int(horizon or 12),
            alpha=float(alpha_r or 10),
        )
        m = result.metrics
        kpis = _kpi_row([
            ("MAE",   _fmt_num(m.get("MAE")),   "среднее |err|"),
            ("RMSE",  _fmt_num(m.get("RMSE")),  "корень MSE"),
            ("MAPE",  _fmt_pct(m.get("MAPE")),  "средний %"),
            ("BIAS",  _fmt_signed(m.get("Bias")), "почти 0"),
        ])
        metric_line = (
            f"MAE: {_fmt_num(m.get('MAE'), 2)} · "
            f"RMSE: {_fmt_num(m.get('RMSE'), 2)} · "
            f"MAPE: {_fmt_pct(m.get('MAPE'))} · "
            f"Bias: {_fmt_signed(m.get('Bias'), 2)}"
        )
        fig = _plot_forecast_v2(result, target_col,
                                show_fit=True,
                                model_color=TS_FORECAST, height=300)
        arx_chip_label = f"ARX · RIDGE α={int(alpha_r or 10)}"
        forecast_card = _card([
            _card_head(
                f"ARX прогноз: {target_col}",
                right=_model_chip(arx_chip_label, TS_ARX, "info"),
            ),
            html.Div(metric_line,
                     className="kb-ts-card__submeta kb-mono"),
            _chart_frame(fig, height=300),
            _legend_row([
                ("actuals", TS_ACTUALS),
                ("fit (ridge)", TS_FIT, "dash"),
                ("forecast", TS_FORECAST),
                ("CI 95%", TS_CI, "band"),
            ]),
        ])
        children = [kpis, forecast_card]
        if result.explainability is not None and not result.explainability.empty:
            coef_fig = _coef_chart(result.explainability, top_n=15)
            coef_card = _card([
                _card_head(
                    "Top-15 коэффициентов модели",
                    right=_chip("ПО АБС. ЗНАЧЕНИЮ", "neutral"),
                ),
                _chart_frame(coef_fig,
                             height=coef_fig.layout.height or 500),
            ])
            children.append(coef_card)
        return html.Div(children)
    except Exception as e:
        return alert_banner(f"Ошибка ARX: {e}", "danger")


# ---------------------------------------------------------------------------
# 7.3 — SARIMAX callback
# ---------------------------------------------------------------------------

@callback(
    Output("sarimax-result", "children"),
    Input("btn-sarimax", "n_clicks"),
    State("sarimax-p", "value"), State("sarimax-d", "value"),
    State("sarimax-q", "value"),
    State("sarimax-P", "value"), State("sarimax-D", "value"),
    State("sarimax-Q", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State("ts-exog-cols", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_sarimax(n, p, d, q, P, D, Q, ds_name, date_col, target_col,
                 horizon, period, exog_cols, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        order = (int(p or 1), int(d or 1), int(q or 1))
        seasonal = (int(P or 1), int(D or 0), int(Q or 1), int(period or 12))
        result = run_sarimax_forecast(
            df, date_col, target_col,
            exog_cols=exog_cols if exog_cols else None,
            order=order, seasonal_order=seasonal,
            horizon=int(horizon or 12),
        )
        m = result.metrics
        kpis = _kpi_row([
            ("MAE",   _fmt_num(m.get("MAE")),     "лучший"),
            ("RMSE",  _fmt_num(m.get("RMSE")),    "лучший"),
            ("MAPE",  _fmt_pct(m.get("MAPE")),    "точность"),
            ("BIAS",  _fmt_signed(m.get("Bias")), "≈ 0"),
        ])
        aic = m.get("AIC", "")
        bic = m.get("BIC", "")
        ll = m.get("LogLik", "")
        submeta = (
            f"({p},{d},{q})({P},{D},{Q},{period}) · "
            + (f"AIC: {_fmt_num(aic, 2)} · " if aic != "" else "")
            + (f"BIC: {_fmt_num(bic, 2)} · " if bic != "" else "")
            + (f"Log-Lik: {_fmt_num(ll, 2)}" if ll != "" else "")
        )
        fig = _plot_forecast_v2(result, target_col,
                                show_fit=True,
                                model_color=TS_FORECAST, height=300)
        forecast_card = _card([
            _card_head(
                f"SARIMAX прогноз: {target_col}",
                right=_model_chip("SARIMAX", TS_SARIMAX, "success"),
            ),
            html.Div(submeta.rstrip(" ·"),
                     className="kb-ts-card__submeta kb-mono"),
            _chart_frame(fig, height=300),
            _legend_row([
                ("actuals", TS_ACTUALS),
                ("fit (in-sample)", TS_FIT, "dash"),
                ("forecast", TS_FORECAST),
                ("CI 95%", TS_CI, "band"),
            ]),
        ])
        children = [kpis, forecast_card]
        if result.notes:
            children.insert(1, alert_banner(result.notes, "info"))
        if result.explainability is not None and not result.explainability.empty:
            children.append(_sarimax_params_table_card(result.explainability))
        return html.Div(children)
    except Exception as e:
        return alert_banner(f"Ошибка SARIMAX: {e}", "danger")


def _sarimax_params_table_card(params: pd.DataFrame) -> html.Div:
    df = params.copy()
    # Normalise columns
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("param", "name", "parameter"):
            colmap[c] = "param"
        elif cl in ("coef", "coefficient", "estimate"):
            colmap[c] = "coef"
        elif cl in ("std_err", "stderr", "se", "std err"):
            colmap[c] = "se"
        elif cl in ("z", "zval", "z_value"):
            colmap[c] = "z"
        elif cl in ("p", "pval", "p_value", "p>|z|"):
            colmap[c] = "p"
        elif cl in ("ci_lower", "lower", "ci_lo", "0.025"):
            colmap[c] = "lo"
        elif cl in ("ci_upper", "upper", "ci_hi", "0.975"):
            colmap[c] = "hi"
    df = df.rename(columns=colmap)

    # Build header
    head = html.Thead(html.Tr([
        html.Th("Параметр", className="kb-ts-tbl__th"),
        html.Th("Coef", className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
        html.Th("Std.Err", className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
        html.Th("z", className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
        html.Th("P>|z|", className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
        html.Th("[0.025, 0.975]",
                className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
    ]))

    rows = []
    sig_count = 0
    total = len(df)
    for _, r in df.iterrows():
        coef = r.get("coef", float("nan"))
        se = r.get("se", float("nan"))
        z = r.get("z", float("nan"))
        p = r.get("p", float("nan"))
        lo = r.get("lo", float("nan"))
        hi = r.get("hi", float("nan"))
        sig = not np.isnan(p) and p < 0.05
        if sig:
            sig_count += 1
        p_txt = f"{p:.4f}" if not np.isnan(p) else "—"
        if not np.isnan(p):
            p_txt = f"{p:.3f}"
        p_cell = (html.Span(p_txt, className="kb-chip kb-chip--success"
                            " kb-ts-pchip") if sig
                  else html.Span(p_txt, className="kb-ts-tbl__pmuted"))
        rows.append(html.Tr([
            html.Td(str(r.get("param", r.iloc[0])), className="kb-mono"),
            html.Td(_fmt_signed(coef),
                    className="kb-mono kb-ts-tbl__td--r"),
            html.Td(_fmt_num(se),
                    className="kb-mono kb-ts-tbl__td--r kb-ts-tbl__td--muted"),
            html.Td(_fmt_num(z, 2),
                    className="kb-mono kb-ts-tbl__td--r"),
            html.Td(p_cell,
                    className="kb-ts-tbl__td--r"),
            html.Td(f"[{_fmt_num(lo, 3)}, {_fmt_num(hi, 3)}]",
                    className="kb-mono kb-ts-tbl__td--r kb-ts-tbl__td--muted"),
        ]))
    tbl = html.Table([head, html.Tbody(rows)],
                     className="kb-ts-tbl kb-ts-tbl--params")

    return _card(
        [
            _card_head(
                "Параметры SARIMAX",
                right=_chip(f"ЗНАЧИМЫХ · {sig_count} / {total}", "neutral"),
            ),
            html.Div(tbl, className="kb-ts-tbl-wrap"),
        ],
        pad=False,
    )


# ---------------------------------------------------------------------------
# 7.4 — Backtest callback
# ---------------------------------------------------------------------------

@callback(
    Output("backtest-result", "children"),
    Input("btn-backtest", "n_clicks"),
    State("bt-model", "value"), State("bt-folds", "value"),
    State("bt-min-train", "value"), State("bt-horizon-fold", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-period", "value"), State("ts-lags", "value"),
    State("ts-exog-cols", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_backtest(n, model, folds, min_train, bt_h,
                  ds_name, date_col, target_col,
                  period, lags_str, exog_cols, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",")
                if x.strip().isdigit()]
        fn = run_naive_forecast if model == "naive" else run_arx_forecast
        kwargs = ({"seasonal": True, "period": int(period or 12)}
                  if model == "naive"
                  else {"lags": lags,
                        "exog_cols": exog_cols if exog_cols else None})

        _, summary_df = rolling_backtest(
            df, date_col, target_col, model_fn=fn,
            n_folds=int(folds or 3), min_train=int(min_train or 24),
            horizon=int(bt_h or 6), **kwargs,
        )

        if summary_df.empty:
            return alert_banner(
                "Недостаточно данных для бэктеста.", "warning")

        means = {}
        stds = {}
        for col in ("MAE", "RMSE", "MAPE", "Bias"):
            if col in summary_df.columns:
                means[col] = summary_df[col].mean()
                stds[col] = summary_df[col].std()
        # sMAPE (if provided by model) — fallback to MAPE
        smape_mean = summary_df["sMAPE"].mean() if "sMAPE" in summary_df.columns \
            else means.get("MAPE", float("nan"))

        kpis = _kpi_row([
            ("СРЕДНЯЯ MAE",   _fmt_num(means.get("MAE")),
             f"± {_fmt_num(stds.get('MAE'), 2)}"),
            ("СРЕДНЯЯ RMSE",  _fmt_num(means.get("RMSE")),
             f"± {_fmt_num(stds.get('RMSE'), 2)}"),
            ("СРЕДНЯЯ MAPE",  _fmt_pct(means.get("MAPE")),
             f"± {_fmt_num(stds.get('MAPE'), 2)}"),
            ("sMAPE",          _fmt_pct(smape_mean), "симметричный"),
            ("BIAS",           _fmt_signed(means.get("Bias")), "≈ 0"),
        ])

        # MAE-per-fold chart
        mae_vals = summary_df["MAE"].tolist() if "MAE" in summary_df.columns else []
        fig = go.Figure()
        xs = list(range(1, len(mae_vals) + 1))
        if mae_vals:
            fig.add_trace(go.Scatter(
                x=xs, y=mae_vals, mode="lines+markers",
                line=dict(color=TS_FIT, width=1.8),
                marker=dict(color=TS_FIT, size=8,
                            line=dict(color=SURFACE_0, width=1.5)),
                name="MAE", showlegend=False,
            ))
            mean_mae = float(np.mean(mae_vals))
            fig.add_hline(y=mean_mae, line_dash="dash",
                          line_color=ACCENT_500, line_width=1.2,
                          annotation_text=f"avg {mean_mae:.2f}",
                          annotation_position="top right",
                          annotation_font=dict(color=ACCENT_300, size=10,
                                               family="JetBrains Mono"),
                          annotation_bgcolor=SURFACE_0)
        lay = _base_layout(200)
        lay.update(dict(
            margin=dict(l=40, r=16, t=14, b=30),
            xaxis=dict(**lay["xaxis"], title="фолд",
                       tickmode="array", tickvals=xs,
                       ticktext=[f"#{i}" for i in xs]),
            hovermode="closest",
        ))
        fig.update_layout(**lay)

        stats_chips = html.Div([
            html.Div(
                [html.Span("MIN", className="kb-ts-stat-chip__k"),
                 html.Span(_fmt_num(np.min(mae_vals)) if mae_vals else "—",
                           className="kb-ts-stat-chip__v"),
                 html.Span(
                     (f"фолд #{int(np.argmin(mae_vals)) + 1}"
                      if mae_vals else "—"),
                     className="kb-ts-stat-chip__k")],
                className="kb-ts-stat-chip"),
            html.Div(
                [html.Span("MAX", className="kb-ts-stat-chip__k"),
                 html.Span(_fmt_num(np.max(mae_vals)) if mae_vals else "—",
                           className="kb-ts-stat-chip__v"),
                 html.Span(
                     (f"фолд #{int(np.argmax(mae_vals)) + 1}"
                      if mae_vals else "—"),
                     className="kb-ts-stat-chip__k")],
                className="kb-ts-stat-chip"),
            html.Div(
                [html.Span("STD", className="kb-ts-stat-chip__k"),
                 html.Span(_fmt_num(np.std(mae_vals)) if mae_vals else "—",
                           className="kb-ts-stat-chip__v"),
                 html.Span("стабильно" if mae_vals else "—",
                           className="kb-ts-stat-chip__k")],
                className="kb-ts-stat-chip"),
        ], className="kb-ts-stat-chips")

        # Folds table
        tbl = _backtest_folds_table(summary_df)

        best_fold = (int(np.argmin(mae_vals)) + 1) if mae_vals else 0
        folds_card = _card(
            [
                _card_head(
                    "Результаты по фолдам",
                    right=_chip(f"ЛУЧШИЙ · #{best_fold}", "success")
                    if best_fold else _chip("—", "neutral"),
                ),
                html.Div(tbl, className="kb-ts-tbl-wrap"),
            ],
            pad=False,
        )
        mae_card = _card(
            [
                _card_head(
                    "MAE по фолдам",
                    right=_chip(f"±{_fmt_num(stds.get('MAE'), 2)}", "neutral"),
                ),
                _chart_frame(fig, height=200),
                stats_chips,
            ],
        )

        split = html.Div([folds_card, mae_card],
                         className="kb-ts-bt-split")

        return html.Div([kpis, split])
    except Exception as e:
        return alert_banner(f"Ошибка бэктеста: {e}", "danger")


def _backtest_folds_table(summary_df: pd.DataFrame) -> html.Table:
    cols = summary_df.columns.tolist()
    head_cells = [html.Th("#", className="kb-ts-tbl__th kb-mono")]
    for c in cols:
        right = c in ("MAE", "RMSE", "MAPE", "sMAPE", "Bias")
        head_cells.append(html.Th(
            c,
            className="kb-ts-tbl__th"
                     + (" kb-ts-tbl__th--r kb-mono" if right else ""),
        ))
    head = html.Thead(html.Tr(head_cells))

    rows = []
    sums = {c: 0.0 for c in cols}
    for idx, r in summary_df.reset_index(drop=True).iterrows():
        cells = [html.Td(str(idx + 1), className="kb-mono")]
        for c in cols:
            v = r[c]
            if c == "Bias" and isinstance(v, (int, float)):
                txt = _fmt_signed(v)
                color_cls = ("kb-ts-tbl__td--pos" if v >= 0
                             else "kb-ts-tbl__td--neg")
                cls = f"kb-mono kb-ts-tbl__td--r {color_cls}"
            elif c in ("MAE", "RMSE"):
                txt = _fmt_num(v)
                cls = "kb-mono kb-ts-tbl__td--r"
            elif c in ("MAPE", "sMAPE"):
                txt = _fmt_pct(v)
                cls = "kb-mono kb-ts-tbl__td--r"
            elif isinstance(v, (int, float)):
                txt = _fmt_num(v)
                cls = "kb-mono kb-ts-tbl__td--r"
            else:
                txt = "—" if pd.isna(v) else str(v)
                cls = "kb-ts-tbl__td--muted"
            cells.append(html.Td(txt, className=cls))
            if isinstance(v, (int, float)) and not pd.isna(v):
                sums[c] = sums.get(c, 0.0) + v
        rows.append(html.Tr(cells))

    # Average row
    n = max(len(summary_df), 1)
    avg_cells = [html.Td("Среднее", className="kb-mono")]
    for c in cols:
        if c in ("MAE", "RMSE"):
            avg_cells.append(html.Td(_fmt_num(sums[c] / n),
                                     className="kb-mono kb-ts-tbl__td--r"))
        elif c in ("MAPE", "sMAPE"):
            avg_cells.append(html.Td(_fmt_pct(sums[c] / n),
                                     className="kb-mono kb-ts-tbl__td--r"))
        elif c == "Bias":
            avg_cells.append(html.Td(_fmt_signed(sums[c] / n),
                                     className="kb-mono kb-ts-tbl__td--r"))
        else:
            avg_cells.append(html.Td(
                (f"{n} фолдов" if c == cols[0] else "—"),
                className="kb-ts-tbl__td--muted"))
    foot = html.Tr(avg_cells, className="kb-ts-tbl__footrow")
    return html.Table([head, html.Tbody(rows + [foot])],
                      className="kb-ts-tbl kb-ts-tbl--folds")


# ---------------------------------------------------------------------------
# 7.5 — ACF/PACF callback
# ---------------------------------------------------------------------------

@callback(
    Output("acf-result", "children"),
    Input("btn-acf", "n_clicks"),
    State("acf-col", "value"), State("acf-nlags", "value"),
    State("ts-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_acf(n, col, nlags, ds_name, raw, prep):
    if not col:
        return alert_banner("Выберите числовую колонку.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        from statsmodels.tsa.stattools import acf, pacf, adfuller
        series = df[col].dropna().values
        if len(series) < 10:
            return alert_banner("Недостаточно данных.", "warning")
        nlags_val = min(int(nlags or 30), len(series) // 2 - 1)
        acf_vals = acf(series, nlags=nlags_val, fft=True)
        pacf_vals = pacf(series, nlags=nlags_val)

        # ADF test
        adf_card = None
        try:
            adf_res = adfuller(series, autolag="AIC")
            adf_stat, adf_p = adf_res[0], adf_res[1]
            adf_crit = adf_res[4]
            is_stationary = adf_p < 0.05
            level = "success" if is_stationary else "warning"
            title = ("ADF-тест: ряд стационарен" if is_stationary
                     else "ADF-тест: ряд не стационарен")
            body_bits = [
                f"ADF stat = {adf_stat:.3f}",
                f"p-value = {adf_p:.4f}",
                "критические значения: "
                + " · ".join(f"{k} = {v:.3f}" for k, v in adf_crit.items()),
                ("требуется дифференцирование (d ≥ 1)" if not is_stationary
                 else "дифференцирование не требуется"),
            ]
            adf_card = html.Div(
                [
                    html.Div(icon("alert" if not is_stationary else "check", 14),
                             className=f"kb-ts-alert__icon"
                                       f" kb-ts-alert__icon--{level}"),
                    html.Div(
                        [
                            html.Div(title, className="kb-ts-alert__title"),
                            html.Div(
                                " — ".join([" · ".join(body_bits[:3]),
                                            body_bits[3]]),
                                className="kb-ts-alert__body kb-mono"),
                        ],
                        className="kb-ts-alert__body-wrap",
                    ),
                ],
                className=f"kb-ts-alert kb-ts-alert--{level}",
            )
        except Exception:
            adf_card = None

        acf_fig = _acf_stems(acf_vals, len(series), ACCENT_500, height=200)
        pacf_fig = _acf_stems(pacf_vals, len(series), TS_FIT, height=200)

        children = []
        if adf_card:
            children.append(adf_card)

        children.append(_card([
            _card_head(
                "ACF — автокорреляция",
                right=_chip(f"ЛАГОВ · {nlags_val} · N = {len(series)}",
                            "neutral"),
            ),
            _chart_frame(acf_fig, height=200),
            _card_head(
                "PACF — частичная автокорреляция",
                right=_chip("СЕЗОННОСТЬ m=12", "info"),
            ),
            _chart_frame(pacf_fig, height=200),
            _legend_row([
                ("ACF stems", ACCENT_500),
                ("PACF stems", TS_FIT),
                ("CI 95% · ±1.96/√N", DANGER, "dash"),
            ]),
        ]))
        return html.Div(children)
    except ImportError:
        return alert_banner("statsmodels не установлен.", "warning")
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# 7.6 — STL decomposition callback
# ---------------------------------------------------------------------------

@callback(
    Output("stl-result", "children"),
    Input("btn-stl", "n_clicks"),
    State("stl-type", "value"), State("stl-robust", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-period", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_stl(n, stl_type, robust_chk, ds_name, date_col, target_col,
             period, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        from statsmodels.tsa.seasonal import STL
        sdf = df[[date_col, target_col]].dropna().sort_values(date_col)
        y = sdf[target_col].astype(float).values
        x = sdf[date_col].values
        per = int(period or 12)
        if len(y) < 2 * per:
            return alert_banner(
                f"Недостаточно данных для STL (нужно ≥ {2 * per}).", "warning")
        if stl_type == "multiplicative":
            logy = np.log(np.clip(y, 1e-9, None))
            stl = STL(logy, period=per,
                      robust="robust" in (robust_chk or []))
            r = stl.fit()
            trend = np.exp(r.trend)
            season = np.exp(r.seasonal)
            resid = y - (trend * season)
            obs = y
        else:
            stl = STL(y, period=per, robust="robust" in (robust_chk or []))
            r = stl.fit()
            trend = r.trend
            season = r.seasonal
            resid = r.resid
            obs = y

        # Strength-of-signal
        var_resid = float(np.nanvar(resid))
        var_det = float(np.nanvar(resid + season))
        var_tr = float(np.nanvar(resid + trend))
        fs = max(0.0, 1 - var_resid / var_det) if var_det > 0 else 0.0
        ft = max(0.0, 1 - var_resid / var_tr) if var_tr > 0 else 0.0
        explained = (1 - (var_resid /
                          float(np.nanvar(y))) if np.nanvar(y) > 0 else 0.0)

        strong = fs >= 0.4
        alert = html.Div(
            [
                html.Div(icon("check" if strong else "alert", 14),
                         className=("kb-ts-alert__icon "
                                    + ("kb-ts-alert__icon--success" if strong
                                       else "kb-ts-alert__icon--warning"))),
                html.Div(
                    [
                        html.Div(
                            "Сильная сезонность" if strong
                            else "Слабая сезонность",
                            className="kb-ts-alert__title"),
                        html.Div(
                            [
                                "Fs = ",
                                html.Span(f"{fs:.2f}", className="kb-mono",
                                          style={"color": ACCENT_300}),
                                " · Ft = ",
                                html.Span(f"{ft:.2f}", className="kb-mono",
                                          style={"color": TEXT_PRI}),
                                f" · тренд и сезонность объясняют "
                                f"{explained * 100:.0f}% дисперсии. "
                                f"Период m = {per}.",
                            ],
                            className="kb-ts-alert__body"),
                    ],
                    className="kb-ts-alert__body-wrap",
                ),
            ],
            className=("kb-ts-alert "
                       + ("kb-ts-alert--success" if strong
                          else "kb-ts-alert--warning")),
        )

        labels_colors = [
            ("НАБЛЮДЕНИЕ", obs, TS_ACTUALS, False),
            ("ТРЕНД", trend, TS_FIT, False),
            ("СЕЗОННОСТЬ", season, ACCENT_500, False),
            ("ОСТАТОК", resid, WARNING, True),
        ]
        panels = []
        for lbl, data, color, zero in labels_colors:
            fig = _stl_panel(x, data, color, height=120, show_zero=zero)
            panels.append(html.Div(
                [
                    html.Div(lbl, className="kb-overline kb-ts-stl-label"),
                    html.Div(_chart_frame(fig, height=120),
                             className="kb-ts-stl-frame"),
                ],
                className="kb-ts-stl-row",
            ))

        stl_card = _card([
            _card_head(
                f"STL · 4-панельная декомпозиция · m = {per}",
                right=_chip(f"{len(y)} ТОЧЕК · {stl_type}", "neutral"),
            ),
            html.Div(panels, className="kb-ts-stl-panels"),
        ])
        return html.Div([alert, stl_card])
    except Exception as e:
        return alert_banner(f"Ошибка STL: {e}", "danger")


# ---------------------------------------------------------------------------
# 7.7 — Diagnostics callback
# ---------------------------------------------------------------------------

@callback(
    Output("diag-result", "children"),
    Input("btn-diag", "n_clicks"),
    State("diag-model", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State("ts-exog-cols", "value"),
    State("ts-lags", "value"), State("ts-alpha", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_diag(n, model, ds_name, date_col, target_col,
              horizon, period, exog_cols, lags_str, alpha_r,
              raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",")
                if x.strip().isdigit()]
        if model == "naive":
            result = run_naive_forecast(
                df, date_col, target_col,
                horizon=int(horizon or 12),
                seasonal=True, period=int(period or 12),
            )
        elif model == "arx":
            result = run_arx_forecast(
                df, date_col, target_col,
                exog_cols=exog_cols if exog_cols else None,
                lags=lags, horizon=int(horizon or 12),
                alpha=float(alpha_r or 10),
            )
        else:
            result = run_sarimax_forecast(
                df, date_col, target_col,
                exog_cols=exog_cols if exog_cols else None,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, int(period or 12)),
                horizon=int(horizon or 12),
            )

        fd = result.forecast_df
        hist = fd[fd["actual"].notna() & fd["forecast"].notna()].copy()
        if hist.empty:
            return alert_banner(
                "Нет данных для диагностики (in-sample fit отсутствует).",
                "warning")
        residuals = (hist["actual"] - hist["forecast"]).values
        fitted = hist["forecast"].values

        from statsmodels.stats.diagnostic import acorr_ljungbox
        try:
            lb_lags = [l for l in (5, 10, 15, 20, 25) if l < len(residuals)]
            lb = acorr_ljungbox(residuals, lags=lb_lags, return_df=True)
        except Exception:
            lb = pd.DataFrame()

        all_pass = (lb["lb_pvalue"].gt(0.05).all()
                    if not lb.empty else True)
        alert = html.Div(
            [
                html.Div(
                    icon("check" if all_pass else "alert", 14),
                    className=("kb-ts-alert__icon "
                               + ("kb-ts-alert__icon--success" if all_pass
                                  else "kb-ts-alert__icon--warning"))),
                html.Div(
                    [
                        html.Div(
                            ("Ljung-Box: остатки не содержат автокорреляции"
                             if all_pass
                             else "Ljung-Box: остатки содержат автокорреляцию"),
                            className="kb-ts-alert__title"),
                        html.Div(
                            ("p-value > 0.05 на всех лагах — модель адекватно "
                             "описывает временную структуру" if all_pass
                             else "на некоторых лагах p < 0.05 — модель "
                                  "не захватывает автокорреляцию"),
                            className="kb-ts-alert__body kb-mono"),
                    ],
                    className="kb-ts-alert__body-wrap",
                ),
            ],
            className=("kb-ts-alert "
                       + ("kb-ts-alert--success" if all_pass
                          else "kb-ts-alert--warning")),
        )

        # ACF of residuals
        from statsmodels.tsa.stattools import acf as acf_fn
        n_lags_res = min(24, len(residuals) // 2 - 1)
        if n_lags_res < 1:
            n_lags_res = 5
        try:
            acf_res = acf_fn(residuals, nlags=n_lags_res, fft=True)
        except Exception:
            acf_res = np.zeros(n_lags_res + 1)
        acf_fig = _acf_stems(acf_res, len(residuals), ACCENT_500, height=200)
        qq_fig = _qq_chart(residuals, height=200)
        time_fig = _resid_time_chart(residuals, height=200)
        fit_fig = _resid_fit_chart(fitted, residuals, height=200)

        # Ljung-Box table
        lb_rows = []
        for lag in lb.index:
            lbstat = lb.loc[lag, "lb_stat"]
            pval = lb.loc[lag, "lb_pvalue"]
            sig = pval < 0.05
            lb_rows.append(html.Tr([
                html.Td(str(int(lag)), className="kb-mono"),
                html.Td(_fmt_num(lbstat, 2),
                        className="kb-mono kb-ts-tbl__td--r"),
                html.Td(_fmt_num(pval, 4),
                        className="kb-mono kb-ts-tbl__td--r"),
                html.Td(
                    _chip("ДА", "danger") if sig else _chip("НЕТ", "success"),
                    className="kb-ts-tbl__td--center"),
            ]))
        lb_table = html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Лаг", className="kb-ts-tbl__th kb-mono"),
                    html.Th("LB-stat",
                            className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
                    html.Th("p-value",
                            className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
                    html.Th("Значимо?",
                            className="kb-ts-tbl__th kb-ts-tbl__th--c"),
                ])),
                html.Tbody(lb_rows),
            ],
            className="kb-ts-tbl kb-ts-tbl--diag",
        )

        left = _card(
            [
                html.H3("Диагностические графики",
                        className="kb-ts-card__title"),
                html.Div(
                    [
                        html.Div(
                            [html.Div("ACF ОСТАТКОВ", className="kb-overline"),
                             _chart_frame(acf_fig, height=200)],
                            className="kb-ts-diag-cell",
                        ),
                        html.Div(
                            [html.Div("Q-Q PLOT", className="kb-overline"),
                             _chart_frame(qq_fig, height=200)],
                            className="kb-ts-diag-cell",
                        ),
                        html.Div(
                            [html.Div("ОСТАТКИ ВО ВРЕМЕНИ",
                                      className="kb-overline"),
                             _chart_frame(time_fig, height=200)],
                            className="kb-ts-diag-cell",
                        ),
                        html.Div(
                            [html.Div("ОСТАТКИ VS ПОДГОНКА",
                                      className="kb-overline"),
                             _chart_frame(fit_fig, height=200)],
                            className="kb-ts-diag-cell",
                        ),
                    ],
                    className="kb-ts-diag-grid",
                ),
            ],
        )
        right = _card(
            [
                _card_head("Ljung-Box тест"),
                html.Div(lb_table, className="kb-ts-tbl-wrap"),
            ],
            pad=False,
        )
        return html.Div(
            [alert, html.Div([left, right], className="kb-ts-diag-split")]
        )
    except Exception as e:
        return alert_banner(f"Ошибка диагностики: {e}", "danger")


# ---------------------------------------------------------------------------
# 7.8 — Compare all models callback
# ---------------------------------------------------------------------------

@callback(
    Output("compare-result", "children"),
    Input("btn-compare", "n_clicks"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State("ts-exog-cols", "value"),
    State("ts-lags", "value"), State("ts-alpha", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_compare(n, ds_name, date_col, target_col,
                 horizon, period, exog_cols, lags_str, alpha_r,
                 raw, prep):
    if not all([date_col, target_col]):
        return alert_banner(
            "Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",")
                if x.strip().isdigit()]
        H = int(horizon or 12)
        results = []
        results.append(("Naive", TS_NAIVE, run_naive_forecast(
            df, date_col, target_col,
            horizon=H, seasonal=True, period=int(period or 12))))
        results.append(("ARX", TS_ARX, run_arx_forecast(
            df, date_col, target_col,
            exog_cols=exog_cols if exog_cols else None,
            lags=lags, horizon=H, alpha=float(alpha_r or 10))))
        try:
            results.append(("SARIMAX", TS_SARIMAX, run_sarimax_forecast(
                df, date_col, target_col,
                exog_cols=exog_cols if exog_cols else None,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, int(period or 12)),
                horizon=H)))
        except Exception:
            pass

        # Metrics table
        metric_cols = ("MAE", "RMSE", "MAPE", "Bias")
        rows_data = []
        for name, color, r in results:
            rows_data.append({
                "name": name, "color": color,
                **{c: r.metrics.get(c, float("nan")) for c in metric_cols},
            })
        # Best per column (min for MAE/RMSE/MAPE, closest-to-zero for Bias)
        best_idx = {}
        for c in metric_cols:
            vals = [r[c] for r in rows_data]
            if c == "Bias":
                best_idx[c] = int(np.argmin([abs(v) for v in vals]))
            else:
                best_idx[c] = int(np.argmin(vals))

        head_cells = [html.Th("Модель", className="kb-ts-tbl__th")]
        for c in metric_cols:
            head_cells.append(html.Th(
                c,
                className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"))
        head = html.Thead(html.Tr(head_cells))

        body_rows = []
        for i, row in enumerate(rows_data):
            cells = [html.Td(
                html.Span(
                    [html.Span(className="kb-ts-cdot",
                               style={"background": row["color"]}),
                     row["name"]],
                    className="kb-ts-model-mark",
                ),
            )]
            for c in metric_cols:
                v = row[c]
                txt = (_fmt_signed(v) if c == "Bias"
                       else (_fmt_pct(v) if c == "MAPE" else _fmt_num(v)))
                best_cls = " kb-ts-tbl__td--best" if best_idx[c] == i else ""
                cells.append(html.Td(
                    txt,
                    className=f"kb-mono kb-ts-tbl__td--r{best_cls}"))
            body_rows.append(html.Tr(cells))
        metrics_table = html.Table(
            [head, html.Tbody(body_rows)],
            className="kb-ts-tbl kb-ts-tbl--compare",
        )

        # Verdict (best by MAPE)
        best_model = rows_data[best_idx["MAPE"]]
        verdict = html.Div(
            [
                html.Div(
                    [
                        _chip("ЛУЧШАЯ МОДЕЛЬ", "success"),
                        html.H2(
                            [
                                html.Span(
                                    className="kb-ts-cdot kb-ts-cdot--lg",
                                    style={"background": best_model["color"]}),
                                html.Span(best_model["name"]),
                            ],
                            className="kb-ts-verdict__title",
                        ),
                        html.Div(
                            f"по MAPE: {_fmt_pct(best_model['MAPE'])} · "
                            f"RMSE: {_fmt_num(best_model['RMSE'], 2)} · "
                            f"стабильное преимущество",
                            className="kb-ts-verdict__meta",
                        ),
                    ],
                    className="kb-ts-verdict__body",
                ),
                html.Div(
                    [
                        html.Button(
                            [icon("file-text", 12), html.Span("В отчёт")],
                            className="kb-btn kb-btn--secondary kb-btn--sm",
                            id="cmp-report-btn", n_clicks=0,
                        ),
                        html.Button(
                            [icon("download", 12), html.Span("Экспорт")],
                            className="kb-btn kb-btn--ghost kb-btn--sm",
                            id="cmp-export-btn", n_clicks=0,
                        ),
                    ],
                    className="kb-ts-verdict__actions",
                ),
            ],
            className="kb-ts-verdict",
        )

        # Overlay chart: actuals + 3 forecasts + SARIMAX CI
        fig = go.Figure()
        # Historical actuals — taken from first result
        base_fd = results[0][2].forecast_df
        hist = base_fd[base_fd["actual"].notna()]
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["actual"],
            mode="lines", name="actuals",
            line=dict(color=TS_ACTUALS, width=1.8),
        ))

        # SARIMAX CI ribbon (if present)
        for name, color, r in results:
            if name != "SARIMAX":
                continue
            rfd = r.forecast_df
            fut = rfd[rfd["actual"].isna() & rfd["forecast"].notna()]
            if not fut.empty and "lower" in fut and fut["lower"].notna().any():
                xs = pd.concat([fut["date"], fut["date"][::-1]])
                ys = pd.concat([fut["upper"], fut["lower"][::-1]])
                fig.add_trace(go.Scatter(
                    x=xs, y=ys,
                    fill="toself", fillcolor="rgba(160,102,200,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="CI SARIMAX", hoverinfo="skip",
                ))

        # Model forecasts
        for name, color, r in results:
            rfd = r.forecast_df
            fut = rfd[rfd["actual"].isna() & rfd["forecast"].notna()]
            if fut.empty:
                continue
            # Bridge from last actual
            bridge_x = pd.concat([hist["date"].iloc[[-1]], fut["date"]])
            bridge_y = pd.concat([hist["actual"].iloc[[-1]], fut["forecast"]])
            fig.add_trace(go.Scatter(
                x=bridge_x, y=bridge_y, mode="lines",
                name=name.lower(),
                line=dict(color=color, width=2),
            ))
            fig.add_trace(go.Scatter(
                x=fut["date"], y=fut["forecast"],
                mode="markers", showlegend=False,
                marker=dict(color=color, size=5),
                hovertemplate=f"{name}: %{{y:.2f}}<extra></extra>",
            ))
        if not hist.empty:
            fig.add_vline(x=hist["date"].iloc[-1],
                          line_dash="dash", line_color=TS_TODAY,
                          line_width=1, opacity=0.7)
        lay = _base_layout(320)
        lay.update(dict(yaxis_title=target_col))
        fig.update_layout(**lay)

        overlay_card = _card([
            _card_head(
                "Overlay · прогнозы на одной оси",
                right=_chip(
                    f"ACTUALS + {len(results)} МОДЕЛИ + CI SARIMAX",
                    "neutral"),
            ),
            _chart_frame(fig, height=320),
            _legend_row(
                [("actuals", TS_ACTUALS)]
                + [(name.lower(), color) for name, color, _ in results]
                + [("CI SARIMAX", "rgba(160,102,200,0.35)", "band")],
            ),
        ])

        metrics_card = _card(
            [
                _card_head(
                    "Метрики по моделям",
                    right=_chip("ЛУЧШЕЕ · ПОДСВЕЧЕНО", "neutral"),
                ),
                html.Div(metrics_table, className="kb-ts-tbl-wrap"),
            ],
            pad=False,
        )
        return html.Div([verdict, metrics_card, overlay_card])
    except Exception as e:
        return alert_banner(f"Ошибка сравнения: {e}", "danger")


# ---------------------------------------------------------------------------
# 7.9 — Anomaly detection callback
# ---------------------------------------------------------------------------

@callback(
    Output("anomaly-result", "children"),
    Input("btn-anomaly", "n_clicks"),
    State("anom-col", "value"), State("anom-method", "value"),
    State("anom-window", "value"), State("anom-thresh", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-period", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_anomaly(n, col, method, window, thresh_tick,
                 ds_name, date_col, period, raw, prep):
    if not col:
        return alert_banner("Выберите числовую колонку.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        threshold = (int(thresh_tick) / 10.0 if thresh_tick is not None
                     else 2.5)
        # slider range 15..50 maps to 1.5..5.0; already encoded that way above
        if threshold > 10:
            threshold = threshold / 10.0
        series = df[col].dropna()
        if series.empty:
            return alert_banner("Нет данных.", "warning")
        anom_df = detect_anomalies(
            series, method=method or "rolling_zscore",
            window=int(window or 12),
            threshold=float(threshold),
            period=int(period or 12),
        )
        anomalies = anom_df[anom_df["is_anomaly"]].copy()
        n_total = len(anom_df)
        n_anom = int(anomalies.shape[0])
        z = anom_df["z_score"] if "z_score" in anom_df.columns else None
        n_pos = int((z > threshold).sum()) if z is not None else 0
        n_neg = int((z < -threshold).sum()) if z is not None else 0
        pct = (100.0 * n_anom / n_total) if n_total else 0.0

        # Optionally align dates
        dates = None
        if date_col and date_col in df.columns:
            sdf = df[[date_col, col]].dropna().sort_values(date_col)
            dates = sdf[date_col].values if len(sdf) == n_total else None
        x_axis = dates if dates is not None else np.arange(n_total)

        # Build chart
        fig = go.Figure()
        # ±σ ribbon
        if "upper" in anom_df.columns and "lower" in anom_df.columns:
            xs = np.concatenate([x_axis, x_axis[::-1]])
            ys = np.concatenate([anom_df["upper"].values,
                                  anom_df["lower"].values[::-1]])
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                fill="toself", fillcolor="rgba(200,80,59,0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"±{threshold:.1f}σ band", hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=x_axis, y=anom_df["value"].values,
            mode="lines", name="value",
            line=dict(color=TS_ACTUALS, width=1.5),
        ))
        if n_anom:
            idx = anomalies.index.values
            if dates is not None:
                ax = dates[idx]
            else:
                ax = idx
            ay = anomalies["value"].values
            fig.add_trace(go.Scatter(
                x=ax, y=ay, mode="markers",
                marker=dict(color=DANGER, size=14, symbol="x-thin",
                            line=dict(color=DANGER, width=2.5)),
                name="аномалия",
                hovertemplate="%{x}: %{y:.2f}<extra></extra>",
            ))
        lay = _base_layout(340)
        lay.update(dict(yaxis_title=col))
        fig.update_layout(**lay)

        # Top-|z| table
        top = anomalies.copy()
        if "z_score" in top.columns:
            top["_abs"] = top["z_score"].abs()
            top = top.sort_values("_abs", ascending=False).drop("_abs", axis=1)
        top_rows = []
        for row_idx, row in top.head(20).iterrows():
            zv = row.get("z_score", float("nan"))
            sev = "danger" if abs(zv) > 3.0 else "warning"
            sev_lbl = "КРИТИЧНО" if sev == "danger" else "ВНИМАНИЕ"
            if dates is not None:
                try:
                    d = pd.to_datetime(dates[row_idx])
                    d_txt = d.strftime("%Y-%m-%d")
                except Exception:
                    d_txt = str(row_idx)
            else:
                d_txt = str(row_idx)
            v = row.get("value", float("nan"))
            top_rows.append(html.Tr([
                html.Td(d_txt, className="kb-mono kb-ts-tbl__td--muted"),
                html.Td(_fmt_num(v, 2),
                        className="kb-mono kb-ts-tbl__td--r"),
                html.Td(_fmt_signed(zv, 2),
                        className=("kb-mono kb-ts-tbl__td--r "
                                   + ("kb-ts-tbl__td--pos" if zv >= 0
                                      else "kb-ts-tbl__td--neg"))),
                html.Td(_chip(sev_lbl, sev),
                        className="kb-ts-tbl__td--c"),
            ]))
        table = html.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Date", className="kb-ts-tbl__th"),
                    html.Th("Value",
                            className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
                    html.Th("Z-Score",
                            className="kb-ts-tbl__th kb-ts-tbl__th--r kb-mono"),
                    html.Th("Severity",
                            className="kb-ts-tbl__th kb-ts-tbl__th--c"),
                ])),
                html.Tbody(top_rows),
            ],
            className="kb-ts-tbl kb-ts-tbl--anom",
        )

        kpis = _kpi_row([
            ("НАЙДЕНО АНОМАЛИЙ", str(n_anom), f"из {n_total} точек"),
            ("ПОЛОЖИТЕЛЬНЫХ", str(n_pos),
             f"z > +{threshold:.1f}σ"),
            ("ОТРИЦАТЕЛЬНЫХ", str(n_neg),
             f"z < -{threshold:.1f}σ"),
            ("% ОТ N", f"{pct:.2f} %", "выше порога"),
        ])

        chart_card = _card([
            _card_head(
                f"{col} · обнаруженные аномалии",
                right=html.Div(
                    [_chip("|z| > 3.0 · КРИТИЧНО", "danger"),
                     _chip(f"{threshold:.1f} ≤ |z| ≤ 3.0 · ВНИМАНИЕ",
                           "warning")],
                    className="kb-ts-chip-row",
                ),
            ),
            _chart_frame(fig, height=340),
            _legend_row([
                ("значение", TS_ACTUALS),
                (f"±{threshold:.1f}σ band", "rgba(200,80,59,0.10)", "band"),
                ("аномалия (X-маркер)", DANGER),
            ]),
        ])
        list_card = _card(
            [
                _card_head(
                    "Аномалии · по |z-score| ↓",
                    right=_chip(f"{min(len(top), 20)} ЗАПИСЕЙ", "neutral"),
                ),
                html.Div(table, className="kb-ts-tbl-wrap"),
            ],
            pad=False,
        )
        if n_anom == 0:
            return html.Div([kpis, chart_card])
        return html.Div([kpis, chart_card, list_card])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")

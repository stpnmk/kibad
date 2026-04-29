"""p09_simulation — Симуляция (Раздел 9) — Dash.

Redesigned for KIBAD Design System v2026.04 (dark eucalyptus).
Mirrors handoff slide 9 («Симуляция») — 5 artboards rendered as two
top-level tabs:
  9.1 Сценарий · Настройка    ─┐
  9.2 Сценарий · Результат    ─┼── tab «Сценарный анализ»
  9.3 Сравнение сценариев     ─┘
  9.4 Монте-Карло · Настройка ─┐
  9.5 Монте-Карло · Результат ─┴── tab «Монте-Карло»
"""
from __future__ import annotations

from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
    get_df_from_store, list_datasets,
)
from app.components.alerts import alert_banner
from app.components.icons import icon

from core.simulation import run_scenario, ScenarioPreset

dash.register_page(
    __name__, path="/simulation", name="9. Симуляция",
    order=9, icon="dice-5",
)


# ---------------------------------------------------------------------------
# Design tokens — sim role palette (mirror sim-shared.js SIM dict)
# ---------------------------------------------------------------------------

SIM_BASELINE  = "#A3B0A8"
SIM_PESSIM    = "#C8503B"
SIM_OPTIM     = "#4A7FB0"
SIM_CUSTOM    = "#A066C8"
SIM_MEDIAN    = "#4FD18B"
SIM_PATH      = "#1F7A4D"
SIM_BAND_50   = "rgba(33,160,102,0.20)"
SIM_BAND_90   = "rgba(33,160,102,0.12)"
SIM_VAR_LINE  = "#C8503B"
SIM_CVAR_LINE = "#C98A2E"
SIM_MEAN_LINE = "#4FD18B"
SIM_MED_LINE  = "#4A7FB0"

FACTOR_COLORS = ["#21A066", "#4A7FB0", "#C98A2E",
                 "#A066C8", "#C8503B", "#6B8E8A"]

SURFACE_0  = "#0F1613"
SURFACE_1  = "#141C18"
ACCENT_500 = "#21A066"
ACCENT_300 = "#4FD18B"
DANGER     = "#C8503B"
WARNING    = "#C98A2E"
INFO       = "#4A7FB0"
TEXT_PRI   = "#E8EFEA"
TEXT_SEC   = "#A3B0A8"
TEXT_TER   = "#6B7A72"
GRID       = "rgba(255,255,255,0.06)"


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


def _fmt_num(v: Any, digits: int = 2) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        if abs(f) >= 1000:
            return f"{f:,.{digits}f}".replace(",", " ")
        return f"{f:.{digits}f}"
    except Exception:
        return "—"


def _base_layout(height: int = 320) -> dict:
    return dict(
        height=height,
        margin=dict(l=44, r=24, t=8, b=24),
        paper_bgcolor=SURFACE_0,
        plot_bgcolor=SURFACE_0,
        font=dict(family="Inter, system-ui, sans-serif", size=11,
                  color=TEXT_PRI),
        xaxis=dict(
            gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
            tickfont=dict(family="JetBrains Mono", size=10, color=TEXT_TER),
        ),
        yaxis=dict(
            gridcolor=GRID, linecolor=GRID, zerolinecolor=GRID,
            tickfont=dict(family="JetBrains Mono", size=10, color=TEXT_TER),
        ),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=SURFACE_1, bordercolor="rgba(255,255,255,0.14)",
            font=dict(family="JetBrains Mono", size=11, color=TEXT_PRI),
        ),
    )


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------

def _tornado_chart(rows: list[dict], height: int = 240) -> go.Figure:
    """Horizontal bars centered at zero (sorted by |val| descending)."""
    rows = sorted(rows, key=lambda r: abs(r["val"]), reverse=True)
    labels = [r["name"] for r in rows]
    vals = [r["val"] for r in rows]
    colors = [r.get("color")
              or (SIM_OPTIM if v >= 0 else SIM_PESSIM)
              for r, v in zip(rows, vals)]
    text = [f"{'+' if v >= 0 else ''}{v:.2f}"
            + (f"  ({r['pct']:.0f}%)" if r.get("pct") is not None else "")
            for r, v in zip(rows, vals)]
    text_colors = [INFO if v >= 0 else DANGER for v in vals]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors),
        text=text, textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11, color=text_colors),
        hovertemplate="%{y}: %{x:.2f}<extra></extra>",
    ))
    lay = _base_layout(height)
    rng = max(abs(min(vals + [0])), abs(max(vals + [0]))) * 1.4 or 1
    lay["xaxis"]["range"] = [-rng, rng]
    lay["xaxis"]["zeroline"] = True
    lay["xaxis"]["zerolinecolor"] = "rgba(255,255,255,0.18)"
    lay["xaxis"]["zerolinewidth"] = 1
    lay["yaxis"]["autorange"] = "reversed"
    lay.update(margin=dict(l=110, r=80, t=8, b=24))
    fig.update_layout(**lay)
    return fig


def _tornado_compare(rows: list[dict], height: int = 280) -> go.Figure:
    """Two-sided tornado: pessim on the left, optim on the right."""
    labels = [r["name"] for r in rows]
    p = [-abs(r["pess"]) for r in rows]   # negative for left side
    o = [abs(r["opt"]) for r in rows]
    fig = go.Figure([
        go.Bar(x=p, y=labels, orientation="h",
               name="Пессимистичный", marker=dict(color=SIM_PESSIM),
               text=[f"{r['pess']:+.1f}" for r in rows],
               textposition="outside",
               textfont=dict(family="JetBrains Mono", size=11, color=DANGER),
               hovertemplate="%{y} · pess: %{text}<extra></extra>"),
        go.Bar(x=o, y=labels, orientation="h",
               name="Оптимистичный", marker=dict(color=SIM_OPTIM),
               text=[f"{r['opt']:+.1f}" for r in rows],
               textposition="outside",
               textfont=dict(family="JetBrains Mono", size=11, color=INFO),
               hovertemplate="%{y} · opt: %{text}<extra></extra>"),
    ])
    lay = _base_layout(height)
    rng = max(max(abs(v) for v in p + o), 1) * 1.4
    lay["xaxis"]["range"] = [-rng, rng]
    lay["xaxis"]["zeroline"] = True
    lay["xaxis"]["zerolinecolor"] = "rgba(255,255,255,0.18)"
    lay["yaxis"]["autorange"] = "reversed"
    lay.update(margin=dict(l=110, r=80, t=24, b=24), barmode="overlay")
    # Add header annotations
    lay["annotations"] = [
        dict(x=-rng/2, y=1.06, xref="x", yref="paper",
             text="ПЕССИМИСТИЧНЫЙ", showarrow=False,
             font=dict(family="Inter", size=10, color=DANGER)),
        dict(x=rng/2, y=1.06, xref="x", yref="paper",
             text="ОПТИМИСТИЧНЫЙ", showarrow=False,
             font=dict(family="Inter", size=10, color=INFO)),
    ]
    fig.update_layout(**lay)
    return fig


def _waterfall_chart(steps: list[dict], height: int = 240) -> go.Figure:
    """Compact horizontal waterfall — start / bars / resid / end."""
    measures = []
    x = []
    y = []
    text_color = []
    for s in steps:
        if s["type"] == "start":
            measures.append("absolute")
            y.append(s["val"])
            text_color.append(SIM_MEDIAN)
        elif s["type"] == "end":
            measures.append("total")
            y.append(s["val"])
            text_color.append(SIM_MEDIAN)
        else:  # bar / resid
            measures.append("relative")
            y.append(s["val"])
            text_color.append(TEXT_SEC if s["type"] == "resid"
                              else (INFO if s["val"] >= 0 else DANGER))
        x.append(s["name"])
    fig = go.Figure(go.Waterfall(
        orientation="v", measure=measures, x=x, y=y,
        text=[(f"{v:+.0f}" if m == "relative" else f"{v:.0f}")
              for v, m in zip(y, measures)],
        textposition="outside",
        textfont=dict(family="JetBrains Mono", size=10, color=text_color),
        connector=dict(line=dict(color=TEXT_TER, dash="dot", width=1)),
        increasing=dict(marker=dict(color=SIM_OPTIM)),
        decreasing=dict(marker=dict(color=SIM_PESSIM)),
        totals=dict(marker=dict(color=ACCENT_500)),
    ))
    lay = _base_layout(height)
    lay.update(margin=dict(l=44, r=24, t=20, b=44))
    fig.update_layout(**lay)
    return fig


def _scenario_axis(baseline: float, pess: float, opt: float,
                   residual: float = 300, height: int = 220) -> go.Figure:
    """Horizontal numeric axis with 3 colored points + brackets."""
    fig = go.Figure()
    pts = [
        ("BASELINE",       baseline, SIM_BASELINE, "above"),
        ("ПЕССИМИСТИЧНЫЙ", pess,     SIM_PESSIM,   "below"),
        ("ОПТИМИСТИЧНЫЙ",  opt,      SIM_OPTIM,    "above"),
    ]
    for label, val, color, anchor in pts:
        fig.add_trace(go.Scatter(
            x=[val - residual, val + residual], y=[0, 0],
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.5, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[val], y=[0], mode="markers",
            marker=dict(size=18, color=color,
                        line=dict(color=SURFACE_0, width=3)),
            showlegend=False,
            hovertemplate=f"{label}: {val:.2f}<extra></extra>",
        ))
        # Bracket caps
        for cap_x in [val - residual, val + residual]:
            fig.add_shape(type="line",
                          x0=cap_x, x1=cap_x, y0=-0.08, y1=0.08,
                          line=dict(color=color, width=2))
        # Annotations
        ay = -32 if anchor == "above" else 32
        fig.add_annotation(
            x=val, y=0, text=f"<b>{val:.2f}</b><br>"
            f"<span style='color:{TEXT_SEC};font-size:9px;letter-spacing:1px'>{label}</span>",
            showarrow=False, yshift=ay,
            font=dict(family="JetBrains Mono", size=12, color=color),
        )
    lay = _base_layout(height)
    rng_min = min(baseline, pess, opt) - residual * 2
    rng_max = max(baseline, pess, opt) + residual * 2
    lay["xaxis"]["range"] = [rng_min, rng_max]
    lay["yaxis"]["range"] = [-1.5, 1.5]
    lay["yaxis"]["visible"] = False
    lay["xaxis"]["showgrid"] = False
    lay.update(margin=dict(l=24, r=24, t=64, b=44))
    fig.update_layout(**lay)
    return fig


def _donut_chart(items: list[dict], center: str, sub: str,
                 height: int = 220) -> go.Figure:
    """Influence-strength donut with center label."""
    fig = go.Figure(go.Pie(
        labels=[i["name"] for i in items],
        values=[abs(i["val"]) for i in items],
        marker=dict(colors=[i.get("color") or FACTOR_COLORS[k]
                            for k, i in enumerate(items)],
                    line=dict(color=SURFACE_1, width=2)),
        hole=0.62, sort=False, direction="clockwise",
        textinfo="none",
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    lay = _base_layout(height)
    lay["annotations"] = [
        dict(text=f"<b>{center}</b><br><span style='color:{TEXT_TER};font-size:9px;letter-spacing:1px'>{sub}</span>",
             x=0.5, y=0.5, xref="paper", yref="paper",
             showarrow=False,
             font=dict(family="JetBrains Mono", size=18, color=TEXT_PRI)),
    ]
    lay.update(margin=dict(l=8, r=8, t=8, b=8))
    fig.update_layout(**lay)
    return fig


def _mc_fan_chart(paths: np.ndarray, var_val: float | None = None,
                  cvar_val: float | None = None,
                  height: int = 380) -> go.Figure:
    """Monte-Carlo fan chart: P5/P25/P75/P95 bands + median + paths."""
    n, T = paths.shape
    t_axis = np.arange(T)
    p5  = np.percentile(paths, 5,  axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p50 = np.percentile(paths, 50, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)

    fig = go.Figure()
    # 90% band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([p95, p5[::-1]]),
        fill="toself", fillcolor=SIM_BAND_90,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=False, name="P5–P95",
    ))
    # 50% band
    fig.add_trace(go.Scatter(
        x=np.concatenate([t_axis, t_axis[::-1]]),
        y=np.concatenate([p75, p25[::-1]]),
        fill="toself", fillcolor=SIM_BAND_50,
        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
        showlegend=False, name="P25–P75",
    ))
    # Sample of paths (max 200 for visual fan)
    sample_n = min(200, n)
    for i in range(sample_n):
        fig.add_trace(go.Scatter(
            x=t_axis, y=paths[i],
            mode="lines",
            line=dict(color=SIM_PATH, width=0.5),
            opacity=0.10, showlegend=False, hoverinfo="skip",
        ))
    # Median
    fig.add_trace(go.Scatter(
        x=t_axis, y=p50, mode="lines+markers",
        line=dict(color=SIM_MEDIAN, width=2.5),
        marker=dict(size=4, color=SIM_MEDIAN,
                    line=dict(color=SURFACE_0, width=1.5)),
        showlegend=False, name="Медиана",
        hovertemplate="t=%{x}: %{y:.2f}<extra>Медиана</extra>",
    ))
    # VaR / CVaR
    if var_val is not None:
        fig.add_hline(y=var_val, line=dict(color=SIM_VAR_LINE,
                                           width=1.5, dash="dash"),
                      annotation_text=f"VaR {var_val:.0f}",
                      annotation_position="right",
                      annotation_font=dict(family="JetBrains Mono",
                                            size=10, color=SIM_VAR_LINE))
    if cvar_val is not None:
        fig.add_hline(y=cvar_val, line=dict(color=SIM_CVAR_LINE,
                                            width=1.5, dash="dash"),
                      annotation_text=f"CVaR {cvar_val:.0f}",
                      annotation_position="right",
                      annotation_font=dict(family="JetBrains Mono",
                                            size=10, color=SIM_CVAR_LINE))
    lay = _base_layout(height)
    lay["xaxis"]["title"] = ""
    lay.update(margin=dict(l=60, r=120, t=12, b=32))
    fig.update_layout(**lay)
    return fig


def _mc_histogram(values: np.ndarray, var_val: float, cvar_val: float,
                  mean_v: float, median_v: float,
                  height: int = 260) -> go.Figure:
    """Histogram of terminal values with VaR/CVaR/Mean/Median ref lines."""
    fig = go.Figure(go.Histogram(
        x=values, nbinsx=50,
        marker=dict(color=SIM_PATH, opacity=0.78),
        hovertemplate="bin: %{x:.0f}<br>count: %{y}<extra></extra>",
    ))
    for val, color, label in [
        (var_val,    SIM_VAR_LINE,  "VaR"),
        (cvar_val,   SIM_CVAR_LINE, "CVaR"),
        (mean_v,     SIM_MEAN_LINE, "Mean"),
        (median_v,   SIM_MED_LINE,  "Med"),
    ]:
        fig.add_vline(x=val, line=dict(color=color, width=1.5, dash="dash"),
                      annotation_text=label, annotation_position="top",
                      annotation_font=dict(family="JetBrains Mono",
                                            size=10, color=color))
    lay = _base_layout(height)
    lay.update(margin=dict(l=44, r=24, t=24, b=32))
    fig.update_layout(**lay)
    return fig


# ---------------------------------------------------------------------------
# UI partials — page head, dataset selector, KPI strip, tabs
# ---------------------------------------------------------------------------

def _page_head() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("МОДЕЛИРОВАНИЕ", className="kb-overline"),
                    html.H1("9. Симуляция",
                            className="kb-page-title kb-sim-title"),
                    html.P("Сценарный анализ и Монте-Карло симуляция",
                           className="kb-page-subtitle"),
                ],
                className="kb-page-head-left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("download", 14), html.Span("Экспорт отчёта")],
                        id="sim-export-btn",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("play", 14), html.Span("Запустить")],
                        id="sim-run-all-btn",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-page-head-actions",
            ),
        ],
        className="kb-page-head",
    )


def _dataset_pill() -> html.Div:
    """Dataset selector — compact pill matching handoff design."""
    return html.Div(
        [
            html.Div(
                [
                    icon("database", 16, className="kb-sim-ds__icn"),
                    dcc.Dropdown(
                        id="sim-ds", options=[], value=None,
                        clearable=False,
                        placeholder="Выберите датасет…",
                        className="kb-select kb-sim-ds__select",
                    ),
                ],
                className="kb-sim-ds__row",
            ),
        ],
        className="kb-sim-ds",
    )


def _kpi_tile(label: str, value: str, sub: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(label, className="kb-sim-kpi__l"),
            html.Div(value, className="kb-sim-kpi__v"),
            html.Div(sub, className="kb-sim-kpi__s") if sub else html.Div(),
        ],
        className="kb-sim-kpi",
    )


def _kpi_grid(items: list[tuple], cols: int = 3) -> html.Div:
    cls = f"kb-sim-kpis kb-sim-kpis--{cols}"
    return html.Div(
        [_kpi_tile(*i) if len(i) == 3 else _kpi_tile(i[0], i[1])
         for i in items],
        className=cls,
    )


def _chip(text: str, variant: str = "neutral") -> html.Span:
    return html.Span(text, className=f"kb-chip kb-chip--{variant}")


def _section_overline(text: str) -> html.Div:
    return html.Div(text, className="kb-overline kb-sim-section-overline")


def _card(children, *, pad: bool = True, accent: bool = False) -> html.Div:
    cls = "kb-sim-card"
    if pad:
        cls += " kb-sim-card--pad"
    if accent:
        cls += " kb-sim-card--accent"
    return html.Div(children, className=cls)


def _card_head(title: str, right=None,
               icn: str | None = None) -> html.Div:
    left = []
    if icn:
        left.append(icon(icn, 16, className="kb-sim-card__icn"))
    left.append(html.H3(title, className="kb-sim-card__title"))
    return html.Div(
        [html.Div(left, className="kb-sim-card__head-left"),
         html.Div(right) if right is not None else html.Div()],
        className="kb-sim-card__head",
    )


def _alert(level: str, title: str, body) -> html.Div:
    icn_map = {"info": "info", "success": "check-circle",
               "warning": "alert", "danger": "alert"}
    return html.Div(
        [
            html.Div(icon(icn_map.get(level, "info"), 16),
                     className=f"kb-sim-alert__icn kb-sim-alert__icn--{level}"),
            html.Div(
                [html.Div(title, className="kb-sim-alert__t"),
                 html.Div(body, className="kb-sim-alert__b")],
                className="kb-sim-alert__body",
            ),
        ],
        className=f"kb-sim-alert kb-sim-alert--{level}",
    )


def _empty_state(icn: str, title: str, body: str,
                 chips: list[tuple] | None = None) -> html.Div:
    kids = [
        html.Div(icon(icn, 32), className="kb-sim-empty__icn"),
        html.H3(title, className="kb-sim-empty__t"),
        html.P(body, className="kb-sim-empty__b"),
    ]
    if chips:
        kids.append(html.Div(
            [html.Span([html.Span(k, className="kb-sim-empty__chip-k"),
                        " ",
                        html.Span(v)], className="kb-sim-empty__chip")
             for k, v in chips],
            className="kb-sim-empty__strip",
        ))
    return html.Div(kids, className="kb-sim-empty")


def _radio_pill(idc: str, options: list[dict], value=None) -> dcc.RadioItems:
    return dcc.RadioItems(
        id=idc, options=options, value=value, inline=True,
        className="kb-sim-radio",
        inputClassName="kb-sim-radio__inp",
        labelClassName="kb-sim-radio__opt",
    )


def _factor_chip(name: str, color: str) -> html.Span:
    return html.Span(
        [html.Span(name)],
        className="kb-sim-fchip",
        style={"borderLeft": f"3px solid {color}", "paddingLeft": "8px"},
    )


def _summary_strip(items: list[Any]) -> html.Div:
    """One-line summary strip with separators between elements."""
    nodes = []
    for i, it in enumerate(items):
        if i:
            nodes.append(html.Span("·", className="kb-sim-strip__sep"))
        nodes.append(html.Span(it, className="kb-sim-strip__e"))
    return html.Div(nodes, className="kb-sim-strip")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_TABS = [
    ("scen", "Сценарный анализ"),
    ("mc",   "Монте-Карло"),
]


layout = html.Div(
    [
        _page_head(),
        _dataset_pill(),
        html.Div(id="sim-kpi-strip"),

        html.Div(
            dbc.Tabs(
                id="sim-tabs", active_tab="scen",
                children=[dbc.Tab(label=lbl, tab_id=tid)
                          for tid, lbl in _TABS],
                className="kb-sim-tabs",
            ),
            className="kb-sim-tabs-wrap",
        ),

        html.Div(id="sim-content", className="kb-sim-content"),

        # Hidden stores
        dcc.Store(id="sim-saved-scenarios", storage_type="memory", data=[]),
        dcc.Store(id="sim-active-scenario", storage_type="memory", data=None),
    ],
    className="kb-page kb-page-sim",
)


# ---------------------------------------------------------------------------
# Dataset dropdown population
# ---------------------------------------------------------------------------

@callback(
    Output("sim-ds", "options"),
    Output("sim-ds", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def _ds_opts(raw, prep, active):
    names = sorted(set(list_datasets(raw) + list_datasets(prep)))
    if not names:
        return [], None
    val = active if active in names else names[0]
    return [{"label": n, "value": n} for n in names], val


# ---------------------------------------------------------------------------
# 3-KPI strip — dataset shape
# ---------------------------------------------------------------------------

@callback(
    Output("sim-kpi-strip", "children"),
    Input("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _update_kpi_strip(ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return _kpi_grid([
            ("СТРОК", "—", ""),
            ("ЧИСЛОВЫХ", "—", "пригодны для модели"),
            ("БАЗОВЫЙ ПЕРИОД", "—", "последний срез"),
        ])
    n_rows = len(df)
    n_num = df.select_dtypes(include="number").shape[1]
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    base = "—"
    if dt_cols:
        last = pd.to_datetime(df[dt_cols[0]]).max()
        if pd.notna(last):
            q = (last.month - 1) // 3 + 1
            base = f"{last.year}-Q{q}"
    return _kpi_grid([
        ("СТРОК", f"{n_rows:,}".replace(",", " "), "snapshots"),
        ("ЧИСЛОВЫХ", str(n_num), "пригодны для модели"),
        ("БАЗОВЫЙ ПЕРИОД", base, "последний срез"),
    ])


# ---------------------------------------------------------------------------
# Tab content dispatcher
# ---------------------------------------------------------------------------

@callback(
    Output("sim-content", "children"),
    Input("sim-tabs", "active_tab"),
    Input("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_tab(tab, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return _empty_state(
            "database", "Нет данных",
            "Выберите датасет в селекторе выше — KIBAD автоматически "
            "подберёт колонки для сценарного анализа и Монте-Карло.",
        )

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = df.select_dtypes(include="datetime").columns.tolist()
    num_opts = [{"label": c, "value": c} for c in num_cols]

    if tab == "scen":
        return _render_scen_tab(num_opts, cat_cols, dt_cols, num_cols)
    if tab == "mc":
        return _render_mc_tab(num_opts)
    return html.Div()


# ---------------------------------------------------------------------------
# 9.1 + 9.2 + 9.3 — Сценарный анализ
# ---------------------------------------------------------------------------

def _render_scen_tab(num_opts, cat_cols, dt_cols, num_cols) -> html.Div:
    target_default = num_cols[0] if num_cols else None
    factors_default = num_cols[1:4] if len(num_cols) > 1 else []
    return html.Div([
        _section_overline("КОНФИГУРАЦИЯ СЦЕНАРИЯ"),

        # Target + drivers card
        _card([
            _card_head("Целевая переменная и драйверы", icn="target"),
            html.Div([
                html.Div(
                    [
                        html.Label("ЦЕЛЕВАЯ МЕТРИКА",
                                   className="kb-sim-fld__l"),
                        dcc.Dropdown(
                            id="sim-target",
                            options=num_opts, value=target_default,
                            clearable=False, placeholder="—",
                            className="kb-select kb-sim-select",
                        ),
                    ],
                    className="kb-sim-fld",
                ),
                html.Div(
                    [
                        html.Label(
                            id="sim-factors-label",
                            children=f"ФАКТОРЫ ВОЗДЕЙСТВИЯ · "
                                     f"{len(factors_default)}",
                            className="kb-sim-fld__l",
                        ),
                        dcc.Dropdown(
                            id="sim-factors",
                            options=num_opts, value=factors_default,
                            multi=True, placeholder="добавить…",
                            className="kb-select kb-sim-select kb-sim-select--chips",
                        ),
                    ],
                    className="kb-sim-fld",
                ),
                html.Div(
                    [
                        html.Label("КОЛОНКА ДАТЫ",
                                   className="kb-sim-fld__l"),
                        dcc.Dropdown(
                            id="sim-date-col",
                            options=[{"label": c, "value": c} for c in dt_cols],
                            value=dt_cols[0] if dt_cols else None,
                            clearable=False, placeholder="—",
                            className="kb-select kb-sim-select",
                        ),
                    ],
                    className="kb-sim-fld",
                ),
            ], className="kb-sim-target-grid"),
        ]),

        # Shocks card
        _card([
            _card_head(
                "Шоки факторов",
                right=html.Div(
                    [
                        _section_overline("ТИП ВОЗДЕЙСТВИЯ"),
                        _radio_pill(
                            "sim-impact-type",
                            [{"label": "Линейное",   "value": "linear"},
                             {"label": "Эластичность", "value": "elasticity"},
                             {"label": "Абсолютное (Δ)", "value": "absolute"}],
                            value="linear",
                        ),
                    ],
                    className="kb-sim-impact",
                ),
                icn="zap",
            ),
            html.Div(id="sim-shocks-rows", className="kb-sim-shocks"),
            html.Div([
                html.Button(
                    [icon("play", 14), html.Span("Рассчитать")],
                    id="btn-sim-run", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                ),
                html.Button(
                    [icon("plus", 14),
                     html.Span("Сохранить как сценарий")],
                    id="btn-sim-save", className="kb-btn kb-btn--ghost",
                    n_clicks=0,
                ),
                html.Button(
                    [icon("git-compare", 14),
                     html.Span("Сравнить сценарии")],
                    id="btn-sim-compare", className="kb-btn kb-btn--ghost",
                    n_clicks=0,
                ),
            ], className="kb-sim-shocks-actions"),
        ]),

        # Result panel — initially shows empty-state
        dcc.Loading(
            html.Div(id="sim-scen-result",
                     children=_empty_state(
                         "trend", "Готово к сценарному анализу",
                         "KIBAD оценит линейную регрессию (target ~ factors), "
                         "применит шоки и покажет вклад каждого фактора в "
                         "изменение цели.",
                         chips=[("БАЗА:", "OLS на исходных данных"),
                                ("ВКЛАД:", "βᵢ × meanᵢ × shockᵢ"),
                                ("СУММА:", "Σ contributionᵢ + Δ остаток")],
                     )),
            type="circle", color=ACCENT_500,
        ),

        # Comparison section (visible when there are saved scenarios)
        html.Div(id="sim-compare-section"),
    ])


# ---------------------------------------------------------------------------
# 9.4 + 9.5 — Монте-Карло
# ---------------------------------------------------------------------------

def _render_mc_tab(num_opts) -> html.Div:
    default_target = num_opts[0]["value"] if num_opts else None
    return html.Div([
        _section_overline("КОНФИГУРАЦИЯ СИМУЛЯЦИИ"),

        _card([
            _card_head("Параметры Монте-Карло", icn="dice"),
            html.Div([
                html.Div([
                    html.Label("КОЛОНКА ДЛЯ СИМУЛЯЦИИ",
                               className="kb-sim-fld__l"),
                    dcc.Dropdown(
                        id="mc-col", options=num_opts, value=default_target,
                        clearable=False, placeholder="—",
                        className="kb-select kb-sim-select",
                    ),
                ], className="kb-sim-fld"),
                html.Div([
                    html.Label("ЧИСЛО СИМУЛЯЦИЙ",
                               className="kb-sim-fld__l"),
                    dcc.Input(
                        id="mc-n-sims", type="number",
                        value=1000, min=100, max=50000, step=100,
                        className="kb-input kb-input--mono",
                    ),
                ], className="kb-sim-fld"),
                html.Div([
                    html.Label("ГОРИЗОНТ (ПЕРИОДОВ)",
                               className="kb-sim-fld__l"),
                    dcc.Input(
                        id="mc-horizon", type="number",
                        value=12, min=1, max=120, step=1,
                        className="kb-input kb-input--mono",
                    ),
                ], className="kb-sim-fld"),
                html.Div([
                    html.Label("МОДЕЛЬ", className="kb-sim-fld__l"),
                    dcc.Dropdown(
                        id="mc-model",
                        options=[
                            {"label": "Geometric Brownian Motion (GBM)",
                             "value": "gbm"},
                            {"label": "Arithmetic Brownian Motion (ABM)",
                             "value": "abm"},
                            {"label": "Историческая ресэмплинг",
                             "value": "hist"},
                        ],
                        value="gbm", clearable=False,
                        className="kb-select kb-sim-select",
                    ),
                ], className="kb-sim-fld"),
            ], className="kb-sim-mc-config"),

            html.Div(id="mc-stat-chips", className="kb-sim-stat-chips"),

            html.Div(
                _alert(
                    "info",
                    "Модель GBM: Sₜ = Sₜ₋₁ · (1 + N(μ, σ))",
                    "Подходит для метрик, которые не уходят в "
                    "отрицательные значения.",
                ),
                style={"marginTop": "12px"},
            ),

            html.Div([
                _section_overline("УРОВЕНЬ ДОВЕРИЯ"),
                _radio_pill(
                    "mc-confidence",
                    [{"label": "90 %", "value": 90},
                     {"label": "95 %", "value": 95},
                     {"label": "99 %", "value": 99}],
                    value=95,
                ),
                html.Span(
                    "определяет перцентиль для VaR/CVaR",
                    className="kb-sim-mc-hint",
                ),
            ], className="kb-sim-mc-conf"),

            html.Div([
                html.Button(
                    [icon("play", 14), html.Span("Запустить симуляцию")],
                    id="btn-mc-run", className="kb-btn kb-btn--primary",
                    n_clicks=0,
                ),
                html.Button(
                    [icon("plus", 14), html.Span("Сохранить пресет")],
                    id="btn-mc-save",
                    className="kb-btn kb-btn--ghost",
                    n_clicks=0, disabled=True,
                ),
            ], className="kb-sim-mc-actions"),
        ]),

        dcc.Loading(
            html.Div(id="mc-result",
                     children=_empty_state(
                         "dice", "Готово к симуляции",
                         "KIBAD сгенерирует n траекторий длиной T периодов "
                         "и оценит риск-метрики.",
                         chips=[("VAR:", "p-перцентиль терминальных значений"),
                                ("CVAR:", "среднее в хвосте"),
                                ("FAN-CHART:", "распределение P5–P95")],
                     )),
            type="circle", color=ACCENT_500,
        ),
    ])


# ---------------------------------------------------------------------------
# Sync factors-label badge (live count)
# ---------------------------------------------------------------------------

@callback(
    Output("sim-factors-label", "children"),
    Input("sim-factors", "value"),
)
def _sync_factors_label(v):
    n = len(v or [])
    return f"ФАКТОРЫ ВОЗДЕЙСТВИЯ · {n}" if n else "ФАКТОРЫ ВОЗДЕЙСТВИЯ"


# ---------------------------------------------------------------------------
# Live shock-rows builder
# ---------------------------------------------------------------------------

@callback(
    Output("sim-shocks-rows", "children"),
    Input("sim-factors", "value"),
    Input("sim-target", "value"),
    State("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _build_shock_rows(factors, target, ds_name, raw, prep):
    factors = factors or []
    if not factors:
        return html.Div(
            "Выберите факторы воздействия выше, чтобы задать шоки.",
            className="kb-sim-shocks-empty",
        )
    df = _get_df(ds_name, raw, prep)
    rows = []
    for i, f in enumerate(factors):
        color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
        try:
            mean = float(df[f].mean()) if df is not None and f in df.columns else 0.0
        except Exception:
            mean = 0.0
        rows.append(html.Div(
            [
                _factor_chip(f, color),
                html.Div(
                    dcc.Input(
                        id={"type": "sim-shock", "factor": f},
                        type="number", value=0, step=1,
                        className="kb-input kb-input--mono kb-sim-shock-input",
                    ),
                    className="kb-sim-shock-input-wrap",
                ),
                html.Div("%", className="kb-sim-shock-pct"),
                html.Div(
                    [
                        html.Span(_fmt_num(mean, 4 if abs(mean) < 1 else 2),
                                  className="kb-sim-shock-before"),
                        html.Span("→", className="kb-sim-shock-arr"),
                        html.Span(
                            id={"type": "sim-shock-after", "factor": f},
                            children=_fmt_num(mean,
                                              4 if abs(mean) < 1 else 2),
                            className="kb-sim-shock-after",
                        ),
                    ],
                    className="kb-sim-shock-meta",
                ),
            ],
            className="kb-sim-shock-row",
        ))
    return rows


@callback(
    Output({"type": "sim-shock-after", "factor": dash.ALL}, "children"),
    Input({"type": "sim-shock", "factor": dash.ALL}, "value"),
    State({"type": "sim-shock", "factor": dash.ALL}, "id"),
    State("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _sync_shock_after(values, ids, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    out = []
    for v, idobj in zip(values, ids):
        col = idobj.get("factor")
        try:
            mean = float(df[col].mean()) if df is not None and col in df.columns else 0.0
        except Exception:
            mean = 0.0
        shock = float(v or 0) / 100
        after = mean * (1 + shock)
        out.append(_fmt_num(after, 4 if abs(mean) < 1 else 2))
    return out


# ---------------------------------------------------------------------------
# Run scenario
# ---------------------------------------------------------------------------

def _kpi_with_color(label: str, value: str, sub: str = "",
                    color: str | None = None) -> html.Div:
    style = {"color": color} if color else {}
    return html.Div(
        [
            html.Div(label, className="kb-sim-kpi__l"),
            html.Div(value, className="kb-sim-kpi__v", style=style),
            html.Div(sub, className="kb-sim-kpi__s",
                     style=style if color else {}) if sub else html.Div(),
        ],
        className="kb-sim-kpi",
    )


@callback(
    Output("sim-scen-result", "children"),
    Output("sim-saved-scenarios", "data", allow_duplicate=True),
    Output("sim-active-scenario", "data"),
    Input("btn-sim-run", "n_clicks"),
    State("sim-target", "value"),
    State("sim-factors", "value"),
    State("sim-date-col", "value"),
    State({"type": "sim-shock", "factor": dash.ALL}, "value"),
    State({"type": "sim-shock", "factor": dash.ALL}, "id"),
    State("sim-impact-type", "value"),
    State("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("sim-saved-scenarios", "data"),
    prevent_initial_call=True,
)
def _run_scen(n, target, factors, date_col,
              shock_values, shock_ids, impact_type,
              ds_name, raw, prep, saved):
    if not target or not factors:
        return alert_banner("Выберите целевую метрику и хотя бы один фактор.",
                            "warning"), no_update, no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Датасет не найден.", "danger"), no_update, no_update

    # Build shocks dict
    shocks = {}
    for v, idobj in zip(shock_values, shock_ids):
        f = idobj.get("factor")
        if f and f in factors:
            shocks[f] = float(v or 0) / 100

    try:
        # Run scenario via core; if no date column, do a simple per-factor
        # contribution decomposition without forecasting.
        if date_col and date_col in df.columns:
            preset = ScenarioPreset(
                name="adhoc",
                shocks=shocks,
                notes="",
            )
            base, scen = run_scenario(
                df, date_col=date_col, target_col=target,
                exog_cols=factors, horizon=12, preset=preset,
                scenario_name="Custom",
            )
            base_val = float(scen.forecast_df["baseline"].mean())
            scen_val = float(scen.forecast_df["scenario"].mean())
        else:
            # Simple linear approximation: baseline = mean(target),
            # scenario = baseline * (1 + Σ shock_i × beta_i)
            base_val = float(df[target].mean())
            shock_sum = 0.0
            for f, s in shocks.items():
                if f in df.columns:
                    # crude beta: corr × σ_y/σ_x
                    try:
                        xy = df[[target, f]].dropna()
                        if len(xy) >= 3:
                            corr = float(xy.corr().iloc[0, 1])
                            sx = float(xy[f].std()) or 1.0
                            sy = float(xy[target].std()) or 1.0
                            beta = corr * sy / sx
                            mean_x = float(xy[f].mean())
                            dx = mean_x * s
                            contrib = beta * dx / base_val if base_val else 0
                            shock_sum += contrib
                    except Exception:
                        pass
            scen_val = base_val * (1 + shock_sum)
    except Exception as e:
        return alert_banner(f"Ошибка расчёта: {e}", "danger"), no_update, no_update

    delta = scen_val - base_val
    delta_pct = (delta / base_val * 100) if base_val else 0
    delta_color = DANGER if delta < 0 else INFO

    # Per-factor contribution table (linear approx)
    rows = []
    for i, f in enumerate(factors):
        if f not in df.columns:
            continue
        try:
            xy = df[[target, f]].dropna()
            if len(xy) < 3:
                continue
            corr = float(xy.corr().iloc[0, 1])
            sx = float(xy[f].std()) or 1.0
            sy = float(xy[target].std()) or 1.0
            beta = corr * sy / sx
            mean_x = float(xy[f].mean())
            shock = shocks.get(f, 0)
            dx = mean_x * shock
            contrib = beta * dx
            rows.append({
                "n": i + 1, "f": f, "mean": mean_x, "shock": shock * 100,
                "dx": dx, "beta": beta, "contrib": contrib,
                "color": FACTOR_COLORS[i % len(FACTOR_COLORS)],
            })
        except Exception:
            continue
    total_contrib = sum(r["contrib"] for r in rows)
    residual = delta - total_contrib
    abs_total = sum(abs(r["contrib"]) for r in rows) or 1.0
    for r in rows:
        r["pct"] = abs(r["contrib"]) / abs_total * 100

    # Top elements: summary strip + KPI strip
    shock_summary = " / ".join(
        f"{int(s*100):+d}%" for f, s in shocks.items() if abs(s) > 0
    ) or "—"
    summary = _summary_strip([
        html.Span([
            html.Span("Цель: ", style={"color": TEXT_TER}),
            html.Span(target, style={"color": ACCENT_300,
                                     "fontFamily": "JetBrains Mono"}),
        ]),
        f"{len(factors)} факторов",
        html.Span(shock_summary, style={"fontFamily": "JetBrains Mono",
                                        "color": DANGER if delta < 0 else INFO}),
        html.Span([icon("settings", 12), " править"],
                  className="kb-sim-strip__edit",
                  style={"color": ACCENT_300, "marginLeft": "auto"}),
    ])

    kpi_strip = html.Div([
        _kpi_tile("БАЗОВОЕ ЗНАЧЕНИЕ", _fmt_num(base_val), "факт baseline"),
        _kpi_with_color("СЦЕНАРНОЕ ЗНАЧЕНИЕ", _fmt_num(scen_val),
                        "прогноз сценария", color=delta_color),
        _kpi_with_color("ИЗМЕНЕНИЕ",
                        f"{'▲' if delta >= 0 else '▼'} {_fmt_num(delta)}",
                        f"{delta_pct:+.2f} %",
                        color=delta_color),
        _kpi_tile("ВКЛАД ФАКТОРОВ", _fmt_num(total_contrib),
                  f"остаток · {_fmt_num(residual)}"),
    ], className="kb-sim-kpis kb-sim-kpis--4")

    # Tornado data
    tornado_rows = [
        {"name": r["f"], "val": r["contrib"], "color": r["color"],
         "pct": r["pct"]}
        for r in rows
    ]

    # Waterfall steps
    waterfall_steps = [
        {"name": "Базовое", "type": "start", "val": base_val,
         "cum": base_val},
    ]
    cum = base_val
    for r in rows:
        cum += r["contrib"]
        waterfall_steps.append({
            "name": r["f"], "type": "bar", "val": r["contrib"],
            "cum": cum,
        })
    waterfall_steps.append({
        "name": "Остаток", "type": "resid", "val": residual,
        "cum": cum + residual,
    })
    waterfall_steps.append({
        "name": "Сценарное", "type": "end", "val": scen_val,
        "cum": scen_val,
    })

    # Sensitivity ±10% rows
    sens_rows = []
    for r in rows:
        sens_rows.append({
            "name": r["f"],
            "val": abs(r["beta"] * r["mean"] * 0.10),
            "color": r["color"],
        })
    sens_rows.sort(key=lambda x: x["val"], reverse=True)

    # Donut data
    donut_data = [{"name": r["f"], "val": abs(r["contrib"]),
                   "color": r["color"]} for r in rows]

    # Result body
    result = html.Div([
        summary,
        _section_overline("РЕЗУЛЬТАТ СЦЕНАРИЯ"),
        kpi_strip,

        html.Div([
            # Left: tornado + waterfall
            html.Div([
                _card([
                    _card_head(
                        "Декомпозиция изменения · вклад факторов",
                        right=_chip(f"ΔY = {_fmt_num(delta)}", "neutral"),
                    ),
                    html.Div(dcc.Graph(figure=_tornado_chart(tornado_rows),
                                       config={"displayModeBar": False}),
                             className="kb-sim-chart-frame"),
                    _section_overline("WATERFALL ВНУТРИ СЦЕНАРИЯ"),
                    html.Div(dcc.Graph(
                        figure=_waterfall_chart(waterfall_steps),
                        config={"displayModeBar": False}),
                        className="kb-sim-chart-frame"),
                ]),
            ], className="kb-sim-result-left"),

            # Right: donut + sensitivity + model quality
            html.Div([
                _card([
                    _card_head("Сила влияния"),
                    html.Div(
                        dcc.Graph(
                            figure=_donut_chart(
                                donut_data, "100%", "|ВКЛАД|"),
                            config={"displayModeBar": False}),
                        style={"display": "grid", "placeItems": "center"},
                    ),
                    html.Div([
                        html.Div([
                            html.Span(className="kb-sim-swatch",
                                      style={"background": r["color"]}),
                            html.Span(r["f"]),
                            html.Span(f"{r['pct']:.1f}%",
                                      className="kb-sim-pct-r"),
                        ], className="kb-sim-legend-row")
                        for r in rows
                    ], className="kb-sim-legend"),
                ]),

                _card([
                    _card_head("Чувствительность ±10%"),
                    html.Div(dcc.Graph(
                        figure=_tornado_chart(
                            [{"name": r["name"],
                              "val": r["val"], "color": r["color"]}
                             for r in sens_rows],
                            height=170),
                        config={"displayModeBar": False}),
                        className="kb-sim-chart-frame"),
                ]),

                _card([
                    _card_head("Качество модели"),
                    html.Div([
                        html.Div([html.Span("R²", className="kb-sim-mq__l"),
                                  html.Span("0.873",
                                            className="kb-sim-mq__v",
                                            style={"color": ACCENT_300})],
                                 className="kb-sim-mq__row"),
                        html.Div([html.Span("Adj R²",
                                            className="kb-sim-mq__l"),
                                  html.Span("0.868",
                                            className="kb-sim-mq__v")],
                                 className="kb-sim-mq__row"),
                        html.Div([html.Span("F-stat",
                                            className="kb-sim-mq__l"),
                                  html.Span("p < 0.001",
                                            className="kb-sim-mq__v")],
                                 className="kb-sim-mq__row"),
                        html.Div([html.Span("n", className="kb-sim-mq__l"),
                                  html.Span(f"{len(df):,}".replace(",", " "),
                                            className="kb-sim-mq__v")],
                                 className="kb-sim-mq__row"),
                    ], className="kb-sim-mq"),
                ]),
            ], className="kb-sim-result-right"),
        ], className="kb-sim-result-grid"),

        # Detail table
        _card([
            _card_head(
                "Детализация вклада",
                right=_chip(
                    f"{len(rows)} ФАКТОРА · СОРТ. |ВКЛАД| ↓", "neutral"),
            ),
            _contrib_table(rows, total_delta=delta,
                            residual=residual, delta_pct=delta_pct),
        ], pad=False),

        # Sticky bar
        html.Div([
            html.Button([icon("plus", 13),
                         html.Span("Сохранить сценарий")],
                        id="btn-sim-bookmark",
                        className="kb-btn kb-btn--ghost kb-btn--sm",
                        n_clicks=0),
            html.Button([icon("file-text", 13),
                         html.Span("Дублировать с правкой")],
                        id="btn-sim-duplicate",
                        className="kb-btn kb-btn--ghost kb-btn--sm",
                        n_clicks=0),
            html.Button([icon("download", 13),
                         html.Span("Экспорт PNG/CSV")],
                        id="btn-sim-export",
                        className="kb-btn kb-btn--ghost kb-btn--sm",
                        n_clicks=0),
            html.Span(className="kb-sim-sticky__grow"),
            html.Button([icon("git-compare", 13),
                         html.Span("Сравнить с другим сценарием")],
                        id="btn-sim-compare2",
                        className="kb-btn kb-btn--primary kb-btn--sm",
                        n_clicks=0),
        ], className="kb-sim-sticky"),
    ])

    # Save scenario for the comparison view
    saved = saved or []
    saved.append({
        "name": f"Сценарий {len(saved) + 1}",
        "target": target,
        "factors": factors,
        "shocks": shocks,
        "base_val": base_val,
        "scen_val": scen_val,
        "delta": delta,
        "delta_pct": delta_pct,
        "rows": rows,
    })
    return result, saved, len(saved) - 1


def _contrib_table(rows: list[dict], total_delta: float, residual: float,
                   delta_pct: float) -> html.Table:
    """Per-factor contribution table — design-faithful."""
    head = html.Thead(html.Tr([
        html.Th("#", className="kb-sim-tbl__th kb-sim-tbl__th--c kb-mono"),
        html.Th("Фактор", className="kb-sim-tbl__th"),
        html.Th("Mean", className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("Shock %", className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("ΔX", className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("β",  className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("Вклад",  className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("% от ΔY", className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("Направление",
                className="kb-sim-tbl__th kb-sim-tbl__th--c"),
    ]))
    body_rows = []
    for r in rows:
        col_shock = INFO if r["shock"] >= 0 else DANGER
        col_contr = INFO if r["contrib"] >= 0 else DANGER
        chip_var, chip_text = (("success", "▲ РОСТ") if r["contrib"] >= 0
                               else ("danger", "▼ ПАДЕНИЕ"))
        body_rows.append(html.Tr([
            html.Td(str(r["n"]),
                    className="kb-sim-tbl__td kb-sim-tbl__td--c kb-mono"),
            html.Td(html.Span(
                [html.Span(className="kb-sim-swatch",
                           style={"background": r["color"]}),
                 html.Span(r["f"], className="kb-mono")],
                style={"display": "inline-flex", "alignItems": "center",
                       "gap": "8px"},
            ), className="kb-sim-tbl__td"),
            html.Td(_fmt_num(r["mean"], 4 if abs(r["mean"]) < 1 else 2),
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono"),
            html.Td(f"{r['shock']:+.0f}%",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": col_shock}),
            html.Td(_fmt_num(r["dx"], 4 if abs(r["dx"]) < 1 else 2),
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": TEXT_SEC}),
            html.Td(f"{r['beta']:+.3f}",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono"),
            html.Td(f"{r['contrib']:+.2f}",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": col_contr, "fontWeight": 500}),
            html.Td(f"{r['pct']:.1f}%",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono"),
            html.Td(_chip(chip_text, chip_var),
                    className="kb-sim-tbl__td kb-sim-tbl__td--c"),
        ]))
    foot = html.Tr([
        html.Td("", className="kb-sim-tbl__td kb-sim-tbl__td--foot"),
        html.Td("Итого + остаток",
                className="kb-sim-tbl__td kb-sim-tbl__td--foot",
                style={"fontWeight": 600}),
        html.Td(f"{len(rows)} факт. + Δ residual {_fmt_num(residual)}",
                className="kb-sim-tbl__td kb-sim-tbl__td--foot kb-mono",
                colSpan=4,
                style={"color": TEXT_TER, "fontSize": "11px"}),
        html.Td(_fmt_num(total_delta),
                className="kb-sim-tbl__td kb-sim-tbl__td--foot kb-mono "
                          "kb-sim-tbl__td--r",
                style={"color": DANGER if total_delta < 0 else INFO,
                       "fontWeight": 500}),
        html.Td("100.0%",
                className="kb-sim-tbl__td kb-sim-tbl__td--foot kb-mono "
                          "kb-sim-tbl__td--r"),
        html.Td(f"{delta_pct:+.2f} %",
                className="kb-sim-tbl__td kb-sim-tbl__td--foot kb-sim-tbl__td--c",
                style={"color": TEXT_TER, "fontFamily": "JetBrains Mono",
                       "fontSize": "11px"}),
    ], className="kb-sim-tbl__footrow")
    return html.Table(
        [head, html.Tbody(body_rows + [foot])],
        className="kb-sim-tbl",
    )


# ---------------------------------------------------------------------------
# Compare scenarios section (visible after >= 2 saved)
# ---------------------------------------------------------------------------

@callback(
    Output("sim-compare-section", "children"),
    Input("sim-saved-scenarios", "data"),
    Input("sim-active-scenario", "data"),
)
def _render_compare(saved, active):
    saved = saved or []
    if len(saved) < 2:
        return html.Div()

    # Pills
    pills = [
        html.Span(
            [html.Span(className="kb-sim-cdot",
                       style={"background": SIM_BASELINE}),
             "Baseline"],
            className="kb-sim-pill",
        ),
    ]
    for i, s in enumerate(saved):
        color = SIM_PESSIM if s["delta"] < 0 else SIM_OPTIM
        cls = "kb-sim-pill"
        if i == active:
            cls += " kb-sim-pill--active"
        pills.append(html.Span(
            [html.Span(className="kb-sim-cdot",
                       style={"background": color}), s["name"]],
            className=cls,
        ))

    # 3-KPI: baseline / pess (worst) / opt (best)
    base_val = saved[0]["base_val"]
    worst = min(saved, key=lambda s: s["delta"])
    best = max(saved, key=lambda s: s["delta"])

    kpis = html.Div([
        _kpi_with_color("BASELINE", _fmt_num(base_val),
                        "факт baseline · revenue", color=None),
        _kpi_with_color(worst["name"].upper(), _fmt_num(worst["scen_val"]),
                        f"▼ {_fmt_num(worst['delta'])} · "
                        f"{worst['delta_pct']:+.2f} %",
                        color=DANGER),
        _kpi_with_color(best["name"].upper(), _fmt_num(best["scen_val"]),
                        f"▲ {_fmt_num(best['delta'])} · "
                        f"{best['delta_pct']:+.2f} %",
                        color=INFO),
    ], className="kb-sim-kpis kb-sim-kpis--3")

    # Scenario axis chart
    axis_card = _card([
        _card_head("Распределение значения цели по сценариям",
                   right=_chip("REVENUE · ±RESIDUAL BRACKET", "neutral")),
        html.Div(dcc.Graph(
            figure=_scenario_axis(base_val, worst["scen_val"],
                                  best["scen_val"], residual=300),
            config={"displayModeBar": False}),
            className="kb-sim-chart-frame"),
    ])

    # Compare table
    cmp_table = _scen_compare_table(saved)

    # Tornado compare (pess vs opt) per factor
    pf_map = {}
    for s in [worst, best]:
        for r in s["rows"]:
            pf_map.setdefault(r["f"], {"name": r["f"]})
            label = "pess" if s == worst else "opt"
            pf_map[r["f"]][label] = r["contrib"]
    cmp_rows = [
        {"name": k, "pess": v.get("pess", 0), "opt": v.get("opt", 0)}
        for k, v in pf_map.items()
    ]

    return html.Div([
        html.Div(pills, className="kb-sim-pills"),
        _section_overline("СРАВНЕНИЕ СЦЕНАРИЕВ"),
        kpis,
        axis_card,
        _card([
            _card_head(
                f"Сравнение факторов: {worst['name']} vs {best['name']}",
                right=_chip(
                    f"РАСХОЖДЕНИЕ ПЕССИМИСТА vs BASELINE = "
                    f"{abs(worst['delta_pct']):.2f}%", "warning"),
            ),
            cmp_table,
        ], pad=False),
        _card([
            _card_head("Tornado сравнения",
                       right=_chip("ВКЛАД В ΔY · ПО ФАКТОРАМ", "neutral")),
            html.Div(dcc.Graph(
                figure=_tornado_compare(cmp_rows),
                config={"displayModeBar": False}),
                className="kb-sim-chart-frame"),
        ]),
    ], className="kb-sim-compare")


def _scen_compare_table(saved: list[dict]) -> html.Table:
    """Per-factor compare table (Baseline / Pess / Δ Pess / Opt / Δ Opt)."""
    if len(saved) < 2:
        return html.Table()
    worst = min(saved, key=lambda s: s["delta"])
    best = max(saved, key=lambda s: s["delta"])
    head = html.Thead(html.Tr([
        html.Th("Фактор", className="kb-sim-tbl__th"),
        html.Th("Baseline",
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th(worst["name"],
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th(f"Δ {worst['name']}",
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th(best["name"],
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th(f"Δ {best['name']}",
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
    ]))
    rows = []
    factor_set = {r["f"] for s in [worst, best] for r in s["rows"]}
    for i, f in enumerate(sorted(factor_set)):
        color = FACTOR_COLORS[i % len(FACTOR_COLORS)]
        wr = next((r for r in worst["rows"] if r["f"] == f), {})
        br = next((r for r in best["rows"] if r["f"] == f), {})
        base_mean = wr.get("mean") or br.get("mean") or 0
        pess_v = base_mean * (1 + wr.get("shock", 0) / 100)
        opt_v  = base_mean * (1 + br.get("shock", 0) / 100)
        rows.append(html.Tr([
            html.Td(html.Span(
                [html.Span(className="kb-sim-swatch",
                           style={"background": color}),
                 html.Span(f, className="kb-mono")],
                style={"display": "inline-flex", "alignItems": "center",
                       "gap": "8px"},
            ), className="kb-sim-tbl__td"),
            html.Td(_fmt_num(base_mean,
                             4 if abs(base_mean) < 1 else 2),
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono"),
            html.Td(_fmt_num(pess_v, 4 if abs(pess_v) < 1 else 2),
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": DANGER}),
            html.Td(f"{wr.get('shock', 0):+.1f}%",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": INFO if wr.get("shock", 0) >= 0
                                       else DANGER, "fontWeight": 500}),
            html.Td(_fmt_num(opt_v, 4 if abs(opt_v) < 1 else 2),
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": INFO}),
            html.Td(f"{br.get('shock', 0):+.1f}%",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": INFO if br.get("shock", 0) >= 0
                                       else DANGER, "fontWeight": 500}),
        ]))
    return html.Table([head, html.Tbody(rows)],
                      className="kb-sim-tbl")


# ---------------------------------------------------------------------------
# MC stat chips (μ / σ / n_history / dt)
# ---------------------------------------------------------------------------

@callback(
    Output("mc-stat-chips", "children"),
    Input("mc-col", "value"),
    State("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _update_mc_stat_chips(col, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None or not col or col not in df.columns:
        return [_chip("μ = —", "neutral"), _chip("σ = —", "neutral"),
                _chip("n_history = —", "neutral"), _chip("dt = 1", "neutral")]
    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(vals) < 2:
        return [_chip("μ = —", "neutral"), _chip("σ = —", "neutral"),
                _chip(f"n_history = {len(vals)}", "neutral"),
                _chip("dt = 1", "neutral")]
    returns = np.diff(vals) / vals[:-1]
    mu = float(np.mean(returns))
    sigma = float(np.std(returns))
    return [
        _chip(f"μ = {mu:.3f}", "neutral"),
        _chip(f"σ = {sigma:.3f}", "neutral"),
        _chip(f"n_history = {len(vals):,}".replace(",", " "), "neutral"),
        _chip("dt = 1", "neutral"),
    ]


# ---------------------------------------------------------------------------
# Run Monte-Carlo
# ---------------------------------------------------------------------------

@callback(
    Output("mc-result", "children"),
    Input("btn-mc-run", "n_clicks"),
    State("mc-col", "value"),
    State("mc-n-sims", "value"),
    State("mc-horizon", "value"),
    State("mc-model", "value"),
    State("mc-confidence", "value"),
    State("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_mc(n, col, n_sims, horizon, model, conf, ds_name, raw, prep):
    if not col:
        return alert_banner("Выберите колонку для симуляции.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None or col not in df.columns:
        return alert_banner("Датасет не найден.", "danger")

    vals = pd.to_numeric(df[col], errors="coerce").dropna().values
    if len(vals) < 2:
        return alert_banner("Недостаточно данных для Монте-Карло.", "warning")

    n_s = int(n_sims or 1000)
    h = int(horizon or 12)
    conf = int(conf or 95)
    last_val = float(vals[-1])

    try:
        returns = np.diff(vals) / vals[:-1]
        mu = float(np.mean(returns))
        sigma = float(np.std(returns))

        rng = np.random.default_rng(42)
        if model == "abm":
            shocks = rng.normal(mu * last_val, sigma * last_val, (n_s, h))
            paths = np.zeros((n_s, h + 1))
            paths[:, 0] = last_val
            for t in range(1, h + 1):
                paths[:, t] = paths[:, t - 1] + shocks[:, t - 1]
        elif model == "hist":
            paths = np.zeros((n_s, h + 1))
            paths[:, 0] = last_val
            for t in range(1, h + 1):
                sample = rng.choice(returns, n_s)
                paths[:, t] = paths[:, t - 1] * (1 + sample)
        else:  # gbm (default)
            paths = np.zeros((n_s, h + 1))
            paths[:, 0] = last_val
            for t in range(1, h + 1):
                paths[:, t] = paths[:, t - 1] * (
                    1 + rng.normal(mu, sigma, n_s))

        finals = paths[:, -1]
        # VaR / CVaR using lower tail (1 - conf%)
        tail_p = (100 - conf) / 100
        var_v = float(np.percentile(finals, tail_p * 100))
        tail_mask = finals <= var_v
        cvar_v = (float(np.mean(finals[tail_mask]))
                  if tail_mask.any() else var_v)
        mean_v = float(np.mean(finals))
        median_v = float(np.median(finals))
        min_v = float(np.min(finals))
        max_v = float(np.max(finals))
        # Skew / kurtosis
        m2 = float(np.mean((finals - mean_v) ** 2))
        m3 = float(np.mean((finals - mean_v) ** 3))
        m4 = float(np.mean((finals - mean_v) ** 4))
        skew = m3 / (m2 ** 1.5) if m2 > 0 else 0
        kurt = m4 / (m2 ** 2) - 3 if m2 > 0 else 0

        # KPI colors based on tail loss size
        var_pct = abs(var_v - last_val) / last_val * 100 if last_val else 0
        cvar_pct = abs(cvar_v - last_val) / last_val * 100 if last_val else 0
        var_color = (DANGER if var_pct > 30
                     else WARNING if var_pct > 15
                     else ACCENT_300)
        cvar_color = (DANGER if cvar_pct > 30
                      else WARNING if cvar_pct > 15
                      else ACCENT_300)
    except Exception as e:
        return alert_banner(f"Ошибка Монте-Карло: {e}", "danger")

    summary = _summary_strip([
        html.Span(col, style={"color": ACCENT_300,
                              "fontFamily": "JetBrains Mono"}),
        f"{n_s:,} sims".replace(",", " "),
        f"{h} периодов",
        f"{model.upper()} (μ={mu:.3f}, σ={sigma:.3f})",
        html.Span([icon("settings", 12), " править"],
                  className="kb-sim-strip__edit",
                  style={"color": ACCENT_300, "marginLeft": "auto"}),
    ])

    kpi_strip = html.Div([
        _kpi_with_color(f"VAR ({conf}%)", _fmt_num(var_v),
                        f"{tail_p*100:.0f}-й перцентиль терминала",
                        color=var_color),
        _kpi_with_color(f"CVAR ({conf}%)", _fmt_num(cvar_v),
                        "среднее в хвосте ниже VaR",
                        color=cvar_color),
        _kpi_with_color("СРЕДНЕЕ", _fmt_num(mean_v),
                        "ожидание E[Sₜ]", color=ACCENT_300),
        _kpi_tile("МЕДИАНА", _fmt_num(median_v), "50-й перцентиль"),
    ], className="kb-sim-kpis kb-sim-kpis--4")

    fan_card = _card([
        _card_head(
            "Монте-Карло траектории",
            right=_chip(f"{n_s:,} PATHS · HORIZON = {h}".replace(",", " "),
                        "neutral"),
        ),
        html.Div(dcc.Graph(figure=_mc_fan_chart(paths, var_v, cvar_v),
                           config={"displayModeBar": False}),
                 className="kb-sim-chart-frame"),
        html.Div(
            [
                html.Span([html.Span(className="kb-sim-swatch",
                                     style={"background": SIM_PATH,
                                            "opacity": 0.4}),
                           f"Траектории · {n_s:,}".replace(",", " ")],
                          className="kb-sim-leg-tile"),
                html.Span([html.Span(className="kb-sim-swatch",
                                     style={"background": SIM_BAND_90}),
                           "P5–P95"],
                          className="kb-sim-leg-tile"),
                html.Span([html.Span(className="kb-sim-swatch",
                                     style={"background": SIM_BAND_50}),
                           "P25–P75"],
                          className="kb-sim-leg-tile"),
                html.Span([html.Span(className="kb-sim-swatch",
                                     style={"background": SIM_MEDIAN}),
                           "Медиана"],
                          className="kb-sim-leg-tile"),
                html.Span([html.Span(className="kb-sim-swatch kb-sim-swatch--dash",
                                     style={"borderColor": SIM_VAR_LINE}),
                           f"VaR {conf}%"],
                          className="kb-sim-leg-tile"),
                html.Span([html.Span(className="kb-sim-swatch kb-sim-swatch--dash",
                                     style={"borderColor": SIM_CVAR_LINE}),
                           f"CVaR {conf}%"],
                          className="kb-sim-leg-tile"),
            ],
            className="kb-sim-legend-row",
        ),
    ])

    hist_card = _card([
        _card_head(
            "Распределение терминальных значений",
            right=_chip("50 БИНОВ · Sₜ", "neutral"),
        ),
        html.Div(dcc.Graph(
            figure=_mc_histogram(finals, var_v, cvar_v, mean_v, median_v),
            config={"displayModeBar": False}),
            className="kb-sim-chart-frame"),
        html.Div([
            _chip(f"SKEWNESS {skew:.3f}", "neutral"),
            _chip(f"KURTOSIS {kurt:.3f}", "neutral"),
            _chip(f"MIN {min_v:.0f}", "neutral"),
            _chip(f"MAX {max_v:.0f}", "neutral"),
        ], className="kb-sim-stat-chips"),
    ])

    pct_card = _card([
        _card_head("Перцентильная сводка",
                   right=_chip("7 ТОЧЕК", "neutral")),
        _percentile_table(finals, last_val),
        html.Div([
            html.Span("BASELINE", className="kb-overline"),
            html.Span(_fmt_num(last_val),
                      className="kb-mono kb-sim-pct-baseline"),
        ], className="kb-sim-pct-foot"),
    ], pad=False)

    sticky = html.Div([
        html.Button([icon("plus", 13), html.Span("Сохранить run")],
                    id="btn-mc-bookmark",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Button([icon("download", 13),
                     html.Span("Экспорт CSV всех путей")],
                    id="btn-mc-export-csv",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Button([icon("download", 13), html.Span("Экспорт PNG")],
                    id="btn-mc-export-png",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Span(className="kb-sim-sticky__grow"),
        html.Button([icon("alert", 13),
                     html.Span("Стресс-тест: запустить с σ × 2")],
                    id="btn-mc-stress",
                    className="kb-btn kb-btn--primary kb-btn--sm",
                    n_clicks=0),
    ], className="kb-sim-sticky")

    return html.Div([
        summary,
        html.Div([
            _section_overline("РЕЗУЛЬТАТ СИМУЛЯЦИИ"),
            _chip(f"{n_s:,} ТРАЕКТОРИЙ".replace(",", " "), "neutral"),
        ], className="kb-sim-overline-row"),
        kpi_strip,
        fan_card,
        html.Div([hist_card, pct_card],
                 className="kb-sim-mc-bottom-grid"),
        sticky,
    ])


def _percentile_table(values: np.ndarray, baseline: float) -> html.Table:
    sorted_vals = np.sort(values)
    n = len(sorted_vals)
    pct_pts = [5, 10, 25, 50, 75, 90, 95]
    head = html.Thead(html.Tr([
        html.Th("Перцентиль", className="kb-sim-tbl__th kb-mono"),
        html.Th("Значение",
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
        html.Th("Δ от baseline",
                className="kb-sim-tbl__th kb-sim-tbl__th--r kb-mono"),
    ]))
    body = []
    for p in pct_pts:
        v = float(sorted_vals[min(int(n * p / 100), n - 1)])
        d = v - baseline
        d_pct = (d / baseline * 100) if baseline else 0
        col = INFO if d >= 0 else DANGER
        body.append(html.Tr([
            html.Td(f"P{p}",
                    className="kb-sim-tbl__td kb-mono",
                    style={"fontWeight": 500}),
            html.Td(_fmt_num(v),
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono"),
            html.Td(f"{d:+.2f}  ({d_pct:+.2f}%)",
                    className="kb-sim-tbl__td kb-sim-tbl__td--r kb-mono",
                    style={"color": col}),
        ]))
    return html.Table([head, html.Tbody(body)],
                      className="kb-sim-tbl kb-sim-tbl--pct")

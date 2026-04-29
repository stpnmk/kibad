"""p17_matching — Сопоставление групп (Раздел 17) — Dash.

Redesigned for KIBAD Design System v2026.04 (dark eucalyptus).
Mirrors handoff slide 17 («Сопоставление групп») — 6 artboards rendered
as a 5-tab strip:

  17.1 Настройка + Pre-balance ─ всегда видимый pre-balance блок сверху
  17.2 PSM (Propensity Score)  ─┐
  17.3 Точное сопоставление    ─┤  4 метод-таба
  17.4 Ближайший сосед (NN)    ─┤
  17.5 CEM (огрубление)        ─┘
  17.6 Сравнение методов       ─ tab «Сравнение» (виден после запусков)
"""
from __future__ import annotations

from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, ctx, dcc, html, no_update
import dash_bootstrap_components as dbc

from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
    get_df_from_store, list_datasets,
)
from app.components.alerts import alert_banner
from app.components.icons import icon

from core.matching import (
    balance_summary, coarsened_exact_match, exact_match,
    nearest_neighbor_match, propensity_score_match,
    standardized_mean_diff,
)

dash.register_page(
    __name__, path="/matching", name="17. Сопоставление",
    order=17, icon="bullseye",
)


# ---------------------------------------------------------------------------
# Design tokens — match role palette (mirror match-shared.js MATCH dict)
# ---------------------------------------------------------------------------

MATCH_TREATMENT = "#A066C8"
MATCH_CONTROL   = "#4A7FB0"
SMD_GOOD        = "#21A066"
SMD_WARN        = "#C98A2E"
SMD_BAD         = "#C8503B"
M_PSM           = "#21A066"
M_EXACT         = "#4A7FB0"
M_NN            = "#C98A2E"
M_CEM           = "#A066C8"

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

METHODS = [
    ("psm",     "PSM (Propensity Score)", M_PSM),
    ("exact",   "Точное сопоставление",    M_EXACT),
    ("nn",      "Ближайший сосед",         M_NN),
    ("cem",     "CEM (огрубление)",        M_CEM),
]


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


def _smd_severity(v: float) -> str:
    a = abs(v)
    if a < 0.1:
        return "good"
    if a < 0.25:
        return "warn"
    return "bad"


def _smd_color(v: float) -> str:
    s = _smd_severity(v)
    return {"good": SMD_GOOD, "warn": SMD_WARN, "bad": SMD_BAD}[s]


def _fmt_num(v: Any, digits: int = 3) -> str:
    try:
        f = float(v)
        if np.isnan(f):
            return "—"
        if abs(f) >= 1000:
            return f"{f:,.0f}".replace(",", " ")
        return f"{f:.{digits}f}"
    except Exception:
        return "—"


def _base_layout(height: int = 320) -> dict:
    return dict(
        height=height,
        margin=dict(l=44, r=24, t=20, b=28),
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

def _smd_bar_chart(rows: list[dict], height: int = 360,
                   x_max: float = 0.5) -> go.Figure:
    """Horizontal SMD bar chart with severity colors + 0.1/0.25 ref lines.

    rows: [{name: str, val: float, dir: 'up'|'down'}]
    """
    rows = sorted(rows, key=lambda r: abs(r["val"]), reverse=True)
    labels = [r["name"] for r in rows]
    vals = [abs(r["val"]) for r in rows]
    colors = [_smd_color(v) for v in vals]
    annotations_text = [
        f"{vals[i]:.3f}  "
        f"<span style='color:{TEXT_TER};font-size:9px'>"
        f"{'▲ treated больше' if r.get('dir') == 'up' else '▼ control больше'}"
        f"</span>"
        for i, r in enumerate(rows)
    ]

    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=colors),
        text=annotations_text, textposition="outside",
        textfont=dict(family="JetBrains Mono", size=11),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    # Reference lines 0.1 (good) and 0.25 (bad)
    fig.add_vline(x=0.1, line=dict(color=SMD_GOOD, width=1, dash="dash"),
                  annotation_text="0.1", annotation_position="bottom",
                  annotation_font=dict(family="JetBrains Mono",
                                        size=10, color=SMD_GOOD))
    fig.add_vline(x=0.25, line=dict(color=SMD_BAD, width=1, dash="dash"),
                  annotation_text="0.25", annotation_position="bottom",
                  annotation_font=dict(family="JetBrains Mono",
                                        size=10, color=SMD_BAD))
    lay = _base_layout(height)
    lay["xaxis"]["range"] = [0, x_max]
    lay["yaxis"]["autorange"] = "reversed"
    lay.update(margin=dict(l=160, r=180, t=14, b=44))
    fig.update_layout(**lay)
    return fig


def _love_plot(before: pd.DataFrame, after: pd.DataFrame,
               height: int = 320) -> go.Figure:
    """Love-plot — before/after pairs with arrow connectors.

    before/after: DataFrame with columns `covariate, abs_smd`.
    """
    # Align on covariate
    merged = before[["covariate", "abs_smd"]].rename(
        columns={"abs_smd": "before"}).merge(
        after[["covariate", "abs_smd"]].rename(
            columns={"abs_smd": "after"}),
        on="covariate", how="left")
    merged = merged.sort_values("before", ascending=False).reset_index(
        drop=True)

    fig = go.Figure()
    # Connector line from before → after
    for _, row in merged.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["before"], row["after"]],
            y=[row["covariate"], row["covariate"]],
            mode="lines",
            line=dict(color=TEXT_TER, width=1, dash="dot"),
            opacity=0.5, showlegend=False, hoverinfo="skip",
        ))
    # Before — open circles colored by severity
    fig.add_trace(go.Scatter(
        x=merged["before"], y=merged["covariate"],
        mode="markers",
        marker=dict(size=10, symbol="circle-open",
                    line=dict(width=1.5,
                              color=[_smd_color(v) for v in merged["before"]])),
        name="До сопоставления",
        hovertemplate="<b>%{y}</b><br>До: %{x:.3f}<extra></extra>",
    ))
    # After — filled circles colored by severity
    fig.add_trace(go.Scatter(
        x=merged["after"], y=merged["covariate"],
        mode="markers",
        marker=dict(size=10, color=[_smd_color(v) for v in merged["after"]],
                    line=dict(width=1, color=SURFACE_0)),
        name="После сопоставления",
        hovertemplate="<b>%{y}</b><br>После: %{x:.3f}<extra></extra>",
    ))
    # Reference lines
    fig.add_vline(x=0.1, line=dict(color=SMD_GOOD, width=1, dash="dash"),
                  annotation_text="0.1", annotation_position="bottom",
                  annotation_font=dict(family="JetBrains Mono",
                                        size=9, color=SMD_GOOD))
    fig.add_vline(x=0.25, line=dict(color=SMD_BAD, width=1, dash="dash"),
                  annotation_text="0.25", annotation_position="bottom",
                  annotation_font=dict(family="JetBrains Mono",
                                        size=9, color=SMD_BAD))
    lay = _base_layout(height)
    lay["xaxis"]["range"] = [0, max(0.5, merged["before"].max() * 1.1)]
    lay["yaxis"]["autorange"] = "reversed"
    lay.update(margin=dict(l=140, r=24, t=14, b=44))
    fig.update_layout(**lay)
    return fig


def _propensity_hist(ps: np.ndarray, treatment: np.ndarray,
                     height: int = 240) -> go.Figure:
    """Overlapping histogram of propensity scores by group."""
    fig = go.Figure([
        go.Histogram(x=ps[treatment == 0], name="Control",
                     marker=dict(color=MATCH_CONTROL, opacity=0.55),
                     nbinsx=30),
        go.Histogram(x=ps[treatment == 1], name="Treatment",
                     marker=dict(color=MATCH_TREATMENT, opacity=0.55),
                     nbinsx=30),
    ])
    lay = _base_layout(height)
    lay.update(barmode="overlay",
               margin=dict(l=44, r=14, t=14, b=28))
    fig.update_layout(**lay)
    return fig


def _distance_hist(distances: np.ndarray,
                   height: int = 240) -> go.Figure:
    """Single histogram of pair distances + median ref line."""
    if len(distances) == 0:
        fig = go.Figure()
        fig.update_layout(**_base_layout(height))
        return fig
    median = float(np.median(distances))
    fig = go.Figure(go.Histogram(
        x=distances, nbinsx=30,
        marker=dict(color=ACCENT_500, opacity=0.78),
        hovertemplate="дистанция: %{x:.3f}<br>пар: %{y}<extra></extra>",
    ))
    fig.add_vline(x=median, line=dict(color=SMD_GOOD, width=1.5, dash="dash"),
                  annotation_text=f"median {median:.3f}",
                  annotation_position="top",
                  annotation_font=dict(family="JetBrains Mono",
                                        size=10, color=SMD_GOOD))
    lay = _base_layout(height)
    lay.update(margin=dict(l=44, r=14, t=24, b=28))
    fig.update_layout(**lay)
    return fig


def _stratum_bars(strata: list[dict]) -> html.Div:
    """Custom HTML stratum bar chart (treatment + control split).

    strata: [{name, t, c}]
    """
    if not strata:
        return html.Div("Нет страт.", className="kb-match-strat-empty")
    max_total = max(s["t"] + s["c"] for s in strata) or 1
    rows = []
    for s in strata:
        total = s["t"] + s["c"]
        width_pct = total / max_total * 100
        t_pct = s["t"] / total * 100 if total else 0
        rows.append(html.Div(
            [
                html.Span(s["name"], className="kb-match-strat-name"),
                html.Div(
                    [
                        html.Div(
                            html.Span(f"t={s['t']}, c={s['c']}",
                                      className="kb-match-strat-lbl"),
                            className="kb-match-strat-bar-inner",
                            style={"background":
                                       f"linear-gradient(90deg, "
                                       f"{MATCH_TREATMENT} 0 {t_pct}%, "
                                       f"{MATCH_CONTROL} {t_pct}% 100%)"},
                        ),
                    ],
                    className="kb-match-strat-bar",
                    style={"width": f"{width_pct}%"},
                ),
                html.Span(str(total), className="kb-match-strat-total"),
            ],
            className="kb-match-strat-row",
        ))
    return html.Div(rows, className="kb-match-strata")


def _cem_bin_grid(covariates: list[str], n_bins: int,
                  ranges: dict[str, tuple] | None = None) -> html.Div:
    """Grid of coarsening bins per covariate (gradient blocks)."""
    if not covariates:
        return html.Div("Нет ковариат.", className="kb-match-cem-empty")
    rows = []
    ranges = ranges or {}
    for cov in covariates:
        rng = ranges.get(cov, (0.10, 9.83))
        bins = []
        for i in range(n_bins):
            t = i / max(n_bins - 1, 1)
            r = round(0x22 + t * (0x1F - 0x22))
            g = round(0x30 + t * (0x7A - 0x30))
            b = round(0x29 + t * (0x4D - 0x29))
            bins.append(html.Div(
                className="kb-match-cem-bin",
                style={"background": f"rgb({r},{g},{b})"}))
        rows.append(html.Div(
            [
                html.Div(
                    [
                        html.Span(cov, className="kb-match-cem-name"),
                        html.Span(
                            f"{n_bins} бинов · range "
                            f"{rng[0]:.2f}–{rng[1]:.2f}",
                            className="kb-match-cem-meta",
                        ),
                    ],
                    className="kb-match-cem-head",
                ),
                html.Div(bins, className="kb-match-cem-bins",
                         style={"gridTemplateColumns":
                                    f"repeat({n_bins}, 1fr)"}),
            ],
            className="kb-match-cem-cov",
        ))
    return html.Div(rows, className="kb-match-cem")


def _love_multi(before: pd.DataFrame,
                after_by_method: dict[str, pd.DataFrame],
                height: int = 380) -> go.Figure:
    """Multi-method love-plot — open circles for before, filled for each method."""
    if before.empty:
        fig = go.Figure()
        fig.update_layout(**_base_layout(height))
        return fig
    merged = before[["covariate", "abs_smd"]].rename(
        columns={"abs_smd": "before"})
    for k, df_a in after_by_method.items():
        merged = merged.merge(
            df_a[["covariate", "abs_smd"]].rename(columns={"abs_smd": k}),
            on="covariate", how="left")
    merged = merged.sort_values("before", ascending=False).reset_index(
        drop=True)

    method_colors = {"psm": M_PSM, "exact": M_EXACT,
                     "nn": M_NN, "cem": M_CEM}

    fig = go.Figure()
    # Before — open circles
    fig.add_trace(go.Scatter(
        x=merged["before"], y=merged["covariate"],
        mode="markers",
        marker=dict(size=10, symbol="circle-open",
                    line=dict(width=1.5,
                              color=[_smd_color(v) for v in merged["before"]])),
        name="До", hoverinfo="skip", showlegend=False,
    ))
    # Connectors per method
    for method, color in method_colors.items():
        if method not in after_by_method:
            continue
        for _, row in merged.iterrows():
            if pd.isna(row.get(method)):
                continue
            fig.add_trace(go.Scatter(
                x=[row["before"], row[method]],
                y=[row["covariate"], row["covariate"]],
                mode="lines",
                line=dict(color=color, width=0.6),
                opacity=0.4, showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=merged[method], y=merged["covariate"],
            mode="markers",
            marker=dict(size=8, color=color,
                        line=dict(width=1, color=SURFACE_0)),
            name=method.upper(),
            hovertemplate=f"<b>%{{y}}</b><br>{method}: %{{x:.3f}}<extra></extra>",
        ))
    fig.add_vline(x=0.1, line=dict(color=SMD_GOOD, width=1, dash="dash"))
    fig.add_vline(x=0.25, line=dict(color=SMD_BAD, width=1, dash="dash"))
    lay = _base_layout(height)
    lay["xaxis"]["range"] = [0, 0.5]
    lay["yaxis"]["autorange"] = "reversed"
    lay.update(margin=dict(l=160, r=24, t=14, b=28))
    fig.update_layout(**lay)
    return fig


# ---------------------------------------------------------------------------
# UI partials — page head, dataset pill, config card, group info
# ---------------------------------------------------------------------------

def _page_head() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("АНАЛИЗ ВЛИЯНИЯ", className="kb-overline"),
                    html.H1("17. Сопоставление групп",
                            className="kb-page-title kb-match-title"),
                    html.P(
                        "Подбор сопоставимых групп для корректного сравнения "
                        "(PSM, Exact, NN, CEM)",
                        className="kb-page-subtitle",
                    ),
                ],
                className="kb-page-head-left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("download", 14),
                         html.Span("Экспорт matched-датасета")],
                        id="match-export-btn",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("git-compare", 14),
                         html.Span("Сравнить методы")],
                        id="match-compare-btn",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-page-head-actions",
            ),
        ],
        className="kb-page-head",
    )


def _dataset_pill_with_kpi() -> html.Div:
    """Compact dataset pill + 2 KPI tiles (rows / cols)."""
    return html.Div(
        [
            html.Div(
                [
                    icon("database", 16, className="kb-match-ds__icn"),
                    dcc.Dropdown(
                        id="match-ds-select", options=[], value=None,
                        clearable=False, placeholder="Выберите датасет…",
                        className="kb-select kb-match-ds__select",
                    ),
                ],
                className="kb-match-ds__row",
            ),
            html.Div(id="match-mini-kpis", className="kb-match-mini-kpis"),
        ],
        className="kb-match-ds-bar",
    )


def _kpi_tile(label: str, value: str, sub: str = "",
              variant: str | None = None,
              marker_color: str | None = None) -> html.Div:
    cls = "kb-match-kpi"
    if variant:
        cls += f" kb-match-kpi--{variant}"
    style = {}
    if marker_color:
        style["--kpi-marker"] = marker_color
        cls += " kb-match-kpi--marker"
    return html.Div(
        [
            html.Div(label, className="kb-match-kpi__l"),
            html.Div(value, className="kb-match-kpi__v"),
            html.Div(sub, className="kb-match-kpi__s") if sub else html.Div(),
        ],
        className=cls, style=style,
    )


def _chip(text: str, variant: str = "neutral",
          mono: bool = False) -> html.Span:
    cls = f"kb-chip kb-chip--{variant}"
    if mono:
        cls += " kb-mono"
    return html.Span(text, className=cls)


def _section_overline(text: str, icn: str | None = None) -> html.Div:
    kids = []
    if icn:
        kids.append(icon(icn, 14, className="kb-match-overline-icn"))
    kids.append(html.Span(text, className="kb-overline"))
    return html.Div(kids, className="kb-match-overline-row")


def _card(children, *, pad: bool = True) -> html.Div:
    cls = "kb-match-card"
    if pad:
        cls += " kb-match-card--pad"
    return html.Div(children, className=cls)


def _card_head(title: str, right=None,
               icn: str | None = None) -> html.Div:
    left = []
    if icn:
        left.append(icon(icn, 16, className="kb-match-card__icn"))
    left.append(html.H3(title, className="kb-match-card__title"))
    return html.Div(
        [html.Div(left, className="kb-match-card__head-left"),
         html.Div(right) if right is not None else html.Div()],
        className="kb-match-card__head",
    )


def _alert(level: str, title: str, body) -> html.Div:
    icn_map = {"info": "info", "success": "check-circle",
               "warning": "alert", "danger": "alert"}
    return html.Div(
        [
            html.Div(icon(icn_map.get(level, "info"), 16),
                     className=f"kb-match-alert__icn kb-match-alert__icn--{level}"),
            html.Div(
                [html.Div(title, className="kb-match-alert__t"),
                 html.Div(body, className="kb-match-alert__b")],
                className="kb-match-alert__body",
            ),
        ],
        className=f"kb-match-alert kb-match-alert--{level}",
    )


def _config_card() -> html.Div:
    return _card(
        [
            _card_head("Конфигурация сопоставления", icn="target"),
            html.Div([
                html.Div(
                    [
                        html.Label("КОЛОНКА ЛЕЧЕНИЯ / TREATMENT",
                                   className="kb-match-fld__l"),
                        dcc.Dropdown(
                            id="match-treatment-col", options=[], value=None,
                            clearable=False, placeholder="—",
                            className="kb-select kb-match-select",
                        ),
                    ],
                    className="kb-match-fld",
                ),
                html.Div(
                    [
                        html.Label(
                            id="match-cov-label",
                            children="КОВАРИАТЫ ДЛЯ СОПОСТАВЛЕНИЯ",
                            className="kb-match-fld__l",
                        ),
                        dcc.Dropdown(
                            id="match-covariates", options=[], value=[],
                            multi=True, placeholder="добавить…",
                            className="kb-select kb-match-select kb-match-select--chips",
                        ),
                    ],
                    className="kb-match-fld",
                ),
            ], className="kb-match-cfg-grid"),
            html.Div(id="match-group-info", className="kb-match-group-info"),
        ],
    )


def _radio_pill(idc: str, options: list[dict], value=None) -> dcc.RadioItems:
    return dcc.RadioItems(
        id=idc, options=options, value=value, inline=True,
        className="kb-match-radio",
        inputClassName="kb-match-radio__inp",
        labelClassName="kb-match-radio__opt",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_TABS = [
    ("psm",     "PSM (Propensity Score)"),
    ("exact",   "Точное сопоставление"),
    ("nn",      "Ближайший сосед"),
    ("cem",     "CEM (огрубление)"),
    ("compare", "Сравнение"),
]


layout = html.Div(
    [
        _page_head(),
        _dataset_pill_with_kpi(),
        _config_card(),

        # Pre-balance section (always visible above the tab strip)
        dcc.Loading(html.Div(id="match-pre-balance"),
                    type="circle", color=ACCENT_500),

        html.Div(
            dbc.Tabs(
                id="match-tabs", active_tab="psm",
                children=[dbc.Tab(label=lbl, tab_id=tid)
                          for tid, lbl in _TABS],
                className="kb-match-tabs",
            ),
            className="kb-match-tabs-wrap",
        ),

        html.Div(id="match-tab-content", className="kb-match-tab-content"),

        # Hidden stores for results across methods
        dcc.Store(id="match-results-store", storage_type="memory", data={}),
    ],
    className="kb-page kb-page-match",
)


# ---------------------------------------------------------------------------
# Dataset dropdown + dependent column options
# ---------------------------------------------------------------------------

@callback(
    Output("match-ds-select", "options"),
    Output("match-ds-select", "value"),
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


@callback(
    Output("match-mini-kpis", "children"),
    Output("match-treatment-col", "options"),
    Output("match-covariates", "options"),
    Input("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _ds_metadata(ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return ([
            _kpi_tile("СТРОК", "—"),
            _kpi_tile("СТОЛБЦОВ", "—"),
        ], [], [])
    n_rows = len(df)
    n_cols = df.shape[1]
    mini = [
        _kpi_tile("СТРОК", f"{n_rows:,}".replace(",", " ")),
        _kpi_tile("СТОЛБЦОВ", str(n_cols)),
    ]
    # Treatment candidates: columns with exactly 2 unique values (excluding NaN)
    t_opts = []
    for c in df.columns:
        try:
            uniq = df[c].dropna().unique()
            if 2 <= len(uniq) <= 2 and pd.api.types.is_numeric_dtype(df[c]):
                t_opts.append(c)
            elif len(uniq) == 2:
                t_opts.append(c)
        except Exception:
            continue
    num_cols = df.select_dtypes(include="number").columns.tolist()
    t_options = [{"label": c, "value": c} for c in t_opts]
    cov_options = [{"label": c, "value": c} for c in num_cols]
    return mini, t_options, cov_options


@callback(
    Output("match-group-info", "children"),
    Output("match-cov-label", "children"),
    Input("match-treatment-col", "value"),
    Input("match-covariates", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _group_info(treatment_col, covs, ds_name, raw, prep):
    n_cov = len(covs or [])
    cov_label = (f"КОВАРИАТЫ ДЛЯ СОПОСТАВЛЕНИЯ · {n_cov}" if n_cov
                 else "КОВАРИАТЫ ДЛЯ СОПОСТАВЛЕНИЯ")
    if not treatment_col:
        return html.Div(), cov_label
    df = _get_df(ds_name, raw, prep)
    if df is None or treatment_col not in df.columns:
        return html.Div(), cov_label
    uniq = df[treatment_col].dropna().unique()
    if len(uniq) != 2:
        return _alert(
            "warning", "Выберите бинарную колонку",
            f"Колонка содержит {len(uniq)} уникальных значений. Нужно ровно 2."
        ), cov_label

    # Identify treated (=1, second value if both numeric) vs control
    if pd.api.types.is_numeric_dtype(df[treatment_col]):
        treated = sorted(uniq, reverse=True)[0]
        control = sorted(uniq, reverse=True)[1]
    else:
        treated = uniq[1]
        control = uniq[0]
    n_t = int((df[treatment_col] == treated).sum())
    n_c = int((df[treatment_col] == control).sum())

    info = html.Div([
        html.Span([
            html.Span(className="kb-match-gi-marker",
                      style={"background": MATCH_TREATMENT}),
            html.Span("Опытная: ", className="kb-match-gi-lbl"),
            html.Span(str(treated), className="kb-mono",
                      style={"color": TEXT_PRI}),
            html.Span(f" (n={n_t:,})".replace(",", " "),
                      className="kb-match-gi-n"),
        ], className="kb-match-gi"),
        html.Span("·", className="kb-match-gi-sep"),
        html.Span([
            html.Span(className="kb-match-gi-marker",
                      style={"background": MATCH_CONTROL}),
            html.Span("Контроль: ", className="kb-match-gi-lbl"),
            html.Span(str(control), className="kb-mono",
                      style={"color": TEXT_PRI}),
            html.Span(f" (n={n_c:,})".replace(",", " "),
                      className="kb-match-gi-n"),
        ], className="kb-match-gi"),
    ])
    return info, cov_label


# ---------------------------------------------------------------------------
# Pre-balance — always visible above the tab strip
# ---------------------------------------------------------------------------

def _prepare_work_df(df, treatment_col):
    """Convert raw DataFrame to a work DataFrame with treatment_col ∈ {0, 1}."""
    if df is None or treatment_col not in df.columns:
        return None
    uniq = df[treatment_col].dropna().unique()
    if len(uniq) != 2:
        return None
    if pd.api.types.is_numeric_dtype(df[treatment_col]):
        treated = sorted(uniq, reverse=True)[0]
    else:
        treated = uniq[1]
    work = df.copy()
    work[treatment_col] = (work[treatment_col] == treated).astype(int)
    return work


@callback(
    Output("match-pre-balance", "children"),
    Input("match-covariates", "value"),
    Input("match-treatment-col", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_pre_balance(covariates, treatment_col, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return html.Div()
    if not treatment_col or not covariates:
        return _alert(
            "info",
            "Конфигурация сопоставления",
            "Выберите колонку лечения (treatment) и числовые ковариаты — "
            "KIBAD рассчитает баланс групп до сопоставления.",
        )
    work = _prepare_work_df(df, treatment_col)
    if work is None:
        return html.Div()

    num_cov = [c for c in covariates if pd.api.types.is_numeric_dtype(work[c])]
    if not num_cov:
        return _alert("warning", "Нет числовых ковариат",
                      "Выберите хотя бы одну числовую ковариату.")

    bal = standardized_mean_diff(work, treatment_col, num_cov)
    summary = balance_summary(bal)

    # Build SMD bar-chart rows with direction indicator (up/down)
    rows = []
    for _, r in bal.iterrows():
        rows.append({
            "name": r["covariate"],
            "val": float(r["abs_smd"]),
            "dir": ("up" if r["mean_treatment"] >= r["mean_control"]
                    else "down"),
        })

    n_t = int((work[treatment_col] == 1).sum())
    n_c = int((work[treatment_col] == 0).sum())
    bal_pct = float(summary["pct_below_01"])
    mean_smd = float(summary["mean_abs_smd"])

    # Severity counts
    cnt_good = int(sum(1 for r in rows if r["val"] < 0.1))
    cnt_warn = int(sum(1 for r in rows if 0.1 <= r["val"] < 0.25))
    cnt_bad  = int(sum(1 for r in rows if r["val"] >= 0.25))

    smd_color = (SMD_GOOD if mean_smd < 0.1
                 else SMD_WARN if mean_smd < 0.25
                 else SMD_BAD)

    kpi_strip = html.Div([
        _kpi_tile("ОПЫТНАЯ", f"{n_t:,}".replace(",", " "),
                  "treated (=1)", marker_color=MATCH_TREATMENT),
        _kpi_tile("КОНТРОЛЬ", f"{n_c:,}".replace(",", " "),
                  "control (=0)", marker_color=MATCH_CONTROL),
        html.Div([
            html.Div("СРЕДНИЙ |SMD|", className="kb-match-kpi__l"),
            html.Div(f"{mean_smd:.3f}", className="kb-match-kpi__v",
                     style={"color": smd_color}),
            html.Div(
                html.Div(className="kb-match-kpi-bar__f",
                         style={"width": f"{min(100, mean_smd / 0.5 * 100)}%",
                                "background": smd_color}),
                className="kb-match-kpi-bar"),
            html.Div("начальный дисбаланс",
                     className="kb-match-kpi__s"),
        ], className="kb-match-kpi"),
        html.Div([
            html.Div("|SMD| < 0.1", className="kb-match-kpi__l"),
            html.Div(f"{bal_pct:.0f}%", className="kb-match-kpi__v",
                     style={"color": ACCENT_300 if bal_pct >= 80
                                                 else TEXT_PRI}),
            html.Div(
                html.Div(className="kb-match-kpi-bar__f",
                         style={"width": f"{bal_pct}%",
                                "background": ACCENT_500}),
                className="kb-match-kpi-bar"),
            html.Div("доля сбалансированных ковариат",
                     className="kb-match-kpi__s"),
        ], className="kb-match-kpi"),
    ], className="kb-match-kpis kb-match-kpis--4")

    smd_card = _card([
        _card_head(
            "SMD по ковариатам · до сопоставления",
            right=html.Div([
                _chip(f"{cnt_good} < 0.1", "success"),
                _chip(f"{cnt_warn} в 0.1–0.25", "warning"),
                _chip(f"{cnt_bad} ≥ 0.25", "danger"),
            ], className="kb-match-chip-row"),
        ),
        html.Div(
            dcc.Graph(figure=_smd_bar_chart(rows, height=300, x_max=0.5),
                      config={"displayModeBar": False}),
            className="kb-match-chart-frame",
        ),
    ])

    alert_kid = (
        _alert("info",
               "Без сопоставления групп прямое сравнение метрик некорректно",
               f"{cnt_bad} ковариат имеют |SMD| ≥ 0.25. Запустите один из "
               "методов ниже, чтобы выровнять распределения.")
        if cnt_bad else
        _alert("success",
               "Группы достаточно сбалансированы",
               f"Все ковариаты имеют |SMD| < 0.25. Можно запустить любой "
               "из методов для уточнения баланса.")
    )

    return html.Div([
        _section_overline("ПРЕДВАРИТЕЛЬНАЯ БАЛАНСИРОВКА (ДО СОПОСТАВЛЕНИЯ)",
                          icn="trend"),
        kpi_strip,
        smd_card,
        alert_kid,
    ])


# ---------------------------------------------------------------------------
# Tab content dispatcher
# ---------------------------------------------------------------------------

@callback(
    Output("match-tab-content", "children"),
    Input("match-tabs", "active_tab"),
    Input("match-results-store", "data"),
    State("match-ds-select", "value"),
    State("match-treatment-col", "value"),
    State("match-covariates", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_tab(tab, results, ds_name, treatment_col, covs, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if tab == "compare":
        return _render_compare(results, df, treatment_col)
    if tab == "psm":
        return _render_psm_tab(df, covs)
    if tab == "exact":
        return _render_exact_tab(df, covs)
    if tab == "nn":
        return _render_nn_tab(df, covs)
    if tab == "cem":
        return _render_cem_tab(df, covs)
    return html.Div()


def _params_card(icn: str, title: str, body, *,
                 chip_text: str, btn_id: str, btn_text: str) -> html.Div:
    return _card([
        _card_head(title, icn=icn),
        body,
        html.Div([
            _chip(chip_text, "neutral", mono=True),
            html.Button(
                [icon("play", 14), html.Span(btn_text)],
                id=btn_id, className="kb-btn kb-btn--primary",
                n_clicks=0,
            ),
        ], className="kb-match-params-foot"),
    ])


# ---------------------------------------------------------------------------
# 17.2 PSM
# ---------------------------------------------------------------------------

def _render_psm_tab(df, covs) -> html.Div:
    cat_opts = []
    if df is not None:
        cat_opts = [{"label": c, "value": c} for c in df.columns]

    body = html.Div([
        html.Div([
            html.Div(
                [
                    html.Div([
                        html.Label("CALIPER (MAX DISTANCE)",
                                   className="kb-match-fld__l"),
                        html.Span("0.20", id="psm-caliper-val",
                                  className="kb-match-slider-val"),
                    ], className="kb-match-slider-hdr"),
                    dcc.Slider(
                        id="psm-caliper", min=0.01, max=1.0, value=0.20,
                        step=0.01, marks=None,
                        tooltip={"always_visible": False,
                                 "placement": "bottom"},
                        className="kb-match-slider",
                    ),
                    html.Div("доля стандартного отклонения логит-скора",
                             className="kb-match-slider-hint"),
                ],
                className="kb-match-fld",
            ),
            html.Div([
                html.Label("СООТНОШЕНИЕ MATCHING",
                           className="kb-match-fld__l"),
                dcc.Dropdown(
                    id="psm-ratio",
                    options=[{"label": f"1:{r}", "value": r}
                             for r in [1, 2, 3, 5]],
                    value=1, clearable=False,
                    className="kb-select kb-match-select",
                ),
            ], className="kb-match-fld"),
        ], className="kb-match-params-grid"),
    ])

    return html.Div([
        _params_card(
            "settings", "Параметры PSM", body,
            chip_text="LOGISTIC + NEAREST NEIGHBOR",
            btn_id="btn-run-psm", btn_text="Запустить PSM",
        ),
        dcc.Loading(html.Div(id="psm-result"),
                    type="circle", color=ACCENT_500),
    ])


# ---------------------------------------------------------------------------
# 17.3 Exact
# ---------------------------------------------------------------------------

def _render_exact_tab(df, covs) -> html.Div:
    cat_opts = []
    if df is not None:
        cat_opts = [{"label": c, "value": c}
                    for c in df.columns
                    if (not pd.api.types.is_numeric_dtype(df[c])
                        or df[c].nunique() <= 12)]

    body = html.Div([
        html.Div([
            html.Label("КАТЕГОРИАЛЬНЫЕ КОЛОНКИ ДЛЯ ТОЧНОГО СОВПАДЕНИЯ",
                       className="kb-match-fld__l"),
            dcc.Dropdown(
                id="exact-cols", options=cat_opts, value=[],
                multi=True, placeholder="—",
                className="kb-select kb-match-select kb-match-select--chips",
            ),
            html.Div(
                "KIBAD сформирует страты — для каждого уникального сочетания "
                "значений отберёт сбалансированную выборку",
                className="kb-match-slider-hint",
            ),
        ], className="kb-match-fld"),
    ])

    return html.Div([
        _params_card(
            "table", "Параметры Exact Matching", body,
            chip_text="STRATIFIED EXACT MATCHING",
            btn_id="btn-run-exact", btn_text="Запустить Exact",
        ),
        dcc.Loading(html.Div(id="exact-result"),
                    type="circle", color=ACCENT_500),
    ])


# ---------------------------------------------------------------------------
# 17.4 NN
# ---------------------------------------------------------------------------

def _render_nn_tab(df, covs) -> html.Div:
    body = html.Div([
        html.Div([
            html.Div([
                html.Label("K (СОСЕДЕЙ)", className="kb-match-fld__l"),
                dcc.Dropdown(
                    id="nn-k",
                    options=[{"label": str(k), "value": k}
                             for k in [1, 2, 3, 5]],
                    value=1, clearable=False,
                    className="kb-select kb-match-select",
                ),
            ], className="kb-match-fld"),
            html.Div([
                html.Label("МЕТРИКА РАССТОЯНИЯ",
                           className="kb-match-fld__l"),
                _radio_pill(
                    "nn-metric",
                    [{"label": "Mahalanobis", "value": "mahalanobis"},
                     {"label": "Euclidean",   "value": "euclidean"}],
                    value="mahalanobis"),
                html.Div(
                    "Mahalanobis учитывает корреляцию признаков; "
                    "Euclidean быстрее на больших выборках",
                    className="kb-match-slider-hint",
                ),
            ], className="kb-match-fld"),
        ], className="kb-match-params-grid"),
    ])

    return html.Div([
        _params_card(
            "target", "Параметры NN-Matching", body,
            chip_text="SCIPY CDIST · NEAREST-K",
            btn_id="btn-run-nn", btn_text="Запустить NN",
        ),
        dcc.Loading(html.Div(id="nn-result"),
                    type="circle", color=ACCENT_500),
    ])


# ---------------------------------------------------------------------------
# 17.5 CEM
# ---------------------------------------------------------------------------

def _render_cem_tab(df, covs) -> html.Div:
    body = html.Div([
        html.Div([
            html.Div([
                html.Label("ЧИСЛО БИНОВ", className="kb-match-fld__l"),
                html.Span("5", id="cem-bins-val",
                          className="kb-match-slider-val"),
            ], className="kb-match-slider-hdr"),
            dcc.Slider(
                id="cem-bins", min=2, max=20, value=5, step=1,
                marks=None,
                tooltip={"always_visible": False, "placement": "bottom"},
                className="kb-match-slider",
            ),
            html.Div(
                "KIBAD разобьёт каждую числовую ковариату на N квантилей "
                "и потребует точное совпадение бинов",
                className="kb-match-slider-hint",
            ),
        ], className="kb-match-fld"),
    ])

    return html.Div([
        _params_card(
            "layers", "Параметры CEM", body,
            chip_text="COARSENED EXACT MATCHING",
            btn_id="btn-run-cem", btn_text="Запустить CEM",
        ),
        dcc.Loading(html.Div(id="cem-result"),
                    type="circle", color=ACCENT_500),
    ])


# ---------------------------------------------------------------------------
# Slider value sync
# ---------------------------------------------------------------------------

@callback(Output("psm-caliper-val", "children"),
          Input("psm-caliper", "value"))
def _sync_caliper(v):
    return f"{float(v or 0.20):.2f}"


@callback(Output("cem-bins-val", "children"),
          Input("cem-bins", "value"))
def _sync_bins(v):
    return str(int(v or 5))


# ---------------------------------------------------------------------------
# Result panel — shared across methods
# ---------------------------------------------------------------------------

def _result_kpi_strip(method_label: str, method_sev: str,
                      n_match_t: int, n_t: int,
                      n_match_c: int, n_c: int,
                      smd_after: float, smd_before: float,
                      bal_pct: float,
                      extras: list[html.Div] | None = None) -> html.Div:
    pct_t = (n_match_t / n_t * 100) if n_t else 0
    pct_c = (n_match_c / n_c * 100) if n_c else 0
    smd_color = (SMD_GOOD if smd_after < 0.1
                 else SMD_WARN if smd_after < 0.25
                 else SMD_BAD)

    method_chip_var = ({"good": "success", "warn": "warning", "bad": "danger"}
                       .get(method_sev, "neutral"))

    method_kpi = html.Div([
        html.Div("МЕТОД", className="kb-match-kpi__l"),
        html.Div(_chip(method_label, method_chip_var, mono=True),
                 style={"marginTop": "8px"}),
    ], className="kb-match-kpi")

    n_t_kpi = html.Div([
        html.Div("ОПЫТНАЯ СОПОСТ.", className="kb-match-kpi__l"),
        html.Div([
            html.Span(f"{n_match_t:,}".replace(",", " "),
                      className="kb-match-kpi-num-big"),
            html.Span(" / ", className="kb-match-kpi-num-sep"),
            html.Span(f"{n_t:,}".replace(",", " "),
                      className="kb-match-kpi-num-tot"),
        ], className="kb-match-kpi__v"),
        html.Div(f"{pct_t:.0f}%", className="kb-match-kpi__s"),
    ], className="kb-match-kpi kb-match-kpi--marker",
       style={"--kpi-marker": MATCH_TREATMENT})

    n_c_kpi = html.Div([
        html.Div("КОНТРОЛЬ СОПОСТ.", className="kb-match-kpi__l"),
        html.Div([
            html.Span(f"{n_match_c:,}".replace(",", " "),
                      className="kb-match-kpi-num-big"),
            html.Span(" / ", className="kb-match-kpi-num-sep"),
            html.Span(f"{n_c:,}".replace(",", " "),
                      className="kb-match-kpi-num-tot"),
        ], className="kb-match-kpi__v"),
        html.Div(f"{pct_c:.1f}%", className="kb-match-kpi__s"),
    ], className="kb-match-kpi kb-match-kpi--marker",
       style={"--kpi-marker": MATCH_CONTROL})

    delta = smd_after - smd_before
    smd_kpi = html.Div([
        html.Div("|SMD| ПОСЛЕ", className="kb-match-kpi__l"),
        html.Div(f"{smd_after:.3f}", className="kb-match-kpi__v",
                 style={"color": smd_color}),
        html.Div(
            f"{'▼' if delta < 0 else '▲'} {delta:+.3f}",
            className="kb-match-kpi__s",
            style={"color": SMD_GOOD if delta < 0 else SMD_BAD},
        ),
    ], className="kb-match-kpi")

    bal_kpi = html.Div([
        html.Div("|SMD| < 0.1", className="kb-match-kpi__l"),
        html.Div(f"{bal_pct:.0f}%", className="kb-match-kpi__v",
                 style={"color": ACCENT_300}),
        html.Div(
            html.Div(className="kb-match-kpi-bar__f",
                     style={"width": f"{bal_pct}%",
                            "background": ACCENT_500}),
            className="kb-match-kpi-bar",
        ),
    ], className="kb-match-kpi")

    tiles = [method_kpi, n_t_kpi, n_c_kpi, smd_kpi, bal_kpi]
    if extras:
        tiles.extend(extras)
    n_cols = len(tiles)
    return html.Div(tiles,
                    className=f"kb-match-kpis kb-match-kpis--{n_cols}")


def _balance_table(before: pd.DataFrame, after: pd.DataFrame) -> html.Table:
    merged = before[["covariate", "mean_treatment", "mean_control",
                     "abs_smd"]].rename(
        columns={"abs_smd": "smd_before"}).merge(
        after[["covariate", "abs_smd"]].rename(
            columns={"abs_smd": "smd_after"}),
        on="covariate", how="left")
    merged = merged.sort_values("smd_before", ascending=False)

    head = html.Thead(html.Tr([
        html.Th("Ковариата", className="kb-match-tbl__th"),
        html.Th("Mean Treat.",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("Mean Control",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("|SMD| До",
                className="kb-match-tbl__th kb-match-tbl__th--r"),
        html.Th("|SMD| После",
                className="kb-match-tbl__th kb-match-tbl__th--r"),
        html.Th("Δ |SMD|",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("Статус",
                className="kb-match-tbl__th kb-match-tbl__th--c"),
    ]))
    body = []
    for _, r in merged.iterrows():
        sev_b = _smd_severity(r["smd_before"])
        sev_a = _smd_severity(r["smd_after"])
        delta = r["smd_after"] - r["smd_before"]
        if sev_a == "good":
            status = _chip("СБАЛАНС.", "success")
        elif sev_a == "warn":
            status = _chip("ПРИЕМЛЕМО", "warning")
        else:
            status = _chip("ДИСБАЛАНС", "danger")
        body.append(html.Tr([
            html.Td(html.Span(r["covariate"], className="kb-mono"),
                    className="kb-match-tbl__td"),
            html.Td(_fmt_num(r["mean_treatment"], 3),
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono"),
            html.Td(_fmt_num(r["mean_control"], 3),
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono"),
            html.Td(html.Span(_fmt_num(r["smd_before"], 3),
                              className=f"kb-match-smd-fill kb-match-smd-fill--{sev_b}"),
                    className="kb-match-tbl__td kb-match-tbl__td--r"),
            html.Td(html.Span(_fmt_num(r["smd_after"], 3),
                              className=f"kb-match-smd-fill kb-match-smd-fill--{sev_a}"),
                    className="kb-match-tbl__td kb-match-tbl__td--r"),
            html.Td(f"{delta:+.3f}",
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono",
                    style={"color": SMD_GOOD if delta < 0 else SMD_BAD,
                           "fontWeight": 500}),
            html.Td(status,
                    className="kb-match-tbl__td kb-match-tbl__td--c"),
        ]))
    return html.Table([head, html.Tbody(body)], className="kb-match-tbl")


def _data_preview(df: pd.DataFrame, n: int = 50) -> html.Table:
    """Compact preview of the matched DataFrame."""
    if df is None or df.empty:
        return html.Table()
    sub = df.head(n)
    head = html.Thead(html.Tr([
        html.Th(c, className="kb-match-tbl__th kb-mono")
        for c in sub.columns
    ]))
    body = []
    for _, r in sub.iterrows():
        cells = []
        for c in sub.columns:
            v = r[c]
            if pd.api.types.is_number(v) and not pd.isna(v):
                txt = _fmt_num(v, 3)
                cells.append(html.Td(txt,
                                     className="kb-match-tbl__td kb-mono kb-match-tbl__td--r"))
            else:
                cells.append(html.Td(("—" if pd.isna(v) else str(v)),
                                     className="kb-match-tbl__td"))
        body.append(html.Tr(cells))
    return html.Table([head, html.Tbody(body)], className="kb-match-tbl")


def _result_panel(result, work_df, treatment_col,
                  method_label: str, method_sev: str,
                  right_panel: html.Div,
                  active_subtab: str = "love",
                  extra_kpis: list[html.Div] | None = None) -> html.Div:
    """Shared result panel — KPI strip + 8/4 grid + sub-tabs (Love/Bal/Preview)."""
    bal_b = result.balance_before
    bal_a = result.balance_after
    n_t = int(result.n_treatment)
    n_c = int(result.n_control)
    n_match_t = int(result.n_matched_treatment)
    n_match_c = int(result.n_matched_control)
    smd_after = float(balance_summary(bal_a)["mean_abs_smd"])
    smd_before = float(balance_summary(bal_b)["mean_abs_smd"])
    bal_pct = float(balance_summary(bal_a)["pct_below_01"])
    n_improved = int(((bal_b["abs_smd"].values - bal_a["abs_smd"].values) > 0).sum())

    # Pre-balance compact strip
    pre_strip = html.Div([
        html.Span([
            html.Span("PRE-BALANCE: ", className="kb-match-prebal-lbl"),
            html.Span(f"{len(bal_b)} ковариат", className="kb-match-prebal-e"),
            html.Span("·", className="kb-match-prebal-sep"),
            html.Span("средн. |SMD| = ", className="kb-match-prebal-e"),
            html.Span(f"{smd_before:.3f}",
                      className="kb-match-prebal-num",
                      style={"color": SMD_BAD}),
            html.Span("·", className="kb-match-prebal-sep"),
            html.Span(
                f"{balance_summary(bal_b)['pct_below_01']:.0f}% < 0.1",
                className="kb-match-prebal-e"),
        ]),
        html.Span([icon("info", 12), " развернуть"],
                  className="kb-match-prebal-eye"),
    ], className="kb-match-prebal-strip")

    kpi_strip = _result_kpi_strip(
        method_label, method_sev,
        n_match_t, n_t, n_match_c, n_c,
        smd_after, smd_before, bal_pct,
        extras=extra_kpis,
    )

    # Sub-tabs: Love-plot / Balance Table / Data Preview
    sub_tabs = html.Div(
        dbc.Tabs(
            id="match-sub-tabs", active_tab=active_subtab,
            children=[
                dbc.Tab(label="Love-plot", tab_id="love"),
                dbc.Tab(label="Таблица баланса", tab_id="bal"),
                dbc.Tab(label="Превью данных", tab_id="prev"),
            ],
            className="kb-match-sub-tabs",
        ),
        className="kb-match-sub-tabs-wrap",
    )

    sub_content = html.Div(
        _balance_table(bal_b, bal_a),
        id="match-sub-tab-content",
        className="kb-match-sub-tab-content",
    )

    # Love-plot card
    love_card = _card([
        _card_head(
            "Love-plot · до и после",
            right=html.Span(
                f"{len(bal_b)} ковариат · {n_improved} улучшено",
                className="kb-match-card__submeta",
            ),
        ),
        html.Div(
            dcc.Graph(figure=_love_plot(bal_b, bal_a, height=320),
                      config={"displayModeBar": False}),
            className="kb-match-chart-frame",
        ),
        html.Div([
            html.Span([
                html.Span(className="kb-match-leg-sw kb-match-leg-sw--open"),
                "До сопоставления",
            ], className="kb-match-leg-tile"),
            html.Span([
                html.Span(className="kb-match-leg-sw",
                          style={"background": SMD_GOOD}),
                "После сопоставления",
            ], className="kb-match-leg-tile"),
        ], className="kb-match-leg-row"),
    ])

    sticky = html.Div([
        html.Button([icon("download", 13),
                     html.Span("Экспорт matched-датасета (CSV)")],
                    id="match-export-csv",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Button([icon("download", 13),
                     html.Span("Экспорт Love-plot (PNG)")],
                    id="match-export-png",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Span(className="kb-match-sticky__grow"),
        html.Button([icon("git-compare", 13),
                     html.Span("Сравнить с другим методом")],
                    id="match-go-compare",
                    className="kb-btn kb-btn--primary kb-btn--sm",
                    n_clicks=0),
    ], className="kb-match-sticky")

    return html.Div([
        pre_strip,
        _section_overline("РЕЗУЛЬТАТ СОПОСТАВЛЕНИЯ"),
        kpi_strip,
        html.Div([
            love_card,
            right_panel,
        ], className="kb-match-result-grid"),
        _card([sub_tabs, sub_content], pad=False),
        sticky,
    ])


# ---------------------------------------------------------------------------
# Run callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("psm-result", "children"),
    Output("match-results-store", "data", allow_duplicate=True),
    Input("btn-run-psm", "n_clicks"),
    State("match-treatment-col", "value"),
    State("match-covariates", "value"),
    State("psm-caliper", "value"),
    State("psm-ratio", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("match-results-store", "data"),
    prevent_initial_call=True,
)
def _run_psm(n, treatment, covs, caliper, ratio, ds_name, raw, prep, store):
    if not all([treatment, covs]):
        return alert_banner(
            "Выберите колонку лечения и хотя бы одну ковариату.",
            "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    work = _prepare_work_df(df, treatment)
    if work is None:
        return alert_banner("Невозможно подготовить датасет.", "danger"), no_update
    num_cov = [c for c in covs if pd.api.types.is_numeric_dtype(work[c])]
    if not num_cov:
        return alert_banner("PSM требует числовые ковариаты.",
                            "warning"), no_update
    try:
        result = propensity_score_match(
            work, treatment, num_cov,
            caliper=float(caliper or 0.20),
            ratio=int(ratio or 1),
        )
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

    # Right panel: propensity histogram
    ps = result.propensity_scores
    treated = work[treatment].values
    common = result.common_support or (0.0, 1.0)
    off_n = int(((ps < common[0]) | (ps > common[1])).sum()) if ps is not None else 0

    right = _card([
        _card_head("Распределение propensity-score"),
        html.Div(
            dcc.Graph(
                figure=_propensity_hist(ps, treated, height=240) if ps is not None else go.Figure(),
                config={"displayModeBar": False}),
            className="kb-match-chart-frame",
        ),
        html.Div([
            html.Span([
                html.Span(className="kb-match-cdot",
                          style={"background": MATCH_TREATMENT}),
                f"Treatment · {int((treated == 1).sum())}",
            ], className="kb-match-stat-chip"),
            html.Span([
                html.Span(className="kb-match-cdot",
                          style={"background": MATCH_CONTROL}),
                f"Control · {int((treated == 0).sum())}",
            ], className="kb-match-stat-chip"),
            _chip(
                f"COMMON SUPPORT: {common[0]:.2f}–{common[1]:.2f}",
                "neutral", mono=True),
            _chip(f"OFF-SUPPORT: {off_n} НАБЛЮДЕНИЙ",
                  "warning" if off_n else "success", mono=True),
        ], className="kb-match-stat-chip-list"),
    ])

    method_label = f"PSM 1:{int(ratio or 1)} · CALIPER={float(caliper or 0.20):.1f}"
    sev = ("good" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.1
           else "warn" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.25
           else "bad")

    panel = _result_panel(result, work, treatment,
                          method_label, sev, right)

    # Save in store for compare tab
    store = store or {}
    store["psm"] = _serialize_result(result, "PSM")
    return panel, store


@callback(
    Output("exact-result", "children"),
    Output("match-results-store", "data", allow_duplicate=True),
    Input("btn-run-exact", "n_clicks"),
    State("match-treatment-col", "value"),
    State("match-covariates", "value"),
    State("exact-cols", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("match-results-store", "data"),
    prevent_initial_call=True,
)
def _run_exact(n, treatment, covs, exact_cols, ds_name, raw, prep, store):
    if not treatment:
        return alert_banner("Выберите колонку лечения.",
                            "warning"), no_update
    if not exact_cols:
        return alert_banner(
            "Выберите хотя бы одну категориальную колонку для точного совпадения.",
            "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    work = _prepare_work_df(df, treatment)
    if work is None:
        return alert_banner("Невозможно подготовить датасет.", "danger"), no_update
    num_cov = [c for c in (covs or []) if pd.api.types.is_numeric_dtype(work[c])]
    try:
        result = exact_match(
            work, treatment, exact_cols,
            covariates=num_cov if num_cov else exact_cols,
        )
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

    # Build strata from result.matched_df
    matched = result.matched_df
    strata_rows = []
    if not matched.empty and exact_cols:
        gb = matched.groupby(exact_cols)[treatment]
        for key, grp in gb:
            t = int((grp == 1).sum())
            c = int((grp == 0).sum())
            name = " · ".join(str(k) for k in
                              (key if isinstance(key, tuple) else (key,)))
            strata_rows.append({"name": name, "t": t, "c": c})
        strata_rows.sort(key=lambda s: s["t"] + s["c"], reverse=True)
    n_strata = len(strata_rows)
    n_strata_total = (
        int(matched.groupby(exact_cols).ngroups) if not matched.empty else 0)
    n_skipped = max(0, n_strata_total - n_strata)
    avg_per_strat = (
        int(np.mean([s["t"] + s["c"] for s in strata_rows]))
        if strata_rows else 0)

    right = _card([
        _card_head("Размер страт"),
        html.Div(_stratum_bars(strata_rows[:10]),
                 className="kb-match-strat-frame"),
        html.Div([
            _chip(f"СРЕДНЕЕ НА СТРАТУ: {avg_per_strat}", "neutral", mono=True),
            _chip(f"ПРОПУЩЕННЫХ СТРАТ (size<2): {n_skipped}",
                  "warning" if n_skipped else "success", mono=True),
        ], className="kb-match-stat-chip-list"),
    ])

    method_label = f"EXACT · {n_strata} страт"
    sev = ("good" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.1
           else "warn" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.25
           else "bad")

    extra_kpis = [html.Div([
        html.Div("СТРАТ ОБРАЗОВАНО", className="kb-match-kpi__l"),
        html.Div(str(n_strata), className="kb-match-kpi__v"),
        html.Div(f"{n_strata - n_skipped} непустых после фильтра",
                 className="kb-match-kpi__s"),
    ], className="kb-match-kpi")]

    panel = _result_panel(result, work, treatment,
                          method_label, sev, right,
                          extra_kpis=extra_kpis)

    store = store or {}
    store["exact"] = _serialize_result(result, "Exact")
    return panel, store


@callback(
    Output("nn-result", "children"),
    Output("match-results-store", "data", allow_duplicate=True),
    Input("btn-run-nn", "n_clicks"),
    State("match-treatment-col", "value"),
    State("match-covariates", "value"),
    State("nn-k", "value"),
    State("nn-metric", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("match-results-store", "data"),
    prevent_initial_call=True,
)
def _run_nn(n, treatment, covs, k, metric, ds_name, raw, prep, store):
    if not all([treatment, covs]):
        return alert_banner(
            "Выберите колонку лечения и ковариаты.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    work = _prepare_work_df(df, treatment)
    if work is None:
        return alert_banner("Невозможно подготовить датасет.", "danger"), no_update
    num_cov = [c for c in covs if pd.api.types.is_numeric_dtype(work[c])]
    if not num_cov:
        return alert_banner("NN требует числовые ковариаты.",
                            "warning"), no_update
    try:
        result = nearest_neighbor_match(
            work, treatment, num_cov,
            n_neighbors=int(k or 1),
            metric=metric or "mahalanobis",
        )
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

    # Distance distribution from match_quality
    distances = np.asarray(result.match_quality.get("distances", []))
    if distances.size:
        d_min = float(distances.min())
        d_max = float(distances.max())
        d_mean = float(distances.mean())
        d_median = float(np.median(distances))
    else:
        d_min = d_max = d_mean = d_median = 0.0

    right = _card([
        _card_head("Распределение расстояний между парами"),
        html.Div(
            dcc.Graph(figure=_distance_hist(distances, height=240),
                      config={"displayModeBar": False}),
            className="kb-match-chart-frame",
        ),
        html.Div([
            html.Span([html.Span("MIN", className="kb-match-stat-chip__k"),
                       html.Span(f"{d_min:.3f}",
                                 className="kb-match-stat-chip__v")],
                      className="kb-match-stat-chip"),
            html.Span([html.Span("MEDIAN", className="kb-match-stat-chip__k"),
                       html.Span(f"{d_median:.3f}",
                                 className="kb-match-stat-chip__v",
                                 style={"color": ACCENT_300})],
                      className="kb-match-stat-chip"),
            html.Span([html.Span("MEAN", className="kb-match-stat-chip__k"),
                       html.Span(f"{d_mean:.3f}",
                                 className="kb-match-stat-chip__v")],
                      className="kb-match-stat-chip"),
            html.Span([html.Span("MAX", className="kb-match-stat-chip__k"),
                       html.Span(f"{d_max:.3f}",
                                 className="kb-match-stat-chip__v")],
                      className="kb-match-stat-chip"),
        ], className="kb-match-stat-chip-list"),
    ])

    method_label = f"NN k={int(k or 1)} · {(metric or 'mahalanobis').title()}"
    sev = ("good" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.1
           else "warn" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.25
           else "bad")

    panel = _result_panel(result, work, treatment,
                          method_label, sev, right)

    store = store or {}
    store["nn"] = _serialize_result(result, "NN")
    return panel, store


@callback(
    Output("cem-result", "children"),
    Output("match-results-store", "data", allow_duplicate=True),
    Input("btn-run-cem", "n_clicks"),
    State("match-treatment-col", "value"),
    State("match-covariates", "value"),
    State("cem-bins", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("match-results-store", "data"),
    prevent_initial_call=True,
)
def _run_cem(n, treatment, covs, bins, ds_name, raw, prep, store):
    if not all([treatment, covs]):
        return alert_banner(
            "Выберите колонку лечения и ковариаты.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    work = _prepare_work_df(df, treatment)
    if work is None:
        return alert_banner("Невозможно подготовить датасет.", "danger"), no_update
    num_cov = [c for c in covs if pd.api.types.is_numeric_dtype(work[c])]
    cov_use = num_cov if num_cov else (covs or [])
    if not cov_use:
        return alert_banner("Выберите ковариаты.", "warning"), no_update
    n_bins = int(bins or 5)
    try:
        result = coarsened_exact_match(
            work, treatment, cov_use, n_bins=n_bins,
        )
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

    # CEM-specific: bin grid + L1 distance
    ranges = {}
    for c in cov_use:
        try:
            ranges[c] = (float(work[c].min()), float(work[c].max()))
        except Exception:
            ranges[c] = (0.0, 0.0)
    l1_dist = float(result.match_quality.get("l1_distance", 0.0))

    right = _card([
        _card_head("Огрубление ковариат"),
        html.Div(_cem_bin_grid(cov_use[:6], n_bins, ranges),
                 className="kb-match-chart-frame"),
    ])

    method_label = f"CEM · {n_bins} бинов"
    sev = ("good" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.1
           else "warn" if balance_summary(result.balance_after)["mean_abs_smd"] < 0.25
           else "bad")

    extra_kpis = [html.Div([
        html.Div("L1-DISTANCE", className="kb-match-kpi__l"),
        html.Div(f"{l1_dist:.3f}", className="kb-match-kpi__v",
                 style={"color": SMD_GOOD if l1_dist < 0.5 else SMD_WARN}),
        html.Div("метрика дисбаланса CEM", className="kb-match-kpi__s"),
    ], className="kb-match-kpi")]

    panel = _result_panel(result, work, treatment,
                          method_label, sev, right,
                          extra_kpis=extra_kpis)

    store = store or {}
    store["cem"] = _serialize_result(result, "CEM")
    return panel, store


# ---------------------------------------------------------------------------
# Result serialization (for compare-tab store)
# ---------------------------------------------------------------------------

def _serialize_result(result, name: str) -> dict:
    """Pack a MatchResult into a JSON-serializable dict for the store."""
    return {
        "name": name,
        "method": result.method,
        "n_treatment": int(result.n_treatment),
        "n_control": int(result.n_control),
        "n_matched_treatment": int(result.n_matched_treatment),
        "n_matched_control": int(result.n_matched_control),
        "balance_before": result.balance_before.to_dict(orient="records"),
        "balance_after": result.balance_after.to_dict(orient="records"),
        "common_support": (list(result.common_support)
                           if result.common_support else None),
        "match_quality": {k: (float(v) if isinstance(v, (int, float, np.floating))
                              else v.tolist() if hasattr(v, "tolist")
                              else v)
                          for k, v in (result.match_quality or {}).items()},
    }


# ---------------------------------------------------------------------------
# 17.6 Сравнение методов
# ---------------------------------------------------------------------------

def _render_compare(results, df, treatment_col) -> html.Div:
    """Multi-method compare view — cards + multi-love-plot + summary table."""
    results = results or {}
    if not results:
        return html.Div([
            _empty_compare(),
        ])

    method_color = {"psm": M_PSM, "exact": M_EXACT,
                    "nn": M_NN, "cem": M_CEM}

    # Compute "best" method by smallest mean |SMD| after
    best_id = None
    best_smd = float("inf")
    for k, r in results.items():
        bal_a = pd.DataFrame(r["balance_after"])
        smd_a = float(balance_summary(bal_a)["mean_abs_smd"])
        if smd_a < best_smd:
            best_smd = smd_a
            best_id = k

    # Method cards
    cards = []
    for k, r in results.items():
        bal_a = pd.DataFrame(r["balance_after"])
        smd_a = float(balance_summary(bal_a)["mean_abs_smd"])
        bal_pct = float(balance_summary(bal_a)["pct_below_01"])
        cards.append(html.Div([
            html.Div(
                _chip("★ BEST", "success") if k == best_id else "",
                className="kb-match-method-badge",
            ),
            html.Div([
                html.Span(className="kb-match-method-marker",
                          style={"background": method_color.get(k, ACCENT_500)}),
                html.H3(r["name"], className="kb-match-method-title"),
            ], className="kb-match-method-top"),
            html.Div([
                html.Div([html.Span("|SMD|", className="kb-match-method-k"),
                          html.Span(f"{smd_a:.3f}",
                                    className="kb-match-method-v",
                                    style={"color": _smd_color(smd_a)})],
                         className="kb-match-method-row"),
                html.Div([html.Span("matched",
                                    className="kb-match-method-k"),
                          html.Span(f"{r['n_matched_treatment']}/"
                                    f"{r['n_treatment']}",
                                    className="kb-match-method-v")],
                         className="kb-match-method-row"),
                html.Div([html.Span("<0.1",
                                    className="kb-match-method-k"),
                          html.Span(f"{bal_pct:.0f}%",
                                    className="kb-match-method-v")],
                         className="kb-match-method-row"),
            ], className="kb-match-method-metrics"),
        ], className=f"kb-match-method-card kb-match-method-card--{k}"))

    cards_grid = html.Div(cards,
                          className="kb-match-method-cards "
                                    f"kb-match-method-cards--{len(cards)}")

    # Multi-love-plot
    bal_b_first = pd.DataFrame(next(iter(results.values()))["balance_before"])
    after_by_method = {
        k: pd.DataFrame(r["balance_after"]) for k, r in results.items()
    }
    multi_card = _card([
        _card_head(
            "Love-plot · все методы",
            right=_chip(
                f"{len(bal_b_first)} КОВАРИАТ · {len(results)} МЕТОДА",
                "neutral", mono=True),
        ),
        html.Div(
            dcc.Graph(
                figure=_love_multi(bal_b_first, after_by_method, height=380),
                config={"displayModeBar": False}),
            className="kb-match-chart-frame",
        ),
        html.Div([
            html.Span([
                html.Span(className="kb-match-leg-sw kb-match-leg-sw--open"),
                "До сопоставления",
            ], className="kb-match-leg-tile"),
            *[html.Span([
                html.Span(className="kb-match-leg-sw",
                          style={"background": method_color[k]}),
                k.upper(),
            ], className="kb-match-leg-tile") for k in results.keys()],
        ], className="kb-match-leg-row"),
    ])

    # Summary table
    summary = _compare_summary_table(results, method_color, best_id)

    sticky = html.Div([
        html.Button([icon("download", 13),
                     html.Span("Экспорт сводки CSV")],
                    id="match-cmp-export-csv",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Button([icon("download", 13),
                     html.Span("Экспорт Love-plot PNG")],
                    id="match-cmp-export-png",
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0),
        html.Span(className="kb-match-sticky__grow"),
        html.Button(
            [icon("check-circle", 13),
             html.Span("Использовать выбранный метод"),
             icon("chevron-down", 11)],
            id="match-cmp-use",
            className="kb-btn kb-btn--primary kb-btn--sm",
            n_clicks=0,
        ),
    ], className="kb-match-sticky")

    return html.Div([
        _section_overline("СРАВНЕНИЕ ВСЕХ ЗАПУЩЕННЫХ МЕТОДОВ"),
        cards_grid,
        multi_card,
        _card([
            _card_head("Сводная таблица методов",
                       right=_chip("★ ЛУЧШЕЕ ПО МЕТРИКЕ", "neutral", mono=True)),
            summary,
        ], pad=False),
        sticky,
    ], className="kb-match-compare")


def _empty_compare() -> html.Div:
    return html.Div(
        [
            html.Div(icon("git-compare", 32), className="kb-match-empty__icn"),
            html.H3("Запустите хотя бы один метод",
                    className="kb-match-empty__t"),
            html.P(
                "Сравнение методов появится автоматически после первого "
                "запуска PSM / Exact / NN / CEM. Запустите 2+ метода — и "
                "KIBAD покажет лучший по метрике |SMD|.",
                className="kb-match-empty__b",
            ),
        ],
        className="kb-match-empty",
    )


def _compare_summary_table(results, method_color, best_id) -> html.Table:
    rows = []
    for k, r in results.items():
        bal_b = pd.DataFrame(r["balance_before"])
        bal_a = pd.DataFrame(r["balance_after"])
        smd_b = float(balance_summary(bal_b)["mean_abs_smd"])
        smd_a = float(balance_summary(bal_a)["mean_abs_smd"])
        bal_pct = float(balance_summary(bal_a)["pct_below_01"])
        n_imp = int(((bal_b["abs_smd"].values - bal_a["abs_smd"].values) > 0).sum())
        l1 = r.get("match_quality", {}).get("l1_distance")
        rows.append({
            "id": k, "name": r["name"],
            "color": method_color.get(k, ACCENT_500),
            "tm": int(r["n_matched_treatment"]),
            "n_t": int(r["n_treatment"]),
            "tPct": (r["n_matched_treatment"] / r["n_treatment"] * 100
                     if r["n_treatment"] else 0),
            "cm": int(r["n_matched_control"]),
            "smdB": smd_b, "smdA": smd_a,
            "imp": n_imp, "tot": len(bal_b),
            "bal": bal_pct,
            "l1": (f"{l1:.3f}" if isinstance(l1, (int, float))
                   and l1 is not None else "—"),
        })
    if not rows:
        return html.Table()

    # Best per column
    best_smd = min(r["smdA"] for r in rows)
    best_t   = max(r["tm"] for r in rows)
    best_bal = max(r["bal"] for r in rows)

    head = html.Thead(html.Tr([
        html.Th("Метод", className="kb-match-tbl__th"),
        html.Th("Treat. matched",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("% treated",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("Control matched",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("|SMD| До",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("|SMD| После",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("Улучш. ков.",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("% <0.1",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
        html.Th("L1-distance",
                className="kb-match-tbl__th kb-match-tbl__th--r kb-mono"),
    ]))
    body = []
    for r in rows:
        def cell(val, is_best):
            cls = "kb-match-tbl__td kb-match-tbl__td--r kb-mono"
            kid = [val]
            if is_best:
                cls += " kb-match-tbl__td--best"
                kid = [val, " ", html.Span("★",
                                            className="kb-match-best-star")]
            return html.Td(kid, className=cls)
        body.append(html.Tr([
            html.Td(html.Span([
                html.Span(className="kb-match-cdot",
                          style={"background": r["color"]}),
                html.Span(r["name"], className="kb-mono",
                          style={"fontWeight": 500}),
            ], style={"display": "inline-flex", "alignItems": "center",
                      "gap": "8px"}),
                    className="kb-match-tbl__td"),
            cell(str(r["tm"]), r["tm"] == best_t),
            cell(f"{r['tPct']:.0f}%", r["tm"] == best_t),
            html.Td(str(r["cm"]),
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono"),
            html.Td(f"{r['smdB']:.3f}",
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono",
                    style={"color": SMD_BAD}),
            cell(f"{r['smdA']:.3f}", r["smdA"] == best_smd),
            html.Td(f"{r['imp']}/{r['tot']}",
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono"),
            cell(f"{r['bal']:.0f}%", r["bal"] == best_bal),
            html.Td(r["l1"],
                    className="kb-match-tbl__td kb-match-tbl__td--r kb-mono"),
        ]))
    # Average row
    avg_t  = np.mean([r["tm"] for r in rows])
    avg_pt = np.mean([r["tPct"] for r in rows])
    avg_c  = np.mean([r["cm"] for r in rows])
    avg_sB = np.mean([r["smdB"] for r in rows])
    avg_sA = np.mean([r["smdA"] for r in rows])
    avg_im = np.mean([r["imp"] for r in rows])
    avg_b  = np.mean([r["bal"] for r in rows])
    foot = html.Tr([
        html.Td("Среднее",
                className="kb-match-tbl__td kb-match-tbl__td--foot",
                style={"fontWeight": 500}),
        html.Td(f"{avg_t:.1f}",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td(f"{avg_pt:.1f}%",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td(f"{avg_c:.1f}",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td(f"{avg_sB:.3f}",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td(f"{avg_sA:.3f}",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td(f"{avg_im:.2f}/{rows[0]['tot']}",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td(f"{avg_b:.1f}%",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
        html.Td("—",
                className="kb-match-tbl__td kb-match-tbl__td--foot kb-match-tbl__td--r kb-mono"),
    ])
    return html.Table([head, html.Tbody(body + [foot])],
                      className="kb-match-tbl")

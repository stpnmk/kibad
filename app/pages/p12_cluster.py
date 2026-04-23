"""p12_cluster — Clustering page (Dash).

Redesigned for KIBAD Design System v2026.04 (dark eucalyptus).
Mirrors handoff slide 12 ("Кластеризация") — 5 artboards rendered as
progressive stages of a single page:
  12.1 Настройка · пустое состояние (когда признаков < 2)
  12.2 Метод локтя · dual-axis Inertia + Silhouette по K
  12.3 Обзор · KPI-полоска + bar-распределение + PCA-скаттер
  12.4 Профили · радарная диаграмма + heatmap центроидов
  12.5 Таблица · сводная таблица с cmarkers + share-bar + mean±std
"""
from __future__ import annotations

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.alerts import alert_banner
from app.components.icons import icon
from core.cluster import run_kmeans, run_elbow, cluster_profiles, pca_transform
from core.audit import log_event

dash.register_page(__name__, path="/cluster", name="12. Кластеризация",
                   order=12, icon="people")


# ---------------------------------------------------------------------------
# Design tokens — per-cluster palette (matches cluster-artboards.js)
# ---------------------------------------------------------------------------

CLUSTER_COLORS = [
    "#21A066",  # accent-500
    "#4A7FB0",  # viz-2 (blue)
    "#C98A2E",  # warning (amber)
    "#C8503B",  # danger (terracotta)
    "#A066C8",  # viz-4 (violet)
    "#6B8E8A",  # viz-6 (teal)
    "#4FD18B",  # accent-300 (mint)
    "#7FB0D9",  # viz-2 light
]


def _cc(i: int) -> str:
    """Cluster colour cycled through ``CLUSTER_COLORS``."""
    return CLUSTER_COLORS[i % len(CLUSTER_COLORS)]


def _fmt_int(n: int | float) -> str:
    try:
        return f"{int(n):,}".replace(",", " ")
    except Exception:
        return str(n)


def _fmt2(v: float | int | None) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{float(v):.2f}"


def _fmt_signed(v: float) -> str:
    return f"{v:+.2f}"


def _load_df(prepared, datasets, ds):
    df = get_df_from_store(prepared, ds)
    if df is None:
        df = get_df_from_store(datasets, ds)
    return df


# ---------------------------------------------------------------------------
# Page head — overline + title + actions (match 12.0 design)
# ---------------------------------------------------------------------------

def _page_head() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("KIBAD · Аналитическая студия",
                             className="kb-overline"),
                    html.H1("Раздел 12 · Кластеризация",
                            className="kb-page-title"),
                    html.Div(
                        "K-Means, Метод локтя, PCA-визуализация — 5 артбордов анализа",
                        className="kb-page-subtitle",
                    ),
                ],
                className="kb-page-head-left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("download", 14), html.Span("Экспорт модели")],
                        id="cl-export-btn",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0, disabled=True,
                    ),
                    html.Button(
                        [icon("target", 14), html.Span("Кластеризовать")],
                        id="cl-run-btn-head",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-page-head-actions",
            ),
        ],
        className="kb-page-head",
    )


# ---------------------------------------------------------------------------
# Dataset picker row — dark 44-px bar
# ---------------------------------------------------------------------------

def _ds_picker() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon("database", 18),
                              style={"color": "var(--accent-300)"}),
                    html.Div(
                        dcc.Dropdown(
                            id="cl-ds",
                            placeholder="Выберите датасет…",
                            clearable=False,
                            className="kb-select kb-select--ds",
                        ),
                        className="kb-ds-picker__select",
                    ),
                ],
                className="kb-ds-picker__left",
            ),
            html.Div(id="cl-ds-meta", className="kb-ds-picker__meta"),
        ],
        className="kb-ds-picker",
    )


# ---------------------------------------------------------------------------
# Config card — features + K + elbow/run buttons (always visible)
# ---------------------------------------------------------------------------

def _config_card() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon("settings", 16),
                              style={"color": "var(--accent-500)"}),
                    html.H3("Настройки кластеризации",
                            className="kb-cluster-cfg__title"),
                ],
                className="kb-cluster-cfg__head",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                id="cl-features-label",
                                className="kb-field-label",
                            ),
                            dcc.Dropdown(
                                id="cl-features", multi=True,
                                placeholder="Выберите ≥ 2 числовых колонки…",
                                className="kb-select kb-select--multi kb-cluster-feats",
                            ),
                        ],
                        className="kb-field",
                    ),
                    html.Div(
                        [
                            html.Label("КЛАСТЕРОВ (K)",
                                       className="kb-field-label"),
                            dcc.Input(
                                id="cl-k", type="number",
                                value=3, min=2, max=20, step=1,
                                className="kb-input kb-input--num mono",
                            ),
                        ],
                        className="kb-field",
                    ),
                    html.Div(
                        [
                            html.Label("\u00A0",
                                       className="kb-field-label"),
                            html.Button(
                                [icon("chart", 14), html.Span("Метод локтя")],
                                id="cl-elbow-btn",
                                className="kb-btn kb-btn--ghost",
                                n_clicks=0,
                            ),
                        ],
                        className="kb-field",
                    ),
                    html.Div(
                        [
                            html.Label("\u00A0",
                                       className="kb-field-label"),
                            html.Button(
                                [icon("target", 14),
                                 html.Span("Кластеризовать")],
                                id="cl-run-btn",
                                className="kb-btn kb-btn--primary",
                                n_clicks=0,
                            ),
                        ],
                        className="kb-field",
                    ),
                ],
                className="kb-cluster-cfg__grid",
            ),
        ],
        className="kb-card kb-cluster-cfg",
    )


# ---------------------------------------------------------------------------
# Empty-big state (12.1)
# ---------------------------------------------------------------------------

def _empty_big() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Span(icon("target", 40),
                              className="kb-cluster-empty__icn")
                ],
                className="kb-cluster-empty__circle",
            ),
            html.H2("Готово к кластеризации",
                    className="kb-cluster-empty__h"),
            html.Div(
                "Выберите датасет и числовые признаки, затем нажмите "
                "«Метод локтя» для подбора оптимального K или "
                "«Кластеризовать» для запуска с текущим K.",
                className="kb-cluster-empty__body",
            ),
            html.Div(
                [
                    html.Span(
                        [icon("info", 10),
                         html.Span("СОВЕТ: 3–8 КЛАСТЕРОВ ОПТИМАЛЬНО")],
                        className="kb-chip kb-chip--info",
                    ),
                    html.Span(
                        [icon("info", 10),
                         html.Span("СОВЕТ: SILHOUETTE ≥ 0.5 ХОРОШО")],
                        className="kb-chip kb-chip--info",
                    ),
                ],
                className="kb-cluster-empty__tips",
            ),
        ],
        className="kb-cluster-empty",
    )


# ---------------------------------------------------------------------------
# Elbow chart (12.2) — dual y-axis: Inertia (viz-2, solid) + Silhouette
# (accent-500, dashed with diamond markers). Vertical warning dashed line at
# best K. Annotation badge "РЕКОМЕНДУЕМЫЙ K = N".
# ---------------------------------------------------------------------------

def _best_k(elbow_df: pd.DataFrame) -> int:
    """Pick best K by highest silhouette; fall back to 1st K on ties/NaN."""
    try:
        sub = elbow_df.dropna(subset=["silhouette"])
        if not sub.empty:
            return int(sub.loc[sub["silhouette"].idxmax(), "k"])
    except Exception:
        pass
    return int(elbow_df["k"].iloc[0])


def _elbow_fig(elbow_df: pd.DataFrame, best_k: int) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    Ks = elbow_df["k"].tolist()

    # Inertia trace — solid accent-blue on left axis
    fig.add_trace(
        go.Scatter(
            x=Ks, y=elbow_df["inertia"],
            mode="lines+markers",
            line=dict(color="#4A7FB0", width=2),
            marker=dict(color="#0A0F0D", size=8,
                        line=dict(color="#4A7FB0", width=2)),
            name="Inertia (SSE)",
            hovertemplate="K=%{x}<br>Inertia=%{y:.0f}<extra></extra>",
        ),
        secondary_y=False,
    )

    # Silhouette — dashed accent-green with diamond markers on right axis
    fig.add_trace(
        go.Scatter(
            x=Ks, y=elbow_df["silhouette"],
            mode="lines+markers",
            line=dict(color="#21A066", width=2, dash="dash"),
            marker=dict(color="#0A0F0D", size=9, symbol="diamond",
                        line=dict(color="#21A066", width=2)),
            name="Silhouette",
            hovertemplate="K=%{x}<br>Silhouette=%{y:.3f}<extra></extra>",
        ),
        secondary_y=True,
    )

    # Vertical recommended-K line
    fig.add_shape(
        type="line", x0=best_k, x1=best_k, y0=0, y1=1,
        xref="x", yref="paper",
        line=dict(color="#C98A2E", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=best_k, y=1.02, xref="x", yref="paper",
        text=f"<b>РЕКОМЕНДУЕМЫЙ K = {best_k}</b>",
        showarrow=False,
        font=dict(color="#E3A953", family="JetBrains Mono",
                  size=11),
        bgcolor="#0F1613", bordercolor="#C98A2E", borderwidth=1,
        borderpad=4, xanchor="left",
    )

    apply_kibad_theme(fig)
    fig.update_layout(
        height=380,
        margin=dict(l=60, r=60, t=40, b=36),
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="K · число кластеров",
                   font=dict(size=11, color="#A3B0A8")),
        tickmode="array", tickvals=Ks,
        tickfont=dict(size=10, color="#A3B0A8", family="JetBrains Mono"),
        showgrid=False, zeroline=False,
        linecolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        title=dict(text="Inertia (SSE)",
                   font=dict(size=11, color="#4A7FB0")),
        tickfont=dict(size=10, color="#A3B0A8", family="JetBrains Mono"),
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        zeroline=False, secondary_y=False,
    )
    fig.update_yaxes(
        title=dict(text="Silhouette",
                   font=dict(size=11, color="#21A066")),
        tickfont=dict(size=10, color="#A3B0A8", family="JetBrains Mono"),
        showgrid=False, zeroline=False, secondary_y=True,
        range=[0, max(0.6, float(elbow_df["silhouette"].max(skipna=True) or 0) * 1.15)],
    )
    return fig


def _render_elbow(elbow_df: pd.DataFrame) -> html.Div:
    best = _best_k(elbow_df)
    best_sil = float(
        elbow_df.loc[elbow_df["k"] == best, "silhouette"].iloc[0]
    )
    fig = _elbow_fig(elbow_df, best)

    # Quality verdict copy
    if best_sil >= 0.5:
        verdict_body = (
            f"Silhouette = {best_sil:.3f} — хорошее разделение "
            "(> 0.5). Точка перегиба на Inertia совпадает с максимумом "
            "Silhouette."
        )
        alert_cls = "kb-alert kb-alert--success"
        alert_icon = ("check", "var(--accent-500)")
    elif best_sil >= 0.25:
        verdict_body = (
            f"Silhouette = {best_sil:.3f} — приемлемое разделение "
            "(0.25–0.5). Точка перегиба найдена; рассмотрите альтернативные K."
        )
        alert_cls = "kb-alert kb-alert--warning"
        alert_icon = ("alert-triangle", "var(--warning)")
    else:
        verdict_body = (
            f"Silhouette = {best_sil:.3f} — слабое разделение "
            "(< 0.25). Попробуйте изменить набор признаков или нормализацию."
        )
        alert_cls = "kb-alert kb-alert--danger"
        alert_icon = ("x-circle", "var(--danger)")

    legend = html.Div(
        [
            html.Span(
                [
                    html.Span(className="kb-legend-dot",
                              style={"background": "#4A7FB0"}),
                    "Inertia (SSE) · левая ось",
                ],
                className="kb-legend-pill",
            ),
            html.Span(
                [
                    html.Span(
                        className="kb-legend-dot kb-legend-dot--diamond",
                        style={"background": "#21A066"},
                    ),
                    "Silhouette · правая ось",
                ],
                className="kb-legend-pill",
            ),
            html.Span(
                [
                    html.Span(
                        className="kb-legend-dot kb-legend-dot--bar",
                        style={"background": "#C98A2E"},
                    ),
                    "Рекомендуемый K",
                ],
                className="kb-legend-pill",
            ),
        ],
        className="kb-legend-row",
    )

    k_min = int(elbow_df["k"].min())
    k_max = int(elbow_df["k"].max())

    return html.Div(
        [
            html.Div("МЕТОД ЛОКТЯ",
                     className="kb-overline kb-cluster-section-overline"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.H3("Inertia vs Silhouette по K"),
                                    html.Div(
                                        [
                                            html.Span(
                                                f"K: {k_min} — {k_max}",
                                                className="kb-chip kb-chip--neutral mono",
                                            ),
                                            html.Span(
                                                f"BEST K = {best}",
                                                className="kb-chip kb-chip--success mono",
                                            ),
                                        ],
                                        className="kb-cluster-chips",
                                    ),
                                ],
                                className="kb-card-title-row",
                            ),
                        ],
                        className="kb-card-head",
                    ),
                    html.Div(
                        dcc.Graph(figure=fig, className="kb-eda-graph",
                                  config={"displayModeBar": False}),
                        className="kb-cluster-chart-wrap",
                    ),
                    legend,
                ],
                className="kb-card kb-cluster-elbow-card",
            ),
            html.Div(
                [
                    html.Span(icon(alert_icon[0], 16),
                              className="kb-alert__ic",
                              style={"color": alert_icon[1]}),
                    html.Div(
                        [
                            html.Div(
                                f"Рекомендованный K = {best} "
                                f"(по Silhouette = {best_sil:.3f})",
                                className="kb-alert__title",
                            ),
                            html.Div(verdict_body,
                                     className="kb-alert__body"),
                        ],
                    ),
                ],
                className=alert_cls,
            ),
        ],
        className="kb-cluster-section",
    )


# ---------------------------------------------------------------------------
# Distribution bar chart (12.3-left): counts per cluster
# ---------------------------------------------------------------------------

def _dist_bar_fig(counts: pd.Series) -> go.Figure:
    colors = [_cc(int(i)) for i in counts.index]
    total = int(counts.sum())
    fig = go.Figure(
        data=go.Bar(
            x=[f"C{i}" for i in counts.index],
            y=counts.values,
            marker=dict(color=colors, line=dict(width=0)),
            text=[_fmt_int(v) for v in counts.values],
            textposition="outside",
            textfont=dict(size=11, color="#E8EFEA",
                          family="JetBrains Mono"),
            hovertemplate="%{x}<br>N = %{y}<br>%{customdata:.1f}%<extra></extra>",
            customdata=(counts.values / max(total, 1) * 100),
        )
    )
    apply_kibad_theme(fig)
    fig.update_layout(
        height=300, margin=dict(l=48, r=16, t=24, b=36),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text="Кластер",
                   font=dict(size=10, color="#6B7A72")),
        tickfont=dict(size=11, color="#A3B0A8",
                      family="JetBrains Mono"),
        showgrid=False, linecolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        title=dict(text="Точек", font=dict(size=10, color="#6B7A72")),
        tickfont=dict(size=10, color="#A3B0A8",
                      family="JetBrains Mono"),
        showgrid=True, gridcolor="rgba(255,255,255,0.06)",
        zeroline=False,
    )
    return fig


# ---------------------------------------------------------------------------
# PCA scatter (12.3-right)
# ---------------------------------------------------------------------------

def _pca_fig(pca_df: pd.DataFrame, labels: np.ndarray,
             explained: tuple[float, ...]) -> go.Figure:
    fig = go.Figure()
    unique = sorted(set(int(x) for x in labels))
    for c in unique:
        mask = labels == c
        fig.add_trace(
            go.Scattergl(
                x=pca_df.loc[mask, "pca_1"],
                y=pca_df.loc[mask, "pca_2"],
                mode="markers",
                marker=dict(color=_cc(c), size=5.5, opacity=0.62,
                            line=dict(width=0)),
                name=f"C{c}",
                hovertemplate=("PC1=%{x:.2f}<br>PC2=%{y:.2f}"
                               f"<br>кластер=C{c}<extra></extra>"),
            )
        )
    apply_kibad_theme(fig)
    ev1 = (explained[0] * 100) if len(explained) > 0 else 0.0
    ev2 = (explained[1] * 100) if len(explained) > 1 else 0.0
    fig.update_layout(
        height=300,
        margin=dict(l=48, r=16, t=16, b=48),
        showlegend=False,
    )
    fig.update_xaxes(
        title=dict(text=f"PC1 · {ev1:.1f}% →",
                   font=dict(size=10, color="#6B7A72")),
        zeroline=True, zerolinecolor="rgba(255,255,255,0.12)",
        zerolinewidth=1,
        tickfont=dict(size=9, color="#A3B0A8",
                      family="JetBrains Mono"),
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
    )
    fig.update_yaxes(
        title=dict(text=f"↑ PC2 · {ev2:.1f}%",
                   font=dict(size=10, color="#6B7A72")),
        zeroline=True, zerolinecolor="rgba(255,255,255,0.12)",
        zerolinewidth=1,
        tickfont=dict(size=9, color="#A3B0A8",
                      family="JetBrains Mono"),
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
    )
    return fig


# ---------------------------------------------------------------------------
# Radar chart (12.4-left): normalized features per cluster
# ---------------------------------------------------------------------------

def _radar_fig(centers_df: pd.DataFrame,
               features: list[str]) -> go.Figure:
    # Min-max normalise centroids across clusters per feature.
    norm = centers_df[features].copy()
    for col in features:
        col_min = norm[col].min()
        col_max = norm[col].max()
        span = col_max - col_min
        if span == 0 or np.isnan(span):
            norm[col] = 0.5
        else:
            norm[col] = (norm[col] - col_min) / span

    fig = go.Figure()
    theta = features + [features[0]]  # close polygon
    for i, row in norm.iterrows():
        r = row[features].tolist() + [row[features[0]]]
        colour = _cc(int(i))
        fig.add_trace(
            go.Scatterpolar(
                r=r, theta=theta, fill="toself",
                fillcolor=colour,
                opacity=0.20,
                line=dict(color=colour, width=1.8),
                marker=dict(color=colour, size=5),
                name=f"C{int(i)}",
                hovertemplate="C%{fullData.name}<br>%{theta}: %{r:.2f}<extra></extra>",
            )
        )
    apply_kibad_theme(fig)
    fig.update_layout(
        height=360, margin=dict(l=30, r=30, t=20, b=20),
        showlegend=False,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["0.25", "0.50", "0.75", "1.00"],
                tickfont=dict(size=9, color="#6B7A72",
                              family="JetBrains Mono"),
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
                showline=False,
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#A3B0A8",
                              family="JetBrains Mono"),
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
            ),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Centroid heatmap (12.4-right): z-score matrix with diverging fill.
# Rendered as CSS grid (same approach as EDA corr heatmap), no Plotly.
# ---------------------------------------------------------------------------

_SURF_RGB = (20, 28, 24)
_RED_RGB  = (200, 80, 59)      # danger
_BLU_RGB  = (74, 127, 176)     # info / viz-2


def _mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(round(a + (b - a) * t) for a, b in zip(c1, c2))


def _z_cell_bg(v: float) -> str:
    """Diverging scale: blue (+) → surface (0) → red (−)."""
    t = max(-2.0, min(2.0, float(v))) / 2.0  # −1..+1
    if t >= 0:
        rgb = _mix(_SURF_RGB, _BLU_RGB, t)
    else:
        rgb = _mix(_SURF_RGB, _RED_RGB, -t)
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def _z_cell_fg(v: float) -> str:
    return "#E8EFEA" if abs(v) > 0.9 else "#A3B0A8"


def _centroid_heatmap(centers_df: pd.DataFrame,
                      features: list[str]) -> html.Div:
    """Standardise centroids per column into z-scores and draw as grid."""
    Z = centers_df[features].copy().astype(float)
    for col in features:
        mean = Z[col].mean()
        std = Z[col].std(ddof=0)
        if not std or np.isnan(std):
            Z[col] = 0.0
        else:
            Z[col] = (Z[col] - mean) / std

    n = len(features)
    cols_css = f"70px repeat({n}, minmax(0, 1fr))"
    children: list = [html.Div(className="kb-cheat-cell kb-cheat-head")]
    for f in features:
        children.append(
            html.Div(f, className="kb-cheat-cell kb-cheat-head",
                     title=f)
        )

    for ci in Z.index:
        row_idx = int(ci)
        children.append(
            html.Div(
                [
                    html.Span(className="kb-cheat-dot",
                              style={"background": _cc(row_idx)}),
                    html.Span(f"C{row_idx}", className="mono"),
                ],
                className="kb-cheat-cell kb-cheat-rowhd",
            )
        )
        for f in features:
            v = float(Z.at[ci, f])
            children.append(
                html.Div(
                    _fmt_signed(v),
                    className="kb-cheat-cell kb-cheat-cell--v mono",
                    style={
                        "background": _z_cell_bg(v),
                        "color": _z_cell_fg(v),
                    },
                    title=f"C{row_idx} · {f}: {v:+.3f}",
                )
            )

    grid = html.Div(
        children,
        className="kb-cheat",
        style={"gridTemplateColumns": cols_css},
    )
    # Vertical colorbar
    bar = html.Div(
        [
            html.Div("+2.0", className="kb-cheat-cbar__tick"),
            html.Div(className="kb-cheat-cbar__gradient"),
            html.Div("-2.0", className="kb-cheat-cbar__tick"),
            html.Div("ЦЕНТРОИД (Z)",
                     className="kb-cheat-cbar__label mono"),
        ],
        className="kb-cheat-cbar",
    )
    return html.Div([grid, bar], className="kb-cheat-wrap")


# ---------------------------------------------------------------------------
# Cluster profile table (12.5)
# ---------------------------------------------------------------------------

def _cluster_marker(ci: int) -> html.Span:
    return html.Span(
        [
            html.Span(str(ci), className="kb-cmarker__ring",
                      style={"background": _cc(ci)}),
            html.Span(f"C{ci}", className="kb-cmarker__name mono"),
        ],
        className="kb-cmarker",
    )


def _share_bar(pct: float, max_pct: float) -> html.Div:
    width = 0 if max_pct <= 0 else max(0.0, min(100.0,
                                                pct / max_pct * 100.0))
    return html.Div(
        [
            html.Div(className="kb-share-bar__fill",
                     style={"width": f"{width:.1f}%"}),
            html.Div(f"{pct:.1f}%", className="kb-share-bar__lbl mono"),
        ],
        className="kb-share-bar",
    )


def _profile_table(profiles: pd.DataFrame,
                   features: list[str]) -> html.Div:
    """Render the full profile table with cluster markers + share-bars."""
    if profiles.empty:
        return html.Div(
            "Нет данных для построения профилей.",
            className="kb-empty-state",
        )

    max_pct = float(profiles["cluster_pct"].max()) if "cluster_pct" in profiles else 1.0
    total_n = int(profiles["cluster_size"].sum())

    # Header rows
    feat_header_cells = [
        html.Th(
            [
                html.Div(f, className="kb-tbl-feat__name"),
                html.Div("mean / std", className="kb-tbl-feat__hint mono"),
            ],
            style={"minWidth": "140px"},
        )
        for f in features
    ]
    thead = html.Thead([
        html.Tr(
            [
                html.Th("Кластер", style={"width": "100px"}),
                html.Th("N", className="mono",
                        style={"width": "80px", "textAlign": "right"}),
                html.Th("Доля", style={"width": "160px"}),
            ] + feat_header_cells
        ),
    ])

    # Body
    body_rows: list = []
    for ci in profiles.index:
        row = profiles.loc[ci]
        feat_cells = []
        for f in features:
            mean = row.get(f"{f}_mean", np.nan)
            std = row.get(f"{f}_std", np.nan)
            feat_cells.append(
                html.Td(
                    [
                        html.Span(_fmt2(mean),
                                  className="kb-tbl-feat__mean mono"),
                        html.Span(f"±{_fmt2(std)}",
                                  className="kb-tbl-feat__std mono"),
                    ],
                    style={"textAlign": "right"},
                )
            )
        body_rows.append(
            html.Tr(
                [
                    html.Td(_cluster_marker(int(ci))),
                    html.Td(_fmt_int(int(row.get("cluster_size", 0))),
                            className="mono",
                            style={"textAlign": "right"}),
                    html.Td(_share_bar(
                        float(row.get("cluster_pct", 0.0)), max_pct)),
                ] + feat_cells
            )
        )

    # Footer: total row with grand stats
    total_cells = []
    for f in features:
        means = profiles[f"{f}_mean"] if f"{f}_mean" in profiles else None
        stds = profiles[f"{f}_std"] if f"{f}_std" in profiles else None
        if means is None:
            total_cells.append(html.Td("—", className="mono"))
            continue
        total_cells.append(
            html.Td(
                [
                    html.Span(_fmt2(means.mean()),
                              className="kb-tbl-feat__mean mono"),
                    html.Span(
                        f"±{_fmt2(stds.mean() if stds is not None else np.nan)}",
                        className="kb-tbl-feat__std mono"),
                ],
                style={"textAlign": "right"},
            )
        )
    footer = html.Tr(
        [
            html.Td("Всего", className="kb-tbl-total__lbl"),
            html.Td(_fmt_int(total_n), className="mono",
                    style={"textAlign": "right"}),
            html.Td("— · 100%", className="mono kb-tbl-total__share"),
        ] + total_cells,
        className="kb-tbl-total",
    )

    table = html.Table(
        [thead, html.Tbody(body_rows + [footer])],
        className="kb-tbl kb-cluster-tbl",
    )

    return html.Div(
        [
            html.Div("СВОДНАЯ ТАБЛИЦА ПО КЛАСТЕРАМ",
                     className="kb-overline kb-cluster-section-overline"),
            html.Div(
                html.Div(table, className="kb-cluster-tbl-scroll"),
                className="kb-card kb-cluster-tbl-card",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span(
                                [icon("rows", 10),
                                 html.Span(
                                     f"{len(profiles)} КЛАСТЕРОВ · "
                                     f"{_fmt_int(total_n)} ОБЪЕКТОВ")],
                                className="kb-chip kb-chip--neutral",
                            ),
                            html.Span("STD — В СКОБКАХ",
                                      className="kb-chip kb-chip--success"),
                        ],
                        className="kb-cluster-tbl-foot__left",
                    ),
                ],
                className="kb-cluster-tbl-foot",
            ),
        ],
        className="kb-cluster-section",
    )


# ---------------------------------------------------------------------------
# 12.3-Overview + 12.4-Profiles + 12.5-Table combined
# ---------------------------------------------------------------------------

def _render_results(
    result, profiles: pd.DataFrame,
    pca_df: pd.DataFrame, explained: tuple[float, ...],
    features: list[str],
) -> html.Div:
    labels = result.labels
    counts = pd.Series(labels).value_counts().sort_index()

    sil = result.silhouette
    sil_tone = (
        "success" if (sil is not None and sil >= 0.5) else
        "warning" if (sil is not None and sil >= 0.25) else
        "danger"
    )
    sil_verbose = (
        "хорошее разделение" if sil_tone == "success" else
        "приемлемое" if sil_tone == "warning" else "слабое"
    )

    kpi_tiles = html.Div(
        [
            _kpi_tile("Кластеров", str(int(result.n_clusters)),
                      "k-means · scaled"),
            _kpi_tile("Объектов", _fmt_int(int(len(labels))),
                      "все строки"),
            _kpi_tile("Признаков", str(len(features)),
                      "стандартизованы"),
            _kpi_tile(
                "Силуэт", _fmt2(sil), sil_verbose, tone=sil_tone,
            ),
        ],
        className="kb-eda-kpis kb-cluster-kpis",
    )

    if sil_tone == "success":
        alert_children = [
            html.Span(icon("check", 16), className="kb-alert__ic",
                      style={"color": "var(--accent-500)"}),
            html.Div(
                [
                    html.Div("Хорошее разделение",
                             className="kb-alert__title"),
                    html.Div(
                        f"Silhouette = {_fmt2(sil)} — кластеры обособлены. "
                        "Модель готова к сохранению и применению.",
                        className="kb-alert__body",
                    ),
                ],
            ),
        ]
        alert_cls = "kb-alert kb-alert--success"
    elif sil_tone == "warning":
        alert_children = [
            html.Span(icon("alert-triangle", 16),
                      className="kb-alert__ic",
                      style={"color": "var(--warning)"}),
            html.Div(
                [
                    html.Div("Приемлемое разделение",
                             className="kb-alert__title"),
                    html.Div(
                        f"Silhouette = {_fmt2(sil)} — кластеры частично "
                        "пересекаются. Попробуйте увеличить K или "
                        "пересмотреть набор признаков.",
                        className="kb-alert__body",
                    ),
                ],
            ),
        ]
        alert_cls = "kb-alert kb-alert--warning"
    else:
        alert_children = [
            html.Span(icon("x-circle", 16),
                      className="kb-alert__ic",
                      style={"color": "var(--danger)"}),
            html.Div(
                [
                    html.Div("Слабое разделение",
                             className="kb-alert__title"),
                    html.Div(
                        f"Silhouette = {_fmt2(sil)} — кластеры плохо "
                        "различимы. Рассмотрите другой набор признаков.",
                        className="kb-alert__body",
                    ),
                ],
            ),
        ]
        alert_cls = "kb-alert kb-alert--danger"
    sil_alert = html.Div(alert_children, className=alert_cls)

    # 12.3 · Overview cards
    total = int(counts.sum())
    overview_grid = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Div(
                            [
                                html.H3("Распределение точек по кластерам"),
                                html.Span(
                                    f"TOTAL {_fmt_int(total)}",
                                    className="kb-chip kb-chip--neutral mono",
                                ),
                            ],
                            className="kb-card-title-row",
                        ),
                        className="kb-card-head",
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=_dist_bar_fig(counts),
                            className="kb-eda-graph",
                            config={"displayModeBar": False},
                        ),
                        className="kb-cluster-chart-wrap",
                    ),
                ],
                className="kb-card",
            ),
            html.Div(
                [
                    html.Div(
                        html.Div(
                            [
                                html.H3("PCA-визуализация кластеров"),
                                html.Span(
                                    "ОБЪЯСН. ДИСПЕРСИЯ "
                                    f"{(sum(explained[:2])*100):.1f}%",
                                    className="kb-chip kb-chip--neutral mono",
                                ),
                            ],
                            className="kb-card-title-row",
                        ),
                        className="kb-card-head",
                    ),
                    html.Div(
                        dcc.Graph(
                            figure=_pca_fig(pca_df, labels, explained),
                            className="kb-eda-graph",
                            config={"displayModeBar": False},
                        ),
                        className="kb-cluster-chart-wrap",
                    ),
                    html.Div(
                        [
                            html.Span(
                                [
                                    html.Span(
                                        className="kb-legend-dot",
                                        style={"background": _cc(int(i))},
                                    ),
                                    f"cluster {int(i)}",
                                ],
                                className="kb-legend-pill",
                            )
                            for i in sorted(set(int(x) for x in labels))
                        ],
                        className="kb-legend-row kb-legend-row--centered",
                    ),
                ],
                className="kb-card",
            ),
        ],
        className="kb-cluster-grid kb-cluster-grid--2",
    )

    # 12.4 · Profiles (radar + heatmap)
    try:
        profiles_grid = html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            html.Div(
                                [
                                    html.H3(
                                        "Радарная диаграмма · "
                                        "нормализованные признаки"
                                    ),
                                    html.Span(
                                        f"{len(features)} ОСЕЙ · "
                                        f"{int(result.n_clusters)} КЛАСТЕРОВ",
                                        className="kb-chip kb-chip--neutral mono",
                                    ),
                                ],
                                className="kb-card-title-row",
                            ),
                            className="kb-card-head",
                        ),
                        html.Div(
                            dcc.Graph(
                                figure=_radar_fig(result.centers_df, features),
                                className="kb-eda-graph",
                                config={"displayModeBar": False},
                            ),
                            className="kb-cluster-chart-wrap",
                        ),
                        html.Div(
                            [
                                html.Span(
                                    [
                                        html.Span(
                                            className="kb-legend-dot",
                                            style={"background": _cc(int(i))},
                                        ),
                                        f"C{int(i)}",
                                    ],
                                    className="kb-legend-pill",
                                )
                                for i in result.centers_df.index
                            ],
                            className="kb-legend-row kb-legend-row--centered",
                        ),
                    ],
                    className="kb-card",
                ),
                html.Div(
                    [
                        html.Div(
                            html.Div(
                                [
                                    html.H3("Центроиды · тепловая карта"),
                                    html.Span(
                                        "Z-SCORE · DIVERGING",
                                        className="kb-chip kb-chip--neutral mono",
                                    ),
                                ],
                                className="kb-card-title-row",
                            ),
                            className="kb-card-head",
                        ),
                        _centroid_heatmap(result.centers_df, features),
                    ],
                    className="kb-card",
                ),
            ],
            className="kb-cluster-grid kb-cluster-grid--2",
        )
    except Exception as exc:
        profiles_grid = alert_banner(
            f"Не удалось построить профили: {exc}", "warning")

    # 12.5 · Full profile table
    table_section = _profile_table(profiles, features)

    return html.Div(
        [
            html.Div("РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ",
                     className="kb-overline kb-cluster-section-overline"),
            kpi_tiles,
            sil_alert,
            html.Div("ОБЗОР",
                     className="kb-overline kb-cluster-section-overline"),
            overview_grid,
            html.Div("ПРОФИЛИ КЛАСТЕРОВ",
                     className="kb-overline kb-cluster-section-overline"),
            html.Div(
                "Профиль показывает нормализованный вклад каждого признака "
                "внутри кластера (0–1) и z-score центроида в общей шкале.",
                className="caption kb-cluster-hint",
            ),
            profiles_grid,
            table_section,
        ],
        className="kb-cluster-section",
    )


def _kpi_tile(label: str, value: str, sub: str = "",
              tone: str = "neutral") -> html.Div:
    return html.Div(
        [
            html.Div(label.upper(),
                     className="kb-overline kb-eda-qtile__lbl"),
            html.Div(value,
                     className=f"kb-eda-qtile__v mono kb-eda-qtile__v--{tone}"),
            html.Div(sub, className="caption"),
        ],
        className="kb-eda-qtile",
    )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

layout = html.Div(
    [
        _page_head(),
        _ds_picker(),
        _config_card(),
        html.Div(id="cl-run-warn"),
        dcc.Loading(
            html.Div(id="cl-results", className="kb-cluster-results"),
            type="circle", color="#21A066",
        ),
    ],
    className="kb-page kb-page-cluster",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("cl-ds", "options"),
    Output("cl-ds", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def update_ds_options(datasets, prepared, active):
    ds = {}
    if prepared:
        ds.update(prepared)
    if datasets:
        for k, v in datasets.items():
            ds.setdefault(k, v)
    if not ds:
        return [], None
    names = list(ds.keys())
    val = active if active in names else names[0]
    return [{"label": n, "value": n} for n in names], val


@callback(
    Output("cl-ds-meta", "children"),
    Input("cl-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def show_ds_meta(ds, datasets, prepared):
    if not ds:
        return ""
    df = _load_df(prepared, datasets, ds)
    if df is None:
        return ""
    num = len(df.select_dtypes(include="number").columns)
    return html.Div(
        [
            html.Span(ds, className="kb-ds-picker__name mono"),
            html.Span("·", className="kb-ds-picker__sep"),
            html.Span(f"{_fmt_int(len(df))} строк × "
                      f"{len(df.columns)} колонок · "
                      f"{num} числовых",
                      className="kb-ds-picker__kv mono"),
        ],
        className="kb-ds-picker__meta-inner",
    )


@callback(
    Output("cl-features", "options"),
    Output("cl-features", "value"),
    Output("cl-features-label", "children"),
    Input("cl-ds", "value"),
    Input("cl-features", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_features(ds, current, datasets, prepared):
    if not ds:
        return [], [], "ЧИСЛОВЫЕ ПРИЗНАКИ"
    df = _load_df(prepared, datasets, ds)
    if df is None:
        return [], [], "ЧИСЛОВЫЕ ПРИЗНАКИ"
    num_cols = df.select_dtypes(include="number").columns.tolist()
    options = [{"label": c, "value": c} for c in num_cols]
    # Preserve user selection across dataset changes when possible
    trig = dash.callback_context.triggered[0]["prop_id"] if dash.callback_context.triggered else ""
    if "cl-features" in trig and current is not None:
        val = [c for c in current if c in num_cols]
    else:
        # Fresh dataset → pre-pick up to 4 first numeric cols
        val = num_cols[: min(4, len(num_cols))]
    n = len(val) if val else 0
    label = f"ЧИСЛОВЫЕ ПРИЗНАКИ · {n} ИЗ {len(num_cols)}"
    return options, val, label


@callback(
    Output("cl-results", "children"),
    Output("cl-run-warn", "children"),
    Input("cl-run-btn", "n_clicks"),
    Input("cl-elbow-btn", "n_clicks"),
    Input("cl-run-btn-head", "n_clicks"),
    State("cl-ds", "value"),
    State("cl-features", "value"),
    State("cl-k", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_clustering(n_run, n_elbow, n_run_head, ds, features, k, datasets, prepared):
    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    if not ds:
        return _empty_big(), ""

    if not features or len(features) < 2:
        warn = html.Div(
            [
                html.Span(icon("alert-triangle", 16),
                          className="kb-alert__ic",
                          style={"color": "var(--warning)"}),
                html.Div(
                    [
                        html.Div("Выберите минимум 2 признака",
                                 className="kb-alert__title"),
                        html.Div(
                            "Кластеризация K-Means требует двух или более "
                            "числовых колонок. Добавьте признаки в поле "
                            "выше, чтобы активировать запуск.",
                            className="kb-alert__body",
                        ),
                    ]
                ),
            ],
            className="kb-alert kb-alert--warning",
        )
        return _empty_big(), warn

    df = _load_df(prepared, datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger"), ""

    try:
        if "cl-elbow-btn" in triggered:
            k_cap = min(10, max(2, len(df) // 2))
            elbow_df = run_elbow(df, features, k_range=range(2, k_cap + 1))
            if elbow_df.empty:
                return alert_banner(
                    "Недостаточно данных для метода локтя.",
                    "warning"), ""
            return _render_elbow(elbow_df), ""

        # Run K-Means
        k_val = int(k or 3)
        result = run_kmeans(df, features, n_clusters=k_val)
        try:
            log_event("cluster", dataset=ds,
                      details=f"kmeans k={k_val} features={features}")
        except Exception:
            pass

        profiles = cluster_profiles(result)
        pca_df, explained = pca_transform(df, features, n_components=2)

        # Align labels with valid (non-NA) rows in pca_df.
        valid_idx = df[features].dropna().index
        labels_series = pd.Series(result.labels, index=valid_idx)
        labels_pca = labels_series.reindex(pca_df.index).values

        # Shadow `result.labels` for downstream PCA plot — but preserve object
        class _View:
            pass
        view = _View()
        view.labels = labels_pca
        view.n_clusters = result.n_clusters
        view.silhouette = result.silhouette
        view.centers_df = result.centers_df

        return _render_results(view, profiles, pca_df,
                               explained, list(features)), ""

    except Exception as exc:
        return alert_banner(f"Ошибка: {exc}", "danger"), ""

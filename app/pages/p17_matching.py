"""
pages/p17_matching.py -- Сопоставление групп (PSM, Exact, NN, CEM).
"""
from __future__ import annotations

import json
import logging

import dash

logger = logging.getLogger(__name__)
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.form import select_input, number_input
from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
    get_df_from_store, get_df_from_stores, save_dataframe, list_datasets,
)
from core.matching import (
    balance_summary,
    coarsened_exact_match,
    exact_match,
    nearest_neighbor_match,
    propensity_score_match,
    standardized_mean_diff,
)

dash.register_page(
    __name__,
    path="/matching",
    name="17. Сопоставление",
    order=17,
    icon="bullseye",
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div([
    page_header("17. Сопоставление групп", "Подбор сопоставимых групп для корректного сравнения"),

    # Hidden stores for matching results
    dcc.Store(id="match-result-store", storage_type="memory"),

    # Dataset selector
    dbc.Row([
        dbc.Col([
            html.Label("Датасет", className="kb-stat-label", style={"marginBottom": "6px"}),
            dcc.Dropdown(id="match-ds-select", placeholder="Выберите датасет"),
        ], md=4),
        dbc.Col(id="match-ds-info", md=8),
    ], className="mb-4"),

    # Configuration
    section_header("Настройка"),
    dbc.Row([
        dbc.Col([
            html.Label("Колонка группы (бинарная)", className="kb-stat-label",
                        style={"marginBottom": "6px"}),
            dcc.Dropdown(id="match-treatment-col", placeholder="Выберите колонку"),
            html.Div(id="match-group-info", className="mt-2"),
        ], md=4),
        dbc.Col([
            html.Label("Ковариаты для сопоставления", className="kb-stat-label",
                        style={"marginBottom": "6px"}),
            dcc.Dropdown(id="match-covariates", multi=True, placeholder="Выберите признаки"),
        ], md=8),
    ], className="mb-4"),

    # Pre-matching balance
    html.Div(id="match-pre-balance", className="mb-4"),

    # Method selection
    section_header("Метод сопоставления"),
    dbc.Tabs([
        dbc.Tab(label="PSM (Propensity Score)", tab_id="tab-psm", children=[
            html.Div([
                html.P(
                    "Оценка склонности через логистическую регрессию. "
                    "Каждому объекту опытной группы подбирается ближайший контроль.",
                    className="kb-text-secondary kb-text-sm mt-2",
                ),
                dbc.Row([
                    dbc.Col([
                        html.Label("Caliper (в стд. откл.)", className="kb-stat-label"),
                        dcc.Slider(id="psm-caliper", min=0.01, max=1.0, value=0.2,
                                   step=0.01, marks={0.1: "0.1", 0.2: "0.2", 0.5: "0.5", 1.0: "1.0"}),
                    ], md=6),
                    dbc.Col([
                        html.Label("Соотношение (контроль : опытная)", className="kb-stat-label"),
                        dcc.Dropdown(id="psm-ratio",
                                     options=[{"label": f"1:{r}", "value": r} for r in [1, 2, 3, 5]],
                                     value=1),
                    ], md=3),
                    dbc.Col([
                        dbc.Button("Запустить PSM", id="btn-run-psm",
                                   color="primary", className="mt-4", style={"width": "100%"}),
                    ], md=3),
                ], className="mt-3"),
            ], className="p-3"),
        ]),
        dbc.Tab(label="Точное сопоставление", tab_id="tab-exact", children=[
            html.Div([
                html.P(
                    "Полное совпадение по категориальным признакам. "
                    "Для каждой страты берётся min(опытная, контроль) наблюдений.",
                    className="kb-text-secondary kb-text-sm mt-2",
                ),
                dbc.Row([
                    dbc.Col([
                        html.Label("Признаки для точного совпадения", className="kb-stat-label"),
                        dcc.Dropdown(id="exact-cols", multi=True, placeholder="Категориальные колонки"),
                    ], md=9),
                    dbc.Col([
                        dbc.Button("Запустить Exact", id="btn-run-exact",
                                   color="primary", className="mt-4", style={"width": "100%"}),
                    ], md=3),
                ], className="mt-3"),
            ], className="p-3"),
        ]),
        dbc.Tab(label="Ближайший сосед (NN)", tab_id="tab-nn", children=[
            html.Div([
                html.P(
                    "Поиск ближайшего контроля по расстоянию Махаланобиса или евклидову.",
                    className="kb-text-secondary kb-text-sm mt-2",
                ),
                dbc.Row([
                    dbc.Col([
                        html.Label("Число соседей", className="kb-stat-label"),
                        dcc.Dropdown(id="nn-k",
                                     options=[{"label": str(k), "value": k} for k in [1, 2, 3, 5]],
                                     value=1),
                    ], md=3),
                    dbc.Col([
                        html.Label("Метрика", className="kb-stat-label"),
                        dbc.RadioItems(id="nn-metric", inline=True, value="mahalanobis",
                                       options=[
                                           {"label": "Махаланобис", "value": "mahalanobis"},
                                           {"label": "Евклидово", "value": "euclidean"},
                                       ]),
                    ], md=6),
                    dbc.Col([
                        dbc.Button("Запустить NN", id="btn-run-nn",
                                   color="primary", className="mt-4", style={"width": "100%"}),
                    ], md=3),
                ], className="mt-3"),
            ], className="p-3"),
        ]),
        dbc.Tab(label="CEM (огрубление)", tab_id="tab-cem", children=[
            html.Div([
                html.P(
                    "Числовые признаки огрубляются в квантильные бины, "
                    "затем точное сопоставление по бинам.",
                    className="kb-text-secondary kb-text-sm mt-2",
                ),
                dbc.Row([
                    dbc.Col([
                        html.Label("Число квантильных бинов", className="kb-stat-label"),
                        dcc.Slider(id="cem-bins", min=2, max=20, value=5, step=1,
                                   marks={2: "2", 5: "5", 10: "10", 20: "20"}),
                    ], md=9),
                    dbc.Col([
                        dbc.Button("Запустить CEM", id="btn-run-cem",
                                   color="primary", className="mt-4", style={"width": "100%"}),
                    ], md=3),
                ], className="mt-3"),
            ], className="p-3"),
        ]),
    ], className="mb-4"),

    # Results area
    html.Div(id="match-results-area"),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("match-ds-select", "options"),
    Output("match-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def populate_datasets(ds_store, prep_store, active_ds):
    names = sorted(set(list_datasets(ds_store) + list_datasets(prep_store)))
    opts = [{"label": n, "value": n} for n in names]
    val = active_ds if active_ds in names else (names[0] if names else None)
    return opts, val


@callback(
    Output("match-ds-info", "children"),
    Output("match-treatment-col", "options"),
    Output("match-covariates", "options"),
    Output("exact-cols", "options"),
    Input("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def on_dataset_select(ds_name, ds_store, prep_store):
    if not ds_name:
        return "", [], [], []

    df = get_df_from_stores(ds_name, prep_store, ds_store)
    if df is None:
        return dbc.Alert("Датасет не найден.", color="warning"), [], [], []

    info = html.Span(f"{len(df):,} строк, {len(df.columns)} колонок",
                     className="kb-text-secondary kb-text-sm")

    all_cols = [{"label": c, "value": c} for c in df.columns]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c]) or df[c].nunique() <= 10]

    cov_opts = [{"label": c, "value": c} for c in num_cols]
    exact_opts = [{"label": c, "value": c} for c in cat_cols]

    return info, all_cols, cov_opts, exact_opts


@callback(
    Output("match-group-info", "children"),
    Input("match-treatment-col", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def show_group_info(treatment_col, ds_name, ds_store, prep_store):
    if not treatment_col or not ds_name:
        return ""

    df = get_df_from_stores(ds_name, prep_store, ds_store)
    if df is None:
        return ""

    unique = df[treatment_col].dropna().unique()
    if len(unique) == 2:
        counts = df[treatment_col].value_counts()
        return html.Div([
            html.Span(f"Группа 0: {unique[0]} ({counts.iloc[0]:,}), ", className="kb-text-sm"),
            html.Span(f"Группа 1: {unique[1]} ({counts.iloc[1]:,})", className="kb-text-sm"),
        ], className="kb-text-secondary")
    elif len(unique) > 2:
        return dbc.Alert(f"Колонка содержит {len(unique)} значений. Нужно ровно 2.", color="danger")
    else:
        return dbc.Alert("Менее 2 уникальных значений.", color="warning")


@callback(
    Output("match-pre-balance", "children"),
    Input("match-covariates", "value"),
    Input("match-treatment-col", "value"),
    State("match-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def show_pre_balance(covariates, treatment_col, ds_name, ds_store, prep_store):
    if not covariates or not treatment_col or not ds_name:
        return ""

    df = get_df_from_stores(ds_name, prep_store, ds_store)
    if df is None:
        return ""

    unique = df[treatment_col].dropna().unique()
    if len(unique) != 2:
        return ""

    work = df.copy()
    val_map = {unique[0]: 0, unique[1]: 1}
    work[treatment_col] = work[treatment_col].map(val_map)

    num_cov = [c for c in covariates if pd.api.types.is_numeric_dtype(work[c])]
    if not num_cov:
        return ""

    bal = standardized_mean_diff(work, treatment_col, num_cov)
    summary = balance_summary(bal)

    # Build horizontal bar chart
    fig = go.Figure()
    colors = [
        "#ef4444" if v > 0.25 else "#f59e0b" if v > 0.1 else "#10b981"
        for v in bal["abs_smd"]
    ]
    fig.add_trace(go.Bar(
        y=bal["covariate"], x=bal["abs_smd"], orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in bal["abs_smd"]],
        textposition="outside",
    ))
    fig.add_vline(x=0.1, line_dash="dash", line_color="#10b981",
                  annotation_text="0.1")
    fig.add_vline(x=0.25, line_dash="dash", line_color="#ef4444",
                  annotation_text="0.25")
    fig.update_layout(
        title="|SMD| до сопоставления",
        xaxis_title="|SMD|",
        height=max(250, len(num_cov) * 30 + 80),
        margin=dict(l=10, r=10, t=40, b=30),
    )
    apply_kibad_theme(fig)

    n_t = int((work[treatment_col] == 1).sum())
    n_c = int((work[treatment_col] == 0).sum())

    stats_row = dbc.Row([
        dbc.Col(stat_card("Опытная", f"{n_t:,}"), md=3),
        dbc.Col(stat_card("Контроль", f"{n_c:,}"), md=3),
        dbc.Col(stat_card("Средний |SMD|", f"{summary['mean_abs_smd']:.3f}"), md=3),
        dbc.Col(stat_card("|SMD| < 0.1", f"{summary['pct_below_01']:.0f}%"), md=3),
    ], className="mb-3")

    return html.Div([
        section_header("Баланс до сопоставления"),
        stats_row,
        dcc.Graph(figure=fig),
    ])


@callback(
    Output("match-results-area", "children"),
    Input("btn-run-psm", "n_clicks"),
    Input("btn-run-exact", "n_clicks"),
    Input("btn-run-nn", "n_clicks"),
    Input("btn-run-cem", "n_clicks"),
    State("match-ds-select", "value"),
    State("match-treatment-col", "value"),
    State("match-covariates", "value"),
    State("exact-cols", "value"),
    State("psm-caliper", "value"),
    State("psm-ratio", "value"),
    State("nn-k", "value"),
    State("nn-metric", "value"),
    State("cem-bins", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_matching(
    psm_clicks, exact_clicks, nn_clicks, cem_clicks,
    ds_name, treatment_col, covariates, exact_cols_val,
    caliper, ratio, nn_k, nn_metric, cem_bins,
    ds_store, prep_store,
):
    triggered = ctx.triggered_id
    if not triggered or not ds_name or not treatment_col:
        return no_update

    df = get_df_from_stores(ds_name, prep_store, ds_store)
    if df is None:
        return dbc.Alert("Датасет не найден.", color="warning")

    unique = df[treatment_col].dropna().unique()
    if len(unique) != 2:
        return dbc.Alert("Колонка группы должна содержать ровно 2 значения.", color="danger")

    work = df.copy()
    val_map = {unique[0]: 0, unique[1]: 1}
    work[treatment_col] = work[treatment_col].map(val_map)

    num_cov = [c for c in (covariates or []) if pd.api.types.is_numeric_dtype(work[c])]

    try:
        if triggered == "btn-run-psm":
            if not num_cov:
                return dbc.Alert("PSM требует числовые ковариаты.", color="danger")
            result = propensity_score_match(work, treatment_col, num_cov,
                                            caliper=caliper or 0.2, ratio=ratio or 1)
        elif triggered == "btn-run-exact":
            if not exact_cols_val:
                return dbc.Alert("Выберите колонки для точного сопоставления.", color="danger")
            result = exact_match(work, treatment_col, exact_cols_val,
                                 covariates=num_cov if num_cov else exact_cols_val)
        elif triggered == "btn-run-nn":
            if not num_cov:
                return dbc.Alert("NN требует числовые ковариаты.", color="danger")
            result = nearest_neighbor_match(work, treatment_col, num_cov,
                                            n_neighbors=nn_k or 1, metric=nn_metric or "mahalanobis")
        elif triggered == "btn-run-cem":
            cov = num_cov if num_cov else (covariates or [])
            if not cov:
                return dbc.Alert("Выберите ковариаты.", color="danger")
            result = coarsened_exact_match(work, treatment_col, cov, n_bins=cem_bins or 5)
        else:
            return no_update
    except Exception as e:
        return dbc.Alert(f"Ошибка: {e}", color="danger")

    # Build results UI
    return _build_results(result, num_cov, work, treatment_col)


def _build_results(result, num_cov, work, treatment_col):
    """Build results layout from MatchResult."""
    summary_after = balance_summary(result.balance_after)
    summary_before = balance_summary(result.balance_before)

    match_rate_t = result.n_matched_treatment / result.n_treatment * 100 if result.n_treatment else 0
    match_rate_c = result.n_matched_control / result.n_control * 100 if result.n_control else 0
    smd_delta = summary_after["mean_abs_smd"] - summary_before["mean_abs_smd"]

    # KPI row
    kpi = dbc.Row([
        dbc.Col(stat_card("Метод", result.method), md=2),
        dbc.Col(stat_card("Опытная сопост.", f"{result.n_matched_treatment:,} ({match_rate_t:.0f}%)"), md=3),
        dbc.Col(stat_card("Контроль сопост.", f"{result.n_matched_control:,} ({match_rate_c:.0f}%)"), md=3),
        dbc.Col(stat_card("|SMD| после", f"{summary_after['mean_abs_smd']:.3f}",
                          delta=f"{smd_delta:+.3f}"), md=2),
        dbc.Col(stat_card("|SMD| < 0.1", f"{summary_after['pct_below_01']:.0f}%"), md=2),
    ], className="mb-4")

    # Love plot
    bal_b = result.balance_before.sort_values("abs_smd", ascending=True)
    bal_a = result.balance_after.sort_values("abs_smd", ascending=True)

    fig_love = go.Figure()
    fig_love.add_trace(go.Scatter(
        x=bal_b["abs_smd"], y=bal_b["covariate"], mode="markers",
        marker=dict(size=9, color="#4a5068", symbol="circle-open", line=dict(width=2)),
        name="До",
    ))
    colors = ["#10b981" if v < 0.1 else "#f59e0b" if v < 0.25 else "#ef4444"
              for v in bal_a["abs_smd"]]
    fig_love.add_trace(go.Scatter(
        x=bal_a["abs_smd"], y=bal_a["covariate"], mode="markers",
        marker=dict(size=11, color=colors, symbol="circle"),
        name="После",
    ))
    # Connecting lines
    for _, row_b in bal_b.iterrows():
        cov = row_b["covariate"]
        row_a_match = bal_a[bal_a["covariate"] == cov]
        if not row_a_match.empty:
            fig_love.add_trace(go.Scatter(
                x=[row_b["abs_smd"], row_a_match.iloc[0]["abs_smd"]],
                y=[cov, cov], mode="lines",
                line=dict(color="#252a3a", width=1), showlegend=False,
            ))
    fig_love.add_vline(x=0.1, line_dash="dash", line_color="#10b981", annotation_text="0.1")
    fig_love.add_vline(x=0.25, line_dash="dash", line_color="#ef4444", annotation_text="0.25")
    fig_love.update_layout(
        height=max(350, len(bal_b) * 30 + 100),
        margin=dict(l=10, r=10, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    apply_kibad_theme(fig_love)

    # Balance table
    bal_merged = result.balance_before[["covariate", "abs_smd"]].rename(
        columns={"abs_smd": "|SMD| до"}
    ).merge(
        result.balance_after[["covariate", "abs_smd"]].rename(
            columns={"abs_smd": "|SMD| после"}
        ), on="covariate", how="left",
    )
    bal_merged["Изменение"] = bal_merged["|SMD| после"] - bal_merged["|SMD| до"]

    # Tabs
    tabs = dbc.Tabs([
        dbc.Tab(label="Love-plot", tab_id="res-love", children=[
            dcc.Graph(figure=fig_love),
        ]),
        dbc.Tab(label="Таблица баланса", tab_id="res-table", children=[
            data_table(
                bal_merged.rename(columns={"covariate": "Ковариата"}),
                id="match-balance-table",
                page_size=20,
            ),
        ]),
        dbc.Tab(label="Данные", tab_id="res-data", children=[
            html.P(f"Сопоставлено: {len(result.matched_df):,} строк", className="kb-text-secondary mt-2"),
            data_table(
                result.matched_df.head(100),
                id="match-data-table",
                page_size=20,
            ),
        ]),
    ])

    improved = (bal_merged["Изменение"] < 0).sum()
    total = len(bal_merged)

    interpretation = html.Div([
        html.Div(
            f"Баланс улучшился по {improved} из {total} ковариат. "
            f"Средний |SMD|: {summary_before['mean_abs_smd']:.3f} -> {summary_after['mean_abs_smd']:.3f}.",
            className="kb-alert kb-alert--info",
        ),
    ], className="mt-3")

    return html.Div([
        section_header("Результаты сопоставления"),
        kpi,
        tabs,
        interpretation,
    ])

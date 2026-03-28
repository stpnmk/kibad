"""p03_group – Group & Aggregate page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from app.state import (
    get_df_from_store, save_dataframe,
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS, STORE_AGG_RESULTS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.cards import stat_card
from app.components.alerts import alert_banner
from core.aggregate import (
    group_aggregate, pivot_view, available_agg_functions,
    to_csv_bytes, to_xlsx_bytes, TIME_BUCKET_MAP,
)
from core.i18n import t
from core.audit import log_event

dash.register_page(__name__, path="/group", name="3. Группировка", order=3, icon="table")

layout = html.Div([
    page_header("3. Группировка", "Сводные таблицы и агрегация данных"),
    dbc.Row([
        dbc.Col([
            html.Label("Датасет", className="kb-stat-label"),
            dcc.Dropdown(id="ga-ds-select", placeholder="Выберите датасет..."),
        ], width=4),
    ], className="mb-3"),
    html.Div(id="ga-stats-row", className="kb-stats-grid"),
    dbc.Card([
        dbc.CardHeader("Настройка группировки"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Группировать по", className="kb-stat-label"),
                    dcc.Dropdown(id="ga-group-cols", multi=True, placeholder="Измерения (строки)..."),
                ], width=4),
                dbc.Col([
                    html.Label("Числовые показатели", className="kb-stat-label"),
                    dcc.Dropdown(id="ga-metric-cols", multi=True, placeholder="Метрики..."),
                ], width=4),
                dbc.Col([
                    html.Label("Функции агрегации", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="ga-agg-funcs", multi=True,
                        options=[{"label": f, "value": f} for f in ["sum", "mean", "count", "median", "min", "max", "std", "nunique"]],
                        value=["sum", "mean"],
                        placeholder="Выберите функции...",
                    ),
                ], width=4),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Временной период", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="ga-time-bucket",
                        options=[{"label": "Без группировки по времени", "value": ""}] +
                                [{"label": k, "value": k} for k in TIME_BUCKET_MAP.keys()],
                        value="",
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Дата-колонка (для временной группировки)", className="kb-stat-label"),
                    dcc.Dropdown(id="ga-date-col", placeholder="Выберите..."),
                ], width=4),
                dbc.Col([
                    dbc.Button("Агрегировать", id="ga-run-btn", color="primary", className="mt-4"),
                ], width=4),
            ], className="mt-3"),
        ]),
    ], className="mb-3"),
    dcc.Loading(
        html.Div(id="ga-results"),
        type="circle", color="#10b981",
    ),
    dcc.Download(id="ga-download"),
])


@callback(
    Output("ga-ds-select", "options"),
    Output("ga-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def update_ds_options(datasets, active):
    if not datasets:
        return [], None
    names = list(datasets.keys())
    val = active if active in names else (names[0] if names else None)
    return [{"label": n, "value": n} for n in names], val


@callback(
    Output("ga-group-cols", "options"),
    Output("ga-metric-cols", "options"),
    Output("ga-date-col", "options"),
    Output("ga-stats-row", "children"),
    Input("ga-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_columns(ds_name, datasets, prepared):
    if not ds_name:
        return [], [], [], ""
    df = get_df_from_store(prepared, ds_name) or get_df_from_store(datasets, ds_name)
    if df is None:
        return [], [], [], ""
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]
    stats = [
        stat_card("Строк", f"{df.shape[0]:,}"),
        stat_card("Столбцов", str(df.shape[1])),
        stat_card("Числовых", str(len(num_cols))),
    ]
    return (
        [{"label": c, "value": c} for c in all_cols],
        [{"label": c, "value": c} for c in num_cols],
        [{"label": c, "value": c} for c in date_cols],
        stats,
    )


@callback(
    Output("ga-results", "children"),
    Output(STORE_AGG_RESULTS, "data"),
    Input("ga-run-btn", "n_clicks"),
    State("ga-ds-select", "value"),
    State("ga-group-cols", "value"),
    State("ga-metric-cols", "value"),
    State("ga-agg-funcs", "value"),
    State("ga-time-bucket", "value"),
    State("ga-date-col", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State(STORE_AGG_RESULTS, "data"),
    prevent_initial_call=True,
)
def run_aggregation(n, ds_name, group_cols, metric_cols, agg_funcs, time_bucket, date_col,
                    datasets, prepared, agg_store):
    if not ds_name or not group_cols or not metric_cols or not agg_funcs:
        return alert_banner("Выберите колонки группировки, метрики и функции.", "warning"), no_update

    df = get_df_from_store(prepared, ds_name) or get_df_from_store(datasets, ds_name)
    if df is None:
        return alert_banner("Датасет не найден.", "danger"), no_update

    try:
        result_df = group_aggregate(
            df, group_cols=group_cols, metric_cols=metric_cols,
            agg_funcs=agg_funcs,
            time_col=date_col if time_bucket else None,
            time_bucket=time_bucket if time_bucket else None,
        )
        log_event("aggregate", dataset=ds_name, details=f"group={group_cols}, agg={agg_funcs}")

        path = save_dataframe(result_df, f"agg_{ds_name}")
        agg_store = agg_store or {}
        agg_store[ds_name] = path

        children = [
            section_header("Результат агрегации"),
            data_table(result_df, id="ga-result-table"),
        ]

        if len(group_cols) == 1 and len(metric_cols) >= 1:
            fig = px.bar(result_df, x=group_cols[0],
                         y=[c for c in result_df.columns if c not in group_cols][:4],
                         barmode="group", title="Агрегация по группам")
            apply_kibad_theme(fig)
            children.append(dcc.Graph(figure=fig))

        return html.Div(children), agg_store

    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

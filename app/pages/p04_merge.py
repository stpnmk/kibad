"""p04_merge – Table merge/join/concat page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd

from app.state import (
    get_df_from_store, save_dataframe, list_datasets,
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.alerts import alert_banner
from core.merge import merge_tables, concat_tables, analyze_key_cardinality, MergeWarning
from core.audit import log_event

dash.register_page(__name__, path="/merge", name="4. Объединение", order=4, icon="diagram-3")

layout = html.Div([
    page_header("4. Объединение таблиц", "JOIN, UNION и конкатенация с диагностикой"),

    dbc.Tabs([
        dbc.Tab(label="JOIN (слияние по ключу)", tab_id="tab-join", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Левая таблица", className="kb-stat-label"),
                    dcc.Dropdown(id="merge-left-ds", placeholder="Выберите..."),
                    html.Div(id="merge-left-preview"),
                ], width=6),
                dbc.Col([
                    html.Label("Правая таблица", className="kb-stat-label"),
                    dcc.Dropdown(id="merge-right-ds", placeholder="Выберите..."),
                    html.Div(id="merge-right-preview"),
                ], width=6),
            ], className="mt-3"),
            html.Hr(),
            dbc.Row([
                dbc.Col([
                    html.Label("Левый ключ", className="kb-stat-label"),
                    dcc.Dropdown(id="merge-left-key", placeholder="Ключ левой таблицы..."),
                ], width=3),
                dbc.Col([
                    html.Label("Правый ключ", className="kb-stat-label"),
                    dcc.Dropdown(id="merge-right-key", placeholder="Ключ правой таблицы..."),
                ], width=3),
                dbc.Col([
                    html.Label("Тип JOIN", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="merge-how",
                        options=[
                            {"label": "LEFT JOIN", "value": "left"},
                            {"label": "INNER JOIN", "value": "inner"},
                            {"label": "RIGHT JOIN", "value": "right"},
                            {"label": "OUTER JOIN", "value": "outer"},
                        ],
                        value="left",
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button("Объединить", id="merge-run-btn", color="primary", className="mt-4"),
                ], width=3),
            ]),
            html.Div(id="merge-diagnostics", className="mt-3"),
            dcc.Loading(html.Div(id="merge-result"), type="circle", color="#10b981"),
        ]),
        dbc.Tab(label="CONCAT (добавление строк)", tab_id="tab-concat", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Датасеты для конкатенации", className="kb-stat-label"),
                    dcc.Dropdown(id="concat-datasets", multi=True, placeholder="Выберите 2+ датасета..."),
                ], width=6),
                dbc.Col([
                    html.Label("Ось", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="concat-axis",
                        options=[
                            {"label": "По строкам (вертикально)", "value": "0"},
                            {"label": "По колонкам (горизонтально)", "value": "1"},
                        ],
                        value="0",
                    ),
                ], width=3),
                dbc.Col([
                    dbc.Button("Конкатенировать", id="concat-run-btn", color="primary", className="mt-4"),
                ], width=3),
            ], className="mt-3"),
            dcc.Loading(html.Div(id="concat-result"), type="circle", color="#10b981"),
        ]),
    ], id="merge-tabs", active_tab="tab-join"),
])


@callback(
    Output("merge-left-ds", "options"),
    Output("merge-right-ds", "options"),
    Output("concat-datasets", "options"),
    Input(STORE_DATASET, "data"),
)
def update_ds_lists(datasets):
    names = list_datasets(datasets)
    opts = [{"label": n, "value": n} for n in names]
    return opts, opts, opts


@callback(
    Output("merge-left-key", "options"),
    Output("merge-left-preview", "children"),
    Input("merge-left-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_left(ds, datasets, prepared):
    if not ds:
        return [], ""
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return [], ""
    cols = [{"label": c, "value": c} for c in df.columns]
    preview = html.Div([
        html.Small(f"{df.shape[0]:,} строк × {df.shape[1]} колонок", className="kb-text-muted"),
        data_table(df.head(3), id="merge-left-tbl"),
    ])
    return cols, preview


@callback(
    Output("merge-right-key", "options"),
    Output("merge-right-preview", "children"),
    Input("merge-right-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_right(ds, datasets, prepared):
    if not ds:
        return [], ""
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return [], ""
    cols = [{"label": c, "value": c} for c in df.columns]
    preview = html.Div([
        html.Small(f"{df.shape[0]:,} строк × {df.shape[1]} колонок", className="kb-text-muted"),
        data_table(df.head(3), id="merge-right-tbl"),
    ])
    return cols, preview


@callback(
    Output("merge-diagnostics", "children"),
    Input("merge-left-key", "value"),
    Input("merge-right-key", "value"),
    State("merge-left-ds", "value"),
    State("merge-right-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def run_diagnostics(lk, rk, lds, rds, datasets, prepared):
    if not all([lk, rk, lds, rds]):
        return ""
    left_df = get_df_from_store(prepared, lds) or get_df_from_store(datasets, lds)
    right_df = get_df_from_store(prepared, rds) or get_df_from_store(datasets, rds)
    if left_df is None or right_df is None:
        return ""
    try:
        card = analyze_key_cardinality(left_df, right_df, [lk], [rk])
        warnings = []
        if card.get("left_null_pct", 0) > 0:
            warnings.append(alert_banner(f"Пустые ключи в левой таблице: {card['left_null_pct']:.1f}%", "warning"))
        if card.get("right_null_pct", 0) > 0:
            warnings.append(alert_banner(f"Пустые ключи в правой таблице: {card['right_null_pct']:.1f}%", "warning"))
        if card.get("many_to_many"):
            warnings.append(alert_banner("Many-to-many связь — возможен взрыв строк!", "danger"))
        match_rate = card.get("match_rate", 0)
        warnings.append(html.Div(f"Совпадение ключей: {match_rate:.1f}%", className="kb-text-secondary kb-text-sm"))
        return html.Div(warnings)
    except Exception:
        return ""


@callback(
    Output("merge-result", "children"),
    Output(STORE_DATASET, "data", allow_duplicate=True),
    Input("merge-run-btn", "n_clicks"),
    State("merge-left-ds", "value"),
    State("merge-right-ds", "value"),
    State("merge-left-key", "value"),
    State("merge-right-key", "value"),
    State("merge-how", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_merge(n, lds, rds, lk, rk, how, datasets, prepared):
    if not all([lds, rds, lk, rk]):
        return alert_banner("Выберите оба датасета и ключи.", "warning"), no_update
    left_df = get_df_from_store(prepared, lds) or get_df_from_store(datasets, lds)
    right_df = get_df_from_store(prepared, rds) or get_df_from_store(datasets, rds)
    if left_df is None or right_df is None:
        return alert_banner("Датасеты не найдены.", "danger"), no_update
    try:
        result_df, warnings = merge_tables(left_df, right_df, [lk], [rk], how=how)
        name = f"{lds}_x_{rds}"
        path = save_dataframe(result_df, name)
        datasets = datasets or {}
        datasets[name] = path
        log_event("merge", dataset=name, details=f"{lds} {how} {rds} on {lk}={rk}")

        warn_divs = [alert_banner(str(w), "warning") for w in warnings]
        return html.Div([
            *warn_divs,
            alert_banner(f"Результат: {result_df.shape[0]:,} строк × {result_df.shape[1]} колонок. Сохранено как «{name}».", "success"),
            data_table(result_df.head(50), id="merge-result-tbl"),
        ]), datasets
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


@callback(
    Output("concat-result", "children"),
    Output(STORE_DATASET, "data", allow_duplicate=True),
    Input("concat-run-btn", "n_clicks"),
    State("concat-datasets", "value"),
    State("concat-axis", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_concat(n, ds_names, axis, datasets, prepared):
    if not ds_names or len(ds_names) < 2:
        return alert_banner("Выберите 2 или более датасетов.", "warning"), no_update
    dfs = []
    for name in ds_names:
        df = get_df_from_store(prepared, name) or get_df_from_store(datasets, name)
        if df is not None:
            dfs.append(df)
    if len(dfs) < 2:
        return alert_banner("Не удалось загрузить датасеты.", "danger"), no_update
    try:
        result_df = concat_tables(dfs, axis=int(axis))
        name = "concat_" + "_".join(ds_names[:3])
        path = save_dataframe(result_df, name)
        datasets = datasets or {}
        datasets[name] = path
        return html.Div([
            alert_banner(f"Конкатенация: {result_df.shape[0]:,} строк × {result_df.shape[1]} колонок. Сохранено как «{name}».", "success"),
            data_table(result_df.head(50), id="concat-result-tbl"),
        ]), datasets
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

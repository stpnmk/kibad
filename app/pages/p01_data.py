"""
pages/p01_data.py -- Dataset ingestion: file upload, PostgreSQL connection, catalog.
"""
from __future__ import annotations

import base64
import io
import json

import dash
from dash import html, dcc, callback, Input, Output, State, dash_table, no_update, ctx
import dash_bootstrap_components as dbc
import pandas as pd

from app.components.upload import upload_zone
from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.form import text_input, select_input
from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_ACTIVE_DS,
    save_dataframe, get_df_from_store, list_datasets,
)
from core.data import load_file, profile_dataframe, describe_numeric, infer_column_types

dash.register_page(
    __name__,
    path="/data",
    name="1. Данные",
    order=1,
    icon="database",
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div([
    page_header("1. Данные", "Загрузка и подключение источников данных"),

    dbc.Tabs([
        # -- Tab 1: File Upload --
        dbc.Tab(label="Загрузка файлов", tab_id="tab-upload", children=[
            html.Div([
                section_header("Загрузка файлов"),
                upload_zone(
                    id="data-upload",
                    label="Перетащите файл или нажмите для выбора",
                    hint="CSV, Excel (.xlsx), Parquet",
                    multiple=True,
                ),
                html.Div(className="mb-3"),
                dbc.Row([
                    dbc.Col([
                        html.Label("Разделитель CSV", className="kb-stat-label",
                                   style={"marginBottom": "6px"}),
                        dcc.Dropdown(
                            id="data-csv-sep",
                            options=[
                                {"label": "Запятая (,)", "value": ","},
                                {"label": "Точка с запятой (;)", "value": ";"},
                                {"label": "Табуляция", "value": "\t"},
                                {"label": "Вертикальная черта (|)", "value": "|"},
                            ],
                            value=",",
                            clearable=False,
                            className="kb-select",
                        ),
                    ], md=3),
                    dbc.Col([
                        html.Div(style={"height": "24px"}),
                        dbc.Button(
                            "Загрузить",
                            id="data-btn-load",
                            color="primary",
                            className="w-100",
                        ),
                    ], md=2),
                ], className="mb-3"),

                dcc.Loading(
                    html.Div(id="data-upload-result"),
                    type="circle", color="#10b981",
                ),
            ], style={"padding": "16px 0"}),
        ]),

        # -- Tab 2: PostgreSQL --
        dbc.Tab(label="PostgreSQL", tab_id="tab-pg", children=[
            html.Div([
                section_header("Подключение к базе данных"),
                dbc.Row([
                    dbc.Col(text_input("Хост", "pg-host", value="localhost"), md=4),
                    dbc.Col(text_input("Порт", "pg-port", value="5432"), md=2),
                    dbc.Col(text_input("База данных", "pg-database"), md=3),
                ]),
                dbc.Row([
                    dbc.Col(text_input("Пользователь", "pg-user"), md=4),
                    dbc.Col([
                        html.Label("Пароль", className="kb-stat-label",
                                   style={"marginBottom": "6px"}),
                        dcc.Input(
                            id="pg-password",
                            type="password",
                            placeholder="Пароль",
                            style={"width": "100%"},
                        ),
                    ], md=4, style={"marginBottom": "12px"}),
                ]),
                dbc.Row([
                    dbc.Col(
                        dbc.Button("Подключиться", id="pg-btn-connect",
                                   color="primary"),
                        md=3,
                    ),
                ]),
                html.Div(id="pg-connect-result", className="mt-3"),
                html.Hr(),
                html.Div([
                    html.Label("SQL-запрос", className="kb-stat-label",
                               style={"marginBottom": "6px"}),
                    dcc.Textarea(
                        id="pg-sql-query",
                        placeholder="SELECT * FROM my_table LIMIT 10000",
                        style={"width": "100%", "height": "120px",
                               "fontFamily": "monospace"},
                    ),
                    html.Label("Имя датасета", className="kb-stat-label mt-2",
                               style={"marginBottom": "6px"}),
                    dcc.Input(id="pg-ds-name", value="pg_query",
                              style={"width": "300px"}),
                    html.Div(className="mb-2"),
                    dbc.Button("Выполнить запрос", id="pg-btn-exec",
                               color="primary"),
                ], className="mt-2"),
                dcc.Loading(
                    html.Div(id="pg-exec-result"),
                    type="circle", color="#10b981",
                ),
            ], style={"padding": "16px 0"}),
        ]),

        # -- Tab 3: Catalog --
        dbc.Tab(label="Каталог датасетов", tab_id="tab-catalog", children=[
            html.Div([
                section_header("Каталог датасетов"),
                html.Div([
                    html.Label("Выберите датасет", className="kb-stat-label",
                               style={"marginBottom": "6px"}),
                    dcc.Dropdown(id="catalog-ds-select", className="kb-select"),
                ], style={"maxWidth": "400px", "marginBottom": "16px"}),
                dcc.Loading(
                    html.Div(id="catalog-preview"),
                    type="circle", color="#10b981",
                ),
            ], style={"padding": "16px 0"}),
        ]),
    ], id="data-tabs", active_tab="tab-upload"),
], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px 16px"})


# ---------------------------------------------------------------------------
# Callback: parse uploaded files
# ---------------------------------------------------------------------------
@callback(
    Output("data-upload-result", "children"),
    Output(STORE_DATASET, "data", allow_duplicate=True),
    Input("data-btn-load", "n_clicks"),
    State("data-upload", "contents"),
    State("data-upload", "filename"),
    State("data-csv-sep", "value"),
    State(STORE_DATASET, "data"),
    prevent_initial_call=True,
)
def load_uploaded_files(n_clicks, contents_list, filenames_list, sep, ds_store):
    if not n_clicks or not contents_list:
        return no_update, no_update

    ds_store = ds_store or {}
    results = []

    if not isinstance(contents_list, list):
        contents_list = [contents_list]
        filenames_list = [filenames_list]

    for content_str, fname in zip(contents_list, filenames_list):
        try:
            # Decode base64 content from dcc.Upload
            content_type, content_data = content_str.split(",", 1)
            decoded = base64.b64decode(content_data)

            df = load_file(decoded, filename=fname, sep=sep)
            ds_name = fname.rsplit(".", 1)[0] if "." in fname else fname
            path = save_dataframe(df, ds_name)
            ds_store[ds_name] = path

            # Build preview
            n_rows, n_cols = df.shape
            n_num = len(df.select_dtypes(include="number").columns)
            n_null = int(df.isnull().sum().sum())

            preview = html.Div([
                dbc.Alert(
                    f"Датасет \u00AB{ds_name}\u00BB загружен: {n_rows:,} строк \u00D7 {n_cols} колонок",
                    color="success",
                ),
                dbc.Row([
                    dbc.Col(stat_card("Строк", f"{n_rows:,}"), md=3),
                    dbc.Col(stat_card("Столбцов", str(n_cols)), md=3),
                    dbc.Col(stat_card("Числовых", str(n_num)), md=3),
                    dbc.Col(stat_card("Пропусков", f"{n_null:,}"), md=3),
                ], className="mb-3"),
                data_table(df.head(10), id=f"preview-{ds_name}", page_size=10),
            ])
            results.append(preview)

        except Exception as e:
            results.append(dbc.Alert(f"{fname}: {e}", color="danger"))

    return html.Div(results), ds_store


# ---------------------------------------------------------------------------
# Callback: PostgreSQL connection (stub -- actual DB logic in services.db)
# ---------------------------------------------------------------------------
@callback(
    Output("pg-connect-result", "children"),
    Input("pg-btn-connect", "n_clicks"),
    State("pg-host", "value"),
    State("pg-port", "value"),
    State("pg-database", "value"),
    State("pg-user", "value"),
    State("pg-password", "value"),
    prevent_initial_call=True,
)
def pg_connect(n_clicks, host, port, database, user, password):
    if not all([host, port, database, user]):
        return dbc.Alert("Заполните хост, порт, базу данных и пользователя.", color="warning")
    try:
        from services.db import test_connection
        ok, msg = test_connection(host, port, database, user, password)
        if ok:
            return dbc.Alert(
                f"Подключено к {database} на {host}:{port}",
                color="success",
            )
        return dbc.Alert(f"Ошибка: {msg}", color="danger")
    except ImportError:
        return dbc.Alert(
            "Модуль services.db не найден. PostgreSQL-подключение недоступно.",
            color="warning",
        )
    except Exception as exc:
        return dbc.Alert(f"Ошибка: {exc}", color="danger")


# ---------------------------------------------------------------------------
# Callback: Execute SQL query
# ---------------------------------------------------------------------------
@callback(
    Output("pg-exec-result", "children"),
    Output(STORE_DATASET, "data", allow_duplicate=True),
    Input("pg-btn-exec", "n_clicks"),
    State("pg-sql-query", "value"),
    State("pg-ds-name", "value"),
    State("pg-host", "value"),
    State("pg-port", "value"),
    State("pg-database", "value"),
    State("pg-user", "value"),
    State("pg-password", "value"),
    State(STORE_DATASET, "data"),
    prevent_initial_call=True,
)
def pg_execute(n_clicks, sql, ds_name, host, port, database, user, password, ds_store):
    if not n_clicks or not sql:
        return no_update, no_update
    try:
        from services.db import query_to_dataframe
        params = {"host": host, "port": port, "database": database,
                  "user": user, "password": password}
        df = query_to_dataframe(**params, query=sql)
        if df.empty:
            return dbc.Alert("Запрос вернул 0 строк.", color="warning"), no_update
        ds_store = ds_store or {}
        path = save_dataframe(df, ds_name or "pg_query")
        ds_store[ds_name or "pg_query"] = path
        return html.Div([
            dbc.Alert(
                f"Датасет \u00AB{ds_name}\u00BB загружен: {len(df):,} строк \u00D7 {len(df.columns)} колонок",
                color="success",
            ),
            data_table(df.head(10), id="pg-preview-tbl", page_size=10),
        ]), ds_store
    except ImportError:
        return dbc.Alert("Модуль services.db не найден.", color="warning"), no_update
    except Exception as exc:
        return dbc.Alert(f"Ошибка: {exc}", color="danger"), no_update


# ---------------------------------------------------------------------------
# Callback: populate catalog dropdown
# ---------------------------------------------------------------------------
@callback(
    Output("catalog-ds-select", "options"),
    Output("catalog-ds-select", "value"),
    Input(STORE_DATASET, "data"),
)
def update_catalog_dropdown(ds_store):
    names = list_datasets(ds_store)
    options = [{"label": n, "value": n} for n in names]
    value = names[0] if names else None
    return options, value


# ---------------------------------------------------------------------------
# Callback: catalog preview
# ---------------------------------------------------------------------------
@callback(
    Output("catalog-preview", "children"),
    Input("catalog-ds-select", "value"),
    State(STORE_DATASET, "data"),
)
def show_catalog_preview(ds_name, ds_store):
    if not ds_name or not ds_store:
        return empty_state(
            icon="",
            title="Нет данных",
            description="Загрузите датасет во вкладке \u00ABЗагрузка файлов\u00BB",
        )

    df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return dbc.Alert("Не удалось загрузить датасет.", color="warning")

    n_rows, n_cols = df.shape
    n_num = len(df.select_dtypes(include="number").columns)
    n_null = int(df.isnull().sum().sum())

    # Schema table
    schema_df = pd.DataFrame({
        "Колонка": df.columns,
        "Тип": df.dtypes.astype(str).values,
        "Заполнено %": (df.notna().mean() * 100).round(1).values,
        "Уникальных": df.nunique().values,
        "Пример": [
            str(df[c].dropna().iloc[0]) if df[c].notna().any() else "\u2014"
            for c in df.columns
        ],
    })

    return html.Div([
        dbc.Row([
            dbc.Col(stat_card("Строк", f"{n_rows:,}"), md=3),
            dbc.Col(stat_card("Столбцов", str(n_cols)), md=3),
            dbc.Col(stat_card("Числовых", str(n_num)), md=3),
            dbc.Col(stat_card("Пропусков", f"{n_null:,}"), md=3),
        ], className="mb-3"),

        html.H5("Предварительный просмотр", className="mb-2"),
        data_table(df.head(10), id="catalog-data-tbl", page_size=10),

        html.H5("Схема данных", className="mt-3 mb-2"),
        data_table(schema_df, id="catalog-schema-tbl", page_size=20),
    ])

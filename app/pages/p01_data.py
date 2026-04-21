"""
pages/p01_data.py — Dataset ingestion (Slide 4 port).

Layout: step-style header with action buttons, tabs for file upload / PostgreSQL /
catalog, a two-column body (dropzone + controls + preview on the left, recent
sources + schema on the right rail).
"""
from __future__ import annotations

import base64

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd

from app.components.alerts import alert_banner
from app.components.cards import card, chip, kpi
from app.components.form import text_input
from app.components.icons import icon
from app.components.layout import empty_state
from app.components.table import data_table
from app.components.upload import upload_zone
from app.state import (
    STORE_DATASET, save_dataframe, get_df_from_store, list_datasets,
)
from core.data import load_file

dash.register_page(
    __name__,
    path="/data",
    name="1. Данные",
    order=1,
    icon="database",
)


# ---------------------------------------------------------------------------
# Presentational helpers
# ---------------------------------------------------------------------------
def _page_head() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Шаг 1", className="kb-overline"),
                    html.H1("Данные", className="kb-page-title"),
                    html.Div(
                        "Загрузка и подключение источников данных",
                        className="kb-page-subtitle",
                    ),
                ],
                className="kb-page-head-left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("database", 14), html.Span("Подключить БД")],
                        id="data-connect-db-btn",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("history", 14), html.Span("История загрузок")],
                        id="data-history-btn",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                ],
                className="kb-page-head-actions",
            ),
        ],
        className="kb-page-head",
    )


def _upload_controls_row() -> html.Div:
    """CSV parsing controls + Load button, horizontally aligned."""
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Разделитель CSV", className="kb-field-label"),
                    dcc.Dropdown(
                        id="data-csv-sep",
                        options=[
                            {"label": "Запятая ( , )",        "value": ","},
                            {"label": "Точка с запятой ( ; )", "value": ";"},
                            {"label": "Табуляция",             "value": "\t"},
                            {"label": "Вертикальная черта ( | )", "value": "|"},
                        ],
                        value=",", clearable=False,
                    ),
                ],
                className="kb-data-ctrl kb-data-ctrl--sep",
            ),
            html.Div(
                [
                    html.Label("Кодировка", className="kb-field-label"),
                    dcc.Dropdown(
                        id="data-encoding",
                        options=[
                            {"label": "UTF-8",      "value": "utf-8"},
                            {"label": "Windows-1251", "value": "cp1251"},
                            {"label": "Latin-1",    "value": "latin-1"},
                        ],
                        value="utf-8", clearable=False,
                    ),
                ],
                className="kb-data-ctrl kb-data-ctrl--enc",
            ),
            html.Div(
                [
                    html.Label("Заголовок", className="kb-field-label"),
                    dcc.Dropdown(
                        id="data-header",
                        options=[
                            {"label": "1-я строка", "value": "0"},
                            {"label": "Нет",        "value": "-1"},
                        ],
                        value="0", clearable=False,
                    ),
                ],
                className="kb-data-ctrl kb-data-ctrl--hdr",
            ),
            html.Button(
                [icon("upload", 14), html.Span("Загрузить")],
                id="data-btn-load",
                className="kb-btn kb-btn--primary",
                n_clicks=0,
            ),
        ],
        className="kb-data-ctrls-row",
    )


def _upload_tab_body() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    # LEFT — main column
                    html.Div(
                        [
                            upload_zone(
                                id="data-upload",
                                label="Перетащите файл или нажмите для выбора",
                                hint="CSV · Excel (.xlsx) · Parquet · до 2 ГБ",
                                multiple=True,
                            ),
                            _upload_controls_row(),
                            dcc.Loading(
                                html.Div(
                                    id="data-upload-result",
                                    className="kb-data-results",
                                ),
                                type="circle", color="var(--accent-500)",
                            ),
                        ],
                        className="kb-data-main",
                    ),

                    # RIGHT rail
                    html.Div(
                        [
                            html.Div(id="data-recent-rail"),
                            html.Div(id="data-schema-rail"),
                        ],
                        className="kb-data-rail",
                    ),
                ],
                className="kb-data-body",
            ),
        ],
        className="kb-data-page",
    )


def _pg_tab_body() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    text_input("Хост", "pg-host", value="localhost"),
                    text_input("Порт", "pg-port", value="5432"),
                    text_input("База данных", "pg-database"),
                ],
                className="kb-data-pg-row",
            ),
            html.Div(
                [
                    text_input("Пользователь", "pg-user"),
                    html.Div(
                        [
                            html.Label("Пароль", className="kb-field-label"),
                            dcc.Input(
                                id="pg-password",
                                type="password",
                                placeholder="Пароль",
                                className="kb-input",
                            ),
                        ],
                        className="kb-data-field",
                    ),
                    html.Button(
                        "Подключиться",
                        id="pg-btn-connect",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-data-pg-row",
            ),
            html.Div(id="pg-connect-result", className="kb-data-results"),

            html.Div(
                [
                    html.Label("SQL-запрос", className="kb-field-label"),
                    dcc.Textarea(
                        id="pg-sql-query",
                        placeholder="SELECT * FROM my_table LIMIT 10000",
                        className="kb-textarea kb-textarea--sql",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Имя датасета", className="kb-field-label"),
                                    dcc.Input(
                                        id="pg-ds-name", value="pg_query",
                                        className="kb-input",
                                    ),
                                ],
                                className="kb-data-field",
                            ),
                            html.Button(
                                [icon("play", 12), html.Span("Выполнить запрос")],
                                id="pg-btn-exec",
                                className="kb-btn kb-btn--primary",
                                n_clicks=0,
                            ),
                        ],
                        className="kb-data-pg-row",
                    ),
                ],
                className="kb-data-pg-sql",
            ),
            dcc.Loading(
                html.Div(id="pg-exec-result", className="kb-data-results"),
                type="circle", color="var(--accent-500)",
            ),
        ],
        className="kb-data-pg",
    )


def _catalog_tab_body() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Выберите датасет", className="kb-field-label"),
                    dcc.Dropdown(id="catalog-ds-select"),
                ],
                className="kb-data-catalog-picker",
            ),
            dcc.Loading(
                html.Div(id="catalog-preview", className="kb-data-results"),
                type="circle", color="var(--accent-500)",
            ),
        ],
        className="kb-data-catalog",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        _page_head(),

        dbc.Tabs(
            [
                dbc.Tab(label="Загрузка файлов",   tab_id="tab-upload",  children=_upload_tab_body()),
                dbc.Tab(label="PostgreSQL",        tab_id="tab-pg",      children=_pg_tab_body()),
                dbc.Tab(label="Каталог датасетов", tab_id="tab-catalog", children=_catalog_tab_body()),
            ],
            id="data-tabs", active_tab="tab-upload",
        ),
    ],
    className="kb-page kb-page-data",
)


# ---------------------------------------------------------------------------
# Upload result renderer
# ---------------------------------------------------------------------------
def _render_upload_result(df: pd.DataFrame, ds_name: str, size_bytes: int) -> html.Div:
    n_rows, n_cols = df.shape
    n_num = len(df.select_dtypes(include="number").columns)
    n_null = int(df.isnull().sum().sum())
    n_dups = int(df.duplicated().sum())
    size_mb = size_bytes / (1024 * 1024)

    # Columns with suspected outliers (|z| > 3 in numeric cols)
    suspect_cols: list[str] = []
    for col in df.select_dtypes(include="number").columns:
        vals = df[col].dropna()
        if len(vals) < 8:
            continue
        std = vals.std()
        if std and std > 0:
            z = (vals - vals.mean()).abs() / std
            if (z > 3).any():
                suspect_cols.append(col)

    alerts: list = [
        html.Div(
            [
                html.Span(icon("check", 16), className="ic"),
                html.Div(
                    [
                        "Датасет ",
                        html.Strong(ds_name),
                        " загружен: ",
                        html.Span(
                            f"{n_rows:,} строк × {n_cols} колонок".replace(",", " "),
                            className="mono",
                        ),
                        " · ",
                        html.Span(f"{size_mb:.1f} МБ", className="mono"),
                    ]
                ),
            ],
            className="kb-alert kb-alert--success",
        )
    ]
    if suspect_cols:
        alerts.append(
            html.Div(
                [
                    html.Span(icon("alert", 16), className="ic"),
                    html.Div(
                        [
                            html.Strong("Качество данных: "),
                            f"колонок с выбросами — {len(suspect_cols)}. ",
                            "Рекомендуется просмотреть: ",
                            html.Span(", ".join(suspect_cols[:8]), className="mono"),
                            ("…" if len(suspect_cols) > 8 else ""),
                        ]
                    ),
                ],
                className="kb-alert kb-alert--warning",
            )
        )

    kpis = html.Div(
        [
            kpi("Строк",      f"{n_rows:,}".replace(",", " ")),
            kpi("Столбцов",   str(n_cols)),
            kpi("Числовых",   str(n_num)),
            kpi("Пропусков",  f"{n_null:,}".replace(",", " ")),
            kpi("Дублей",     f"{n_dups:,}".replace(",", " ")),
        ],
        className="kb-data-kpis",
    )

    preview = card(
        title="Предпросмотр",
        head_right=html.Div(
            [
                chip(ds_name, "neutral"),
                html.Button(
                    [icon("funnel", 12), html.Span("Фильтры")],
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                ),
                html.Button(
                    [icon("grid", 12), html.Span(f"Столбцы ({n_cols})")],
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                ),
            ],
            className="kb-data-preview-head-tools",
        ),
        size="md",
        children=[
            html.Div(
                data_table(df.head(50), id=f"preview-{ds_name}", page_size=10),
                className="kb-data-preview-tbl",
            )
        ],
    )

    return html.Div([html.Div(alerts, className="kb-data-alerts"), kpis, preview])


# ---------------------------------------------------------------------------
# Callback: upload files
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
            _, content_data = content_str.split(",", 1)
            decoded = base64.b64decode(content_data)

            df = load_file(decoded, filename=fname, sep=sep)
            ds_name = fname.rsplit(".", 1)[0] if "." in fname else fname
            path = save_dataframe(df, ds_name)
            ds_store[ds_name] = path
            results.append(_render_upload_result(df, ds_name, len(decoded)))
        except Exception as exc:
            results.append(alert_banner(f"{fname}: {exc}", "danger"))

    return html.Div(results), ds_store


# ---------------------------------------------------------------------------
# Right-rail: recent sources + schema of active dataset
# ---------------------------------------------------------------------------
def _recent_source_row(name: str, meta: str) -> html.Div:
    return html.Div(
        [
            html.Div(icon("table", 14), className="kb-data-recent-icon"),
            html.Div(
                [
                    html.Div(name, className="kb-data-recent-name"),
                    html.Div(meta, className="kb-data-recent-meta"),
                ],
                className="kb-data-recent-text",
            ),
        ],
        className="kb-data-recent-row",
    )


@callback(
    Output("data-recent-rail", "children"),
    Output("data-schema-rail", "children"),
    Input(STORE_DATASET, "data"),
)
def update_rail(ds_store):
    ds_store = ds_store or {}
    names = list_datasets(ds_store)

    # Recent sources card
    if not names:
        recent_body = html.Div(
            "Пока нет загруженных датасетов.",
            className="kb-data-recent-empty",
        )
    else:
        rows = []
        for name in names[-5:][::-1]:
            df = get_df_from_store(ds_store, name)
            meta = (f"{df.shape[0]:,} × {df.shape[1]}".replace(",", " ")
                    if df is not None else "—")
            rows.append(_recent_source_row(name, meta))
        recent_body = html.Div(rows, className="kb-data-recent-list")

    recent_card = card(
        title=None,
        size="sm",
        children=[
            html.Div("Недавние источники", className="kb-overline"),
            recent_body,
        ],
    )

    # Schema card — based on the last dataset loaded (most recent entry)
    if not names:
        schema_card = card(
            title=None,
            size="sm",
            children=[
                html.Div("Схема", className="kb-overline"),
                html.Div("Загрузите датасет, чтобы увидеть схему.",
                         className="kb-data-schema-empty"),
            ],
        )
    else:
        last = names[-1]
        df = get_df_from_store(ds_store, last)
        if df is None:
            schema_card = card(
                title=None, size="sm",
                children=[
                    html.Div("Схема", className="kb-overline"),
                    html.Div("Не удалось прочитать датасет.",
                             className="kb-data-schema-empty"),
                ],
            )
        else:
            total = df.shape[1]
            shown_cols = list(df.columns[:8])
            rows = []
            for col in shown_cols:
                rows.append(
                    html.Div(
                        [
                            html.Span(col, className="kb-data-schema-name"),
                            html.Span(str(df[col].dtype),
                                      className="kb-data-schema-type"),
                        ],
                        className="kb-data-schema-row",
                    )
                )
            if total > len(shown_cols):
                rows.append(
                    html.Div(
                        f"+ {total - len(shown_cols)} more…",
                        className="kb-data-schema-more",
                    )
                )
            schema_card = card(
                title=None, size="sm",
                children=[
                    html.Div("Схема", className="kb-overline"),
                    html.Div(
                        f"{total} колонок · auto-detected · {last}",
                        className="kb-data-schema-summary",
                    ),
                    html.Div(rows, className="kb-data-schema-list"),
                ],
            )

    return recent_card, schema_card


# ---------------------------------------------------------------------------
# PostgreSQL callbacks
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
        return alert_banner(
            "Заполните хост, порт, базу данных и пользователя.", "warning"
        )
    try:
        from services.db import test_connection
        ok, msg = test_connection(host, port, database, user, password)
        if ok:
            return alert_banner(f"Подключено к {database} на {host}:{port}", "success")
        return alert_banner(f"Ошибка: {msg}", "danger")
    except ImportError:
        return alert_banner(
            "Модуль services.db не найден. PostgreSQL-подключение недоступно.",
            "warning",
        )
    except Exception as exc:
        return alert_banner(f"Ошибка: {exc}", "danger")


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
            return alert_banner("Запрос вернул 0 строк.", "warning"), no_update

        ds_store = ds_store or {}
        path = save_dataframe(df, ds_name or "pg_query")
        ds_store[ds_name or "pg_query"] = path

        return html.Div(
            [
                alert_banner(
                    f"Датасет «{ds_name}» загружен: {len(df):,} × {len(df.columns)}.".replace(",", " "),
                    "success",
                ),
                data_table(df.head(50), id="pg-preview-tbl", page_size=10),
            ]
        ), ds_store
    except ImportError:
        return alert_banner("Модуль services.db не найден.", "warning"), no_update
    except Exception as exc:
        return alert_banner(f"Ошибка: {exc}", "danger"), no_update


# ---------------------------------------------------------------------------
# Catalog callbacks
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
            description="Загрузите датасет во вкладке «Загрузка файлов»",
        )

    df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return alert_banner("Не удалось загрузить датасет.", "warning")

    n_rows, n_cols = df.shape
    n_num = len(df.select_dtypes(include="number").columns)
    n_null = int(df.isnull().sum().sum())
    n_dups = int(df.duplicated().sum())

    schema_df = pd.DataFrame(
        {
            "Колонка": df.columns,
            "Тип": df.dtypes.astype(str).values,
            "Заполнено %": (df.notna().mean() * 100).round(1).values,
            "Уникальных": df.nunique().values,
            "Пример": [
                str(df[c].dropna().iloc[0]) if df[c].notna().any() else "—"
                for c in df.columns
            ],
        }
    )

    kpis = html.Div(
        [
            kpi("Строк",     f"{n_rows:,}".replace(",", " ")),
            kpi("Столбцов",  str(n_cols)),
            kpi("Числовых",  str(n_num)),
            kpi("Пропусков", f"{n_null:,}".replace(",", " ")),
            kpi("Дублей",    f"{n_dups:,}".replace(",", " ")),
        ],
        className="kb-data-kpis",
    )

    return html.Div(
        [
            kpis,
            card(
                title="Предпросмотр",
                head_right=chip(ds_name, "neutral"),
                children=[data_table(df.head(50), id="catalog-data-tbl", page_size=10)],
            ),
            card(
                title="Схема данных",
                children=[data_table(schema_df, id="catalog-schema-tbl", page_size=20)],
            ),
        ]
    )

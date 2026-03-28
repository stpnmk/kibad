"""
pages/p02_prepare.py -- Data preparation wizard: 9-step pipeline.
"""
from __future__ import annotations

import json

import dash
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np

from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.form import select_input, number_input, checklist_input
from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
    get_df_from_store, save_dataframe, list_datasets,
)
from core.prepare import (
    parse_dates, resample_timeseries, impute_missing, remove_outliers,
    deduplicate, add_lags, add_rolling, add_ema, add_buckets, normalize,
    add_interaction,
)

dash.register_page(
    __name__,
    path="/prepare",
    name="2. Подготовка",
    order=2,
    icon="wrench",
)

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------
_STEPS = [
    ("1. Маппинг колонок", "tab-prep-mapping"),
    ("2. Типы данных", "tab-prep-types"),
    ("3. Парсинг дат", "tab-prep-dates"),
    ("4. Числовой парсинг", "tab-prep-numeric"),
    ("5. Заполнение пропусков", "tab-prep-impute"),
    ("6. Удаление выбросов", "tab-prep-outliers"),
    ("7. Дедупликация", "tab-prep-dedup"),
    ("8. Ресэмплинг", "tab-prep-resample"),
    ("9. Генерация фич", "tab-prep-features"),
]

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div([
    page_header("2. Подготовка данных", "Очистка, трансформация и обогащение"),

    # Dataset selector
    dbc.Row([
        dbc.Col([
            html.Label("Датасет для обработки", className="kb-stat-label",
                       style={"marginBottom": "6px"}),
            dcc.Dropdown(id="prep-ds-select", className="kb-select"),
        ], md=4),
    ], className="mb-3"),

    # Before stats
    dbc.Row(id="prep-stats-row", className="mb-3"),

    # Tabs wizard
    dbc.Tabs(
        [dbc.Tab(label=label, tab_id=tab_id) for label, tab_id in _STEPS],
        id="prep-step-tabs",
        active_tab="tab-prep-mapping",
    ),
    html.Div(id="prep-step-content", className="mt-3"),

    # Result area
    dcc.Loading(
        html.Div(id="prep-result"),
        type="circle", color="#10b981",
    ),
], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px 16px"})


# ---------------------------------------------------------------------------
# Callback: populate dataset dropdown
# ---------------------------------------------------------------------------
@callback(
    Output("prep-ds-select", "options"),
    Output("prep-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def populate_ds_dropdown(ds_store, active_ds):
    names = list_datasets(ds_store)
    options = [{"label": n, "value": n} for n in names]
    value = active_ds if active_ds in names else (names[0] if names else None)
    return options, value


# ---------------------------------------------------------------------------
# Callback: stats row
# ---------------------------------------------------------------------------
@callback(
    Output("prep-stats-row", "children"),
    Input("prep-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_stats_row(ds_name, ds_store, prep_store):
    if not ds_name:
        return []
    # Use prepared version if available, else raw
    df = get_df_from_store(prep_store, ds_name) if prep_store else None
    if df is None:
        df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return []

    n_rows, n_cols = df.shape
    n_null = int(df.isnull().sum().sum())
    n_dup = int(df.duplicated().sum())

    return [
        dbc.Col(stat_card("Строк", f"{n_rows:,}"), md=3),
        dbc.Col(stat_card("Столбцов", str(n_cols)), md=3),
        dbc.Col(stat_card("Пропусков", f"{n_null:,}"), md=3),
        dbc.Col(stat_card("Дублей", f"{n_dup:,}"), md=3),
    ]


# ---------------------------------------------------------------------------
# Callback: render step content
# ---------------------------------------------------------------------------
@callback(
    Output("prep-step-content", "children"),
    Input("prep-step-tabs", "active_tab"),
    Input("prep-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_step(active_tab, ds_name, ds_store, prep_store):
    if not ds_name:
        return empty_state(
            icon="",
            title="Данные не загружены",
            description="Загрузите датасет на странице Данные",
        )

    df = get_df_from_store(prep_store, ds_name) if prep_store else None
    if df is None:
        df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return dbc.Alert("Не удалось загрузить датасет.", color="warning")

    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]

    # --- Step 1: Column Mapping ---
    if active_tab == "tab-prep-mapping":
        return html.Div([
            section_header("Маппинг колонок", "Переименование столбцов"),
            html.P("Выберите столбцы для переименования. Оставьте пустым, чтобы не менять.",
                   style={"color": "var(--text-muted)", "fontSize": "0.88rem"}),
            html.Div([
                dbc.Row([
                    dbc.Col(html.Span(c, style={"fontWeight": "600"}), md=4),
                    dbc.Col(dcc.Input(
                        id={"type": "rename-input", "index": i},
                        value=c,
                        placeholder="Новое имя",
                        style={"width": "100%"},
                    ), md=4),
                ], className="mb-2")
                for i, c in enumerate(all_cols[:20])  # limit to 20 cols in UI
            ]),
            dbc.Button("Применить переименование", id="prep-btn-rename",
                       color="primary", className="mt-2"),
        ])

    # --- Step 2: Type overrides ---
    if active_tab == "tab-prep-types":
        type_options = ["(auto)", "datetime", "numeric", "categorical", "boolean"]
        rows = []
        for i, c in enumerate(all_cols[:30]):
            rows.append(dbc.Row([
                dbc.Col(html.Span(f"{c} ({df[c].dtype})", style={"fontSize": "0.85rem"}), md=5),
                dbc.Col(dcc.Dropdown(
                    id={"type": "type-override", "index": i},
                    options=[{"label": t, "value": t} for t in type_options],
                    value="(auto)",
                    clearable=False,
                    className="kb-select",
                ), md=4),
            ], className="mb-1"))
        return html.Div([
            section_header("Переопределение типов"),
            html.P("Измените тип данных столбца вручную.",
                   style={"color": "var(--text-muted)", "fontSize": "0.88rem"}),
            html.Div(rows),
            dbc.Button("Применить типы", id="prep-btn-types", color="primary", className="mt-2"),
        ])

    # --- Step 3: Date parsing ---
    if active_tab == "tab-prep-dates":
        return html.Div([
            section_header("Парсинг дат"),
            select_input("Столбец с датой", "prep-date-col",
                         options=all_cols, value=date_cols[0] if date_cols else None),
            dbc.Button("Парсить даты", id="prep-btn-dates", color="primary"),
        ])

    # --- Step 4: Numeric parsing ---
    if active_tab == "tab-prep-numeric":
        return html.Div([
            section_header("Числовой парсинг"),
            html.P("Преобразование текстовых столбцов в числовой тип (удаление пробелов, запятых).",
                   style={"color": "var(--text-muted)", "fontSize": "0.88rem"}),
            select_input("Столбцы для преобразования", "prep-num-cols",
                         options=cat_cols, multi=True),
            dbc.Button("Преобразовать в числа", id="prep-btn-numeric", color="primary"),
        ])

    # --- Step 5: Imputation ---
    if active_tab == "tab-prep-impute":
        cols_with_nulls = [c for c in all_cols if df[c].isnull().any()]
        return html.Div([
            section_header("Заполнение пропусков"),
            select_input("Столбцы с пропусками", "prep-impute-cols",
                         options=cols_with_nulls, multi=True,
                         value=cols_with_nulls[:5]),
            select_input("Метод заполнения", "prep-impute-method",
                         options=[
                             {"label": "Медиана", "value": "median"},
                             {"label": "Среднее", "value": "mean"},
                             {"label": "Мода", "value": "mode"},
                             {"label": "Заполнить вперёд (ffill)", "value": "ffill"},
                             {"label": "Заполнить назад (bfill)", "value": "bfill"},
                             {"label": "Нулём", "value": "zero"},
                             {"label": "Удалить строки", "value": "drop"},
                         ],
                         value="median"),
            dbc.Button("Заполнить пропуски", id="prep-btn-impute", color="primary"),
        ])

    # --- Step 6: Outlier removal ---
    if active_tab == "tab-prep-outliers":
        return html.Div([
            section_header("Удаление выбросов"),
            select_input("Числовые столбцы", "prep-outlier-cols",
                         options=num_cols, multi=True,
                         value=num_cols[:3]),
            select_input("Метод", "prep-outlier-method",
                         options=[
                             {"label": "IQR (межквартильный размах)", "value": "iqr"},
                             {"label": "Z-score", "value": "zscore"},
                         ],
                         value="iqr"),
            number_input("Порог (IQR multiplier / Z-score)", "prep-outlier-threshold",
                         value=1.5, min_val=0.5, max_val=5, step=0.1),
            dbc.Button("Удалить выбросы", id="prep-btn-outliers", color="primary"),
        ])

    # --- Step 7: Deduplication ---
    if active_tab == "tab-prep-dedup":
        n_dup = int(df.duplicated().sum())
        return html.Div([
            section_header("Дедупликация"),
            dbc.Alert(f"Обнаружено дубликатов: {n_dup:,}", color="info"),
            select_input("Подмножество столбцов (пусто = все)", "prep-dedup-cols",
                         options=all_cols, multi=True),
            dbc.Button("Удалить дубликаты", id="prep-btn-dedup", color="primary"),
        ])

    # --- Step 8: Resample ---
    if active_tab == "tab-prep-resample":
        return html.Div([
            section_header("Ресэмплинг временного ряда"),
            select_input("Столбец даты", "prep-resample-datecol",
                         options=date_cols if date_cols else all_cols,
                         value=date_cols[0] if date_cols else None),
            select_input("Период", "prep-resample-freq",
                         options=[
                             {"label": "День", "value": "D"},
                             {"label": "Неделя", "value": "W"},
                             {"label": "Месяц", "value": "MS"},
                             {"label": "Квартал", "value": "QS"},
                             {"label": "Год", "value": "YS"},
                         ],
                         value="MS"),
            select_input("Числовые столбцы для агрегации", "prep-resample-valuecols",
                         options=num_cols, multi=True,
                         value=num_cols[:3]),
            dbc.Button("Ресэмплить", id="prep-btn-resample", color="primary"),
        ])

    # --- Step 9: Feature engineering ---
    if active_tab == "tab-prep-features":
        return html.Div([
            section_header("Генерация фич"),
            dbc.Accordion([
                dbc.AccordionItem([
                    select_input("Столбец", "prep-lag-col",
                                 options=num_cols, value=num_cols[0] if num_cols else None),
                    number_input("Лаги (через запятую, напр. 1,3,7)", "prep-lag-periods",
                                 value=1),
                    dbc.Button("Добавить лаги", id="prep-btn-lags", color="primary",
                               size="sm"),
                ], title="Лаги"),
                dbc.AccordionItem([
                    select_input("Столбец", "prep-roll-col",
                                 options=num_cols, value=num_cols[0] if num_cols else None),
                    number_input("Окно", "prep-roll-window", value=7, min_val=2, max_val=365),
                    dbc.Button("Добавить скользящее среднее", id="prep-btn-rolling",
                               color="primary", size="sm"),
                ], title="Скользящее среднее"),
                dbc.AccordionItem([
                    select_input("Столбцы для нормализации", "prep-norm-cols",
                                 options=num_cols, multi=True),
                    select_input("Метод", "prep-norm-method",
                                 options=[
                                     {"label": "Min-Max [0, 1]", "value": "minmax"},
                                     {"label": "Z-score", "value": "zscore"},
                                     {"label": "Robust (медиана)", "value": "robust"},
                                 ],
                                 value="minmax"),
                    dbc.Button("Нормализовать", id="prep-btn-normalize", color="primary",
                               size="sm"),
                ], title="Нормализация"),
                dbc.AccordionItem([
                    select_input("Столбец", "prep-bucket-col",
                                 options=num_cols, value=num_cols[0] if num_cols else None),
                    number_input("Количество бинов", "prep-bucket-n", value=5,
                                 min_val=2, max_val=20),
                    dbc.Button("Создать бины", id="prep-btn-buckets", color="primary",
                               size="sm"),
                ], title="Бинирование (бакеты)"),
            ], start_collapsed=True),
        ])

    return html.Div()


# ---------------------------------------------------------------------------
# Callback: Apply preparation steps (unified handler)
# ---------------------------------------------------------------------------
@callback(
    Output("prep-result", "children"),
    Output(STORE_PREPARED, "data", allow_duplicate=True),
    Output("prep-stats-row", "children", allow_duplicate=True),
    # Inputs -- buttons for each step
    Input("prep-btn-impute", "n_clicks"),
    Input("prep-btn-outliers", "n_clicks"),
    Input("prep-btn-dedup", "n_clicks"),
    Input("prep-btn-dates", "n_clicks"),
    Input("prep-btn-resample", "n_clicks"),
    Input("prep-btn-lags", "n_clicks"),
    Input("prep-btn-rolling", "n_clicks"),
    Input("prep-btn-normalize", "n_clicks"),
    Input("prep-btn-buckets", "n_clicks"),
    # States
    State("prep-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    # Imputation
    State("prep-impute-cols", "value"),
    State("prep-impute-method", "value"),
    # Outliers
    State("prep-outlier-cols", "value"),
    State("prep-outlier-method", "value"),
    State("prep-outlier-threshold", "value"),
    # Dedup
    State("prep-dedup-cols", "value"),
    # Date parsing
    State("prep-date-col", "value"),
    # Resample
    State("prep-resample-datecol", "value"),
    State("prep-resample-freq", "value"),
    State("prep-resample-valuecols", "value"),
    # Lags
    State("prep-lag-col", "value"),
    State("prep-lag-periods", "value"),
    # Rolling
    State("prep-roll-col", "value"),
    State("prep-roll-window", "value"),
    # Normalize
    State("prep-norm-cols", "value"),
    State("prep-norm-method", "value"),
    # Buckets
    State("prep-bucket-col", "value"),
    State("prep-bucket-n", "value"),
    prevent_initial_call=True,
)
def apply_step(
    n_imp, n_out, n_ded, n_dat, n_res, n_lag, n_roll, n_norm, n_bkt,
    ds_name, ds_store, prep_store,
    imp_cols, imp_method,
    out_cols, out_method, out_thresh,
    dedup_cols,
    date_col,
    resample_datecol, resample_freq, resample_valuecols,
    lag_col, lag_periods,
    roll_col, roll_window,
    norm_cols, norm_method,
    bucket_col, bucket_n,
):
    if not ds_name:
        return no_update, no_update, no_update

    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update, no_update

    # Load working dataframe
    prep_store = prep_store or {}
    df = get_df_from_store(prep_store, ds_name)
    if df is None:
        df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return dbc.Alert("Датасет не найден.", color="danger"), no_update, no_update

    before_rows = len(df)
    before_nulls = int(df.isnull().sum().sum())
    operation = ""

    try:
        if triggered == "prep-btn-impute" and imp_cols:
            for col in imp_cols:
                if col in df.columns:
                    df = impute_missing(df, col, method=imp_method)
            operation = f"Заполнение пропусков ({imp_method})"

        elif triggered == "prep-btn-outliers" and out_cols:
            for col in out_cols:
                if col in df.columns:
                    df = remove_outliers(df, col, method=out_method,
                                         threshold=float(out_thresh or 1.5))
            operation = f"Удаление выбросов ({out_method})"

        elif triggered == "prep-btn-dedup":
            subset = dedup_cols if dedup_cols else None
            df = deduplicate(df, subset=subset)
            operation = "Дедупликация"

        elif triggered == "prep-btn-dates" and date_col:
            df = parse_dates(df, date_col)
            operation = f"Парсинг дат ({date_col})"

        elif triggered == "prep-btn-resample" and resample_datecol and resample_freq:
            value_cols = resample_valuecols or df.select_dtypes(include="number").columns.tolist()
            df = resample_timeseries(df, resample_datecol, resample_freq, value_cols)
            operation = f"Ресэмплинг ({resample_freq})"

        elif triggered == "prep-btn-lags" and lag_col:
            periods = [int(lag_periods)] if lag_periods else [1]
            df = add_lags(df, lag_col, periods)
            operation = f"Лаги ({lag_col}, {periods})"

        elif triggered == "prep-btn-rolling" and roll_col:
            window = int(roll_window or 7)
            df = add_rolling(df, roll_col, window)
            operation = f"Скользящее среднее ({roll_col}, окно={window})"

        elif triggered == "prep-btn-normalize" and norm_cols:
            df = normalize(df, norm_cols, method=norm_method)
            operation = f"Нормализация ({norm_method})"

        elif triggered == "prep-btn-buckets" and bucket_col:
            n_bins = int(bucket_n or 5)
            df = add_buckets(df, bucket_col, n_bins=n_bins)
            operation = f"Бинирование ({bucket_col}, {n_bins} бинов)"

        else:
            return dbc.Alert("Выберите параметры операции.", color="warning"), no_update, no_update

    except Exception as exc:
        return dbc.Alert(f"Ошибка: {exc}", color="danger"), no_update, no_update

    # Save result
    path = save_dataframe(df, f"{ds_name}_prepared")
    prep_store[ds_name] = path

    after_rows = len(df)
    after_nulls = int(df.isnull().sum().sum())
    n_dup = int(df.duplicated().sum())

    # Before/after card
    result_card = html.Div([
        dbc.Alert(f"Применено: {operation}", color="success"),
        dbc.Row([
            dbc.Col(stat_card("Строк (до)", f"{before_rows:,}"), md=3),
            dbc.Col(stat_card("Строк (после)", f"{after_rows:,}"), md=3),
            dbc.Col(stat_card("Пропусков (до)", f"{before_nulls:,}"), md=3),
            dbc.Col(stat_card("Пропусков (после)", f"{after_nulls:,}"), md=3),
        ], className="mb-3"),
        data_table(df.head(10), id="prep-result-tbl", page_size=10),
    ])

    stats_row = [
        dbc.Col(stat_card("Строк", f"{after_rows:,}"), md=3),
        dbc.Col(stat_card("Столбцов", str(df.shape[1])), md=3),
        dbc.Col(stat_card("Пропусков", f"{after_nulls:,}"), md=3),
        dbc.Col(stat_card("Дублей", f"{n_dup:,}"), md=3),
    ]

    return result_card, prep_store, stats_row

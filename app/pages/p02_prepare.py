"""p02_prepare — Data preparation wizard: 9-step pipeline.

Design port of Slide 16 «Подготовка данных». Layout:
- Hero header with overline ("Шаг N из 9 · {step}"), H1, dataset
  subtitle, and action buttons (dataset select, reset all, save).
- Custom 9-step horizontal stepper (replaces `dbc.Tabs`): each node
  shows done/active/pending state with a connecting accent line. Nodes
  are clickable to jump between steps.
- Four-KPI stats row (rows / cols / nulls / dupes).
- Split layout — left: step form card; right: «История преобразований»
  panel tracking applied operations in session.
- Result card below, showing before/after metrics and a preview table.

Core callbacks preserved from the previous version; only the UI shell
and styling are rebuilt on `kb-*` classes.
"""
from __future__ import annotations

import dash
from dash import (
    ALL, Input, Output, State, callback, ctx, dcc, html, no_update,
)
import dash_bootstrap_components as dbc
import pandas as pd

from app.components.alerts import alert_banner
from app.components.cards import chip
from app.components.icons import icon
from app.components.layout import empty_state
from app.components.table import data_table
from app.state import (
    STORE_ACTIVE_DS, STORE_DATASET, STORE_PREPARED,
    get_df_from_store, list_datasets, save_dataframe,
)
from core.prepare import (
    add_buckets, add_lags, add_rolling, deduplicate, impute_missing,
    normalize, parse_dates, remove_outliers, resample_timeseries,
)

dash.register_page(
    __name__,
    path="/prepare",
    name="2. Подготовка",
    order=2,
    icon="wrench",
)


# ---------------------------------------------------------------------------
# Step catalogue — (tab_id, short label, full title for overline)
# ---------------------------------------------------------------------------
_STEPS: list[tuple[str, str, str]] = [
    ("tab-prep-mapping",  "Маппинг",         "Маппинг колонок"),
    ("tab-prep-types",    "Типы",            "Переопределение типов"),
    ("tab-prep-dates",    "Даты",            "Парсинг дат"),
    ("tab-prep-numeric",  "Числа",           "Числовой парсинг"),
    ("tab-prep-impute",   "Пропуски",        "Заполнение пропусков"),
    ("tab-prep-outliers", "Выбросы",         "Удаление выбросов"),
    ("tab-prep-dedup",    "Дедуп",           "Дедупликация"),
    ("tab-prep-resample", "Ресэмплинг",      "Ресэмплинг временного ряда"),
    ("tab-prep-features", "Генерация фич",   "Генерация фич"),
]
_STEP_INDEX = {tab: i for i, (tab, _s, _t) in enumerate(_STEPS)}

# Map each step's op code back to its tab — used to mark the step as "done"
# in the history stepper when its apply button is clicked.
_OP_TO_TAB = {
    "dates":     "tab-prep-dates",
    "impute":    "tab-prep-impute",
    "outliers":  "tab-prep-outliers",
    "dedup":     "tab-prep-dedup",
    "resample":  "tab-prep-resample",
    # feature-engineering ops all count as the last step
    "lags":      "tab-prep-features",
    "rolling":   "tab-prep-features",
    "normalize": "tab-prep-features",
    "buckets":   "tab-prep-features",
}


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------
def _hero() -> html.Div:
    """Top hero — overline (filled by callback), H1, subtitle, action row."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(id="prep-overline", className="kb-overline"),
                    html.H1("Подготовка данных",
                            className="kb-h1 kb-prep-hero__title"),
                    html.Div(id="prep-subtitle", className="kb-body-l kb-prep-hero__sub"),
                ],
                className="kb-prep-hero__left",
            ),
            html.Div(
                [
                    dcc.Dropdown(
                        id="prep-ds-select",
                        className="kb-select kb-prep-hero__ds",
                        placeholder="Выберите датасет",
                        clearable=False,
                    ),
                    html.Button(
                        [icon("refresh", 12), html.Span("Сбросить шаги")],
                        id="prep-btn-reset",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("check", 12), html.Span("Сохранить как prepared")],
                        id="prep-btn-save",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                        disabled=True,
                    ),
                ],
                className="kb-prep-hero__right",
            ),
        ],
        className="kb-card kb-prep-hero",
    )


def _wizard(active_tab: str, done_set: set[str]) -> html.Div:
    """9-step horizontal stepper matching Slide 16.

    A 9-column grid. Each node is a circular badge + short label, with a
    connecting line between siblings. Whole node is clickable via pattern-
    matching id ``{"type": "prep-step", "tab": ...}``.
    """
    nodes: list = []
    for i, (tab, short, _full) in enumerate(_STEPS):
        if tab in done_set:
            state_cls = "is-done"
        elif tab == active_tab:
            state_cls = "is-active"
        else:
            state_cls = ""
        # connecting line between step i and i+1 — accent if THIS step is
        # done (segment to the right), otherwise muted.
        line = None
        if i < len(_STEPS) - 1:
            line_cls = "kb-wizard__line"
            if tab in done_set:
                line_cls += " is-done"
            line = html.Div(className=line_cls)

        badge_content = icon("check", 12) if tab in done_set else str(i + 1)
        nodes.append(
            html.Button(
                [
                    line,
                    html.Div(badge_content, className="kb-wizard__node"),
                    html.Div(short, className="kb-wizard__label"),
                ],
                id={"type": "prep-step", "tab": tab},
                className=f"kb-wizard__step {state_cls}".rstrip(),
                n_clicks=0,
            )
        )
    return html.Div(nodes, className="kb-wizard")


def _kpi_row(df: pd.DataFrame | None) -> html.Div:
    """Four-KPI row — rows / cols / nulls / dupes with trend captions."""
    def tile(label: str, value: str, trend: str,
             value_cls: str = "kb-stat-value") -> html.Div:
        return html.Div(
            [
                html.Div(label, className="kb-stat-label"),
                html.Div(
                    [
                        html.Div(value, className=value_cls),
                        html.Div(trend, className="kb-stat-hint"),
                    ],
                    className="kb-stat-row",
                ),
            ],
            className="kb-stat-card",
        )

    if df is None:
        return html.Div(
            [tile(lbl, "—", "—") for lbl in ("Строк", "Столбцов", "Пропусков", "Дублей")],
            className="kb-prep-stats",
        )

    n_rows, n_cols = df.shape
    nulls_by_col = df.isnull().sum()
    n_null = int(nulls_by_col.sum())
    n_null_cols = int((nulls_by_col > 0).sum())
    n_dup = int(df.duplicated().sum())

    null_value_cls = "kb-stat-value" + (" kb-stat-value--warn" if n_null else "")
    dup_value_cls = "kb-stat-value" + (" kb-stat-value--warn" if n_dup else "")

    return html.Div(
        [
            tile("Строк",     f"{n_rows:,}".replace(",", " "), "всего"),
            tile("Столбцов",  str(n_cols), "в датасете"),
            tile("Пропусков", f"{n_null:,}".replace(",", " "),
                 f"в {n_null_cols} столбцах" if n_null_cols else "нет",
                 value_cls=null_value_cls),
            tile("Дублей",    f"{n_dup:,}".replace(",", " "),
                 "полных" if n_dup else "нет",
                 value_cls=dup_value_cls),
        ],
        className="kb-prep-stats",
    )


def _history_panel(history: list[dict]) -> html.Div:
    """Right-column history panel — list of applied operations."""
    if not history:
        body = html.Div(
            [
                html.Div(icon("settings", 18),
                         className="kb-prep-history__empty-icon"),
                html.Div("Ещё ни один шаг не применён",
                         className="kb-prep-history__empty-title"),
                html.Div(
                    "Выберите шаг слева и нажмите «Применить». История преобразований появится здесь.",
                    className="kb-prep-history__empty-desc",
                ),
            ],
            className="kb-prep-history__empty",
        )
    else:
        items = []
        for n, entry in enumerate(history, start=1):
            items.append(
                html.Div(
                    [
                        html.Div(icon("check", 12),
                                 className="kb-prep-history__num"),
                        html.Div(
                            [
                                html.Div(entry["op"], className="kb-prep-history__op"),
                                html.Div(entry.get("detail", ""),
                                         className="kb-prep-history__detail"),
                            ],
                            className="kb-prep-history__body",
                        ),
                        html.Div(entry.get("time", ""),
                                 className="kb-prep-history__time"),
                    ],
                    className="kb-prep-history__item",
                )
            )
        body = html.Div(items, className="kb-prep-history__list")

    head = html.Div(
        [
            html.Div(
                [
                    html.H3("История преобразований"),
                    html.Div(
                        f"{len(history)} шагов применено"
                        + (" · откат недоступен" if history else ""),
                        className="caption",
                    ),
                ],
                className="kb-card-title",
            ),
        ],
        className="kb-card-head",
    )

    return html.Div([head, body], className="kb-card kb-prep-history")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        # Page-local stores
        dcc.Store(id="prep-active-step", data=_STEPS[0][0]),
        dcc.Store(id="prep-history-store", data=[]),

        _hero(),

        html.Div(id="prep-stepper-wrap"),

        html.Div(id="prep-stats-wrap"),

        # split: form (left) + history (right)
        html.Div(
            [
                html.Div(
                    [
                        html.Div(id="prep-step-content"),
                        dcc.Loading(
                            html.Div(id="prep-result"),
                            type="circle", color="var(--accent-500)",
                        ),
                    ],
                    className="kb-prep-layout__main",
                ),
                html.Div(id="prep-history-wrap",
                         className="kb-prep-layout__side"),
            ],
            className="kb-prep-layout",
        ),
    ],
    className="kb-page kb-prep-page",
)


# ---------------------------------------------------------------------------
# Callback: dataset dropdown
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
# Callback: stepper click → set active step
# ---------------------------------------------------------------------------
@callback(
    Output("prep-active-step", "data"),
    Input({"type": "prep-step", "tab": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_step_click(_clicks):
    trig = ctx.triggered_id
    if not isinstance(trig, dict):
        return no_update
    if not any(_clicks or []):
        return no_update
    return trig.get("tab", no_update)


# ---------------------------------------------------------------------------
# Callback: render wizard stepper (active + history-driven done marks)
# ---------------------------------------------------------------------------
@callback(
    Output("prep-stepper-wrap", "children"),
    Input("prep-active-step", "data"),
    Input("prep-history-store", "data"),
)
def render_stepper(active_tab, history):
    active_tab = active_tab or _STEPS[0][0]
    done = {entry.get("tab") for entry in (history or []) if entry.get("tab")}
    return _wizard(active_tab, done)


# ---------------------------------------------------------------------------
# Callback: overline + subtitle in the hero
# ---------------------------------------------------------------------------
@callback(
    Output("prep-overline", "children"),
    Output("prep-subtitle", "children"),
    Output("prep-btn-save", "disabled"),
    Input("prep-active-step", "data"),
    Input("prep-ds-select", "value"),
    Input("prep-history-store", "data"),
)
def render_hero_text(active_tab, ds_name, history):
    active_tab = active_tab or _STEPS[0][0]
    idx = _STEP_INDEX.get(active_tab, 0)
    _t, _short, full_title = _STEPS[idx]
    overline = f"Шаг {idx + 1} из 9 · {full_title}"

    if ds_name:
        subtitle = [
            "Очистка, трансформация и обогащение · датасет ",
            html.Span(ds_name, className="mono",
                       style={"color": "var(--text-primary)"}),
        ]
    else:
        subtitle = "Очистка, трансформация и обогащение"

    save_disabled = not (history and ds_name)
    return overline, subtitle, save_disabled


# ---------------------------------------------------------------------------
# Callback: stats row
# ---------------------------------------------------------------------------
@callback(
    Output("prep-stats-wrap", "children"),
    Input("prep-ds-select", "value"),
    Input("prep-history-store", "data"),  # re-render after operation
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_stats_row(ds_name, _history, ds_store, prep_store):
    if not ds_name:
        return _kpi_row(None)
    df = get_df_from_store(prep_store, ds_name) if prep_store else None
    if df is None:
        df = get_df_from_store(ds_store, ds_name)
    return _kpi_row(df)


# ---------------------------------------------------------------------------
# Callback: history panel
# ---------------------------------------------------------------------------
@callback(
    Output("prep-history-wrap", "children"),
    Input("prep-history-store", "data"),
)
def render_history(history):
    return _history_panel(history or [])


# ---------------------------------------------------------------------------
# Callback: render step form
# ---------------------------------------------------------------------------
def _step_head(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [
            html.Div(
                [html.H3(title), html.Div(subtitle, className="caption")],
                className="kb-card-title",
            ),
        ],
        className="kb-card-head",
    )


def _btn_apply(text: str, op: str) -> html.Button:
    """Apply button with pattern-matching ID ``{type: prep-apply, op}``."""
    return html.Button(
        [icon("play", 12), html.Span(text)],
        id={"type": "prep-apply", "op": op},
        className="kb-btn kb-btn--primary",
        n_clicks=0,
    )


def _mk_select(label: str, op: str, key: str, options: list,
               value=None, multi: bool = False) -> html.Div:
    """Labeled Dropdown with pattern-match id ``{type:prep-arg, op, key}``."""
    if options and isinstance(options[0], str):
        opts = [{"label": o, "value": o} for o in options]
    else:
        opts = options
    return html.Div([
        html.Label(label, className="kb-stat-label",
                   style={"marginBottom": "6px"}),
        dcc.Dropdown(
            id={"type": "prep-arg", "op": op, "key": key},
            options=opts,
            value=value,
            multi=multi,
            placeholder="Выберите...",
            clearable=True,
            className="kb-select",
        ),
    ], style={"marginBottom": "12px"})


def _mk_number(label: str, op: str, key: str, value=None,
               min_val=None, max_val=None, step=None) -> html.Div:
    """Labeled number input with pattern-match id."""
    return html.Div([
        html.Label(label, className="kb-stat-label",
                   style={"marginBottom": "6px"}),
        dcc.Input(
            id={"type": "prep-arg", "op": op, "key": key},
            type="number",
            value=value,
            min=min_val,
            max=max_val,
            step=step,
            style={"width": "100%"},
        ),
    ], style={"marginBottom": "12px"})


@callback(
    Output("prep-step-content", "children"),
    Input("prep-active-step", "data"),
    Input("prep-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_step(active_tab, ds_name, ds_store, prep_store):
    if not ds_name:
        return html.Div(
            empty_state(
                icon="",
                title="Данные не загружены",
                description="Загрузите датасет на странице «Данные».",
            ),
            className="kb-card",
        )

    df = get_df_from_store(prep_store, ds_name) if prep_store else None
    if df is None:
        df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return html.Div(
            alert_banner("Не удалось загрузить датасет.", level="warning"),
            className="kb-card",
        )

    active_tab = active_tab or _STEPS[0][0]
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]

    # --- Step 1: Column Mapping (informational placeholder) ---------------
    if active_tab == "tab-prep-mapping":
        return html.Div(
            [
                _step_head("Маппинг колонок",
                          "Переименование столбцов и быстрый обзор состава"),
                html.Div(
                    [
                        html.Div("Столбцы датасета",
                                 className="kb-field-label"),
                        html.Div(
                            [html.Span(c, className="kb-mchip") for c in all_cols[:60]],
                            className="kb-prep-colchips",
                        ),
                    ],
                    className="kb-field",
                ),
                alert_banner(
                    "Переименование выполняется на странице «Данные». "
                    "Здесь отображается текущая схема для контекста.",
                    level="info",
                ),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 2: Type overrides (informational placeholder) --------------
    if active_tab == "tab-prep-types":
        rows = []
        for c in all_cols[:30]:
            rows.append(
                html.Div(
                    [
                        html.Div(c, className="kb-prep-typerow__col mono"),
                        html.Div(str(df[c].dtype),
                                 className="kb-prep-typerow__type"),
                    ],
                    className="kb-prep-typerow",
                )
            )
        return html.Div(
            [
                _step_head("Переопределение типов",
                          "Текущие dtypes по столбцам датасета"),
                html.Div(rows, className="kb-prep-typegrid"),
                alert_banner(
                    "Типы переопределяются автоматически при парсинге дат / чисел "
                    "на следующих шагах.",
                    level="info",
                ),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 3: Date parsing --------------------------------------------
    if active_tab == "tab-prep-dates":
        return html.Div(
            [
                _step_head("Парсинг дат",
                          "Преобразование текстового столбца в datetime (формат auto)"),
                _mk_select("Столбец с датой", op="dates", key="col",
                           options=all_cols,
                           value=date_cols[0] if date_cols else (all_cols[0] if all_cols else None)),
                html.Div(_btn_apply("Парсить даты", "dates"),
                         className="kb-prep-actions"),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 4: Numeric parsing (informational) -------------------------
    if active_tab == "tab-prep-numeric":
        return html.Div(
            [
                _step_head(
                    "Числовой парсинг",
                    "Преобразование текстовых столбцов в числовой тип "
                    "(удаление пробелов, запятых)",
                ),
                _mk_select("Столбцы для преобразования",
                           op="numeric", key="cols",
                           options=cat_cols, multi=True),
                alert_banner(
                    "Отдельная операция недоступна — используйте «Парсинг дат» "
                    "или «Генерация фич → Нормализация» для обработки числовых.",
                    level="warning",
                ),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 5: Imputation ----------------------------------------------
    if active_tab == "tab-prep-impute":
        cols_with_nulls = [c for c in all_cols if df[c].isnull().any()]
        null_counts = {c: int(df[c].isnull().sum()) for c in cols_with_nulls}
        return html.Div(
            [
                _step_head(
                    "Заполнение пропусков",
                    "Выберите столбцы и метод заполнения. "
                    "Применяется к копии датасета — исходник не изменится.",
                ),
                _mk_select(
                    f"Столбцы с пропусками · {len(cols_with_nulls)} с NA",
                    op="impute", key="cols",
                    options=cols_with_nulls, multi=True,
                    value=cols_with_nulls[:5],
                ),
                _mk_select(
                    "Метод заполнения", op="impute", key="method",
                    options=[
                        {"label": "Медиана",                  "value": "median"},
                        {"label": "Среднее",                  "value": "mean"},
                        {"label": "Мода",                     "value": "mode"},
                        {"label": "Заполнить вперёд (ffill)", "value": "ffill"},
                        {"label": "Заполнить назад (bfill)",  "value": "bfill"},
                        {"label": "Нулём",                    "value": "zero"},
                        {"label": "Удалить строки",           "value": "drop"},
                    ],
                    value="median",
                ),
                (
                    alert_banner(
                        f"Топ-столбцы по пропускам: "
                        + ", ".join(
                            f"{c} ({null_counts[c]})"
                            for c in sorted(null_counts, key=null_counts.get, reverse=True)[:3]
                        ),
                        level="info",
                    )
                    if cols_with_nulls else
                    alert_banner("В датасете нет пропусков — шаг можно пропустить.",
                                 level="success")
                ),
                html.Div(
                    _btn_apply("Заполнить пропуски", "impute"),
                    className="kb-prep-actions",
                ),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 6: Outlier removal -----------------------------------------
    if active_tab == "tab-prep-outliers":
        return html.Div(
            [
                _step_head("Удаление выбросов",
                          "IQR или Z-score по выбранным числовым столбцам"),
                _mk_select("Числовые столбцы", op="outliers", key="cols",
                           options=num_cols, multi=True,
                           value=num_cols[:3]),
                _mk_select(
                    "Метод", op="outliers", key="method",
                    options=[
                        {"label": "IQR (межквартильный размах)", "value": "iqr"},
                        {"label": "Z-score",                     "value": "zscore"},
                    ],
                    value="iqr",
                ),
                _mk_number("Порог (IQR multiplier / Z-score)",
                           op="outliers", key="threshold",
                           value=1.5, min_val=0.5, max_val=5, step=0.1),
                html.Div(_btn_apply("Удалить выбросы", "outliers"),
                         className="kb-prep-actions"),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 7: Deduplication -------------------------------------------
    if active_tab == "tab-prep-dedup":
        n_dup = int(df.duplicated().sum())
        return html.Div(
            [
                _step_head("Дедупликация",
                          "Удаление повторяющихся строк по выбранным столбцам"),
                alert_banner(f"Обнаружено дубликатов: {n_dup:,}".replace(",", " "),
                             level="info" if n_dup else "success"),
                _mk_select("Подмножество столбцов (пусто = все)",
                           op="dedup", key="cols",
                           options=all_cols, multi=True),
                html.Div(_btn_apply("Удалить дубликаты", "dedup"),
                         className="kb-prep-actions"),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 8: Resample ------------------------------------------------
    if active_tab == "tab-prep-resample":
        return html.Div(
            [
                _step_head("Ресэмплинг временного ряда",
                          "Агрегация по периоду (день / неделя / месяц / …)"),
                _mk_select(
                    "Столбец даты", op="resample", key="datecol",
                    options=date_cols if date_cols else all_cols,
                    value=date_cols[0] if date_cols else None,
                ),
                _mk_select(
                    "Период", op="resample", key="freq",
                    options=[
                        {"label": "День",     "value": "D"},
                        {"label": "Неделя",   "value": "W"},
                        {"label": "Месяц",    "value": "MS"},
                        {"label": "Квартал",  "value": "QS"},
                        {"label": "Год",      "value": "YS"},
                    ],
                    value="MS",
                ),
                _mk_select("Числовые столбцы для агрегации",
                           op="resample", key="valuecols",
                           options=num_cols, multi=True, value=num_cols[:3]),
                html.Div(_btn_apply("Ресэмплить", "resample"),
                         className="kb-prep-actions"),
            ],
            className="kb-card kb-card--lg",
        )

    # --- Step 9: Feature engineering -------------------------------------
    if active_tab == "tab-prep-features":
        return html.Div(
            [
                _step_head("Генерация фич",
                          "Лаги, скользящие средние, нормализация, бинирование"),
                dbc.Accordion([
                    dbc.AccordionItem([
                        _mk_select("Столбец", op="lags", key="col",
                                   options=num_cols,
                                   value=num_cols[0] if num_cols else None),
                        _mk_number(
                            "Лаги (через запятую, напр. 1,3,7)",
                            op="lags", key="periods", value=1,
                        ),
                        html.Div(_btn_apply("Добавить лаги", "lags"),
                                 className="kb-prep-actions"),
                    ], title="Лаги"),
                    dbc.AccordionItem([
                        _mk_select("Столбец", op="rolling", key="col",
                                   options=num_cols,
                                   value=num_cols[0] if num_cols else None),
                        _mk_number("Окно", op="rolling", key="window",
                                   value=7, min_val=2, max_val=365),
                        html.Div(_btn_apply("Добавить скользящее среднее",
                                            "rolling"),
                                 className="kb-prep-actions"),
                    ], title="Скользящее среднее"),
                    dbc.AccordionItem([
                        _mk_select("Столбцы для нормализации",
                                   op="normalize", key="cols",
                                   options=num_cols, multi=True),
                        _mk_select(
                            "Метод", op="normalize", key="method",
                            options=[
                                {"label": "Min-Max [0, 1]",    "value": "minmax"},
                                {"label": "Z-score",           "value": "zscore"},
                                {"label": "Robust (медиана)",  "value": "robust"},
                            ],
                            value="minmax",
                        ),
                        html.Div(_btn_apply("Нормализовать", "normalize"),
                                 className="kb-prep-actions"),
                    ], title="Нормализация"),
                    dbc.AccordionItem([
                        _mk_select("Столбец", op="buckets", key="col",
                                   options=num_cols,
                                   value=num_cols[0] if num_cols else None),
                        _mk_number("Количество бинов", op="buckets", key="n",
                                   value=5, min_val=2, max_val=20),
                        html.Div(_btn_apply("Создать бины", "buckets"),
                                 className="kb-prep-actions"),
                    ], title="Бинирование (бакеты)"),
                ], start_collapsed=True),
            ],
            className="kb-card kb-card--lg",
        )

    return html.Div()


# ---------------------------------------------------------------------------
# Callback: apply transformation step
# ---------------------------------------------------------------------------
@callback(
    Output("prep-result", "children"),
    Output(STORE_PREPARED, "data", allow_duplicate=True),
    Output("prep-history-store", "data", allow_duplicate=True),
    # Apply buttons (pattern-matched) + reset
    Input({"type": "prep-apply", "op": ALL}, "n_clicks"),
    Input("prep-btn-reset", "n_clicks"),
    # Form field values (pattern-matched, parallel to `arg_ids`)
    State({"type": "prep-arg", "op": ALL, "key": ALL}, "value"),
    State({"type": "prep-arg", "op": ALL, "key": ALL}, "id"),
    # Shared state
    State("prep-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("prep-history-store", "data"),
    prevent_initial_call=True,
)
def apply_step(apply_clicks, n_reset,
               arg_values, arg_ids,
               ds_name, ds_store, prep_store, history):
    if not ds_name:
        return no_update, no_update, no_update

    triggered = ctx.triggered_id
    if not triggered:
        return no_update, no_update, no_update

    prep_store = dict(prep_store or {})
    history = list(history or [])

    # Reset: drop prepared snapshot + clear history
    if triggered == "prep-btn-reset":
        if not n_reset:
            return no_update, no_update, no_update
        prep_store.pop(ds_name, None)
        return (
            alert_banner("Все шаги сброшены — активна исходная копия датасета.",
                         level="info"),
            prep_store,
            [],
        )

    # Only continue for apply-button clicks
    if not (isinstance(triggered, dict) and triggered.get("type") == "prep-apply"):
        return no_update, no_update, no_update
    if not any(apply_clicks or []):
        return no_update, no_update, no_update

    op = triggered.get("op")

    # Build {key: value} dict for args belonging to this op
    args: dict = {}
    for aid, val in zip(arg_ids or [], arg_values or []):
        if isinstance(aid, dict) and aid.get("op") == op:
            args[aid.get("key")] = val

    df = get_df_from_store(prep_store, ds_name)
    if df is None:
        df = get_df_from_store(ds_store, ds_name)
    if df is None:
        return alert_banner("Датасет не найден.", level="danger"), no_update, no_update

    before_rows = len(df)
    before_nulls = int(df.isnull().sum().sum())
    operation = ""
    detail = ""

    def _warn():
        return (
            alert_banner("Выберите параметры операции.", level="warning"),
            no_update, no_update,
        )

    try:
        if op == "impute":
            imp_cols = args.get("cols") or []
            imp_method = args.get("method") or "median"
            if not imp_cols:
                return _warn()
            for col in imp_cols:
                if col in df.columns:
                    df = impute_missing(df, col, method=imp_method)
            operation = "Заполнение пропусков"
            detail = f"{len(imp_cols)} столбцов · метод = {imp_method}"

        elif op == "outliers":
            out_cols = args.get("cols") or []
            out_method = args.get("method") or "iqr"
            out_thresh = args.get("threshold")
            if not out_cols:
                return _warn()
            for col in out_cols:
                if col in df.columns:
                    df = remove_outliers(df, col, method=out_method,
                                         threshold=float(out_thresh or 1.5))
            operation = "Удаление выбросов"
            detail = f"{len(out_cols)} столбцов · {out_method} > {out_thresh}"

        elif op == "dedup":
            dedup_cols = args.get("cols")
            subset = dedup_cols if dedup_cols else None
            df = deduplicate(df, subset=subset)
            operation = "Дедупликация"
            detail = (f"по {len(subset)} столбцам" if subset else "по всем столбцам")

        elif op == "dates":
            date_col = args.get("col")
            if not date_col:
                return _warn()
            df = parse_dates(df, date_col)
            operation = "Парсинг дат"
            detail = f"{date_col} · формат auto"

        elif op == "resample":
            resample_datecol = args.get("datecol")
            resample_freq = args.get("freq")
            resample_valuecols = args.get("valuecols")
            if not resample_datecol or not resample_freq:
                return _warn()
            value_cols = resample_valuecols or df.select_dtypes(
                include="number").columns.tolist()
            df = resample_timeseries(df, resample_datecol, resample_freq, value_cols)
            operation = "Ресэмплинг"
            detail = f"{resample_datecol} · {resample_freq} · {len(value_cols)} метрик"

        elif op == "lags":
            lag_col = args.get("col")
            lag_periods = args.get("periods")
            if not lag_col:
                return _warn()
            periods = [int(lag_periods)] if lag_periods else [1]
            df = add_lags(df, lag_col, periods)
            operation = "Лаги"
            detail = f"{lag_col} · лаги {periods}"

        elif op == "rolling":
            roll_col = args.get("col")
            roll_window = args.get("window")
            if not roll_col:
                return _warn()
            window = int(roll_window or 7)
            df = add_rolling(df, roll_col, window)
            operation = "Скользящее среднее"
            detail = f"{roll_col} · окно {window}"

        elif op == "normalize":
            norm_cols = args.get("cols") or []
            norm_method = args.get("method") or "minmax"
            if not norm_cols:
                return _warn()
            df = normalize(df, norm_cols, method=norm_method)
            operation = "Нормализация"
            detail = f"{len(norm_cols)} столбцов · {norm_method}"

        elif op == "buckets":
            bucket_col = args.get("col")
            bucket_n = args.get("n")
            if not bucket_col:
                return _warn()
            n_bins = int(bucket_n or 5)
            df = add_buckets(df, bucket_col, n_bins=n_bins)
            operation = "Бинирование"
            detail = f"{bucket_col} · {n_bins} бинов"

        else:
            return _warn()

    except Exception as exc:
        return (
            alert_banner(f"Ошибка: {exc}", level="danger"),
            no_update, no_update,
        )

    # Persist the prepared snapshot
    path = save_dataframe(df, f"{ds_name}_prepared")
    prep_store[ds_name] = path

    after_rows = len(df)
    after_nulls = int(df.isnull().sum().sum())

    # Append to history
    tab_for_op = _OP_TO_TAB.get(op, "tab-prep-features")
    history.append({
        "tab":    tab_for_op,
        "op":     operation,
        "detail": detail,
        "time":   pd.Timestamp.now().strftime("%H:%M"),
        "rows":   after_rows,
        "nulls":  after_nulls,
    })

    # Result card — before/after KPIs + preview table
    def mini(label: str, value: str, tone: str = "") -> html.Div:
        val_cls = "kb-stat-value" + (f" {tone}" if tone else "")
        return html.Div(
            [
                html.Div(label, className="kb-stat-label"),
                html.Div(value, className=val_cls),
            ],
            className="kb-stat-card",
        )

    result_card = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3(f"Применено: {operation}"),
                            html.Div(detail or "—", className="caption"),
                        ],
                        className="kb-card-title",
                    ),
                    chip("готово", variant="success"),
                ],
                className="kb-card-head",
            ),
            html.Div(
                [
                    mini("Строк (до)",      f"{before_rows:,}".replace(",", " ")),
                    mini("Строк (после)",   f"{after_rows:,}".replace(",", " "),
                         "kb-stat-value--ok" if after_rows != before_rows else ""),
                    mini("Пропусков (до)",  f"{before_nulls:,}".replace(",", " "),
                         "kb-stat-value--warn" if before_nulls else ""),
                    mini("Пропусков (после)", f"{after_nulls:,}".replace(",", " "),
                         "kb-stat-value--ok" if after_nulls < before_nulls else ""),
                ],
                className="kb-prep-result-stats",
            ),
            data_table(df.head(10), id="prep-result-tbl", page_size=10),
        ],
        className="kb-card kb-prep-result",
    )

    return result_card, prep_store, history

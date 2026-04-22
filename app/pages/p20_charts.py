"""p20_charts – Chart constructor (Dash).

Design port: hero + 2-col split. LEFT — configuration card with a 6-category
tile picker for 18 chart types, data-field selectors that react to the picked
chart type, and a display block (title / palette / options / top-N / height).
RIGHT — canvas card with suggestion chip, full-size ``dcc.Graph``, auto-insight
list and a PNG/HTML download button.

Chart logic is unchanged from the previous version — only the UI shell is
reshaped, ``ch-type`` is replaced by a ``dcc.Store`` backing a tile grid, and
the buggy ``df = get_df(prep) or get_df(ds)`` pattern is fixed via ``_load_df``.
"""
from __future__ import annotations

import base64
from typing import Any

import dash
from dash import (
    ALL, Input, Output, State, callback, ctx, dcc, html, no_update,
)
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from app.components.alerts import alert_banner
from app.components.cards import chip
from app.components.icons import icon
from app.components.layout import empty_state
from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_ACTIVE_DS, STORE_DATASET, STORE_PREPARED,
    get_df_from_store, list_datasets,
)

dash.register_page(
    __name__,
    path="/charts",
    name="20. Графики",
    order=20,
    icon="bar-chart",
)


# ---------------------------------------------------------------------------
# Chart type catalog
# ---------------------------------------------------------------------------
# ``needs`` — space-separated tokens controlling which data fields are shown:
#   "x"   — X required
#   "y"   — Y required
#   "col" — Color/group
#   "sz"  — Size
# ``opts`` — tokens enabling display options:
#   "sort" · "topn" · "logy" · "labels"
# ---------------------------------------------------------------------------
CHART_TYPES: list[dict[str, str]] = [
    dict(key="bar",         label="Столбчатый",        cat="compare",   needs="x y col", opts="sort topn logy labels"),
    dict(key="bar_h",       label="Горизонтальный",    cat="compare",   needs="x y col", opts="sort topn logy labels"),
    dict(key="pie",         label="Круговой",          cat="compare",   needs="x y",     opts="topn labels"),
    dict(key="donut",       label="Кольцевой",         cat="compare",   needs="x y",     opts="topn labels"),
    dict(key="funnel",      label="Воронка",           cat="compare",   needs="x y",     opts="sort topn"),

    dict(key="line",        label="Линейный",          cat="time",      needs="x y col", opts="logy"),
    dict(key="area",        label="Площадной",         cat="time",      needs="x y col", opts="logy"),
    dict(key="candlestick", label="Свечи (OHLC)",      cat="time",      needs="x y",     opts=""),
    dict(key="bar_race",    label="Гонка столбцов",    cat="time",      needs="x y col", opts="logy"),

    dict(key="histogram",   label="Гистограмма",       cat="dist",      needs="x",       opts="logy"),
    dict(key="box",         label="Ящик с усами",      cat="dist",      needs="y col",   opts="logy"),
    dict(key="violin",      label="Скрипичный",        cat="dist",      needs="y col",   opts="logy"),

    dict(key="scatter",     label="Точечный",          cat="relation",  needs="x y col", opts="logy"),
    dict(key="bubble",      label="Пузырьковый",       cat="relation",  needs="x y col sz", opts="logy"),
    dict(key="heatmap",     label="Тепловая карта",    cat="relation",  needs="",        opts=""),

    dict(key="treemap",     label="Дерево",            cat="hier",      needs="x y col", opts="topn"),
    dict(key="sunburst",    label="Солнечный",         cat="hier",      needs="x y col", opts="topn"),

    dict(key="dual_axis",   label="Двойная ось",       cat="combo",     needs="x y col", opts=""),
]

CATEGORIES: list[tuple[str, str, str]] = [
    ("compare",  "Сравнение",     "bar-chart"),
    ("time",     "Время",         "trend"),
    ("dist",     "Распределение", "chart"),
    ("relation", "Связи",         "target"),
    ("hier",     "Иерархия",      "layers"),
    ("combo",    "Комбо",         "columns"),
]

_BY_KEY = {ct["key"]: ct for ct in CHART_TYPES}

COLOR_SCHEMES: dict[str, Any] = {
    "По умолчанию": None,
    "Синий":        px.colors.sequential.Blues,
    "Красный":      px.colors.sequential.Reds,
    "Зелёный":      px.colors.sequential.Greens,
    "Пастельный":   px.colors.qualitative.Pastel,
}

OPTION_LABELS: list[tuple[str, str]] = [
    ("legend", "Легенда"),
    ("labels", "Подписи"),
    ("sort",   "Сортировать"),
    ("logy",   "Лог. шкала Y"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_df(prepared, datasets, ds):
    """Prefer prepared, fall back to raw. Safe against the ``bool(df)`` trap."""
    if not ds:
        return None
    df = get_df_from_store(prepared, ds)
    if df is None:
        df = get_df_from_store(datasets, ds)
    return df


def _get_insights(chart_key: str, df: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    try:
        if chart_key in ("bar", "bar_h") and y_col and y_col in df.columns and x_col in df.columns:
            grp = df.groupby(x_col, observed=True)[y_col].sum().sort_values(ascending=False)
            if grp.empty:
                return []
            top_label = str(grp.index[0])
            top_val = grp.iloc[0]
            last_val = grp.iloc[-1]
            gap = round((top_val - last_val) / top_val * 100, 1) if top_val != 0 else 0
            return [f"Лидер: {top_label} ({top_val:,.1f}). Отставание последнего: {gap}%"]
        elif chart_key in ("line", "area") and y_col and y_col in df.columns:
            s = df[y_col].dropna()
            if len(s) < 2:
                return []
            first, last = s.iloc[0], s.iloc[-1]
            pct = round((last - first) / first * 100, 1) if first != 0 else 0
            trend = "растущий" if pct > 2 else "падающий" if pct < -2 else "нейтральный"
            return [f"Тренд: {trend}. {first:,.1f} → {last:,.1f} ({pct:+.1f}%)"]
        elif chart_key == "scatter" and x_col and y_col:
            sub = df[[x_col, y_col]].dropna()
            if len(sub) < 3:
                return []
            r = sub[x_col].corr(sub[y_col])
            strength = "сильная" if abs(r) >= 0.7 else "умеренная" if abs(r) >= 0.4 else "слабая"
            return [f"Корреляция: {r:.2f} ({strength})"]
        elif chart_key == "histogram" and x_col and x_col in df.columns:
            s = df[x_col].dropna()
            skew = s.skew()
            dist = "нормальное" if abs(skew) < 0.5 else "перекос вправо" if skew > 0 else "перекос влево"
            return [f"Среднее: {s.mean():,.2f}. Медиана: {s.median():,.2f}. Распределение: {dist}"]
        elif chart_key in ("pie", "donut") and x_col and y_col:
            grp = df.groupby(x_col, observed=True)[y_col].sum()
            total = grp.sum()
            if total == 0:
                return []
            top_label = str(grp.idxmax())
            top_pct = grp.max() / total * 100
            return [f"Топ: {top_label} = {top_pct:.1f}%. Категорий: {len(grp)}"]
    except Exception:
        pass
    return []


def _suggest_chart(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return ""
    try:
        x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
        y_is_num = pd.api.types.is_numeric_dtype(df[y_col])
        x_nunique = df[x_col].nunique()
        if x_is_date and y_is_num:
            return "Рекомендуем «Линейный» — идеально для временных рядов"
        if x_is_num and y_is_num:
            return "Рекомендуем «Точечный» — две числовые оси"
        if not x_is_num and y_is_num and x_nunique <= 10:
            return "Рекомендуем «Круговой» — мало категорий"
        if not x_is_num and y_is_num:
            return "Рекомендуем «Столбчатый»"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------
def _hero() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Инструмент · Визуализация", className="kb-overline"),
                    html.H1("20. Конструктор графиков",
                            className="kb-h1 kb-charts-hero__title"),
                    html.Div(
                        "18 типов графиков без кода — с автовыводами и экспортом",
                        className="kb-body-l kb-charts-hero__sub",
                    ),
                ],
                className="kb-charts-hero__left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("refresh", 12), html.Span("Сбросить")],
                        id="ch-btn-reset",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("play", 12), html.Span("Построить")],
                        id="ch-btn-build",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-charts-hero__right",
            ),
        ],
        className="kb-card kb-charts-hero",
    )


def _tile_grid(active_key: str) -> list[html.Div]:
    """Render 6 category groups, each with a pill-tile grid."""
    groups = []
    for cat_key, cat_label, cat_icon in CATEGORIES:
        tiles = []
        for ct in CHART_TYPES:
            if ct["cat"] != cat_key:
                continue
            is_on = ct["key"] == active_key
            tiles.append(
                html.Button(
                    ct["label"],
                    id={"type": "ch-tile", "kind": ct["key"]},
                    className="kb-ch-tile" + (" is-on" if is_on else ""),
                    n_clicks=0,
                )
            )
        groups.append(
            html.Div(
                [
                    html.Div(
                        [icon(cat_icon, 12), html.Span(cat_label)],
                        className="kb-ch-tile-group__head",
                    ),
                    html.Div(tiles, className="kb-ch-tile-grid"),
                ],
                className="kb-ch-tile-group",
            )
        )
    return groups


def _field(label: str, inner: Any, wrap_id: str | None = None, hidden: bool = False) -> html.Div:
    kwargs: dict = {"className": "kb-field"}
    if wrap_id:
        kwargs["id"] = wrap_id
    if hidden:
        kwargs["style"] = {"display": "none"}
    return html.Div([html.Div(label, className="kb-field-label"), inner], **kwargs)


def _config_card() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Настройка графика"),
                            html.Div("Выберите тип и укажите данные",
                                     className="caption"),
                        ],
                        className="kb-card-title",
                    ),
                    chip("Конфиг", variant="neutral"),
                ],
                className="kb-card-head",
            ),

            # — Dataset —
            _field(
                "Датасет",
                dcc.Dropdown(
                    id="ch-ds-select",
                    className="kb-select",
                    placeholder="Выберите датасет",
                    clearable=False,
                ),
            ),

            # — Chart type picker —
            html.Div(
                [
                    html.Div("Тип графика", className="kb-field-label"),
                    html.Div(id="ch-tile-wrap",
                             className="kb-ch-tile-wrap",
                             children=_tile_grid("bar")),
                ],
                className="kb-field",
            ),

            # — Data fields (conditionally hidden by callback) —
            html.Div(
                [
                    html.Div("Данные", className="kb-overline kb-ch-section"),
                    _field("Ось X",
                           dcc.Dropdown(id="ch-x", className="kb-select",
                                        placeholder="Колонка X"),
                           wrap_id="ch-field-x"),
                    _field("Ось Y",
                           dcc.Dropdown(id="ch-y", className="kb-select",
                                        placeholder="Колонка Y"),
                           wrap_id="ch-field-y"),
                    _field("Цвет / группа",
                           dcc.Dropdown(id="ch-color", className="kb-select",
                                        placeholder="(не разбивать)"),
                           wrap_id="ch-field-color"),
                    _field("Размер пузырька",
                           dcc.Dropdown(id="ch-size", className="kb-select",
                                        placeholder="(равный)"),
                           wrap_id="ch-field-size", hidden=True),
                ],
                className="kb-ch-group",
            ),

            # — Display —
            html.Div(
                [
                    html.Div("Оформление", className="kb-overline kb-ch-section"),
                    _field("Заголовок",
                           dcc.Input(id="ch-title", className="kb-input",
                                     placeholder="Заголовок графика",
                                     debounce=True, value="")),
                    _field("Палитра",
                           dcc.Dropdown(
                               id="ch-scheme",
                               className="kb-select",
                               options=[{"label": k, "value": k} for k in COLOR_SCHEMES],
                               value="По умолчанию",
                               clearable=False,
                           )),
                    # — options pills —
                    html.Div(
                        [
                            html.Div("Опции", className="kb-field-label"),
                            html.Div(id="ch-opts-wrap",
                                     className="kb-ch-opts"),
                        ],
                        className="kb-field",
                    ),
                    html.Div(
                        [
                            _field(
                                "Топ N",
                                dcc.Input(id="ch-topn", type="number", min=0,
                                          step=1, value=0,
                                          className="kb-input"),
                                wrap_id="ch-field-topn",
                            ),
                            _field(
                                "Высота",
                                dcc.Input(id="ch-height", type="number",
                                          min=200, max=1200, step=50,
                                          value=500, className="kb-input"),
                            ),
                        ],
                        className="kb-ch-row-2",
                    ),
                ],
                className="kb-ch-group",
            ),
        ],
        className="kb-card kb-charts-config",
    )


def _canvas_card() -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Предпросмотр"),
                            html.Div("Нажмите «Построить» после настройки",
                                     className="caption"),
                        ],
                        className="kb-card-title",
                    ),
                    html.Div(id="ch-dl-slot", className="kb-card-head-actions"),
                ],
                className="kb-card-head",
            ),
            html.Div(id="ch-suggestion", className="kb-ch-suggestion"),
            html.Div(id="ch-alert"),
            dcc.Loading(
                type="circle",
                color="#21A066",
                children=[
                    html.Div(id="ch-canvas-body",
                             className="kb-ch-canvas-body",
                             children=empty_state(
                                 "", "График не построен",
                                 "Выберите тип, укажите данные и нажмите «Построить»",
                             )),
                ],
            ),
            html.Div(id="ch-insights", className="kb-ch-insights"),
        ],
        className="kb-card kb-charts-canvas",
    )


layout = html.Div(
    [
        dcc.Store(id="ch-type-store", data="bar"),
        _hero(),
        html.Div(
            [_config_card(), _canvas_card()],
            className="kb-charts-grid",
        ),
    ],
    className="kb-page kb-page--charts",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

# --- Dataset select — populate from stores -----------------------------------
@callback(
    Output("ch-ds-select", "options"),
    Output("ch-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_ACTIVE_DS, "data"),
    State("ch-ds-select", "value"),
)
def _ch_populate_ds(ds_data, active_ds, current):
    names = list_datasets(ds_data)
    opts = [{"label": n, "value": n} for n in names]
    if current in names:
        val = current
    elif active_ds in names:
        val = active_ds
    elif names:
        val = names[0]
    else:
        val = None
    return opts, val


# --- Columns dropdowns — populate based on chosen dataset ---------------------
@callback(
    Output("ch-x", "options"),
    Output("ch-y", "options"),
    Output("ch-color", "options"),
    Output("ch-size", "options"),
    Input("ch-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _ch_populate_cols(ds, ds_data, prep_data):
    df = _load_df(prep_data, ds_data, ds)
    if df is None:
        return [], [], [], []
    num_df = df.select_dtypes(include="number")
    all_opts = [{"label": c, "value": c} for c in df.columns]
    num_opts = [{"label": c, "value": c} for c in num_df.columns]
    none_plus = [{"label": "(нет)", "value": ""}] + all_opts
    none_plus_num = [{"label": "(нет)", "value": ""}] + num_opts
    return all_opts, all_opts, none_plus, none_plus_num


# --- Tile click → chart type store + tile active state ------------------------
@callback(
    Output("ch-type-store", "data"),
    Output("ch-tile-wrap", "children"),
    Input({"type": "ch-tile", "kind": ALL}, "n_clicks"),
    State("ch-type-store", "data"),
    prevent_initial_call=True,
)
def _ch_tile_click(_clicks, current):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or trig.get("type") != "ch-tile":
        return no_update, no_update
    key = trig["kind"]
    if key == current:
        return no_update, no_update
    return key, _tile_grid(key)


# --- Conditional visibility of data fields + options grid ---------------------
@callback(
    Output("ch-field-x", "style"),
    Output("ch-field-y", "style"),
    Output("ch-field-color", "style"),
    Output("ch-field-size", "style"),
    Output("ch-field-topn", "style"),
    Output("ch-opts-wrap", "children"),
    Input("ch-type-store", "data"),
    State({"type": "ch-opt", "key": ALL}, "id"),
    State({"type": "ch-opt", "key": ALL}, "data-on"),
)
def _ch_fields_visibility(chart_key, existing_ids, existing_on):
    ct = _BY_KEY.get(chart_key, _BY_KEY["bar"])
    needs = set(ct["needs"].split())
    opts = set(ct["opts"].split())

    def sty(show: bool) -> dict:
        return {} if show else {"display": "none"}

    # preserve previously-toggled options
    prev_on: set[str] = set()
    if existing_ids and existing_on:
        for id_obj, is_on in zip(existing_ids, existing_on):
            if is_on:
                prev_on.add(id_obj.get("key"))
    # new chart may not support previously-toggled opts → drop them
    prev_on &= (opts | {"legend"})  # "legend" is universal
    # sensible default: legend on, nothing else
    if not existing_ids:
        prev_on = {"legend"}

    tiles = []
    for key, label in OPTION_LABELS:
        supported = (key == "legend") or (key in opts)
        is_on = key in prev_on
        tiles.append(
            html.Button(
                label,
                id={"type": "ch-opt", "key": key},
                className=(
                    "kb-chip kb-chip--toggle"
                    + (" is-on" if is_on else "")
                    + ("" if supported else " is-off")
                ),
                disabled=not supported,
                n_clicks=0,
                **{"data-on": is_on},
            )
        )

    return (
        sty("x" in needs),
        sty("y" in needs),
        sty("col" in needs),
        sty("sz" in needs),
        sty("topn" in opts),
        tiles,
    )


# --- Option chip toggle -------------------------------------------------------
@callback(
    Output({"type": "ch-opt", "key": ALL}, "className"),
    Output({"type": "ch-opt", "key": ALL}, "data-on"),
    Input({"type": "ch-opt", "key": ALL}, "n_clicks"),
    State({"type": "ch-opt", "key": ALL}, "id"),
    State({"type": "ch-opt", "key": ALL}, "data-on"),
    State({"type": "ch-opt", "key": ALL}, "disabled"),
    prevent_initial_call=True,
)
def _ch_opt_toggle(_clicks, ids, states, disabled):
    trig = ctx.triggered_id
    new_states = list(states)
    classes = []
    for i, (id_obj, is_on, is_dis) in enumerate(zip(ids, states, disabled)):
        if trig and isinstance(trig, dict) and id_obj == trig and not is_dis:
            is_on = not is_on
            new_states[i] = is_on
        classes.append(
            "kb-chip kb-chip--toggle"
            + (" is-on" if is_on else "")
            + (" is-off" if is_dis else "")
        )
    return classes, new_states


# --- Suggestion chip ---------------------------------------------------------
@callback(
    Output("ch-suggestion", "children"),
    Input("ch-x", "value"),
    Input("ch-y", "value"),
    Input("ch-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _ch_show_suggestion(x_col, y_col, ds, ds_data, prep_data):
    df = _load_df(prep_data, ds_data, ds)
    if df is None:
        return ""
    text = _suggest_chart(df, x_col or "", y_col or "")
    if not text:
        return ""
    return chip(text, variant="info")


# --- Reset button ------------------------------------------------------------
@callback(
    Output("ch-x", "value"),
    Output("ch-y", "value"),
    Output("ch-color", "value"),
    Output("ch-size", "value"),
    Output("ch-title", "value"),
    Output("ch-topn", "value"),
    Output("ch-height", "value"),
    Output("ch-scheme", "value"),
    Input("ch-btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def _ch_reset(_n):
    return None, None, "", "", "", 0, 500, "По умолчанию"


# --- Build chart -------------------------------------------------------------
@callback(
    Output("ch-canvas-body", "children"),
    Output("ch-insights", "children"),
    Output("ch-alert", "children"),
    Output("ch-dl-slot", "children"),
    Input("ch-btn-build", "n_clicks"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State("ch-ds-select", "value"),
    State("ch-type-store", "data"),
    State("ch-x", "value"),
    State("ch-y", "value"),
    State("ch-color", "value"),
    State("ch-size", "value"),
    State("ch-title", "value"),
    State("ch-scheme", "value"),
    State({"type": "ch-opt", "key": ALL}, "id"),
    State({"type": "ch-opt", "key": ALL}, "data-on"),
    State("ch-topn", "value"),
    State("ch-height", "value"),
    prevent_initial_call=True,
)
def _ch_build(n_clicks, ds_data, prep_data, active_ds, chart_type,
              x_col, y_col, color_col, size_col, title, scheme,
              opt_ids, opt_states, top_n, height):
    def _empty(msg="График не построен", sub=""):
        return empty_state("", msg, sub)

    if not active_ds:
        return _empty(), "", alert_banner("Нет данных — выберите датасет.", "warning"), ""

    df = _load_df(prep_data, ds_data, active_ds)
    if df is None or df.empty:
        return _empty(), "", alert_banner("Датасет не найден.", "danger"), ""

    # --- decode option chip states ---
    on_set: set[str] = set()
    if opt_ids and opt_states:
        for id_obj, is_on in zip(opt_ids, opt_states):
            if is_on:
                on_set.add(id_obj.get("key"))
    show_legend = "legend" in on_set
    show_labels = "labels" in on_set
    sort_vals   = "sort" in on_set
    log_y       = "logy" in on_set

    color_col = color_col or None
    size_col  = size_col or None
    top_n     = int(top_n) if top_n else 0
    height    = int(height) if height else 500
    title     = title or ""

    palette = COLOR_SCHEMES.get(scheme)
    disc_kwargs = {"color_discrete_sequence": palette} if palette else {}

    try:
        fig = None

        if chart_type == "bar":
            plot_df = df.sort_values(y_col, ascending=False) if sort_vals and y_col in df.columns else df
            if top_n > 0:
                plot_df = plot_df.head(top_n)
            fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "bar_h":
            plot_df = df.sort_values(y_col, ascending=False) if sort_vals and y_col in df.columns else df
            if top_n > 0:
                plot_df = plot_df.head(top_n)
            fig = px.bar(plot_df, x=y_col, y=x_col, color=color_col, orientation="h", **disc_kwargs)

        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "area":
            fig = px.area(df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, nbins=30, **disc_kwargs)

        elif chart_type == "box":
            fig = px.box(df, x=color_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "violin":
            fig = px.violin(df, x=color_col, y=y_col, color=color_col, box=True, **disc_kwargs)

        elif chart_type == "pie":
            plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
            if top_n > 0:
                plot_df = plot_df.nlargest(top_n, y_col)
            fig = px.pie(plot_df, names=x_col, values=y_col, **disc_kwargs)

        elif chart_type == "donut":
            plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
            if top_n > 0:
                plot_df = plot_df.nlargest(top_n, y_col)
            fig = px.pie(plot_df, names=x_col, values=y_col, hole=0.45, **disc_kwargs)

        elif chart_type == "treemap":
            path_cols = [color_col, x_col] if color_col else [x_col]
            fig = px.treemap(df, path=path_cols, values=y_col, **disc_kwargs)

        elif chart_type == "sunburst":
            path_cols = [color_col, x_col] if color_col else [x_col]
            fig = px.sunburst(df, path=path_cols, values=y_col, **disc_kwargs)

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                return _empty(), "", alert_banner(
                    "Нужно минимум 2 числовых столбца.", "warning"), ""
            corr = num_df.corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
            ))

        elif chart_type == "bubble":
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                             size_max=60, **disc_kwargs)

        elif chart_type == "funnel":
            plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
            if sort_vals:
                plot_df = plot_df.sort_values(y_col, ascending=False)
            if top_n > 0:
                plot_df = plot_df.head(top_n)
            fig = go.Figure(go.Funnel(
                y=plot_df[x_col].astype(str).tolist(),
                x=plot_df[y_col].tolist(),
                textinfo="value+percent total",
            ))

        elif chart_type == "dual_axis":
            if not y_col or not color_col:
                return _empty(), "", alert_banner(
                    "Для двойной оси укажите Y (столбцы) и Цвет (вторая ось).",
                    "warning"), ""
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=y_col),
                          secondary_y=False)
            fig.add_trace(
                go.Scatter(x=df[x_col], y=df[color_col], name=color_col,
                           mode="lines+markers"),
                secondary_y=True,
            )
            fig.update_yaxes(title_text=y_col, secondary_y=False)
            fig.update_yaxes(title_text=color_col, secondary_y=True)

        elif chart_type == "candlestick":
            return _empty(), "", alert_banner(
                "Для свечного графика используйте 4 числовых столбца (Open / "
                "High / Low / Close) и столбец даты. Укажите X=дата, Y=Open.",
                "info"), ""

        elif chart_type == "bar_race":
            if not color_col:
                return _empty(), "", alert_banner(
                    "Для анимации укажите «Цвет» как кадр анимации.",
                    "warning"), ""
            fig = px.bar(
                df.sort_values(y_col, ascending=False), x=x_col, y=y_col,
                animation_frame=color_col,
                range_y=[0, df[y_col].max() * 1.1],
            )

        if fig is None:
            return _empty(), "", alert_banner("Неизвестный тип графика.", "danger"), ""

        # --- layout ---
        fig.update_layout(
            height=height,
            showlegend=show_legend,
            title={"text": title, "font": {"size": 16}} if title else None,
        )
        if log_y:
            fig.update_yaxes(type="log")
        if show_labels and chart_type in ("bar", "bar_h"):
            fig.update_traces(texttemplate="%{value:,.0f}", textposition="outside")
        elif show_labels and chart_type in ("pie", "donut"):
            fig.update_traces(textinfo="label+percent")

        apply_kibad_theme(fig)

        # --- insights ---
        lines = _get_insights(chart_type, df, x_col or "", y_col or "")
        insights_div = ""
        if lines:
            insights_div = html.Div(
                [
                    html.Div("Автовыводы", className="kb-overline"),
                    html.Ul([html.Li(l) for l in lines],
                            className="kb-ch-insight-list"),
                ],
            )

        # --- download button ---
        dl_btn: Any = ""
        try:
            img_bytes = fig.to_image(format="png")
            b64 = base64.b64encode(img_bytes).decode()
            dl_btn = html.A(
                html.Button(
                    [icon("download", 12), html.Span("PNG")],
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0,
                ),
                href=f"data:image/png;base64,{b64}",
                download="chart.png",
            )
        except Exception:
            html_str = fig.to_html()
            b64 = base64.b64encode(html_str.encode()).decode()
            dl_btn = html.A(
                html.Button(
                    [icon("download", 12), html.Span("HTML")],
                    className="kb-btn kb-btn--ghost kb-btn--sm",
                    n_clicks=0,
                ),
                href=f"data:text/html;base64,{b64}",
                download="chart.html",
            )

        graph = dcc.Graph(figure=fig, className="kb-ch-graph",
                          config={"displaylogo": False})

        return graph, insights_div, "", dl_btn

    except Exception as exc:
        return _empty(), "", alert_banner(f"Ошибка построения: {exc}", "danger"), ""

"""
p07_timeseries.py -- Временные ряды: прогнозирование, ACF/PACF, аномалии (Dash).

Методы: Наивный, ARX, SARIMAX, STL декомпозиция, диагностика резидуалов.
"""
from __future__ import annotations

import logging
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_ACTIVE_DS, STORE_PREPARED, STORE_FORECAST,
    get_df_from_store, list_datasets,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.form import select_input, slider_input, number_input, text_input
from app.components.alerts import alert_banner

from core.models import (
    run_naive_forecast, run_arx_forecast, run_sarimax_forecast,
    rolling_backtest, detect_anomalies, ForecastResult,
    run_stl_decomposition, compute_residual_diagnostics,
)

dash.register_page(
    __name__,
    path="/timeseries",
    name="7. Временные ряды",
    order=7,
    icon="graph-up",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_df(ds_name, raw, prep):
    if not ds_name:
        return None
    df = get_df_from_store(prep, ds_name) if prep and ds_name in (prep or {}) else get_df_from_store(raw, ds_name)
    return df


def _plot_forecast(result: ForecastResult, target_col: str, title: str) -> go.Figure:
    fd = result.forecast_df
    hist = fd[fd["actual"].notna()].copy()
    fut = fd[fd["actual"].isna() & fd["forecast"].notna()].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["actual"],
        mode="lines", name="Факт (история)",
        line=dict(color="#e4e7ee", width=2.5),
    ))
    if not hist["forecast"].dropna().empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["forecast"],
            mode="lines", name="Подгонка модели",
            line=dict(color="#4b9eff", width=1.5, dash="dot"), opacity=0.8,
        ))
    if not hist.empty and not fut.empty:
        boundary_x = hist["date"].iloc[-1]
        fig.add_shape(
            type="line",
            x0=boundary_x, x1=boundary_x,
            y0=0, y1=1, yref="paper",
            line=dict(dash="dash", color="#8891a5", width=1.5),
            opacity=0.6,
        )
        fig.add_annotation(
            x=boundary_x, y=1, yref="paper",
            text="Граница прогноза", showarrow=False,
            font=dict(size=11, color="#8891a5"),
            xanchor="left", yanchor="top",
        )
    if not fut.empty and "lower" in fut.columns and fut["lower"].notna().any():
        fig.add_trace(go.Scatter(
            x=pd.concat([fut["date"], fut["date"][::-1]]),
            y=pd.concat([fut["upper"], fut["lower"][::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% доверительный интервал", hoverinfo="skip",
        ))
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["forecast"],
            mode="lines+markers", name="Прогноз",
            line=dict(color="#ef4444", width=2.5), marker=dict(size=5),
        ))
    m = result.metrics
    metric_text = " | ".join([f"{k}: {v}" for k, v in m.items()])
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18, x=0),
        xaxis_title="Дата", yaxis_title=target_col,
        annotations=[dict(
            xref="paper", yref="paper", x=0.01, y=1.03,
            text=metric_text, showarrow=False,
            font=dict(size=11, color="#8891a5"),
        )],
        height=450,
    )
    return apply_kibad_theme(fig)


def _metric_cards(metrics: dict) -> list:
    return [stat_card(k, str(v)) for k, v in metrics.items()]


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_method_info = dbc.Accordion([
    dbc.AccordionItem([
        dbc.Row([
            dbc.Col([
                html.Strong("Наивный / Сезонный наивный", style={"color": "#e4e7ee"}),
                html.P("Последнее значение или значение прошлого сезона. "
                       "Базовая линия, горизонт 1–3 периода.",
                       style={"fontSize": "0.83rem", "color": "#8891a5", "marginBottom": 0}),
            ], md=4),
            dbc.Col([
                html.Strong("ARX (Ridge + лаги)", style={"color": "#e4e7ee"}),
                html.P("Авторегрессия + внешние факторы, Ridge-регуляризация. "
                       "Лучше при экзогенных факторах, горизонт 3–24.",
                       style={"fontSize": "0.83rem", "color": "#8891a5", "marginBottom": 0}),
            ], md=4),
            dbc.Col([
                html.Strong("SARIMAX", style={"color": "#e4e7ee"}),
                html.P("Сезонная ARIMA с доверительными интервалами. "
                       "Нужна чёткая сезонность, ≥ 50 наблюдений.",
                       style={"fontSize": "0.83rem", "color": "#8891a5", "marginBottom": 0}),
            ], md=4),
        ]),
    ], title="Справка по методам", item_id="methods-info"),
], start_collapsed=True, className="mb-3",
   style={"fontSize": "0.9rem"})


layout = dbc.Container([
    page_header("7. Временные ряды",
                "Прогнозирование и анализ динамики",
                "bi-graph-up"),

    _method_info,

    # Row 1: dataset + series config
    dbc.Row([
        dbc.Col(select_input("Датасет", "ts-ds-select", [],
                             placeholder="Выберите датасет..."), md=3),
        dbc.Col(select_input("Колонка даты", "ts-date-col", []), md=3),
        dbc.Col(select_input("Целевая переменная", "ts-target-col", []), md=3),
        dbc.Col(select_input("Внешние факторы (exog)", "ts-exog-cols", [],
                             multi=True), md=3),
    ], className="mb-2 align-items-end"),

    # Row 2: model hyperparams (Naive / ARX)
    dbc.Row([
        dbc.Col(slider_input("Горизонт прогноза", "ts-horizon", 1, 60, 12, 1), md=3),
        dbc.Col(slider_input("Сезонный период", "ts-period", 1, 52, 12, 1), md=3),
        dbc.Col(text_input("Лаги AR (через запятую)", "ts-lags",
                           value="1,2,3,12"), md=3),
        dbc.Col(slider_input("Регуляризация Ridge (alpha)", "ts-arx-alpha",
                             1, 100, 1, 1), md=3),
    ], className="mb-2"),

    # Row 3: SARIMAX params — always in DOM so all tabs can read them as State
    dbc.Accordion([
        dbc.AccordionItem([
            dbc.Row([
                dbc.Col([
                    html.P("Несезонная часть (p, d, q)",
                           style={"fontWeight": "bold", "color": "#e4e7ee", "marginBottom": "6px"}),
                    slider_input("p (AR)", "sarimax-p", 0, 5, 1, 1),
                    slider_input("d (дифф.)", "sarimax-d", 0, 2, 1, 1),
                    slider_input("q (MA)", "sarimax-q", 0, 5, 1, 1),
                ], md=6),
                dbc.Col([
                    html.P("Сезонная часть (P, D, Q)",
                           style={"fontWeight": "bold", "color": "#e4e7ee", "marginBottom": "6px"}),
                    slider_input("P", "sarimax-P", 0, 3, 1, 1),
                    slider_input("D", "sarimax-D", 0, 2, 0, 1),
                    slider_input("Q", "sarimax-Q", 0, 3, 1, 1),
                ], md=6),
            ]),
        ], title="Параметры SARIMAX (p,d,q)(P,D,Q) — используются во вкладках SARIMAX, Бэктест, Диагностика, Сравнение"),
    ], start_collapsed=True, className="mb-3"),

    html.Hr(className="mb-3"),

    # Tabs
    dbc.Tabs(id="ts-tabs", active_tab="tab-naive", children=[
        dbc.Tab(label="Наивный", tab_id="tab-naive"),
        dbc.Tab(label="ARX", tab_id="tab-arx"),
        dbc.Tab(label="SARIMAX", tab_id="tab-sarimax"),
        dbc.Tab(label="Бэктестинг", tab_id="tab-backtest"),
        dbc.Tab(label="ACF/PACF", tab_id="tab-acf"),
        dbc.Tab(label="Декомпозиция", tab_id="tab-stl"),
        dbc.Tab(label="Диагностика", tab_id="tab-diagnostics"),
        dbc.Tab(label="Сравнение", tab_id="tab-compare"),
        dbc.Tab(label="Аномалии", tab_id="tab-anomaly"),
    ]),

    html.Div(id="ts-tab-content", className="mt-3"),

], fluid=True, className="kb-page")


# ---------------------------------------------------------------------------
# Dataset & column dropdowns
# ---------------------------------------------------------------------------
@callback(
    Output("ts-ds-select", "options"),
    Output("ts-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def _ds_opts(raw, prep, active_ds):
    names = sorted(set(list_datasets(raw) + list_datasets(prep)))
    opts = [{"label": n, "value": n} for n in names]
    val = active_ds if active_ds in names else (names[0] if names else None)
    return opts, val


@callback(
    Output("ts-date-col", "options"),
    Output("ts-target-col", "options"),
    Output("ts-exog-cols", "options"),
    Input("ts-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _col_opts(ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return [], [], []
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])
               or (df[c].dtype == object and pd.to_datetime(df[c], errors="coerce").notna().mean() > 0.8)]
    num_cols = df.select_dtypes(include="number").columns.tolist()
    return (
        [{"label": c, "value": c} for c in dt_cols],
        [{"label": c, "value": c} for c in num_cols],
        [{"label": c, "value": c} for c in num_cols],
    )


# ---------------------------------------------------------------------------
# Tab content
# ---------------------------------------------------------------------------
@callback(
    Output("ts-tab-content", "children"),
    Input("ts-tabs", "active_tab"),
    Input("ts-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_tab(tab, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return empty_state("bi-graph-up", "Нет данных",
                           "Выберите датасет с временным рядом.")

    if tab == "tab-naive":
        return html.Div([
            section_header("Наивный / Сезонный наивный прогноз"),
            dbc.RadioItems(
                id="naive-type", className="mb-3",
                options=[
                    {"label": "Сезонный наивный", "value": "seasonal"},
                    {"label": "Наивный (последнее значение)", "value": "last"},
                ], value="seasonal", inline=True,
            ),
            dbc.Button("Запустить наивный прогноз", id="btn-naive", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="naive-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-arx":
        return html.Div([
            section_header("ARX — авторегрессия с внешними факторами (Ridge)"),
            html.P("Параметры модели задаются в панели выше (Лаги AR, Регуляризация Ridge).",
                   style={"color": "#8891a5", "fontSize": "0.85rem", "marginBottom": "12px"}),
            dbc.Button("Запустить ARX", id="btn-arx", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="arx-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-sarimax":
        return html.Div([
            section_header("SARIMAX -- сезонная ARIMA"),
            html.P("Параметры (p,d,q)(P,D,Q) задаются в аккордеоне «Параметры SARIMAX» выше.",
                   style={"color": "#8891a5", "fontSize": "0.85rem", "marginBottom": "12px"}),
            dbc.Button("Запустить SARIMAX", id="btn-sarimax", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="sarimax-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-backtest":
        return html.Div([
            section_header("Скользящее бэктестирование"),
            dbc.Row([
                dbc.Col(select_input("Модель", "bt-model",
                                     [{"label": "Сезонный наивный", "value": "naive"},
                                      {"label": "ARX", "value": "arx"},
                                      {"label": "SARIMAX", "value": "sarimax"}],
                                     value="naive"), md=3),
                dbc.Col(slider_input("Фолдов", "bt-folds", 2, 8, 3, 1), md=3),
                dbc.Col(slider_input("Мин. обуч. периодов", "bt-min-train", 12, 60, 24, 1), md=3),
                dbc.Col(slider_input("Горизонт фолда", "bt-horizon-fold", 1, 24, 6, 1), md=3),
            ]),
            dbc.Button("Запустить бэктест", id="btn-backtest", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="backtest-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-acf":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        return html.Div([
            section_header("ACF / PACF анализ"),
            dbc.Row([
                dbc.Col(select_input("Колонка", "acf-col", num_cols), md=4),
                dbc.Col(slider_input("Число лагов", "acf-nlags", 5, 60, 30, 1), md=4),
            ]),
            dbc.Button("Построить ACF/PACF", id="btn-acf", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="acf-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-stl":
        return html.Div([
            section_header("STL декомпозиция (Seasonal and Trend decomposition using Loess)"),
            dbc.Row([
                dbc.Col([
                    dbc.RadioItems(
                        id="stl-model-type",
                        options=[
                            {"label": "Аддитивная (Y = T + S + R)", "value": "additive"},
                            {"label": "Мультипликативная (Y = T × S × R, log-transform)", "value": "multiplicative"},
                        ],
                        value="additive", inline=True, className="mb-2",
                    ),
                    dbc.Checklist(
                        id="stl-robust",
                        options=[{"label": "Устойчивое STL (robust=True, рекомендуется)", "value": "robust"}],
                        value=["robust"], inline=True,
                    ),
                ], md=8),
            ], className="mb-3"),
            dbc.Button("Декомпозировать", id="btn-stl", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="stl-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-diagnostics":
        return html.Div([
            section_header("Диагностика резидуалов модели"),
            html.P("Выберите модель и параметры в панели выше, затем запустите диагностику.",
                   style={"color": "#8891a5", "fontSize": "0.85rem", "marginBottom": "12px"}),
            dbc.Row([
                dbc.Col(select_input("Модель для диагностики", "diag-model",
                                     [{"label": "Наивный (сезонный)", "value": "naive"},
                                      {"label": "ARX", "value": "arx"},
                                      {"label": "SARIMAX", "value": "sarimax"}],
                                     value="arx"), md=4),
            ], className="mb-3"),
            dbc.Button("Запустить диагностику", id="btn-diagnostics", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="diagnostics-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-compare":
        return html.Div([
            section_header("Сравнение моделей"),
            html.P("Запускает Наивный, ARX и SARIMAX с текущими параметрами. "
                   "Показывает метрики рядом и совмещённый прогноз.",
                   style={"color": "#8891a5", "fontSize": "0.85rem", "marginBottom": "12px"}),
            dbc.Button("Сравнить модели", id="btn-compare", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="compare-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-anomaly":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        return html.Div([
            section_header("Детекция аномалий"),
            dbc.Row([
                dbc.Col(select_input("Колонка", "anom-col", num_cols), md=3),
                dbc.Col([
                    html.Label("Метод", style={"color": "#e4e7ee", "fontSize": "0.85rem"}),
                    dbc.RadioItems(
                        id="anom-method",
                        options=[
                            {"label": "Rolling z-score", "value": "rolling_zscore"},
                            {"label": "STL резидуалы", "value": "stl_residual"},
                        ],
                        value="rolling_zscore", inline=True,
                    ),
                ], md=3),
                dbc.Col(slider_input("Размер окна", "anom-window", 5, 60, 12, 1), md=3),
                dbc.Col(slider_input("Порог z", "anom-thresh", 15, 50, 25, 1,
                                     marks={15: "1.5", 25: "2.5", 30: "3.0", 50: "5.0"}), md=3),
            ]),
            dbc.Button("Найти аномалии", id="btn-anomaly", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="anomaly-result"), type="circle", color="#10b981"),
        ])

    return html.Div()


# ---------------------------------------------------------------------------
# Naive forecast callback
# ---------------------------------------------------------------------------
@callback(
    Output("naive-result", "children"),
    Input("btn-naive", "n_clicks"),
    State("naive-type", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_naive(n, naive_type, ds_name, date_col, target_col, horizon, period, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        result = run_naive_forecast(
            df, date_col, target_col,
            horizon=int(horizon or 12),
            seasonal=(naive_type == "seasonal"),
            period=int(period or 12),
        )
        fig = _plot_forecast(result, target_col, f"Наивный прогноз: {target_col}")
        return html.Div([
            dbc.Row([dbc.Col(c) for c in _metric_cards(result.metrics)]),
            dcc.Graph(figure=fig),
            data_table(result.forecast_df.tail(30), "naive-fc-table", page_size=10),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# ARX forecast callback
# ---------------------------------------------------------------------------
@callback(
    Output("arx-result", "children"),
    Input("btn-arx", "n_clicks"),
    State("ts-arx-alpha", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-lags", "value"),
    State("ts-exog-cols", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_arx(n, alpha_r, ds_name, date_col, target_col, horizon, lags_str, exog_cols, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",") if x.strip().isdigit()]
        if not lags:
            return alert_banner("Укажите лаги (например: 1,2,3,12).", "warning")
        result = run_arx_forecast(
            df, date_col, target_col,
            exog_cols=exog_cols if exog_cols else None,
            lags=lags,
            horizon=int(horizon or 12),
            alpha=float(alpha_r or 1),
        )
        fig = _plot_forecast(result, target_col, f"ARX прогноз: {target_col}")
        children = [
            dbc.Row([dbc.Col(c) for c in _metric_cards(result.metrics)]),
        ]
        if exog_cols:
            children.append(dbc.Alert(
                "Будущие значения внешних факторов заданы как последнее наблюдение "
                "(константная экстраполяция). Прогноз может быть смещён, если факторы меняются.",
                color="info", className="mb-2",
            ))
        children.append(dcc.Graph(figure=fig))
        if result.explainability is not None:
            coef_df = result.explainability.copy()
            if "coefficient" in coef_df.columns and "feature" in coef_df.columns:
                coef_df["abs"] = coef_df["coefficient"].abs()
                coef_df = coef_df.sort_values("abs", ascending=False).drop("abs", axis=1)
                fig_c = px.bar(
                    coef_df.head(15), x="coefficient", y="feature", orientation="h",
                    title="Коэффициенты ARX (топ-15)",
                    color="coefficient", color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    labels={"coefficient": "Коэффициент", "feature": "Фактор"},
                )
                children.append(dcc.Graph(figure=apply_kibad_theme(fig_c)))
        children.append(data_table(result.forecast_df.tail(30), "arx-fc-table", page_size=10))
        return html.Div(children)
    except Exception as e:
        import traceback as _tb
        import logging as _log
        _log.getLogger("werkzeug").info("ARX_FULL_TRACEBACK: %s", _tb.format_exc())
        return alert_banner(f"Ошибка ARX: {e}", "danger")


# ---------------------------------------------------------------------------
# SARIMAX callback
# ---------------------------------------------------------------------------
@callback(
    Output("sarimax-result", "children"),
    Input("btn-sarimax", "n_clicks"),
    State("sarimax-p", "value"), State("sarimax-d", "value"), State("sarimax-q", "value"),
    State("sarimax-P", "value"), State("sarimax-D", "value"), State("sarimax-Q", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State("ts-exog-cols", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_sarimax(n, p, d, q, P, D, Q, ds_name, date_col, target_col, horizon, period, exog_cols, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        result = run_sarimax_forecast(
            df, date_col, target_col,
            exog_cols=exog_cols if exog_cols else None,
            order=(int(p or 1), int(d or 1), int(q or 1)),
            seasonal_order=(int(P or 1), int(D or 0), int(Q or 1), int(period or 12)),
            horizon=int(horizon or 12),
        )
        fig = _plot_forecast(result, target_col,
                             f"SARIMAX({p},{d},{q})({P},{D},{Q},{period}): {target_col}")
        children = [
            dbc.Row([dbc.Col(c) for c in _metric_cards(result.metrics)]),
        ]
        if exog_cols:
            children.append(dbc.Alert(
                "Будущие значения внешних факторов заданы как последнее наблюдение "
                "(константная экстраполяция). Прогноз может быть смещён, если факторы меняются.",
                color="info", className="mb-2",
            ))
        if result.notes:
            children.append(alert_banner(result.notes, "info"))
        children.append(dcc.Graph(figure=fig))
        if result.explainability is not None:
            children.append(data_table(result.explainability, "sarimax-params-table", page_size=15))
        children.append(data_table(result.forecast_df.tail(30), "sarimax-fc-table", page_size=10))
        return html.Div(children)
    except Exception as e:
        return alert_banner(f"Ошибка SARIMAX: {e}", "danger")


# ---------------------------------------------------------------------------
# Backtest callback
# ---------------------------------------------------------------------------
@callback(
    Output("backtest-result", "children"),
    Input("btn-backtest", "n_clicks"),
    State("bt-model", "value"), State("bt-folds", "value"),
    State("bt-min-train", "value"), State("bt-horizon-fold", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-period", "value"), State("ts-lags", "value"),
    State("ts-exog-cols", "value"),
    State("sarimax-p", "value"), State("sarimax-d", "value"), State("sarimax-q", "value"),
    State("sarimax-P", "value"), State("sarimax-D", "value"), State("sarimax-Q", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_backtest(n, model, folds, min_train, bt_h, ds_name, date_col, target_col,
                  period, lags_str, exog_cols,
                  p, d, q, P, D, Q,
                  raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",") if x.strip().isdigit()]
        if model == "naive":
            fn = run_naive_forecast
            kwargs = {"seasonal": True, "period": int(period or 12)}
        elif model == "arx":
            fn = run_arx_forecast
            kwargs = {"lags": lags, "exog_cols": exog_cols if exog_cols else None}
        else:  # sarimax
            fn = run_sarimax_forecast
            kwargs = {
                "order": (int(p or 1), int(d or 1), int(q or 1)),
                "seasonal_order": (int(P or 1), int(D or 0), int(Q or 1), int(period or 12)),
                "exog_cols": exog_cols if exog_cols else None,
            }

        fold_results, summary_df = rolling_backtest(
            df, date_col, target_col, model_fn=fn,
            n_folds=int(folds or 3), min_train=int(min_train or 24),
            horizon=int(bt_h or 6), **kwargs,
        )

        children = []
        if not summary_df.empty:
            metric_cols = [c for c in ["MAE", "RMSE", "MAPE", "sMAPE", "Bias"] if c in summary_df.columns]
            avg = summary_df[metric_cols].mean().round(4).to_dict()
            children.append(dbc.Row([dbc.Col(stat_card(k, str(v))) for k, v in avg.items()]))
            children.append(data_table(summary_df, "bt-summary-table", page_size=10))
        else:
            children.append(alert_banner("Недостаточно данных для бэктеста.", "warning"))
        return html.Div(children)
    except Exception as e:
        return alert_banner(f"Ошибка бэктеста: {e}", "danger")


# ---------------------------------------------------------------------------
# ACF/PACF callback  (BUG-3 fix: ADF stationarity warning)
# ---------------------------------------------------------------------------
@callback(
    Output("acf-result", "children"),
    Input("btn-acf", "n_clicks"),
    State("acf-col", "value"), State("acf-nlags", "value"),
    State("ts-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_acf(n, col, nlags, ds_name, raw, prep):
    if not col:
        return alert_banner("Выберите числовую колонку.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        from statsmodels.tsa.stattools import acf, pacf, adfuller
        series = df[col].dropna().values
        nlags_val = min(int(nlags or 30), len(series) // 2 - 1)

        # ADF stationarity test before ACF/PACF (BUG-3 fix)
        adf_stat, adf_pvalue, *_ = adfuller(series, autolag="AIC")
        stationarity_banner = None
        if adf_pvalue > 0.05:
            stationarity_banner = dbc.Alert(
                f"Ряд нестационарен (ADF p={adf_pvalue:.3f}). "
                "Рассмотрите дифференцирование (d=1) перед анализом ACF/PACF. "
                "Текущий ACF/PACF может давать неверные оценки порядков p и q.",
                color="warning", className="mb-3",
            )

        acf_vals = acf(series, nlags=nlags_val, fft=True)
        pacf_vals = pacf(series, nlags=nlags_val)
        ci_bound = 1.96 / np.sqrt(len(series))

        fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])

        for i, (vals, col_idx) in enumerate([(acf_vals, 1), (pacf_vals, 2)]):
            lags = list(range(len(vals)))
            fig.add_trace(go.Bar(x=lags, y=vals, marker_color="#4b9eff", name="ACF" if i == 0 else "PACF",
                                 showlegend=False), row=1, col=col_idx)
            fig.add_hline(y=ci_bound, line_dash="dash", line_color="#ef4444", row=1, col=col_idx)
            fig.add_hline(y=-ci_bound, line_dash="dash", line_color="#ef4444", row=1, col=col_idx)

        fig.update_layout(
            title=f"ACF / PACF: {col}  |  ADF p={adf_pvalue:.3f} ({'стационарен' if adf_pvalue <= 0.05 else 'нестационарен'})",
            height=400,
        )
        apply_kibad_theme(fig)

        children = []
        if stationarity_banner:
            children.append(stationarity_banner)
        children += [
            dcc.Graph(figure=fig),
            html.P(
                "Пунктирные линии -- 95% доверительные границы. "
                "Значимые лаги ACF указывают на порядок q (MA), "
                "значимые лаги PACF -- на порядок p (AR).",
                style={"color": "#8891a5", "fontSize": "0.85rem"},
            ),
        ]
        return html.Div(children)
    except ImportError:
        return alert_banner("statsmodels не установлен.", "warning")
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# STL Decomposition callback  (P1-A)
# ---------------------------------------------------------------------------
@callback(
    Output("stl-result", "children"),
    Input("btn-stl", "n_clicks"),
    State("stl-model-type", "value"),
    State("stl-robust", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-period", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_stl(n, model_type, robust_val, ds_name, date_col, target_col, period, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        robust = bool(robust_val)
        multiplicative = (model_type == "multiplicative")
        res = run_stl_decomposition(
            df, date_col, target_col,
            period=int(period or 12),
            robust=robust,
            multiplicative=multiplicative,
        )

        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=["Наблюдения", "Тренд", "Сезонность", "Остаток"],
            shared_xaxes=True,
            vertical_spacing=0.06,
        )
        x = res.dates
        traces = [
            (res.observed, "#e4e7ee", "Факт"),
            (res.trend, "#4b9eff", "Тренд"),
            (res.seasonal, "#10b981", "Сезонность"),
            (res.residual, "#f59e0b", "Остаток"),
        ]
        for row, (y_vals, color, name) in enumerate(traces, 1):
            fig.add_trace(go.Scatter(x=x, y=y_vals, mode="lines", name=name,
                                     line=dict(color=color, width=1.8)), row=row, col=1)

        model_label = "мультипликативная" if multiplicative else "аддитивная"
        fs = res.seasonality_strength
        fs_interp = "сильная (> 0.6)" if fs > 0.6 else ("слабая (< 0.3)" if fs < 0.3 else "умеренная")
        fig.update_layout(
            title=f"STL декомпозиция: {target_col} | Модель: {model_label} | Период: {period}",
            height=700, showlegend=False,
        )
        apply_kibad_theme(fig)

        strength_color = "success" if fs > 0.6 else ("warning" if fs > 0.3 else "secondary")
        return html.Div([
            dbc.Alert(
                f"Сила сезонности Fs = {fs:.3f} — {fs_interp}. "
                f"Fs > 0.6 — STL эффективен, Fs < 0.3 — сезонность слабая.",
                color=strength_color, className="mb-2",
            ),
            dcc.Graph(figure=fig),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка STL: {e}", "danger")


# ---------------------------------------------------------------------------
# Residual Diagnostics callback  (P1-B)
# ---------------------------------------------------------------------------
@callback(
    Output("diagnostics-result", "children"),
    Input("btn-diagnostics", "n_clicks"),
    State("diag-model", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State("ts-lags", "value"), State("ts-arx-alpha", "value"),
    State("ts-exog-cols", "value"),
    State("sarimax-p", "value"), State("sarimax-d", "value"), State("sarimax-q", "value"),
    State("sarimax-P", "value"), State("sarimax-D", "value"), State("sarimax-Q", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_diagnostics(n, diag_model, ds_name, date_col, target_col,
                     horizon, period, lags_str, alpha_r, exog_cols,
                     p, d, q, P, D, Q, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        from scipy import stats as _scipy_stats

        if diag_model == "naive":
            result = run_naive_forecast(df, date_col, target_col,
                                        horizon=int(horizon or 12), seasonal=True,
                                        period=int(period or 12))
        elif diag_model == "arx":
            lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",") if x.strip().isdigit()] or [1, 2, 3, 12]
            result = run_arx_forecast(df, date_col, target_col,
                                      exog_cols=exog_cols if exog_cols else None,
                                      lags=lags, horizon=int(horizon or 12),
                                      alpha=float(alpha_r or 1))
        else:  # sarimax
            result = run_sarimax_forecast(df, date_col, target_col,
                                          exog_cols=exog_cols if exog_cols else None,
                                          order=(int(p or 1), int(d or 1), int(q or 1)),
                                          seasonal_order=(int(P or 1), int(D or 0), int(Q or 1), int(period or 12)),
                                          horizon=int(horizon or 12))

        diag = compute_residual_diagnostics(result)
        residuals = diag.residuals
        fitted_vals = diag.fitted
        lb = diag.ljung_box
        acf_vals = diag.acf_residuals
        ci = diag.ci_bound

        # Ljung-Box summary
        lb_rows = [{"Лаг": int(row.name), "LB-статистика": round(row["lb_stat"], 3),
                    "p-value": round(row["lb_pvalue"], 4),
                    "Белый шум": "✓" if row["lb_pvalue"] > 0.05 else "✗"}
                   for _, row in lb.iterrows()]
        lb_pass = all(r["p-value"] > 0.05 for r in lb_rows)
        lb_banner_color = "success" if lb_pass else "warning"
        lb_banner_text = ("Тест Льюнга–Бокса: резидуалы — белый шум (p > 0.05 для всех лагов). "
                          "Модель адекватна." if lb_pass else
                          "Тест Льюнга–Бокса: обнаружена автокорреляция резидуалов (p ≤ 0.05). "
                          "Рассмотрите увеличение p или q.")

        # 4-panel diagnostics chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "ACF резидуалов", "Q-Q график (нормальность)",
                "Резидуалы по времени", "Резидуалы vs Подгонка",
            ],
            vertical_spacing=0.15, horizontal_spacing=0.1,
        )

        # ACF of residuals
        lags_acf = list(range(len(acf_vals)))
        fig.add_trace(go.Bar(x=lags_acf, y=acf_vals, marker_color="#4b9eff",
                             name="ACF резид.", showlegend=False), row=1, col=1)
        fig.add_hline(y=ci, line_dash="dash", line_color="#ef4444", row=1, col=1)
        fig.add_hline(y=-ci, line_dash="dash", line_color="#ef4444", row=1, col=1)

        # Q-Q plot
        (osm, osr), (slope, intercept, _) = _scipy_stats.probplot(residuals)
        qq_line_y = slope * np.array([osm[0], osm[-1]]) + intercept
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                  marker=dict(color="#10b981", size=4),
                                  name="Q-Q", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[osm[0], osm[-1]], y=qq_line_y, mode="lines",
                                  line=dict(color="#ef4444", dash="dash"),
                                  showlegend=False), row=1, col=2)

        # Residuals over time
        fig.add_trace(go.Scatter(x=list(range(len(residuals))), y=residuals, mode="lines",
                                  line=dict(color="#f59e0b", width=1.2),
                                  name="Резидуалы", showlegend=False), row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#8891a5", row=2, col=1)

        # Residuals vs fitted
        fig.add_trace(go.Scatter(x=fitted_vals, y=residuals, mode="markers",
                                  marker=dict(color="#a78bfa", size=4, opacity=0.7),
                                  name="Резид. vs Подгонка", showlegend=False), row=2, col=2)
        fig.add_hline(y=0, line_dash="dot", line_color="#8891a5", row=2, col=2)

        fig.update_layout(title=f"Диагностика резидуалов: {result.model_name}", height=600)
        apply_kibad_theme(fig)

        lb_df = pd.DataFrame(lb_rows)
        return html.Div([
            dbc.Alert(lb_banner_text, color=lb_banner_color, className="mb-2"),
            data_table(lb_df, "lb-table", page_size=5),
            dcc.Graph(figure=fig),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка диагностики: {e}", "danger")


# ---------------------------------------------------------------------------
# Model Comparison callback  (P2-C)
# ---------------------------------------------------------------------------
@callback(
    Output("compare-result", "children"),
    Input("btn-compare", "n_clicks"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-target-col", "value"),
    State("ts-horizon", "value"), State("ts-period", "value"),
    State("ts-lags", "value"), State("ts-arx-alpha", "value"),
    State("ts-exog-cols", "value"),
    State("sarimax-p", "value"), State("sarimax-d", "value"), State("sarimax-q", "value"),
    State("sarimax-P", "value"), State("sarimax-D", "value"), State("sarimax-Q", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_compare(n, ds_name, date_col, target_col,
                 horizon, period, lags_str, alpha_r, exog_cols,
                 p, d, q, P, D, Q, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")

    lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",") if x.strip().isdigit()] or [1, 2, 3, 12]
    results = {}
    errors = []

    for model_name, run_fn, kwargs in [
        ("Наивный", run_naive_forecast,
         {"seasonal": True, "period": int(period or 12)}),
        ("ARX", run_arx_forecast,
         {"lags": lags, "exog_cols": exog_cols if exog_cols else None,
          "alpha": float(alpha_r or 1)}),
        ("SARIMAX", run_sarimax_forecast,
         {"order": (int(p or 1), int(d or 1), int(q or 1)),
          "seasonal_order": (int(P or 1), int(D or 0), int(Q or 1), int(period or 12)),
          "exog_cols": exog_cols if exog_cols else None}),
    ]:
        try:
            results[model_name] = run_fn(df, date_col, target_col,
                                         horizon=int(horizon or 12), **kwargs)
        except Exception as e:
            errors.append(f"{model_name}: {e}")

    if not results:
        return alert_banner("Все модели завершились с ошибкой: " + "; ".join(errors), "danger")

    # Metrics comparison table
    metrics_rows = []
    all_metric_keys = []
    for mn, res in results.items():
        row = {"Модель": mn}
        row.update(res.metrics)
        metrics_rows.append(row)
        all_metric_keys += list(res.metrics.keys())
    metric_keys = list(dict.fromkeys(all_metric_keys))  # unique, ordered
    metrics_df = pd.DataFrame(metrics_rows)

    # Highlight best per metric (lower = better for MAE/RMSE/MAPE/sMAPE; Bias closest to 0)
    best_info = []
    for mk in metric_keys:
        if mk not in metrics_df.columns:
            continue
        col_vals = metrics_df[mk]
        if mk == "Bias":
            best_idx = col_vals.abs().idxmin()
        else:
            best_idx = col_vals.idxmin()
        best_info.append(f"{mk}: **{metrics_df.loc[best_idx, 'Модель']}**")

    # Overlay forecast chart
    colors = {"Наивный": "#f59e0b", "ARX": "#4b9eff", "SARIMAX": "#ef4444"}
    fig = go.Figure()

    # Historical actual from first available result
    first_res = next(iter(results.values()))
    hist = first_res.forecast_df[first_res.forecast_df["actual"].notna()]
    fig.add_trace(go.Scatter(x=hist["date"], y=hist["actual"],
                              mode="lines", name="Факт",
                              line=dict(color="#e4e7ee", width=2)))

    for mn, res in results.items():
        fut = res.forecast_df[res.forecast_df["actual"].isna() & res.forecast_df["forecast"].notna()]
        color = colors.get(mn, "#8891a5")
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["forecast"],
                                  mode="lines+markers", name=mn,
                                  line=dict(color=color, width=2), marker=dict(size=5)))

    fig.update_layout(
        title=f"Сравнение прогнозов: {target_col}",
        hovermode="x unified", height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    apply_kibad_theme(fig)

    children = [
        html.H6("Лучшая модель по метрикам:", style={"color": "#e4e7ee", "marginBottom": "6px"}),
        html.P(" | ".join(best_info), style={"color": "#10b981", "fontSize": "0.9rem", "marginBottom": "12px"}),
        data_table(metrics_df, "compare-metrics-table", page_size=5),
        dcc.Graph(figure=fig),
    ]
    if errors:
        children.insert(0, alert_banner("Ошибки: " + "; ".join(errors), "warning"))

    return html.Div(children)


# ---------------------------------------------------------------------------
# Anomaly detection callback  (P3-A severity, P3-B STL method)
# ---------------------------------------------------------------------------
@callback(
    Output("anomaly-result", "children"),
    Input("btn-anomaly", "n_clicks"),
    State("anom-col", "value"), State("anom-window", "value"),
    State("anom-thresh", "value"), State("anom-method", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"), State("ts-period", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_anomaly(n, col, window, thresh_tick, method, ds_name, date_col, period, raw, prep):
    if not col:
        return alert_banner("Выберите числовую колонку.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        threshold = (thresh_tick or 25) / 10.0  # slider 15..50 -> 1.5..5.0
        anom_df = detect_anomalies(df[col].dropna(), method=method,
                                   window=int(window or 12), threshold=threshold,
                                   period=int(period or 12))
        anomalies = anom_df[anom_df["is_anomaly"]] if "is_anomaly" in anom_df.columns else pd.DataFrame()

        # Determine x-axis: use date column if available
        if date_col and date_col in df.columns:
            x_full = pd.to_datetime(df[date_col].dropna().iloc[:len(anom_df)]).values
        else:
            x_full = list(range(len(anom_df)))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_full,
            y=anom_df["value"] if "value" in anom_df.columns else anom_df.iloc[:, 0],
            mode="lines", name=col, line=dict(color="#e4e7ee"),
        ))

        # CI band
        if "upper" in anom_df.columns and "lower" in anom_df.columns:
            fig.add_trace(go.Scatter(
                x=list(x_full) + list(x_full[::-1]),
                y=list(anom_df["upper"]) + list(anom_df["lower"][::-1]),
                fill="toself", fillcolor="rgba(239,68,68,0.07)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"±{threshold}σ", hoverinfo="skip",
            ))

        if not anomalies.empty:
            anom_idx = anomalies.index.tolist()
            anom_x = [x_full[i] for i in anom_idx if i < len(x_full)]
            anom_vals = anomalies["value"].values if "value" in anomalies.columns else anomalies.iloc[:, 0].values
            fig.add_trace(go.Scatter(
                x=anom_x, y=anom_vals,
                mode="markers", name="Аномалии",
                marker=dict(color="#ef4444", size=10, symbol="x"),
            ))

        method_label = "Rolling z-score" if method == "rolling_zscore" else "STL резидуалы"
        fig.update_layout(
            title=f"Аномалии: {col} | Метод: {method_label} | Порог z={threshold:.1f}",
            height=420,
        )
        apply_kibad_theme(fig)

        # Anomaly table with severity sorted desc
        display_anomalies = anomalies.copy()
        if "severity" in display_anomalies.columns:
            display_anomalies = display_anomalies.sort_values("severity", ascending=False)
        display_cols = [c for c in ["value", "z_score", "severity", "is_anomaly"] if c in display_anomalies.columns]

        return html.Div([
            stat_card("Найдено аномалий", len(anomalies)),
            dcc.Graph(figure=fig),
            data_table(display_anomalies[display_cols].head(50), "anomaly-table", page_size=10)
            if not anomalies.empty else html.Div(),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")

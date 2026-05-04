"""
p07_timeseries.py -- Временные ряды: прогнозирование, ACF/PACF, аномалии (Dash).

Методы: Наивный, ARX, SARIMAX (+ Prophet в try/except).
"""
from __future__ import annotations

import json
import logging
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

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
        # Use add_shape + add_annotation instead of add_vline to avoid
        # Plotly's internal shapeannotation._mean(sum([Timestamp])) which
        # fails with pandas 2.x
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
    cards = []
    for k, v in metrics.items():
        cards.append(stat_card(k, str(v)))
    return cards


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

    # Row 2: model hyperparams
    dbc.Row([
        dbc.Col(slider_input("Горизонт прогноза", "ts-horizon", 1, 60, 12, 1), md=3),
        dbc.Col(slider_input("Сезонный период", "ts-period", 1, 52, 12, 1), md=3),
        dbc.Col(text_input("Лаги AR (через запятую)", "ts-lags",
                           value="1,2,3,12"), md=3),
        dbc.Col(slider_input("Регуляризация Ridge (alpha)", "ts-arx-alpha",
                             1, 100, 1, 1), md=3),
    ], className="mb-3"),

    html.Hr(className="mb-3"),

    # Tabs
    dbc.Tabs(id="ts-tabs", active_tab="tab-naive", children=[
        dbc.Tab(label="Наивный", tab_id="tab-naive"),
        dbc.Tab(label="ARX", tab_id="tab-arx"),
        dbc.Tab(label="SARIMAX", tab_id="tab-sarimax"),
        dbc.Tab(label="Бэктестинг", tab_id="tab-backtest"),
        dbc.Tab(label="ACF/PACF", tab_id="tab-acf"),
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
            dbc.Row([
                dbc.Col([
                    html.P("Несезонная часть (p, d, q)", style={"fontWeight": "bold", "color": "#e4e7ee"}),
                    slider_input("p (AR)", "sarimax-p", 0, 5, 1, 1),
                    slider_input("d (дифф.)", "sarimax-d", 0, 2, 1, 1),
                    slider_input("q (MA)", "sarimax-q", 0, 5, 1, 1),
                ], md=6),
                dbc.Col([
                    html.P("Сезонная часть (P, D, Q)", style={"fontWeight": "bold", "color": "#e4e7ee"}),
                    slider_input("P", "sarimax-P", 0, 3, 1, 1),
                    slider_input("D", "sarimax-D", 0, 2, 0, 1),
                    slider_input("Q", "sarimax-Q", 0, 3, 1, 1),
                ], md=6),
            ]),
            dbc.Button("Запустить SARIMAX", id="btn-sarimax", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="sarimax-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-backtest":
        return html.Div([
            section_header("Скользящее бэктестирование"),
            dbc.Row([
                dbc.Col(select_input("Модель", "bt-model",
                                     [{"label": "Сезонный наивный", "value": "naive"},
                                      {"label": "ARX", "value": "arx"}],
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

    if tab == "tab-anomaly":
        num_cols = df.select_dtypes(include="number").columns.tolist()
        return html.Div([
            section_header("Детекция аномалий (rolling z-score)"),
            dbc.Row([
                dbc.Col(select_input("Колонка", "anom-col", num_cols), md=3),
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
            dcc.Graph(figure=fig),
        ]
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
        tb_str = _tb.format_exc()
        # Log full traceback as an INFO message visible in server output
        import logging as _log
        _log.getLogger("werkzeug").info("ARX_FULL_TRACEBACK: %s", tb_str)
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
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_backtest(n, model, folds, min_train, bt_h, ds_name, date_col, target_col,
                  period, lags_str, exog_cols, raw, prep):
    if not all([date_col, target_col]):
        return alert_banner("Выберите колонку даты и целевую переменную.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        lags = [int(x.strip()) for x in (lags_str or "1,2,3,12").split(",") if x.strip().isdigit()]
        fn = run_naive_forecast if model == "naive" else run_arx_forecast
        kwargs = ({"seasonal": True, "period": int(period or 12)}
                  if model == "naive"
                  else {"lags": lags, "exog_cols": exog_cols if exog_cols else None})

        fold_results, summary_df = rolling_backtest(
            df, date_col, target_col, model_fn=fn,
            n_folds=int(folds or 3), min_train=int(min_train or 24),
            horizon=int(bt_h or 6), **kwargs,
        )

        children = []
        if not summary_df.empty:
            avg = summary_df[["MAE", "RMSE", "MAPE", "Bias"]].mean().round(4).to_dict()
            children.append(dbc.Row([dbc.Col(stat_card(k, str(v))) for k, v in avg.items()]))
            children.append(data_table(summary_df, "bt-summary-table", page_size=10))
        else:
            children.append(alert_banner("Недостаточно данных для бэктеста.", "warning"))
        return html.Div(children)
    except Exception as e:
        return alert_banner(f"Ошибка бэктеста: {e}", "danger")


# ---------------------------------------------------------------------------
# ACF/PACF callback
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
        from statsmodels.tsa.stattools import acf, pacf
        series = df[col].dropna().values
        nlags_val = min(int(nlags or 30), len(series) // 2 - 1)
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

        fig.update_layout(title=f"ACF / PACF: {col}", height=400)
        apply_kibad_theme(fig)

        return html.Div([
            dcc.Graph(figure=fig),
            html.P(
                "Пунктирные линии -- 95% доверительные границы. "
                "Значимые лаги ACF указывают на порядок q (MA), "
                "значимые лаги PACF -- на порядок p (AR).",
                style={"color": "#8891a5", "fontSize": "0.85rem"},
            ),
        ])
    except ImportError:
        return alert_banner("statsmodels не установлен.", "warning")
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# Anomaly detection callback
# ---------------------------------------------------------------------------
@callback(
    Output("anomaly-result", "children"),
    Input("btn-anomaly", "n_clicks"),
    State("anom-col", "value"), State("anom-window", "value"),
    State("anom-thresh", "value"),
    State("ts-ds-select", "value"),
    State("ts-date-col", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_anomaly(n, col, window, thresh_tick, ds_name, date_col, raw, prep):
    if not col:
        return alert_banner("Выберите числовую колонку.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger")
    try:
        threshold = (thresh_tick or 25) / 10.0  # slider 15..50 -> 1.5..5.0
        anom_df = detect_anomalies(df[col].dropna(), window=int(window or 12),
                                   threshold=threshold)
        anomalies = anom_df[anom_df["is_anomaly"]] if "is_anomaly" in anom_df.columns else pd.DataFrame()

        fig = go.Figure()
        x_vals = list(range(len(anom_df)))
        fig.add_trace(go.Scatter(x=x_vals, y=anom_df["value"] if "value" in anom_df.columns else anom_df.iloc[:, 0],
                                 mode="lines", name=col, line=dict(color="#e4e7ee")))
        if not anomalies.empty:
            anom_idx = anomalies.index.tolist()
            anom_vals = anomalies["value"].values if "value" in anomalies.columns else anomalies.iloc[:, 0].values
            fig.add_trace(go.Scatter(x=anom_idx, y=anom_vals,
                                     mode="markers", name="Аномалии",
                                     marker=dict(color="#ef4444", size=10, symbol="x")))
        fig.update_layout(title=f"Аномалии: {col} (порог z={threshold:.1f})", height=400)
        apply_kibad_theme(fig)

        return html.Div([
            stat_card("Найдено аномалий", len(anomalies)),
            dcc.Graph(figure=fig),
            data_table(anomalies.head(50), "anomaly-table", page_size=10) if not anomalies.empty else html.Div(),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")

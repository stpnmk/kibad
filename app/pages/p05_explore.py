"""p05_explore – Exploratory analysis page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.cards import stat_card
from app.components.alerts import alert_banner
from core.explore import (
    plot_timeseries, plot_histogram, plot_boxplot, plot_violin,
    plot_correlation_heatmap,
    build_pivot, plot_pivot_bar, plot_waterfall, plot_stl_decomposition, compute_kpi,
)
from core.insights import analyze_dataset, format_insights_markdown, score_data_quality

dash.register_page(__name__, path="/explore", name="5. Исследование", order=5, icon="search")

layout = html.Div([
    page_header("5. Исследовательский анализ", "Распределения, корреляции, профилирование"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="exp-ds-select", placeholder="Выберите датасет..."),
        ], width=4),
    ], className="mb-3"),
    html.Div(id="exp-info", className="mb-3"),

    dbc.Tabs(id="exp-tabs", active_tab="tab-auto", children=[
        dbc.Tab(label="Авто-анализ", tab_id="tab-auto"),
        dbc.Tab(label="Временные ряды", tab_id="tab-ts"),
        dbc.Tab(label="Распределения", tab_id="tab-dist"),
        dbc.Tab(label="Корреляции", tab_id="tab-corr"),
        dbc.Tab(label="Попарные графики", tab_id="tab-pair"),
        dbc.Tab(label="KPI-трекер", tab_id="tab-kpi"),
        dbc.Tab(label="Профиль данных", tab_id="tab-profile"),
    ]),
    dcc.Loading(html.Div(id="exp-tab-content"), type="circle", color="#00c896"),
])


@callback(
    Output("exp-ds-select", "options"),
    Output("exp-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def update_ds(datasets, active):
    if not datasets:
        return [], None
    names = list(datasets.keys())
    val = active if active in names else (names[0] if names else None)
    return [{"label": n, "value": n} for n in names], val


@callback(
    Output("exp-info", "children"),
    Input("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def show_info(ds, datasets, prepared):
    if not ds:
        return empty_state("", "Данные не загружены", "Загрузите датасет на странице Данные")
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return ""
    return html.Div([
        stat_card("Строк", f"{df.shape[0]:,}"),
        stat_card("Столбцов", str(df.shape[1])),
        stat_card("Числовых", str(len(df.select_dtypes(include='number').columns))),
    ], className="kb-stats-grid")


@callback(
    Output("exp-tab-content", "children"),
    Input("exp-tabs", "active_tab"),
    Input("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_tab(tab, ds, datasets, prepared):
    if not ds:
        return empty_state("", "Выберите датасет", "")
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "warning")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    if tab == "tab-auto":
        try:
            insights = analyze_dataset(df)
            md = format_insights_markdown(insights)
            score = score_data_quality(df)
            return html.Div([
                section_header("Автоматический анализ"),
                html.Div([
                    stat_card("Качество данных", f"{score:.0f}/100"),
                ], className="kb-stats-grid"),
                dcc.Markdown(md, style={"color": "#8b92a8", "fontSize": "0.9rem"}),
            ])
        except Exception as e:
            return alert_banner(f"Ошибка авто-анализа: {e}", "warning")

    elif tab == "tab-ts":
        if not dt_cols or not num_cols:
            return alert_banner("Нужна хотя бы одна дата-колонка и числовая колонка.", "info")
        return html.Div([
            section_header("Временные ряды"),
            dbc.Row([
                dbc.Col([dcc.Dropdown(id="exp-ts-date", options=[{"label": c, "value": c} for c in dt_cols], value=dt_cols[0] if dt_cols else None, placeholder="Дата...")], width=4),
                dbc.Col([dcc.Dropdown(id="exp-ts-vals", options=[{"label": c, "value": c} for c in num_cols], value=num_cols[:2], multi=True, placeholder="Значения...")], width=8),
            ]),
            html.Div(id="exp-ts-chart"),
        ])

    elif tab == "tab-dist":
        if not num_cols:
            return alert_banner("Нет числовых колонок.", "info")
        figs = []
        for col in num_cols[:6]:
            fig = plot_histogram(df, col)
            apply_kibad_theme(fig)
            figs.append(dbc.Col(dcc.Graph(figure=fig), width=6))
        return html.Div([section_header("Распределения"), dbc.Row(figs)])

    elif tab == "tab-corr":
        if len(num_cols) < 2:
            return alert_banner("Нужно минимум 2 числовых колонки.", "info")
        fig = plot_correlation_heatmap(df, num_cols)
        apply_kibad_theme(fig)
        return html.Div([section_header("Матрица корреляций"), dcc.Graph(figure=fig)])

    elif tab == "tab-pair":
        if len(num_cols) < 2:
            return alert_banner("Нужно минимум 2 числовых колонки.", "info")
        cols = num_cols[:5]
        fig = px.scatter_matrix(df[cols], dimensions=cols, title="Попарные графики")
        apply_kibad_theme(fig)
        fig.update_layout(height=700)
        return html.Div([section_header("Попарные графики (Scatter Matrix)"), dcc.Graph(figure=fig)])

    elif tab == "tab-kpi":
        if not dt_cols or not num_cols:
            return alert_banner("Нужна дата-колонка и числовая колонка.", "info")
        try:
            kpis = compute_kpi(df, dt_cols[0], num_cols[0])
            cards = [stat_card(k, str(v)) for k, v in kpis.items()]
            return html.Div([section_header("KPI"), html.Div(cards, className="kb-stats-grid")])
        except Exception as e:
            return alert_banner(f"Ошибка KPI: {e}", "warning")

    elif tab == "tab-profile":
        desc = df.describe(include="all").T
        desc = desc.round(2)
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Колонка", "Пропуски"]
        miss["% пропусков"] = (miss["Пропуски"] / len(df) * 100).round(1)
        return html.Div([
            section_header("Профиль данных"),
            html.H4("Описательная статистика"),
            data_table(desc.reset_index().rename(columns={"index": "Колонка"}), id="exp-profile-desc"),
            html.H4("Пропуски", className="mt-3"),
            data_table(miss[miss["Пропуски"] > 0], id="exp-profile-miss"),
        ])

    return ""


@callback(
    Output("exp-ts-chart", "children"),
    Input("exp-ts-date", "value"),
    Input("exp-ts-vals", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def render_ts(date_col, val_cols, ds, datasets, prepared):
    if not date_col or not val_cols or not ds:
        return ""
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return ""
    try:
        fig = plot_timeseries(df, date_col, val_cols if isinstance(val_cols, list) else [val_cols])
        apply_kibad_theme(fig)
        return dcc.Graph(figure=fig)
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "warning")

"""p25_autoanalyst – One-click full analysis pipeline (Dash)."""
import logging

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

logger = logging.getLogger(__name__)

from app.state import (
    get_df_from_store, get_df_from_stores, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.cards import stat_card
from app.components.alerts import alert_banner
from core.insights import analyze_dataset, format_insights_markdown, score_data_quality
from core.explore import plot_correlation_heatmap, plot_histogram
from core.audit import log_event

dash.register_page(__name__, path="/auto", name="25. Автоанализ", order=25, icon="robot")

layout = html.Div([
    page_header("25. Автоматический анализ", "Полный анализ датасета в один клик"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="auto-ds", placeholder="Выберите датасет..."), width=4),
        dbc.Col(dbc.Button("Запустить полный анализ", id="auto-run-btn", color="primary"), width="auto"),
    ], className="mb-3"),
    dcc.Loading(html.Div(id="auto-results"), type="circle", color="#10b981"),
])


@callback(
    Output("auto-ds", "options"),
    Output("auto-ds", "value"),
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
    Output("auto-results", "children"),
    Input("auto-run-btn", "n_clicks"),
    State("auto-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_auto_analysis(n, ds, datasets, prepared):
    if not ds:
        return alert_banner("Выберите датасет.", "warning")

    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    log_event("auto_analysis", dataset=ds)
    sections = []

    # 1. Data profile
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    miss_pct = df.isnull().mean().mean() * 100

    sections.append(html.Div([
        section_header("Профиль данных"),
        html.Div([
            stat_card("Строк", f"{len(df):,}"),
            stat_card("Столбцов", str(df.shape[1])),
            stat_card("Числовых", str(len(num_cols))),
            stat_card("Текстовых", str(len(cat_cols))),
            stat_card("Дат", str(len(dt_cols))),
            stat_card("Пропусков", f"{miss_pct:.1f}%"),
        ], className="kb-stats-grid"),
    ]))

    # 2. Quality score
    try:
        score = score_data_quality(df)
        level = "success" if score >= 80 else ("warning" if score >= 50 else "danger")
        sections.append(alert_banner(f"Качество данных: {score:.0f}/100", level))
    except Exception:
        logger.exception("score_data_quality failed")

    # 3. Auto insights
    try:
        insights = analyze_dataset(df)
        md = format_insights_markdown(insights)
        sections.append(html.Div([
            section_header("Авто-инсайты"),
            dcc.Markdown(md, style={"color": "#9ba3b8"}),
        ]))
    except Exception as e:
        sections.append(alert_banner(f"Авто-инсайты недоступны: {e}", "info"))

    # 4. Distributions
    if num_cols:
        figs = []
        for col in num_cols[:4]:
            try:
                fig = plot_histogram(df, col)
                apply_kibad_theme(fig)
                figs.append(dbc.Col(dcc.Graph(figure=fig), width=6))
            except Exception:
                logger.warning("plot_histogram failed for column %s", col)
        if figs:
            sections.append(html.Div([section_header("Распределения"), dbc.Row(figs)]))

    # 5. Correlation
    if len(num_cols) >= 2:
        try:
            fig = plot_correlation_heatmap(df, num_cols[:10])
            apply_kibad_theme(fig)
            sections.append(html.Div([section_header("Корреляции"), dcc.Graph(figure=fig)]))
        except Exception:
            logger.exception("plot_correlation_heatmap failed")

    # 6. Descriptive stats
    try:
        desc = df.describe(include="all").round(2).reset_index()
        desc = desc.rename(columns={"index": "Статистика"})
        sections.append(html.Div([
            section_header("Описательная статистика"),
            data_table(desc, id="auto-desc-tbl"),
        ]))
    except Exception:
        logger.exception("describe stats failed")

    # 7. Missing values
    miss = df.isnull().sum()
    miss = miss[miss > 0].reset_index()
    miss.columns = ["Колонка", "Пропуски"]
    miss["% пропусков"] = (miss["Пропуски"] / len(df) * 100).round(1)
    if not miss.empty:
        sections.append(html.Div([
            section_header("Пропущенные значения"),
            data_table(miss, id="auto-miss-tbl"),
        ]))

    return html.Div(sections)

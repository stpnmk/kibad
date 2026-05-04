"""p18_weighted – Weighted Averages page (Dash).

Sections:
  1. Portfolio Weighted Averages  — wa, wstd, wmin, wmax per metric per group
  2. Mix/Rate Decomposition       — decompose change in WA rate between two periods
  3. Simplified Duration          — Macaulay / Modified / DV01 from WAR + WAM
"""
from __future__ import annotations

import logging

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update

from app.components.alerts import alert_banner
from app.components.cards import stat_card
from app.components.layout import empty_state, page_header, section_header
from app.components.table import data_table
from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_ACTIVE_DS,
    STORE_DATASET,
    STORE_PREPARED,
    get_df_from_stores,
)
from core.aggregate import to_csv_bytes, to_xlsx_bytes
from core.weighted_avg import (
    mix_rate_decomposition,
    portfolio_weighted_averages,
    simplified_duration,
    weighted_average,
    weighted_std,
)

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path="/weighted",
    name="Средневзвешенные",
    order=18,
    icon="percent",
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = html.Div([
    page_header("Средневзвешенные значения", "Портфельный анализ и декомпозиция"),

    # ── Dataset selector ──────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Label("Датасет", className="kb-stat-label"),
            dcc.Dropdown(id="wa-ds-select", placeholder="Выберите датасет..."),
        ], md=4),
    ], className="mb-3"),

    html.Div(id="wa-stats-row", className="kb-stats-grid mb-3"),

    dbc.Tabs(id="wa-tabs", active_tab="tab-portfolio", children=[

        # ── Tab 1: Portfolio Weighted Averages ────────────────────────────
        dbc.Tab(label="Портфельные СВ", tab_id="tab-portfolio", children=[
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Колонка весов", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-weight-col", placeholder="Вес (объём, сумма)..."),
                    ], md=3),
                    dbc.Col([
                        html.Label("Метрики (числовые)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-metric-cols", multi=True, placeholder="Ставка, доходность..."),
                    ], md=4),
                    dbc.Col([
                        html.Label("Группировка (опционально)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-group-cols", multi=True, placeholder="Сегмент, продукт..."),
                    ], md=3),
                    dbc.Col([
                        dbc.Button("Рассчитать", id="wa-run-btn", color="primary", className="mt-4"),
                    ], md=2),
                ]),
            ]), className="mb-3"),
            dcc.Loading(html.Div(id="wa-portfolio-result"), type="circle", color="#10b981"),
            dcc.Download(id="wa-download"),
        ]),

        # ── Tab 2: Mix/Rate Decomposition ─────────────────────────────────
        dbc.Tab(label="Микс/ставка (декомпозиция)", tab_id="tab-decomp", children=[
            dbc.Card(dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.Label("Датасет периода A (база)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-ds-a", placeholder="Период A..."),
                    ], md=3),
                    dbc.Col([
                        html.Label("Датасет периода B (факт)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-ds-b", placeholder="Период B..."),
                    ], md=3),
                    dbc.Col([
                        html.Label("Колонка весов", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-decomp-weight", placeholder="Вес..."),
                    ], md=2),
                    dbc.Col([
                        html.Label("Колонка ставки", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-decomp-rate", placeholder="Ставка..."),
                    ], md=2),
                    dbc.Col([
                        html.Label("Группа (категория)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-decomp-group", placeholder="Сегмент..."),
                    ], md=2),
                ]),
                dbc.Row([
                    dbc.Col(dbc.Button("Декомпозиция", id="wa-decomp-btn", color="primary", className="mt-3"), md=2),
                ]),
            ]), className="mb-3"),
            dcc.Loading(html.Div(id="wa-decomp-result"), type="circle", color="#10b981"),
        ]),

        # ── Tab 3: Simplified Duration ─────────────────────────────────────
        dbc.Tab(label="Дюрация (упрощённая)", tab_id="tab-duration", children=[
            dbc.Card(dbc.CardBody([
                html.P("Вычисляет дюрацию Маколея, модифицированную дюрацию и DV01 "
                       "из средневзвешенной ставки и срока.",
                       style={"color": "#8891a5", "fontSize": "0.875rem"}),
                dbc.Row([
                    dbc.Col([
                        html.Label("Колонка весов (объём)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-dur-weight", placeholder="Объём..."),
                    ], md=3),
                    dbc.Col([
                        html.Label("Колонка ставки (%)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-dur-rate", placeholder="Ставка..."),
                    ], md=3),
                    dbc.Col([
                        html.Label("Колонка срока (месяцы)", className="kb-stat-label"),
                        dcc.Dropdown(id="wa-dur-maturity", placeholder="Срок, мес..."),
                    ], md=3),
                    dbc.Col([
                        html.Label("Купонных периодов/год", className="kb-stat-label"),
                        dcc.Dropdown(
                            id="wa-dur-freq",
                            options=[
                                {"label": "12 (ежемесячно)", "value": 12},
                                {"label": "4 (ежеквартально)", "value": 4},
                                {"label": "2 (полугодово)", "value": 2},
                                {"label": "1 (ежегодно)", "value": 1},
                            ],
                            value=12,
                        ),
                    ], md=2),
                    dbc.Col([
                        dbc.Button("Рассчитать", id="wa-dur-btn", color="primary", className="mt-4"),
                    ], md=1),
                ]),
            ]), className="mb-3"),
            dcc.Loading(html.Div(id="wa-duration-result"), type="circle", color="#10b981"),
        ]),
    ]),
])


# ---------------------------------------------------------------------------
# Callbacks: dataset selector
# ---------------------------------------------------------------------------

@callback(
    Output("wa-ds-select", "options"),
    Output("wa-ds-select", "value"),
    Output("wa-ds-a", "options"),
    Output("wa-ds-b", "options"),
    Input(STORE_DATASET, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def _update_ds_list(datasets, active):
    if not datasets:
        return [], None, [], []
    names = list(datasets.keys())
    opts = [{"label": n, "value": n} for n in names]
    val = active if active in names else (names[0] if names else None)
    return opts, val, opts, opts


@callback(
    Output("wa-weight-col", "options"),
    Output("wa-metric-cols", "options"),
    Output("wa-group-cols", "options"),
    Output("wa-decomp-weight", "options"),
    Output("wa-decomp-rate", "options"),
    Output("wa-decomp-group", "options"),
    Output("wa-dur-weight", "options"),
    Output("wa-dur-rate", "options"),
    Output("wa-dur-maturity", "options"),
    Output("wa-stats-row", "children"),
    Input("wa-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _update_cols(ds_name, datasets, prepared):
    empty = [[], [], [], [], [], [], [], [], [], ""]
    if not ds_name:
        return empty
    df = get_df_from_stores(ds_name, datasets, prepared)
    if df is None:
        return empty
    num_cols = [{"label": c, "value": c} for c in df.select_dtypes(include="number").columns]
    all_cols = [{"label": c, "value": c} for c in df.columns]
    stats = [
        stat_card("Строк", f"{df.shape[0]:,}"),
        stat_card("Столбцов", str(df.shape[1])),
        stat_card("Числовых", str(len(num_cols))),
    ]
    return (
        num_cols, num_cols, all_cols,
        num_cols, num_cols, all_cols,
        num_cols, num_cols, num_cols,
        stats,
    )


# ---------------------------------------------------------------------------
# Tab 1: Portfolio Weighted Averages
# ---------------------------------------------------------------------------

@callback(
    Output("wa-portfolio-result", "children"),
    Input("wa-run-btn", "n_clicks"),
    State("wa-ds-select", "value"),
    State("wa-weight-col", "value"),
    State("wa-metric-cols", "value"),
    State("wa-group-cols", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_portfolio(n, ds_name, weight_col, metric_cols, group_cols, datasets, prepared):
    if not ds_name or not weight_col or not metric_cols:
        return alert_banner("Выберите датасет, колонку весов и метрики.", "warning")
    df = get_df_from_stores(ds_name, datasets, prepared)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")
    try:
        result = portfolio_weighted_averages(
            df,
            weight_col=weight_col,
            metric_cols=metric_cols,
            group_cols=group_cols or None,
        )
        if result.empty:
            return alert_banner("Нет данных для расчёта.", "warning")

        # Summary stat cards (total WA per metric)
        if not group_cols:
            cards = [stat_card(f"{m} (СВ)", f"{result[f'{m}_wa'].iloc[0]:.4f}" if f"{m}_wa" in result.columns else "—")
                     for m in metric_cols]
            weight_total = result[f"{weight_col}_sum"].iloc[0]
            cards.insert(0, stat_card(f"∑ {weight_col}", f"{weight_total:,.2f}"))
            stats_row = html.Div(cards, className="kb-stats-grid mb-3")
        else:
            stats_row = html.Div()

        # Bar charts for each metric's WA by group
        charts = []
        if group_cols and len(group_cols) == 1:
            for mc in metric_cols:
                wa_col = f"{mc}_wa"
                if wa_col not in result.columns:
                    continue
                fig = px.bar(
                    result, x=group_cols[0], y=wa_col,
                    error_y=f"{mc}_wstd" if f"{mc}_wstd" in result.columns else None,
                    title=f"Средневзвешенное: {mc}",
                    labels={wa_col: f"СВ {mc}", group_cols[0]: group_cols[0]},
                )
                apply_kibad_theme(fig)
                charts.append(dcc.Graph(figure=fig, style={"marginBottom": "16px"}))

        return html.Div([
            section_header("Результат"),
            stats_row,
            data_table(result, id="wa-result-tbl"),
            dbc.Row([
                dbc.Col(dbc.Button("⬇ CSV", id="wa-dl-csv-btn", color="secondary",
                                   size="sm", className="me-2"), width="auto"),
                dbc.Col(dbc.Button("⬇ Excel", id="wa-dl-xlsx-btn", color="secondary",
                                   size="sm"), width="auto"),
            ], className="mt-2 mb-3"),
            *charts,
        ])
    except Exception as e:
        logger.exception("portfolio_weighted_averages failed")
        return alert_banner(f"Ошибка расчёта: {e}", "danger")


@callback(
    Output("wa-download", "data"),
    Input("wa-dl-csv-btn", "n_clicks"),
    State("wa-ds-select", "value"),
    State("wa-weight-col", "value"),
    State("wa-metric-cols", "value"),
    State("wa-group-cols", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _dl_wa_csv(n, ds_name, weight_col, metric_cols, group_cols, datasets, prepared):
    if not ds_name or not weight_col or not metric_cols:
        return no_update
    df = get_df_from_stores(ds_name, datasets, prepared)
    if df is None:
        return no_update
    result = portfolio_weighted_averages(df, weight_col=weight_col,
                                         metric_cols=metric_cols, group_cols=group_cols or None)
    return dcc.send_bytes(to_csv_bytes(result), f"wa_{ds_name}.csv")


@callback(
    Output("wa-download", "data", allow_duplicate=True),
    Input("wa-dl-xlsx-btn", "n_clicks"),
    State("wa-ds-select", "value"),
    State("wa-weight-col", "value"),
    State("wa-metric-cols", "value"),
    State("wa-group-cols", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _dl_wa_xlsx(n, ds_name, weight_col, metric_cols, group_cols, datasets, prepared):
    if not ds_name or not weight_col or not metric_cols:
        return no_update
    df = get_df_from_stores(ds_name, datasets, prepared)
    if df is None:
        return no_update
    result = portfolio_weighted_averages(df, weight_col=weight_col,
                                         metric_cols=metric_cols, group_cols=group_cols or None)
    return dcc.send_bytes(to_xlsx_bytes(result), f"wa_{ds_name}.xlsx")


# ---------------------------------------------------------------------------
# Tab 2: Mix/Rate Decomposition
# ---------------------------------------------------------------------------

@callback(
    Output("wa-decomp-result", "children"),
    Input("wa-decomp-btn", "n_clicks"),
    State("wa-ds-a", "value"),
    State("wa-ds-b", "value"),
    State("wa-decomp-weight", "value"),
    State("wa-decomp-rate", "value"),
    State("wa-decomp-group", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_decomp(n, ds_a, ds_b, weight_col, rate_col, group_col, datasets, prepared):
    if not all([ds_a, ds_b, weight_col, rate_col, group_col]):
        return alert_banner("Заполните все поля для декомпозиции.", "warning")
    df_a = get_df_from_stores(ds_a, datasets, prepared)
    df_b = get_df_from_stores(ds_b, datasets, prepared)
    if df_a is None or df_b is None:
        return alert_banner("Не удалось загрузить датасеты.", "danger")
    try:
        result = mix_rate_decomposition(df_a, df_b, weight_col, rate_col, group_col)
        if result.empty:
            return alert_banner("Нет данных для декомпозиции.", "warning")

        # Summary totals
        total_mix = result["mix_effect"].sum()
        total_rate = result["rate_effect"].sum()
        total_total = result["total_effect"].sum()
        wa_a = weighted_average(
            pd.to_numeric(df_a[rate_col], errors="coerce"),
            pd.to_numeric(df_a[weight_col], errors="coerce").fillna(0),
        )
        wa_b = weighted_average(
            pd.to_numeric(df_b[rate_col], errors="coerce"),
            pd.to_numeric(df_b[weight_col], errors="coerce").fillna(0),
        )

        stats_row = html.Div([
            stat_card("СВ период A", f"{wa_a:.4f}"),
            stat_card("СВ период B", f"{wa_b:.4f}"),
            stat_card("Δ итого", f"{total_total:+.4f}"),
            stat_card("Эффект микса", f"{total_mix:+.4f}"),
            stat_card("Эффект ставки", f"{total_rate:+.4f}"),
        ], className="kb-stats-grid mb-3")

        # Waterfall-style bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Эффект микса",
            x=result["group"].astype(str),
            y=result["mix_effect"],
            marker_color="#4b9eff",
        ))
        fig.add_trace(go.Bar(
            name="Эффект ставки",
            x=result["group"].astype(str),
            y=result["rate_effect"],
            marker_color="#ef4444",
        ))
        fig.update_layout(
            barmode="stack",
            title="Декомпозиция изменения СВ: микс vs ставка",
            xaxis_title=group_col,
            yaxis_title="Эффект",
        )
        apply_kibad_theme(fig)

        return html.Div([
            section_header("Результат декомпозиции"),
            stats_row,
            data_table(result, id="wa-decomp-tbl"),
            dcc.Graph(figure=fig),
        ])
    except Exception as e:
        logger.exception("mix_rate_decomposition failed")
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# Tab 3: Simplified Duration
# ---------------------------------------------------------------------------

@callback(
    Output("wa-duration-result", "children"),
    Input("wa-dur-btn", "n_clicks"),
    State("wa-ds-select", "value"),
    State("wa-dur-weight", "value"),
    State("wa-dur-rate", "value"),
    State("wa-dur-maturity", "value"),
    State("wa-dur-freq", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_duration(n, ds_name, weight_col, rate_col, maturity_col, freq, datasets, prepared):
    if not all([ds_name, weight_col, rate_col, maturity_col]):
        return alert_banner("Выберите датасет, веса, ставку и срок.", "warning")
    df = get_df_from_stores(ds_name, datasets, prepared)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")
    try:
        w = pd.to_numeric(df[weight_col], errors="coerce").fillna(0)
        r = pd.to_numeric(df[rate_col], errors="coerce")
        m = pd.to_numeric(df[maturity_col], errors="coerce")

        war = weighted_average(r, w)   # Weighted Average Rate
        wam = weighted_average(m, w)   # Weighted Average Maturity (months)
        wstd_r = weighted_std(r, w)
        wstd_m = weighted_std(m, w)

        dur = simplified_duration(war, wam, coupon_freq_per_year=int(freq))

        # Stat cards
        stats_row = html.Div([
            stat_card("WAR (ставка, %)", f"{war:.4f}"),
            stat_card("WAM (срок, мес)", f"{wam:.2f}"),
            stat_card("Дюрация Маколея (лет)", f"{dur['macaulay_years']:.3f}"),
            stat_card("Мод. дюрация", f"{dur['modified_duration']:.3f}"),
            stat_card("DV01", f"{dur['dv01']:.6f}"),
        ], className="kb-stats-grid mb-3")

        # Distribution scatter (rate vs maturity, sized by weight)
        fig = px.scatter(
            df.assign(
                _rate=pd.to_numeric(df[rate_col], errors="coerce"),
                _mat=pd.to_numeric(df[maturity_col], errors="coerce"),
                _w=pd.to_numeric(df[weight_col], errors="coerce").fillna(0),
            ).dropna(subset=["_rate", "_mat"]),
            x="_mat", y="_rate", size="_w",
            labels={"_mat": f"{maturity_col} (мес)", "_rate": f"{rate_col} (%)", "_w": weight_col},
            title=f"Распределение: {rate_col} vs {maturity_col} (размер = {weight_col})",
            opacity=0.65,
        )
        # Add WAR and WAM lines
        fig.add_hline(y=war, line_dash="dash", line_color="#10b981",
                      annotation_text=f"WAR={war:.3f}%")
        fig.add_vline(x=wam, line_dash="dash", line_color="#f59e0b",
                      annotation_text=f"WAM={wam:.1f}м")
        apply_kibad_theme(fig)

        return html.Div([
            section_header("Результат дюрационного анализа"),
            stats_row,
            dbc.Row([
                dbc.Col([
                    dbc.Card(dbc.CardBody([
                        html.H6("Формулы", className="card-title"),
                        html.P(f"WAR = {war:.4f}%  (∑ w·r / ∑ w)", className="mb-1"),
                        html.P(f"WAM = {wam:.2f} мес", className="mb-1"),
                        html.P(f"Дюрация Маколея = WAM / 12 = {dur['macaulay_years']:.3f} лет", className="mb-1"),
                        html.P(f"Мод. дюрация = Маколей / (1 + WAR/{freq}) = {dur['modified_duration']:.3f}", className="mb-1"),
                        html.P(f"DV01 = −Мод. дюрация / 10000 = {dur['dv01']:.6f}", className="mb-0"),
                    ]))
                ], md=4),
                dbc.Col(dcc.Graph(figure=fig), md=8),
            ]),
        ])
    except Exception as e:
        logger.exception("simplified_duration failed")
        return alert_banner(f"Ошибка: {e}", "danger")

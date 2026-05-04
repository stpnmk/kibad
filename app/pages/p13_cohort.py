"""p13_cohort.py – Когортный анализ (Dash): удержание, отток, CLV."""
import logging

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
    get_df_from_store, get_df_from_stores, list_datasets,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.alerts import alert_banner
from core.cohort import (
    build_cohort_table, retention_table, churn_rate_table,
    average_retention_curve, compute_clv,
)

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/cohort", name="13. Когорты", order=13, icon="people")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div([
    page_header("13. Когортный анализ", "Удержание клиентов, отток и LTV по когортам"),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id="coh-ds", placeholder="Выберите датасет..."), md=4),
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("Настройка параметров"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("ID клиента", className="kb-label"),
                    dcc.Dropdown(id="coh-id-col", placeholder="Выберите колонку...", className="kb-select"),
                ], md=3),
                dbc.Col([
                    html.Label("Дата активности", className="kb-label"),
                    dcc.Dropdown(id="coh-act-col", placeholder="Выберите колонку...", className="kb-select"),
                ], md=3),
                dbc.Col([
                    html.Label("Дата привлечения (необязательно)", className="kb-label"),
                    dcc.Dropdown(id="coh-acq-col",
                                 options=[{"label": "(авто — первая активность)", "value": "__auto__"}],
                                 value="__auto__",
                                 className="kb-select"),
                ], md=3),
                dbc.Col([
                    html.Label("Период когорты", className="kb-label"),
                    dcc.Dropdown(id="coh-freq",
                                 options=[
                                     {"label": "Месяц", "value": "MS"},
                                     {"label": "Квартал", "value": "QS"},
                                 ],
                                 value="MS", clearable=False, className="kb-select"),
                ], md=1),
                dbc.Col([
                    html.Label("Макс. периодов", className="kb-label"),
                    dcc.Input(id="coh-max-offset", type="number", value=12, min=3, max=36,
                              style={"width": "100%"}),
                ], md=2),
            ], className="mb-3"),
            dbc.Button("▶ Построить когортный анализ", id="btn-cohort", color="primary"),
        ]),
    ], className="mb-3"),

    dcc.Loading(html.Div(id="coh-results"), type="circle", color="#10b981"),
], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px 16px"})


# ---------------------------------------------------------------------------
# Populate dropdowns
# ---------------------------------------------------------------------------
@callback(
    Output("coh-ds", "options"),
    Output("coh-ds", "value"),
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
    Output("coh-id-col", "options"),
    Output("coh-act-col", "options"),
    Output("coh-acq-col", "options"),
    Input("coh-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_cols(ds, datasets, prepared):
    if not ds:
        return [], [], [{"label": "(авто — первая активность)", "value": "__auto__"}]
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return [], [], [{"label": "(авто — первая активность)", "value": "__auto__"}]
    all_cols = [{"label": c, "value": c} for c in df.columns]
    acq_opts = [{"label": "(авто — первая активность)", "value": "__auto__"}] + all_cols
    return all_cols, all_cols, acq_opts


# ---------------------------------------------------------------------------
# Main compute callback
# ---------------------------------------------------------------------------
@callback(
    Output("coh-results", "children"),
    Input("btn-cohort", "n_clicks"),
    State("coh-ds", "value"),
    State("coh-id-col", "value"),
    State("coh-act-col", "value"),
    State("coh-acq-col", "value"),
    State("coh-freq", "value"),
    State("coh-max-offset", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_cohort(n, ds, id_col, act_col, acq_col_val, freq, max_offset, datasets, prepared):
    if not all([ds, id_col, act_col]):
        return alert_banner("Выберите датасет, ID клиента и дату активности.", "warning")

    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    acq_col = None if (not acq_col_val or acq_col_val == "__auto__") else acq_col_val
    max_offset = int(max_offset or 12)

    try:
        cohort_counts = build_cohort_table(
            df, id_col, act_col,
            acquisition_date_col=acq_col,
            cohort_freq=freq or "MS",
            max_offset=max_offset,
        )
    except Exception as e:
        logger.exception("Cohort build error")
        return alert_banner(f"Ошибка построения когорт: {e}", "danger")

    if cohort_counts.empty:
        return alert_banner("Не удалось построить когорты. Проверьте колонки с датами.", "warning")

    ret = retention_table(cohort_counts)
    churn = churn_rate_table(ret)
    avg_ret = average_retention_curve(cohort_counts)

    n_cohorts = len(cohort_counts)
    total_customers = int(cohort_counts[0].sum()) if 0 in cohort_counts.columns else 0
    avg_ret_1 = f"{avg_ret.get(1, 0):.0%}" if 1 in avg_ret.index else "—"
    avg_ret_3 = f"{avg_ret.get(3, 0):.0%}" if 3 in avg_ret.index else "—"

    # Format index for display
    def _fmt(idx):
        if hasattr(idx, "strftime"):
            return idx.strftime("%Y-%m")
        return str(idx)

    # ── Heatmap ──────────────────────────────────────────────────────────────
    disp_ret = ret.copy()
    disp_ret.index = [_fmt(c) for c in disp_ret.index]
    fig_heat = go.Figure(data=go.Heatmap(
        z=(disp_ret.values * 100),
        x=[f"Период {c}" for c in disp_ret.columns],
        y=disp_ret.index.tolist(),
        colorscale="Blues",
        text=np.round(disp_ret.values * 100, 1),
        texttemplate="%{text:.0f}%",
        textfont=dict(size=9),
        hoverongaps=False,
        colorbar=dict(title="%"),
        hovertemplate="Когорта: %{y}<br>%{x}<br>Удержание: %{z:.1f}%<extra></extra>",
    ))
    fig_heat.update_layout(
        title="Тепловая карта удержания",
        height=max(320, n_cohorts * 32 + 100),
        yaxis=dict(autorange="reversed"),
    )
    apply_kibad_theme(fig_heat)

    # ── Churn heatmap ─────────────────────────────────────────────────────────
    disp_churn = (churn * 100).round(1)
    disp_churn.index = [_fmt(c) for c in disp_churn.index]
    churn_vals = disp_churn.values.copy()
    churn_vals[np.isnan(churn_vals)] = 0
    fig_churn = go.Figure(data=go.Heatmap(
        z=churn_vals,
        x=[f"Период {c}" for c in disp_churn.columns],
        y=disp_churn.index.tolist(),
        colorscale="Reds",
        text=np.round(churn_vals, 1),
        texttemplate="%{text:.0f}%",
        textfont=dict(size=9),
        hoverongaps=False,
        colorbar=dict(title="%"),
        hovertemplate="Когорта: %{y}<br>%{x}<br>Отток: %{z:.1f}%<extra></extra>",
    ))
    fig_churn.update_layout(
        title="Отток по периодам",
        height=max(320, n_cohorts * 32 + 100),
        yaxis=dict(autorange="reversed"),
    )
    apply_kibad_theme(fig_churn)

    # ── Average retention curve ───────────────────────────────────────────────
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=[f"Период {k}" for k in avg_ret.index],
        y=(avg_ret.values * 100).round(1),
        mode="lines+markers+text",
        text=[f"{v:.0f}%" for v in avg_ret.values * 100],
        textposition="top center",
        line=dict(color="#10b981", width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.12)",
    ))
    fig_curve.update_layout(
        title="Средняя кривая удержания (взвешенная по размеру когорты)",
        yaxis=dict(title="Удержание, %", range=[0, 108]),
        height=380,
    )
    apply_kibad_theme(fig_curve)

    # ── CLV section ───────────────────────────────────────────────────────────
    clv_section = html.Div([
        section_header("Расчёт CLV (Customer Lifetime Value)"),
        dbc.Row([
            dbc.Col([
                html.Label("ARPU (выручка на клиента за период)", className="kb-label"),
                dcc.Input(id="coh-arpu", type="number", value=1000, min=0.01, step=100,
                          style={"width": "100%"}),
            ], md=3),
            dbc.Col([
                html.Label("Годовая ставка дисконтирования, %", className="kb-label"),
                dcc.Input(id="coh-discount", type="number", value=12, min=0, max=100, step=1,
                          style={"width": "100%"}),
            ], md=3),
            dbc.Col([
                html.Label("Горизонт (месяцев)", className="kb-label"),
                dcc.Input(id="coh-horizon", type="number", value=12, min=1, max=60, step=1,
                          style={"width": "100%"}),
            ], md=3),
            dbc.Col([
                dbc.Button("💰 Рассчитать CLV", id="btn-clv", color="success", className="mt-4"),
            ], md=3),
        ], className="mb-3"),
        dcc.Loading(html.Div(id="coh-clv-result"), type="circle", color="#10b981"),
        # Hidden store for retention data
        dcc.Store(id="coh-ret-store", data=ret.to_json()),
    ])

    # ── Profiles table ────────────────────────────────────────────────────────
    ret_display = (ret * 100).round(1)
    ret_display.index = [_fmt(c) for c in ret_display.index]
    ret_display = ret_display.reset_index()
    ret_display.columns = ["Когорта"] + [f"Период {c}" for c in ret_display.columns[1:]]

    return html.Div([
        section_header("Результаты когортного анализа"),
        html.Div([
            stat_card("Когорт", str(n_cohorts)),
            stat_card("Клиентов", f"{total_customers:,}"),
            stat_card("Удержание (пер. 1)", avg_ret_1),
            stat_card("Удержание (пер. 3)", avg_ret_3),
        ], className="kb-stats-grid mb-3"),

        dbc.Tabs([
            dbc.Tab(label="🔥 Тепловая карта", tab_id="coh-tab-heat"),
            dbc.Tab(label="📉 Отток", tab_id="coh-tab-churn"),
            dbc.Tab(label="📈 Средняя кривая", tab_id="coh-tab-curve"),
            dbc.Tab(label="💰 CLV", tab_id="coh-tab-clv"),
            dbc.Tab(label="📋 Данные", tab_id="coh-tab-data"),
        ], id="coh-result-tabs", active_tab="coh-tab-heat"),
        html.Div(id="coh-tab-content", className="mt-3"),

        dcc.Store(id="coh-ret-store", data=ret.to_json()),
        dcc.Store(id="coh-heat-fig", data=fig_heat.to_json()),
        dcc.Store(id="coh-churn-fig", data=fig_churn.to_json()),
        dcc.Store(id="coh-curve-fig", data=fig_curve.to_json()),
        dcc.Store(id="coh-ret-display", data=ret_display.to_json()),
    ])


@callback(
    Output("coh-tab-content", "children"),
    Input("coh-result-tabs", "active_tab"),
    State("coh-heat-fig", "data"),
    State("coh-churn-fig", "data"),
    State("coh-curve-fig", "data"),
    State("coh-ret-display", "data"),
)
def render_coh_tab(tab, heat_json, churn_json, curve_json, ret_json):
    import plotly.io as pio
    if tab == "coh-tab-heat" and heat_json:
        return dcc.Graph(figure=pio.from_json(heat_json))
    if tab == "coh-tab-churn" and churn_json:
        return dcc.Graph(figure=pio.from_json(churn_json))
    if tab == "coh-tab-curve" and curve_json:
        return dcc.Graph(figure=pio.from_json(curve_json))
    if tab == "coh-tab-clv":
        return html.Div([
            section_header("CLV — Customer Lifetime Value"),
            dbc.Row([
                dbc.Col([
                    html.Label("ARPU", className="kb-label"),
                    dcc.Input(id="coh-arpu", type="number", value=1000, min=0.01, step=100,
                              style={"width": "100%"}),
                ], md=3),
                dbc.Col([
                    html.Label("Ставка дисконтирования, %/год", className="kb-label"),
                    dcc.Input(id="coh-discount", type="number", value=12, min=0, max=100,
                              style={"width": "100%"}),
                ], md=3),
                dbc.Col([
                    html.Label("Горизонт (месяцев)", className="kb-label"),
                    dcc.Input(id="coh-horizon", type="number", value=12, min=1, max=60,
                              style={"width": "100%"}),
                ], md=3),
                dbc.Col([
                    dbc.Button("💰 Рассчитать CLV", id="btn-clv", color="success", className="mt-4"),
                ], md=3),
            ], className="mb-3"),
            dcc.Loading(html.Div(id="coh-clv-result"), type="circle", color="#10b981"),
        ])
    if tab == "coh-tab-data" and ret_json:
        try:
            df_ret = pd.read_json(ret_json)
            return html.Div([
                section_header("Таблица удержания (%)"),
                data_table(df_ret, id="coh-ret-tbl", page_size=20),
            ])
        except Exception:
            logger.warning("Failed to parse retention JSON for data tab", exc_info=True)
            return html.Div()
    return html.Div()


@callback(
    Output("coh-clv-result", "children"),
    Input("btn-clv", "n_clicks"),
    State("coh-arpu", "value"),
    State("coh-discount", "value"),
    State("coh-horizon", "value"),
    State("coh-ret-store", "data"),
    prevent_initial_call=True,
)
def compute_clv_cb(n, arpu, discount_pct, horizon, ret_json):
    if not ret_json:
        return alert_banner("Сначала постройте когортный анализ.", "warning")
    try:
        ret = pd.read_json(ret_json)
        ret.columns = [int(c) for c in ret.columns]
        clv = compute_clv(ret, float(arpu or 1000), float(discount_pct or 12) / 100.0, int(horizon or 12))
        if clv.empty:
            return alert_banner("Не удалось рассчитать CLV.", "warning")

        def _fmt(idx):
            if hasattr(idx, "strftime"):
                return idx.strftime("%Y-%m")
            return str(idx)

        clv_df = clv.reset_index()
        clv_df.columns = ["Когорта", "CLV"]
        clv_df["Когорта"] = clv_df["Когорта"].apply(_fmt)
        clv_df["CLV"] = clv_df["CLV"].round(2)

        fig = go.Figure(go.Bar(
            x=clv_df["Когорта"], y=clv_df["CLV"],
            marker_color="#10b981",
            hovertemplate="Когорта: %{x}<br>CLV: %{y:,.2f}<extra></extra>",
        ))
        fig.update_layout(
            title=f"CLV по когортам (ARPU={arpu}, ставка={discount_pct}%, горизонт={horizon} мес.)",
            xaxis_title="Когорта", yaxis_title="CLV",
            height=380,
        )
        apply_kibad_theme(fig)

        avg_clv = clv_df["CLV"].mean()
        max_clv = clv_df["CLV"].max()
        return html.Div([
            html.Div([
                stat_card("Средний CLV", f"{avg_clv:,.0f}"),
                stat_card("Макс. CLV", f"{max_clv:,.0f}"),
            ], className="kb-stats-grid mb-3"),
            dcc.Graph(figure=fig),
            data_table(clv_df, id="clv-tbl", page_size=20),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка расчёта CLV: {e}", "danger")

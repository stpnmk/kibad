"""p09_simulation – Scenario simulation page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.cards import stat_card
from app.components.alerts import alert_banner
from core.simulation import run_scenario
from core.audit import log_event

dash.register_page(__name__, path="/simulation", name="9. Симуляция", order=9, icon="dice-5")

layout = html.Div([
    page_header("9. Сценарное моделирование", "Шоковые сценарии и Монте-Карло"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="sim-ds", placeholder="Выберите датасет..."), width=4),
    ], className="mb-3"),

    dbc.Tabs(id="sim-tabs", active_tab="tab-scenario", children=[
        dbc.Tab(label="Сценарный анализ", tab_id="tab-scenario"),
        dbc.Tab(label="Монте-Карло", tab_id="tab-mc"),
    ]),
    dcc.Loading(html.Div(id="sim-content"), type="circle", color="#10b981"),
])


@callback(
    Output("sim-ds", "options"),
    Output("sim-ds", "value"),
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
    Output("sim-content", "children"),
    Input("sim-tabs", "active_tab"),
    Input("sim-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_tab(tab, ds, datasets, prepared):
    if not ds:
        return empty_state("", "Выберите датасет", "")
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        return alert_banner("Нет числовых колонок.", "info")

    if tab == "tab-scenario":
        return html.Div([
            section_header("Сценарный анализ", "Изменение параметров и оценка воздействия"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Целевая метрика", className="kb-stat-label"),
                            dcc.Dropdown(id="sim-target", options=[{"label": c, "value": c} for c in num_cols],
                                         value=num_cols[0] if num_cols else None),
                        ], width=4),
                        dbc.Col([
                            html.Label("Факторы для шока", className="kb-stat-label"),
                            dcc.Dropdown(id="sim-factors", options=[{"label": c, "value": c} for c in num_cols],
                                         multi=True, value=num_cols[1:3] if len(num_cols) > 1 else []),
                        ], width=4),
                        dbc.Col([
                            html.Label("Величина шока (%)", className="kb-stat-label"),
                            dcc.Input(id="sim-shock-pct", type="number", value=-10, style={"width": "100%"}),
                        ], width=2),
                        dbc.Col([
                            dbc.Button("Рассчитать", id="sim-run-btn", color="primary", className="mt-4"),
                        ], width=2),
                    ]),
                ]),
            ], className="mb-3"),
            html.Div(id="sim-scenario-result"),
        ])

    elif tab == "tab-mc":
        return html.Div([
            section_header("Монте-Карло симуляция"),
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Колонка", className="kb-stat-label"),
                            dcc.Dropdown(id="mc-col", options=[{"label": c, "value": c} for c in num_cols],
                                         value=num_cols[0] if num_cols else None),
                        ], width=4),
                        dbc.Col([
                            html.Label("Кол-во симуляций", className="kb-stat-label"),
                            dcc.Input(id="mc-n-sims", type="number", value=1000, min=100, max=50000, style={"width": "100%"}),
                        ], width=3),
                        dbc.Col([
                            html.Label("Горизонт (периодов)", className="kb-stat-label"),
                            dcc.Input(id="mc-horizon", type="number", value=12, min=1, max=120, style={"width": "100%"}),
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Запустить", id="mc-run-btn", color="primary", className="mt-4"),
                        ], width=2),
                    ]),
                ]),
            ], className="mb-3"),
            html.Div(id="mc-result"),
        ])

    return ""


@callback(
    Output("sim-scenario-result", "children"),
    Input("sim-run-btn", "n_clicks"),
    State("sim-ds", "value"),
    State("sim-target", "value"),
    State("sim-factors", "value"),
    State("sim-shock-pct", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_scenario_cb(n, ds, target, factors, shock_pct, datasets, prepared):
    if not all([ds, target, factors]):
        return alert_banner("Заполните все поля.", "warning")
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")
    try:
        shock = {f: float(shock_pct) / 100.0 for f in factors}
        result = run_scenario(df, target_col=target, shocks=shock)
        log_event("simulation", dataset=ds, details=f"scenario shock={shock_pct}%")

        base_val = result.get("base_value", 0)
        new_val = result.get("scenario_value", 0)
        delta = new_val - base_val
        delta_pct = (delta / abs(base_val) * 100) if base_val else 0

        return html.Div([
            html.Div([
                stat_card("Базовое значение", f"{base_val:,.2f}"),
                stat_card("Сценарное значение", f"{new_val:,.2f}"),
                stat_card("Изменение", f"{delta:+,.2f}", delta=f"{delta_pct:+.1f}%"),
            ], className="kb-stats-grid"),
            data_table(pd.DataFrame(result.get("factor_impacts", [])), id="sim-impact-tbl") if result.get("factor_impacts") else "",
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


@callback(
    Output("mc-result", "children"),
    Input("mc-run-btn", "n_clicks"),
    State("sim-ds", "value"),
    State("mc-col", "value"),
    State("mc-n-sims", "value"),
    State("mc-horizon", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_mc(n, ds, col, n_sims, horizon, datasets, prepared):
    if not all([ds, col]):
        return alert_banner("Выберите колонку.", "warning")
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")
    try:
        # Simple Monte Carlo simulation (geometric Brownian motion)
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(vals) < 2:
            return alert_banner("Недостаточно данных для Монте-Карло.", "warning")

        returns = np.diff(vals) / vals[:-1]
        mu = np.mean(returns)
        sigma = np.std(returns)
        n_s = int(n_sims or 1000)
        h = int(horizon or 12)
        last_val = vals[-1]

        paths = np.zeros((n_s, h + 1))
        paths[:, 0] = last_val
        for t in range(1, h + 1):
            paths[:, t] = paths[:, t - 1] * (1 + np.random.normal(mu, sigma, n_s))

        final_vals = paths[:, -1]
        var_95 = float(np.percentile(final_vals, 5))
        cvar_95 = float(np.mean(final_vals[final_vals <= var_95]))
        mean_val = float(np.mean(final_vals))
        median_val = float(np.median(final_vals))
        log_event("simulation", dataset=ds, details=f"monte_carlo n={n_s}")

        fig = go.Figure()
        for i in range(min(100, n_s)):
            fig.add_trace(go.Scatter(y=paths[i], mode="lines", opacity=0.1,
                                     line=dict(width=0.5, color="#10b981"), showlegend=False))
        fig.update_layout(title="Монте-Карло траектории", xaxis_title="Период", yaxis_title=col)
        apply_kibad_theme(fig)

        return html.Div([
            html.Div([
                stat_card("VaR (95%)", f"{var_95:,.2f}"),
                stat_card("CVaR (95%)", f"{cvar_95:,.2f}"),
                stat_card("Среднее", f"{mean_val:,.2f}"),
                stat_card("Медиана", f"{median_val:,.2f}"),
            ], className="kb-stats-grid"),
            dcc.Graph(figure=fig),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")

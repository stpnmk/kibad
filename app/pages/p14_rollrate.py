"""
app/pages/p14_rollrate.py – Roll-Rate / матрица миграций (Dash).

Марковские цепи: матрица переходов, прогноз на n периодов,
стационарное распределение, динамика ставок.
"""
from __future__ import annotations

import json

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, dash_table, no_update
import dash_bootstrap_components as dbc

from app.state import (
    get_df_from_store, get_df_from_stores, list_datasets, save_dataframe,
    STORE_DATASET, STORE_ACTIVE_DS, STORE_PREPARED,
)
from app.figure_theme import apply_kibad_theme
from core.rollrate import (
    auto_bucket,
    build_transition_matrix,
    matrix_power,
    steady_state,
    roll_forward_rates,
    cure_rates,
    transition_time_series,
    BUCKET_ORDER,
)

dash.register_page(
    __name__,
    path="/rollrate",
    name="14. Roll Rate",
    order=14,
    icon="percent",
)

COLORS = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H3("14. Roll-Rate матрица"), className="mb-1")),
    dbc.Row(dbc.Col(html.P(
        "Матрица миграции и Марковские цепи", className="text-muted mb-3",
    ))),

    # Hidden stores for page-level results
    dcc.Store(id="rr-result-store", data=None),

    dbc.Row([
        # --- Settings panel ---
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H6("Параметры", className="mb-3"),
                dbc.Label("ID займа"),
                dcc.Dropdown(id="rr-loan-id-col", placeholder="Выберите столбец"),
                dbc.Label("Период наблюдения (дата)", className="mt-2"),
                dcc.Dropdown(id="rr-period-col", placeholder="Выберите столбец"),
                dbc.Label("Статус / DPD колонка", className="mt-2"),
                dcc.Dropdown(id="rr-bucket-col", placeholder="Выберите столбец"),
                dbc.Label("Тип колонки", className="mt-2"),
                dbc.RadioItems(
                    id="rr-col-type",
                    options=[
                        {"label": "Бакет (категория)", "value": "bucket"},
                        {"label": "DPD (число)", "value": "dpd"},
                    ],
                    value="bucket",
                    inline=True,
                ),
                html.Div(id="rr-edges-container", children=[
                    dbc.Label("Границы бакетов", className="mt-2"),
                    dbc.Input(id="rr-edges", value="0,1,30,60,90,180", type="text"),
                ], style={"display": "none"}),
                dbc.Label("Горизонт прогноза (месяцев)", className="mt-2"),
                dbc.Input(id="rr-horizon", type="number", value=3, min=1, max=24, step=1),
                dbc.Button(
                    "Рассчитать", id="rr-run-btn", color="primary",
                    className="mt-3 w-100",
                ),
                html.Div(id="rr-row-count", className="text-muted mt-2"),
            ])),
        ], md=3),

        # --- Results ---
        dbc.Col([
            dcc.Loading(type="circle", color="#10b981", children=[
                html.Div(id="rr-alert"),
                dbc.Tabs(id="rr-tabs", active_tab="tab-matrix", children=[
                    dbc.Tab(label="Матрица переходов", tab_id="tab-matrix"),
                    dbc.Tab(label="Прогноз (n-периодов)", tab_id="tab-forecast"),
                    dbc.Tab(label="Динамика ставок", tab_id="tab-dynamics"),
                    dbc.Tab(label="Таблица", tab_id="tab-table"),
                    dbc.Tab(label="Марковские цепи", tab_id="tab-markov"),
                ]),
                html.Div(id="rr-tab-content", className="mt-3"),
            ]),
        ], md=9),
    ]),
], fluid=True, className="p-4")


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("rr-edges-container", "style"),
    Input("rr-col-type", "value"),
)
def toggle_edges(col_type):
    return {"display": "block"} if col_type == "dpd" else {"display": "none"}


@callback(
    [Output("rr-loan-id-col", "options"),
     Output("rr-period-col", "options"),
     Output("rr-bucket-col", "options"),
     Output("rr-row-count", "children")],
    [Input(STORE_DATASET, "data"),
     Input(STORE_PREPARED, "data"),
     Input(STORE_ACTIVE_DS, "data")],
)
def populate_columns(ds_data, prep_data, active_ds):
    ds = active_ds or next(iter(prep_data or {}), None) or next(iter(ds_data or {}), None)
    if not ds:
        return [], [], [], ""
    df = get_df_from_stores(ds, prep_data, ds_data)
    if df is None:
        return [], [], [], ""
    cols = [{"label": c, "value": c} for c in df.columns]
    return cols, cols, cols, f"Строк: {len(df):,}"


@callback(
    [Output("rr-result-store", "data"),
     Output("rr-alert", "children")],
    Input("rr-run-btn", "n_clicks"),
    [State(STORE_DATASET, "data"),
     State(STORE_PREPARED, "data"),
     State(STORE_ACTIVE_DS, "data"),
     State("rr-loan-id-col", "value"),
     State("rr-period-col", "value"),
     State("rr-bucket-col", "value"),
     State("rr-col-type", "value"),
     State("rr-edges", "value"),
     State("rr-horizon", "value")],
    prevent_initial_call=True,
)
def run_rollrate(n_clicks, ds_data, prep_data, active_ds,
                 loan_id_col, period_col, bucket_col,
                 col_type, edges_str, horizon):
    if not active_ds or not loan_id_col or not period_col or not bucket_col:
        return no_update, dbc.Alert("Заполните все параметры.", color="warning")

    df = get_df_from_stores(active_ds, prep_data, ds_data)
    if df is None or df.empty:
        return no_update, dbc.Alert("Нет данных.", color="danger")

    try:
        work_df = df.copy()
        if col_type == "dpd":
            edges = [int(e.strip()) for e in edges_str.split(",")]
            work_df["_bucket"] = auto_bucket(
                pd.to_numeric(work_df[bucket_col], errors="coerce"), edges=edges,
            )
            used_bucket_col = "_bucket"
        else:
            used_bucket_col = bucket_col

        count_m, rate_m = build_transition_matrix(
            work_df, loan_id_col=loan_id_col, period_col=period_col,
            bucket_col=used_bucket_col,
        )
        ts_df = transition_time_series(
            work_df, loan_id_col=loan_id_col, period_col=period_col,
            bucket_col=used_bucket_col,
        )

        # Serialize for store
        path_work = save_dataframe(work_df, f"rr_work_{active_ds}")
        result = {
            "count_m": count_m.to_json(),
            "rate_m": rate_m.to_json(),
            "ts_df": ts_df.to_json(date_format="iso") if not ts_df.empty else "{}",
            "work_df_path": path_work,
            "used_bucket_col": used_bucket_col,
            "period_col": period_col,
            "horizon": horizon or 3,
        }
        return result, dbc.Alert("Матрица миграций построена.", color="success", duration=4000)
    except Exception as exc:
        return no_update, dbc.Alert(f"Ошибка: {exc}", color="danger")


@callback(
    Output("rr-tab-content", "children"),
    [Input("rr-tabs", "active_tab"),
     Input("rr-result-store", "data")],
)
def render_tab(active_tab, result_data):
    if not result_data:
        return dbc.Alert(
            "Настройте параметры и нажмите «Рассчитать».", color="info",
        )

    count_m = pd.read_json(result_data["count_m"])
    rate_m = pd.read_json(result_data["rate_m"])
    horizon = result_data.get("horizon", 3)

    active_buckets = [b for b in BUCKET_ORDER if b in rate_m.index and rate_m.loc[b].sum() > 0]
    if not active_buckets:
        active_buckets = [b for b in BUCKET_ORDER if b in rate_m.index]

    # ---- Tab: Matrix ----
    if active_tab == "tab-matrix":
        rate_display = rate_m.loc[active_buckets, active_buckets] if active_buckets else rate_m
        z_vals = rate_display.values * 100
        text_vals = [[f"{z_vals[i, j]:.1f}%" for j in range(z_vals.shape[1])] for i in range(z_vals.shape[0])]

        fig_matrix = go.Figure(data=go.Heatmap(
            z=z_vals, x=active_buckets, y=active_buckets,
            colorscale="RdBu_r", zmin=0, zmax=100,
            text=text_vals, texttemplate="%{text}",
            hovertemplate="Из: %{y}<br>В: %{x}<br>Ставка: %{z:.1f}%<extra></extra>",
            colorbar=dict(title="%"),
        ))
        fig_matrix.update_layout(
            title="Матрица переходов (% строки)",
            xaxis_title="Статус в следующем периоде",
            yaxis_title="Статус в текущем периоде",
        )
        apply_kibad_theme(fig_matrix)

        # Key metrics
        rf = roll_forward_rates(rate_m)
        cr = cure_rates(rate_m)
        non_absorbing = [b for b in active_buckets if b not in ("Списан", "Закрыт")]
        avg_rf = float(rf[non_absorbing].mean()) * 100 if non_absorbing else 0.0
        avg_cr = float(cr[non_absorbing].mean()) * 100 if non_absorbing else 0.0
        wo_rate = float(rate_m.loc["90+", "Списан"]) * 100 if "90+" in rate_m.index and "Списан" in rate_m.columns else 0.0
        net_flow = avg_rf - avg_cr

        metrics_row = dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Small("Ср. ставка ухудшения", className="text-muted"),
                html.H5(f"{avg_rf:.1f}%"),
            ])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Small("Ср. ставка улучшения", className="text-muted"),
                html.H5(f"{avg_cr:.1f}%"),
            ])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Small("Приток в списание", className="text-muted"),
                html.H5(f"{wo_rate:.1f}%"),
            ])), md=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.Small("Чистый поток", className="text-muted"),
                html.H5(f"{net_flow:+.1f}%",
                         className="text-danger" if net_flow > 0 else "text-success"),
            ])), md=3),
        ], className="mb-3")

        return html.Div([
            html.P(
                "Строки = текущий статус | Столбцы = следующий статус. "
                "Читайте построчно: из статуса X, Y% переходят в статус Z.",
                className="text-muted",
            ),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_matrix), md=7),
                dbc.Col(metrics_row, md=5),
            ]),
        ])

    # ---- Tab: Forecast ----
    if active_tab == "tab-forecast":
        try:
            rate_m_active = rate_m.loc[active_buckets, active_buckets]
            # Uniform starting vector as fallback
            start_vec = np.array([1.0 / len(active_buckets)] * len(active_buckets))

            projections = {0: {b: v for b, v in zip(active_buckets, start_vec)}}
            for t in range(1, horizon + 1):
                T_n = matrix_power(rate_m_active, t)
                proj_vec = start_vec @ T_n.values
                projections[t] = {b: float(proj_vec[i]) for i, b in enumerate(active_buckets)}

            proj_df = pd.DataFrame(projections).T
            proj_df.index.name = "Период"
            proj_df = proj_df * 100

            fig_proj = go.Figure()
            for i, bucket in enumerate(active_buckets):
                if bucket in proj_df.columns:
                    fig_proj.add_trace(go.Scatter(
                        x=proj_df.index, y=proj_df[bucket], name=bucket,
                        stackgroup="one",
                        line=dict(color=COLORS[i % len(COLORS)]),
                        hovertemplate=f"{bucket}<br>Период: %{{x}}<br>Доля: %{{y:.1f}}%<extra></extra>",
                    ))
            try:
                ss = steady_state(rate_m_active)
                for i, bucket in enumerate(active_buckets):
                    if bucket in ss.index:
                        fig_proj.add_hline(
                            y=float(ss[bucket]) * 100, line_dash="dash",
                            line_color=COLORS[i % len(COLORS)], opacity=0.4,
                        )
            except Exception:
                pass

            fig_proj.update_layout(
                title=f"Прогноз распределения портфеля на {horizon} периодов",
                xaxis_title="Период", yaxis_title="Доля портфеля (%)",
                hovermode="x unified",
            )
            apply_kibad_theme(fig_proj)

            table_data = proj_df.round(2).reset_index()
            table_data.columns = [str(c) for c in table_data.columns]

            return html.Div([
                dcc.Graph(figure=fig_proj),
                html.P("Пунктирные линии -- стационарное распределение.", className="text-muted"),
                html.H6("Таблица проекции", className="mt-3"),
                dash_table.DataTable(
                    data=table_data.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in table_data.columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "padding": "6px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#191c24", "color": "#e4e7ee"},
                    style_data={"backgroundColor": "#111318", "color": "#e4e7ee"},
                ),
            ])
        except Exception as exc:
            return dbc.Alert(f"Ошибка прогноза: {exc}", color="danger")

    # ---- Tab: Dynamics ----
    if active_tab == "tab-dynamics":
        ts_raw = result_data.get("ts_df", "{}")
        if ts_raw == "{}":
            return dbc.Alert("Нужно 2+ периода для анализа динамики.", color="info")
        ts_df = pd.read_json(ts_raw)
        if ts_df.empty or "period" not in ts_df.columns or ts_df["period"].nunique() < 2:
            return dbc.Alert("Нужно 2+ периода для анализа динамики.", color="info")

        buckets_available = [b for b in BUCKET_ORDER if b in ts_df["source_bucket"].unique()]

        figs = []
        for metric, title, y_col_name in [
            ("Roll-forward", "Динамика ставок: Roll-forward", "roll_forward_rate"),
            ("Cure", "Динамика ставок: Cure", "cure_rate"),
        ]:
            fig_dyn = go.Figure()
            for i, bucket in enumerate(buckets_available):
                bdf = ts_df[ts_df["source_bucket"] == bucket].sort_values("period")
                if y_col_name in bdf.columns:
                    fig_dyn.add_trace(go.Scatter(
                        x=bdf["period"].astype(str), y=bdf[y_col_name] * 100,
                        name=bucket, mode="lines+markers",
                        line=dict(color=COLORS[i % len(COLORS)]),
                    ))
            fig_dyn.update_layout(
                title=title, xaxis_title="Период", yaxis_title="%",
                hovermode="x unified",
            )
            apply_kibad_theme(fig_dyn)
            figs.append(dcc.Graph(figure=fig_dyn))

        return html.Div(figs)

    # ---- Tab: Table ----
    if active_tab == "tab-table":
        count_display = count_m.loc[active_buckets, active_buckets].astype(int)
        rate_display = (rate_m.loc[active_buckets, active_buckets] * 100).round(2)

        csv_string = rate_m.to_csv()

        return html.Div([
            html.H6("Матрица переходов: количество"),
            dash_table.DataTable(
                data=count_display.reset_index().to_dict("records"),
                columns=[{"name": str(c), "id": str(c)} for c in count_display.reset_index().columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "6px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#191c24", "color": "#e4e7ee"},
                style_data={"backgroundColor": "#111318", "color": "#e4e7ee"},
            ),
            html.H6("Матрица переходов: ставки (%)", className="mt-4"),
            dash_table.DataTable(
                data=rate_display.reset_index().to_dict("records"),
                columns=[{"name": str(c), "id": str(c)} for c in rate_display.reset_index().columns],
                style_table={"overflowX": "auto"},
                style_cell={"textAlign": "center", "padding": "6px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#191c24", "color": "#e4e7ee"},
                style_data={"backgroundColor": "#111318", "color": "#e4e7ee"},
            ),
            html.A(
                dbc.Button("Скачать матрицу ставок (CSV)", color="secondary", className="mt-3"),
                href=f"data:text/csv;charset=utf-8,{csv_string}",
                download="rollrate_matrix.csv",
            ),
        ])

    # ---- Tab: Markov ----
    if active_tab == "tab-markov":
        active_mask = rate_m.sum(axis=1) > 0
        mat_df = rate_m.loc[active_mask, active_mask]
        states = list(mat_df.index)
        auto_absorbing = [s for s in states if mat_df.loc[s, s] >= 0.99]
        absorbing_states = auto_absorbing or ([states[-1]] if states else [])
        transient_states = [s for s in states if s not in absorbing_states]

        if not transient_states or len(absorbing_states) >= len(states):
            return dbc.Alert(
                "Невозможно выделить транзитные и поглощающие состояния. "
                "Проверьте данные.", color="warning",
            )

        try:
            Q = mat_df.loc[transient_states, transient_states].values.astype(float)
            R = mat_df.loc[transient_states, absorbing_states].values.astype(float)
            I_mat = np.eye(len(transient_states))
            N = np.linalg.inv(I_mat - Q)

            t_steps = N.sum(axis=1)
            B = N @ R
            B_df = pd.DataFrame(B, index=transient_states, columns=absorbing_states).round(4)

            # Absorption probability heatmap
            fig_abs = px.imshow(
                B_df, text_auto=".2%", color_continuous_scale="RdYlGn_r",
                aspect="auto",
                title="Вероятности поглощения (Lifetime PD по бакетам)",
                labels=dict(color="Вероятность"),
            )
            fig_abs.update_layout(height=max(250, len(transient_states) * 60 + 100))
            apply_kibad_theme(fig_abs)

            # Time to absorption bar chart
            t_df = pd.DataFrame({
                "Состояние": transient_states,
                "Шагов до поглощения": t_steps.round(2),
            })
            fig_time = px.bar(
                t_df, x="Состояние", y="Шагов до поглощения",
                color="Шагов до поглощения", color_continuous_scale="RdYlGn_r",
                title="Ожидаемое время до дефолта/списания (в периодах)",
                text_auto=".1f",
            )
            fig_time.update_layout(showlegend=False, height=350)
            apply_kibad_theme(fig_time)

            # Fundamental matrix table
            N_df = pd.DataFrame(N, index=transient_states, columns=transient_states).round(3)

            return html.Div([
                html.H6("Марковские цепи: Поглощающие состояния"),
                html.P(
                    f"Поглощающие состояния: {', '.join(absorbing_states)}. "
                    f"Транзитные: {', '.join(transient_states)}.",
                    className="text-muted",
                ),
                dcc.Graph(figure=fig_abs),
                html.Hr(),
                dcc.Graph(figure=fig_time),
                html.Hr(),
                html.H6("Фундаментальная матрица N = (I - Q)^-1"),
                html.P(
                    "N[i,j] = ожидаемое число посещений состояния j при старте из i до поглощения",
                    className="text-muted",
                ),
                dash_table.DataTable(
                    data=N_df.reset_index().to_dict("records"),
                    columns=[{"name": str(c), "id": str(c)} for c in N_df.reset_index().columns],
                    style_table={"overflowX": "auto"},
                    style_cell={"textAlign": "center", "padding": "6px"},
                    style_header={"fontWeight": "bold", "backgroundColor": "#191c24", "color": "#e4e7ee"},
                    style_data={"backgroundColor": "#111318", "color": "#e4e7ee"},
                ),
            ])
        except np.linalg.LinAlgError:
            return dbc.Alert(
                "Матрица (I - Q) вырождена. Проверьте, что транзитные состояния "
                "не образуют замкнутый цикл.", color="danger",
            )

    return html.Div()

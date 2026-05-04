"""p15_abc.py – ABC-XYZ анализ (Dash): классификация по вкладу и стабильности."""
import logging

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
    get_df_from_store, get_df_from_stores,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.alerts import alert_banner

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/abc", name="15. ABC-XYZ", order=15, icon="tags")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div([
    page_header("15. ABC-XYZ анализ", "Классификация объектов по вкладу и стабильности"),

    dbc.Alert([
        html.Strong("ABC"), " — классификация по принципу Парето: A = 80% объёма, B = следующие 15%, C = остаток. ",
        html.Strong("XYZ"), " — по стабильности (коэф. вариации CV): X < 10% (стабильные), 10–25% = Y, > 25% = Z. ",
        html.Strong("Кросс-матрица"), " даёт 9 сегментов: AX — ключевые стабильные, CZ — малозначимые нестабильные.",
    ], color="info", className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id="abc-ds", placeholder="Выберите датасет..."), md=4),
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("Параметры анализа"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Объект (SKU / клиент / продукт)", className="kb-label"),
                    dcc.Dropdown(id="abc-item-col", className="kb-select",
                                 placeholder="Выберите колонку..."),
                ], md=3),
                dbc.Col([
                    html.Label("Значение (выручка / объём)", className="kb-label"),
                    dcc.Dropdown(id="abc-value-col", className="kb-select",
                                 placeholder="Выберите числовую колонку..."),
                ], md=3),
                dbc.Col([
                    html.Label("Временная колонка (для XYZ)", className="kb-label"),
                    dcc.Dropdown(id="abc-time-col", className="kb-select",
                                 placeholder="Необязательно..."),
                ], md=3),
                dbc.Col([
                    html.Label("XYZ включить", className="kb-label"),
                    dcc.Checklist(id="abc-use-xyz",
                                  options=[{"label": " Включить XYZ-анализ", "value": "yes"}],
                                  value=[], style={"marginTop": "8px", "color": "#b0bccf"}),
                ], md=3),
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    html.Label("Порог A, %", className="kb-label"),
                    dcc.Input(id="abc-thresh-a", type="number", value=80, min=50, max=95,
                              style={"width": "100%"}),
                ], md=2),
                dbc.Col([
                    html.Label("Порог A+B, %", className="kb-label"),
                    dcc.Input(id="abc-thresh-ab", type="number", value=95, min=51, max=99,
                              style={"width": "100%"}),
                ], md=2),
                dbc.Col([
                    html.Label("Порог X (CV, %)", className="kb-label"),
                    dcc.Input(id="abc-thresh-x", type="number", value=10, min=1, max=30,
                              style={"width": "100%"}),
                ], md=2),
                dbc.Col([
                    html.Label("Порог Y (CV, %)", className="kb-label"),
                    dcc.Input(id="abc-thresh-y", type="number", value=25, min=2, max=60,
                              style={"width": "100%"}),
                ], md=2),
                dbc.Col([
                    dbc.Button("▶ Запустить ABC-XYZ", id="btn-abc", color="primary",
                               className="mt-4"),
                ], md=4),
            ]),
        ]),
    ], className="mb-3"),

    dcc.Loading(html.Div(id="abc-results"), type="circle", color="#10b981"),
], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px 16px"})


# ---------------------------------------------------------------------------
# Dataset / column dropdowns
# ---------------------------------------------------------------------------
@callback(
    Output("abc-ds", "options"),
    Output("abc-ds", "value"),
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
    Output("abc-item-col", "options"),
    Output("abc-value-col", "options"),
    Output("abc-time-col", "options"),
    Input("abc-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_cols(ds, datasets, prepared):
    if not ds:
        return [], [], []
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return [], [], []
    all_cols = [{"label": c, "value": c} for c in df.columns]
    num_cols = [{"label": c, "value": c} for c in df.select_dtypes(include="number").columns]
    cat_cols = [{"label": c, "value": c} for c in df.select_dtypes(
        include=["object", "category"]).columns]
    return (cat_cols or all_cols), num_cols, all_cols


# ---------------------------------------------------------------------------
# Main compute callback
# ---------------------------------------------------------------------------
@callback(
    Output("abc-results", "children"),
    Input("btn-abc", "n_clicks"),
    State("abc-ds", "value"),
    State("abc-item-col", "value"),
    State("abc-value-col", "value"),
    State("abc-time-col", "value"),
    State("abc-use-xyz", "value"),
    State("abc-thresh-a", "value"),
    State("abc-thresh-ab", "value"),
    State("abc-thresh-x", "value"),
    State("abc-thresh-y", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_abc(n, ds, item_col, value_col, time_col, use_xyz_list,
            thresh_a, thresh_ab, thresh_x, thresh_y,
            datasets, prepared):
    if not all([ds, item_col, value_col]):
        return alert_banner("Выберите датасет, объект и числовое значение.", "warning")

    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    abc_a = float(thresh_a or 80)
    abc_b = float(thresh_ab or 95)
    xyz_x = float(thresh_x or 10)
    xyz_y = float(thresh_y or 25)
    use_xyz = bool(use_xyz_list)

    try:
        # ── ABC calculation ───────────────────────────────────────────────────
        agg = df.groupby(item_col, dropna=False)[value_col].sum().reset_index()
        agg.columns = ["item", "total"]
        agg = agg.sort_values("total", ascending=False).reset_index(drop=True)
        total_sum = agg["total"].sum()
        agg["share"] = agg["total"] / total_sum * 100
        agg["cumshare"] = agg["share"].cumsum()

        def _abc(cumshare):
            if cumshare <= abc_a:
                return "A"
            elif cumshare <= abc_b:
                return "B"
            return "C"

        agg["ABC"] = agg["cumshare"].apply(_abc)

        # ── XYZ calculation ───────────────────────────────────────────────────
        if use_xyz and time_col and time_col in df.columns:
            work = df[[item_col, time_col, value_col]].copy()
            work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
            work = work.dropna(subset=[time_col])
            work["_period"] = work[time_col].dt.to_period("M").dt.to_timestamp()
            period_agg = work.groupby([item_col, "_period"])[value_col].sum().reset_index()
            pivot = period_agg.pivot(index=item_col, columns="_period",
                                     values=value_col).fillna(0)
            cv_series = (pivot.std(axis=1) / pivot.mean(axis=1).replace(0, np.nan) * 100)
            cv_df = cv_series.reset_index()
            cv_df.columns = ["item", "CV"]
            cv_df["CV"] = cv_df["CV"].fillna(999).round(1)

            def _xyz(cv):
                if cv <= xyz_x:
                    return "X"
                elif cv <= xyz_y:
                    return "Y"
                return "Z"

            cv_df["XYZ"] = cv_df["CV"].apply(_xyz)
            agg = agg.merge(cv_df, on="item", how="left")
            agg["XYZ"] = agg["XYZ"].fillna("Z")
            agg["CV"] = agg["CV"].fillna(0)
            agg["Класс"] = agg["ABC"] + agg["XYZ"]
        else:
            use_xyz = False
            agg["Класс"] = agg["ABC"]

        # ── Metrics ───────────────────────────────────────────────────────────
        n_items = len(agg)
        for cls in ["A", "B", "C"]:
            if cls not in agg["ABC"].values:
                pass
        n_a = int((agg["ABC"] == "A").sum())
        n_b = int((agg["ABC"] == "B").sum())
        n_c = int((agg["ABC"] == "C").sum())
        share_a = agg.loc[agg["ABC"] == "A", "total"].sum() / total_sum * 100

        # ── ABC Pareto chart ──────────────────────────────────────────────────
        _COLOR_MAP = {"A": "#10b981", "B": "#3b82f6", "C": "#707a94"}
        bar_colors = [_COLOR_MAP.get(c, "#707a94") for c in agg["ABC"]]

        fig_abc = go.Figure()
        fig_abc.add_trace(go.Bar(
            x=agg["item"].astype(str), y=agg["share"],
            marker_color=bar_colors, name="Доля, %",
            hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        ))
        fig_abc.add_trace(go.Scatter(
            x=agg["item"].astype(str), y=agg["cumshare"],
            mode="lines", name="Кумулятивная %",
            line=dict(color="#ef4444", width=2), yaxis="y2",
        ))
        fig_abc.add_hline(y=abc_a, line_dash="dash", line_color="#10b981",
                          annotation_text=f"A: {abc_a:.0f}%",
                          annotation_font_color="#10b981", yref="y2")
        fig_abc.add_hline(y=abc_b, line_dash="dash", line_color="#3b82f6",
                          annotation_text=f"A+B: {abc_b:.0f}%",
                          annotation_font_color="#3b82f6", yref="y2")
        fig_abc.update_layout(
            title="ABC-анализ: кривая Парето",
            yaxis=dict(title="Доля, %"),
            yaxis2=dict(title="Кумулятивная доля, %", overlaying="y",
                        side="right", range=[0, 105]),
            height=420, xaxis_tickangle=-45,
            legend=dict(orientation="h", y=-0.18, x=0),
        )
        apply_kibad_theme(fig_abc)

        children = [
            section_header("Результаты ABC-XYZ анализа"),
            html.Div([
                stat_card("Объектов", str(n_items)),
                stat_card("Класс A", f"{n_a} ({n_a/n_items*100:.0f}%)"),
                stat_card("Класс B", f"{n_b} ({n_b/n_items*100:.0f}%)"),
                stat_card("Класс C", f"{n_c} ({n_c/n_items*100:.0f}%)"),
                stat_card("Доля A в объёме", f"{share_a:.1f}%"),
            ], className="kb-stats-grid mb-3"),
        ]

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs_list = [
            dbc.Tab(label="📊 ABC-график", tab_id="abc-t-chart"),
            dbc.Tab(label="📋 Таблица", tab_id="abc-t-table"),
        ]
        tab_contents = {
            "abc-t-chart": dcc.Graph(figure=fig_abc),
            "abc-t-table": data_table(
                agg.rename(columns={"item": item_col, "total": "Сумма",
                                     "share": "Доля,%", "cumshare": "КумДоля,%"}),
                id="abc-tbl", page_size=25,
            ),
        }

        if use_xyz:
            # Cross-matrix heatmap
            matrix = agg.groupby(["ABC", "XYZ"]).size().reset_index(name="count")
            all_abc = ["A", "B", "C"]
            all_xyz = ["X", "Y", "Z"]
            pivot_matrix = matrix.pivot(index="ABC", columns="XYZ", values="count").reindex(
                index=all_abc, columns=all_xyz).fillna(0).astype(int)

            fig_matrix = go.Figure(data=go.Heatmap(
                z=pivot_matrix.values,
                x=[f"XYZ: {v}" for v in pivot_matrix.columns],
                y=[f"ABC: {v}" for v in pivot_matrix.index],
                colorscale="Greens",
                text=pivot_matrix.values, texttemplate="%{text}",
                hovertemplate="ABC=%{y}, XYZ=%{x}<br>Объектов: %{z}<extra></extra>",
            ))
            fig_matrix.update_layout(
                title="Матрица ABC×XYZ (количество объектов)",
                height=340,
                yaxis=dict(autorange="reversed"),
                margin=dict(l=56, r=36, t=48, b=48),
            )
            apply_kibad_theme(fig_matrix)

            # XYZ distribution chart
            xyz_counts = agg["XYZ"].value_counts().reset_index()
            xyz_counts.columns = ["XYZ", "count"]
            fig_xyz = px.bar(xyz_counts, x="XYZ", y="count",
                             color="XYZ",
                             color_discrete_map={"X": "#10b981", "Y": "#f59e0b", "Z": "#ef4444"},
                             title="Распределение по XYZ классам",
                             labels={"XYZ": "Класс", "count": "Объектов"})
            fig_xyz.update_layout(showlegend=False, height=340, margin=dict(l=56, r=36, t=48, b=48))
            apply_kibad_theme(fig_xyz)

            tabs_list += [
                dbc.Tab(label="🔀 Матрица ABC×XYZ", tab_id="abc-t-matrix"),
                dbc.Tab(label="📈 XYZ-распределение", tab_id="abc-t-xyz"),
            ]
            tab_contents["abc-t-matrix"] = dcc.Graph(figure=fig_matrix)
            tab_contents["abc-t-xyz"] = dcc.Graph(figure=fig_xyz)

        # Render tabs inline (store in dcc.Store for tab switching)
        first_tab_id = tabs_list[0].tab_id
        children += [
            dbc.Tabs(tabs_list, id="abc-inner-tabs", active_tab=first_tab_id),
            html.Div(tab_contents.get(first_tab_id, html.Div()), id="abc-inner-content"),
            dcc.Store(id="abc-tab-store",
                      data={k: v.to_plotly_json() if hasattr(v, 'to_plotly_json') else None
                            for k, v in tab_contents.items()}),
        ]

        # Simplify: render everything at once
        tab_divs = []
        for tab_id, content in tab_contents.items():
            tab_divs.append(html.Div(content, id=f"abct-{tab_id}",
                                     style={"display": "block"}))

        return html.Div([
            section_header("Результаты ABC-XYZ анализа"),
            html.Div([
                stat_card("Объектов", str(n_items)),
                stat_card("Класс A", f"{n_a} ({n_a/n_items*100:.0f}%)"),
                stat_card("Класс B", f"{n_b} ({n_b/n_items*100:.0f}%)"),
                stat_card("Класс C", f"{n_c} ({n_c/n_items*100:.0f}%)"),
                stat_card("Доля A в объёме", f"{share_a:.1f}%"),
            ], className="kb-stats-grid mb-3"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_abc), md=12 if not use_xyz else 8),
            ] + ([dbc.Col(dcc.Graph(figure=fig_matrix), md=4)] if use_xyz else []),
                className="mb-3"),
        ] + ([dbc.Row([dbc.Col(dcc.Graph(figure=fig_xyz))])] if use_xyz else []) + [
            section_header("Детальная таблица"),
            data_table(
                agg.rename(columns={"item": item_col, "total": "Сумма",
                                     "share": "Доля%", "cumshare": "КумДоля%"}),
                id="abc-tbl", page_size=30,
            ),
        ])

    except Exception as e:
        logger.exception("ABC-XYZ error")
        return alert_banner(f"Ошибка: {e}", "danger")

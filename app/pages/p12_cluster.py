"""p12_cluster – Clustering page (Dash)."""
import logging

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from app.state import (
    get_df_from_store, get_df_from_stores,
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header
from app.components.table import data_table
from app.components.cards import stat_card
from app.components.alerts import alert_banner
from core.cluster import run_kmeans, run_elbow, cluster_profiles, pca_transform
from core.audit import log_event

logger = logging.getLogger(__name__)

dash.register_page(__name__, path="/cluster", name="12. Кластеризация", order=12, icon="people")

CLUSTER_COLORS = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6",
                  "#06b6d4", "#f97316", "#ec4899", "#84cc16", "#a78bfa"]

layout = html.Div([
    page_header("12. Кластеризация", "K-Means, Метод локтя, PCA-визуализация"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="cl-ds", placeholder="Выберите датасет..."), width=4),
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("Настройки кластеризации"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Числовые признаки", className="kb-label"),
                    dcc.Dropdown(id="cl-features", multi=True, placeholder="Выберите колонки..."),
                ], width=6),
                dbc.Col([
                    html.Label("Кластеров (K)", className="kb-label"),
                    dcc.Input(id="cl-k", type="number", value=3, min=2, max=20,
                              style={"width": "100%"}),
                ], width=2),
                dbc.Col([
                    dbc.Button("Метод локтя", id="cl-elbow-btn", color="secondary",
                               outline=True, className="mt-4 me-2"),
                    dbc.Button("Кластеризовать", id="cl-run-btn", color="primary",
                               className="mt-4"),
                ], width=4),
            ]),
        ]),
    ], className="mb-3"),

    dcc.Loading(html.Div(id="cl-results"), type="circle", color="#10b981"),
])


@callback(
    Output("cl-ds", "options"),
    Output("cl-ds", "value"),
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
    Output("cl-features", "options"),
    Input("cl-ds", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
)
def update_features(ds, datasets, prepared):
    if not ds:
        return []
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return []
    num = df.select_dtypes(include="number").columns.tolist()
    return [{"label": c, "value": c} for c in num]


@callback(
    Output("cl-results", "children"),
    Input("cl-run-btn", "n_clicks"),
    Input("cl-elbow-btn", "n_clicks"),
    State("cl-ds", "value"),
    State("cl-features", "value"),
    State("cl-k", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_clustering(n_run, n_elbow, ds, features, k, datasets, prepared):
    if not ds or not features or len(features) < 2:
        return alert_banner("Выберите минимум 2 числовых признака.", "warning")

    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    try:
        # ── Elbow Method ─────────────────────────────────────────────────────
        if "elbow" in triggered:
            max_k = min(10, len(df) - 1)
            elbow_df = run_elbow(df, features, k_range=range(2, max_k + 1))
            if elbow_df.empty:
                return alert_banner("Недостаточно данных для построения кривой.", "warning")

            best_k = int(elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"])

            # Dual-axis chart: inertia (left) + silhouette (right) on same plot
            fig_elbow = go.Figure()

            fig_elbow.add_trace(go.Scatter(
                x=elbow_df["k"], y=elbow_df["inertia"],
                name="Инерция (SSE)",
                line=dict(color="#3b82f6", width=2),
                mode="lines+markers",
                yaxis="y1",
                hovertemplate="K=%{x}<br>Инерция=%{y:,.1f}<extra></extra>",
            ))

            fig_elbow.add_trace(go.Scatter(
                x=elbow_df["k"], y=elbow_df["silhouette"],
                name="Силуэтный коэффициент",
                line=dict(color="#10b981", width=2, dash="dot"),
                mode="lines+markers",
                yaxis="y2",
                hovertemplate="K=%{x}<br>Силуэт=%{y:.3f}<extra></extra>",
            ))

            fig_elbow.add_vline(
                x=best_k,
                line_dash="dash", line_color="#f59e0b", line_width=2,
                annotation_text=f"Рекомендуемое K={best_k}",
                annotation_position="top right",
                annotation_font_color="#f59e0b",
            )

            fig_elbow.update_layout(
                title="Кривая локтя: инерция и силуэтный коэффициент",
                xaxis=dict(title="Количество кластеров (K)", dtick=1),
                yaxis=dict(title="Инерция (SSE)", title_font=dict(color="#3b82f6")),
                yaxis2=dict(
                    title="Силуэтный коэффициент",
                    title_font=dict(color="#10b981"),
                    overlaying="y", side="right",
                    tickformat=".3f",
                ),
                legend=dict(x=0.01, y=0.99),
                hovermode="x unified",
                height=420,
            )
            apply_kibad_theme(fig_elbow)

            return html.Div([
                section_header("Метод локтя",
                               "Ищите точку 'перегиба' инерции и максимум силуэтного коэффициента"),
                dcc.Graph(figure=fig_elbow),
                dbc.Alert([
                    html.Strong(f"Рекомендуемое K по силуэтному коэффициенту: {best_k}"),
                    html.Br(),
                    html.Small(
                        "Инерция (SSE) — сумма квадратов расстояний до центроидов. "
                        "Ищите точку «локтя», где снижение резко замедляется. "
                        "Силуэтный коэффициент — мера компактности и разделённости (−1…1): "
                        "выбирайте K с максимальным значением."
                    ),
                ], color="success", className="mt-3"),
            ])

        # ── K-Means ───────────────────────────────────────────────────────────
        n_clusters = int(k or 3)
        result = run_kmeans(df, features, n_clusters=n_clusters)
        log_event("cluster", details={"dataset": ds, "k": n_clusters, "features": features})

        sil_val = result.silhouette
        sil_str = f"{sil_val:.3f}" if sil_val == sil_val else "—"

        # Silhouette quality level
        if sil_val == sil_val:
            if sil_val > 0.5:
                sil_quality = "Хорошее разделение"
                sil_color = "success"
            elif sil_val > 0.25:
                sil_quality = "Приемлемое разделение"
                sil_color = "warning"
            else:
                sil_quality = "Слабое разделение — попробуйте другое K"
                sil_color = "danger"
        else:
            sil_quality = ""
            sil_color = "secondary"

        # ── Cluster size bar chart ────────────────────────────────────────────
        size_df = (
            pd.Series(result.labels, name="cluster")
            .value_counts().sort_index().reset_index()
        )
        size_df.columns = ["cluster", "count"]
        size_df["cluster"] = size_df["cluster"].astype(str)

        fig_size = px.bar(
            size_df, x="cluster", y="count",
            color="cluster",
            color_discrete_sequence=CLUSTER_COLORS,
            labels={"cluster": "Кластер", "count": "Количество точек"},
            title="Распределение точек по кластерам",
        )
        fig_size.update_layout(showlegend=False, height=360)
        apply_kibad_theme(fig_size)

        # ── PCA 2D visualization ──────────────────────────────────────────────
        pca_df, explained = pca_transform(df, features, n_components=2)
        valid_idx = result.df_with_labels.dropna(subset=["cluster"]).index
        pca_df = pca_df.loc[pca_df.index.isin(valid_idx)].copy()
        cluster_series = result.df_with_labels.loc[pca_df.index, "cluster"]
        pca_df["cluster"] = cluster_series.astype(str).values

        exp_pct = [f"{v:.1%}" for v in explained]
        x_label = f"PC1 ({exp_pct[0] if exp_pct else ''})"
        y_label = f"PC2 ({exp_pct[1] if len(exp_pct) > 1 else ''})"
        total_exp = sum(explained[:2]) * 100

        fig_pca = px.scatter(
            pca_df, x="pca_1", y="pca_2", color="cluster",
            title=f"PCA-визуализация кластеров (объяснённая дисперсия: {total_exp:.1f}%)",
            labels={"pca_1": x_label, "pca_2": y_label},
            color_discrete_sequence=CLUSTER_COLORS,
        )
        apply_kibad_theme(fig_pca)
        fig_pca.update_traces(marker_size=5, marker_opacity=0.75)

        # ── Radar chart (normalized cluster profiles) ─────────────────────────
        profiles = cluster_profiles(result)
        mean_cols = [c for c in profiles.columns if c.endswith("_mean")]
        feature_names = [c.replace("_mean", "") for c in mean_cols]

        norm_df = profiles[mean_cols].copy()
        norm_df.columns = feature_names
        for col in norm_df.columns:
            col_min, col_max = norm_df[col].min(), norm_df[col].max()
            if col_max > col_min:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.5

        fig_radar = go.Figure()
        for idx, cluster_id in enumerate(norm_df.index):
            vals = norm_df.loc[cluster_id].tolist()
            cats = feature_names + [feature_names[0]]
            vals_closed = vals + [vals[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats,
                fill="toself",
                name=f"Кластер {cluster_id}",
                line=dict(color=CLUSTER_COLORS[idx % len(CLUSTER_COLORS)]),
                opacity=0.7,
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Нормализованные профили кластеров (0–1)",
            height=400,
        )
        apply_kibad_theme(fig_radar)

        # ── Centroid heatmap ──────────────────────────────────────────────────
        centers = result.centers_df.copy()
        centers.index = [f"Кластер {i}" for i in centers.index]

        fig_heat = go.Figure(data=go.Heatmap(
            z=centers.values,
            x=centers.columns.tolist(),
            y=centers.index.tolist(),
            colorscale="RdBu_r",
            text=np.round(centers.values, 2),
            texttemplate="%{text}",
            hovertemplate="Кластер: %{y}<br>Признак: %{x}<br>Значение: %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Центроиды кластеров",
            xaxis_title="Признак",
            yaxis_title="Кластер",
            height=max(280, 60 + n_clusters * 55),
        )
        apply_kibad_theme(fig_heat)

        # ── Profiles table ────────────────────────────────────────────────────
        profiles_display = profiles.reset_index()

        return html.Div([
            section_header("Результаты кластеризации"),

            # Stat cards
            html.Div([
                stat_card("Кластеров", str(n_clusters)),
                stat_card("Объектов", f"{len(result.labels):,}"),
                stat_card("Признаков", str(len(features))),
                stat_card("Силуэт", sil_str),
            ], className="kb-stats-grid"),

            # Silhouette quality
            dbc.Alert(sil_quality, color=sil_color, className="mt-2 mb-3") if sil_quality else html.Div(),

            # Cluster size + PCA side by side
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_size), md=5),
                dbc.Col(dcc.Graph(figure=fig_pca), md=7),
            ], className="mb-3"),

            # Radar + Heatmap
            section_header("Профили кластеров", "Сравнение нормализованных характеристик по кластерам"),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_radar), md=6),
                dbc.Col(dcc.Graph(figure=fig_heat), md=6),
            ], className="mb-3"),

            # Profiles table
            section_header("Сводная таблица", "Средние и стандартные отклонения признаков по кластерам"),
            data_table(profiles_display, id="cl-profiles-tbl"),
        ])

    except ValueError as e:
        return alert_banner(f"Ошибка входных данных: {e}", "warning")
    except Exception as e:
        logger.exception("Ошибка кластеризации")
        return alert_banner(f"Не удалось выполнить кластеризацию: {e}", "danger")

"""p12_cluster – Clustering page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px

from app.state import (
    get_df_from_store, save_dataframe,
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.cards import stat_card
from app.components.alerts import alert_banner
from core.cluster import run_kmeans, run_elbow, cluster_profiles, pca_transform
from core.audit import log_event

dash.register_page(__name__, path="/cluster", name="12. Кластеризация", order=12, icon="people")

layout = html.Div([
    page_header("12. Кластеризация", "K-Means, Elbow Method, PCA-визуализация"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="cl-ds", placeholder="Выберите датасет..."), width=4),
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("Настройки кластеризации"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Числовые признаки", className="kb-stat-label"),
                    dcc.Dropdown(id="cl-features", multi=True, placeholder="Выберите колонки..."),
                ], width=6),
                dbc.Col([
                    html.Label("Количество кластеров (K)", className="kb-stat-label"),
                    dcc.Input(id="cl-k", type="number", value=3, min=2, max=20, style={"width": "100%"}),
                ], width=2),
                dbc.Col([
                    dbc.Button("Elbow", id="cl-elbow-btn", color="secondary", outline=True, className="mt-4 me-2"),
                    dbc.Button("Кластеризовать", id="cl-run-btn", color="primary", className="mt-4"),
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
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_features(ds, datasets, prepared):
    if not ds:
        return []
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
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

    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    ctx = dash.callback_context
    triggered = ctx.triggered[0]["prop_id"] if ctx.triggered else ""

    try:
        if "elbow" in triggered:
            elbow_df = run_elbow(df, features, max_k=min(10, len(df) - 1))
            fig = px.line(elbow_df, x="k", y="inertia", markers=True,
                          title="Метод локтя (Elbow Method)")
            apply_kibad_theme(fig)
            return html.Div([section_header("Метод локтя"), dcc.Graph(figure=fig)])

        # K-Means
        result_df, labels = run_kmeans(df, features, n_clusters=int(k or 3))
        log_event("cluster", dataset=ds, details=f"kmeans k={k} features={features}")

        profiles = cluster_profiles(result_df, features, label_col="cluster")

        # PCA 2D visualization
        pca_df = pca_transform(df, features, n_components=2)
        pca_df["cluster"] = labels.astype(str)
        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="cluster",
                             title="PCA-визуализация кластеров")
        apply_kibad_theme(fig_pca)

        return html.Div([
            section_header("Результаты кластеризации"),
            html.Div([
                stat_card("Кластеров", str(int(k or 3))),
                stat_card("Объектов", f"{len(result_df):,}"),
                stat_card("Признаков", str(len(features))),
            ], className="kb-stats-grid"),
            dcc.Graph(figure=fig_pca),
            html.H4("Профили кластеров", className="mt-3"),
            data_table(profiles, id="cl-profiles-tbl"),
        ])

    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")

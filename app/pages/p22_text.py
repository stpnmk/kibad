"""p22_text – Text analytics page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from collections import Counter
import re

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.alerts import alert_banner

dash.register_page(__name__, path="/text", name="22. Текст", order=22, icon="card-text")

layout = html.Div([
    page_header("22. Текстовая аналитика", "Частотный анализ, биграммы, поиск ключевых слов"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="txt-ds", placeholder="Выберите датасет..."), width=4),
        dbc.Col(dcc.Dropdown(id="txt-col", placeholder="Текстовая колонка..."), width=4),
    ], className="mb-3"),

    dbc.Tabs(id="txt-tabs", active_tab="tab-freq", children=[
        dbc.Tab(label="Частоты слов", tab_id="tab-freq"),
        dbc.Tab(label="Биграммы", tab_id="tab-bigram"),
        dbc.Tab(label="Поиск по ключевым словам", tab_id="tab-search"),
    ]),
    dcc.Loading(html.Div(id="txt-content"), type="circle", color="#10b981"),
])


@callback(
    Output("txt-ds", "options"),
    Output("txt-ds", "value"),
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
    Output("txt-col", "options"),
    Input("txt-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_cols(ds, datasets, prepared):
    if not ds:
        return []
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return []
    text_cols = df.select_dtypes(include="object").columns.tolist()
    return [{"label": c, "value": c} for c in text_cols]


@callback(
    Output("txt-content", "children"),
    Input("txt-tabs", "active_tab"),
    Input("txt-col", "value"),
    State("txt-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_tab(tab, col, ds, datasets, prepared):
    if not ds or not col:
        return empty_state("", "Выберите датасет и текстовую колонку", "")

    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    texts = df[col].dropna().astype(str).tolist()
    if not texts:
        return alert_banner("Колонка пуста.", "info")

    all_words = []
    for t in texts:
        words = re.findall(r'\b\w+\b', t.lower())
        all_words.extend(words)

    # Filter stop words (basic Russian + English)
    _STOP = {"и", "в", "на", "с", "по", "не", "что", "как", "для", "это",
             "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of"}
    all_words = [w for w in all_words if w not in _STOP and len(w) > 2]

    if tab == "tab-freq":
        freq = Counter(all_words).most_common(30)
        freq_df = pd.DataFrame(freq, columns=["Слово", "Частота"])
        fig = px.bar(freq_df, x="Частота", y="Слово", orientation="h",
                     title="Топ-30 слов по частоте")
        apply_kibad_theme(fig)
        fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
        return html.Div([section_header("Частотный анализ"), dcc.Graph(figure=fig)])

    elif tab == "tab-bigram":
        bigrams = []
        for t in texts:
            words = [w for w in re.findall(r'\b\w+\b', t.lower()) if w not in _STOP and len(w) > 2]
            for i in range(len(words) - 1):
                bigrams.append(f"{words[i]} {words[i+1]}")
        freq = Counter(bigrams).most_common(20)
        bg_df = pd.DataFrame(freq, columns=["Биграмма", "Частота"])
        fig = px.bar(bg_df, x="Частота", y="Биграмма", orientation="h",
                     title="Топ-20 биграмм")
        apply_kibad_theme(fig)
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        return html.Div([section_header("Биграммы"), dcc.Graph(figure=fig)])

    elif tab == "tab-search":
        return html.Div([
            section_header("Поиск по ключевым словам"),
            dcc.Input(id="txt-search-input", placeholder="Введите слово для поиска...",
                      style={"width": "100%", "marginBottom": "12px"}),
            dbc.Button("Найти", id="txt-search-btn", color="primary"),
            html.Div(id="txt-search-result"),
        ])

    return ""


@callback(
    Output("txt-search-result", "children"),
    Input("txt-search-btn", "n_clicks"),
    State("txt-search-input", "value"),
    State("txt-col", "value"),
    State("txt-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def search_text(n, query, col, ds, datasets, prepared):
    if not query or not ds or not col:
        return ""
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return ""
    mask = df[col].astype(str).str.contains(query, case=False, na=False)
    matches = df[mask]
    return html.Div([
        alert_banner(f"Найдено {len(matches)} строк с «{query}»", "info"),
        data_table(matches.head(50), id="txt-search-tbl"),
    ])

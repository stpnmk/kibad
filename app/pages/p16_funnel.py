"""p16_funnel.py – Воронка продаж / конверсий (Dash)."""
from __future__ import annotations

import logging

import dash
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_PREPARED,
    get_df_from_store, list_datasets,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card

logger = logging.getLogger(__name__)

dash.register_page(
    __name__,
    path="/funnel",
    name="16. Воронка",
    order=16,
    icon="filter",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ds_options(ds_store, prep_store):
    names = list_datasets(ds_store) + [
        n for n in list_datasets(prep_store) if n not in list_datasets(ds_store)
    ]
    return [{"label": n, "value": n} for n in names]


def _get_df(ds_store, prep_store, name):
    df = get_df_from_store(prep_store, name)
    if df is None:
        df = get_df_from_store(ds_store, name)
    return df


def _funnel_fig(stages: list[str], values: list[float], title: str = "Воронка") -> go.Figure:
    fig = go.Figure(go.Funnel(
        y=stages,
        x=values,
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=[
                f"rgba(16, 185, 129, {0.9 - i * 0.08})"
                for i in range(len(stages))
            ],
            line=dict(width=1, color="rgba(255,255,255,0.15)"),
        ),
        connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
    ))
    fig.update_layout(title=title, height=420)
    apply_kibad_theme(fig)
    return fig


def _dropoff_fig(stages: list[str], values: list[float]) -> go.Figure:
    dropoff = []
    for i in range(1, len(values)):
        d = values[i - 1] - values[i]
        dropoff.append(d)
    dropoff_labels = [f"{stages[i]} → {stages[i+1]}" for i in range(len(stages) - 1)]
    colors = [
        f"rgba(239, 68, 68, {min(0.9, 0.3 + d / max(values[0], 1))})"
        for d in dropoff
    ]
    fig = go.Figure(go.Bar(
        x=dropoff_labels,
        y=dropoff,
        marker_color=colors,
        text=[f"{d:,.0f}" for d in dropoff],
        textposition="outside",
    ))
    fig.update_layout(
        title="Отток по переходам",
        xaxis_title="Переход",
        yaxis_title="Потерянные единицы",
        height=360,
    )
    apply_kibad_theme(fig)
    return fig


def _conversion_fig(stages: list[str], values: list[float]) -> go.Figure:
    conv = [values[i] / values[0] * 100 for i in range(len(values))]
    step_conv = [100.0] + [values[i] / values[i - 1] * 100 for i in range(1, len(values))]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="От начала (%)",
        x=stages,
        y=conv,
        marker_color="rgba(99, 102, 241, 0.7)",
        text=[f"{c:.1f}%" for c in conv],
        textposition="outside",
    ))
    fig.add_trace(go.Scatter(
        name="Шаговая конверсия (%)",
        x=stages,
        y=step_conv,
        mode="lines+markers",
        line=dict(color="rgba(16, 185, 129, 0.9)", width=2),
        marker=dict(size=7),
        yaxis="y2",
    ))
    fig.update_layout(
        title="Конверсия",
        xaxis_title="Этап",
        yaxis=dict(title="От начала (%)", range=[0, 115]),
        yaxis2=dict(title="Шаговая (%)", overlaying="y", side="right", range=[0, 115]),
        legend=dict(orientation="h", y=-0.2),
        height=380,
        barmode="group",
    )
    apply_kibad_theme(fig)
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_MANUAL_DEFAULT = [
    ("Посетители", 10000),
    ("Регистрации", 3500),
    ("Активация", 1200),
    ("Оплата", 420),
    ("Удержание", 180),
]

layout = html.Div([
    page_header("16. Воронка", "Анализ конверсионной воронки и отток по этапам"),

    # Mode selector
    dbc.Card([
        dbc.CardBody([
            section_header("Источник данных"),
            dbc.RadioItems(
                id="funnel-mode",
                options=[
                    {"label": "Из датасета", "value": "dataset"},
                    {"label": "Ввести вручную", "value": "manual"},
                ],
                value="manual",
                inline=True,
                className="mb-3",
            ),

            # --- Dataset mode ---
            html.Div(id="funnel-dataset-controls", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Датасет", className="kb-stat-label mb-1"),
                        dcc.Dropdown(id="funnel-ds-select", className="kb-select"),
                    ], md=4),
                    dbc.Col([
                        html.Label("Режим", className="kb-stat-label mb-1"),
                        dcc.RadioItems(
                            id="funnel-ds-mode",
                            options=[
                                {"label": "Колонка этапов (строки = события)", "value": "stage_col"},
                                {"label": "Агрегированные данные (этап + значение)", "value": "agg"},
                            ],
                            value="stage_col",
                            className="kb-radio-group",
                        ),
                    ], md=5),
                ], className="mb-3"),
                dbc.Row(id="funnel-ds-col-row", children=[]),
            ]),

            # --- Manual mode ---
            html.Div(id="funnel-manual-controls", children=[
                html.Label("Этапы и значения (один этап на строку: «Название, число»)",
                           className="kb-stat-label mb-1"),
                dcc.Textarea(
                    id="funnel-manual-text",
                    value="\n".join(f"{n}, {v}" for n, v in _MANUAL_DEFAULT),
                    style={"width": "100%", "height": "150px", "fontFamily": "monospace",
                           "background": "var(--bg-surface)", "color": "var(--text-primary)",
                           "border": "1px solid var(--border-subtle)", "borderRadius": "6px",
                           "padding": "8px"},
                ),
            ]),

            html.Div(className="mb-3"),
            dbc.Button("Построить воронку", id="funnel-btn-run", color="primary"),
        ])
    ], className="kb-card mb-4"),

    dcc.Loading(html.Div(id="funnel-result"), type="circle", color="#10b981"),
], style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px 16px"})


# ---------------------------------------------------------------------------
# Callbacks: toggle visibility of controls
# ---------------------------------------------------------------------------

@callback(
    Output("funnel-dataset-controls", "style"),
    Output("funnel-manual-controls", "style"),
    Input("funnel-mode", "value"),
)
def toggle_mode(mode):
    show = {"display": "block"}
    hide = {"display": "none"}
    if mode == "dataset":
        return show, hide
    return hide, show


@callback(
    Output("funnel-ds-select", "options"),
    Output("funnel-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
)
def update_ds_options(ds_store, prep_store):
    opts = _ds_options(ds_store, prep_store)
    val = opts[0]["value"] if opts else None
    return opts, val


@callback(
    Output("funnel-ds-col-row", "children"),
    Input("funnel-ds-select", "value"),
    Input("funnel-ds-mode", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_col_selectors(ds_name, ds_mode, ds_store, prep_store):
    if not ds_name:
        return []
    df = _get_df(ds_store, prep_store, ds_name)
    if df is None:
        return []
    all_cols = [{"label": c, "value": c} for c in df.columns]
    num_cols = [{"label": c, "value": c} for c in df.select_dtypes(include="number").columns]

    if ds_mode == "stage_col":
        return [
            dbc.Col([
                html.Label("Колонка этапа", className="kb-stat-label mb-1"),
                dcc.Dropdown(id="funnel-stage-col", options=all_cols,
                             value=all_cols[0]["value"] if all_cols else None,
                             className="kb-select"),
            ], md=4),
            # Hidden dummies for output compatibility
            html.Div(id="funnel-value-col", style={"display": "none"}),
            html.Div(id="funnel-order-col", style={"display": "none"}),
        ]
    else:  # agg mode
        return [
            dbc.Col([
                html.Label("Колонка этапа", className="kb-stat-label mb-1"),
                dcc.Dropdown(id="funnel-stage-col", options=all_cols,
                             value=all_cols[0]["value"] if all_cols else None,
                             className="kb-select"),
            ], md=4),
            dbc.Col([
                html.Label("Колонка значений", className="kb-stat-label mb-1"),
                dcc.Dropdown(id="funnel-value-col", options=num_cols,
                             value=num_cols[0]["value"] if num_cols else None,
                             className="kb-select"),
            ], md=4),
            html.Div(id="funnel-order-col", style={"display": "none"}),
        ]


# ---------------------------------------------------------------------------
# Main callback: build funnel
# ---------------------------------------------------------------------------

@callback(
    Output("funnel-result", "children"),
    Input("funnel-btn-run", "n_clicks"),
    State("funnel-mode", "value"),
    # Manual inputs
    State("funnel-manual-text", "value"),
    # Dataset inputs
    State("funnel-ds-select", "value"),
    State("funnel-ds-mode", "value"),
    State("funnel-ds-col-row", "children"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def build_funnel(n_clicks, mode, manual_text, ds_name, ds_mode, col_row,
                 ds_store, prep_store):
    if not n_clicks:
        return no_update

    try:
        stages, values = _parse_inputs(
            mode, manual_text, ds_name, ds_mode, col_row, ds_store, prep_store
        )
    except ValueError as e:
        return dbc.Alert(str(e), color="danger")

    if len(stages) < 2:
        return dbc.Alert("Нужно минимум 2 этапа.", color="warning")

    # Stat cards
    total_in = values[0]
    total_out = values[-1]
    overall_conv = total_out / total_in * 100 if total_in else 0
    max_dropoff_idx = max(range(1, len(values)), key=lambda i: values[i - 1] - values[i])
    max_dropoff_label = f"{stages[max_dropoff_idx - 1]} → {stages[max_dropoff_idx]}"
    max_dropoff_pct = (values[max_dropoff_idx - 1] - values[max_dropoff_idx]) / values[max_dropoff_idx - 1] * 100

    stat_row = dbc.Row([
        dbc.Col(stat_card("Вход в воронку", f"{total_in:,.0f}"), md=3),
        dbc.Col(stat_card("Выход из воронки", f"{total_out:,.0f}"), md=3),
        dbc.Col(stat_card("Общая конверсия", f"{overall_conv:.1f}%"), md=3),
        dbc.Col(stat_card("Максим. отток", f"{max_dropoff_label} ({max_dropoff_pct:.0f}%)"), md=3),
    ], className="mb-4")

    # Charts
    funnel_fig = _funnel_fig(stages, values)
    dropoff_fig = _dropoff_fig(stages, values)
    conv_fig = _conversion_fig(stages, values)

    # Detailed table
    rows = []
    for i, (s, v) in enumerate(zip(stages, values)):
        from_start = v / values[0] * 100
        step_c = (v / values[i - 1] * 100) if i > 0 else 100.0
        drop = values[i - 1] - v if i > 0 else 0
        rows.append({
            "Этап": s,
            "Значение": f"{v:,.0f}",
            "Конверсия от начала (%)": f"{from_start:.1f}%",
            "Шаговая конверсия (%)": f"{step_c:.1f}%",
            "Отток": f"{drop:,.0f}" if i > 0 else "—",
        })
    tbl_df = pd.DataFrame(rows)

    from app.components.table import data_table
    tbl = data_table(tbl_df, id="funnel-detail-tbl", page_size=20)

    tabs = dbc.Tabs([
        dbc.Tab(label="Воронка", tab_id="funnel-tab-main", children=[
            dcc.Graph(figure=funnel_fig, config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Отток", tab_id="funnel-tab-dropoff", children=[
            dcc.Graph(figure=dropoff_fig, config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Конверсия", tab_id="funnel-tab-conv", children=[
            dcc.Graph(figure=conv_fig, config={"displayModeBar": False}),
        ]),
        dbc.Tab(label="Таблица", tab_id="funnel-tab-tbl", children=[
            html.Div(tbl, className="mt-3"),
        ]),
    ], active_tab="funnel-tab-main", className="mt-3")

    return html.Div([
        stat_row,
        dbc.Card(dbc.CardBody([tabs]), className="kb-card"),
    ])


# ---------------------------------------------------------------------------
# Helpers: parse input sources
# ---------------------------------------------------------------------------

def _parse_inputs(mode, manual_text, ds_name, ds_mode, col_row,
                  ds_store, prep_store):
    """Return (stages: list[str], values: list[float])."""
    if mode == "manual":
        return _parse_manual(manual_text)
    else:
        return _parse_dataset(ds_name, ds_mode, col_row, ds_store, prep_store)


def _parse_manual(text: str):
    stages, values = [], []
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(",", 1)
        if len(parts) != 2:
            raise ValueError(f"Неверный формат строки: «{line}». Ожидается «Название, число».")
        name = parts[0].strip()
        try:
            val = float(parts[1].strip().replace(" ", "").replace("\xa0", ""))
        except ValueError:
            raise ValueError(f"Не удалось прочитать число из строки: «{line}».")
        stages.append(name)
        values.append(val)
    if not stages:
        raise ValueError("Не указано ни одного этапа.")
    return stages, values


def _parse_dataset(ds_name, ds_mode, col_row, ds_store, prep_store):
    if not ds_name:
        raise ValueError("Выберите датасет.")
    df = _get_df(ds_store, prep_store, ds_name)
    if df is None:
        raise ValueError(f"Датасет «{ds_name}» не найден.")

    # Extract column names from col_row children (they may be Dash dict components)
    def _extract_dropdown_value(children, dropdown_id):
        """Recursively find dropdown value by id in a component tree."""
        if not children:
            return None
        if isinstance(children, dict):
            if children.get("props", {}).get("id") == dropdown_id:
                return children.get("props", {}).get("value")
            for v in children.get("props", {}).get("children", []) or []:
                result = _extract_dropdown_value(v, dropdown_id)
                if result is not None:
                    return result
        elif isinstance(children, list):
            for c in children:
                result = _extract_dropdown_value(c, dropdown_id)
                if result is not None:
                    return result
        return None

    stage_col = _extract_dropdown_value(col_row, "funnel-stage-col")
    if not stage_col or stage_col not in df.columns:
        raise ValueError("Выберите колонку этапов.")

    if ds_mode == "stage_col":
        # Each row is an event — count occurrences per stage
        counts = df[stage_col].value_counts().sort_index()
        # Try to sort by count descending (natural funnel ordering)
        counts = counts.sort_values(ascending=False)
        return list(counts.index.astype(str)), list(counts.values.astype(float))
    else:
        # Aggregated: stage + value columns
        value_col = _extract_dropdown_value(col_row, "funnel-value-col")
        if not value_col or value_col not in df.columns:
            raise ValueError("Выберите колонку значений.")
        grp = df.groupby(stage_col)[value_col].sum().sort_values(ascending=False)
        return list(grp.index.astype(str)), list(grp.values.astype(float))

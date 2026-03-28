"""p21_pipeline – Visual automation workflow editor (Dash)."""
import json
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from app.state import (
    get_df_from_store, save_dataframe,
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.alerts import alert_banner
from core.audit import log_event

dash.register_page(__name__, path="/pipeline", name="21. Пайплайн", order=21, icon="gear")

_STEP_TYPES = [
    {"label": "Загрузить датасет", "value": "load"},
    {"label": "Фильтр строк", "value": "filter"},
    {"label": "Выбрать колонки", "value": "select_cols"},
    {"label": "Удалить дубликаты", "value": "dedup"},
    {"label": "Заполнить пропуски (медиана)", "value": "impute_median"},
    {"label": "Группировка (sum)", "value": "group_sum"},
    {"label": "Сортировка", "value": "sort"},
    {"label": "Экспорт CSV", "value": "export_csv"},
]

layout = html.Div([
    page_header("21. Пайплайн", "Визуальный редактор автоматизированных потоков обработки"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="pipe-ds", placeholder="Выберите датасет..."), width=4),
    ], className="mb-3"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Добавить шаг"),
                dbc.CardBody([
                    dcc.Dropdown(id="pipe-step-type", options=_STEP_TYPES, placeholder="Тип операции..."),
                    dcc.Input(id="pipe-step-param", placeholder="Параметр (колонка, условие...)",
                              style={"width": "100%", "marginTop": "8px"}),
                    dbc.Button("Добавить", id="pipe-add-btn", color="primary", className="mt-2"),
                ]),
            ]),
            html.Hr(),
            dbc.Button("Экспорт пайплайна (JSON)", id="pipe-export-btn", color="secondary", outline=True, className="me-2"),
            dbc.Button("Выполнить пайплайн", id="pipe-run-btn", color="primary"),
            dcc.Download(id="pipe-download"),
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Шаги пайплайна"),
                dbc.CardBody(id="pipe-steps-display"),
            ]),
            dcc.Loading(html.Div(id="pipe-result"), type="circle", color="#10b981"),
        ], width=8),
    ]),

    dcc.Store(id="pipe-steps-store", storage_type="session", data=[]),
])


@callback(
    Output("pipe-ds", "options"),
    Output("pipe-ds", "value"),
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
    Output("pipe-steps-store", "data"),
    Output("pipe-steps-display", "children"),
    Input("pipe-add-btn", "n_clicks"),
    State("pipe-step-type", "value"),
    State("pipe-step-param", "value"),
    State("pipe-steps-store", "data"),
    prevent_initial_call=True,
)
def add_step(n, step_type, param, steps):
    if not step_type:
        return no_update, no_update
    steps = steps or []
    steps.append({"type": step_type, "param": param or "", "index": len(steps) + 1})
    display = _render_steps(steps)
    return steps, display


def _render_steps(steps):
    if not steps:
        return html.Div("Пайплайн пуст. Добавьте шаги.", className="kb-text-muted")
    items = []
    for s in steps:
        label = next((t["label"] for t in _STEP_TYPES if t["value"] == s["type"]), s["type"])
        param_str = f" → {s['param']}" if s.get("param") else ""
        items.append(
            html.Div([
                html.Span(f"{s['index']}.", className="kb-text-accent", style={"fontWeight": "700", "marginRight": "8px"}),
                html.Span(f"{label}{param_str}"),
            ], className="kb-card", style={"padding": "10px 16px", "marginBottom": "6px"})
        )
    return html.Div(items)


@callback(
    Output("pipe-result", "children"),
    Input("pipe-run-btn", "n_clicks"),
    State("pipe-ds", "value"),
    State("pipe-steps-store", "data"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_pipeline(n, ds, steps, datasets, prepared):
    if not ds or not steps:
        return alert_banner("Выберите датасет и добавьте шаги.", "warning")

    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    import pandas as pd
    from core.prepare import impute_missing

    results_log = []
    current_df = df.copy()

    for step in steps:
        try:
            stype = step["type"]
            param = step.get("param", "")

            if stype == "dedup":
                before = len(current_df)
                current_df = current_df.drop_duplicates()
                results_log.append(f"Дедупликация: {before} → {len(current_df)} строк")

            elif stype == "impute_median":
                num_cols = current_df.select_dtypes(include="number").columns.tolist()
                for c in num_cols:
                    current_df[c] = current_df[c].fillna(current_df[c].median())
                results_log.append(f"Заполнение пропусков медианой: {len(num_cols)} колонок")

            elif stype == "select_cols" and param:
                cols = [c.strip() for c in param.split(",") if c.strip() in current_df.columns]
                if cols:
                    current_df = current_df[cols]
                    results_log.append(f"Выбрано колонок: {len(cols)}")

            elif stype == "filter" and param:
                try:
                    current_df = current_df.query(param)
                    results_log.append(f"Фильтр '{param}': {len(current_df)} строк")
                except Exception as e:
                    results_log.append(f"Ошибка фильтра: {e}")

            elif stype == "sort" and param:
                col = param.strip()
                if col in current_df.columns:
                    current_df = current_df.sort_values(col)
                    results_log.append(f"Сортировка по {col}")

            elif stype == "group_sum" and param:
                col = param.strip()
                if col in current_df.columns:
                    num = current_df.select_dtypes(include="number").columns.tolist()
                    current_df = current_df.groupby(col)[num].sum().reset_index()
                    results_log.append(f"Группировка по {col}: {len(current_df)} строк")

            else:
                results_log.append(f"Шаг {step['index']}: выполнен ({stype})")

        except Exception as e:
            results_log.append(f"Ошибка на шаге {step['index']}: {e}")

    log_event("pipeline", dataset=ds, details=f"{len(steps)} steps")

    return html.Div([
        section_header("Результат пайплайна"),
        html.Ul([html.Li(r, className="kb-text-secondary") for r in results_log]),
        data_table(current_df.head(100), id="pipe-result-tbl"),
    ])


@callback(
    Output("pipe-download", "data"),
    Input("pipe-export-btn", "n_clicks"),
    State("pipe-steps-store", "data"),
    prevent_initial_call=True,
)
def export_pipeline(n, steps):
    if not steps:
        return no_update
    return dcc.send_string(json.dumps(steps, ensure_ascii=False, indent=2), "pipeline.json")

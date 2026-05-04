"""p08_attribution – Factor Attribution / Decomposition page (Dash)."""
import logging

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

from app.state import (
    get_df_from_store, get_df_from_stores, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS, STORE_ATTRIBUTION,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.alerts import alert_banner
from core.attribution import (
    additive_attribution, multiplicative_attribution,
    regression_attribution, shapley_attribution,
    waterfall_data, AttributionResult,
)
from core.explore import plot_waterfall
from core.i18n import t
from core.audit import log_event

dash.register_page(__name__, path="/attribution", name="8. Атрибуция", order=8, icon="pie-chart")


def _kpi_card(label: str, value: str) -> html.Div:
    return html.Div([
        html.Div(label, style={"fontSize": "0.72rem", "fontWeight": "600",
                               "textTransform": "uppercase", "letterSpacing": "0.08em",
                               "color": "var(--text-muted)", "marginBottom": "4px"}),
        html.Div(value, style={"fontSize": "1.4rem", "fontWeight": "700",
                               "color": "var(--text-primary)"}),
    ], style={
        "background": "var(--bg-card)",
        "border": "1px solid var(--border-subtle)",
        "borderRadius": "10px",
        "padding": "16px 20px",
    })

_METHODS = {
    "additive": "Аддитивный (линейный)",
    "multiplicative": "Мультипликативный (ratio)",
    "regression": "Регрессионный",
    "shapley": "Шэпли-аппроксимация",
}

layout = html.Div([
    page_header("8. Факторный анализ", "Декомпозиция вклада факторов в изменение показателя"),
    dbc.Accordion([
        dbc.AccordionItem([
            dcc.Markdown("""
Каждая строка — один сегмент/период. Для каждого драйвера нужны две колонки: текущий и предыдущий период.

| сегмент | выручка_тек | выручка_пред | стоимость_тек | стоимость_пред |
|---|---|---|---|---|
| Север | 12 500 000 | 11 200 000 | 7 800 000 | 6 900 000 |

Используйте суффиксы `_тек`/`_пред` или `_current`/`_last`.
            """, style={"color": "#9ba3b8"}),
        ], title="Ожидаемый формат данных"),
    ], start_collapsed=True, className="mb-3"),

    dbc.Row([
        dbc.Col(dcc.Dropdown(id="attr-ds", placeholder="Выберите датасет..."), width=4),
    ], className="mb-3"),

    dbc.Card([
        dbc.CardHeader("Настройки декомпозиции"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Целевой показатель (текущий)", className="kb-stat-label"),
                    dcc.Dropdown(id="attr-target", placeholder="Целевая колонка..."),
                ], width=3),
                dbc.Col([
                    html.Label("Целевой (предыдущий)", className="kb-stat-label"),
                    dcc.Dropdown(id="attr-target-prev", placeholder="Предыдущее значение..."),
                ], width=3),
                dbc.Col([
                    html.Label("Факторы-драйверы (текущие)", className="kb-stat-label"),
                    dcc.Dropdown(id="attr-drivers", multi=True, placeholder="Драйверы..."),
                ], width=3),
                dbc.Col([
                    html.Label("Факторы (предыдущие)", className="kb-stat-label"),
                    dcc.Dropdown(id="attr-drivers-prev", multi=True, placeholder="Предыдущие значения..."),
                ], width=3),
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Метод декомпозиции", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="attr-method",
                        options=[{"label": v, "value": k} for k, v in _METHODS.items()],
                        value="additive",
                    ),
                ], width=4),
                dbc.Col([
                    dbc.Button("Рассчитать", id="attr-run-btn", color="primary", className="mt-4"),
                ], width=4),
            ], className="mt-3"),
        ]),
    ], className="mb-3"),
    dcc.Loading(html.Div(id="attr-results"), type="circle", color="#10b981"),
])


@callback(
    Output("attr-ds", "options"),
    Output("attr-ds", "value"),
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
    Output("attr-target", "options"),
    Output("attr-target-prev", "options"),
    Output("attr-drivers", "options"),
    Output("attr-drivers-prev", "options"),
    Input("attr-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_cols(ds, datasets, prepared):
    if not ds:
        return [], [], [], []
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return [], [], [], []
    num = df.select_dtypes(include="number").columns.tolist()
    opts = [{"label": c, "value": c} for c in num]
    return opts, opts, opts, opts


@callback(
    Output("attr-results", "children"),
    Output(STORE_ATTRIBUTION, "data"),
    Input("attr-run-btn", "n_clicks"),
    State("attr-ds", "value"),
    State("attr-target", "value"),
    State("attr-target-prev", "value"),
    State("attr-drivers", "value"),
    State("attr-drivers-prev", "value"),
    State("attr-method", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State(STORE_ATTRIBUTION, "data"),
    prevent_initial_call=True,
)
def run_attribution(n, ds, target, target_prev, drivers, drivers_prev, method,
                    datasets, prepared, attr_store):
    if not all([ds, target, target_prev, drivers, drivers_prev]):
        return alert_banner("Заполните все поля.", "warning"), no_update

    if len(drivers) != len(drivers_prev):
        return alert_banner("Количество текущих и предыдущих факторов должно совпадать.", "warning"), no_update

    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return alert_banner("Датасет не найден.", "danger"), no_update

    try:
        func_map = {
            "additive": additive_attribution,
            "multiplicative": multiplicative_attribution,
            "regression": regression_attribution,
            "shapley": shapley_attribution,
        }
        func = func_map[method]
        result: AttributionResult = func(
            df, target_col=target, target_prev_col=target_prev,
            driver_cols=drivers, driver_prev_cols=drivers_prev,
        )
        log_event("attribution", dataset=ds, details=f"method={method}")

        attr_store = attr_store or []
        attr_store.append({"dataset": ds, "method": method, "target": target})

        contrib_df = result.contributions
        total_delta = result.target_delta
        residual = result.residual
        explained = contrib_df["contribution"].abs().sum()
        explained_pct = (explained / abs(total_delta) * 100) if total_delta else 0.0

        # ── KPI row ──────────────────────────────────────────────────────────
        arrow = "▲" if total_delta >= 0 else "▼"
        kpi_row = dbc.Row([
            dbc.Col(_kpi_card("Изменение целевого", f"{arrow} {total_delta:+,.2f}"), md=3),
            dbc.Col(_kpi_card("Объяснено", f"{explained_pct:.1f}%"), md=3),
            dbc.Col(_kpi_card("Метод", _METHODS.get(method, method)), md=3),
            dbc.Col(_kpi_card("Остаток", f"{residual:+,.2f}"), md=3),
        ], className="mb-4")

        # ── Horizontal bar — contributions ────────────────────────────────
        contrib_sorted = contrib_df.sort_values("contribution", key=lambda s: s.abs(), ascending=True)
        bar_colors = ["#ef4444" if v < 0 else "#4f8ef7" for v in contrib_sorted["contribution"]]
        fig_bar = go.Figure(go.Bar(
            y=contrib_sorted["driver"],
            x=contrib_sorted["contribution"],
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v:+,.2f}" for v in contrib_sorted["contribution"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="Вклад факторов",
            xaxis_title="Вклад",
            height=max(300, len(contrib_df) * 40 + 80),
            margin=dict(l=10, r=60, t=48, b=36),
        )
        apply_kibad_theme(fig_bar)

        # ── Donut — share of absolute contribution ────────────────────────
        abs_contrib = contrib_df.copy()
        abs_contrib["abs_c"] = abs_contrib["contribution"].abs()
        abs_contrib = abs_contrib[abs_contrib["abs_c"] > 0]
        fig_donut = go.Figure(go.Pie(
            labels=abs_contrib["driver"],
            values=abs_contrib["abs_c"],
            hole=0.55,
            textinfo="label+percent",
            textfont_size=11,
        ))
        fig_donut.update_layout(
            title="Доля факторов (|вклад|)",
            height=340,
            margin=dict(l=10, r=10, t=48, b=10),
            showlegend=False,
        )
        apply_kibad_theme(fig_donut)

        # ── Waterfall ──────────────────────────────────────────────────────
        waterfall_el = ""
        try:
            wf = waterfall_data(result)
            fig_wf = plot_waterfall(wf)
            apply_kibad_theme(fig_wf)
            waterfall_el = dcc.Graph(figure=fig_wf)
        except Exception:
            pass

        charts_row = dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_bar), md=7),
            dbc.Col(dcc.Graph(figure=fig_donut), md=5),
        ], className="mb-3")

        return html.Div([
            section_header("Результаты декомпозиции"),
            kpi_row,
            charts_row,
            waterfall_el,
            section_header("Таблица вкладов"),
            data_table(contrib_df, id="attr-contrib-tbl"),
        ]), attr_store

    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

"""p08_attribution – Factor Attribution / Decomposition page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS, STORE_ATTRIBUTION,
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
            """, style={"color": "#8891a5"}),
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
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
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

    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
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

        children = [section_header("Результаты декомпозиции")]

        # Contributions table
        contrib_df = result.contributions
        children.append(data_table(contrib_df, id="attr-contrib-tbl"))

        # Waterfall chart
        try:
            wf = waterfall_data(result)
            fig = plot_waterfall(wf)
            apply_kibad_theme(fig)
            children.append(dcc.Graph(figure=fig))
        except Exception:
            pass

        return html.Div(children), attr_store

    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update

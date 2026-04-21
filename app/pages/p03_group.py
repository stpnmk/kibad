"""p03_group — Group & Aggregate page.

Design port of Slide 5 «Группировка и агрегация». Layout:
- Hero header with overline «Шаг 3», H1, subtitle, and Reset / export
  actions on the right.
- Two-column split (1fr 1fr) — LEFT: configuration card with dataset
  badge, multi-selects for group-by / metric columns, a grid of pill-
  checkboxes for aggregation functions, and an optional time-bucket
  row; RIGHT: result card with title+subtitle head, three KPI tiles
  (group count / totals), a bar chart, and a data-table preview.

Keeps the existing ``core.aggregate`` integration — only the UI shell
is reshaped to match the handoff.
"""
from __future__ import annotations

import dash
from dash import (
    ALL, Input, Output, State, callback, ctx, dcc, html, no_update,
)
import pandas as pd
import plotly.express as px

from app.components.alerts import alert_banner
from app.components.cards import chip
from app.components.icons import icon
from app.components.table import data_table
from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_ACTIVE_DS, STORE_AGG_RESULTS, STORE_DATASET, STORE_PREPARED,
    get_df_from_store, list_datasets, save_dataframe,
)
from core.aggregate import TIME_BUCKET_MAP, group_aggregate
from core.audit import log_event

dash.register_page(
    __name__,
    path="/group",
    name="3. Группировка",
    order=3,
    icon="table",
)


# ---------------------------------------------------------------------------
# Aggregation functions shown as pill-checkboxes (matches handoff slide 5)
# ---------------------------------------------------------------------------
_AGG_FUNCS: list[tuple[str, str]] = [
    ("sum",      "sum"),
    ("mean",     "mean"),
    ("median",   "median"),
    ("count",    "count"),
    ("std",      "std"),
    ("min",      "min"),
    ("max",      "max"),
    ("nunique",  "nunique"),
]
_DEFAULT_AGGS: list[str] = ["sum", "mean"]

_TIME_LABELS: dict[str, str] = {
    "":        "Без времени",
    "W-MON":   "Неделя",
    "month":   "Месяц",
    "quarter": "Квартал",
    "year":    "Год",
}


# ---------------------------------------------------------------------------
# UI builders
# ---------------------------------------------------------------------------
def _hero() -> html.Div:
    """Top hero card — overline + H1 + subtitle, action buttons on right."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Шаг 3 · Агрегация", className="kb-overline"),
                    html.H1("Группировка и агрегация",
                            className="kb-h1 kb-group-hero__title"),
                    html.Div(
                        "Сводные таблицы по одной или нескольким колонкам",
                        className="kb-body-l kb-group-hero__sub",
                    ),
                ],
                className="kb-group-hero__left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("refresh", 12), html.Span("Сбросить")],
                        id="ga-btn-reset",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("download", 12), html.Span("Экспорт CSV")],
                        id="ga-btn-export",
                        className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                        disabled=True,
                    ),
                ],
                className="kb-group-hero__right",
            ),
        ],
        className="kb-card kb-group-hero",
    )


def _agg_grid(selected: list[str]) -> html.Div:
    """Pill-checkbox grid (4 columns) for aggregation functions."""
    selected = set(selected or [])
    cells = []
    for value, label in _AGG_FUNCS:
        is_on = value in selected
        cells.append(
            html.Button(
                [
                    html.Span(
                        icon("check", 10) if is_on else "",
                        className="kb-check__box",
                    ),
                    html.Span(label, className="kb-check__label mono"),
                ],
                id={"type": "ga-agg", "fn": value},
                className="kb-check" + (" is-on" if is_on else ""),
                n_clicks=0,
            )
        )
    return html.Div(cells, className="kb-agg-grid")


def _config_card() -> html.Div:
    """LEFT card — all configuration inputs."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Настройка группировки"),
                            html.Div("Определите ключи и показатели",
                                     className="caption"),
                        ],
                        className="kb-card-title",
                    ),
                    chip("Конфиг", variant="neutral"),
                ],
                className="kb-card-head",
            ),

            # Dataset select — Dropdown stays, displayed as a 1-up row
            html.Div(
                [
                    html.Div("Датасет", className="kb-field-label"),
                    dcc.Dropdown(
                        id="ga-ds-select",
                        className="kb-select",
                        placeholder="Выберите датасет",
                        clearable=False,
                    ),
                    html.Div(id="ga-ds-meta", className="kb-group-ds-meta"),
                ],
                className="kb-field",
            ),

            # Group-by columns
            html.Div(
                [
                    html.Div("Колонки группировки", className="kb-field-label"),
                    dcc.Dropdown(
                        id="ga-group-cols",
                        className="kb-select",
                        multi=True,
                        placeholder="Измерения для строк…",
                    ),
                ],
                className="kb-field",
            ),

            # Target columns
            html.Div(
                [
                    html.Div("Целевые колонки", className="kb-field-label"),
                    dcc.Dropdown(
                        id="ga-metric-cols",
                        className="kb-select",
                        multi=True,
                        placeholder="Числовые метрики…",
                    ),
                ],
                className="kb-field",
            ),

            # Aggregation functions — pill-checkbox grid
            html.Div(
                [
                    html.Div("Агрегации", className="kb-field-label"),
                    html.Div(id="ga-agg-grid-wrap"),
                    dcc.Store(id="ga-agg-funcs-store", data=_DEFAULT_AGGS),
                ],
                className="kb-field",
            ),

            # Time bucket (optional)
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Временное ведро", className="kb-field-label"),
                            dcc.Dropdown(
                                id="ga-time-bucket",
                                className="kb-select",
                                options=[
                                    {"label": "Без группировки по времени",
                                     "value": ""},
                                ] + [
                                    {"label": _TIME_LABELS.get(k, k),
                                     "value": k}
                                    for k in TIME_BUCKET_MAP.keys()
                                ],
                                value="",
                                clearable=False,
                            ),
                        ],
                        className="kb-group-timegrid__col",
                    ),
                    html.Div(
                        [
                            html.Div("Дата-колонка", className="kb-field-label"),
                            dcc.Dropdown(
                                id="ga-date-col",
                                className="kb-select",
                                placeholder="—",
                            ),
                        ],
                        className="kb-group-timegrid__col",
                    ),
                ],
                className="kb-group-timegrid",
            ),

            # Action row
            html.Div(
                [
                    html.Button(
                        [icon("play", 12), html.Span("Сгруппировать")],
                        id="ga-run-btn",
                        className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                ],
                className="kb-group-actions",
            ),
        ],
        className="kb-card kb-card--lg kb-group-config",
    )


def _result_empty_state() -> html.Div:
    """Placeholder shown in the right column before the first run."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.H3("Результат"),
                         html.Div("Запустите группировку, чтобы увидеть результат",
                                  className="caption")],
                        className="kb-card-title",
                    ),
                    chip("—", variant="neutral"),
                ],
                className="kb-card-head",
            ),
            html.Div(
                [
                    html.Div(icon("table", 24),
                             className="kb-group-empty__icon"),
                    html.Div("Конфигурация ещё не применена",
                             className="kb-group-empty__title"),
                    html.Div(
                        "Выберите датасет, укажите колонки группировки и "
                        "целевые метрики, затем нажмите «Сгруппировать».",
                        className="kb-group-empty__desc",
                    ),
                ],
                className="kb-group-empty",
            ),
        ],
        className="kb-card kb-card--lg kb-group-result",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        _hero(),

        html.Div(
            [
                _config_card(),
                html.Div(id="ga-results", children=_result_empty_state()),
            ],
            className="kb-group-layout",
        ),

        dcc.Download(id="ga-download"),
    ],
    className="kb-page kb-group-page",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("ga-ds-select", "options"),
    Output("ga-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def update_ds_options(datasets, active):
    names = list_datasets(datasets)
    if not names:
        return [], None
    val = active if active in names else names[0]
    return [{"label": n, "value": n} for n in names], val


@callback(
    Output("ga-group-cols", "options"),
    Output("ga-metric-cols", "options"),
    Output("ga-date-col", "options"),
    Output("ga-date-col", "value"),
    Output("ga-ds-meta", "children"),
    Input("ga-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_columns(ds_name, datasets, prepared):
    if not ds_name:
        return [], [], [], None, ""
    df = get_df_from_store(prepared, ds_name) or get_df_from_store(datasets, ds_name)
    if df is None:
        return [], [], [], None, ""
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    date_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]

    meta = html.Span(
        f"{df.shape[0]:,} × {df.shape[1]}".replace(",", " "),
        className="mono kb-group-ds-meta__text",
    )
    return (
        [{"label": c, "value": c} for c in all_cols],
        [{"label": c, "value": c} for c in num_cols],
        [{"label": c, "value": c} for c in date_cols],
        date_cols[0] if date_cols else None,
        meta,
    )


# ---------------------------------------------------------------------------
# Pill-checkbox grid for aggregation functions
# ---------------------------------------------------------------------------
@callback(
    Output("ga-agg-grid-wrap", "children"),
    Input("ga-agg-funcs-store", "data"),
)
def render_agg_grid(selected):
    return _agg_grid(selected or [])


@callback(
    Output("ga-agg-funcs-store", "data"),
    Input({"type": "ga-agg", "fn": ALL}, "n_clicks"),
    State("ga-agg-funcs-store", "data"),
    prevent_initial_call=True,
)
def toggle_agg(clicks, current):
    trig = ctx.triggered_id
    if not isinstance(trig, dict) or not any(clicks or []):
        return no_update
    fn = trig.get("fn")
    current = list(current or [])
    if fn in current:
        current.remove(fn)
    else:
        current.append(fn)
    # Preserve handoff order
    order = [v for v, _ in _AGG_FUNCS]
    current.sort(key=lambda x: order.index(x) if x in order else 99)
    return current


# ---------------------------------------------------------------------------
# Reset button
# ---------------------------------------------------------------------------
@callback(
    Output("ga-group-cols", "value"),
    Output("ga-metric-cols", "value"),
    Output("ga-agg-funcs-store", "data", allow_duplicate=True),
    Output("ga-time-bucket", "value"),
    Output("ga-results", "children", allow_duplicate=True),
    Output("ga-btn-export", "disabled", allow_duplicate=True),
    Input("ga-btn-reset", "n_clicks"),
    prevent_initial_call=True,
)
def on_reset(n):
    if not n:
        return no_update, no_update, no_update, no_update, no_update, no_update
    return [], [], list(_DEFAULT_AGGS), "", _result_empty_state(), True


# ---------------------------------------------------------------------------
# Run aggregation
# ---------------------------------------------------------------------------
def _fmt_big(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if abs(v) >= 1e9:
        return f"{v / 1e9:.2f}B"
    if abs(v) >= 1e6:
        return f"{v / 1e6:.2f}M"
    if abs(v) >= 1e3:
        return f"{v / 1e3:.1f}K"
    if v == int(v):
        return f"{int(v):,}".replace(",", " ")
    return f"{v:.2f}"


@callback(
    Output("ga-results", "children", allow_duplicate=True),
    Output(STORE_AGG_RESULTS, "data"),
    Output("ga-btn-export", "disabled"),
    Input("ga-run-btn", "n_clicks"),
    State("ga-ds-select", "value"),
    State("ga-group-cols", "value"),
    State("ga-metric-cols", "value"),
    State("ga-agg-funcs-store", "data"),
    State("ga-time-bucket", "value"),
    State("ga-date-col", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    State(STORE_AGG_RESULTS, "data"),
    prevent_initial_call=True,
)
def run_aggregation(n, ds_name, group_cols, metric_cols, agg_funcs,
                    time_bucket, date_col, datasets, prepared, agg_store):
    if not n:
        return no_update, no_update, no_update
    if not ds_name:
        return (
            html.Div(
                alert_banner("Сначала выберите датасет.", level="warning"),
                className="kb-card kb-card--lg",
            ),
            no_update, True,
        )
    if not group_cols or not metric_cols or not agg_funcs:
        return (
            html.Div(
                alert_banner(
                    "Выберите колонки группировки, метрики и хотя бы одну "
                    "функцию агрегации.",
                    level="warning",
                ),
                className="kb-card kb-card--lg",
            ),
            no_update, True,
        )

    df = get_df_from_store(prepared, ds_name) or get_df_from_store(datasets, ds_name)
    if df is None:
        return (
            html.Div(alert_banner("Датасет не найден.", level="danger"),
                     className="kb-card kb-card--lg"),
            no_update, True,
        )

    try:
        result_df = group_aggregate(
            df,
            group_cols=group_cols,
            metric_cols=metric_cols,
            agg_funcs=agg_funcs,
            time_col=date_col if time_bucket else None,
            time_bucket=time_bucket if time_bucket else None,
        )
    except Exception as exc:
        return (
            html.Div(alert_banner(f"Ошибка: {exc}", level="danger"),
                     className="kb-card kb-card--lg"),
            no_update, True,
        )

    log_event("aggregate",
              details={"dataset": ds_name, "group": group_cols,
                        "agg": agg_funcs})
    path = save_dataframe(result_df, f"agg_{ds_name}")
    agg_store = dict(agg_store or {})
    agg_store[ds_name] = path

    # --- KPI tiles above the chart ---------------------------------------
    n_groups = len(result_df)
    first_metric = metric_cols[0]
    # Look for sum-of-first-metric column (pattern used by group_aggregate)
    sum_col = next(
        (c for c in result_df.columns
         if c.startswith(f"{first_metric}_") and c.endswith("sum")),
        None,
    )
    mean_col = next(
        (c for c in result_df.columns
         if c.startswith(f"{first_metric}_") and c.endswith("mean")),
        None,
    )

    def kpi_tile(label, value, tone: str = "") -> html.Div:
        return html.Div(
            [
                html.Div(label, className="kb-stat-label"),
                html.Div(value, className="kb-stat-value " + tone),
            ],
            className="kb-stat-card",
        )

    kpi_row = html.Div(
        [
            kpi_tile("Групп", str(n_groups)),
            kpi_tile(
                f"Σ {first_metric}" if sum_col else f"◦ {first_metric}",
                _fmt_big(result_df[sum_col].sum()) if sum_col else "—",
            ),
            kpi_tile(
                f"X̄ {first_metric}" if mean_col else "—",
                _fmt_big(result_df[mean_col].mean()) if mean_col else "—",
            ),
        ],
        className="kb-group-kpis",
    )

    # --- Chart (optional — bar) ------------------------------------------
    graph = None
    value_cols = [c for c in result_df.columns if c not in group_cols]
    if len(group_cols) == 1 and value_cols:
        try:
            # Take up to 12 groups + up to 4 metric columns to keep it readable
            top = result_df.head(12)
            ycols = value_cols[:4]
            fig = px.bar(
                top,
                x=group_cols[0],
                y=ycols,
                barmode="group",
                title=None,
            )
            apply_kibad_theme(fig)
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", y=-0.2, x=0, title=""),
            )
            graph = dcc.Graph(figure=fig, config={"displayModeBar": False})
        except Exception:
            graph = None

    # --- Result card (head + KPI + chart + table) ------------------------
    head = html.Div(
        [
            html.Div(
                [
                    html.H3("Результат"),
                    html.Div(
                        f"Группировка по {len(group_cols)} ключ"
                        f"{'у' if len(group_cols) == 1 else 'ам'} · "
                        f"{n_groups:,} строк".replace(",", " "),
                        className="caption",
                    ),
                ],
                className="kb-card-title",
            ),
            chip("готово", variant="success"),
        ],
        className="kb-card-head",
    )

    body: list = [kpi_row]
    if graph is not None:
        body.append(html.Div(graph, className="kb-group-result__chart"))
    body.append(
        html.Div(
            [
                html.Div("Превью таблицы", className="kb-overline"),
                data_table(result_df.head(50), id="ga-result-table",
                           page_size=10),
            ],
            className="kb-group-result__table",
        )
    )

    return (
        html.Div([head, *body],
                 className="kb-card kb-card--lg kb-group-result"),
        agg_store,
        False,
    )


# ---------------------------------------------------------------------------
# Export CSV
# ---------------------------------------------------------------------------
@callback(
    Output("ga-download", "data"),
    Input("ga-btn-export", "n_clicks"),
    State("ga-ds-select", "value"),
    State(STORE_AGG_RESULTS, "data"),
    prevent_initial_call=True,
)
def on_export(n, ds_name, agg_store):
    if not n or not ds_name or not agg_store or ds_name not in agg_store:
        return no_update
    path = agg_store[ds_name]
    try:
        df = pd.read_parquet(path)
    except Exception:
        return no_update
    return dcc.send_data_frame(df.to_csv, f"agg_{ds_name}.csv", index=False)

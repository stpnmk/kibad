"""p04_merge – Table merge (JOIN/UNION) page.

Design port of Slide 6 "Объединение таблиц". Primary flow on the page is
JOIN with a side-by-side layout: source selectors on top, JOIN-type Venn
cards in the middle, keys/diagnostics card on the left, result preview
card on the right. The UNION tab is kept for parity with existing logic.
"""
from __future__ import annotations

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from app.state import (
    get_df_from_store, save_dataframe, list_datasets,
    STORE_DATASET, STORE_PREPARED,
)
from app.components.cards import card, chip, kpi
from app.components.icons import icon
from app.components.alerts import alert_banner
from app.components.table import data_table
from core.merge import merge_tables, concat_tables, analyze_key_cardinality
from core.audit import log_event

dash.register_page(__name__, path="/merge", name="4. Объединение", order=4, icon="diagram-3")


# ---------------------------------------------------------------------------
# Visual atoms
# ---------------------------------------------------------------------------
def _venn(kind: str) -> html.Span:
    """Two-circle Venn diagram rendered via CSS (see .kb-venn-* rules)."""
    return html.Span(className=f"kb-venn kb-venn--{kind}")


def _join_type_label(kind: str, caption: str) -> html.Div:
    """Custom label for a JOIN-type radio option — Venn + caption + active chip."""
    return html.Div(
        [
            _venn(kind),
            html.Span(caption.upper(), className="kb-join-caption"),
            html.Span("активно", className="kb-join-active-pill"),
        ],
        className="kb-join-card",
    )


def _page_head() -> html.Div:
    """Top band: step overline, title, subtitle, and right-aligned tools."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Шаг 3", className="kb-overline"),
                    html.H1("Объединение таблиц", className="kb-page-title"),
                    html.Div(
                        "JOIN / UNION с диагностикой совместимости ключей",
                        className="kb-page-subtitle",
                    ),
                ],
                className="kb-page-head-left",
            ),
            html.Div(
                [
                    html.Button(
                        [icon("history", 14), html.Span("История")],
                        id="merge-history-btn", className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                    html.Button(
                        [icon("file-text", 14), html.Span("SQL-превью")],
                        id="merge-sql-btn", className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                ],
                className="kb-page-head-actions",
            ),
        ],
        className="kb-page-head",
    )


def _dataset_pickers() -> html.Div:
    """Row with «Таблица A» + link glyph + «Таблица B» selectors."""
    return html.Div(
        [
            html.Div(
                [
                    html.Label("Таблица A", className="kb-field-label"),
                    dcc.Dropdown(
                        id="merge-left-ds", placeholder="Выберите датасет…",
                        className="kb-merge-select",
                    ),
                    html.Div(id="merge-left-caption", className="kb-merge-caption"),
                ],
                className="kb-merge-side",
            ),
            html.Div(icon("link", 16), className="kb-merge-link"),
            html.Div(
                [
                    html.Label("Таблица B", className="kb-field-label"),
                    dcc.Dropdown(
                        id="merge-right-ds", placeholder="Выберите датасет…",
                        className="kb-merge-select",
                    ),
                    html.Div(id="merge-right-caption", className="kb-merge-caption"),
                ],
                className="kb-merge-side",
            ),
        ],
        className="kb-merge-pickers",
    )


def _join_type_picker() -> html.Div:
    """4-card RadioItems with Venn diagrams (INNER / LEFT / RIGHT / OUTER)."""
    options = [
        {"label": _join_type_label("inner", "Inner"), "value": "inner"},
        {"label": _join_type_label("left",  "Left"),  "value": "left"},
        {"label": _join_type_label("right", "Right"), "value": "right"},
        {"label": _join_type_label("outer", "Outer"), "value": "outer"},
    ]
    return html.Div(
        [
            html.Label("Тип JOIN", className="kb-field-label"),
            dcc.RadioItems(
                id="merge-how",
                options=options,
                value="left",
                className="kb-join-types",
                labelClassName="kb-join-label",
                inputClassName="kb-join-input",
            ),
        ],
        className="kb-merge-join-types",
    )


def _keys_card() -> html.Div:
    """Left body card — keys setup + diagnostics + primary action."""
    return card(
        title="Ключи соединения",
        subtitle="Сопоставьте колонки попарно",
        head_right=html.Div(
            [icon("plus", 12), html.Span("ключ")],
            className="kb-btn kb-btn--text kb-btn--sm",
            style={"cursor": "default"},
        ),
        size="lg",
        children=[
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("A · левая", className="kb-keys-col-label"),
                            dcc.Dropdown(
                                id="merge-left-key", placeholder="Ключ A…",
                                className="kb-merge-keyselect",
                            ),
                        ],
                        className="kb-keys-col",
                    ),
                    html.Div(
                        icon("arrows-lr", 14),
                        className="kb-keys-pair-glyph",
                    ),
                    html.Div(
                        [
                            html.Div("B · правая", className="kb-keys-col-label"),
                            dcc.Dropdown(
                                id="merge-right-key", placeholder="Ключ B…",
                                className="kb-merge-keyselect",
                            ),
                        ],
                        className="kb-keys-col",
                    ),
                ],
                className="kb-keys-pair",
            ),
            html.Div(id="merge-diagnostics", className="kb-merge-diagnostics"),
            html.Div(
                [
                    html.Button(
                        [icon("play", 12), html.Span("Выполнить JOIN")],
                        id="merge-run-btn", className="kb-btn kb-btn--primary",
                        n_clicks=0,
                    ),
                    html.Button(
                        "Диагностика",
                        id="merge-diag-btn", className="kb-btn kb-btn--ghost",
                        n_clicks=0,
                    ),
                ],
                className="kb-merge-actions",
            ),
        ],
    )


def _result_card() -> html.Div:
    """Right body card — KPI row + preview table. Populated by callbacks."""
    return card(
        title="Предпросмотр результата",
        subtitle="Выберите датасеты и ключи",
        head_right=html.Div(id="merge-match-chip"),
        size="lg",
        children=[
            html.Div(id="merge-kpis", className="kb-merge-kpis"),
            dcc.Loading(
                html.Div(id="merge-result", className="kb-merge-result"),
                type="circle", color="var(--accent-500)",
            ),
        ],
    )


def _concat_tab() -> html.Div:
    """UNION/CONCAT tab — kept functional, lighter styling."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Датасеты для объединения",
                                className="kb-field-label",
                            ),
                            dcc.Dropdown(
                                id="concat-datasets", multi=True,
                                placeholder="Выберите 2 и более…",
                            ),
                        ],
                        style={"flex": "2"},
                    ),
                    html.Div(
                        [
                            html.Label("Ось", className="kb-field-label"),
                            dcc.Dropdown(
                                id="concat-axis",
                                options=[
                                    {"label": "По строкам (UNION)",      "value": "0"},
                                    {"label": "По колонкам (side-by-side)", "value": "1"},
                                ],
                                value="0",
                            ),
                        ],
                        style={"flex": "1"},
                    ),
                    html.Div(
                        html.Button(
                            [icon("play", 12), html.Span("Объединить")],
                            id="concat-run-btn", className="kb-btn kb-btn--primary",
                            n_clicks=0,
                        ),
                        style={"alignSelf": "flex-end"},
                    ),
                ],
                className="kb-concat-controls",
            ),
            dcc.Loading(
                html.Div(id="concat-result"),
                type="circle", color="var(--accent-500)",
            ),
        ],
        style={"paddingTop": "16px"},
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        _page_head(),

        dbc.Tabs(
            [
                dbc.Tab(
                    label="JOIN", tab_id="tab-join",
                    children=html.Div(
                        [
                            _dataset_pickers(),
                            _join_type_picker(),
                            html.Div(
                                [_keys_card(), _result_card()],
                                className="kb-merge-body",
                            ),
                        ],
                        className="kb-merge-page",
                    ),
                ),
                dbc.Tab(label="UNION", tab_id="tab-concat", children=_concat_tab()),
            ],
            id="merge-tabs", active_tab="tab-join",
        ),
    ],
    className="kb-page kb-page-merge",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("merge-left-ds", "options"),
    Output("merge-right-ds", "options"),
    Output("concat-datasets", "options"),
    Input(STORE_DATASET, "data"),
)
def update_ds_lists(datasets):
    names = list_datasets(datasets)
    opts = [{"label": n, "value": n} for n in names]
    return opts, opts, opts


def _load_df(name, datasets, prepared):
    if not name:
        return None
    return get_df_from_store(prepared, name) or get_df_from_store(datasets, name)


def _shape_caption(df) -> str:
    return f"{df.shape[0]:,} × {df.shape[1]}".replace(",", " ")


@callback(
    Output("merge-left-key", "options"),
    Output("merge-left-caption", "children"),
    Input("merge-left-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_left(ds, datasets, prepared):
    df = _load_df(ds, datasets, prepared)
    if df is None:
        return [], ""
    cols = [{"label": c, "value": c} for c in df.columns]
    return cols, _shape_caption(df)


@callback(
    Output("merge-right-key", "options"),
    Output("merge-right-caption", "children"),
    Input("merge-right-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_right(ds, datasets, prepared):
    df = _load_df(ds, datasets, prepared)
    if df is None:
        return [], ""
    cols = [{"label": c, "value": c} for c in df.columns]
    return cols, _shape_caption(df)


@callback(
    Output("merge-diagnostics", "children"),
    Output("merge-match-chip", "children"),
    Output("merge-kpis", "children"),
    Input("merge-left-key", "value"),
    Input("merge-right-key", "value"),
    Input("merge-how", "value"),
    State("merge-left-ds", "value"),
    State("merge-right-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def diagnostics(lk, rk, how, lds, rds, datasets, prepared):
    if not all([lk, rk, lds, rds]):
        return "", "", ""

    left_df = _load_df(lds, datasets, prepared)
    right_df = _load_df(rds, datasets, prepared)
    if left_df is None or right_df is None:
        return "", "", ""

    try:
        c = analyze_key_cardinality(left_df, right_df, [lk], [rk])
    except Exception as ex:
        return alert_banner(f"Ошибка анализа: {ex}", "danger"), "", ""

    a_rows = int(left_df.shape[0])
    b_rows = int(right_df.shape[0])
    left_null_pct = float(left_df[lk].isna().mean() * 100) if a_rows else 0.0
    right_null_pct = float(right_df[rk].isna().mean() * 100) if b_rows else 0.0

    # Match rate — fraction of A-keys present in B.
    left_keys = left_df[lk].dropna()
    right_key_set = set(right_df[rk].dropna().unique().tolist())
    if len(left_keys):
        matched = left_keys.isin(right_key_set).sum()
        match_rate = float(matched) * 100.0 / len(left_keys)
    else:
        match_rate = 0.0

    join_type = c.get("join_type", "unknown")
    duplicates = int(a_rows - int(c.get("left_unique_keys") or a_rows))

    warns: list = []
    left_dtype = str(left_df[lk].dtype)
    right_dtype = str(right_df[rk].dtype)
    if left_dtype != right_dtype:
        warns.append(
            html.Div(
                [
                    icon("alert", 14),
                    html.Div(
                        [
                            html.Strong("Несовпадение типов"),
                            f" на ключе {lk} ({left_dtype}) ↔ {rk} ({right_dtype}). "
                            "Потребуется приведение типов.",
                        ]
                    ),
                ],
                className="kb-callout kb-callout--warning",
            )
        )
    if left_null_pct > 0:
        warns.append(
            alert_banner(
                f"Пустые ключи в таблице A: {left_null_pct:.1f}%", "warning"
            )
        )
    if right_null_pct > 0:
        warns.append(
            alert_banner(
                f"Пустые ключи в таблице B: {right_null_pct:.1f}%", "warning"
            )
        )
    if join_type == "N:M":
        warns.append(
            alert_banner("Many-to-many связь — возможен взрыв строк.", "danger")
        )

    if match_rate >= 99.0:
        match_chip = chip(f"{match_rate:.0f}% match", "success")
    elif match_rate >= 80.0:
        match_chip = chip(f"{match_rate:.0f}% match", "info")
    else:
        match_chip = chip(f"{match_rate:.0f}% match", "warning")

    expected = {
        "inner": min(a_rows, b_rows),
        "left":  a_rows,
        "right": b_rows,
        "outer": max(a_rows, b_rows),
    }.get(how, a_rows)

    kpi_row = html.Div(
        [
            kpi("A строк", f"{a_rows:,}".replace(",", " ")),
            kpi("B строк", f"{b_rows:,}".replace(",", " ")),
            kpi("Ожидаемо", f"{expected:,}".replace(",", " ")),
            kpi("Дубликаты", f"{duplicates:,}".replace(",", " ")),
        ],
        className="kb-merge-kpi-row",
    )

    return html.Div(warns), match_chip, kpi_row


@callback(
    Output("merge-result", "children"),
    Output(STORE_DATASET, "data", allow_duplicate=True),
    Input("merge-run-btn", "n_clicks"),
    State("merge-left-ds", "value"),
    State("merge-right-ds", "value"),
    State("merge-left-key", "value"),
    State("merge-right-key", "value"),
    State("merge-how", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_merge(n, lds, rds, lk, rk, how, datasets, prepared):
    if not n:
        return no_update, no_update
    if not all([lds, rds, lk, rk]):
        return alert_banner("Выберите оба датасета и ключи.", "warning"), no_update

    left_df = _load_df(lds, datasets, prepared)
    right_df = _load_df(rds, datasets, prepared)
    if left_df is None or right_df is None:
        return alert_banner("Датасеты не найдены.", "danger"), no_update

    try:
        result_df, warnings = merge_tables(left_df, right_df, [lk], [rk], how=how)
    except Exception as ex:
        return alert_banner(f"Ошибка: {ex}", "danger"), no_update

    name = f"{lds}_x_{rds}"
    path = save_dataframe(result_df, name)
    datasets = datasets or {}
    datasets[name] = path
    log_event("merge", dataset=name, details=f"{lds} {how} {rds} on {lk}={rk}")

    children = [alert_banner(str(w), "warning") for w in warnings]
    children.append(
        alert_banner(
            f"Сохранено «{name}» · {result_df.shape[0]:,} × {result_df.shape[1]}.",
            "success",
        )
    )
    children.append(data_table(result_df.head(50), id="merge-result-tbl"))
    return html.Div(children), datasets


@callback(
    Output("concat-result", "children"),
    Output(STORE_DATASET, "data", allow_duplicate=True),
    Input("concat-run-btn", "n_clicks"),
    State("concat-datasets", "value"),
    State("concat-axis", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_concat(n, ds_names, axis, datasets, prepared):
    if not n:
        return no_update, no_update
    if not ds_names or len(ds_names) < 2:
        return alert_banner("Выберите 2 или более датасетов.", "warning"), no_update

    dfs = []
    for name in ds_names:
        df = _load_df(name, datasets, prepared)
        if df is not None:
            dfs.append(df)
    if len(dfs) < 2:
        return alert_banner("Не удалось загрузить датасеты.", "danger"), no_update

    try:
        result_df = concat_tables(dfs, axis=int(axis))
    except Exception as ex:
        return alert_banner(f"Ошибка: {ex}", "danger"), no_update

    name = "concat_" + "_".join(ds_names[:3])
    path = save_dataframe(result_df, name)
    datasets = datasets or {}
    datasets[name] = path

    return html.Div(
        [
            alert_banner(
                f"Сохранено «{name}» · {result_df.shape[0]:,} × {result_df.shape[1]}.",
                "success",
            ),
            data_table(result_df.head(50), id="concat-result-tbl"),
        ]
    ), datasets

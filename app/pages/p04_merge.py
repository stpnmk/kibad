"""p04_merge – Table merge (JOIN/UNION) page.

Design port of Slide 6 "Объединение таблиц". Primary flow on the page is
JOIN with a side-by-side layout: source selectors on top, JOIN-type Venn
cards in the middle, keys/diagnostics card on the left, result preview
card on the right. The UNION tab is kept for parity with existing logic.
"""
from __future__ import annotations

import dash
from dash import (
    ALL, Input, Output, State, callback, clientside_callback, ctx, dcc, html,
    no_update,
)
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
    """Left body card — keys setup + diagnostics + primary action.

    The body hosts a two-column "visual-mapper" of table columns. A
    clientside callback draws curved SVG paths between paired pills;
    colour-codes compatible (accent-green solid) vs dtype-mismatched
    (amber dashed) pairs.
    """
    return card(
        title="Ключи соединения",
        subtitle="Кликните колонку в A, затем в B — добавится пара",
        head_right=html.Button(
            "Сбросить",
            id="merge-keys-reset",
            className="kb-keys-map__reset",
            n_clicks=0,
        ),
        size="lg",
        children=[
            # Hint / status strip (dynamically rendered).
            html.Div(id="merge-keys-hint", className="kb-keys-map__hint"),
            # Pill map canvas.
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(id="merge-keys-left-head", className="kb-keys-map__head"),
                            html.Div(id="merge-keys-left-col"),
                        ],
                        className="kb-keys-map__col kb-keys-map__col--left",
                    ),
                    # Centre gutter stays empty — the SVG overlay spans the whole map.
                    html.Div(className="kb-keys-map__col kb-keys-map__col--center"),
                    html.Div(
                        [
                            html.Div(id="merge-keys-right-head", className="kb-keys-map__head"),
                            html.Div(id="merge-keys-right-col"),
                        ],
                        className="kb-keys-map__col kb-keys-map__col--right",
                    ),
                    # Overlay — clientside JS populates innerHTML with <svg>.
                    html.Div(id="merge-keys-svg-host", className="kb-keys-links"),
                ],
                className="kb-keys-map",
                id="merge-keys-map",
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
            # State stores for the visual mapper.
            dcc.Store(id="merge-keys-store", data=[]),
            dcc.Store(id="merge-keys-pending", data=None),
            dcc.Store(id="merge-keys-dtypes", data={"left": {}, "right": {}}),
            dcc.Store(id="merge-keys-draw-trigger", data=0),
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


# ---------------------------------------------------------------------------
# Visual key-mapper — pill rendering
# ---------------------------------------------------------------------------
def _pill(side: str, col: str, dtype: str, *, pending: bool, matched: bool,
          mismatch: bool) -> html.Div:
    """One clickable column-pill in the key-mapper."""
    cls = ["kb-key-pill"]
    if pending:
        cls.append("is-pending")
    if matched:
        cls.append("is-matched")
    if mismatch:
        cls.append("is-mismatch")
    return html.Div(
        [
            html.Span(col, className="kb-key-pill__name"),
            html.Span(dtype, className="kb-key-pill__type"),
        ],
        # Pattern-match id. Dash serialises this as a JSON-string in the DOM
        # `id` attribute, which the SVG-drawer parses to find pills.
        id={"type": "mkp", "side": side, "col": col},
        className=" ".join(cls),
        n_clicks=0,
    )


def _render_pills(
    df, side: str, pairs: list, pending: dict | None, other_dtypes: dict
) -> list:
    """Render the full pill list for one side. `other_dtypes` is the opposite
    side's {col: dtype} map, used to colour-code mismatches."""
    if df is None:
        return [html.Div("Выберите датасет…", className="kb-keys-map__empty")]
    out = []
    paired_cols = {p[side] for p in pairs if p.get(side)}
    for col in df.columns:
        dtype = str(df[col].dtype)
        is_pending = bool(pending and pending.get("side") == side and pending.get("col") == col)
        is_matched = col in paired_cols
        # Mismatch iff this col is part of a pair whose opposite dtype differs.
        mismatch = False
        if is_matched:
            other_side = "right" if side == "left" else "left"
            partner = next(
                (p[other_side] for p in pairs if p.get(side) == col), None
            )
            if partner is not None:
                other_dtype = other_dtypes.get(partner)
                if other_dtype is not None and other_dtype != dtype:
                    mismatch = True
        out.append(
            _pill(
                side, col, dtype,
                pending=is_pending, matched=is_matched, mismatch=mismatch,
            )
        )
    return out


# Reset pairs & pending whenever either dataset is swapped.
@callback(
    Output("merge-keys-store", "data", allow_duplicate=True),
    Output("merge-keys-pending", "data", allow_duplicate=True),
    Input("merge-left-ds", "value"),
    Input("merge-right-ds", "value"),
    prevent_initial_call=True,
)
def reset_on_ds_change(_lds, _rds):
    return [], None


# Single source of truth for the pill map: renders both columns + heads
# + captions + dtypes whenever the dataset or the stored pairs/pending
# change. Keeping it as one callback removes the race on `merge-keys-dtypes`.
@callback(
    Output("merge-keys-left-col", "children"),
    Output("merge-keys-right-col", "children"),
    Output("merge-keys-left-head", "children"),
    Output("merge-keys-right-head", "children"),
    Output("merge-left-caption", "children"),
    Output("merge-right-caption", "children"),
    Output("merge-keys-dtypes", "data"),
    Input("merge-left-ds", "value"),
    Input("merge-right-ds", "value"),
    Input("merge-keys-store", "data"),
    Input("merge-keys-pending", "data"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_pill_map(lds, rds, pairs, pending, datasets, prepared):
    left_df = _load_df(lds, datasets, prepared)
    right_df = _load_df(rds, datasets, prepared)

    left_dtypes = (
        {c: str(left_df[c].dtype) for c in left_df.columns}
        if left_df is not None else {}
    )
    right_dtypes = (
        {c: str(right_df[c].dtype) for c in right_df.columns}
        if right_df is not None else {}
    )

    left_pills = _render_pills(left_df, "left",  pairs or [], pending, right_dtypes)
    right_pills = _render_pills(right_df, "right", pairs or [], pending, left_dtypes)

    def _head(letter: str, ds_name):
        if not ds_name:
            return f"{letter} · —"
        return [
            html.Span(f"{letter} · ", className="kb-keys-map__head"),
            html.Span(str(ds_name).upper(), className="kb-keys-map__head-name"),
        ]

    left_cap = _shape_caption(left_df) if left_df is not None else ""
    right_cap = _shape_caption(right_df) if right_df is not None else ""

    return (
        left_pills,
        right_pills,
        _head("A", lds),
        _head("B", rds),
        left_cap,
        right_cap,
        {"left": left_dtypes, "right": right_dtypes},
    )


# ---------------------------------------------------------------------------
# Visual key-mapper — click interaction (add / toggle / switch-selection)
# ---------------------------------------------------------------------------
@callback(
    Output("merge-keys-store", "data", allow_duplicate=True),
    Output("merge-keys-pending", "data", allow_duplicate=True),
    Input({"type": "mkp", "side": ALL, "col": ALL}, "n_clicks"),
    Input("merge-keys-reset", "n_clicks"),
    State("merge-keys-store", "data"),
    State("merge-keys-pending", "data"),
    prevent_initial_call=True,
)
def handle_pill_click(pill_clicks, reset_n, pairs, pending):
    trig = ctx.triggered_id
    if trig == "merge-keys-reset":
        if reset_n:
            return [], None
        return no_update, no_update
    # Ignore the initial pattern-match fire with all zeros.
    if not any(pill_clicks or []):
        return no_update, no_update
    if not isinstance(trig, dict):
        return no_update, no_update

    side = trig.get("side")
    col = trig.get("col")
    pairs = list(pairs or [])

    # Click on an already-paired pill → remove that pair, clear pending.
    idx = next((i for i, p in enumerate(pairs) if p.get(side) == col), None)
    if idx is not None:
        pairs.pop(idx)
        return pairs, None

    # No pending yet → this becomes the pending selection.
    if not pending:
        return no_update, {"side": side, "col": col}

    # Pending on the same side → switch the selection.
    if pending.get("side") == side:
        return no_update, {"side": side, "col": col}

    # Pending on the opposite side → form a pair.
    new_pair = {
        "left":  col if side == "left" else pending["col"],
        "right": col if side == "right" else pending["col"],
    }
    # Guard: prevent duplicate pair (if user clicks same combo twice).
    if any(p["left"] == new_pair["left"] and p["right"] == new_pair["right"] for p in pairs):
        return no_update, None
    pairs.append(new_pair)
    return pairs, None


# ---------------------------------------------------------------------------
# Hint / status strip above the pill-map
# ---------------------------------------------------------------------------
@callback(
    Output("merge-keys-hint", "children"),
    Input("merge-keys-store", "data"),
    Input("merge-keys-pending", "data"),
)
def render_hint(pairs, pending):
    pairs = pairs or []
    left = html.Span(
        f"Пар настроено: {len(pairs)}" if pairs else
        "Кликните колонку в A, затем в B — так добавляется пара"
    )
    right = html.Span()
    if pending:
        side_ru = "A" if pending["side"] == "left" else "B"
        right = html.Span(
            [
                "Ожидается пара для ",
                html.Strong(f'{side_ru} · {pending["col"]}'),
            ]
        )
    return [left, right]


# ---------------------------------------------------------------------------
# Clientside trigger — redraws SVG links whenever pairs / dtypes change.
# ---------------------------------------------------------------------------
clientside_callback(
    """
    function(pairs, dtypes) {
        if (window.kbDrawMergeKeys) {
            setTimeout(function() { window.kbDrawMergeKeys(pairs, dtypes); }, 30);
        }
        return (Array.isArray(pairs) ? pairs.length : 0) + ':' + Date.now();
    }
    """,
    Output("merge-keys-draw-trigger", "data"),
    Input("merge-keys-store", "data"),
    Input("merge-keys-dtypes", "data"),
)


# ---------------------------------------------------------------------------
# Diagnostics — aggregates match-rate, dtype mismatches, nulls, cardinality
# across all configured key pairs.
# ---------------------------------------------------------------------------
@callback(
    Output("merge-diagnostics", "children"),
    Output("merge-match-chip", "children"),
    Output("merge-kpis", "children"),
    Input("merge-keys-store", "data"),
    Input("merge-how", "value"),
    State("merge-left-ds", "value"),
    State("merge-right-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def diagnostics(pairs, how, lds, rds, datasets, prepared):
    pairs = pairs or []
    if not pairs or not lds or not rds:
        return "", "", ""

    left_df = _load_df(lds, datasets, prepared)
    right_df = _load_df(rds, datasets, prepared)
    if left_df is None or right_df is None:
        return "", "", ""

    # Guard: only keep pairs whose cols still exist in both sides.
    pairs = [
        p for p in pairs
        if p.get("left") in left_df.columns and p.get("right") in right_df.columns
    ]
    if not pairs:
        return "", "", ""

    lks = [p["left"] for p in pairs]
    rks = [p["right"] for p in pairs]

    try:
        c = analyze_key_cardinality(left_df, right_df, lks, rks)
    except Exception as ex:
        return alert_banner(f"Ошибка анализа: {ex}", "danger"), "", ""

    a_rows = int(left_df.shape[0])
    b_rows = int(right_df.shape[0])
    left_null_pct = float(left_df[lks].isna().any(axis=1).mean() * 100) if a_rows else 0.0
    right_null_pct = float(right_df[rks].isna().any(axis=1).mean() * 100) if b_rows else 0.0

    # Match rate — fraction of A-key-tuples present in B (composite-key aware).
    left_keys = left_df[lks].dropna()
    right_keys = right_df[rks].dropna()
    if len(left_keys) and len(right_keys):
        right_tuples = set(map(tuple, right_keys.itertuples(index=False, name=None)))
        left_tuples = list(map(tuple, left_keys.itertuples(index=False, name=None)))
        matched = sum(1 for t in left_tuples if t in right_tuples)
        match_rate = matched * 100.0 / len(left_tuples)
    else:
        match_rate = 0.0

    join_type = c.get("join_type", "unknown")
    duplicates = int(a_rows - int(c.get("left_unique_keys") or a_rows))

    warns: list = []
    mismatches = [
        (p["left"], str(left_df[p["left"]].dtype),
         p["right"], str(right_df[p["right"]].dtype))
        for p in pairs
        if str(left_df[p["left"]].dtype) != str(right_df[p["right"]].dtype)
    ]
    if mismatches:
        pair_txt: list = []
        for i, (lk, ld, rk, rd) in enumerate(mismatches):
            if i:
                pair_txt.append(", ")
            pair_txt.append(f"{lk} ({ld}) ↔ {rk} ({rd})")
        warns.append(
            html.Div(
                [
                    icon("alert", 14),
                    html.Div(
                        [
                            html.Strong("Несовпадение типов"),
                            " на ключ" + ("ах " if len(mismatches) > 1 else "е "),
                            *pair_txt,
                            ". Потребуется приведение типов.",
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
    State("merge-keys-store", "data"),
    State("merge-how", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_merge(n, lds, rds, pairs, how, datasets, prepared):
    if not n:
        return no_update, no_update
    pairs = pairs or []
    if not lds or not rds or not pairs:
        return alert_banner(
            "Выберите оба датасета и хотя бы одну пару ключей.", "warning"
        ), no_update

    left_df = _load_df(lds, datasets, prepared)
    right_df = _load_df(rds, datasets, prepared)
    if left_df is None or right_df is None:
        return alert_banner("Датасеты не найдены.", "danger"), no_update

    lks = [p["left"] for p in pairs]
    rks = [p["right"] for p in pairs]

    try:
        mr = merge_tables(left_df, right_df, lks, rks, how=how)
    except Exception as ex:
        return alert_banner(f"Ошибка: {ex}", "danger"), no_update
    result_df = mr.df
    warnings = mr.warnings

    name = f"{lds}_x_{rds}"
    path = save_dataframe(result_df, name)
    datasets = datasets or {}
    datasets[name] = path
    keys_txt = ", ".join(f"{lk}={rk}" for lk, rk in zip(lks, rks))
    log_event(
        "merge",
        details={
            "dataset": name, "left": lds, "right": rds,
            "how": how, "keys": keys_txt,
        },
    )

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

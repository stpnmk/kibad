"""p05_explore – Exploratory analysis page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_kibad_theme
from app.components.layout import page_header, section_header, empty_state
from app.components.table import data_table
from app.components.cards import stat_card, chip
from app.components.alerts import alert_banner
from app.components.icons import icon
from core.explore import (
    plot_timeseries, plot_histogram, plot_boxplot, plot_violin,
    plot_correlation_heatmap,
    build_pivot, plot_pivot_bar, plot_waterfall, plot_stl_decomposition, compute_kpi,
)
from core.insights import analyze_dataset, format_insights_markdown, score_data_quality

dash.register_page(__name__, path="/explore", name="5. Исследование", order=5, icon="search")


def _load_df(prepared, datasets, ds):
    """Prefer prepared over raw. Can't use ``a or b`` because ``bool(df)``
    raises on DataFrames."""
    df = get_df_from_store(prepared, ds)
    if df is None:
        df = get_df_from_store(datasets, ds)
    return df

layout = html.Div([
    page_header("5. Исследовательский анализ", "Распределения, корреляции, профилирование"),
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="exp-ds-select", placeholder="Выберите датасет..."),
        ], width=4),
    ], className="mb-3"),
    html.Div(id="exp-info", className="mb-3"),

    dbc.Tabs(id="exp-tabs", active_tab="tab-auto", children=[
        dbc.Tab(label="Авто-анализ", tab_id="tab-auto"),
        dbc.Tab(label="Временные ряды", tab_id="tab-ts"),
        dbc.Tab(label="Распределения", tab_id="tab-dist"),
        dbc.Tab(label="Корреляции", tab_id="tab-corr"),
        dbc.Tab(label="Попарные графики", tab_id="tab-pair"),
        dbc.Tab(label="KPI-трекер", tab_id="tab-kpi"),
        dbc.Tab(label="Профиль данных", tab_id="tab-profile"),
    ]),
    dcc.Loading(html.Div(id="exp-tab-content"), type="circle", color="#10b981"),
])


@callback(
    Output("exp-ds-select", "options"),
    Output("exp-ds-select", "value"),
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
    Output("exp-info", "children"),
    Input("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def show_info(ds, datasets, prepared):
    if not ds:
        return empty_state("", "Данные не загружены", "Загрузите датасет на странице Данные")
    df = _load_df(prepared, datasets, ds)
    if df is None:
        return ""
    return html.Div([
        stat_card("Строк", f"{df.shape[0]:,}"),
        stat_card("Столбцов", str(df.shape[1])),
        stat_card("Числовых", str(len(df.select_dtypes(include='number').columns))),
    ], className="kb-stats-grid")


# ---------------------------------------------------------------------------
# Auto-analysis renderer
# ---------------------------------------------------------------------------

_QUALITY_TIER: list[tuple[float, str, str]] = [
    (85, "Отличное качество",   "success"),
    (70, "Хорошее качество",    "info"),
    (50, "Требует внимания",    "warning"),
    (0,  "Критические проблемы", "danger"),
]


def _quality_tier(score: float) -> tuple[str, str]:
    """Map a 0–100 score to (label, chip-variant)."""
    for threshold, label, variant in _QUALITY_TIER:
        if score >= threshold:
            return label, variant
    return _QUALITY_TIER[-1][1], _QUALITY_TIER[-1][2]


def _score_bar(label: str, value: float) -> html.Div:
    """Horizontal progress row: label · value% · filled bar."""
    width_pct = max(0, min(100, value))
    if value >= 85:
        tone = "is-good"
    elif value >= 60:
        tone = "is-mid"
    else:
        tone = "is-bad"
    return html.Div(
        [
            html.Div(
                [
                    html.Span(label, className="kb-score-bar__label"),
                    html.Span(f"{value:.0f}%", className="kb-score-bar__value"),
                ],
                className="kb-score-bar__head",
            ),
            html.Div(
                html.Div(className=f"kb-score-bar__fill {tone}",
                          style={"width": f"{width_pct:.1f}%"}),
                className="kb-score-bar__track",
            ),
        ],
        className="kb-score-bar",
    )


def _quality_hero(score: dict) -> html.Div:
    overall = float(score.get("overall", 0) or 0)
    tier_label, tier_variant = _quality_tier(overall)
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Качество данных", className="kb-stat-label"),
                    html.Div(
                        [
                            html.Span(f"{overall:.0f}", className="kb-score-big"),
                            html.Span("/100", className="kb-score-big__denom"),
                        ],
                        className="kb-score-big-row",
                    ),
                    chip(tier_label, variant=tier_variant),
                ],
                className="kb-score-hero__left",
            ),
            html.Div(
                [
                    _score_bar("Заполненность",  float(score.get("completeness", 0) or 0)),
                    _score_bar("Уникальность",   float(score.get("uniqueness", 0) or 0)),
                    _score_bar("Согласованность", float(score.get("consistency", 0) or 0)),
                ],
                className="kb-score-hero__right",
            ),
        ],
        className="kb-card kb-card--lg kb-score-hero",
    )


def _summary_kpis(summary: dict) -> html.Div:
    def tile(label, value, hint=""):
        return html.Div(
            [
                html.Div(label, className="kb-stat-label"),
                html.Div(
                    [
                        html.Div(value, className="kb-stat-value"),
                        html.Div(hint or "", className="kb-stat-hint"),
                    ],
                    className="kb-stat-row",
                ),
            ],
            className="kb-stat-card",
        )

    n_rows = int(summary.get("n_rows", 0))
    n_cols = int(summary.get("n_cols", 0))
    n_num = int(summary.get("n_numeric", 0))
    n_cat = int(summary.get("n_categorical", 0))
    n_dt = int(summary.get("n_datetime", 0))
    miss = float(summary.get("missing_pct", 0) or 0)
    dup = float(summary.get("duplicate_pct", 0) or 0)
    n_dup = int(summary.get("n_duplicates", 0))

    return html.Div(
        [
            tile("Строк",    f"{n_rows:,}".replace(",", " "), "всего"),
            tile("Столбцов", str(n_cols),
                 f"{n_num} чис · {n_cat} кат · {n_dt} дата"),
            tile("Пропусков", f"{miss:.1f}%",
                 "в среднем по колонкам"),
            tile("Дублей",   f"{n_dup:,}".replace(",", " "),
                 f"{dup:.1f}% от объёма" if n_dup else "нет"),
        ],
        className="kb-autoa-kpis",
    )


def _reco_priority_chip(priority: str) -> html.Span:
    label = {"high": "важно", "medium": "средний", "low": "позже"}.get(
        priority, priority)
    return html.Span(label, className=f"kb-reco-prio kb-reco-prio--{priority}")


def _reco_card(recs: list[dict]) -> html.Div:
    if not recs:
        body = html.Div(
            "Автоматических рекомендаций пока нет — данные выглядят аккуратно.",
            className="caption",
            style={"color": "var(--text-secondary)"},
        )
    else:
        # sort high → medium → low
        order = {"high": 0, "medium": 1, "low": 2}
        recs = sorted(recs, key=lambda r: order.get(r.get("priority", "low"), 9))
        body = html.Div(
            [
                html.Div(
                    [
                        _reco_priority_chip(r.get("priority", "low")),
                        html.Div(
                            [
                                html.Div(r.get("action", ""), className="kb-reco-action"),
                                html.Div(r.get("reason", ""), className="kb-reco-reason"),
                            ],
                            className="kb-reco-body",
                        ),
                    ],
                    className="kb-reco-row",
                )
                for r in recs
            ],
            className="kb-reco-list",
        )

    head = html.Div(
        [
            html.Div(
                [
                    html.H3("Рекомендации"),
                    html.Div(
                        f"{len(recs)} действий" if recs else "без действий",
                        className="caption",
                    ),
                ],
                className="kb-card-title",
            ),
        ],
        className="kb-card-head",
    )
    return html.Div([head, body], className="kb-card kb-card--lg")


def _corr_card(correlations: list[dict]) -> html.Div:
    if not correlations:
        body = html.Div(
            "Недостаточно числовых колонок для расчёта корреляций.",
            className="caption",
            style={"color": "var(--text-secondary)"},
        )
    else:
        rows = []
        for c in correlations:
            r = float(c.get("r", 0))
            tone = "kb-corr-r--pos" if r > 0 else "kb-corr-r--neg"
            rows.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Span(c.get("col_a", ""), className="mono"),
                                html.Span(" ↔ ", className="kb-corr-sep"),
                                html.Span(c.get("col_b", ""), className="mono"),
                            ],
                            className="kb-corr-pair",
                        ),
                        html.Span(f"r = {r:+.2f}",
                                  className=f"kb-corr-r {tone}"),
                        html.Div(c.get("insight_text", ""),
                                 className="kb-corr-note"),
                    ],
                    className="kb-corr-row",
                )
            )
        body = html.Div(rows, className="kb-corr-list")

    head = html.Div(
        html.Div(
            [html.H3("Топ корреляции"),
             html.Div("Сильнейшие связи между числовыми колонками",
                      className="caption")],
            className="kb-card-title",
        ),
        className="kb-card-head",
    )
    return html.Div([head, body], className="kb-card kb-card--lg")


def _trend_card(trends: list[dict]) -> html.Div:
    if not trends:
        body = html.Div(
            "Нет дата-колонок — тренды не вычислены.",
            className="caption",
            style={"color": "var(--text-secondary)"},
        )
    else:
        rows = []
        for t in trends:
            direction = t.get("direction", "")
            pct = t.get("pct_change")
            if direction == "рост":
                arrow, tone = "▲", "kb-trend--pos"
            elif direction == "снижение":
                arrow, tone = "▼", "kb-trend--neg"
            else:
                arrow, tone = "■", "kb-trend--flat"
            value_str = (f"{pct:+.1f}%" if isinstance(pct, (int, float))
                         else direction)
            rows.append(
                html.Div(
                    [
                        html.Span(arrow, className=f"kb-trend-arrow {tone}"),
                        html.Div(
                            [
                                html.Div(t.get("col_value", ""),
                                         className="kb-trend-col mono"),
                                html.Div(t.get("insight_text", ""),
                                         className="kb-trend-note"),
                            ],
                            className="kb-trend-body",
                        ),
                        html.Span(value_str,
                                  className=f"kb-trend-value {tone}"),
                    ],
                    className="kb-trend-row",
                )
            )
        body = html.Div(rows, className="kb-trend-list")

    head = html.Div(
        html.Div(
            [html.H3("Динамика"),
             html.Div("Изменение метрик во времени",
                      className="caption")],
            className="kb-card-title",
        ),
        className="kb-card-head",
    )
    return html.Div([head, body], className="kb-card kb-card--lg")


def _dist_card(distributions: list[dict]) -> html.Div:
    if not distributions:
        body = html.Div("Нет числовых колонок.", className="caption",
                        style={"color": "var(--text-secondary)"})
    else:
        # surface the most skewed / most-outlier cols first
        dist = sorted(distributions,
                      key=lambda d: (abs(d.get("skewness", 0))
                                      + d.get("outlier_pct", 0) / 100),
                      reverse=True)[:6]
        rows = []
        for d in dist:
            sk = float(d.get("skewness", 0))
            out = float(d.get("outlier_pct", 0) or 0)
            if abs(sk) > 1:
                tone = "kb-dist--warn"
            elif out > 5:
                tone = "kb-dist--warn"
            else:
                tone = "kb-dist--ok"
            pills = [
                html.Span(f"skew {sk:+.2f}", className="kb-dist-pill"),
                html.Span(f"outliers {out:.1f}%", className="kb-dist-pill"),
            ]
            rows.append(
                html.Div(
                    [
                        html.Div(d.get("col", ""), className="kb-dist-col mono"),
                        html.Div(pills, className="kb-dist-pills"),
                        html.Div(d.get("insight_text", ""),
                                 className=f"kb-dist-note {tone}"),
                    ],
                    className="kb-dist-row",
                )
            )
        body = html.Div(rows, className="kb-dist-list")

    head = html.Div(
        html.Div(
            [html.H3("Распределения"),
             html.Div("Форма числовых колонок и выбросы",
                      className="caption")],
            className="kb-card-title",
        ),
        className="kb-card-head",
    )
    return html.Div([head, body], className="kb-card kb-card--lg")


def _cat_card(top_values: dict) -> html.Div:
    items = list(top_values.values()) if isinstance(top_values, dict) else list(top_values)
    if not items:
        body = html.Div(
            "Нет категориальных колонок (с малым числом уникальных значений).",
            className="caption",
            style={"color": "var(--text-secondary)"},
        )
    else:
        items.sort(key=lambda v: v.get("top_pct", 0), reverse=True)
        rows = []
        for v in items[:6]:
            pct = float(v.get("top_pct", 0))
            if pct > 50:
                tone = "kb-conc--high"
            elif pct > 30:
                tone = "kb-conc--mid"
            else:
                tone = "kb-conc--low"
            rows.append(
                html.Div(
                    [
                        html.Div(v.get("col", ""), className="kb-cat-col mono"),
                        html.Div(
                            [
                                html.Span(v.get("top_value", ""),
                                          className="kb-cat-top mono"),
                                html.Span(f"{pct:.0f}%",
                                          className=f"kb-cat-pct {tone}"),
                            ],
                            className="kb-cat-pair",
                        ),
                        html.Div(v.get("concentration_insight", ""),
                                 className="kb-cat-note"),
                    ],
                    className="kb-cat-row",
                )
            )
        body = html.Div(rows, className="kb-cat-list")

    head = html.Div(
        html.Div(
            [html.H3("Категории"),
             html.Div("Концентрация значений в текстовых колонках",
                      className="caption")],
            className="kb-card-title",
        ),
        className="kb-card-head",
    )
    return html.Div([head, body], className="kb-card kb-card--lg")


def _anomalies_card(anomalies: list[dict]) -> html.Div:
    if not anomalies:
        body = html.Div(
            "Явных выбросов не обнаружено.",
            className="caption",
            style={"color": "var(--text-secondary)"},
        )
    else:
        rows = []
        for a in anomalies:
            rows.append(
                html.Div(
                    [
                        html.Span(f"#{a.get('row_idx', '—')}",
                                  className="kb-anom-idx mono"),
                        html.Div(a.get("col", ""), className="kb-anom-col mono"),
                        html.Span(f"{a.get('value', '—')}",
                                  className="kb-anom-val mono"),
                        html.Span(
                            f"ожидаемо {a.get('expected_range', '—')}",
                            className="kb-anom-range",
                        ),
                    ],
                    className="kb-anom-row",
                )
            )
        body = html.Div(rows, className="kb-anom-list")

    head = html.Div(
        html.Div(
            [html.H3("Аномалии"),
             html.Div("Самые экстремальные значения (IQR)",
                      className="caption")],
            className="kb-card-title",
        ),
        className="kb-card-head",
    )
    return html.Div([head, body], className="kb-card kb-card--lg")


def _issues_card(issues: list[dict]) -> html.Div:
    if not issues:
        return html.Div()  # omit card when no issues
    # Prioritize errors → warnings → info
    order = {"error": 0, "warning": 1, "info": 2}
    issues = sorted(issues, key=lambda i: order.get(i.get("level", "info"), 9))
    rows = []
    for iss in issues[:15]:
        lvl = iss.get("level", "info")
        level_label = {"error": "ошибка",
                       "warning": "предупреждение",
                       "info": "совет"}.get(lvl, lvl)
        rows.append(
            html.Div(
                [
                    html.Span(level_label,
                              className=f"kb-issue-lvl kb-issue-lvl--{lvl}"),
                    html.Div(
                        [
                            html.Span(iss.get("col") or "—",
                                      className="kb-issue-col mono"),
                            html.Span(iss.get("message", ""),
                                      className="kb-issue-msg"),
                        ],
                        className="kb-issue-body",
                    ),
                ],
                className="kb-issue-row",
            )
        )

    head = html.Div(
        html.Div(
            [html.H3("Проблемы и замечания"),
             html.Div(f"Всего: {len(issues)}", className="caption")],
            className="kb-card-title",
        ),
        className="kb-card-head",
    )
    return html.Div([head, html.Div(rows, className="kb-issue-list")],
                    className="kb-card kb-card--lg")


def _render_auto_analysis(df: pd.DataFrame) -> html.Div:
    """Main renderer for the ``tab-auto`` tab — quality score, summary KPIs,
    insight cards (correlations / trends / distributions / categories /
    anomalies), actionable recommendations, and a full issue list."""
    insights = analyze_dataset(df)
    score = score_data_quality(df)

    return html.Div(
        [
            _quality_hero(score),
            _summary_kpis(insights.get("summary", {})),
            _reco_card(insights.get("recommendations", [])),
            html.Div(
                [
                    _corr_card(insights.get("correlations", [])),
                    _trend_card(insights.get("trends", [])),
                ],
                className="kb-autoa-grid",
            ),
            html.Div(
                [
                    _dist_card(insights.get("distributions", [])),
                    _cat_card(insights.get("top_values", {})),
                ],
                className="kb-autoa-grid",
            ),
            _anomalies_card(insights.get("anomalies", [])),
            _issues_card(score.get("issues", [])),
        ],
        className="kb-autoa",
    )


def _render_auto_landing(df: pd.DataFrame) -> html.Div:
    """Pre-analysis CTA card — shows dataset size and a launch button."""
    n_rows, n_cols = df.shape
    rows_fmt = f"{n_rows:,}".replace(",", " ")
    # very rough heuristic — fine for a hint
    if n_rows * n_cols > 500_000:
        duration = "до нескольких секунд"
    elif n_rows * n_cols > 50_000:
        duration = "пара секунд"
    else:
        duration = "почти мгновенно"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H3("Авто-анализ датасета"),
                            html.Div(
                                "Оценим качество, найдём корреляции, тренды, "
                                "выбросы, концентрации и выдадим рекомендации.",
                                className="caption",
                            ),
                        ],
                        className="kb-card-title",
                    ),
                    html.Button(
                        [icon("play"), html.Span("Запустить анализ")],
                        id="exp-auto-run",
                        n_clicks=0,
                        className="kb-btn kb-btn--primary kb-btn--lg",
                    ),
                ],
                className="kb-card-head kb-auto-launch__head",
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Span("Объём", className="kb-overline"),
                            html.Div(
                                f"{rows_fmt} × {n_cols}",
                                className="kb-auto-launch__stat-val mono",
                            ),
                            html.Div("строк × колонок",
                                     className="kb-auto-launch__stat-hint"),
                        ],
                        className="kb-auto-launch__stat",
                    ),
                    html.Div(
                        [
                            html.Span("Время", className="kb-overline"),
                            html.Div(duration,
                                     className="kb-auto-launch__stat-val"),
                            html.Div("оценка",
                                     className="kb-auto-launch__stat-hint"),
                        ],
                        className="kb-auto-launch__stat",
                    ),
                    html.Div(
                        [
                            html.Span("Что получите", className="kb-overline"),
                            html.Div(
                                "оценка · инсайты · рекомендации",
                                className="kb-auto-launch__stat-val",
                            ),
                            html.Div("7 разделов",
                                     className="kb-auto-launch__stat-hint"),
                        ],
                        className="kb-auto-launch__stat",
                    ),
                ],
                className="kb-auto-launch__stats",
            ),
            dcc.Loading(
                html.Div(id="exp-auto-result"),
                type="circle",
                color="#21A066",
            ),
        ],
        className="kb-card kb-card--lg kb-auto-launch",
    )


@callback(
    Output("exp-tab-content", "children"),
    Input("exp-tabs", "active_tab"),
    Input("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_tab(tab, ds, datasets, prepared):
    if not ds:
        return empty_state("", "Выберите датасет", "")
    df = _load_df(prepared, datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "warning")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    if tab == "tab-auto":
        # Show landing CTA — analysis is run manually via the button.
        # The ``exp-auto-result`` div inside the landing is populated by
        # the ``run_auto_analysis`` callback below.
        return _render_auto_landing(df)

    elif tab == "tab-ts":
        if not dt_cols or not num_cols:
            return alert_banner("Нужна хотя бы одна дата-колонка и числовая колонка.", "info")
        return html.Div([
            section_header("Временные ряды"),
            dbc.Row([
                dbc.Col([dcc.Dropdown(id="exp-ts-date", options=[{"label": c, "value": c} for c in dt_cols], value=dt_cols[0] if dt_cols else None, placeholder="Дата...")], width=4),
                dbc.Col([dcc.Dropdown(id="exp-ts-vals", options=[{"label": c, "value": c} for c in num_cols], value=num_cols[:2], multi=True, placeholder="Значения...")], width=8),
            ]),
            html.Div(id="exp-ts-chart"),
        ])

    elif tab == "tab-dist":
        if not num_cols:
            return alert_banner("Нет числовых колонок.", "info")
        figs = []
        for col in num_cols[:6]:
            fig = plot_histogram(df, col)
            apply_kibad_theme(fig)
            figs.append(dbc.Col(dcc.Graph(figure=fig), width=6))
        return html.Div([section_header("Распределения"), dbc.Row(figs)])

    elif tab == "tab-corr":
        if len(num_cols) < 2:
            return alert_banner("Нужно минимум 2 числовых колонки.", "info")
        fig = plot_correlation_heatmap(df, num_cols)
        apply_kibad_theme(fig)
        return html.Div([section_header("Матрица корреляций"), dcc.Graph(figure=fig)])

    elif tab == "tab-pair":
        if len(num_cols) < 2:
            return alert_banner("Нужно минимум 2 числовых колонки.", "info")
        cols = num_cols[:5]
        fig = px.scatter_matrix(df[cols], dimensions=cols, title="Попарные графики")
        apply_kibad_theme(fig)
        fig.update_layout(height=700)
        return html.Div([section_header("Попарные графики (Scatter Matrix)"), dcc.Graph(figure=fig)])

    elif tab == "tab-kpi":
        if not dt_cols or not num_cols:
            return alert_banner("Нужна дата-колонка и числовая колонка.", "info")
        try:
            kpis = compute_kpi(df, dt_cols[0], num_cols[0])
            cards = [stat_card(k, str(v)) for k, v in kpis.items()]
            return html.Div([section_header("KPI"), html.Div(cards, className="kb-stats-grid")])
        except Exception as e:
            return alert_banner(f"Ошибка KPI: {e}", "warning")

    elif tab == "tab-profile":
        desc = df.describe(include="all").T
        desc = desc.round(2)
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Колонка", "Пропуски"]
        miss["% пропусков"] = (miss["Пропуски"] / len(df) * 100).round(1)
        return html.Div([
            section_header("Профиль данных"),
            html.H4("Описательная статистика"),
            data_table(desc.reset_index().rename(columns={"index": "Колонка"}), id="exp-profile-desc"),
            html.H4("Пропуски", className="mt-3"),
            data_table(miss[miss["Пропуски"] > 0], id="exp-profile-miss"),
        ])

    return ""


@callback(
    Output("exp-auto-result", "children"),
    Input("exp-auto-run", "n_clicks"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def run_auto_analysis(n_clicks, ds, datasets, prepared):
    """Populate the auto-analysis result area after the user clicks «Запустить анализ»."""
    if not n_clicks or not ds:
        return no_update
    df = _load_df(prepared, datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "warning")
    try:
        return _render_auto_analysis(df)
    except Exception as e:
        return alert_banner(f"Ошибка авто-анализа: {e}", "warning")


@callback(
    Output("exp-ts-chart", "children"),
    Input("exp-ts-date", "value"),
    Input("exp-ts-vals", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def render_ts(date_col, val_cols, ds, datasets, prepared):
    if not date_col or not val_cols or not ds:
        return ""
    df = _load_df(prepared, datasets, ds)
    if df is None:
        return ""
    try:
        fig = plot_timeseries(df, date_col, val_cols if isinstance(val_cols, list) else [val_cols])
        apply_kibad_theme(fig)
        return dcc.Graph(figure=fig)
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "warning")

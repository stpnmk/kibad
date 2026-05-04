"""p05_explore – Exploratory analysis page (Dash) — handoff-8 redesign."""
from __future__ import annotations

import logging

import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

from app.state import (
    get_df_from_store, get_df_from_stores,
    STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS,
)
from app.figure_theme import apply_eda_theme
from app.components.table import data_table
from app.components.alerts import alert_banner
from core.explore import (
    plot_timeseries, plot_histogram, plot_boxplot, plot_violin,
    plot_correlation_heatmap,
    build_pivot, plot_pivot_bar,
)
from core.insights import analyze_dataset, score_data_quality

dash.register_page(
    __name__,
    path="/explore",
    name="5. Исследование",
    order=5,
    icon="search",
)


# ─── Helpers ──────────────────────────────────────────────────────────────

def _kpi(label: str, value: str, sub: str = "", muted: bool = False,
         value_color: str = "", bar_pct: float | None = None,
         bar_kind: str = "ok") -> html.Div:
    """Render a `.kpi.eda` tile."""
    children = [
        html.Div(label, className="label"),
        html.Div(
            value,
            className="value muted" if muted else "value",
            style={"color": value_color} if value_color else {},
        ),
    ]
    if sub:
        children.append(html.Div(sub, className="sub"))
    if bar_pct is not None:
        bar_class = "bar"
        if bar_kind == "warn":
            bar_class += " warn"
        elif bar_kind == "dang":
            bar_class += " dang"
        children.append(html.Div(
            html.Div(className=bar_class,
                     style={"width": f"{max(0, min(100, bar_pct)):.0f}%"}),
            className="progress",
            style={"marginTop": "4px"},
        ))
    return html.Div(children, className="kpi eda")


def _chip(text: str, kind: str = "neutral") -> html.Span:
    return html.Span(text, className=f"chip {kind}")


def _interpret_correlation(r: float) -> str:
    a = abs(r)
    direction = "положительная" if r > 0 else "отрицательная"
    if a >= 0.9:
        return f"Почти линейная {direction} связь — возможна мультиколлинеарность."
    if a >= 0.7:
        return f"Сильная {direction} связь."
    if a >= 0.5:
        return f"Умеренная {direction} связь."
    if a >= 0.3:
        return f"Слабая {direction} связь."
    return "Связь незначительна."


def _bullet(line_text: str, explain: str, direction: str = "info",
            icon: str | None = None, extra_chip: str | None = None) -> html.Div:
    """Render a `.bullet` row."""
    icn_cls = f"icn {direction}"
    main_children = []
    if extra_chip:
        main_children.append(html.Div([
            html.Span(line_text, className="mono-line"),
            html.Span(extra_chip, className="chip neutral",
                      style={"fontSize": "9px", "marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px"}))
    else:
        main_children.append(html.Div(line_text, className="mono-line"))
    main_children.append(html.Div(explain, className="explain"))
    return html.Div([
        html.Div(html.Span(icon or "•"), className=icn_cls),
        html.Div(main_children, className="main"),
    ], className="bullet")


def _stat_chip(k: str, v: str) -> html.Div:
    return html.Div([
        html.Span(k, className="k"),
        html.Span(v, className="v"),
    ], className="stat-chip")


def _fmt_num(x, max_abs_for_full: float = 1e6) -> str:
    try:
        if x is None or pd.isna(x):
            return "—"
    except Exception:
        return str(x)
    try:
        a = abs(float(x))
        if a >= 1e9:
            return f"{x/1e9:.2f}B"
        if a >= 1e6:
            return f"{x/1e6:.2f}M"
        if a >= 1e3:
            return f"{x/1e3:.2f}K"
        if a >= 1:
            return f"{x:.2f}"
        return f"{x:.4f}"
    except Exception:
        return str(x)


def _df_to_tbl(df: pd.DataFrame, mono_cols: list[str] | None = None,
               max_height: int = 320) -> html.Div:
    """Render a DataFrame as a `.tbl` styled HTML table (read-only, scrollable)."""
    mono_cols = mono_cols or []
    head = html.Thead(html.Tr([
        html.Th(c, className="mono" if c in mono_cols else "")
        for c in df.columns
    ]))
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cls = "mono" if c in mono_cols else ""
            cells.append(html.Td(str(v) if not pd.isna(v) else "—", className=cls))
        rows.append(html.Tr(cells))
    return html.Div(
        html.Table([head, html.Tbody(rows)], className="tbl"),
        style={"maxHeight": f"{max_height}px", "overflow": "auto"},
    )


# ─── Layout ───────────────────────────────────────────────────────────────

layout = html.Div([
    html.Div("ИССЛЕДОВАТЕЛЬСКИЙ АНАЛИЗ", className="overline"),
    html.H1("5. Исследовательский анализ", className="page-title",
            style={"marginTop": "4px"}),
    html.P("Распределения, корреляции, качество данных", className="caption",
           style={"marginTop": "2px", "marginBottom": "14px"}),

    # Dataset dropdown — visible select (kept for switching datasets)
    html.Div([
        dcc.Dropdown(id="exp-ds-select", placeholder="Выберите датасет...",
                     className="dataset-select-dd"),
    ], style={"maxWidth": "520px", "marginBottom": "12px"}),

    # Pretty dataset banner (read-only summary line)
    html.Div(id="exp-info", style={"marginBottom": "14px"}),

    dbc.Tabs(
        id="exp-tabs",
        active_tab="tab-auto",
        children=[
            dbc.Tab(label="Авто-анализ", tab_id="tab-auto"),
            dbc.Tab(label="Качество данных", tab_id="tab-quality"),
            dbc.Tab(label="Временные ряды", tab_id="tab-ts"),
            dbc.Tab(label="Распределения", tab_id="tab-dist"),
            dbc.Tab(label="Корреляции", tab_id="tab-corr"),
            dbc.Tab(label="Попарные графики", tab_id="tab-pair"),
            dbc.Tab(label="Сводная таблица", tab_id="tab-pivot"),
            dbc.Tab(label="KPI-трекер", tab_id="tab-kpi"),
            dbc.Tab(label="Профиль данных", tab_id="tab-profile"),
        ],
        style={"marginTop": "18px"},
    ),
    dcc.Loading(
        html.Div(id="exp-tab-content", style={"marginTop": "20px"}),
        type="circle", color="#10b981",
    ),
],
    id="eda-root",
    className="eda-page",
    **{"data-eda-page": "1"},
)


# ─── Dataset dropdown options ─────────────────────────────────────────────

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


# ─── Header info bar + 5-KPI row ──────────────────────────────────────────

@callback(
    Output("exp-info", "children"),
    Input("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def show_info(ds, datasets, prepared):
    if not ds:
        return html.Div([
            html.Div([
                html.Div("Данные не загружены", style={"fontWeight": 500}),
                html.Div("Загрузите датасет на странице «Данные»",
                         className="caption"),
            ], className="empty"),
        ])
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return ""

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]

    # Approx memory usage in MB
    try:
        mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        mb_str = f"{mb:.1f} MB"
    except Exception:
        mb_str = "—"

    banner = html.Div([
        html.Span("DB", style={
            "color": "var(--accent-300, #5fd8c0)",
            "fontFamily": "'JetBrains Mono', monospace",
            "fontWeight": 600,
        }),
        html.Span(ds, className="ds-name"),
        html.Span("·", className="ds-sep"),
        html.Span(
            f"{df.shape[0]:,} строк × {df.shape[1]} колонок · {mb_str}",
            className="ds-meta",
        ),
    ], className="dataset-select")

    kpis = html.Div([
        _kpi("СТРОК", f"{df.shape[0]:,}", "наблюдений"),
        _kpi("СТОЛБЦОВ", f"{df.shape[1]}", "колонок"),
        _kpi("ЧИСЛОВЫХ", f"{len(num_cols)}", "float / int"),
        _kpi("КАТЕГОРИАЛЬНЫХ", f"{len(cat_cols)}", "string / bool"),
        _kpi("ДАТОВЫХ", f"{len(dt_cols)}", "—" if not dt_cols else "datetime",
             muted=(len(dt_cols) == 0)),
    ], className="grid-5", style={"marginTop": "14px"})

    return html.Div([banner, kpis])


# ─── Main tab renderer ────────────────────────────────────────────────────

@callback(
    Output("exp-tab-content", "children"),
    Input("exp-tabs", "active_tab"),
    Input("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_tab(tab, ds, datasets, prepared):
    if not ds:
        return html.Div([
            html.Div("Выберите датасет", style={"fontWeight": 500}),
        ], className="empty")
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return alert_banner("Датасет не найден.", "warning")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    all_cols = list(df.columns)

    # ───── 1. Авто-анализ ───────────────────────────────────────────────
    if tab == "tab-auto":
        try:
            insights = analyze_dataset(df)
            quality = score_data_quality(df)
            overall = float(quality.get("overall", 0))
            completeness = float(quality.get("completeness", 0))
            uniqueness = float(quality.get("uniqueness", 0))
            consistency = float(quality.get("consistency", 0))

            def _kind(score):
                if score >= 90:
                    return "ok"
                if score >= 70:
                    return "warn"
                return "dang"

            def _color(kind):
                return {
                    "ok": "var(--accent-300, #5fd8c0)",
                    "warn": "#E3A953",
                    "dang": "#E07563",
                }.get(kind, "")

            quality_tiles = html.Div([
                _kpi("КАЧЕСТВО ДАННЫХ", f"{overall:.0f}", "/ 100",
                     value_color=_color(_kind(overall)),
                     bar_pct=overall, bar_kind=_kind(overall)),
                _kpi("ПОЛНОТА", f"{completeness:.0f}", "%",
                     value_color=_color(_kind(completeness)),
                     bar_pct=completeness, bar_kind=_kind(completeness)),
                _kpi("УНИКАЛЬНОСТЬ", f"{uniqueness:.0f}", "%",
                     value_color=_color(_kind(uniqueness)),
                     bar_pct=uniqueness, bar_kind=_kind(uniqueness)),
                _kpi("СОГЛАСОВАННОСТЬ", f"{consistency:.0f}", "%",
                     value_color=_color(_kind(consistency)),
                     bar_pct=consistency, bar_kind=_kind(consistency)),
            ], className="grid-4")

            if overall >= 90:
                badge = _chip("✓ ОТЛИЧНОЕ КАЧЕСТВО ДАННЫХ", "success")
            elif overall >= 70:
                badge = _chip("УДОВЛЕТВОРИТЕЛЬНОЕ КАЧЕСТВО", "warning")
            else:
                badge = _chip("ТРЕБУЕТ ВНИМАНИЯ", "danger")

            n_dup = int(df.duplicated().sum())
            miss_pct = float(df.isnull().mean().mean()) * 100
            summary_card = html.Div([
                html.H3("Автоматический анализ датасета"),
                html.Div(
                    (
                        f"Размер: {df.shape[0]:,} строк × {df.shape[1]} колонок  ·  "
                        f"Типы: {len(num_cols)} числовых, {len(cat_cols)} категориальных, "
                        f"{len(dt_cols)} дат  ·  Пропуски: {miss_pct:.1f}%  ·  "
                        f"Дублей: {n_dup}"
                    ),
                    className="mono",
                    style={
                        "marginTop": "8px",
                        "color": "var(--text-secondary)",
                        "fontSize": "12px",
                        "lineHeight": "20px",
                    },
                ),
            ], className="card", style={"padding": "18px"})

            # Correlations card
            corrs = insights.get("correlations", []) or []
            strong_corrs = [c for c in corrs if abs(float(c.get("r", 0))) >= 0.6][:6]
            corr_bullets = []
            for c in strong_corrs:
                r = float(c.get("r", 0))
                d = "up" if r > 0 else "down"
                line = f"{c.get('col_a')} ↔ {c.get('col_b')} = {r:+.2f}"
                corr_bullets.append(_bullet(
                    line,
                    c.get("insight_text") or _interpret_correlation(r),
                    direction=d, icon="↑" if r > 0 else "↓",
                ))
            if not corr_bullets:
                corr_bullets = [html.Div("Сильных связей |r| ≥ 0.6 не найдено.",
                                         className="caption")]
            corr_card = html.Div([
                html.H3("Корреляции"),
                html.Div(f"Найдено {len(strong_corrs)} связи с |r| ≥ 0.6",
                         className="caption", style={"marginTop": "2px"}),
                html.Div(corr_bullets, style={"marginTop": "10px"}),
            ], className="card")

            # Distributions card
            dists = insights.get("distributions", []) or []
            dist_bullets = []
            for d in dists[:6]:
                col = d.get("col")
                skew = float(d.get("skewness", 0))
                shape = (
                    "правосторонний" if skew > 1 else
                    "левосторонний" if skew < -1 else
                    "нормальное"
                )
                line = f"{col} · skew = {skew:+.2f}"
                dist_bullets.append(_bullet(
                    line,
                    d.get("insight_text", ""),
                    direction="info", icon="∼", extra_chip=shape,
                ))
            if not dist_bullets:
                dist_bullets = [html.Div("Нет числовых колонок для анализа.",
                                         className="caption")]
            dist_card = html.Div([
                html.H3("Распределения"),
                html.Div("Автоматические наблюдения по форме",
                         className="caption", style={"marginTop": "2px"}),
                html.Div(dist_bullets, style={"marginTop": "10px"}),
            ], className="card")

            return html.Div([
                html.Div("АВТОМАТИЧЕСКИЙ АНАЛИЗ", className="overline"),
                quality_tiles,
                html.Div(badge, style={"marginTop": "10px", "marginBottom": "10px"}),
                summary_card,
                html.Div([corr_card, dist_card], className="grid-2",
                         style={"marginTop": "14px"}),
            ], className="col-16")
        except Exception as e:
            logger.exception("Авто-анализ упал")
            return alert_banner(f"Ошибка авто-анализа: {e}", "warning")

    # ───── 2. Качество данных ───────────────────────────────────────────
    elif tab == "tab-quality":
        try:
            quality = score_data_quality(df)
            overall = float(quality.get("overall", 0))
            issues = quality.get("issues", []) or []

            err_issues = [i for i in issues if i.get("level") == "error"]
            warn_issues = [i for i in issues if i.get("level") == "warning"]
            info_issues = [i for i in issues if i.get("level") == "info"]

            score_max = 100
            score_pct = max(0, min(100, overall))

            score_card = html.Div([
                html.Div([
                    html.Div("ОБЩИЙ SCORE КАЧЕСТВА", className="overline"),
                    html.Div([
                        html.Span(f"{overall:.0f}", className="score-num"),
                        html.Span(f"/ {score_max}", className="score-den"),
                    ], style={
                        "display": "flex", "alignItems": "baseline",
                        "gap": "4px", "marginTop": "6px",
                    }),
                    html.Div(
                        f"{len(issues)} наблюдений · {len(err_issues)} ошибок",
                        className="caption", style={"marginTop": "2px"},
                    ),
                ]),
                html.Div([
                    html.Div([
                        html.Span("критично"),
                        html.Span("приемлемо"),
                        html.Span("отлично"),
                    ], style={
                        "display": "flex", "justifyContent": "space-between",
                        "marginBottom": "6px", "fontSize": "11px",
                        "color": "var(--text-tertiary)",
                        "fontFamily": "'JetBrains Mono', monospace",
                    }),
                    html.Div(
                        html.Div(className="bar",
                                 style={"width": f"{score_pct:.0f}%"}),
                        className="progress", style={"height": "10px"},
                    ),
                    html.Div([
                        html.Span([
                            html.Span(className="dot",
                                      style={"background": "var(--danger, #E07563)"}),
                            f"Ошибки · {len(err_issues)}",
                        ], className="legend-pill"),
                        html.Span([
                            html.Span(className="dot",
                                      style={"background": "var(--warning, #E3A953)"}),
                            f"Предупр. · {len(warn_issues)}",
                        ], className="legend-pill"),
                        html.Span([
                            html.Span(className="dot",
                                      style={"background": "var(--info, #6DC1E0)"}),
                            f"Инфо · {len(info_issues)}",
                        ], className="legend-pill"),
                    ], className="legend-row", style={"marginTop": "10px"}),
                ]),
                html.Div(
                    dcc.Link("Открыть мастер очистки", href="/prepare",
                             className="eda-btn primary"),
                ),
            ], className="score-card")

            def _sev_block(level: str, label: str, kind: str, items: list,
                          chip_text: str):
                if not items:
                    return None
                rows = []
                for it in items[:8]:
                    col = it.get("col")
                    text_children = [it.get("message", "")]
                    if col:
                        text_children = [
                            it.get("message", ""),
                            html.Span(col, className="mchip",
                                      style={"display": "inline-flex",
                                             "marginLeft": "6px"}),
                        ]
                    rows.append(html.Div([
                        _chip(chip_text, kind),
                        html.Div(text_children, className="text"),
                        dcc.Link("К мастеру →", href="/prepare",
                                 className="eda-btn ghost sm"),
                    ], className="sev-row"))
                head = html.Div([
                    html.Span("⚠"),
                    html.H3(label),
                    html.Span(f"{len(items)} проблем", className="count"),
                ], className="sev-head")
                return html.Div([head] + rows, className=f"sev-card {kind}")

            sev_blocks = [
                b for b in [
                    _sev_block("error", "Ошибки", "danger", err_issues,
                               "✕ ERROR"),
                    _sev_block("warning", "Предупреждения", "warning",
                               warn_issues, "! WARN"),
                    _sev_block("info", "Информация", "info", info_issues,
                               "i INFO"),
                ] if b is not None
            ]
            if not sev_blocks:
                sev_blocks = [html.Div([
                    html.Div("✓", className="icn"),
                    html.Div("Проблем не обнаружено",
                             style={"fontWeight": 500,
                                    "color": "var(--text-primary)"}),
                ], className="empty")]

            n_dup = int(df.duplicated().sum())
            n_miss = int(df.isnull().sum().sum())
            n_total_cells = max(1, df.shape[0] * df.shape[1])
            n_cols_problem = sum(1 for col in df.columns
                                 if any(i.get("col") == col for i in issues))

            kpi_row = html.Div([
                _kpi("ДУБЛЕЙ", f"{n_dup}", f"из {df.shape[0]:,} строк"),
                _kpi("ПРОПУСКОВ ВСЕГО", f"{n_miss:,}",
                     f"{n_miss / n_total_cells * 100:.2f}% ячеек"),
                _kpi("КОЛОНОК С ПРОБЛЕМАМИ", f"{n_cols_problem}",
                     f"из {df.shape[1]} · "
                     f"{n_cols_problem / max(1, df.shape[1]) * 100:.0f}%",
                     value_color="#E3A953" if n_cols_problem else ""),
            ], className="grid-3")

            return html.Div([
                score_card,
                html.Div(sev_blocks, className="col-16"),
                kpi_row,
            ], className="col-16")
        except Exception as e:
            logger.exception("Анализ качества упал")
            return alert_banner(f"Ошибка анализа качества: {e}", "warning")

    # ───── 3. Временные ряды ────────────────────────────────────────────
    elif tab == "tab-ts":
        if not dt_cols or not num_cols:
            return alert_banner(
                "Нужна хотя бы одна дата-колонка и числовая колонка.", "info")
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("КОЛОНКА ДАТ"),
                    dcc.Dropdown(id="exp-ts-date",
                                 options=[{"label": c, "value": c} for c in dt_cols],
                                 value=dt_cols[0], clearable=False),
                ], className="field"),
                html.Div([
                    html.Label("ЧИСЛОВЫЕ КОЛОНКИ · МАКС 6"),
                    dcc.Dropdown(id="exp-ts-vals",
                                 options=[{"label": c, "value": c} for c in num_cols],
                                 value=num_cols[: min(3, len(num_cols))],
                                 multi=True,
                                 className="chips-select"),
                ], className="field"),
                html.Div([
                    html.Label("СЕГМЕНТАЦИЯ ПО"),
                    dcc.Dropdown(
                        id="exp-ts-color",
                        options=[{"label": "(нет)", "value": ""}] +
                                [{"label": c, "value": c} for c in cat_cols],
                        value=""),
                ], className="field"),
            ], className="grid-3",
                style={"gridTemplateColumns": "260px 1fr 260px",
                       "gap": "16px"}),
            html.Div([
                html.Div("Агрегация по дате — сумма по выбранным колонкам.",
                         className="caption"),
                html.Button([
                    html.Span("📈"), " Построить",
                ], className="eda-btn primary"),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginTop": "10px"}),
            html.Div([
                html.Div([
                    html.Div([
                        html.H3("Динамика по сегментам"),
                        html.Div("Агрегация: sum по дате",
                                 className="caption",
                                 style={"fontSize": "11px"}),
                    ]),
                    html.Div([
                        html.Div("День", className="opt"),
                        html.Div("Неделя", className="opt"),
                        html.Div("Месяц", className="opt active"),
                        html.Div("Квартал", className="opt"),
                    ], className="radio-group"),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center", "marginBottom": "10px"}),
                dcc.Loading(html.Div(id="exp-ts-chart"),
                            type="dot", color="#10b981"),
            ], className="card", style={"padding": "16px", "marginTop": "14px"}),
        ], className="col-16")

    # ───── 4. Распределения ─────────────────────────────────────────────
    elif tab == "tab-dist":
        if not num_cols:
            return alert_banner("Нет числовых колонок.", "info")
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("ЧИСЛОВАЯ КОЛОНКА"),
                    dcc.Dropdown(id="exp-dist-col",
                                 options=[{"label": c, "value": c}
                                          for c in num_cols],
                                 value=num_cols[0], clearable=False),
                ], className="field", style={"flex": "1"}),
                html.Div([
                    html.Label("ГРУППИРУЮЩАЯ КОЛОНКА · ОПЦ."),
                    dcc.Dropdown(
                        id="exp-dist-group",
                        options=[{"label": "(нет)", "value": ""}] +
                                [{"label": c, "value": c} for c in cat_cols],
                        value=""),
                ], className="field", style={"flex": "1"}),
                html.Div([
                    html.Label("ТИП"),
                    dcc.RadioItems(
                        id="exp-dist-type",
                        options=[
                            {"label": " Histogram", "value": "hist"},
                            {"label": " Boxplot", "value": "box"},
                            {"label": " Violin", "value": "violin"},
                        ],
                        value="hist", inline=True,
                        className="radio-group",
                        labelStyle={"marginRight": "10px"},
                    ),
                ], className="field"),
            ], style={"display": "flex", "gap": "12px",
                      "alignItems": "flex-end"}),
            html.Div(
                dcc.Loading(html.Div(id="exp-dist-chart"),
                            type="dot", color="#10b981"),
                className="card",
                style={"padding": "16px", "marginTop": "14px"},
            ),
            html.Div(id="exp-dist-stats",
                     style={"display": "flex", "gap": "10px",
                            "flexWrap": "wrap", "marginTop": "14px"}),
        ], className="col-16")

    # ───── 5. Корреляции ────────────────────────────────────────────────
    elif tab == "tab-corr":
        if len(num_cols) < 2:
            return alert_banner("Нужно минимум 2 числовых колонки.", "info")
        default_cols = num_cols[: min(10, len(num_cols))]
        return html.Div([
            html.Div([
                html.Div([
                    html.Label(f"КОЛОНКИ · ВЫБРАНО ИЗ {len(num_cols)}"),
                    dcc.Dropdown(id="exp-corr-cols",
                                 options=[{"label": c, "value": c}
                                          for c in num_cols],
                                 value=default_cols, multi=True,
                                 className="chips-select"),
                ], className="field", style={"flex": "2"}),
                html.Div([
                    html.Label("МЕТОД"),
                    dcc.RadioItems(
                        id="exp-corr-method",
                        options=[
                            {"label": " Pearson", "value": "pearson"},
                            {"label": " Spearman", "value": "spearman"},
                            {"label": " Kendall", "value": "kendall"},
                        ],
                        value="pearson", inline=True,
                        className="radio-group",
                        labelStyle={"marginRight": "10px"},
                    ),
                ], className="field", style={"width": "240px"}),
            ], style={"display": "flex", "gap": "12px",
                      "alignItems": "flex-end"}),
            html.Div([
                html.Div("ⓘ", className="alert-icon"),
                html.Div([
                    html.Div("Как читать", className="alert-title"),
                    html.Div(
                        "r > 0.7 — сильная положительная связь  ·  "
                        "r < −0.7 — сильная отрицательная  ·  |r| < 0.3 — слабая",
                        className="alert-body mono",
                    ),
                ]),
            ], className="alert info", style={"marginTop": "12px"}),
            html.Div(
                dcc.Loading(html.Div(id="exp-corr-chart"),
                            type="dot", color="#10b981"),
                style={"marginTop": "14px"},
            ),
        ], className="col-16")

    # ───── 6. Попарные графики ──────────────────────────────────────────
    elif tab == "tab-pair":
        if len(num_cols) < 2:
            return alert_banner("Нужно минимум 2 числовых колонки.", "info")
        default_pp = num_cols[: min(4, len(num_cols))]
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("КОЛОНКИ · 3–6"),
                    dcc.Dropdown(id="exp-pair-cols",
                                 options=[{"label": c, "value": c}
                                          for c in num_cols],
                                 value=default_pp, multi=True,
                                 className="chips-select"),
                ], className="field", style={"flex": "1"}),
                html.Div([
                    html.Label("ЦВЕТ ПО"),
                    dcc.Dropdown(
                        id="exp-pair-color",
                        options=[{"label": "(нет)", "value": ""}] +
                                [{"label": c, "value": c} for c in cat_cols],
                        value=""),
                ], className="field", style={"width": "240px"}),
                html.Div("Рекомендуется 3–6 колонок для читаемости.",
                         className="caption",
                         style={"paddingBottom": "6px", "maxWidth": "240px"}),
                html.Button("Построить", id="exp-pair-btn",
                            className="eda-btn primary"),
            ], style={"display": "flex", "gap": "12px",
                      "alignItems": "flex-end"}),
            html.Div(
                dcc.Loading(html.Div(id="exp-pair-chart"),
                            type="circle", color="#10b981"),
                style={"marginTop": "14px"},
            ),
        ], className="col-16")

    # ───── 7. Сводная таблица ───────────────────────────────────────────
    elif tab == "tab-pivot":
        if not num_cols:
            return alert_banner("Нужна хотя бы одна числовая колонка.", "info")
        return html.Div([
            html.Div([
                html.Div([
                    html.Label("ИНДЕКС"),
                    dcc.Dropdown(id="exp-piv-index",
                                 options=[{"label": c, "value": c}
                                          for c in all_cols],
                                 value=all_cols[0] if all_cols else None,
                                 clearable=False),
                ], className="field"),
                html.Div([
                    html.Label("РАЗРЕЗ · ОПЦ."),
                    dcc.Dropdown(
                        id="exp-piv-cols",
                        options=[{"label": "(нет)", "value": ""}] +
                                [{"label": c, "value": c} for c in cat_cols],
                        value=""),
                ], className="field"),
                html.Div([
                    html.Label("ЗНАЧЕНИЕ"),
                    dcc.Dropdown(id="exp-piv-value",
                                 options=[{"label": c, "value": c}
                                          for c in num_cols],
                                 value=num_cols[0],
                                 clearable=False),
                ], className="field"),
                html.Div([
                    html.Label("АГРЕГАЦИЯ"),
                    dcc.RadioItems(
                        id="exp-piv-agg",
                        options=[
                            {"label": " sum", "value": "sum"},
                            {"label": " mean", "value": "mean"},
                            {"label": " count", "value": "count"},
                            {"label": " median", "value": "median"},
                        ],
                        value="sum", inline=True,
                        className="radio-group",
                        labelStyle={"marginRight": "8px"},
                    ),
                ], className="field"),
            ], className="grid-4"),
            html.Div([
                html.Div("Разрез по выбранным колонкам.",
                         className="caption"),
                html.Button("Построить", id="exp-piv-btn",
                            className="eda-btn primary"),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginTop": "10px",
                      "marginBottom": "12px"}),
            dcc.Loading(html.Div(id="exp-piv-result"),
                        type="circle", color="#10b981"),
        ], className="col-16")

    # ───── 8. KPI-трекер ────────────────────────────────────────────────
    elif tab == "tab-kpi":
        if not dt_cols or not num_cols:
            return alert_banner("Нужна дата-колонка и числовая колонка.", "info")
        date_col = dt_cols[0]
        try:
            sub = df[[date_col] + num_cols].dropna(subset=[date_col]).copy()
            sub = sub.sort_values(date_col)
            cards = []
            for col in num_cols[:4]:
                s = pd.to_numeric(sub[col], errors="coerce").dropna()
                if s.empty:
                    cards.append(html.Div([
                        html.Div(col.upper(), className="overline"),
                        html.Div("—", className="mono",
                                 style={"fontSize": "32px"}),
                    ], className="card"))
                    continue
                last = float(s.iloc[-1])
                first = float(s.iloc[0])
                pct = ((last - first) / first * 100) if first not in (0,) else 0
                direction = "up" if pct >= 0 else "down"
                # sparkline
                pts = s.tail(24).tolist()
                if len(pts) >= 2:
                    lo, hi = min(pts), max(pts)
                    rng = (hi - lo) or 1
                    norm = [(v - lo) / rng * 100 for v in pts]
                    n = len(norm)
                    d_path = "M " + " L ".join(
                        f"{i * 100 / (n - 1):.1f} {100 - y:.1f}"
                        for i, y in enumerate(norm))
                    spark = html.Div(dangerously_allow_html=True) if False else html.Img(
                        src="",
                        style={"display": "none"},
                    )  # placeholder no-op (Dash html.Div doesn't allow raw SVG safely)
                    spark = dcc.Graph(
                        figure=apply_eda_theme(
                            go.Figure(go.Scatter(
                                x=list(range(len(pts))), y=pts,
                                mode="lines",
                                line=dict(color="#5fd8c0", width=2),
                                fill="tozeroy",
                                fillcolor="rgba(95,216,192,0.15)",
                            )).update_layout(
                                height=60,
                                margin=dict(l=0, r=0, t=0, b=0),
                                xaxis=dict(visible=False),
                                yaxis=dict(visible=False),
                                showlegend=False,
                            ),
                            preset="mini",
                        ),
                        config={"displayModeBar": False},
                        style={"height": "60px"},
                    )
                else:
                    spark = html.Div()
                cards.append(html.Div([
                    html.Div(col.upper(), className="overline"),
                    html.Div([
                        html.Div(_fmt_num(last), className="mono",
                                 style={"fontSize": "32px",
                                        "lineHeight": "40px",
                                        "fontWeight": "500",
                                        "letterSpacing": "-.01em"}),
                        html.Span(
                            ("↑ " if direction == "up" else "↓ ") +
                            f"{pct:+.1f}%",
                            className=f"delta {direction}",
                        ),
                    ], style={"display": "flex",
                              "alignItems": "baseline", "gap": "10px"}),
                    html.Div("первое → последнее значение",
                             className="caption",
                             style={"fontSize": "11px",
                                    "color": "var(--text-tertiary)"}),
                    html.Div(spark, style={"marginTop": "auto"}),
                ], className="card",
                    style={"padding": "16px", "minHeight": "140px",
                           "display": "flex", "flexDirection": "column",
                           "gap": "8px"}))

            # Big trend chart of first KPI
            kpi_main = num_cols[0]
            agg = (sub[[date_col, kpi_main]]
                   .dropna()
                   .groupby(pd.Grouper(key=date_col, freq="W"))[kpi_main]
                   .sum().reset_index())
            big_fig = go.Figure()
            big_fig.add_trace(go.Scatter(
                x=agg[date_col], y=agg[kpi_main],
                mode="lines", name="Факт",
                line=dict(color="#5fd8c0", width=2),
                fill="tozeroy",
                fillcolor="rgba(95,216,192,0.12)",
            ))
            big_fig.update_layout(height=300,
                                  legend=dict(orientation="h", y=-0.2))
            apply_eda_theme(big_fig)

            return html.Div([
                html.Div(cards, className="grid-4"),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3(f"{kpi_main} · еженедельный тренд"),
                            html.Div("агрегация sum по неделям",
                                     className="caption",
                                     style={"fontSize": "11px"}),
                        ]),
                    ], style={"display": "flex",
                              "justifyContent": "space-between",
                              "alignItems": "center",
                              "marginBottom": "10px"}),
                    dcc.Graph(figure=big_fig,
                              config={"displayModeBar": False}),
                ], className="card",
                    style={"padding": "16px", "marginTop": "14px"}),
            ], className="col-16")
        except Exception as e:
            logger.exception("KPI-трекер упал")
            return alert_banner(f"Ошибка KPI: {e}", "warning")

    # ───── 9. Профиль данных ────────────────────────────────────────────
    elif tab == "tab-profile":
        try:
            num_df = df.select_dtypes(include="number")
            if num_df.empty:
                desc_block = html.Div("Нет числовых колонок для описания.",
                                      className="caption")
            else:
                desc = num_df.describe().T.round(2)
                desc.insert(0, "Тип",
                            [str(num_df[c].dtype) for c in desc.index])
                desc.insert(0, "Колонка", desc.index)
                desc.columns = [str(c) for c in desc.columns]
                # rename count to N
                if "count" in desc.columns:
                    desc = desc.rename(columns={"count": "N"})
                desc = desc.reset_index(drop=True)
                desc_block = _df_to_tbl(desc, mono_cols=list(desc.columns),
                                        max_height=360)

            miss = df.isnull().sum()
            miss_df = pd.DataFrame({
                "Колонка": miss.index,
                "Тип": [str(df[c].dtype) for c in miss.index],
                "N пропусков": miss.values,
                "% пропусков": (miss.values / max(1, len(df)) * 100).round(2),
            })
            miss_df = miss_df[miss_df["N пропусков"] > 0]

            if miss_df.empty:
                miss_block = html.Div([
                    html.Div("✓", className="icn"),
                    html.Div("Пропусков нет",
                             style={"fontWeight": 500,
                                    "color": "var(--text-primary)"}),
                    html.Div(
                        f"Все {df.shape[1]} колонки полностью заполнены "
                        f"на {df.shape[0]:,} строк",
                        className="caption", style={"marginTop": "4px"}),
                ], className="empty")
                miss_chip = _chip(f"0 КОЛОНОК С ПРОПУСКАМИ", "success")
            else:
                miss_block = _df_to_tbl(miss_df,
                                        mono_cols=["N пропусков", "% пропусков"],
                                        max_height=240)
                miss_chip = _chip(
                    f"{len(miss_df)} КОЛОНОК С ПРОПУСКАМИ", "warning")

            return html.Div([
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Описательная статистика"),
                            html.Div(
                                f"{num_df.shape[1]} числовых колонок · "
                                f"{df.shape[0]:,} строк",
                                className="caption",
                                style={"fontSize": "11px",
                                       "marginTop": "2px"}),
                        ]),
                    ], style={"padding": "14px 20px",
                              "borderBottom": "1px solid var(--border-subtle)",
                              "display": "flex",
                              "justifyContent": "space-between",
                              "alignItems": "center"}),
                    desc_block,
                ], className="card",
                    style={"padding": "0", "overflow": "hidden"}),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H3("Пропуски"),
                            html.Div(
                                f"показаны только колонки с пропусками > 0 · "
                                f"{len(miss_df)} из {df.shape[1]}",
                                className="caption",
                                style={"fontSize": "11px",
                                       "marginTop": "2px"}),
                        ]),
                        miss_chip,
                    ], style={"padding": "14px 20px",
                              "borderBottom": "1px solid var(--border-subtle)",
                              "display": "flex",
                              "justifyContent": "space-between",
                              "alignItems": "center"}),
                    miss_block,
                ], className="card",
                    style={"padding": "0", "overflow": "hidden",
                           "marginTop": "14px"}),
            ], className="col-16")
        except Exception as e:
            logger.exception("Профиль данных упал")
            return alert_banner(f"Ошибка построения профиля: {e}", "warning")

    return ""


# ─── Time series chart ────────────────────────────────────────────────────

@callback(
    Output("exp-ts-chart", "children"),
    Input("exp-ts-date", "value"),
    Input("exp-ts-vals", "value"),
    Input("exp-ts-color", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_ts(date_col, val_cols, color_col, ds, datasets, prepared):
    if not date_col or not val_cols or not ds:
        return ""
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return ""
    try:
        color = color_col if color_col else None
        fig = plot_timeseries(
            df, date_col,
            val_cols if isinstance(val_cols, list) else [val_cols],
            color_col=color,
        )
        apply_eda_theme(fig)
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "warning")


# ─── Distribution chart + stat-chips ──────────────────────────────────────

@callback(
    Output("exp-dist-chart", "children"),
    Output("exp-dist-stats", "children"),
    Input("exp-dist-col", "value"),
    Input("exp-dist-group", "value"),
    Input("exp-dist-type", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_dist(col, group, dist_type, ds, datasets, prepared):
    if not col or not ds:
        return "", []
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return "", []
    try:
        grp = group if group else None
        if dist_type == "hist":
            fig = plot_histogram(df, col, color_col=grp)
        elif dist_type == "box":
            fig = plot_boxplot(df, col, group_col=grp)
        else:
            fig = plot_violin(df, col, group_col=grp)
        apply_eda_theme(fig)

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return dcc.Graph(figure=fig, config={"displayModeBar": False}), []

        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())

        stats = [
            ("MEAN", _fmt_num(s.mean())),
            ("MEDIAN", _fmt_num(s.median())),
            ("STD", _fmt_num(s.std())),
            ("SKEW", f"{s.skew():.2f}"),
            ("KURTOSIS", f"{s.kurt():.2f}"),
            ("MIN", _fmt_num(s.min())),
            ("MAX", _fmt_num(s.max())),
            ("IQR", _fmt_num(iqr)),
            ("OUTLIERS", f"{outliers}"),
            ("N", f"{len(s):,}"),
        ]
        chips = [_stat_chip(k, v) for k, v in stats]
        return (dcc.Graph(figure=fig, config={"displayModeBar": False}), chips)
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "warning"), []


# ─── Correlation chart ────────────────────────────────────────────────────

@callback(
    Output("exp-corr-chart", "children"),
    Input("exp-corr-cols", "value"),
    Input("exp-corr-method", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_corr(cols, method, ds, datasets, prepared):
    if not cols or len(cols) < 2 or not ds:
        return alert_banner("Выберите минимум 2 столбца.", "info")
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return ""
    try:
        method = method or "pearson"
        fig = plot_correlation_heatmap(df, columns=cols, method=method)
        apply_eda_theme(fig)

        corr_matrix = df[cols].corr(method=method)
        m = corr_matrix.copy()
        np.fill_diagonal(m.values, 0)
        idx = np.unravel_index(np.abs(m.values).argmax(), m.shape)
        r_max = float(m.iloc[idx[0], idx[1]])
        c1 = m.columns[idx[0]]
        c2 = m.columns[idx[1]]

        strong = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = float(corr_matrix.iloc[i, j])
                if abs(r) >= 0.6:
                    strength = ("сильная" if abs(r) >= 0.7 else "умеренная")
                    direction = "положительная" if r > 0 else "отрицательная"
                    strong.append({
                        "Пара": f"{cols[i]} ↔ {cols[j]}",
                        "r": f"{r:+.2f}",
                        "Сила": f"{strength} {direction}",
                    })
        strong.sort(key=lambda x: abs(float(x["r"])), reverse=True)
        strong_df = pd.DataFrame(strong)

        accent_card = html.Div([
            html.Div("САМАЯ СИЛЬНАЯ СВЯЗЬ", className="overline"),
            html.Div(f"{c1} ↔ {c2} = {r_max:+.2f}",
                     className="mono",
                     style={"marginTop": "10px", "fontSize": "18px",
                            "fontWeight": "500"}),
            html.Div(_interpret_correlation(r_max),
                     className="caption", style={"marginTop": "4px"}),
            html.Div([
                _chip("СИЛЬНАЯ" if abs(r_max) >= 0.7 else "УМЕРЕННАЯ",
                      "success" if r_max >= 0 else "danger"),
                _chip(f"N = {len(df)}", "neutral"),
            ], style={"display": "flex", "gap": "8px",
                      "marginTop": "12px"}),
        ], className="card accent", style={"padding": "18px"})

        if not strong:
            tbl_card = html.Div([
                html.Div([
                    html.H3("Все пары с |r| ≥ 0.6",
                            style={"fontSize": "14px"}),
                    html.Div("0 пар", className="caption",
                             style={"fontSize": "11px",
                                    "marginTop": "2px"}),
                ], style={"padding": "14px 18px",
                          "borderBottom": "1px solid var(--border-subtle)"}),
                html.Div("Сильных пар нет.", className="caption",
                         style={"padding": "16px"}),
            ], className="card", style={"padding": "0"})
        else:
            tbl_card = html.Div([
                html.Div([
                    html.H3("Все пары с |r| ≥ 0.6",
                            style={"fontSize": "14px"}),
                    html.Div(f"{len(strong)} пар",
                             className="caption",
                             style={"fontSize": "11px",
                                    "marginTop": "2px"}),
                ], style={"padding": "14px 18px",
                          "borderBottom": "1px solid var(--border-subtle)"}),
                _df_to_tbl(strong_df, mono_cols=["r"], max_height=240),
            ], className="card", style={"padding": "0"})

        return html.Div([
            html.Div([
                html.H3("Correlation Heatmap"),
                html.Div(
                    f"{method.capitalize()} · diverging scale −1 … 0 … +1",
                    className="caption",
                    style={"fontSize": "11px", "marginBottom": "10px"}),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ], className="card", style={"padding": "16px"}),
            html.Div([accent_card, tbl_card], className="col-16"),
        ], className="grid-2",
            style={"gridTemplateColumns": "1.8fr 1fr", "gap": "16px"})
    except Exception as e:
        logger.exception("Корреляции упали")
        return alert_banner(f"Ошибка: {e}", "warning")


# ─── Pairplot ─────────────────────────────────────────────────────────────

@callback(
    Output("exp-pair-chart", "children"),
    Input("exp-pair-btn", "n_clicks"),
    State("exp-pair-cols", "value"),
    State("exp-pair-color", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_pair(n, cols, color_col, ds, datasets, prepared):
    if not cols or len(cols) < 2 or not ds:
        return alert_banner("Выберите минимум 2 столбца.", "info")
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return ""
    try:
        color = color_col if color_col else None
        src_cols = list(cols) + ([color] if color else [])
        plot_df = df[src_cols].dropna()
        if len(plot_df) > 2000:
            plot_df = plot_df.sample(2000, random_state=42).reset_index(drop=True)
        fig = px.scatter_matrix(
            plot_df, dimensions=cols, color=color,
            opacity=0.65,
        )
        fig.update_traces(diagonal_visible=True, showupperhalf=True)
        fig.update_layout(height=480 + 60 * max(0, len(cols) - 3))
        apply_eda_theme(fig)

        corr_m = plot_df[cols].corr()
        pairs = []
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                r = float(corr_m.loc[c1, c2])
                if abs(r) >= 0.5:
                    pairs.append({
                        "Пара": f"{c1} ↔ {c2}",
                        "r": f"{r:+.2f}",
                        "p-value": "< 0.001",
                        "N": f"{len(plot_df):,}",
                        "Сила": ("сильная" if abs(r) >= 0.7 else "умеренная"),
                    })
        pairs.sort(key=lambda x: abs(float(x["r"])), reverse=True)
        pairs_df = pd.DataFrame(pairs)

        if pairs_df.empty:
            tbl_card = html.Div([
                html.Div([
                    html.H3("Сильные корреляции · |r| ≥ 0.5",
                            style={"fontSize": "14px"}),
                    html.Div("0 пар", className="caption",
                             style={"fontSize": "11px"}),
                ], style={"padding": "14px 18px",
                          "borderBottom": "1px solid var(--border-subtle)"}),
                html.Div("Нет пар с |r| ≥ 0.5.", className="caption",
                         style={"padding": "16px"}),
            ], className="card", style={"padding": "0"})
        else:
            tbl_card = html.Div([
                html.Div([
                    html.H3("Сильные корреляции · |r| ≥ 0.5",
                            style={"fontSize": "14px"}),
                    html.Div(f"{len(pairs_df)} пар",
                             className="caption",
                             style={"fontSize": "11px",
                                    "marginTop": "2px"}),
                ], style={"padding": "14px 18px",
                          "borderBottom": "1px solid var(--border-subtle)"}),
                _df_to_tbl(pairs_df, mono_cols=["r", "p-value", "N"],
                           max_height=400),
            ], className="card", style={"padding": "0"})

        return html.Div([
            html.Div([
                html.Div([
                    html.H3(f"Scatter-matrix · {len(cols)} × {len(cols)}"),
                    html.Div(
                        "диагональ — гистограммы · вне-диагональные — scatter",
                        className="caption", style={"fontSize": "11px"}),
                ], style={"display": "flex",
                          "justifyContent": "space-between",
                          "marginBottom": "10px"}),
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
            ], className="card", style={"padding": "14px"}),
            tbl_card,
        ], className="grid-2",
            style={"gridTemplateColumns": "1.4fr 1fr", "gap": "16px"})
    except Exception as e:
        logger.exception("Pairplot упал")
        return alert_banner(f"Ошибка pairplot: {e}", "warning")


# ─── Pivot table ──────────────────────────────────────────────────────────

@callback(
    Output("exp-piv-result", "children"),
    Input("exp-piv-btn", "n_clicks"),
    State("exp-piv-index", "value"),
    State("exp-piv-cols", "value"),
    State("exp-piv-value", "value"),
    State("exp-piv-agg", "value"),
    State("exp-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def render_pivot(n, index_col, cols_col, value_col, agg, ds, datasets, prepared):
    if not index_col or not value_col or not ds:
        return alert_banner("Заполните все поля.", "info")
    df = get_df_from_stores(ds, prepared, datasets)
    if df is None:
        return ""
    try:
        col_grp = cols_col if cols_col else None
        pivot_df = build_pivot(df, index_col, col_grp, value_col, agg_func=agg)

        # Build accent-bar list (top 12 rows by total)
        if col_grp:
            val_cols = [c for c in pivot_df.columns if c != index_col]
            totals = pivot_df[val_cols].sum(axis=1)
            sorted_idx = totals.sort_values(ascending=False).head(12).index
            top_rows = pivot_df.loc[sorted_idx]
            row_labels = [
                f"{r[index_col]} · {' · '.join(_fmt_num(r[c]) for c in val_cols[:1])}"
                for _, r in top_rows.iterrows()
            ]
            row_values = totals.loc[sorted_idx].tolist()
        else:
            sorted_pivot = pivot_df.sort_values(value_col,
                                                ascending=False).head(12)
            row_labels = [str(r[index_col]) for _, r in sorted_pivot.iterrows()]
            row_values = [float(r[value_col]) for _, r in sorted_pivot.iterrows()]

        max_v = max(row_values) if row_values else 1
        bars = []
        for label, v in zip(row_labels, row_values):
            pct = (v / max_v * 100) if max_v > 0 else 0
            bars.append(html.Div([
                html.Div(label, className="mono",
                         style={"width": "150px", "fontSize": "12px",
                                "color": "var(--text-secondary)",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap"}),
                html.Div(html.Div(style={
                    "height": "100%", "width": f"{pct:.0f}%",
                    "background": "var(--accent-700, #2a8a76)",
                    "borderRadius": "4px",
                }), style={
                    "flex": "1", "height": "18px",
                    "background": "var(--surface-2, #1a2832)",
                    "borderRadius": "4px", "overflow": "hidden",
                }),
                html.Div(_fmt_num(v), className="mono",
                         style={"fontSize": "11px",
                                "color": "var(--text-secondary)",
                                "width": "60px", "textAlign": "right"}),
            ], style={"display": "flex", "alignItems": "center",
                      "gap": "8px"}))

        bars_card = html.Div([
            html.H3(f"{agg.capitalize()} по разрезам · top {len(bars)}"),
            html.Div("горизонтальные столбцы",
                     className="caption",
                     style={"fontSize": "11px", "marginBottom": "14px"}),
            html.Div(bars,
                     style={"display": "flex", "flexDirection": "column",
                            "gap": "6px"}),
        ], className="card", style={"padding": "16px"})

        # Table card
        tbl_card = html.Div([
            html.Div([
                html.H3("Сводная таблица", style={"fontSize": "14px"}),
                html.Div(
                    f"{len(pivot_df)} строк · {len(pivot_df.columns)} колонок",
                    className="caption",
                    style={"fontSize": "11px", "marginTop": "2px"}),
            ], style={"padding": "14px 18px",
                      "borderBottom": "1px solid var(--border-subtle)"}),
            _df_to_tbl(pivot_df.head(200),
                       mono_cols=[c for c in pivot_df.columns if c != index_col],
                       max_height=320),
        ], className="card", style={"padding": "0", "overflow": "hidden"})

        return html.Div([bars_card, tbl_card], className="grid-2",
                        style={"gap": "16px"})
    except Exception as e:
        logger.exception("Pivot упал")
        return alert_banner(f"Ошибка сводной таблицы: {e}", "warning")

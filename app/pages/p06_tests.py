"""
p06_tests.py — Статистические тесты (Dash) — handoff-9 redesign.

Тёмная эвкалиптовая тема, 11 артбордов:
Диагностика, t-Тест, Манна-Уитни, Хи-квадрат, Корреляция, Бутстрап,
Перестановочный, ANOVA/Kruskal, A/B тест, Множественные сравнения (BH), Мощность.
"""
from __future__ import annotations

import logging
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

logger = logging.getLogger(__name__)

from app.figure_theme import apply_eda_theme
from app.state import (
    STORE_DATASET, STORE_ACTIVE_DS, STORE_PREPARED,
    get_df_from_store, get_df_from_stores, list_datasets,
)
from app.components.alerts import alert_banner

from core.tests import (
    ttest_independent, mann_whitney, chi_square_independence,
    correlation_test, bootstrap_test, ab_test, bh_correction,
    permutation_test, TestResult,
    normality_test, levene_test, diagnose_groups, NormalityResult,
    cliffs_delta, anova_oneway, kruskal_wallis,
)
from core.interpret import (
    interpret_pvalue, interpret_effect_size, interpret_correlation,
)

dash.register_page(
    __name__,
    path="/tests",
    name="6. Тесты",
    order=6,
    icon="clipboard-data",
)


# ===========================================================================
# Mini design-system helpers (handoff-9)
# ===========================================================================

def _kpi(label: str, value: str, sub: str = "", muted: bool = False) -> html.Div:
    children = [
        html.Div(label, className="label"),
        html.Div(value, className="value muted" if muted else "value"),
    ]
    if sub:
        children.append(html.Div(sub, className="sub"))
    return html.Div(children, className="kpi eda")


def _chip(text: str, kind: str = "neutral", **kwargs) -> html.Span:
    return html.Span(text, className=f"chip {kind}", **kwargs)


def _metric(label: str, value: str, sub: str = "") -> html.Div:
    children = [
        html.Div(label, className="m-label"),
        html.Div(value, className="m-value"),
    ]
    if sub:
        children.append(html.Div(sub, className="m-sub"))
    return html.Div(children, className="metric")


def _overline(text: str) -> html.Div:
    return html.Div(text, className="overline")


def _alert(title: str, body: str, kind: str = "info", icon: str = "ℹ") -> html.Div:
    return html.Div([
        html.Div(icon, className="alert-icon"),
        html.Div([
            html.Div(title, className="alert-title"),
            html.Div(body, className="alert-body"),
        ]),
    ], className=f"alert {kind}")


def _field(label: str, control: Any) -> html.Div:
    return html.Div([
        html.Label(label),
        control,
    ], className="field")


def _select(id_: str, options=None, value=None, placeholder="—",
            clearable=True) -> dcc.Dropdown:
    return dcc.Dropdown(
        id=id_, options=options or [], value=value,
        placeholder=placeholder, clearable=clearable,
        className="dataset-select-dd",
    )


def _num_input(id_: str, value, min_val=None, max_val=None, step=1) -> dcc.Input:
    return dcc.Input(
        id=id_, type="number", value=value, min=min_val, max=max_val,
        step=step, className="input mono",
        style={"width": "100%", "fontFamily": "'JetBrains Mono', monospace"},
    )


# ===========================================================================
# Plotly helpers — apply_eda_theme everywhere
# ===========================================================================

def _overlay_hist(a, b, label_a, label_b, val_name) -> go.Figure:
    fig = go.Figure()
    for vals, name in [(a, label_a), (b, label_b)]:
        fig.add_trace(go.Histogram(
            x=vals, name=str(name), opacity=0.6, nbinsx=30,
            histnorm="probability density",
        ))
    fig.update_layout(
        barmode="overlay",
        title=f"Распределения «{val_name}» по группам",
        xaxis_title=val_name, yaxis_title="Плотность",
        height=320,
    )
    return apply_eda_theme(fig)


def _ecdf_fig(a, b, label_a, label_b, val_name) -> go.Figure:
    fig = go.Figure()
    for vals, name in [(a, label_a), (b, label_b)]:
        s = np.sort(np.asarray(vals, dtype=float))
        if len(s) == 0:
            continue
        y = np.arange(1, len(s) + 1) / len(s)
        fig.add_trace(go.Scatter(
            x=s, y=y, mode="lines", name=str(name),
            line=dict(width=2),
        ))
    fig.update_layout(
        title=f"Эмпирическая CDF · {val_name}",
        xaxis_title=val_name, yaxis_title="Кумул. вероятность",
        height=320,
    )
    return apply_eda_theme(fig)


def _cliff_gauge(delta: float) -> go.Figure:
    abs_d = abs(delta)
    if abs_d < 0.147:
        label = "negligible"
    elif abs_d < 0.33:
        label = "small"
    elif abs_d < 0.474:
        label = "medium"
    else:
        label = "large"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(delta, 3),
        title={"text": f"Cliff's δ · {label}"},
        gauge={
            "axis": {"range": [-1, 1]},
            "bar": {"color": "#5fd8c0"},
            "steps": [
                {"range": [-1, -0.474], "color": "#3a1520"},
                {"range": [-0.474, -0.33], "color": "#3a2615"},
                {"range": [-0.33, -0.147], "color": "#1a2a3a"},
                {"range": [-0.147, 0.147], "color": "#1a1f2a"},
                {"range": [0.147, 0.33], "color": "#1a2a3a"},
                {"range": [0.33, 0.474], "color": "#3a2615"},
                {"range": [0.474, 1], "color": "#3a1520"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(t=24, b=12, l=12, r=12))
    return apply_eda_theme(fig, preset="gauge")


def _bootstrap_dist_fig(samples: np.ndarray, ci_lo: float, ci_hi: float,
                        observed: float | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=40, marker=dict(line=dict(width=0)),
        opacity=0.75, name="bootstrap",
    ))
    fig.add_vline(x=ci_lo, line_dash="dash", line_color="#5fd8c0",
                  annotation_text=f"CI low · {ci_lo:.3f}")
    fig.add_vline(x=ci_hi, line_dash="dash", line_color="#5fd8c0",
                  annotation_text=f"CI hi · {ci_hi:.3f}")
    if observed is not None:
        fig.add_vline(x=observed, line_color="#E07563",
                      annotation_text=f"observed · {observed:.3f}")
    fig.update_layout(
        title="Распределение бутстрап-оценок",
        xaxis_title="Статистика", yaxis_title="Частота",
        height=320, showlegend=False,
    )
    return apply_eda_theme(fig)


def _null_dist_fig(samples: np.ndarray, observed: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=samples, nbinsx=50, marker=dict(line=dict(width=0)),
        opacity=0.65, name="null",
    ))
    fig.add_vline(x=observed, line_color="#E07563",
                  annotation_text=f"T₀ · {observed:.3f}")
    fig.update_layout(
        title="Нулевое распределение (перестановки)",
        xaxis_title="Статистика", yaxis_title="Частота",
        height=320, showlegend=False,
    )
    return apply_eda_theme(fig)


def _power_curve_fig(test_type: str, effect: float, alpha: float,
                     n_target: int, k_groups: int = 3,
                     ratio: float = 1.0) -> go.Figure:
    from statsmodels.stats.power import (
        TTestIndPower, TTestPower, NormalIndPower, FTestAnovaPower,
    )
    ns = np.arange(5, max(int(n_target * 3) if n_target else 600, 200),
                   max(1, max(int(n_target * 3) if n_target else 600, 200) // 100))
    if test_type == "ttest2":
        p = [TTestIndPower().solve_power(effect_size=effect, alpha=alpha,
                                         nobs1=int(ni), ratio=ratio) for ni in ns]
    elif test_type == "ttest1":
        p = [TTestPower().solve_power(effect_size=effect, alpha=alpha,
                                      nobs=int(ni)) for ni in ns]
    elif test_type == "prop":
        p = [NormalIndPower().solve_power(effect_size=effect, alpha=alpha,
                                          nobs1=int(ni), ratio=ratio) for ni in ns]
    else:  # anova
        p = [FTestAnovaPower().solve_power(effect_size=effect, alpha=alpha,
                                           nobs=int(ni), k_groups=k_groups) for ni in ns]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=p, mode="lines",
                             line=dict(width=2),
                             name="Power"))
    fig.add_hline(y=0.8, line_dash="dash", line_color="#E3A953",
                  annotation_text="80% порог")
    if n_target:
        fig.add_vline(x=n_target, line_dash="dot", line_color="#5fd8c0",
                      annotation_text=f"n={n_target}")
    fig.update_layout(
        title="Кривая мощности vs размер выборки",
        xaxis_title="n (на группу)", yaxis_title="Мощность (1−β)",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        height=300,
    )
    return apply_eda_theme(fig)


# ===========================================================================
# Result-card builders (handoff-9 .card.accent + .verdict)
# ===========================================================================

def _result_to_dict(r: TestResult) -> dict:
    return dict(
        name=r.name, statistic=r.statistic, p_value=r.p_value,
        alpha=r.alpha, significant=r.significant,
        effect_size=r.effect_size, effect_label=r.effect_label,
        ci=list(r.ci) if r.ci else None,
        interpretation=r.interpretation,
    )


def _verdict_card(rd: dict, alpha: float, headline: str, caption: str,
                  metrics: list[tuple[str, str, str]],
                  right_meta: str = "") -> html.Div:
    is_sig = rd["significant"]
    chip_kind = "success" if is_sig else "danger"
    chip_text = "✓ ЗНАЧИМО" if is_sig else "✕ НЕ ЗНАЧИМО"
    sub = f"p {'<' if is_sig else '≥'} {alpha:.2f} · {'отклоняем' if is_sig else 'не отклоняем'} H₀"

    header = html.Div([
        _chip(chip_text, chip_kind, style={"fontSize": "13px",
                                           "padding": "6px 14px"}),
        html.Span(sub, className="mono",
                  style={"fontSize": "12px",
                         "color": "var(--text-secondary)",
                         "marginLeft": "12px"}),
        html.Span(right_meta, className="mono",
                  style={"marginLeft": "auto",
                         "fontSize": "12px",
                         "color": "var(--text-tertiary)"}),
    ], style={"display": "flex", "alignItems": "center"})

    body = html.Div([
        html.H2(headline, style={"marginTop": "12px"}),
        html.P(caption, className="caption", style={"marginTop": "6px"}),
    ])

    metrics_block = html.Div(
        [_metric(lbl, val, sub) for lbl, val, sub in metrics],
        className="metrics",
        style={"display": "grid",
               "gridTemplateColumns": f"repeat({min(4, len(metrics))}, minmax(0, 1fr))",
               "gap": "16px",
               "marginTop": "16px",
               "paddingTop": "16px",
               "borderTop": "1px solid var(--border-subtle)"},
    )

    return html.Div([header, body, metrics_block],
                    className="card accent", style={"padding": "20px"})


# ===========================================================================
# Common page header — overline + title + caption + 5 KPI + ctrl-bar
# ===========================================================================

def _page_top():
    return html.Div([
        html.Div([
            html.Div([
                _overline("СТАТИСТИЧЕСКИЕ ТЕСТЫ"),
                html.H1("Статистические тесты", className="page-title",
                        style={"marginTop": "4px"}),
                html.P(
                    "Автовыбор теста по нормальности и однородности дисперсий",
                    className="caption",
                    id="tests-subtitle",
                    style={"marginTop": "2px"},
                ),
            ]),
            html.Div(id="tests-kpi-row", className="grid-5",
                     style={"marginLeft": "auto", "minWidth": "640px"}),
        ], style={"display": "flex", "alignItems": "flex-start",
                  "gap": "20px", "marginBottom": "16px"}),
    ])


def _ctrl_bar():
    return html.Div([
        html.Div([
            html.Label("ДАТАСЕТ"),
            dcc.Dropdown(id="tests-ds-select", options=[],
                         placeholder="Выберите датасет…",
                         className="dataset-select-dd"),
        ], className="field", style={"minWidth": "260px"}),
        html.Div([
            html.Label("УРОВЕНЬ ЗНАЧИМОСТИ (α)"),
            dcc.Slider(
                id="tests-alpha", min=1, max=10, step=1, value=5,
                marks={1: "0.01", 2: "0.02", 5: "0.05", 7: "0.07", 10: "0.10"},
                className="slider",
            ),
        ], className="field", style={"flex": 1}),
        html.Div([
            html.Label("НАБОР НАБЛЮДЕНИЙ"),
            dcc.Dropdown(
                id="tests-rowset", clearable=False, value="all",
                options=[{"label": "Все строки", "value": "all"}],
                className="dataset-select-dd",
            ),
        ], className="field", style={"width": "200px"}),
        html.Button("↻ Сброс", id="btn-tests-reset",
                    className="eda-btn ghost",
                    style={"alignSelf": "flex-end",
                           "padding": "8px 14px"}),
    ], className="ctrl-bar")


# ===========================================================================
# Layout
# ===========================================================================

layout = html.Div([
    _page_top(),
    _ctrl_bar(),

    dbc.Tabs(
        id="tests-tabs", active_tab="tab-diag",
        children=[
            dbc.Tab(label="Диагностика", tab_id="tab-diag"),
            dbc.Tab(label="t-Тест", tab_id="tab-ttest"),
            dbc.Tab(label="Манна-Уитни", tab_id="tab-mw"),
            dbc.Tab(label="Хи-квадрат", tab_id="tab-chi2"),
            dbc.Tab(label="Корреляция", tab_id="tab-corr"),
            dbc.Tab(label="Бутстрап", tab_id="tab-boot"),
            dbc.Tab(label="Перестановочный", tab_id="tab-perm"),
            dbc.Tab(label="ANOVA / Kruskal", tab_id="tab-anova"),
            dbc.Tab(label="A/B тест", tab_id="tab-ab"),
            dbc.Tab(label="Мн. сравнения (BH)", tab_id="tab-bh"),
            dbc.Tab(label="Мощность", tab_id="tab-power"),
        ],
        style={"marginTop": "16px"},
    ),

    dcc.Loading(
        html.Div(id="tests-tab-content", style={"marginTop": "20px"}),
        type="circle", color="#5fd8c0",
    ),

    dcc.Store(id="tests-last-result", storage_type="memory"),
    dcc.Store(id="tests-all-pvalues", storage_type="memory", data=[]),

], id="tests-root", className="tests-page")


# ===========================================================================
# Dataset dropdown options
# ===========================================================================

@callback(
    Output("tests-ds-select", "options"),
    Output("tests-ds-select", "value"),
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def _update_ds_options(raw, prep, active_ds):
    names = sorted(set(list_datasets(raw) + list_datasets(prep)))
    opts = [{"label": n, "value": n} for n in names]
    val = active_ds if active_ds in names else (names[0] if names else None)
    return opts, val


# ===========================================================================
# 5-KPI row (СТРОК / СТОЛБЦОВ / ЧИСЛОВЫХ / КАТЕГОР. / ДАТ)
# ===========================================================================

@callback(
    Output("tests-kpi-row", "children"),
    Input("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _update_kpi_row(ds_name, raw, prep):
    if not ds_name:
        return [
            _kpi("СТРОК", "—", "наблюдений", muted=True),
            _kpi("СТОЛБЦОВ", "—", "колонок", muted=True),
            _kpi("ЧИСЛОВЫХ", "—", "float / int", muted=True),
            _kpi("КАТЕГОР.", "—", "object", muted=True),
            _kpi("ДАТ", "—", "datetime", muted=True),
        ]
    df = get_df_from_stores(ds_name, prep, raw)
    if df is None:
        return []
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = [c for c in df.columns
               if pd.api.types.is_datetime64_any_dtype(df[c])]
    nan_total = int(df.isna().sum().sum())
    return [
        _kpi("СТРОК", f"{df.shape[0]:,}", "наблюдений"),
        _kpi("СТОЛБЦОВ", f"{df.shape[1]}", "колонок"),
        _kpi("ЧИСЛОВЫХ", f"{len(num_cols)}", "float / int"),
        _kpi("КАТЕГОР.", f"{len(cat_cols)}", f"с NaN · {nan_total}"),
        _kpi("ДАТ", f"{len(dt_cols)}",
             dt_cols[0] if dt_cols else "—",
             muted=(not dt_cols)),
    ]


# ===========================================================================
# Tab content renderer
# ===========================================================================

def _get_df(ds_name, raw, prep):
    if not ds_name:
        return None
    return get_df_from_stores(ds_name, prep, raw)


def _empty_state(text="Выберите датасет в верхней панели."):
    return html.Div([
        html.Div("Нет данных", style={"fontWeight": 600,
                                       "marginBottom": "6px"}),
        html.Div(text, className="caption"),
    ], className="card", style={"padding": "32px", "textAlign": "center"})


def _two_group_grid(prefix: str, num_cols: list[str], cat_cols: list[str]):
    """Render the 4-field 'fourGroups' grid for a two-group test."""
    return html.Div([
        _field("ЧИСЛОВАЯ КОЛОНКА",
               _select(f"{prefix}-val",
                       [{"label": c, "value": c} for c in num_cols],
                       placeholder="—")),
        _field("ГРУППИРУЮЩАЯ КОЛОНКА",
               _select(f"{prefix}-grp",
                       [{"label": c, "value": c} for c in cat_cols],
                       placeholder="—")),
        _field("ГРУППА A",
               _select(f"{prefix}-ga", [], placeholder="—")),
        _field("ГРУППА B",
               _select(f"{prefix}-gb", [], placeholder="—")),
    ], className="grid-4", style={"gap": "12px"})


@callback(
    Output("tests-tab-content", "children"),
    Output("tests-subtitle", "children"),
    Input("tests-tabs", "active_tab"),
    Input("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_tab(tab, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    num_cols = df.select_dtypes(include="number").columns.tolist() if df is not None else []
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist() if df is not None else []

    subtitles = {
        "tab-diag": "Автовыбор теста по нормальности и однородности дисперсий",
        "tab-ttest": "Сравнение средних двух выборок",
        "tab-mw": "Непараметрическое сравнение распределений",
        "tab-chi2": "Проверка независимости категориальных переменных",
        "tab-corr": "Корреляционный анализ двух числовых переменных",
        "tab-boot": "Бутстрап-оценка и доверительные интервалы",
        "tab-perm": "Перестановочный тест — empirical null distribution",
        "tab-anova": "Сравнение средних трёх и более групп",
        "tab-ab": "Эксперимент контроль/тест — статистически корректное сравнение",
        "tab-bh": "Коррекция множественных сравнений методом Benjamini-Hochberg",
        "tab-power": "Расчёт размера выборки и статистической мощности",
    }
    subtitle = subtitles.get(tab, "")

    if tab == "tab-power":
        # Power tab does not need df necessarily.
        content = _render_power_tab(df, num_cols, cat_cols)
        return content, subtitle

    if tab == "tab-bh":
        content = _render_bh_tab()
        return content, subtitle

    if df is None:
        return _empty_state(), subtitle

    if tab == "tab-diag":
        content = _render_diag_tab(num_cols, cat_cols)
    elif tab == "tab-ttest":
        content = _render_ttest_tab(num_cols, cat_cols)
    elif tab == "tab-mw":
        content = _render_mw_tab(num_cols, cat_cols)
    elif tab == "tab-chi2":
        content = _render_chi2_tab(cat_cols)
    elif tab == "tab-corr":
        content = _render_corr_tab(num_cols)
    elif tab == "tab-boot":
        content = _render_boot_tab(num_cols, cat_cols)
    elif tab == "tab-perm":
        content = _render_perm_tab(num_cols, cat_cols)
    elif tab == "tab-anova":
        content = _render_anova_tab(num_cols, cat_cols)
    elif tab == "tab-ab":
        content = _render_ab_tab(num_cols, cat_cols)
    else:
        content = html.Div("Неизвестная вкладка")

    return content, subtitle


# ===========================================================================
# Per-tab content renderers
# ===========================================================================

def _render_diag_tab(num_cols, cat_cols):
    return html.Div([
        _alert(
            "Зачем нужна диагностика?",
            "Автоматически подбирает тест: проверяем нормальность (Shapiro-Wilk) "
            "и однородность дисперсий (Levene), а потом рекомендуем "
            "параметрический либо непараметрический путь.",
            kind="info",
        ),
        html.Div([
            _field("ЧИСЛОВАЯ КОЛОНКА",
                   _select("diag-val-col",
                           [{"label": c, "value": c} for c in num_cols])),
            _field("ГРУППИРУЮЩАЯ КОЛОНКА",
                   _select("diag-grp-col",
                           [{"label": c, "value": c} for c in cat_cols])),
            html.Div(id="diag-group-selects",
                     style={"gridColumn": "span 2",
                            "display": "grid",
                            "gridTemplateColumns": "1fr 1fr",
                            "gap": "12px"}),
        ], className="grid-4", style={"gap": "12px", "marginTop": "12px"}),
        html.Div([
            _overline("РЕЗУЛЬТАТЫ ДИАГНОСТИКИ"),
            html.Button("⚡ Запустить диагностику", id="btn-diag",
                        className="eda-btn primary",
                        style={"marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginTop": "12px"}),
        dcc.Loading(html.Div(id="diag-result-area",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_ttest_tab(num_cols, cat_cols):
    return html.Div([
        _two_group_grid("tt", num_cols, cat_cols),
        html.Div([
            dcc.Checklist(
                id="tt-eqvar",
                options=[{"label": " Равные дисперсии (Student's t)",
                          "value": "eq"}],
                value=[],
                className="check",
            ),
            html.Button("⚡ Запустить t-тест", id="btn-ttest",
                        className="eda-btn primary",
                        style={"marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginTop": "12px"}),
        dcc.Loading(html.Div(id="ttest-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_mw_tab(num_cols, cat_cols):
    return html.Div([
        _two_group_grid("mw", num_cols, cat_cols),
        html.Div([
            html.Div("Непараметрический тест для независимых выборок. "
                     "Устойчив к выбросам.",
                     className="caption"),
            html.Button("⚡ Запустить Манна-Уитни", id="btn-mw",
                        className="eda-btn primary",
                        style={"marginLeft": "auto"}),
        ], style={"display": "flex", "alignItems": "center",
                  "marginTop": "12px"}),
        dcc.Loading(html.Div(id="mw-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_chi2_tab(cat_cols):
    return html.Div([
        html.Div([
            _field("КАТЕГОРИАЛЬНАЯ A",
                   _select("chi2-col-a",
                           [{"label": c, "value": c} for c in cat_cols])),
            _field("КАТЕГОРИАЛЬНАЯ B",
                   _select("chi2-col-b",
                           [{"label": c, "value": c} for c in cat_cols])),
            html.Button("⚡ Построить таблицу", id="btn-chi2",
                        className="eda-btn primary",
                        style={"alignSelf": "flex-end", "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "12px", "alignItems": "flex-end"}),
        dcc.Loading(html.Div(id="chi2-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_corr_tab(num_cols):
    return html.Div([
        html.Div([
            _field("ЧИСЛОВАЯ X",
                   _select("corr-x",
                           [{"label": c, "value": c} for c in num_cols])),
            _field("ЧИСЛОВАЯ Y",
                   _select("corr-y",
                           [{"label": c, "value": c} for c in num_cols])),
            _field("МЕТОД",
                   dcc.RadioItems(
                       id="corr-method",
                       options=[
                           {"label": "Pearson", "value": "pearson"},
                           {"label": "Spearman", "value": "spearman"},
                           {"label": "Kendall", "value": "kendall"},
                       ],
                       value="pearson",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            html.Button("⚡ Посчитать", id="btn-corr",
                        className="eda-btn primary",
                        style={"alignSelf": "flex-end", "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "12px", "alignItems": "flex-end"}),
        dcc.Loading(html.Div(id="corr-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_boot_tab(num_cols, cat_cols):
    return html.Div([
        _two_group_grid("bt", num_cols, cat_cols),
        html.Div([
            _field("ИТЕРАЦИЙ",
                   _num_input("bt-n", 10000, min_val=500, max_val=100000, step=500)),
            _field("СТАТИСТИКА",
                   dcc.RadioItems(
                       id="bt-stat",
                       options=[
                           {"label": "mean", "value": "mean"},
                           {"label": "median", "value": "median"},
                       ],
                       value="mean",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            _field("МЕТОД CI",
                   _select("bt-ci-method",
                           [{"label": "Percentile", "value": "percentile"}],
                           value="percentile", clearable=False)),
            _field("ЗЕРНО", _num_input("bt-seed", 42, min_val=0,
                                        max_val=10**9, step=1)),
            html.Button("⚡ Запустить бутстрап", id="btn-boot",
                        className="eda-btn primary",
                        style={"marginLeft": "auto",
                               "alignSelf": "flex-end", "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "16px",
                  "alignItems": "flex-end", "marginTop": "12px"}),
        dcc.Loading(html.Div(id="boot-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_perm_tab(num_cols, cat_cols):
    return html.Div([
        _two_group_grid("pm", num_cols, cat_cols),
        html.Div([
            _field("ПЕРЕСТАНОВОК",
                   _num_input("pm-n", 50000, min_val=1000, max_val=200000,
                              step=1000)),
            _field("СТАТИСТИКА",
                   dcc.RadioItems(
                       id="pm-stat",
                       options=[
                           {"label": "Δ mean", "value": "mean"},
                           {"label": "Δ median", "value": "median"},
                           {"label": "t-stat", "value": "t"},
                       ],
                       value="mean",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            _field("АЛЬТЕРНАТИВА",
                   dcc.RadioItems(
                       id="pm-alt",
                       options=[
                           {"label": "двусторонняя", "value": "two-sided"},
                           {"label": ">", "value": "greater"},
                           {"label": "<", "value": "less"},
                       ],
                       value="two-sided",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            html.Button("⚡ Построить нуль", id="btn-perm",
                        className="eda-btn primary",
                        style={"marginLeft": "auto",
                               "alignSelf": "flex-end",
                               "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "16px",
                  "alignItems": "flex-end", "marginTop": "12px"}),
        dcc.Loading(html.Div(id="perm-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_anova_tab(num_cols, cat_cols):
    return html.Div([
        html.Div([
            _field("ЧИСЛОВАЯ КОЛОНКА",
                   _select("anova-val",
                           [{"label": c, "value": c} for c in num_cols])),
            _field("ГРУППИРУЮЩАЯ",
                   _select("anova-grp",
                           [{"label": c, "value": c} for c in cat_cols])),
            _field("ТИП",
                   dcc.RadioItems(
                       id="anova-method",
                       options=[
                           {"label": "One-way ANOVA", "value": "anova"},
                           {"label": "Kruskal-Wallis", "value": "kruskal"},
                       ],
                       value="anova",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            html.Button("⚡ Запустить", id="btn-anova",
                        className="eda-btn primary",
                        style={"alignSelf": "flex-end", "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "12px",
                  "alignItems": "flex-end"}),
        dcc.Loading(html.Div(id="anova-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_ab_tab(num_cols, cat_cols):
    return html.Div([
        _two_group_grid("ab", num_cols, cat_cols),
        html.Div([
            _field("МЕТРИКИ", html.Div([
                html.Span("conversion", className="ms-chip"),
                html.Span("revenue · mean", className="ms-chip"),
                html.Span("arpu", className="ms-chip muted"),
                html.Span("retention_7d", className="ms-chip muted"),
                html.Span("+ добавить",
                          style={"color": "var(--text-tertiary)",
                                 "fontSize": "12px"}),
            ], className="chips-select")),
            _field("ГИПОТЕЗА",
                   dcc.RadioItems(
                       id="ab-alt",
                       options=[
                           {"label": "two-sided", "value": "two-sided"},
                           {"label": "> control", "value": "greater"},
                       ],
                       value="two-sided",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            html.Button("🧪 Анализ A/B", id="btn-ab",
                        className="eda-btn primary",
                        style={"marginLeft": "auto",
                               "alignSelf": "flex-end",
                               "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "16px",
                  "alignItems": "flex-end", "marginTop": "12px"}),
        dcc.Loading(html.Div(id="ab-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_bh_tab():
    return html.Div([
        _alert(
            "Множественные сравнения · BH (Benjamini-Hochberg)",
            "Контролирует ожидаемую долю ложноположительных открытий (FDR). "
            "Менее консервативен, чем Бонферрони. Используйте, когда "
            "запущено много независимых тестов.",
            kind="info",
        ),
        html.Div([
            html.Div("ФИЛЬТР:", className="caption"),
            dcc.RadioItems(
                id="bh-filter",
                options=[
                    {"label": "Все", "value": "all"},
                    {"label": "Значимые", "value": "sig"},
                    {"label": "Не значимые", "value": "ns"},
                ],
                value="all",
                className="radio-group",
                inputStyle={"display": "none"},
                labelClassName="opt",
            ),
            html.Div([
                _chip("FDR · 0.05", "neutral"),
                _chip("МЕТОД · BH", "accent"),
                html.Button("⚡ Применить BH", id="btn-bh",
                            className="eda-btn primary",
                            style={"padding": "6px 12px"}),
            ], style={"marginLeft": "auto", "display": "flex",
                      "gap": "8px", "alignItems": "center"}),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "12px", "padding": "12px 16px",
                  "background": "var(--surface-1)",
                  "border": "1px solid var(--border-subtle)",
                  "borderRadius": "10px", "marginTop": "12px"}),
        dcc.Loading(html.Div(id="bh-result",
                             style={"marginTop": "12px"}),
                    type="circle", color="#5fd8c0"),
    ], className="col-16")


def _render_power_tab(df, num_cols, cat_cols):
    return html.Div([
        # Top control row
        html.Div([
            _field("ТИП ТЕСТА",
                   _select("pw-test-type",
                           [
                               {"label": "t-test · 2 sample", "value": "ttest2"},
                               {"label": "t-test · 1 sample", "value": "ttest1"},
                               {"label": "Пропорции (z-тест)", "value": "prop"},
                               {"label": "ANOVA (k групп)", "value": "anova"},
                           ],
                           value="ttest2", clearable=False)),
            _field("РЕШИТЬ ДЛЯ",
                   dcc.RadioItems(
                       id="pw-solve-for",
                       options=[
                           {"label": "n · размер", "value": "nobs"},
                           {"label": "power", "value": "power"},
                           {"label": "effect", "value": "effect_size"},
                       ],
                       value="nobs",
                       className="radio-group",
                       inputStyle={"display": "none"},
                       labelClassName="opt",
                   )),
            _field("ALPHA", _num_input("pw-alpha", 0.05,
                                        min_val=0.001, max_val=0.5, step=0.001)),
            _field("ALTERNATIVE",
                   _select("pw-alt",
                           [{"label": "two-sided", "value": "two-sided"},
                            {"label": "greater", "value": "greater"},
                            {"label": "less", "value": "less"}],
                           value="two-sided", clearable=False)),
            html.Button("○ Рассчитать", id="btn-power",
                        className="eda-btn primary",
                        style={"marginLeft": "auto",
                               "alignSelf": "flex-end",
                               "padding": "8px 14px"}),
        ], style={"display": "flex", "gap": "16px",
                  "alignItems": "flex-end"}),

        html.Div([
            # left col: parameters card + interpretation alert
            html.Div([
                html.Div([
                    _overline("ПАРАМЕТРЫ"),
                    html.Div([
                        _field("N · РАЗМЕР ВЫБОРКИ (на группу)",
                               _num_input("pw-n", 50, min_val=2, max_val=10**6,
                                          step=1)),
                        _field("POWER · (1 − β)",
                               _num_input("pw-power", 0.80,
                                          min_val=0.5, max_val=0.999, step=0.01)),
                        _field("EFFECT SIZE · COHEN'S D",
                               _num_input("pw-effect", 0.35,
                                          min_val=0.01, max_val=5.0, step=0.01)),
                        _field("RATIO N₂/N₁",
                               _num_input("pw-ratio", 1.0,
                                          min_val=0.1, max_val=10.0, step=0.1)),
                        _field("K групп (ANOVA)",
                               _num_input("pw-k-groups", 3, min_val=2,
                                          max_val=20, step=1)),
                    ], style={"display": "flex",
                              "flexDirection": "column",
                              "gap": "12px",
                              "marginTop": "10px"}),
                ], className="card", style={"padding": "16px"}),
                _alert(
                    "Интерпретация",
                    "Задайте 2 из 3 параметров (effect / power / n) — "
                    "KIBAD рассчитает третий. По умолчанию n=259 наблюдений на "
                    "группу для d=0.35 при power=0.80.",
                    kind="info",
                ),
                # Auto-effect-size from data (compact)
                html.Div([
                    _overline("Авторасчёт effect size из данных"),
                    html.Div([
                        _field("ЧИСЛОВАЯ КОЛОНКА",
                               _select("pw-col-num",
                                       [{"label": c, "value": c} for c in num_cols])),
                        _field("ГРУППИРУЮЩАЯ",
                               _select("pw-col-grp",
                                       [{"label": c, "value": c} for c in cat_cols])),
                        html.Button("↗ Вычислить", id="btn-calc-effect",
                                    className="eda-btn ghost",
                                    style={"alignSelf": "flex-end",
                                           "padding": "6px 12px"}),
                    ], style={"display": "flex", "gap": "12px",
                              "alignItems": "flex-end",
                              "marginTop": "8px"}),
                    html.Div(id="pw-effect-hint",
                             style={"marginTop": "8px"}),
                ], className="card", style={"padding": "14px"}),
            ], className="col-16"),

            # right col: result card + power curve + 4 KPIs
            html.Div(id="power-result", className="col-16",
                     children=_power_initial_placeholder()),
        ], className="grid-2",
            style={"gridTemplateColumns": "1fr 1.4fr",
                   "gap": "16px",
                   "marginTop": "16px"}),
    ], className="col-16")


def _power_initial_placeholder():
    return [html.Div([
        html.Div("Нажмите «Рассчитать», чтобы получить искомый параметр.",
                 className="caption"),
    ], className="card", style={"padding": "20px"})]


# ===========================================================================
# Group-selectors auto-fill
# ===========================================================================

def _make_group_callback(prefix):
    @callback(
        Output(f"{prefix}-ga", "options"),
        Output(f"{prefix}-gb", "options"),
        Output(f"{prefix}-ga", "value"),
        Output(f"{prefix}-gb", "value"),
        Input(f"{prefix}-grp", "value"),
        State("tests-ds-select", "value"),
        State(STORE_DATASET, "data"),
        State(STORE_PREPARED, "data"),
        prevent_initial_call=True,
    )
    def _update_groups(grp_col, ds_name, raw, prep):
        df = _get_df(ds_name, raw, prep)
        if df is None or not grp_col or grp_col not in df.columns:
            return [], [], None, None
        vals = df[grp_col].dropna().unique().tolist()
        opts = [{"label": str(v), "value": str(v)} for v in vals]
        v_a = str(vals[0]) if vals else None
        v_b = str(vals[1]) if len(vals) > 1 else v_a
        return opts, opts, v_a, v_b


for _p in ["tt", "mw", "bt", "pm", "ab"]:
    _make_group_callback(_p)


@callback(
    Output("diag-group-selects", "children"),
    Input("diag-grp-col", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _diag_group_options(grp_col, ds_name, raw, prep):
    df = _get_df(ds_name, raw, prep)
    if df is None or not grp_col:
        return []
    groups = sorted(df[grp_col].dropna().unique().astype(str).tolist())
    if len(groups) < 2:
        return [alert_banner("Нужно минимум 2 уникальных значения.", "warning")]
    opts = [{"label": g, "value": g} for g in groups]
    return [
        _field("ГРУППА A",
               _select("diag-ga", opts, value=groups[0], clearable=False)),
        _field("ГРУППА B",
               _select("diag-gb", opts,
                       value=groups[1] if len(groups) > 1 else groups[0],
                       clearable=False)),
    ]


# ===========================================================================
# t-test callback
# ===========================================================================

@callback(
    Output("ttest-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-ttest", "n_clicks"),
    State("tt-val", "value"), State("tt-grp", "value"),
    State("tt-ga", "value"), State("tt-gb", "value"),
    State("tt-eqvar", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_ttest(n, val_col, grp_col, ga, gb, eqvar, alpha_tick,
               ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        a = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        b = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        if len(a) < 2 or len(b) < 2:
            return alert_banner("Недостаточно данных в группах.", "warning"), no_update
        eq = "eq" in (eqvar or [])
        result = ttest_independent(a, b, alpha=alpha, equal_var=eq,
                                   label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": "t-Тест",
                                     "var": f"{val_col} × {grp_col}",
                                     "p": rd["p_value"]}]

        d_val = rd["effect_size"] or 0.0
        ci = rd["ci"] or (np.nan, np.nan)
        diff = float(np.mean(a) - np.mean(b))

        verdict = _verdict_card(
            rd, alpha,
            headline=("Средние значимо различаются"
                      if rd["significant"] else
                      "Средние значимо не различаются"),
            caption=(f"Δ среднего = {diff:.4g}. "
                     f"{interpret_effect_size(d_val, 'cohen_d')}"),
            metrics=[
                ("T-СТАТИСТИКА", f"{rd['statistic']:.4g}",
                 f"n = {len(a)} / {len(b)}"),
                ("P-VALUE", f"{rd['p_value']:.4g}",
                 f"{'<' if rd['significant'] else '≥'} α = {alpha}"),
                ("COHEN'S D", f"{d_val:.3f}", rd["effect_label"] or ""),
                ("95% CI",
                 f"[{ci[0]:.3g}, {ci[1]:.3g}]" if ci[0] == ci[0] else "—",
                 ""),
            ],
            right_meta=("Welch's t-test"
                        if not eq else "Student's t-test"),
        )

        plots = html.Div([
            html.Div(dcc.Graph(figure=_overlay_hist(
                a.values, b.values, str(ga), str(gb), val_col)),
                    className="card", style={"padding": "12px"}),
            html.Div(dcc.Graph(figure=_ecdf_fig(
                a.values, b.values, str(ga), str(gb), val_col)),
                    className="card", style={"padding": "12px"}),
        ], className="grid-2", style={"gap": "12px", "marginTop": "12px"})

        return html.Div([verdict, plots]), pv_list
    except Exception as e:
        logger.exception("ttest")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# Mann-Whitney callback
# ===========================================================================

@callback(
    Output("mw-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-mw", "n_clicks"),
    State("mw-val", "value"), State("mw-grp", "value"),
    State("mw-ga", "value"), State("mw-gb", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_mw(n, val_col, grp_col, ga, gb, alpha_tick, ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        a = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        b = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        if len(a) < 2 or len(b) < 2:
            return alert_banner("Недостаточно данных в группах.", "warning"), no_update
        result = mann_whitney(a, b, alpha=alpha,
                              label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": "Манна-Уитни",
                                     "var": f"{val_col} × {grp_col}",
                                     "p": rd["p_value"]}]
        delta, delta_label = cliffs_delta(a, b)
        prob_a_gt_b = (delta + 1) / 2.0

        # Verdict (left side) and gauge (right side) — custom 2-col layout
        is_sig = rd["significant"]
        chip_kind = "success" if is_sig else "danger"
        chip_text = "✓ ЗНАЧИМО" if is_sig else "✕ НЕ ЗНАЧИМО"

        left_block = html.Div([
            html.Div([
                _chip(chip_text, chip_kind,
                      style={"fontSize": "13px", "padding": "6px 14px"}),
                html.Span("ранги групп смещены" if is_sig else "ранги почти равны",
                          className="mono",
                          style={"fontSize": "12px",
                                 "color": "var(--text-secondary)",
                                 "marginLeft": "12px"}),
            ], style={"display": "flex", "alignItems": "center"}),
            html.H2("Распределения различны по положению"
                    if is_sig else "Положение распределений сходно",
                    style={"marginTop": "12px"}),
            html.P(f"Вероятность P(A > B) = {prob_a_gt_b:.3f}.",
                   className="caption", style={"marginTop": "6px"}),
            html.Div([
                _metric("U-СТАТИСТИКА", f"{rd['statistic']:.4g}", "rank-sum"),
                _metric("Z-SCORE",
                        f"{rd['statistic']:.3g}", ""),
                _metric("P-VALUE", f"{rd['p_value']:.4g}",
                        f"{'<' if is_sig else '≥'} α = {alpha}"),
                _metric("P(A>B)", f"{prob_a_gt_b:.3f}", "доминирование"),
            ], className="metrics",
               style={"display": "grid",
                      "gridTemplateColumns": "repeat(4, minmax(0, 1fr))",
                      "gap": "16px", "marginTop": "16px",
                      "paddingTop": "16px",
                      "borderTop": "1px solid var(--border-subtle)"}),
        ])

        right_block = html.Div([
            _overline("CLIFF'S δ"),
            dcc.Graph(figure=_cliff_gauge(delta),
                      config={"displayModeBar": False}),
            html.Div(f"δ = {delta:+.3f} · {delta_label}",
                     className="mono",
                     style={"textAlign": "center",
                            "fontSize": "11px",
                            "color": "var(--text-tertiary)"}),
        ], style={"borderLeft": "1px solid var(--border-subtle)",
                  "paddingLeft": "20px"})

        verdict = html.Div([
            html.Div([left_block, right_block],
                     style={"display": "grid",
                            "gridTemplateColumns": "1fr 280px",
                            "gap": "20px", "alignItems": "center"}),
        ], className="card accent", style={"padding": "20px"})

        plots = html.Div([
            html.Div(dcc.Graph(figure=_overlay_hist(
                a.values, b.values, str(ga), str(gb), val_col)),
                    className="card", style={"padding": "12px"}),
            html.Div(dcc.Graph(figure=_ecdf_fig(
                a.values, b.values, str(ga), str(gb), val_col)),
                    className="card", style={"padding": "12px"}),
        ], className="grid-2", style={"gap": "12px", "marginTop": "12px"})

        return html.Div([verdict, plots]), pv_list
    except Exception as e:
        logger.exception("mw")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# Chi-square callback
# ===========================================================================

@callback(
    Output("chi2-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-chi2", "n_clicks"),
    State("chi2-col-a", "value"), State("chi2-col-b", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_chi2(n, col_a, col_b, alpha_tick, ds_name, raw, prep, all_pv):
    if not col_a or not col_b:
        return alert_banner("Выберите две категориальные колонки.",
                            "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        result = chi_square_independence(df, col_a, col_b, alpha=alpha)
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": "Хи-квадрат",
                                     "var": f"{col_a} × {col_b}",
                                     "p": rd["p_value"]}]

        ct = pd.crosstab(df[col_a], df[col_b])
        # Expected (independence): row_total * col_total / n
        row_tot = ct.sum(axis=1).values.reshape(-1, 1)
        col_tot = ct.sum(axis=0).values.reshape(1, -1)
        n_total = int(ct.values.sum())
        expected = row_tot * col_tot / max(n_total, 1)
        residuals = (ct.values - expected) / np.sqrt(np.maximum(expected, 1e-9))

        verdict = _verdict_card(
            rd, alpha,
            headline=("Переменные не независимы"
                      if rd["significant"] else
                      "Связь между переменными не найдена"),
            caption=(f"Cramer's V = {rd['effect_size']:.3f}"
                     if rd["effect_size"] is not None else
                     "Cramer's V — связь между категориями"),
            metrics=[
                ("χ²", f"{rd['statistic']:.4g}", ""),
                ("P-VALUE", f"{rd['p_value']:.4g}",
                 f"{'<' if rd['significant'] else '≥'} α = {alpha}"),
                ("CRAMER'S V",
                 f"{rd['effect_size']:.3f}" if rd["effect_size"] else "—",
                 rd["effect_label"] or ""),
                ("N", f"{n_total:,}", "наблюдений"),
            ],
            right_meta="chi-square independence",
        )

        # Heatmap (Plotly)
        fig = go.Figure(data=go.Heatmap(
            z=ct.values,
            x=[str(c) for c in ct.columns],
            y=[str(r) for r in ct.index],
            text=ct.values, texttemplate="%{text}",
            colorscale="Blues",
        ))
        fig.update_layout(title="Таблица сопряжённости",
                          height=380)
        apply_eda_theme(fig)

        # Residuals table — top 20 by |z|
        rows = []
        flat = []
        for i, r_lbl in enumerate(ct.index):
            for j, c_lbl in enumerate(ct.columns):
                flat.append((str(r_lbl), str(c_lbl),
                             int(ct.values[i, j]),
                             float(expected[i, j]),
                             float(residuals[i, j])))
        flat.sort(key=lambda x: -abs(x[4]))
        for r_lbl, c_lbl, o, e, z in flat[:20]:
            sig = abs(z) > 1.96
            rows.append(html.Tr([
                html.Td(r_lbl), html.Td(c_lbl),
                html.Td(f"{o}", className="mono"),
                html.Td(f"{e:.1f}", className="mono"),
                html.Td(f"{z:+.2f}", className="mono"),
            ], className="significant" if sig else ""))

        residuals_card = html.Div([
            html.H3("Остатки Пирсона (standardized)",
                    style={"fontSize": "14px"}),
            html.Div("|z| > 1.96 — значимое отклонение от независимости",
                     className="caption",
                     style={"fontSize": "11px", "marginBottom": "10px"}),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Сегмент"), html.Th("Категория"),
                    html.Th("O", className="mono"),
                    html.Th("E", className="mono"),
                    html.Th("z", className="mono"),
                ])),
                html.Tbody(rows),
            ], className="tbl"),
        ], className="card", style={"padding": "16px"})

        plots = html.Div([
            html.Div(dcc.Graph(figure=fig), className="card",
                     style={"padding": "12px"}),
            residuals_card,
        ], className="grid-2",
            style={"gridTemplateColumns": "1.4fr 1fr",
                   "gap": "16px", "marginTop": "12px"})

        return html.Div([verdict, plots]), pv_list
    except Exception as e:
        logger.exception("chi2")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# Correlation callback
# ===========================================================================

@callback(
    Output("corr-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-corr", "n_clicks"),
    State("corr-x", "value"), State("corr-y", "value"),
    State("corr-method", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_corr(n, x_col, y_col, method, alpha_tick, ds_name, raw, prep, all_pv):
    if not x_col or not y_col:
        return alert_banner("Выберите обе переменные.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        sub = df[[x_col, y_col]].dropna()
        if len(sub) < 3:
            return alert_banner("Недостаточно данных.", "warning"), no_update
        result = correlation_test(sub[x_col], sub[y_col],
                                  method=method or "pearson", alpha=alpha,
                                  label_x=x_col, label_y=y_col)
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": "Корреляция",
                                     "var": f"{x_col} × {y_col}",
                                     "p": rd["p_value"]}]
        r = float(rd["statistic"])
        r2 = r * r
        ci = rd.get("ci") or (np.nan, np.nan)
        # Fisher z
        try:
            fz = 0.5 * np.log((1 + r) / (1 - r))
        except Exception:
            fz = float("nan")

        # Verdict card (left), scatter (right)
        is_sig = rd["significant"]
        if abs(r) >= 0.7:
            head = "Сильная " + ("положительная" if r > 0 else "отрицательная") + " связь"
        elif abs(r) >= 0.4:
            head = "Умеренная " + ("положительная" if r > 0 else "отрицательная") + " связь"
        else:
            head = "Слабая связь"

        left_card = html.Div([
            _chip("✓ ЗНАЧИМА" if is_sig else "✕ НЕ ЗНАЧИМА",
                  "success" if is_sig else "danger"),
            html.H2(head, style={"marginTop": "10px"}),
            html.P(interpret_correlation(r, p=rd["p_value"]),
                   className="caption", style={"marginTop": "6px"}),
            html.Div([
                _metric(f"R ({(method or 'pearson').upper()})",
                        f"{r:+.3f}", rd["effect_label"] or ""),
                _metric("P-VALUE", f"{rd['p_value']:.4g}",
                        f"{'<' if is_sig else '≥'} α = {alpha}"),
                _metric("R²", f"{r2:.3f}",
                        f"{r2*100:.1f}% дисперсии"),
                _metric("N", f"{len(sub):,}", "пар"),
                _metric("95% CI",
                        (f"[{ci[0]:+.3g}, {ci[1]:+.3g}]"
                         if ci and ci[0] == ci[0] else "—"), ""),
                _metric("FISHER z", f"{fz:.3f}", ""),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(2, 1fr)",
                      "gap": "16px", "marginTop": "18px",
                      "paddingTop": "18px",
                      "borderTop": "1px solid var(--border-subtle)"}),
        ], className="card accent", style={"padding": "20px"})

        # Scatter + OLS
        try:
            fig = px.scatter(sub, x=x_col, y=y_col, trendline="ols",
                             title=f"Scatter + OLS · {x_col} × {y_col}")
        except Exception:
            fig = px.scatter(sub, x=x_col, y=y_col,
                             title=f"Scatter · {x_col} × {y_col}")
        fig.update_layout(height=360)
        apply_eda_theme(fig)

        right_card = html.Div(dcc.Graph(figure=fig),
                              className="card",
                              style={"padding": "12px"})

        return html.Div([
            html.Div([left_card, right_card],
                     className="grid-2",
                     style={"gridTemplateColumns": "1fr 1.2fr",
                            "gap": "16px"}),
        ]), pv_list
    except Exception as e:
        logger.exception("corr")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# Bootstrap callback
# ===========================================================================

@callback(
    Output("boot-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-boot", "n_clicks"),
    State("bt-val", "value"), State("bt-grp", "value"),
    State("bt-ga", "value"), State("bt-gb", "value"),
    State("bt-n", "value"), State("bt-stat", "value"),
    State("bt-seed", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_boot(n, val_col, grp_col, ga, gb, n_boot, stat, seed,
              alpha_tick, ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        a = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        b = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        result = bootstrap_test(a, b, statistic=stat or "mean",
                                n_bootstrap=int(n_boot or 5000), alpha=alpha,
                                label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": "Бутстрап",
                                     "var": f"{val_col} × {grp_col}",
                                     "p": rd["p_value"]}]

        # Reproduce bootstrap samples for chart (use seed)
        rng = np.random.default_rng(int(seed or 42))
        a_arr = a.to_numpy(dtype=float)
        b_arr = b.to_numpy(dtype=float)
        n_iter = int(n_boot or 5000)
        if stat == "median":
            stats_arr = np.array([
                np.median(rng.choice(a_arr, len(a_arr), replace=True))
                - np.median(rng.choice(b_arr, len(b_arr), replace=True))
                for _ in range(min(n_iter, 5000))
            ])
        else:
            stats_arr = np.array([
                np.mean(rng.choice(a_arr, len(a_arr), replace=True))
                - np.mean(rng.choice(b_arr, len(b_arr), replace=True))
                for _ in range(min(n_iter, 5000))
            ])
        ci = rd.get("ci") or (np.percentile(stats_arr, 2.5),
                              np.percentile(stats_arr, 97.5))
        observed = (np.mean(a_arr) - np.mean(b_arr)
                    if stat != "median" else
                    np.median(a_arr) - np.median(b_arr))
        se = float(np.std(stats_arr, ddof=1))
        bias = float(np.mean(stats_arr) - observed)

        is_sig = rd["significant"]
        left = html.Div([
            _chip("✓ ЗНАЧИМО" if is_sig else "✕ НЕ ЗНАЧИМО",
                  "success" if is_sig else "danger"),
            html.H2(f"Δ {stat or 'mean'} {'положительна' if observed > 0 else 'отрицательна'}",
                    style={"marginTop": "10px"}),
            html.P(("95% доверительный интервал не накрывает ноль."
                    if is_sig else
                    "95% доверительный интервал содержит ноль."),
                   className="caption", style={"marginTop": "6px"}),
            html.Div([
                _metric(f"Δ {(stat or 'mean').upper()}",
                        f"{observed:.4g}", "точечная оценка"),
                _metric("P-VALUE", f"{rd['p_value']:.4g}",
                        f"{'<' if is_sig else '≥'} α = {alpha}"),
                _metric("95% CI LOW", f"{ci[0]:.4g}", ""),
                _metric("95% CI HIGH", f"{ci[1]:.4g}", ""),
                _metric("SE (BOOTSTRAP)", f"{se:.4g}", ""),
                _metric("BIAS", f"{bias:+.4g}",
                        "negligible" if abs(bias) < 0.01 * max(abs(observed), 1)
                        else "bias detected"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(2, 1fr)",
                      "gap": "16px", "marginTop": "16px",
                      "paddingTop": "16px",
                      "borderTop": "1px solid var(--border-subtle)"}),
        ], className="card accent", style={"padding": "20px"})

        right = html.Div(
            dcc.Graph(figure=_bootstrap_dist_fig(stats_arr, ci[0], ci[1],
                                                 observed=observed)),
            className="card", style={"padding": "12px"})

        return html.Div([
            html.Div([left, right], className="grid-2",
                     style={"gridTemplateColumns": "1fr 1.3fr",
                            "gap": "16px"}),
        ]), pv_list
    except Exception as e:
        logger.exception("boot")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# Permutation callback
# ===========================================================================

@callback(
    Output("perm-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-perm", "n_clicks"),
    State("pm-val", "value"), State("pm-grp", "value"),
    State("pm-ga", "value"), State("pm-gb", "value"),
    State("pm-n", "value"), State("pm-stat", "value"),
    State("pm-alt", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_perm(n, val_col, grp_col, ga, gb, n_perm, stat_kind, alt,
              alpha_tick, ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        a = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        b = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        n_iter = int(n_perm or 10000)
        result = permutation_test(a, b, n_perm=n_iter, alpha=alpha,
                                  label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": "Перестановочный",
                                     "var": f"{val_col} × {grp_col}",
                                     "p": rd["p_value"]}]

        # Build a null distribution for visualization (smaller, fast).
        a_arr = a.to_numpy(dtype=float)
        b_arr = b.to_numpy(dtype=float)
        observed = float(np.mean(a_arr) - np.mean(b_arr))
        rng = np.random.default_rng(42)
        pool = np.concatenate([a_arr, b_arr])
        n_a = len(a_arr)
        n_viz = min(n_iter, 5000)
        null_diffs = np.empty(n_viz)
        for i in range(n_viz):
            rng.shuffle(pool)
            null_diffs[i] = np.mean(pool[:n_a]) - np.mean(pool[n_a:])

        is_sig = rd["significant"]
        pct = float(np.mean(null_diffs <= observed) * 100)
        n_extreme = int(np.sum(np.abs(null_diffs) >= abs(observed)))

        left = html.Div([
            html.Div([
                _chip("✓ ЗНАЧИМО" if is_sig else "✕ НЕ ЗНАЧИМО",
                      "success" if is_sig else "danger"),
                _chip("EXACT P-VALUE", "neutral",
                      style={"marginLeft": "8px"}),
            ]),
            html.H2("Наблюдаемая статистика в хвосте"
                    if is_sig else "Наблюдаемая статистика типична",
                    style={"marginTop": "12px"}),
            html.P((f"Только {n_extreme} из {n_viz} перестановок "
                    "дают значение, равное или экстремальнее наблюдаемого."),
                   className="caption", style={"marginTop": "6px"}),
            html.Div([
                _metric("T НАБЛЮД.", f"{observed:.4g}",
                        f"Δ {(stat_kind or 'mean')}"),
                _metric("P-VALUE", f"{rd['p_value']:.4g}",
                        f"{n_extreme} / {n_viz}"),
                _metric("MEAN(NULL)", f"{float(np.mean(null_diffs)):.3g}", "≈ 0"),
                _metric("SD(NULL)", f"{float(np.std(null_diffs)):.3g}", ""),
                _metric("PERCENTILE", f"{pct:.2f}", "right tail"),
                _metric("ДВУХСТОР. p", f"{rd['p_value']:.4g}", ""),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(2, 1fr)",
                      "gap": "16px", "marginTop": "18px",
                      "paddingTop": "18px",
                      "borderTop": "1px solid var(--border-subtle)"}),
        ], className="card accent", style={"padding": "20px"})

        right = html.Div(
            dcc.Graph(figure=_null_dist_fig(null_diffs, observed)),
            className="card", style={"padding": "12px"})

        return html.Div([
            html.Div([left, right], className="grid-2",
                     style={"gridTemplateColumns": "1fr 1.3fr",
                            "gap": "16px"}),
        ]), pv_list
    except Exception as e:
        logger.exception("perm")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# ANOVA / Kruskal callback
# ===========================================================================

@callback(
    Output("anova-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-anova", "n_clicks"),
    State("anova-val", "value"), State("anova-grp", "value"),
    State("anova-method", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_anova(n, val_col, grp_col, method, alpha_tick, ds_name, raw, prep, all_pv):
    if not val_col or not grp_col:
        return alert_banner("Выберите числовую и группирующую колонки.",
                            "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        grouped = {k: v.dropna() for k, v in df.groupby(grp_col)[val_col]}
        labels = sorted(grouped.keys(), key=str)
        arrays = [grouped[k] for k in labels]
        str_labels = [str(lb) for lb in labels]
        if len(arrays) < 2:
            return alert_banner("Нужно минимум 2 группы.", "warning"), no_update

        if method == "kruskal":
            result = kruskal_wallis(*arrays, alpha=alpha, labels=str_labels)
            test_name = "Kruskal-Wallis"
        else:
            result = anova_oneway(*arrays, alpha=alpha, labels=str_labels)
            test_name = "One-way ANOVA"

        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [{"test": test_name,
                                     "var": f"{val_col} × {grp_col}",
                                     "p": rd["p_value"]}]

        # Verdict (header + 4 metrics on right)
        is_sig = rd["significant"]
        n_total = sum(len(a) for a in arrays)
        df_between = len(arrays) - 1
        df_within = n_total - len(arrays)

        # ω² adjusted (approx for ANOVA)
        ss_between = sum(len(a) * (np.mean(a) - np.mean(np.concatenate(arrays))) ** 2
                         for a in arrays)
        ss_within = sum(((a - np.mean(a)) ** 2).sum() for a in arrays)
        ms_within = ss_within / max(df_within, 1)
        omega_sq = ((ss_between - df_between * ms_within)
                    / (ss_between + ss_within + ms_within))
        omega_sq = max(0.0, float(omega_sq))

        chip_kind = "success" if is_sig else "danger"
        chip_text = "✓ ЗНАЧИМО" if is_sig else "✕ НЕ ЗНАЧИМО"

        es_label = "η²" if method != "kruskal" else "ε²"

        verdict = html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        _chip(chip_text, chip_kind,
                              style={"fontSize": "13px",
                                     "padding": "6px 14px"}),
                        _chip("МЕЖДУ ГРУППАМИ", "neutral",
                              style={"marginLeft": "8px"}),
                    ]),
                    html.H2("Групповые средние различаются"
                            if is_sig else "Групповые средние сходны",
                            style={"marginTop": "12px"}),
                    html.P(f"{test_name} · {len(arrays)} групп · "
                           f"n = {n_total:,}",
                           className="caption", style={"marginTop": "6px"}),
                ]),
                html.Div([
                    _metric("F-STAT" if method != "kruskal" else "H-STAT",
                            f"{rd['statistic']:.3f}",
                            (f"df = {df_between}, {df_within}"
                             if method != "kruskal" else
                             f"df = {df_between}")),
                    _metric("P-VALUE", f"{rd['p_value']:.4g}",
                            f"{'<' if is_sig else '≥'} α = {alpha}"),
                    _metric(es_label,
                            (f"{rd['effect_size']:.3f}"
                             if rd["effect_size"] is not None else "—"),
                            rd["effect_label"] or ""),
                    _metric("ω²", f"{omega_sq:.3f}", "adjusted"),
                ], style={"display": "grid",
                          "gridTemplateColumns": "repeat(4, 1fr)",
                          "gap": "20px",
                          "paddingLeft": "20px",
                          "borderLeft": "1px solid var(--border-subtle)"}),
            ], style={"display": "flex",
                      "justifyContent": "space-between",
                      "alignItems": "flex-start",
                      "gap": "16px"}),
        ], className="card accent", style={"padding": "20px"})

        # Post-hoc table from result.details
        posthoc = result.details.get("posthoc", [])
        n_sig_ph = sum(1 for ph in posthoc if ph.get("significant"))

        if posthoc:
            rows = []
            for ph in posthoc:
                pair = ph.get("pair", "")
                p_disp = ph.get("p_adj", ph.get("p_value", float("nan")))
                rows.append(html.Tr([
                    html.Td(pair),
                    html.Td(f"{ph.get('diff', 0):+.4g}", className="mono"),
                    html.Td(
                        (f"{ph.get('ci_low', float('nan')):+.4g}"
                         if ph.get("ci_low") is not None else "—"),
                        className="mono"),
                    html.Td(
                        (f"{ph.get('ci_high', float('nan')):+.4g}"
                         if ph.get("ci_high") is not None else "—"),
                        className="mono"),
                    html.Td(f"{p_disp:.4g}", className="mono"),
                    html.Td(_chip("✓", "success") if ph.get("significant")
                            else _chip("—", "neutral")),
                ], className="significant" if ph.get("significant") else ""))

            posthoc_card = html.Div([
                html.Div([
                    html.Div([
                        html.H3("Post-hoc · попарные сравнения",
                                style={"fontSize": "14px"}),
                        html.Div("значимые пары подсвечены · family-wise α = "
                                 + f"{alpha}",
                                 className="caption",
                                 style={"fontSize": "11px",
                                        "marginTop": "2px"}),
                    ]),
                    html.Div([
                        _chip(f"{n_sig_ph} значимы", "success"),
                        _chip(f"{len(posthoc) - n_sig_ph} не знач.", "neutral"),
                    ], style={"marginLeft": "auto", "display": "flex",
                              "gap": "8px"}),
                ], style={"display": "flex", "alignItems": "center",
                          "padding": "14px 20px",
                          "borderBottom": "1px solid var(--border-subtle)"}),
                html.Div([
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Пара"),
                            html.Th("Diff", className="mono"),
                            html.Th("CI low", className="mono"),
                            html.Th("CI high", className="mono"),
                            html.Th("p-adj", className="mono"),
                            html.Th("Значим?"),
                        ])),
                        html.Tbody(rows),
                    ], className="tbl"),
                ], style={"maxHeight": "320px", "overflow": "auto"}),
            ], className="card", style={"padding": 0, "overflow": "hidden",
                                         "marginTop": "12px"})
        else:
            posthoc_card = html.Div()

        # Box-plot for groups
        plot_df = df[[val_col, grp_col]].dropna()
        fig_box = px.box(plot_df, x=grp_col, y=val_col, color=grp_col,
                         title=f"Распределение «{val_col}» по «{grp_col}»")
        fig_box.update_layout(height=320, showlegend=False)
        apply_eda_theme(fig_box)
        box_card = html.Div(dcc.Graph(figure=fig_box),
                            className="card",
                            style={"padding": "12px",
                                   "marginTop": "12px"})

        return html.Div([verdict, posthoc_card, box_card]), pv_list
    except Exception as e:
        logger.exception("anova")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# A/B test callback
# ===========================================================================

@callback(
    Output("ab-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-ab", "n_clicks"),
    State("ab-val", "value"), State("ab-grp", "value"),
    State("ab-ga", "value"), State("ab-gb", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_ab(n, val_col, grp_col, ga, gb, alpha_tick,
            ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        ctrl = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        trt = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        if len(ctrl) < 2 or len(trt) < 2:
            return alert_banner("Недостаточно данных.", "warning"), no_update

        ab_res = ab_test(ctrl, trt, alpha=alpha,
                         label_ctrl=str(ga), label_trt=str(gb))

        # Pull primary p-value (ttest)
        primary_p = float("nan")
        primary_sig = False
        if "ttest" in ab_res and hasattr(ab_res["ttest"], "p_value"):
            primary_p = ab_res["ttest"].p_value
            primary_sig = ab_res["ttest"].significant

        pv_list = (all_pv or []) + [{"test": "A/B test",
                                     "var": f"{val_col} × {grp_col}",
                                     "p": primary_p}]

        # Group statistics
        n_a, n_b = len(ctrl), len(trt)
        m_a, m_b = float(np.mean(ctrl)), float(np.mean(trt))
        sd_a = float(np.std(ctrl, ddof=1)) if n_a > 1 else 0.0
        sd_b = float(np.std(trt, ddof=1)) if n_b > 1 else 0.0
        se_a = sd_a / np.sqrt(n_a) if n_a > 0 else 0.0
        se_b = sd_b / np.sqrt(n_b) if n_b > 0 else 0.0
        ci_a = (m_a - 1.96 * se_a, m_a + 1.96 * se_a)
        ci_b = (m_b - 1.96 * se_b, m_b + 1.96 * se_b)
        lift = (m_b - m_a) / m_a * 100 if m_a != 0 else float("nan")
        delta_pp = (m_b - m_a) * 100  # if metric is proportion

        # Two cards
        card_ctrl = html.Div([
            html.Div([
                html.Div([
                    _chip(f"CONTROL · {ga}", "neutral"),
                    html.H3("Базовая когорта",
                            style={"marginTop": "8px",
                                   "fontSize": "15px"}),
                ]),
            ], style={"display": "flex",
                      "justifyContent": "space-between",
                      "alignItems": "center"}),
            html.Div([
                _metric("n", f"{n_a:,}", "obs"),
                _metric("μ", f"{m_a:.4g}", ""),
                _metric("σ", f"{sd_a:.4g}", ""),
                _metric("SE", f"±{se_a:.4g}", ""),
                _metric("CI 95%",
                        f"[{ci_a[0]:.3g}, {ci_a[1]:.3g}]", ""),
                _metric("Δ vs ctrl", "—", ""),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(3, 1fr)",
                      "gap": "16px", "marginTop": "16px"}),
        ], className="card", style={"padding": "20px"})

        card_trt = html.Div([
            html.Div([
                html.Div([
                    _chip(f"TREATMENT · {gb}", "accent"),
                    html.H3("Новая вариация",
                            style={"marginTop": "8px",
                                   "fontSize": "15px"}),
                ]),
            ], style={"display": "flex",
                      "justifyContent": "space-between",
                      "alignItems": "center"}),
            html.Div([
                _metric("n", f"{n_b:,}", "obs"),
                _metric("μ", f"{m_b:.4g}",
                        f"{'+' if lift>=0 else ''}{lift:.2f}%"),
                _metric("σ", f"{sd_b:.4g}", ""),
                _metric("SE", f"±{se_b:.4g}", ""),
                _metric("CI 95%",
                        f"[{ci_b[0]:.3g}, {ci_b[1]:.3g}]", ""),
                _metric("Δ vs ctrl",
                        f"{m_b - m_a:+.4g}", "разница"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(3, 1fr)",
                      "gap": "16px", "marginTop": "16px"}),
        ], className="card accent", style={"padding": "20px"})

        # Sticky bar (result)
        conf_pct = (1 - primary_p) * 100 if primary_p == primary_p else 0
        rec = "ВЫКАТИТЬ" if primary_sig and lift > 0 else (
            "ОТКЛОНИТЬ" if primary_sig and lift < 0 else "ПРОДОЛЖИТЬ ТЕСТ")

        sticky = html.Div([
            html.Div([
                _overline("РЕЗУЛЬТАТ A/B"),
                html.Div([
                    html.Span(f"LIFT {lift:+.2f}%", className="lift",
                              style={"fontSize": "22px",
                                     "fontWeight": 600,
                                     "fontFamily": "'JetBrains Mono', monospace"}),
                    _chip(f"{'HIGH' if conf_pct > 95 else 'MEDIUM'} CONFIDENCE · "
                          f"{conf_pct:.1f}%",
                          "success" if primary_sig else "warning"),
                ], style={"display": "flex", "alignItems": "center",
                          "gap": "12px", "marginTop": "4px"}),
            ]),
            html.Div([
                _metric("P-VALUE", f"{primary_p:.4g}",
                        "ttest"),
                _metric("MDE @ 80%", f"d ≈ 0.20",
                        "минимум"),
                _metric("ДО СТАТ-ЗНАЧ.",
                        "достигнуто" if primary_sig else "не достигнуто",
                        ""),
                _metric("РЕКОМЕНДАЦИЯ", rec,
                        "auto"),
            ], style={"display": "grid",
                      "gridTemplateColumns": "repeat(4, 1fr)",
                      "gap": "20px", "marginLeft": "auto"}),
        ], className="sticky-bar",
            style={"display": "flex", "alignItems": "center",
                   "gap": "20px", "marginTop": "12px",
                   "padding": "16px 20px",
                   "background": "var(--surface-1)",
                   "border": "1px solid var(--border-subtle)",
                   "borderRadius": "12px"})

        return html.Div([
            html.Div([card_ctrl, card_trt], className="grid-2",
                     style={"gap": "16px"}),
            sticky,
        ]), pv_list
    except Exception as e:
        logger.exception("ab")
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ===========================================================================
# Diagnostics callback
# ===========================================================================

@callback(
    Output("diag-result-area", "children"),
    Input("btn-diag", "n_clicks"),
    State("diag-val-col", "value"),
    State("diag-grp-col", "value"),
    State("diag-ga", "value"),
    State("diag-gb", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _run_diagnostics(n, val_col, grp_col, ga, gb, alpha_tick,
                     ds_name, raw, prep):
    if not all([val_col, grp_col, ga, gb, ds_name]):
        return alert_banner("Заполните все поля.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")
    alpha = (alpha_tick or 5) / 100
    a_s = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
    b_s = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
    if len(a_s) < 3 or len(b_s) < 3:
        return alert_banner("В каждой группе нужно минимум 3 наблюдения.",
                            "warning")

    try:
        diag = diagnose_groups(a_s, b_s, alpha=alpha,
                               label_a=str(ga), label_b=str(gb))
    except Exception as e:
        return alert_banner(f"Ошибка диагностики: {e}", "danger")

    norm_a, norm_b, lev = diag["norm_a"], diag["norm_b"], diag["levene"]

    # Recommendation card (.card.accent)
    rec_card = html.Div([
        html.Div([
            html.Div([
                _chip("РЕКОМЕНДУЕМЫЙ ТЕСТ", "accent"),
                html.H2(diag["rec_name"], style={"marginTop": "10px"}),
                html.P(diag["rec_reason"], className="caption",
                       style={"marginTop": "6px", "maxWidth": "640px"}),
            ], style={"flex": 1}),
            html.Div([
                _metric("НОРМАЛЬНОСТЬ A",
                        "✓ да" if norm_a.is_normal else "✕ нет",
                        f"{norm_a.test_name} p = {norm_a.p_value:.4g}"),
                _metric("НОРМАЛЬНОСТЬ B",
                        "✓ да" if norm_b.is_normal else "✕ нет",
                        f"{norm_b.test_name} p = {norm_b.p_value:.4g}"),
                _metric("ДИСПЕРСИИ",
                        "РАЗНЫЕ" if lev.significant else "РАВНЫ",
                        f"Levene p = {lev.p_value:.4g}"),
            ], style={"display": "flex", "gap": "20px",
                      "paddingLeft": "16px",
                      "borderLeft": "1px solid var(--border-subtle)"}),
        ], style={"display": "flex",
                  "justifyContent": "space-between",
                  "alignItems": "flex-start", "gap": "16px"}),
    ], className="card accent", style={"padding": "20px"})

    # Three side-by-side cards
    def _norm_block(nr: NormalityResult, label: str, group_label: str) -> html.Div:
        ok = nr.is_normal
        kind = "success" if ok else "danger"
        chip_text = "✓ PASS" if ok else "✕ FAIL"
        return html.Div([
            html.Div([
                html.Div([
                    _overline(f"ГРУППА · {group_label}"),
                    html.H3("Нормальна" if ok else "Не нормальна",
                            style={"marginTop": "6px"}),
                ]),
                _chip(chip_text, kind),
            ], style={"display": "flex",
                      "justifyContent": "space-between",
                      "alignItems": "flex-start"}),
            html.Div([
                _metric(nr.test_name + " W", f"{nr.statistic:.3f}", ""),
                _metric("p-value", f"{nr.p_value:.4g}",
                        # NormalityResult: is_normal == True ⇔ p ≥ α
                        # (нулевую гипотезу о нормальности НЕ отвергаем).
                        "< α" if not nr.is_normal else "≥ α"),
                _metric("Skewness", f"{nr.skewness:+.2f}", nr.skew_label),
                _metric("Kurtosis", f"{nr.kurtosis:+.2f}", nr.kurt_label),
            ], style={"display": "grid",
                      "gridTemplateColumns": "1fr 1fr",
                      "gap": "10px", "marginTop": "14px"}),
        ], className=f"card {'success' if ok else 'danger'}",
           style={"padding": "16px"})

    levene_block = html.Div([
        html.Div([
            html.Div([
                _overline("ОДНОРОДНОСТЬ ДИСПЕРСИЙ"),
                html.H3("Levene", style={"marginTop": "6px"}),
            ]),
            _chip("✕ FAIL" if lev.significant else "✓ PASS",
                  "danger" if lev.significant else "success"),
        ], style={"display": "flex",
                  "justifyContent": "space-between",
                  "alignItems": "flex-start"}),
        html.Div([
            _metric("F-stat", f"{lev.statistic:.3f}", ""),
            _metric("p-value", f"{lev.p_value:.4g}",
                    "< α" if lev.significant else "≥ α"),
            _metric("σ²ₐ", f"{float(np.var(a_s, ddof=1)):.3g}", ""),
            _metric("σ²ᵦ", f"{float(np.var(b_s, ddof=1)):.3g}", ""),
        ], style={"display": "grid",
                  "gridTemplateColumns": "1fr 1fr",
                  "gap": "10px", "marginTop": "14px"}),
    ], className="card", style={"padding": "16px"})

    grid = html.Div([
        _norm_block(norm_a, "Группа A", str(ga)),
        _norm_block(norm_b, "Группа B", str(gb)),
        levene_block,
    ], className="grid-3", style={"gap": "12px", "marginTop": "12px"})

    warnings = diag.get("warnings", [])
    warn_block = html.Div(
        [_alert("Предупреждение", w, kind="warning", icon="⚠")
         for w in warnings]) if warnings else html.Div()

    return html.Div([rec_card, grid, warn_block])


# ===========================================================================
# BH correction callback
# ===========================================================================

@callback(
    Output("bh-result", "children"),
    Input("btn-bh", "n_clicks"),
    Input("bh-filter", "value"),
    State("tests-all-pvalues", "data"),
    State("tests-alpha", "value"),
    prevent_initial_call=True,
)
def _run_bh(n, fltr, all_pv, alpha_tick):
    if not all_pv:
        return alert_banner(
            "Нет p-значений. Сначала запустите тесты на других вкладках.",
            "warning")
    alpha = (alpha_tick or 5) / 100
    try:
        # all_pv is list of {"test","var","p"}
        p_only = [float(x["p"]) for x in all_pv if x.get("p") == x.get("p")]
        corrections = bh_correction(p_only, alpha=alpha)
        # corrections returned in same input order
        rows = []
        for i, c in enumerate(corrections):
            entry = all_pv[i] if i < len(all_pv) else {}
            sig = c["significant"]
            if fltr == "sig" and not sig:
                continue
            if fltr == "ns" and sig:
                continue
            rows.append(html.Tr([
                html.Td(entry.get("test", "—")),
                html.Td(entry.get("var", "—"), className="mono",
                        style={"fontSize": "12px"}),
                html.Td(f"{c['p_value']:.5f}", className="mono"),
                html.Td(f"{c['adjusted_p']:.5f}", className="mono"),
                html.Td(f"{i + 1}", className="mono"),
                html.Td(_chip("✓ значим", "success") if sig
                        else _chip("—", "neutral")),
            ], className="significant" if sig else ""))

        n_total = len(corrections)
        n_sig_raw = sum(1 for c in corrections if c["p_value"] < alpha)
        n_sig_bh = sum(1 for c in corrections if c["significant"])
        fdr_actual = (n_sig_raw - n_sig_bh) / n_sig_raw * 100 if n_sig_raw else 0

        table_card = html.Div([
            html.Div(html.Table([
                html.Thead(html.Tr([
                    html.Th("Тест"), html.Th("Переменная"),
                    html.Th("p raw", className="mono"),
                    html.Th("p adj (BH)", className="mono"),
                    html.Th("Ранг"), html.Th("Значим?"),
                ])),
                html.Tbody(rows),
            ], className="tbl"),
                style={"maxHeight": "320px", "overflow": "auto"}),
        ], className="card", style={"padding": 0,
                                     "overflow": "hidden",
                                     "marginTop": "12px"})

        kpis = html.Div([
            _kpi("ВСЕГО ТЕСТОВ", f"{n_total}", "собрано"),
            _kpi("ЗНАЧИМЫХ (RAW)", f"{n_sig_raw}", "p < α"),
            _kpi("ЗНАЧИМЫХ (BH)", f"{n_sig_bh}", "после поправки"),
            _kpi("FDR ФАКТ.", f"{fdr_actual:.1f}%",
                 f"ожидаемо ≤ {alpha*100:.0f}%"),
        ], className="grid-4",
            style={"gap": "12px", "marginTop": "12px"})

        return html.Div([table_card, kpis])
    except Exception as e:
        logger.exception("bh")
        return alert_banner(f"Ошибка: {e}", "danger")


# ===========================================================================
# Power analysis callback
# ===========================================================================

@callback(
    Output("power-result", "children"),
    Input("btn-power", "n_clicks"),
    State("pw-test-type", "value"),
    State("pw-solve-for", "value"),
    State("pw-effect", "value"),
    State("pw-alpha", "value"),
    State("pw-power", "value"),
    State("pw-n", "value"),
    State("pw-k-groups", "value"),
    State("pw-ratio", "value"),
    State("pw-alt", "value"),
    prevent_initial_call=True,
)
def _run_power(n_click, test_type, solve_for, effect, alpha, power_val,
               n_obs, k_groups, ratio, alt):
    from statsmodels.stats.power import (
        TTestIndPower, TTestPower, NormalIndPower, FTestAnovaPower,
    )
    try:
        effect = float(effect or 0.35)
        alpha = float(alpha or 0.05)
        power_val = float(power_val or 0.80)
        n_obs = int(n_obs or 50)
        k_groups = int(k_groups or 3)
        ratio = float(ratio or 1.0)

        # Build kwargs and overwrite what we're solving for.
        kw_common = {"effect_size": effect, "alpha": alpha,
                     "power": power_val, "nobs1": n_obs}
        if solve_for == "nobs":
            kw_common["nobs1"] = None
        elif solve_for == "power":
            kw_common["power"] = None
        elif solve_for == "effect_size":
            kw_common["effect_size"] = None

        if test_type == "ttest2":
            analysis = TTestIndPower()
            kw = dict(kw_common); kw["ratio"] = ratio
            result_val = analysis.solve_power(**kw)
        elif test_type == "ttest1":
            analysis = TTestPower()
            kw = {"effect_size": kw_common["effect_size"],
                  "alpha": kw_common["alpha"],
                  "power": kw_common["power"],
                  "nobs": kw_common["nobs1"]}
            result_val = analysis.solve_power(**kw)
        elif test_type == "prop":
            analysis = NormalIndPower()
            kw = dict(kw_common); kw["ratio"] = ratio
            result_val = analysis.solve_power(**kw)
        else:  # anova
            analysis = FTestAnovaPower()
            kw = {"effect_size": kw_common["effect_size"],
                  "alpha": kw_common["alpha"],
                  "power": kw_common["power"],
                  "nobs": kw_common["nobs1"],
                  "k_groups": k_groups}
            result_val = analysis.solve_power(**kw)

        if result_val is None or not np.isfinite(result_val):
            return [alert_banner("Не удалось вычислить результат.", "danger")]

        # Solve label / display value
        if solve_for == "nobs":
            n_target = int(np.ceil(result_val))
            display_val = f"{n_target}"
            overline_text = "ИСКОМЫЙ РАЗМЕР ВЫБОРКИ (НА ГРУППУ)"
            sub_text = f"всего {n_target * 2} наблюдений"
        elif solve_for == "power":
            display_val = f"{result_val:.1%}"
            n_target = n_obs
            overline_text = "ИСКОМАЯ МОЩНОСТЬ"
            sub_text = f"при n = {n_obs}, d = {effect}"
        else:
            display_val = f"{result_val:.4f}"
            n_target = n_obs
            overline_text = "ИСКОМЫЙ РАЗМЕР ЭФФЕКТА"
            sub_text = f"d при n = {n_obs}, power = {power_val}"

        # Big result card (.card.accent)
        big = html.Div([
            html.Div([
                _overline(overline_text),
                html.Div(display_val, className="mono",
                         style={"fontSize": "44px",
                                "lineHeight": "52px",
                                "fontWeight": 500,
                                "color": "var(--accent-300, #5fd8c0)",
                                "letterSpacing": "-0.02em",
                                "marginTop": "6px"}),
                html.Div(sub_text, className="caption",
                         style={"marginTop": "2px"}),
            ]),
            html.Div([
                _chip("ДОСТИЖИМО В ТЕКУЩЕМ ДАТАСЕТЕ", "accent"),
                _chip(f"MDE @ n={n_target} → d = {effect:.2f}", "success"),
                _chip("accuracy · ± 2", "neutral"),
            ], style={"display": "flex", "flexDirection": "column",
                      "gap": "10px", "alignItems": "flex-end"}),
        ], className="card accent",
            style={"padding": "24px",
                   "display": "flex",
                   "justifyContent": "space-between",
                   "alignItems": "center"})

        # Power curve
        curve_n = (n_target if solve_for == "nobs"
                   else n_obs)
        fig = _power_curve_fig(test_type, effect, alpha, curve_n,
                               k_groups, ratio)
        curve_card = html.Div(dcc.Graph(figure=fig),
                              className="card",
                              style={"padding": "12px"})

        # 4 KPIs at different power levels
        def _solve_n_at_power(p_val: float) -> str:
            try:
                if test_type == "ttest2":
                    v = TTestIndPower().solve_power(
                        effect_size=effect, alpha=alpha,
                        power=p_val, nobs1=None, ratio=ratio)
                elif test_type == "ttest1":
                    v = TTestPower().solve_power(
                        effect_size=effect, alpha=alpha,
                        power=p_val, nobs=None)
                elif test_type == "prop":
                    v = NormalIndPower().solve_power(
                        effect_size=effect, alpha=alpha,
                        power=p_val, nobs1=None, ratio=ratio)
                else:
                    v = FTestAnovaPower().solve_power(
                        effect_size=effect, alpha=alpha,
                        power=p_val, nobs=None, k_groups=k_groups)
                if v is None or not np.isfinite(v):
                    return "—"
                return f"{int(np.ceil(v))}"
            except Exception:
                return "—"

        kpis_row = html.Div([
            _kpi("@ POWER 0.7", _solve_n_at_power(0.7), "obs"),
            _kpi("@ POWER 0.8", _solve_n_at_power(0.8), "obs"),
            _kpi("@ POWER 0.9", _solve_n_at_power(0.9), "obs"),
            _kpi("@ POWER 0.95", _solve_n_at_power(0.95), "obs"),
        ], className="grid-4", style={"gap": "12px"})

        return [big, curve_card, kpis_row]
    except Exception as e:
        logger.exception("power")
        return [alert_banner(f"Ошибка анализа мощности: {e}", "danger")]


# ===========================================================================
# Compute effect size from data (auxiliary)
# ===========================================================================

@callback(
    Output("pw-effect", "value"),
    Output("pw-effect-hint", "children"),
    Input("btn-calc-effect", "n_clicks"),
    State("pw-col-num", "value"),
    State("pw-col-grp", "value"),
    State("pw-test-type", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def _calc_effect_from_data(n, col_num, col_grp, test_type, ds_name, raw, prep):
    if not col_num:
        return no_update, alert_banner("Выберите числовую колонку.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return no_update, alert_banner("Датасет не загружен.", "danger")
    try:
        series = df[col_num].dropna()
        if test_type in ("ttest2", "anova") and col_grp:
            groups = {k: v.dropna().values
                      for k, v in df.groupby(col_grp)[col_num]}
            names = list(groups.keys())
            arrays = [groups[k] for k in names]
            if test_type == "ttest2":
                if len(arrays) < 2:
                    return no_update, alert_banner(
                        "Нужно минимум 2 группы.", "warning")
                a1, a2 = arrays[0], arrays[1]
                n1, n2 = len(a1), len(a2)
                pooled = np.sqrt(((n1 - 1) * a1.std() ** 2
                                  + (n2 - 1) * a2.std() ** 2)
                                 / max(n1 + n2 - 2, 1))
                d = abs(a1.mean() - a2.mean()) / pooled if pooled > 0 else 0.0
                hint = (f"Cohen's d = {d:.4f}  ({names[0]}: μ={a1.mean():.3g}, "
                        f"n={n1}  vs  {names[1]}: μ={a2.mean():.3g}, n={n2})")
                return round(float(d), 4), _alert("Готово", hint, "success", "✓")
            else:  # anova → Cohen's f
                grand = series.mean()
                ss_b = sum(len(a) * (a.mean() - grand) ** 2 for a in arrays)
                ss_w = sum(((a - a.mean()) ** 2).sum() for a in arrays)
                tot = ss_b + ss_w
                eta_sq = ss_b / tot if tot > 0 else 0.0
                f = np.sqrt(eta_sq / (1 - eta_sq)) if 0 < eta_sq < 1 else 0.0
                hint = (f"Cohen's f = {f:.4f}  (η² = {eta_sq:.4f}, "
                        f"{len(arrays)} групп)")
                return round(float(f), 4), _alert("Готово", hint, "success", "✓")
        else:
            mu, sigma = series.mean(), series.std()
            d = abs(mu) / sigma if sigma > 0 else 0.0
            hint = (f"Cohen's d = {d:.4f}  (μ={mu:.4g}, σ={sigma:.4g}, "
                    f"n={len(series)})")
            return round(float(d), 4), _alert("Готово", hint, "success", "✓")
    except Exception as e:
        return no_update, alert_banner(f"Ошибка: {e}", "danger")

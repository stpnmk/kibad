"""
p06_tests.py -- Статистические тесты (Dash).

t-тест, Манна-Уитни, хи-квадрат, корреляция, бутстрап, перестановочный,
A/B-тест, пакетное тестирование с поправкой BH.
"""
from __future__ import annotations

import json
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

from app.figure_theme import apply_kibad_theme
from app.state import (
    STORE_DATASET, STORE_ACTIVE_DS, STORE_PREPARED,
    STORE_TEST_RESULTS, get_df_from_store, list_datasets,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.cards import stat_card
from app.components.table import data_table
from app.components.form import select_input, slider_input, number_input, checklist_input
from app.components.alerts import alert_banner

from core.tests import (
    ttest_independent, mann_whitney, chi_square_independence,
    correlation_test, bootstrap_test, ab_test, bh_correction,
    permutation_test, TestResult,
    normality_test, levene_test, diagnose_groups, NormalityResult,
    cliffs_delta, anova_oneway, kruskal_wallis,
)

dash.register_page(
    __name__,
    path="/tests",
    name="6. Тесты",
    order=6,
    icon="clipboard-data",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result_to_dict(r: TestResult) -> dict:
    return dict(
        name=r.name, statistic=r.statistic, p_value=r.p_value,
        alpha=r.alpha, significant=r.significant,
        effect_size=r.effect_size, effect_label=r.effect_label,
        ci=list(r.ci) if r.ci else None,
        interpretation=r.interpretation,
    )


def _result_card_div(r: dict, alpha: float) -> html.Div:
    is_sig = r["significant"]
    color = "#1b3a26" if is_sig else "#3a1b1b"
    border = "#10b981" if is_sig else "#ef4444"
    verdict = "ЗНАЧИМО" if is_sig else "НЕ ЗНАЧИМО"
    icon = "bi-check-circle-fill" if is_sig else "bi-x-circle-fill"

    children = [
        html.Div([
            html.I(className=f"bi {icon}", style={"marginRight": "8px"}),
            html.Strong(r["name"]),
            html.Span(f" -- {verdict} при alpha={alpha}", style={"marginLeft": "8px"}),
        ], style={"fontSize": "1.05rem", "marginBottom": "6px"}),
        html.Div([
            html.Span(f"p-value = {r['p_value']:.6f}", style={"marginRight": "16px"}),
            html.Span(f"Статистика = {r['statistic']:.4f}", style={"marginRight": "16px"}),
            html.Span(
                f"Размер эффекта = {r['effect_size']:.4f}" if r["effect_size"] is not None else "Размер эффекта = N/A",
                style={"marginRight": "16px"},
            ),
            html.Span(r["effect_label"] or ""),
        ], style={"fontSize": "0.85rem", "color": "#9ba3b8"}),
    ]
    if r.get("ci"):
        children.append(
            html.Div(f"95% ДИ: [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
                      style={"fontSize": "0.85rem", "color": "#9ba3b8", "marginTop": "4px"})
        )
    children.append(
        html.Div(r["interpretation"],
                 style={"marginTop": "8px", "fontSize": "0.9rem", "color": "#c8cad4"})
    )
    return html.Div(children, style={
        "background": color, "border": f"1px solid {border}",
        "borderRadius": "8px", "padding": "14px 18px", "marginBottom": "12px",
    })


def _overlay_hist(a_vals, b_vals, label_a, label_b, val_name) -> go.Figure:
    fig = go.Figure()
    for vals, name, color in [(a_vals, label_a, "#4b9eff"), (b_vals, label_b, "#ef4444")]:
        fig.add_trace(go.Histogram(
            x=vals, name=name, opacity=0.55,
            marker_color=color, nbinsx=30, histnorm="probability density",
        ))
        fig.add_vline(x=float(np.mean(vals)), line_dash="dash", line_color=color,
                      annotation_text=f"mu={np.mean(vals):.2f}")
    fig.update_layout(
        barmode="overlay",
        title=f"Распределение '{val_name}' по группам",
        xaxis_title=val_name, yaxis_title="Плотность",
        height=380,
    )
    return apply_kibad_theme(fig)


def _ecdf_fig(a_vals, b_vals, label_a, label_b, val_name) -> go.Figure:
    fig = go.Figure()
    for vals, name, color in [(a_vals, label_a, "#4b9eff"), (b_vals, label_b, "#ef4444")]:
        sorted_v = np.sort(vals)
        y = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        fig.add_trace(go.Scatter(
            x=sorted_v, y=y, name=name, mode="lines",
            line=dict(color=color, width=2),
            hovertemplate=f"{name}<br>x=%{{x:.3f}}<br>ECDF=%{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(
        title=f"ECDF: '{val_name}'",
        xaxis_title=val_name, yaxis_title="Кумулятивная вероятность",
        height=340,
    )
    return apply_kibad_theme(fig)


def _cliff_gauge(delta: float) -> go.Figure:
    abs_d = abs(delta)
    thresholds = [0, 0.147, 0.33, 0.474, 1.0]
    labels = ["negligible", "small", "medium", "large"]
    colors = ["#505872", "#3b82f6", "#f59e0b", "#ef4444"]
    idx = next((i for i, t in enumerate(thresholds[1:]) if abs_d <= t), 3)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(delta, 3),
        number={"font": {"size": 26}},
        title={"text": "Cliff's δ", "font": {"size": 14}},
        gauge={
            "axis": {"range": [-1, 1], "tickwidth": 1},
            "bar": {"color": colors[idx], "thickness": 0.25},
            "steps": [
                {"range": [-1, -0.474], "color": "#3a1520"},
                {"range": [-0.474, -0.33], "color": "#3a2615"},
                {"range": [-0.33, -0.147], "color": "#1a2a3a"},
                {"range": [-0.147, 0.147], "color": "#1a1f2a"},
                {"range": [0.147, 0.33], "color": "#1a2a3a"},
                {"range": [0.33, 0.474], "color": "#3a2615"},
                {"range": [0.474, 1], "color": "#3a1520"},
            ],
            "threshold": {"line": {"color": colors[idx], "width": 3}, "value": delta},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=40, b=20, l=24, r=24))
    apply_kibad_theme(fig, preset="gauge")
    return fig


def _ci_bars(a_vals, b_vals, label_a, label_b, val_name) -> go.Figure:
    fig = go.Figure()
    for vals, name, color in [(a_vals, label_a, "#4b9eff"), (b_vals, label_b, "#ef4444")]:
        m = np.mean(vals)
        se = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
        fig.add_trace(go.Bar(
            x=[name], y=[m], marker_color=color,
            error_y=dict(type="data", symmetric=True, array=[1.96 * se]),
            text=[f"{m:.3f}"], textposition="outside",
        ))
    fig.update_layout(
        title=f"Среднее +/- 95% ДИ: '{val_name}'",
        yaxis_title=val_name, showlegend=False, barmode="group", height=380,
    )
    return apply_kibad_theme(fig)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container([
    page_header("6. Статистические тесты",
                "Проверка гипотез: t-тест, Манна-Уитни, хи-квадрат, корреляция, бутстрап, A/B",
                "bi-clipboard-data"),

    # Controls row
    dbc.Row([
        dbc.Col(select_input("Датасет", "tests-ds-select", [], placeholder="Выберите датасет..."), md=3),
        dbc.Col(slider_input("Уровень значимости (alpha)", "tests-alpha", 1, 10, 5, 1,
                             marks={1: "0.01", 5: "0.05", 10: "0.10"}), md=4),
        dbc.Col(html.Div(id="tests-data-stats"), md=5),
    ], className="mb-3"),

    html.Hr(),

    # Tabs
    dbc.Tabs(id="tests-tabs", active_tab="tab-diag", children=[
        dbc.Tab(label="🔍 Диагностика", tab_id="tab-diag"),
        dbc.Tab(label="t-Тест", tab_id="tab-ttest"),
        dbc.Tab(label="Манна-Уитни", tab_id="tab-mw"),
        dbc.Tab(label="Хи-квадрат", tab_id="tab-chi2"),
        dbc.Tab(label="Корреляция", tab_id="tab-corr"),
        dbc.Tab(label="Бутстрап", tab_id="tab-boot"),
        dbc.Tab(label="Перестановочный", tab_id="tab-perm"),
        dbc.Tab(label="ANOVA / Краскел-Уоллис", tab_id="tab-anova"),
        dbc.Tab(label="A/B тест", tab_id="tab-ab"),
        dbc.Tab(label="Мн. сравнения (BH)", tab_id="tab-bh"),
        dbc.Tab(label="⚡ Мощность и размер выборки", tab_id="tab-power"),
    ]),

    html.Div(id="tests-tab-content", className="mt-3"),

    # Hidden stores for intermediate results
    dcc.Store(id="tests-last-result", storage_type="memory"),
    dcc.Store(id="tests-all-pvalues", storage_type="memory", data=[]),

], fluid=True, className="kb-page")


# ---------------------------------------------------------------------------
# Populate dataset dropdown
# ---------------------------------------------------------------------------
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


@callback(
    Output("tests-data-stats", "children"),
    Input("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _show_stats(ds_name, raw, prep):
    if not ds_name:
        return ""
    df = get_df_from_store(prep, ds_name) if prep and ds_name in (prep or {}) else get_df_from_store(raw, ds_name)
    if df is None:
        return ""
    num_c = len(df.select_dtypes(include="number").columns)
    cat_c = len(df.select_dtypes(include="object").columns)
    return dbc.Row([
        dbc.Col(stat_card("Строк", f"{len(df):,}")),
        dbc.Col(stat_card("Числовых", num_c)),
        dbc.Col(stat_card("Категориальных", cat_c)),
    ])


# ---------------------------------------------------------------------------
# Tab content renderer
# ---------------------------------------------------------------------------

def _get_df(ds_name, raw, prep):
    if not ds_name:
        return None
    df = get_df_from_store(prep, ds_name) if prep and ds_name in (prep or {}) else get_df_from_store(raw, ds_name)
    return df


def _two_group_controls(prefix: str, num_cols, cat_cols):
    """Reusable controls for two-group tests."""
    return [
        dbc.Row([
            dbc.Col(select_input("Числовая колонка", f"{prefix}-val", num_cols), md=6),
            dbc.Col(select_input("Группирующая колонка", f"{prefix}-grp", cat_cols), md=6),
        ]),
        dbc.Row([
            dbc.Col(select_input("Группа A", f"{prefix}-ga", []), md=6),
            dbc.Col(select_input("Группа B", f"{prefix}-gb", []), md=6),
        ]),
    ]


@callback(
    Output("tests-tab-content", "children"),
    Input("tests-tabs", "active_tab"),
    Input("tests-ds-select", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def _render_tab(tab, ds_name, raw, prep):
    if tab == "tab-power":
        df = _get_df(ds_name, raw, prep)
        has_data = df is not None
        num_opts = (
            [{"label": c, "value": c} for c in df.select_dtypes(include="number").columns.tolist()]
            if has_data else []
        )
        cat_opts = (
            [{"label": c, "value": c} for c in df.select_dtypes(include=["object", "category"]).columns.tolist()]
            if has_data else []
        )
        return html.Div([
            section_header("Анализ мощности и размер выборки",
                           "Расчёт необходимого n, мощности теста и минимального обнаруживаемого эффекта"),
            dbc.Alert(
                "Мощность (1-β) — вероятность обнаружить эффект, если он существует. "
                "Стандарт: 80% мощности при alpha=0.05. Задайте 2 из 3 параметров — KIBAD рассчитает третий.",
                color="info", className="mb-3",
            ),
            dbc.Row([
                dbc.Col([
                    html.Label("Тип теста", className="kb-stat-label"),
                    dcc.Dropdown(id="pw-test-type",
                        options=[
                            {"label": "t-тест (две независимые группы)", "value": "ttest2"},
                            {"label": "t-тест (одна выборка)", "value": "ttest1"},
                            {"label": "Пропорции (z-тест)", "value": "prop"},
                            {"label": "ANOVA (k групп)", "value": "anova"},
                        ],
                        value="ttest2", clearable=False, className="kb-select",
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Что вычислить", className="kb-stat-label"),
                    dcc.Dropdown(id="pw-solve-for",
                        options=[
                            {"label": "Размер выборки (n)", "value": "nobs"},
                            {"label": "Мощность (power)", "value": "power"},
                            {"label": "Размер эффекта (effect size)", "value": "effect_size"},
                        ],
                        value="nobs", clearable=False, className="kb-select",
                    ),
                ], md=4),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(number_input("Размер эффекта (Cohen's d / f)", "pw-effect", value=0.5, min_val=0.01, max_val=5.0, step=0.01), md=3),
                dbc.Col(number_input("Уровень значимости (alpha)", "pw-alpha", value=0.05, min_val=0.001, max_val=0.5, step=0.001), md=3),
                dbc.Col(number_input("Мощность (1 - beta)", "pw-power", value=0.80, min_val=0.5, max_val=0.999, step=0.01), md=3),
                dbc.Col(number_input("Размер выборки (n на группу)", "pw-n", value=50, min_val=2, max_val=100000, step=1), md=3),
            ], className="mb-3"),
            dbc.Row([
                dbc.Col(number_input("Кол-во групп (ANOVA)", "pw-k-groups", value=3, min_val=2, max_val=20, step=1), md=3),
            ], className="mb-3"),
            dbc.Button("⚡ Рассчитать", id="btn-power", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="power-result"), type="circle", color="#10b981"),

            html.Hr(className="my-4"),
            html.Small("Автоматический расчёт размера эффекта из данных",
                       className="fw-bold d-block mb-2",
                       style={"color": "#8891a5", "textTransform": "uppercase",
                              "fontSize": "0.75rem", "letterSpacing": "0.05em"}),
            dbc.Row([
                dbc.Col([
                    html.Label("Числовая колонка", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="pw-col-num", options=num_opts,
                        placeholder="Выберите числовую колонку..." if has_data else "Загрузите датасет...",
                        disabled=not has_data,
                    ),
                ], md=4),
                dbc.Col([
                    html.Label("Группирующая колонка (для t-теста 2 выб. / ANOVA)", className="kb-stat-label"),
                    dcc.Dropdown(
                        id="pw-col-grp", options=cat_opts,
                        placeholder="Опционально..." if has_data else "Загрузите датасет...",
                        disabled=not has_data,
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Button("↗ Вычислить из данных", id="btn-calc-effect",
                               color="outline-primary", size="sm", className="mt-4",
                               disabled=not has_data),
                ], md=4),
            ]),
            html.Div(id="pw-effect-hint", className="mt-2"),
        ])

    df = _get_df(ds_name, raw, prep)
    if df is None:
        return empty_state("bi-clipboard-data", "Нет данных",
                           "Выберите датасет в выпадающем списке выше.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if tab == "tab-diag":
        if not num_cols or not cat_cols:
            return alert_banner(
                "Для диагностики нужна хотя бы одна числовая и одна категориальная колонка.",
                "warning",
            )
        return html.Div([
            section_header(
                "Диагностика данных перед выбором теста",
                "Проверьте нормальность и однородность дисперсий — KIBAD подберёт оптимальный тест",
            ),
            dbc.Alert(
                "Зачем это нужно? Выбор корректного статистического теста зависит от структуры данных. "
                "Здесь вы оцениваете нормальность каждой группы и однородность дисперсий — "
                "и получаете автоматическую рекомендацию, какой тест использовать.",
                color="info", className="mb-3",
            ),
            dbc.Row([
                dbc.Col(select_input("Числовая колонка", "diag-val-col", num_cols), md=4),
                dbc.Col(select_input("Группирующая колонка", "diag-grp-col", cat_cols), md=4),
                dbc.Col(html.Div(id="diag-group-selects"), md=4),
            ], className="mb-2"),
            dbc.Button("🔍 Запустить диагностику", id="btn-diag", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="diag-result-area"), type="circle", color="#10b981"),
        ])

    if tab == "tab-ttest":
        return html.Div([
            section_header("t-тест (Уэлча / Стьюдента)",
                           "H0: Средние двух групп равны."),
            *_two_group_controls("tt", num_cols, cat_cols),
            dbc.Checklist(
                id="tt-eqvar",
                options=[{"label": " Предполагать равные дисперсии (Стьюдент)", "value": "eq"}],
                value=[], inline=True,
                className="mb-3", style={"color": "#9ba3b8"},
            ),
            dbc.Button("Запустить t-тест", id="btn-ttest", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="ttest-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-mw":
        return html.Div([
            section_header("U-тест Манна-Уитни",
                           "Непараметрический тест для двух независимых групп."),
            *_two_group_controls("mw", num_cols, cat_cols),
            dbc.Button("Запустить Манна-Уитни", id="btn-mw", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="mw-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-chi2":
        return html.Div([
            section_header("Хи-квадрат тест независимости",
                           "Проверка связи между двумя категориальными переменными."),
            dbc.Row([
                dbc.Col(select_input("Колонка A", "chi2-col-a", cat_cols), md=6),
                dbc.Col(select_input("Колонка B", "chi2-col-b", cat_cols), md=6),
            ]),
            dbc.Button("Запустить хи-квадрат", id="btn-chi2", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="chi2-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-corr":
        return html.Div([
            section_header("Корреляционный тест",
                           "Пирсон (линейная) или Спирмен (монотонная) связь."),
            dbc.Row([
                dbc.Col(select_input("Переменная X", "corr-x", num_cols), md=4),
                dbc.Col(select_input("Переменная Y", "corr-y", num_cols), md=4),
                dbc.Col(select_input("Метод", "corr-method",
                                     [{"label": "Пирсон", "value": "pearson"},
                                      {"label": "Спирмен", "value": "spearman"},
                                      {"label": "Кендалл", "value": "kendall"}],
                                     value="pearson"), md=4),
            ]),
            dbc.Button("Запустить тест корреляции", id="btn-corr", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="corr-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-boot":
        return html.Div([
            section_header("Бутстрап-тест",
                           "Непараметрический тест на основе повторных выборок."),
            *_two_group_controls("bt", num_cols, cat_cols),
            dbc.Row([
                dbc.Col(number_input("Число итераций", "bt-n", value=5000, min_val=500, max_val=50000, step=500), md=4),
                dbc.Col(select_input("Статистика", "bt-stat",
                                     [{"label": "Среднее", "value": "mean"},
                                      {"label": "Медиана", "value": "median"}],
                                     value="mean"), md=4),
            ]),
            dbc.Button("Запустить бутстрап", id="btn-boot", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="boot-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-perm":
        return html.Div([
            section_header("Перестановочный тест",
                           "Непараметрический тест: случайное перемешивание меток."),
            *_two_group_controls("pm", num_cols, cat_cols),
            dbc.Row([
                dbc.Col(number_input("Число перестановок", "pm-n", value=10000, min_val=1000, max_val=100000, step=1000), md=4),
            ]),
            dbc.Button("Запустить перестановочный тест", id="btn-perm", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="perm-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-anova":
        return html.Div([
            section_header("ANOVA / Краскел-Уоллис",
                           "Сравнение средних (медиан) в ≥3 группах с post-hoc анализом."),
            dbc.Alert(
                "ANOVA — параметрический тест (требует нормальность). "
                "Краскел-Уоллис — непараметрический аналог. "
                "Post-hoc анализ автоматически запускается при значимом результате.",
                color="info", className="mb-3",
            ),
            dbc.Row([
                dbc.Col(select_input("Числовая колонка", "anova-val", num_cols), md=4),
                dbc.Col(select_input("Группирующая колонка", "anova-grp", cat_cols), md=4),
                dbc.Col(select_input("Метод", "anova-method",
                                     [{"label": "ANOVA (параметрический)", "value": "anova"},
                                      {"label": "Краскел-Уоллис (непараметрический)", "value": "kruskal"}],
                                     value="anova"), md=4),
            ]),
            dbc.Button("Запустить анализ", id="btn-anova", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="anova-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-ab":
        return html.Div([
            section_header("A/B тест",
                           "Комплексный анализ двух групп с несколькими метриками."),
            *_two_group_controls("ab", num_cols, cat_cols),
            dbc.Button("Запустить A/B тест", id="btn-ab", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="ab-result"), type="circle", color="#10b981"),
        ])

    if tab == "tab-bh":
        return html.Div([
            section_header("Поправка Бенджамини-Хохберга (BH/FDR)",
                           "Коррекция p-значений при множественных сравнениях."),
            html.P("Запустите несколько тестов на других вкладках, затем нажмите кнопку ниже "
                   "для пакетной коррекции.", style={"color": "#9ba3b8"}),
            dbc.Button("Применить поправку BH", id="btn-bh", color="primary", className="mb-3"),
            dcc.Loading(html.Div(id="bh-result"), type="circle", color="#10b981"),
        ])

    return html.Div("Выберите вкладку.")


# ---------------------------------------------------------------------------
# Group selectors auto-fill
# ---------------------------------------------------------------------------
def _make_group_callback(prefix):
    @callback(
        Output(f"{prefix}-ga", "options"),
        Output(f"{prefix}-gb", "options"),
        Input(f"{prefix}-grp", "value"),
        State("tests-ds-select", "value"),
        State(STORE_DATASET, "data"),
        State(STORE_PREPARED, "data"),
        prevent_initial_call=True,
    )
    def _update_groups(grp_col, ds_name, raw, prep):
        df = _get_df(ds_name, raw, prep)
        if df is None or not grp_col or grp_col not in df.columns:
            return [], []
        vals = df[grp_col].dropna().unique().tolist()
        opts = [{"label": str(v), "value": str(v)} for v in vals]
        return opts, opts

for _p in ["tt", "mw", "bt", "pm", "ab"]:
    _make_group_callback(_p)


# ---------------------------------------------------------------------------
# t-test callback
# ---------------------------------------------------------------------------
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
def _run_ttest(n, val_col, grp_col, ga, gb, eqvar, alpha_tick, ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        a = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        b = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        result = ttest_independent(a, b, alpha=alpha, equal_var=("eq" in (eqvar or [])),
                                   label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [rd["p_value"]]

        fig_hist = _overlay_hist(a.values, b.values, str(ga), str(gb), val_col)
        fig_ci = _ci_bars(a.values, b.values, str(ga), str(gb), val_col)

        return html.Div([
            _result_card_div(rd, alpha),
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_hist), md=7),
                dbc.Col(dcc.Graph(figure=fig_ci), md=5),
            ]),
        ]), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# Mann-Whitney callback
# ---------------------------------------------------------------------------
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
        result = mann_whitney(a, b, alpha=alpha, label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [rd["p_value"]]

        delta, delta_label = cliffs_delta(a, b)
        _CLIFF_COLORS = {"negligible": "#707a94", "small": "#3b82f6", "medium": "#f59e0b", "large": "#ef4444"}
        cliff_color = _CLIFF_COLORS.get(delta_label, "#707a94")
        cliff_card = html.Div([
            html.Strong("Cliff's δ (непараметрический размер эффекта): "),
            html.Span(f"{delta:+.4f}", style={"color": cliff_color, "fontWeight": "700", "fontSize": "1.1rem"}),
            html.Span(f"  — {delta_label}", style={"color": cliff_color, "marginLeft": "8px"}),
            html.Br(),
            html.Small("Порог: |δ| < 0.147 = negligible, 0.147–0.33 = small, 0.33–0.474 = medium, > 0.474 = large",
                       style={"color": "#707a94"}),
        ], style={
            "background": "#111318", "border": f"1px solid {cliff_color}33",
            "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "12px",
        })

        fig_hist = _overlay_hist(a.values, b.values, str(ga), str(gb), val_col)
        fig_ecdf = _ecdf_fig(a.values, b.values, str(ga), str(gb), val_col)
        fig_gauge = _cliff_gauge(delta)

        return html.Div([
            _result_card_div(rd, alpha),
            cliff_card,
            dbc.Row([
                dbc.Col(dcc.Graph(figure=fig_gauge), md=4),
                dbc.Col(dcc.Graph(figure=fig_hist), md=8),
            ]),
            dcc.Graph(figure=fig_ecdf),
        ]), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# Chi-square callback
# ---------------------------------------------------------------------------
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
        return alert_banner("Выберите две категориальные колонки.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        result = chi_square_independence(df, col_a, col_b, alpha=alpha)
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [rd["p_value"]]

        ct = pd.crosstab(df[col_a], df[col_b])
        fig = go.Figure(data=go.Heatmap(
            z=ct.values, x=[str(c) for c in ct.columns],
            y=[str(r) for r in ct.index],
            colorscale="Blues", text=ct.values, texttemplate="%{text}",
        ))
        fig.update_layout(title="Таблица сопряжённости", height=400)
        apply_kibad_theme(fig)

        return html.Div([
            _result_card_div(rd, alpha),
            dcc.Graph(figure=fig),
        ]), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# Correlation callback
# ---------------------------------------------------------------------------
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
        result = correlation_test(df[x_col].dropna(), df[y_col].dropna(),
                                  method=method or "pearson", alpha=alpha,
                                  label_x=x_col, label_y=y_col)
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [rd["p_value"]]

        sub = df[[x_col, y_col]].dropna()
        fig = px.scatter(sub, x=x_col, y=y_col, trendline="ols",
                         title=f"Корреляция ({method}): {x_col} vs {y_col}")
        apply_kibad_theme(fig)

        return html.Div([
            _result_card_div(rd, alpha),
            dcc.Graph(figure=fig),
        ]), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# Bootstrap callback
# ---------------------------------------------------------------------------
@callback(
    Output("boot-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-boot", "n_clicks"),
    State("bt-val", "value"), State("bt-grp", "value"),
    State("bt-ga", "value"), State("bt-gb", "value"),
    State("bt-n", "value"), State("bt-stat", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_boot(n, val_col, grp_col, ga, gb, n_boot, stat, alpha_tick, ds_name, raw, prep, all_pv):
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
        pv_list = (all_pv or []) + [rd["p_value"]]

        fig_hist = _overlay_hist(a.values, b.values, str(ga), str(gb), val_col)
        return html.Div([
            _result_card_div(rd, alpha),
            dcc.Graph(figure=fig_hist),
        ]), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# Permutation test callback
# ---------------------------------------------------------------------------
@callback(
    Output("perm-result", "children"),
    Output("tests-all-pvalues", "data", allow_duplicate=True),
    Input("btn-perm", "n_clicks"),
    State("pm-val", "value"), State("pm-grp", "value"),
    State("pm-ga", "value"), State("pm-gb", "value"),
    State("pm-n", "value"),
    State("tests-alpha", "value"),
    State("tests-ds-select", "value"),
    State(STORE_DATASET, "data"), State(STORE_PREPARED, "data"),
    State("tests-all-pvalues", "data"),
    prevent_initial_call=True,
)
def _run_perm(n, val_col, grp_col, ga, gb, n_perm, alpha_tick, ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        a = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        b = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        result = permutation_test(a, b, n_perm=int(n_perm or 10000),
                                  alpha=alpha, label_a=str(ga), label_b=str(gb))
        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [rd["p_value"]]

        return html.Div([
            _result_card_div(rd, alpha),
            dcc.Graph(figure=_overlay_hist(a.values, b.values, str(ga), str(gb), val_col)),
        ]), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# ANOVA / Kruskal-Wallis callback
# ---------------------------------------------------------------------------
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
        return alert_banner("Выберите числовую и группирующую колонки.", "warning"), no_update
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
        else:
            result = anova_oneway(*arrays, alpha=alpha, labels=str_labels)

        rd = _result_to_dict(result)
        pv_list = (all_pv or []) + [rd["p_value"]]

        children = [_result_card_div(rd, alpha)]

        # Box plot for group comparison
        plot_df = df[[val_col, grp_col]].dropna()
        fig_box = px.box(plot_df, x=grp_col, y=val_col, color=grp_col,
                         title=f"Распределение «{val_col}» по группам «{grp_col}»")
        apply_kibad_theme(fig_box)
        children.append(dcc.Graph(figure=fig_box))

        # Post-hoc table
        posthoc = result.details.get("posthoc", [])
        if posthoc:
            children.append(section_header("Post-hoc анализ",
                                           "Попарные сравнения групп"))
            ph_rows = []
            for ph in posthoc:
                row = {"Пара": ph["pair"], "Значимо": "Да" if ph["significant"] else "Нет"}
                if "p_adj" in ph:
                    row["p (сырое)"] = f"{ph['p_raw']:.6f}"
                    row["p (Бонферрони)"] = f"{ph['p_adj']:.6f}"
                else:
                    row["p-value"] = f"{ph['p_value']:.6f}"
                ph_rows.append(row)
            ph_df = pd.DataFrame(ph_rows)
            children.append(data_table(ph_df, "anova-posthoc-tbl", page_size=20))

        # Effect size card
        es_name = "η²" if method != "kruskal" else "ε²"
        es_val = result.effect_size
        es_color = "#10b981" if result.effect_label == "незначительный" else (
            "#3b82f6" if result.effect_label == "малый" else (
                "#f59e0b" if result.effect_label == "средний" else "#ef4444"
            )
        )
        es_card = html.Div([
            html.Strong(f"Размер эффекта ({es_name}): "),
            html.Span(f"{es_val:.4f}", style={"color": es_color, "fontWeight": "700", "fontSize": "1.1rem"}),
            html.Span(f"  — {result.effect_label}", style={"color": es_color, "marginLeft": "8px"}),
        ], style={
            "background": "#111318", "border": f"1px solid {es_color}33",
            "borderRadius": "8px", "padding": "12px 16px", "marginBottom": "12px",
        })
        children.insert(1, es_card)

        return html.Div(children), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# A/B test callback
# ---------------------------------------------------------------------------
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
def _run_ab(n, val_col, grp_col, ga, gb, alpha_tick, ds_name, raw, prep, all_pv):
    if not all([val_col, grp_col, ga, gb]):
        return alert_banner("Заполните все поля.", "warning"), no_update
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Данные не найдены.", "danger"), no_update
    alpha = (alpha_tick or 5) / 100
    try:
        control = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
        treatment = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()
        ab_res = ab_test(control, treatment, alpha=alpha,
                         label_ctrl=str(ga), label_trt=str(gb))

        children = []
        for key in ["ttest", "mann_whitney", "bootstrap"]:
            if key in ab_res and hasattr(ab_res[key], "name"):
                rd = _result_to_dict(ab_res[key])
                children.append(_result_card_div(rd, alpha))

        pv_list = all_pv or []
        if "ttest" in ab_res and hasattr(ab_res["ttest"], "p_value"):
            pv_list = pv_list + [ab_res["ttest"].p_value]

        fig_hist = _overlay_hist(control.values, treatment.values, str(ga), str(gb), val_col)
        fig_ci = _ci_bars(control.values, treatment.values, str(ga), str(gb), val_col)
        children.append(dbc.Row([
            dbc.Col(dcc.Graph(figure=fig_hist), md=7),
            dbc.Col(dcc.Graph(figure=fig_ci), md=5),
        ]))

        return html.Div(children), pv_list
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger"), no_update


# ---------------------------------------------------------------------------
# BH correction callback
# ---------------------------------------------------------------------------
@callback(
    Output("bh-result", "children"),
    Input("btn-bh", "n_clicks"),
    State("tests-all-pvalues", "data"),
    State("tests-alpha", "value"),
    prevent_initial_call=True,
)
def _run_bh(n, all_pv, alpha_tick):
    if not all_pv:
        return alert_banner("Нет p-значений. Запустите тесты на других вкладках.", "warning")
    alpha = (alpha_tick or 5) / 100
    try:
        corrections = bh_correction(all_pv, alpha=alpha)
        rows = []
        for i, c in enumerate(corrections):
            rows.append({
                "Тест #": i + 1,
                "p-value": f"{c['p_value']:.6f}",
                "Скорр. p": f"{c['adjusted_p']:.6f}",
                "Порог BH": f"{c['bh_threshold']:.6f}",
                "Значимо (BH)": "Да" if c["significant"] else "Нет",
            })
        tbl = pd.DataFrame(rows)
        n_sig = sum(1 for c in corrections if c["significant"])
        n_total = len(corrections)

        return html.Div([
            stat_card("Тестов", n_total),
            stat_card("Значимых после BH", n_sig),
            html.Div(className="mb-3"),
            data_table(tbl, "bh-table", page_size=20),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")


# ---------------------------------------------------------------------------
# Diagnostics callbacks
# ---------------------------------------------------------------------------

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
        return html.Div()
    groups = sorted(df[grp_col].dropna().unique().astype(str).tolist())
    if len(groups) < 2:
        return alert_banner("Нужно минимум 2 группы.", "warning")
    opts = [{"label": g, "value": g} for g in groups]
    return html.Div([
        html.Label("Группа A", className="kb-label"),
        dcc.Dropdown(id="diag-ga", options=opts, value=groups[0], clearable=False),
        html.Label("Группа B", className="kb-label mt-2"),
        dcc.Dropdown(id="diag-gb", options=opts, value=groups[1] if len(groups) > 1 else groups[0], clearable=False),
    ])


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
def _run_diagnostics(n, val_col, grp_col, ga, gb, alpha_tick, ds_name, raw, prep):
    if not all([val_col, grp_col, ga, gb, ds_name]):
        return alert_banner("Заполните все поля.", "warning")
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return alert_banner("Датасет не найден.", "danger")

    alpha = (alpha_tick or 5) / 100
    a_s = df[df[grp_col].astype(str) == str(ga)][val_col].dropna()
    b_s = df[df[grp_col].astype(str) == str(gb)][val_col].dropna()

    if len(a_s) < 3 or len(b_s) < 3:
        return alert_banner("В каждой группе нужно минимум 3 наблюдения.", "warning")

    try:
        diag = diagnose_groups(a_s, b_s, alpha=alpha, label_a=str(ga), label_b=str(gb))
    except Exception as e:
        return alert_banner(f"Ошибка диагностики: {e}", "danger")

    # ── Recommendation card ───────────────────────────────────────────────
    rec_colors = {
        "ttest_welch":   ("#10b981", "#0d2b1f"),
        "ttest_student": ("#10b981", "#0d2b1f"),
        "mann_whitney":  ("#3b82f6", "#0d1b2f"),
        "bootstrap":     ("#f59e0b", "#2b1f06"),
    }
    border_col, bg_col = rec_colors.get(diag["rec_test"], ("#10b981", "#0d2b1f"))

    rec_card = html.Div([
        html.Div("✅ Рекомендуемый тест", style={
            "fontSize": "0.7rem", "fontWeight": "700",
            "textTransform": "uppercase", "letterSpacing": "0.08em",
            "color": border_col, "marginBottom": "4px",
        }),
        html.Div(diag["rec_name"], style={
            "fontSize": "1.3rem", "fontWeight": "800", "color": border_col,
        }),
        html.Div(diag["rec_reason"], style={
            "fontSize": "0.88rem", "color": "#c8cad4", "marginTop": "6px",
        }),
    ], style={
        "background": bg_col, "border": f"2px solid {border_col}",
        "borderRadius": "10px", "padding": "18px 22px", "marginBottom": "20px",
    })

    # ── Normality cards per group ─────────────────────────────────────────
    def _norm_card(nr: NormalityResult, label: str, n: int) -> html.Div:
        is_norm = nr.is_normal
        border = "#10b981" if is_norm else "#ef4444"
        bg = "#0d2b1f" if is_norm else "#2b0d0d"
        icon = "✅" if is_norm else "❌"
        verdict = "нормальное" if is_norm else "ненормальное"
        return html.Div([
            html.Strong(f"{icon} {label} (n={n}) — распределение: {verdict}"),
            html.Br(),
            html.Span(
                f"Тест: {nr.test_name}  |  W={nr.statistic}  |  p={nr.p_value}  |  "
                f"Асимметрия={nr.skewness} ({nr.skew_label})  |  "
                f"Эксцесс={nr.kurtosis} ({nr.kurt_label})",
                style={"fontSize": "0.83rem", "color": "#8891a5"},
            ),
        ], style={
            "background": bg, "borderLeft": f"5px solid {border}",
            "borderRadius": "8px", "padding": "14px 18px", "marginBottom": "10px",
        })

    norm_row = dbc.Row([
        dbc.Col(_norm_card(diag["norm_a"], f"Группа A: {ga}", diag["n_a"]), md=6),
        dbc.Col(_norm_card(diag["norm_b"], f"Группа B: {gb}", diag["n_b"]), md=6),
    ], className="mb-3")

    # ── Levene test card ──────────────────────────────────────────────────
    lev = diag["levene"]
    lev_is_sig = lev.significant
    lev_border = "#ef4444" if lev_is_sig else "#10b981"
    lev_bg = "#2b0d0d" if lev_is_sig else "#0d2b1f"
    lev_icon = "❌" if lev_is_sig else "✅"
    lev_verdict = ("дисперсии различаются (гетероскедастичность)"
                   if lev_is_sig else "дисперсии однородны (гомоскедастичность)")

    levene_card = html.Div([
        html.Strong(f"{lev_icon} Тест Левена: {lev_verdict}"),
        html.Br(),
        html.Span(
            f"W={lev.statistic}  |  p={lev.p_value}",
            style={"fontSize": "0.83rem", "color": "#8891a5"},
        ),
    ], style={
        "background": lev_bg, "borderLeft": f"5px solid {lev_border}",
        "borderRadius": "8px", "padding": "14px 18px", "marginBottom": "20px",
    })

    # ── Summary message ───────────────────────────────────────────────────
    both_normal = diag["both_normal"]
    equal_var = diag["equal_var"]
    if both_normal and equal_var:
        summary = "Обе группы нормальны, дисперсии однородны → рекомендуется t-тест Стьюдента"
        s_color = "success"
    elif both_normal and not equal_var:
        summary = "Обе группы нормальны, дисперсии различаются → рекомендуется t-тест Уэлча"
        s_color = "info"
    else:
        summary = ("Хотя бы одна группа ненормальна → рекомендуется Манна-Уитни или Бутстрап. "
                   "Перейдите на соответствующую вкладку.")
        s_color = "warning"

    warnings_div = html.Div([
        dbc.Alert(w, color="warning", className="mb-1")
        for w in diag.get("warnings", [])
    ]) if diag.get("warnings") else html.Div()

    return html.Div([
        section_header("Результаты диагностики"),
        rec_card,
        section_header("Нормальность распределений"),
        norm_row,
        section_header("Однородность дисперсий (тест Левена)"),
        levene_card,
        dbc.Alert(summary, color=s_color),
        warnings_div,
    ])


# ---------------------------------------------------------------------------
# Power analysis callback
# ---------------------------------------------------------------------------
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
    prevent_initial_call=True,
)
def _run_power(n, test_type, solve_for, effect, alpha, power, n_obs, k_groups):
    from statsmodels.stats.power import TTestIndPower, TTestPower, NormalIndPower, FTestAnovaPower
    from statsmodels.stats.proportion import proportion_effectsize

    try:
        effect = float(effect or 0.5)
        alpha = float(alpha or 0.05)
        power_val = float(power or 0.8)
        n_obs = int(n_obs or 50)
        k_groups = int(k_groups or 3)

        # Decide which parameter is None (to be solved)
        kwargs = {"effect_size": effect, "alpha": alpha, "power": power_val, "nobs1": n_obs}
        if solve_for == "nobs":
            kwargs["nobs1"] = None
        elif solve_for == "power":
            kwargs["power"] = None
        elif solve_for == "effect_size":
            kwargs["effect_size"] = None

        result_val = None
        label = ""

        if test_type == "ttest2":
            analysis = TTestIndPower()
            kw = {k: v for k, v in kwargs.items()}
            kw["ratio"] = 1.0
            result_val = analysis.solve_power(**kw)
            label_map = {"nobs": "Размер выборки (n на группу)", "power": "Мощность", "effect_size": "Размер эффекта (Cohen's d)"}
        elif test_type == "ttest1":
            analysis = TTestPower()
            kw = {"effect_size": kwargs["effect_size"], "alpha": kwargs["alpha"],
                  "power": kwargs["power"], "nobs": kwargs["nobs1"]}
            if solve_for == "nobs":
                kw["nobs"] = None
            result_val = analysis.solve_power(**kw)
            label_map = {"nobs": "Размер выборки (n)", "power": "Мощность", "effect_size": "Размер эффекта (Cohen's d)"}
        elif test_type == "prop":
            analysis = NormalIndPower()
            kw = {k: v for k, v in kwargs.items()}
            kw["ratio"] = 1.0
            result_val = analysis.solve_power(**kw)
            label_map = {"nobs": "Размер выборки (n на группу)", "power": "Мощность", "effect_size": "Размер эффекта (Cohen's h)"}
        elif test_type == "anova":
            analysis = FTestAnovaPower()
            kw = {"effect_size": kwargs["effect_size"], "alpha": kwargs["alpha"],
                  "power": kwargs["power"], "nobs": kwargs["nobs1"], "k_groups": k_groups}
            if solve_for == "nobs":
                kw["nobs"] = None
            result_val = analysis.solve_power(**kw)
            label_map = {"nobs": "Размер выборки (n на группу)", "power": "Мощность", "effect_size": "Размер эффекта (Cohen's f)"}

        if result_val is None:
            return alert_banner("Не удалось вычислить результат.", "danger")

        solve_label = label_map.get(solve_for, solve_for)
        if solve_for == "nobs":
            display_val = f"{int(np.ceil(result_val))}"
            unit = "наблюдений"
        elif solve_for == "power":
            display_val = f"{result_val:.1%}"
            unit = ""
        else:
            display_val = f"{result_val:.4f}"
            unit = ""

        # Interpretation
        if solve_for == "power":
            pwr = result_val
        else:
            pwr = power_val
        if pwr >= 0.8:
            pwr_color, pwr_verdict = "#10b981", "Достаточная мощность (≥ 80%)"
        elif pwr >= 0.6:
            pwr_color, pwr_verdict = "#f59e0b", "Умеренная мощность (60–80%)"
        else:
            pwr_color, pwr_verdict = "#ef4444", "Недостаточная мощность (< 60%) — высок риск ошибки II рода"

        # Power curve: n vs power
        ns = np.arange(5, max(int(n_obs * 3), 200), max(1, int(n_obs * 3) // 100))
        if test_type == "ttest2":
            curve_power = [TTestIndPower().solve_power(
                effect_size=effect, alpha=alpha, nobs1=ni, ratio=1.0) for ni in ns]
        elif test_type == "ttest1":
            curve_power = [TTestPower().solve_power(
                effect_size=effect, alpha=alpha, nobs=ni) for ni in ns]
        elif test_type == "prop":
            curve_power = [NormalIndPower().solve_power(
                effect_size=effect, alpha=alpha, nobs1=ni, ratio=1.0) for ni in ns]
        else:
            curve_power = [FTestAnovaPower().solve_power(
                effect_size=effect, alpha=alpha, nobs=ni, k_groups=k_groups) for ni in ns]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ns, y=curve_power, mode="lines", name="Мощность",
            line=dict(color="#10b981", width=2),
            hovertemplate="n=%{x}<br>Мощность=%{y:.1%}<extra></extra>",
        ))
        fig.add_hline(y=0.8, line_dash="dash", line_color="#f59e0b",
                      annotation_text="80% порог", annotation_font_color="#f59e0b")
        fig.add_hline(y=0.95, line_dash="dot", line_color="#ef4444",
                      annotation_text="95% порог", annotation_font_color="#ef4444")
        if solve_for == "nobs":
            n_needed = int(np.ceil(result_val))
            fig.add_shape(type="line", x0=n_needed, x1=n_needed, y0=0, y1=1, yref="paper",
                          line=dict(dash="dash", color="#3b82f6", width=1.5))
            fig.add_annotation(x=n_needed, y=1, yref="paper", text=f"n={n_needed}",
                               showarrow=False, font=dict(size=11, color="#3b82f6"),
                               xanchor="left", yanchor="top")
        fig.update_layout(
            title="Кривая мощности: n → мощность теста",
            xaxis_title="Размер выборки (n)",
            yaxis_title="Мощность (1−β)",
            yaxis=dict(tickformat=".0%", range=[0, 1.05]),
            height=380,
        )
        apply_kibad_theme(fig)

        return html.Div([
            html.Div([
                html.Div(solve_label, style={
                    "fontSize": "0.65rem", "fontWeight": "700",
                    "textTransform": "uppercase", "letterSpacing": "0.1em",
                    "color": pwr_color, "marginBottom": "4px",
                }),
                html.Div(f"{display_val} {unit}".strip(), style={
                    "fontSize": "2.2rem", "fontWeight": "800", "color": pwr_color, "lineHeight": "1.1",
                }),
                html.Div(pwr_verdict, style={"fontSize": "0.85rem", "color": "#b0bccf", "marginTop": "6px"}),
            ], style={
                "background": "#111318", "border": f"2px solid {pwr_color}",
                "borderRadius": "10px", "padding": "20px 28px", "marginBottom": "20px",
                "maxWidth": "480px",
            }),
            html.Div([
                html.Strong("Параметры расчёта: "),
                html.Span(f"effect={effect:.3f}  alpha={alpha:.3f}  power={power_val:.0%}  n={n_obs}",
                          style={"color": "#707a94", "fontSize": "0.85rem"}),
            ], className="mb-3"),
            dcc.Graph(figure=fig),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка анализа мощности: {e}", "danger")


# ---------------------------------------------------------------------------
# Compute effect size from data
# ---------------------------------------------------------------------------
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
            groups = {k: v.dropna().values for k, v in df.groupby(col_grp)[col_num]}
            group_names = list(groups.keys())
            arrays = [groups[k] for k in group_names]
            if test_type == "ttest2":
                if len(arrays) < 2:
                    return no_update, alert_banner("Нужно минимум 2 группы.", "warning")
                a1, a2 = arrays[0], arrays[1]
                n1, n2 = len(a1), len(a2)
                pooled_std = np.sqrt(((n1 - 1) * a1.std() ** 2 + (n2 - 1) * a2.std() ** 2) / (n1 + n2 - 2))
                d = abs(a1.mean() - a2.mean()) / pooled_std if pooled_std > 0 else 0.0
                hint = (f"Cohen's d = {d:.4f}  "
                        f"({group_names[0]}: μ={a1.mean():.3f}, n={n1}  "
                        f"vs  {group_names[1]}: μ={a2.mean():.3f}, n={n2})")
                return round(float(d), 4), dbc.Alert(f"✓ {hint}", color="success",
                                                     className="mt-1 py-1 px-3")
            else:  # anova
                grand_mean = series.mean()
                ss_between = sum(len(a) * (a.mean() - grand_mean) ** 2 for a in arrays)
                ss_within = sum(((a - a.mean()) ** 2).sum() for a in arrays)
                total = ss_between + ss_within
                eta_sq = ss_between / total if total > 0 else 0.0
                f_cohen = np.sqrt(eta_sq / (1 - eta_sq)) if 0 < eta_sq < 1 else 0.0
                hint = (f"Cohen's f = {f_cohen:.4f}  "
                        f"(η² = {eta_sq:.4f}, {len(arrays)} групп)")
                return round(float(f_cohen), 4), dbc.Alert(f"✓ {hint}", color="success",
                                                           className="mt-1 py-1 px-3")
        else:
            # one-sample: effect vs 0
            mu, sigma = series.mean(), series.std()
            d = abs(mu) / sigma if sigma > 0 else 0.0
            hint = (f"Cohen's d = {d:.4f}  "
                    f"(μ={mu:.4f}, σ={sigma:.4f}, n={len(series)})")
            return round(float(d), 4), dbc.Alert(f"✓ {hint}", color="success",
                                                 className="mt-1 py-1 px-3")
    except Exception as e:
        return no_update, alert_banner(f"Ошибка: {e}", "danger")

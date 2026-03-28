"""
p06_tests.py -- Статистические тесты (Dash).

t-тест, Манна-Уитни, хи-квадрат, корреляция, бутстрап, перестановочный,
A/B-тест, пакетное тестирование с поправкой BH.
"""
from __future__ import annotations

import json
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html, no_update
import dash_bootstrap_components as dbc

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
        ], style={"fontSize": "0.85rem", "color": "#8891a5"}),
    ]
    if r.get("ci"):
        children.append(
            html.Div(f"95% ДИ: [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]",
                      style={"fontSize": "0.85rem", "color": "#8891a5", "marginTop": "4px"})
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
    dbc.Tabs(id="tests-tabs", active_tab="tab-ttest", children=[
        dbc.Tab(label="t-Тест", tab_id="tab-ttest"),
        dbc.Tab(label="Манна-Уитни", tab_id="tab-mw"),
        dbc.Tab(label="Хи-квадрат", tab_id="tab-chi2"),
        dbc.Tab(label="Корреляция", tab_id="tab-corr"),
        dbc.Tab(label="Бутстрап", tab_id="tab-boot"),
        dbc.Tab(label="Перестановочный", tab_id="tab-perm"),
        dbc.Tab(label="A/B тест", tab_id="tab-ab"),
        dbc.Tab(label="Мн. сравнения (BH)", tab_id="tab-bh"),
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
    Input(STORE_DATASET, "data"),
    Input(STORE_PREPARED, "data"),
)
def _update_ds_options(raw, prep):
    names = sorted(set(list_datasets(raw) + list_datasets(prep)))
    return [{"label": n, "value": n} for n in names]


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
    df = _get_df(ds_name, raw, prep)
    if df is None:
        return empty_state("bi-clipboard-data", "Нет данных",
                           "Выберите датасет в выпадающем списке выше.")

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    if tab == "tab-ttest":
        return html.Div([
            section_header("t-тест (Уэлча / Стьюдента)",
                           "H0: Средние двух групп равны."),
            *_two_group_controls("tt", num_cols, cat_cols),
            dbc.Checklist(
                id="tt-eqvar",
                options=[{"label": " Предполагать равные дисперсии (Стьюдент)", "value": "eq"}],
                value=[], inline=True,
                className="mb-3", style={"color": "#8891a5"},
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
                                      {"label": "Спирмен", "value": "spearman"}],
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
                   "для пакетной коррекции.", style={"color": "#8891a5"}),
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

        fig_hist = _overlay_hist(a.values, b.values, str(ga), str(gb), val_col)

        return html.Div([
            _result_card_div(rd, alpha),
            dcc.Graph(figure=fig_hist),
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

        return html.Div([
            _result_card_div(rd, alpha),
            _overlay_hist(a.values, b.values, str(ga), str(gb), val_col).__class__.__name__
            and dcc.Graph(figure=_overlay_hist(a.values, b.values, str(ga), str(gb), val_col)),
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
                "Порог BH": f"{c['threshold']:.6f}",
                "Значимо (BH)": "Да" if c["significant_bh"] else "Нет",
            })
        tbl = pd.DataFrame(rows)
        n_sig = sum(1 for c in corrections if c["significant_bh"])
        n_total = len(corrections)

        return html.Div([
            stat_card("Тестов", n_total),
            stat_card("Значимых после BH", n_sig),
            html.Div(className="mb-3"),
            data_table(tbl, "bh-table", page_size=20),
        ])
    except Exception as e:
        return alert_banner(f"Ошибка: {e}", "danger")

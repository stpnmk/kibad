"""
17. Сопоставление групп — Propensity Score Matching, Exact, NN, CEM.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.state import (
    dataset_selectbox,
    get_active_df,
    init_state,
    store_prepared,
)
from app.styles import inject_all_css, page_header, section_header
from app.components.ux import (
    interpretation_box,
    kpi_cards_row,
    method_card,
    no_data_placeholder,
    step_progress,
)
from core.audit import log_event
from core.matching import (
    MatchResult,
    balance_summary,
    coarsened_exact_match,
    exact_match,
    nearest_neighbor_match,
    propensity_score_match,
    standardized_mean_diff,
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – Сопоставление", layout="wide")
init_state()
inject_all_css()
page_header(
    "17. Сопоставление групп",
    "Подбор сопоставимых групп для корректного сравнения",
    "🔀",
)

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------

chosen = dataset_selectbox("Датасет", key="match_ds")
if not chosen:
    no_data_placeholder()
    st.stop()

df = get_active_df()
if df is None or df.empty:
    no_data_placeholder()
    st.stop()

st.caption(f"📐 {df.shape[0]:,} строк × {df.shape[1]} колонок")

# ---------------------------------------------------------------------------
# Step progress
# ---------------------------------------------------------------------------

step_progress(
    ["Настройка", "Сопоставление", "Диагностика"],
    current=0 if "_match_result" not in st.session_state else 2,
)

st.divider()

# ---------------------------------------------------------------------------
# Step 1 — Configuration
# ---------------------------------------------------------------------------

section_header("Настройка сопоставления", "⚙️")

col_left, col_right = st.columns([1, 1])

with col_left:
    # Treatment column — binary group indicator
    all_cols = df.columns.tolist()
    treatment_col = st.selectbox(
        "Колонка группы (бинарная: 0/1 или 2 уникальных значения)",
        options=[""] + all_cols,
        key="match_treatment",
        help="Колонка, определяющая принадлежность к группе (тест/контроль, опытная/контрольная)",
    )

    if treatment_col:
        unique_vals = df[treatment_col].dropna().unique()
        if len(unique_vals) == 2:
            # Map to 0/1
            val_map = {unique_vals[0]: 0, unique_vals[1]: 1}
            st.info(
                f"Группы: **{unique_vals[0]}** → контроль (0), "
                f"**{unique_vals[1]}** → опытная (1)"
            )
            # Let user swap if needed
            swap = st.checkbox("Поменять группы местами", key="match_swap")
            if swap:
                val_map = {unique_vals[0]: 1, unique_vals[1]: 0}
                st.info(
                    f"Группы: **{unique_vals[0]}** → опытная (1), "
                    f"**{unique_vals[1]}** → контроль (0)"
                )
        elif len(unique_vals) > 2:
            st.error(
                f"Колонка **{treatment_col}** содержит {len(unique_vals)} уникальных значений. "
                "Выберите колонку ровно с 2 уникальными значениями."
            )
            st.stop()
        else:
            st.error("Колонка содержит менее 2 уникальных значений.")
            st.stop()
    else:
        st.warning("Выберите колонку группы для продолжения.")
        st.stop()

with col_right:
    # Covariate selection
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    available_covariates = [c for c in numeric_cols + cat_cols if c != treatment_col]

    covariates = st.multiselect(
        "Ковариаты для сопоставления",
        options=available_covariates,
        default=[c for c in numeric_cols if c != treatment_col][:6],
        key="match_covariates",
        help="Признаки, по которым нужно обеспечить сопоставимость групп",
    )

    if not covariates:
        st.warning("Выберите хотя бы одну ковариату.")
        st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Pre-matching balance preview
# ---------------------------------------------------------------------------

section_header("Баланс до сопоставления", "📊")

df_work = df.copy()
df_work[treatment_col] = df_work[treatment_col].map(val_map)

# Show only numeric covariates in balance table
num_covariates = [c for c in covariates if pd.api.types.is_numeric_dtype(df_work[c])]

if num_covariates:
    balance_pre = standardized_mean_diff(df_work, treatment_col, num_covariates)

    n_t = int((df_work[treatment_col] == 1).sum())
    n_c = int((df_work[treatment_col] == 0).sum())
    summary = balance_summary(balance_pre)

    kpi_cards_row(
        {
            "Опытная группа": (f"{n_t:,}", None),
            "Контрольная группа": (f"{n_c:,}", None),
            "Средний |SMD|": (f"{summary['mean_abs_smd']:.3f}", None),
            "Ковариат с |SMD| < 0.1": (f"{summary['pct_below_01']:.0f}%", None),
        }
    )

    # Quick balance bar chart
    fig_pre = go.Figure()
    fig_pre.add_trace(
        go.Bar(
            y=balance_pre["covariate"],
            x=balance_pre["abs_smd"],
            orientation="h",
            marker_color=[
                "#e74c3c" if v > 0.25 else "#f39c12" if v > 0.1 else "#27ae60"
                for v in balance_pre["abs_smd"]
            ],
            text=[f"{v:.3f}" for v in balance_pre["abs_smd"]],
            textposition="outside",
        )
    )
    fig_pre.add_vline(x=0.1, line_dash="dash", line_color="green", annotation_text="0.1")
    fig_pre.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="0.25")
    fig_pre.update_layout(
        title="|SMD| до сопоставления",
        xaxis_title="|SMD|",
        yaxis_title="",
        template="plotly_white",
        height=max(300, len(num_covariates) * 35 + 100),
        margin=dict(l=10, r=10, t=40, b=30),
    )
    st.plotly_chart(fig_pre, use_container_width=True)

    if summary["mean_abs_smd"] < 0.1:
        interpretation_box(
            "Группы уже сбалансированы",
            "Средний |SMD| < 0.1 — группы уже достаточно похожи. "
            "Сопоставление может не требоваться.",
            icon="✅",
        )
    elif summary["mean_abs_smd"] > 0.25:
        interpretation_box(
            "Значительный дисбаланс",
            "Средний |SMD| > 0.25 — группы существенно различаются. "
            "Рекомендуется сопоставление для корректного сравнения.",
            icon="⚠️",
        )
else:
    st.info("Нет числовых ковариат для предварительного анализа баланса.")

st.divider()

# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------

section_header("Метод сопоставления", "🔧")

method_tabs = st.tabs([
    "🎯 PSM (Propensity Score)",
    "🔗 Точное сопоставление",
    "📐 Ближайший сосед",
    "📦 CEM (огрубление)",
])

method_choice = None
method_params: dict = {}

with method_tabs[0]:
    method_card(
        "Propensity Score Matching",
        "Оценка склонности через логистическую регрессию. "
        "Каждому объекту опытной группы подбирается ближайший контроль "
        "по вероятности попадания в опытную группу.",
        "Много числовых ковариат, нужно учесть все сразу.",
        "Числовые ковариаты, бинарная группа.",
        icon="🎯",
    )
    col1, col2 = st.columns(2)
    with col1:
        psm_caliper = st.slider(
            "Caliper (в σ propensity score)",
            0.01, 1.0, 0.2, 0.01,
            key="psm_caliper",
            help="Максимальное допустимое расстояние. Меньше = строже отбор, меньше пар.",
        )
    with col2:
        psm_ratio = st.selectbox(
            "Соотношение (контроль : опытная)",
            [1, 2, 3, 5],
            key="psm_ratio",
            help="Сколько контрольных объектов подбирать на каждый опытный.",
        )
    if st.button("▶ Запустить PSM", type="primary", key="btn_psm"):
        method_choice = "psm"
        method_params = {"caliper": psm_caliper, "ratio": psm_ratio}

with method_tabs[1]:
    method_card(
        "Точное сопоставление (Exact Matching)",
        "Группы формируются из объектов с полным совпадением "
        "по выбранным категориальным признакам.",
        "Мало категориальных признаков с небольшим числом категорий.",
        "Категориальные ковариаты.",
        icon="🔗",
    )
    exact_options = [c for c in covariates if c in cat_cols or df_work[c].nunique() <= 10]
    if exact_options:
        exact_selected = st.multiselect(
            "Признаки для точного сопоставления",
            exact_options,
            default=exact_options[:3],
            key="exact_cols",
        )
        if st.button("▶ Запустить Exact Matching", type="primary", key="btn_exact"):
            if exact_selected:
                method_choice = "exact"
                method_params = {"exact_cols": exact_selected}
            else:
                st.error("Выберите хотя бы один признак.")
    else:
        st.warning("Нет подходящих категориальных признаков (≤ 10 уникальных значений).")

with method_tabs[2]:
    method_card(
        "Ближайший сосед (Nearest Neighbor)",
        "Для каждого объекта опытной группы находится ближайший по расстоянию "
        "объект из контрольной группы (Махаланобис или евклидово расстояние).",
        "Числовые ковариаты, нужна точность в каждой паре.",
        "Числовые ковариаты.",
        icon="📐",
    )
    col1, col2 = st.columns(2)
    with col1:
        nn_k = st.selectbox("Число соседей", [1, 2, 3, 5], key="nn_k")
    with col2:
        nn_metric = st.radio(
            "Метрика расстояния",
            ["mahalanobis", "euclidean"],
            format_func=lambda x: "Махаланобис" if x == "mahalanobis" else "Евклидово",
            key="nn_metric",
        )
    if st.button("▶ Запустить NN Matching", type="primary", key="btn_nn"):
        method_choice = "nn"
        method_params = {"n_neighbors": nn_k, "metric": nn_metric}

with method_tabs[3]:
    method_card(
        "Coarsened Exact Matching (CEM)",
        "Числовые признаки огрубляются в квантильные бины, затем "
        "выполняется точное сопоставление по бинам. "
        "Сохраняет веса для взвешенного анализа.",
        "Смесь числовых и категориальных признаков.",
        "Любые ковариаты.",
        icon="📦",
    )
    cem_bins = st.slider(
        "Число квантильных бинов",
        2, 20, 5, 1,
        key="cem_bins",
        help="Больше бинов = точнее сопоставление, но меньше совпадений.",
    )
    if st.button("▶ Запустить CEM", type="primary", key="btn_cem"):
        method_choice = "cem"
        method_params = {"n_bins": cem_bins}


# ---------------------------------------------------------------------------
# Run matching
# ---------------------------------------------------------------------------

if method_choice:
    with st.spinner("Выполняется сопоставление…"):
        try:
            # Prepare data — encode treatment to 0/1
            df_encoded = df.copy()
            df_encoded[treatment_col] = df_encoded[treatment_col].map(val_map)

            # Filter to numeric covariates for methods that require them
            num_cov = [c for c in covariates if pd.api.types.is_numeric_dtype(df_encoded[c])]

            if method_choice == "psm":
                if not num_cov:
                    st.error("PSM требует хотя бы одну числовую ковариату.")
                    st.stop()
                result = propensity_score_match(
                    df_encoded, treatment_col, num_cov,
                    caliper=method_params["caliper"],
                    ratio=method_params["ratio"],
                )
            elif method_choice == "exact":
                result = exact_match(
                    df_encoded, treatment_col,
                    method_params["exact_cols"],
                    covariates=num_cov if num_cov else method_params["exact_cols"],
                )
            elif method_choice == "nn":
                if not num_cov:
                    st.error("NN Matching требует хотя бы одну числовую ковариату.")
                    st.stop()
                result = nearest_neighbor_match(
                    df_encoded, treatment_col, num_cov,
                    n_neighbors=method_params["n_neighbors"],
                    metric=method_params["metric"],
                )
            elif method_choice == "cem":
                result = coarsened_exact_match(
                    df_encoded, treatment_col, num_cov if num_cov else covariates,
                    n_bins=method_params["n_bins"],
                )
            else:
                st.error("Неизвестный метод.")
                st.stop()

            st.session_state["_match_result"] = result
            log_event("matching_run", {
                "method": method_choice,
                "n_covariates": len(covariates),
                "n_matched_t": result.n_matched_treatment,
                "n_matched_c": result.n_matched_control,
            })
            st.rerun()

        except Exception as e:
            st.error(f"Ошибка сопоставления: {e}")
            st.stop()


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if "_match_result" not in st.session_state:
    st.info("👆 Выберите метод и нажмите кнопку для запуска сопоставления.")
    st.stop()

result: MatchResult = st.session_state["_match_result"]

st.divider()
section_header("Результаты сопоставления", "📈")

# KPI row
match_rate_t = (
    result.n_matched_treatment / result.n_treatment * 100
    if result.n_treatment > 0 else 0
)
match_rate_c = (
    result.n_matched_control / result.n_control * 100
    if result.n_control > 0 else 0
)
summary_after = balance_summary(result.balance_after)
summary_before = balance_summary(result.balance_before)
smd_delta = summary_after["mean_abs_smd"] - summary_before["mean_abs_smd"]

kpi_cards_row(
    {
        "Метод": (result.method, None),
        "Опытная сопоставлена": (f"{result.n_matched_treatment:,} ({match_rate_t:.0f}%)", None),
        "Контроль сопоставлен": (f"{result.n_matched_control:,} ({match_rate_c:.0f}%)", None),
        "Средний |SMD| после": (
            f"{summary_after['mean_abs_smd']:.3f}",
            f"{smd_delta:+.3f}",
        ),
    }
)

# ---------------------------------------------------------------------------
# Tabs: Balance / Distributions / Love Plot / Data / Export
# ---------------------------------------------------------------------------

tab_balance, tab_love, tab_dist, tab_ps, tab_data = st.tabs([
    "📊 Баланс",
    "❤️ Love-plot",
    "📉 Распределения",
    "🎯 Propensity Score",
    "📋 Данные и экспорт",
])

# --- Tab: Balance table ---
with tab_balance:
    section_header("Таблица баланса до и после сопоставления", "📊")

    bal_before = result.balance_before.copy()
    bal_after = result.balance_after.copy()
    bal_before = bal_before.rename(columns={
        "mean_treatment": "Среднее (опыт) до",
        "mean_control": "Среднее (контроль) до",
        "smd": "SMD до",
        "abs_smd": "|SMD| до",
        "variance_ratio": "Var ratio до",
    })
    bal_after = bal_after.rename(columns={
        "mean_treatment": "Среднее (опыт) после",
        "mean_control": "Среднее (контроль) после",
        "smd": "SMD после",
        "abs_smd": "|SMD| после",
        "variance_ratio": "Var ratio после",
    })

    merged = bal_before.merge(
        bal_after[["covariate", "Среднее (опыт) после", "Среднее (контроль) после",
                   "SMD после", "|SMD| после", "Var ratio после"]],
        on="covariate",
        how="left",
    )
    merged["Δ |SMD|"] = merged["|SMD| после"] - merged["|SMD| до"]
    merged = merged.rename(columns={"covariate": "Ковариата"})

    def _color_smd(val):
        if pd.isna(val):
            return ""
        v = abs(val)
        if v < 0.1:
            return "background-color: #d4edda"
        if v < 0.25:
            return "background-color: #fff3cd"
        return "background-color: #f8d7da"

    styled = merged.style.map(
        _color_smd, subset=["|SMD| до", "|SMD| после"]
    ).format(precision=4)

    st.dataframe(styled, use_container_width=True, height=min(600, len(merged) * 40 + 60))

    # Interpretation
    improved = (merged["Δ |SMD|"] < 0).sum()
    total = len(merged)
    interpretation_box(
        "Интерпретация баланса",
        f"Баланс улучшился по **{improved} из {total}** ковариат.\n\n"
        f"- Средний |SMD| до: **{summary_before['mean_abs_smd']:.3f}**, "
        f"после: **{summary_after['mean_abs_smd']:.3f}**\n"
        f"- Ковариат с |SMD| < 0.1: до **{summary_before['pct_below_01']:.0f}%**, "
        f"после **{summary_after['pct_below_01']:.0f}%**\n\n"
        "✅ |SMD| < 0.1 — хороший баланс\n"
        "⚠️ 0.1 ≤ |SMD| < 0.25 — допустимо\n"
        "🔴 |SMD| ≥ 0.25 — значительный дисбаланс",
    )


# --- Tab: Love plot ---
with tab_love:
    section_header("Love-plot: SMD до и после", "❤️")
    st.caption(
        "Love-plot показывает |SMD| до (серые точки) и после (цветные точки) сопоставления. "
        "Цель — чтобы все цветные точки оказались левее порога 0.1."
    )

    bal_b = result.balance_before.sort_values("abs_smd", ascending=True)
    bal_a = result.balance_after.sort_values("abs_smd", ascending=True)

    fig_love = go.Figure()

    # Before (gray)
    fig_love.add_trace(
        go.Scatter(
            x=bal_b["abs_smd"],
            y=bal_b["covariate"],
            mode="markers",
            marker=dict(size=10, color="#bdc3c7", symbol="circle-open", line=dict(width=2)),
            name="До сопоставления",
        )
    )

    # After (colored)
    colors = [
        "#27ae60" if v < 0.1 else "#f39c12" if v < 0.25 else "#e74c3c"
        for v in bal_a["abs_smd"]
    ]
    fig_love.add_trace(
        go.Scatter(
            x=bal_a["abs_smd"],
            y=bal_a["covariate"],
            mode="markers",
            marker=dict(size=12, color=colors, symbol="circle"),
            name="После сопоставления",
        )
    )

    # Connecting lines
    for _, row_b in bal_b.iterrows():
        cov = row_b["covariate"]
        row_a = bal_a[bal_a["covariate"] == cov]
        if not row_a.empty:
            fig_love.add_trace(
                go.Scatter(
                    x=[row_b["abs_smd"], row_a.iloc[0]["abs_smd"]],
                    y=[cov, cov],
                    mode="lines",
                    line=dict(color="#95a5a6", width=1),
                    showlegend=False,
                )
            )

    fig_love.add_vline(x=0.1, line_dash="dash", line_color="green",
                       annotation_text="Хороший баланс (0.1)")
    fig_love.add_vline(x=0.25, line_dash="dash", line_color="red",
                       annotation_text="Порог (0.25)")

    fig_love.update_layout(
        template="plotly_white",
        height=max(400, len(bal_b) * 35 + 120),
        margin=dict(l=10, r=10, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_love, use_container_width=True)


# --- Tab: Distributions ---
with tab_dist:
    section_header("Распределения ковариат до и после сопоставления", "📉")

    dist_cov = st.selectbox(
        "Выберите ковариату",
        options=num_covariates if num_covariates else result.covariates,
        key="dist_cov_select",
    )

    if dist_cov and pd.api.types.is_numeric_dtype(df_work[dist_cov]):
        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown("**До сопоставления**")
            fig_b = go.Figure()
            vals_t = df_work.loc[df_work[treatment_col] == 1, dist_cov].dropna()
            vals_c = df_work.loc[df_work[treatment_col] == 0, dist_cov].dropna()
            fig_b.add_trace(go.Histogram(
                x=vals_t, name="Опытная", opacity=0.6,
                marker_color="#3498db", nbinsx=30,
            ))
            fig_b.add_trace(go.Histogram(
                x=vals_c, name="Контроль", opacity=0.6,
                marker_color="#e74c3c", nbinsx=30,
            ))
            fig_b.update_layout(
                barmode="overlay", template="plotly_white",
                height=350, margin=dict(l=10, r=10, t=30, b=30),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_b, use_container_width=True)

        with col_after:
            st.markdown("**После сопоставления**")
            mdf = result.matched_df
            fig_a = go.Figure()
            vals_t_m = mdf.loc[mdf[result.treatment_col] == 1, dist_cov].dropna()
            vals_c_m = mdf.loc[mdf[result.treatment_col] == 0, dist_cov].dropna()
            fig_a.add_trace(go.Histogram(
                x=vals_t_m, name="Опытная", opacity=0.6,
                marker_color="#3498db", nbinsx=30,
            ))
            fig_a.add_trace(go.Histogram(
                x=vals_c_m, name="Контроль", opacity=0.6,
                marker_color="#e74c3c", nbinsx=30,
            ))
            fig_a.update_layout(
                barmode="overlay", template="plotly_white",
                height=350, margin=dict(l=10, r=10, t=30, b=30),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_a, use_container_width=True)

        # QQ-plot style comparison
        st.markdown("**Квантиль-квантиль (QQ) — опытная vs контроль (после)**")
        if len(vals_t_m) > 2 and len(vals_c_m) > 2:
            quantiles = np.linspace(0, 1, min(100, min(len(vals_t_m), len(vals_c_m))))
            q_t = np.quantile(vals_t_m, quantiles)
            q_c = np.quantile(vals_c_m, quantiles)

            fig_qq = go.Figure()
            fig_qq.add_trace(go.Scatter(
                x=q_c, y=q_t, mode="markers",
                marker=dict(size=5, color="#3498db"),
                name="Квантили",
            ))
            rng_min = min(q_c.min(), q_t.min())
            rng_max = max(q_c.max(), q_t.max())
            fig_qq.add_trace(go.Scatter(
                x=[rng_min, rng_max], y=[rng_min, rng_max],
                mode="lines", line=dict(color="#e74c3c", dash="dash"),
                name="Идеальное совпадение",
            ))
            fig_qq.update_layout(
                xaxis_title="Контроль (квантили)",
                yaxis_title="Опытная (квантили)",
                template="plotly_white",
                height=400,
                margin=dict(l=10, r=10, t=30, b=30),
            )
            st.plotly_chart(fig_qq, use_container_width=True)
    else:
        st.info("Выберите числовую ковариату для просмотра распределений.")


# --- Tab: Propensity Score diagnostics ---
with tab_ps:
    section_header("Диагностика Propensity Score", "🎯")

    if result.propensity_scores is not None and result.method == "PSM":
        work_ps = df_work.dropna(subset=[treatment_col] + result.covariates).copy()
        work_ps["_ps"] = result.propensity_scores[:len(work_ps)]

        # Distribution of PS by group
        fig_ps = go.Figure()
        ps_t = work_ps.loc[work_ps[treatment_col] == 1, "_ps"]
        ps_c = work_ps.loc[work_ps[treatment_col] == 0, "_ps"]

        fig_ps.add_trace(go.Histogram(
            x=ps_t, name="Опытная", opacity=0.6,
            marker_color="#3498db", nbinsx=40,
        ))
        fig_ps.add_trace(go.Histogram(
            x=ps_c, name="Контроль", opacity=0.6,
            marker_color="#e74c3c", nbinsx=40,
        ))
        fig_ps.update_layout(
            barmode="overlay",
            title="Распределение propensity score по группам",
            xaxis_title="Propensity Score",
            template="plotly_white",
            height=400,
            margin=dict(l=10, r=10, t=40, b=30),
            legend=dict(orientation="h"),
        )

        if result.common_support:
            fig_ps.add_vrect(
                x0=result.common_support[0], x1=result.common_support[1],
                fillcolor="green", opacity=0.08,
                annotation_text="Common support",
            )

        st.plotly_chart(fig_ps, use_container_width=True)

        # Quality metrics
        quality = result.match_quality
        interpretation_box(
            "Параметры PSM",
            f"- **Caliper**: {quality.get('caliper', '—')} σ "
            f"(абс. = {quality.get('caliper_abs', '—')})\n"
            f"- **Common support**: [{quality.get('common_support', ('—', '—'))[0]}, "
            f"{quality.get('common_support', ('—', '—'))[1]}]\n"
            f"- **Обрезано (common support)**: {quality.get('n_trimmed_common_support', '—')} наблюдений\n"
            f"- **Соотношение**: 1:{quality.get('ratio', 1)}",
        )
    else:
        st.info(
            "Диагностика propensity score доступна только для метода PSM. "
            "Текущий метод: **" + result.method + "**."
        )


# --- Tab: Data & export ---
with tab_data:
    section_header("Сопоставленные данные", "📋")

    mdf = result.matched_df
    st.caption(f"Сопоставлено: {len(mdf):,} строк")

    st.dataframe(mdf.head(100), use_container_width=True, height=400)

    col_save, col_csv, col_xlsx = st.columns(3)

    with col_save:
        save_name = st.text_input(
            "Сохранить как датасет",
            value=f"{chosen}_matched",
            key="match_save_name",
        )
        if st.button("💾 Сохранить в KIBAD", type="primary", key="btn_save_matched"):
            store_prepared(save_name, mdf)
            st.success(f"Датасет **{save_name}** сохранён ({len(mdf):,} строк).")
            log_event("matching_save", {"name": save_name, "rows": len(mdf)})

    with col_csv:
        csv_buf = io.StringIO()
        mdf.to_csv(csv_buf, index=False)
        st.download_button(
            "📥 Скачать CSV",
            data=csv_buf.getvalue(),
            file_name=f"{chosen}_matched.csv",
            mime="text/csv",
            key="dl_csv_match",
        )

    with col_xlsx:
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            mdf.to_excel(writer, index=False, sheet_name="Matched")
            result.balance_after.to_excel(writer, index=False, sheet_name="Balance")
        st.download_button(
            "📥 Скачать Excel",
            data=xlsx_buf.getvalue(),
            file_name=f"{chosen}_matched.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="dl_xlsx_match",
        )

    # Reset button
    st.divider()
    if st.button("🔄 Сбросить результаты и начать заново", key="btn_reset_match"):
        st.session_state.pop("_match_result", None)
        st.rerun()

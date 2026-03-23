"""
pages/17_WeightedAvg.py – Weighted portfolio metrics page for KIBAD.

Weighted averages, portfolio summary treemap, mix-rate decomposition,
and simplified duration analysis.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df, get_dataset, list_dataset_names
from core.audit import log_event
from app.components.ux import interpretation_box
from app.styles import inject_all_css, page_header, section_header
from core.weighted_avg import (
    weighted_average,
    portfolio_weighted_averages,
    mix_rate_decomposition,
    simplified_duration,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – Взвешенные метрики", layout="wide")
init_state()
inject_all_css()

page_header("17. Взвешенные метрики", "Средневзвешенные показатели с декомпозицией", "⚖️")

COLORS = px.colors.qualitative.Plotly

# Keyword hints for auto-detection of metric columns
METRIC_KEYWORDS = ("rate", "ltv", "maturity", "score", "coupon", "yield", "ставк", "дох", "зрел")


# ---------------------------------------------------------------------------
# Dataset & Settings
# ---------------------------------------------------------------------------

chosen = dataset_selectbox("Датасет", key="wa_ds_sel")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.error("Нет данных. Сначала загрузите датасет.")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if not num_cols:
    st.error("Датасет не содержит числовых колонок.")
    st.stop()

# Auto-detect metric columns
auto_metrics = [
    c for c in num_cols
    if any(kw in c.lower() for kw in METRIC_KEYWORDS) and c != (st.session_state.get("wa_weight_col") or num_cols[0])
] or [c for c in num_cols if c != (st.session_state.get("wa_weight_col") or num_cols[0])][:4]

with st.expander("⚙️ Настройки", expanded=True):
    col_w1, col_w2, col_w3 = st.columns(3)
    with col_w1:
        weight_col = st.selectbox(
            "Вес (Balance / EAD / Объём)",
            options=num_cols,
            key="wa_weight_col",
        )
        metric_cols = st.multiselect(
            "Метрики для взвешивания",
            options=[c for c in num_cols if c != weight_col],
            default=auto_metrics[:min(4, len(auto_metrics))],
            key="wa_metric_cols",
        )
    with col_w2:
        group_cols = st.multiselect(
            "Группировка (до 3 уровней)",
            options=cat_cols,
            key="wa_group_cols",
        )
        if len(group_cols) > 3:
            st.warning("Рекомендуется не более 3 уровней группировки.")
        dataset_names = list_dataset_names()
        other_names = [n for n in dataset_names if n != chosen]
        ds_b_options = ["(не выбрано)"] + other_names
        ds_b_sel = st.selectbox("Датасет B (для сравнения)", options=ds_b_options, key="wa_ds_b")
        ds_b_name = None if ds_b_sel == "(не выбрано)" else ds_b_sel
    with col_w3:
        calc_duration = st.checkbox("Рассчитать дюрацию", value=False, key="wa_duration")
        coupon_freq = st.number_input(
            "Частота купона (раз/год)",
            min_value=1,
            max_value=12,
            value=12,
            step=1,
            key="wa_coupon_freq",
        )
        cost_of_funds = st.number_input(
            "Стоимость фондирования (%)",
            min_value=0.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            key="wa_cof",
        )
        st.caption(f"Строк: **{len(df):,}**")

    run_btn = st.button("▶ Рассчитать", type="primary", key="wa_run")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_summary, tab_breakdown, tab_compare = st.tabs([
    "📊 Сводка",
    "🔬 Разбивка",
    "↔️ Сравнение",
])

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

if run_btn:
    with st.spinner("Расчёт взвешенных метрик…"):
        if not metric_cols:
            st.error("Выберите хотя бы одну метрику.")
        else:
            try:
                # Portfolio-level (no groups)
                portfolio_stats = portfolio_weighted_averages(df, weight_col, metric_cols)

                # Grouped stats if groups selected
                grouped_stats = None
                if group_cols:
                    grouped_stats = portfolio_weighted_averages(df, weight_col, metric_cols, group_cols)

                # Dataset B stats
                portfolio_stats_b = None
                if ds_b_name:
                    df_b = get_dataset(ds_b_name)
                    if df_b is not None:
                        common_metrics = [c for c in metric_cols if c in df_b.columns]
                        if weight_col in df_b.columns and common_metrics:
                            portfolio_stats_b = portfolio_weighted_averages(df_b, weight_col, common_metrics)
                        else:
                            st.warning("Датасет B не содержит нужных колонок для сравнения.")

                # Duration
                duration_result = None
                if calc_duration and metric_cols:
                    # Find WAR: first metric with 'rate' in name, or first metric
                    war_col = next((c for c in metric_cols if "rate" in c.lower() or "ставк" in c.lower()), metric_cols[0])
                    wam_col = next((c for c in metric_cols if "matur" in c.lower() or "зрел" in c.lower()), None)
                    war_val = float(portfolio_stats[f"{war_col}_wa"].iloc[0]) if f"{war_col}_wa" in portfolio_stats.columns else 0.0
                    wam_val = float(portfolio_stats[f"{wam_col}_wa"].iloc[0]) if wam_col and f"{wam_col}_wa" in portfolio_stats.columns else 12.0
                    duration_result = simplified_duration(war_val, wam_val, int(coupon_freq))
                    duration_result["war_col"] = war_col
                    duration_result["war_val"] = war_val
                    duration_result["spread"] = war_val - cost_of_funds

                st.session_state["wa_result"] = {
                    "portfolio_stats": portfolio_stats,
                    "grouped_stats": grouped_stats,
                    "portfolio_stats_b": portfolio_stats_b,
                    "duration_result": duration_result,
                    "metric_cols": metric_cols,
                    "group_cols": group_cols,
                    "weight_col": weight_col,
                    "ds_b_name": ds_b_name,
                }
                log_event("weighted_avg_run", {
                    "weight_col": weight_col,
                    "metric_cols": metric_cols,
                    "group_cols": group_cols,
                    "ds_b": ds_b_name,
                })
                st.success("Расчёт завершён.")
            except Exception as exc:
                st.error(f"Ошибка: {exc}")

wa_result = st.session_state.get("wa_result")

# ---------------------------------------------------------------------------
# Tab 1: Summary
# ---------------------------------------------------------------------------

with tab_summary:
    if wa_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    else:
        pstats = wa_result["portfolio_stats"]
        pstats_b = wa_result["portfolio_stats_b"]
        mc = wa_result["metric_cols"]
        wc = wa_result["weight_col"]
        duration_result = wa_result["duration_result"]

        # WA metric cards
        n_metrics = len(mc)
        n_cols = min(6, n_metrics)
        cols = st.columns(n_cols) if n_metrics > 0 else []
        for i, metric in enumerate(mc):
            wa_key = f"{metric}_wa"
            if wa_key not in pstats.columns:
                continue
            val_a = pstats[wa_key].iloc[0]
            val_a_str = f"{val_a:.4f}" if not pd.isna(val_a) else "N/A"

            delta_val = None
            if pstats_b is not None and wa_key in pstats_b.columns:
                val_b = pstats_b[wa_key].iloc[0]
                if not pd.isna(val_a) and not pd.isna(val_b):
                    delta_val = round(float(val_a - val_b), 4)

            cols[i % n_cols].metric(
                f"WA {metric}",
                val_a_str,
                delta=str(delta_val) if delta_val is not None else None,
            )

        # Charts
        col_l, col_r = st.columns(2)

        with col_l:
            if wa_result["grouped_stats"] is not None and wa_result["group_cols"]:
                grp_stats = wa_result["grouped_stats"]
                gc = wa_result["group_cols"][0]
                w_sum_col = f"{wc}_sum"
                first_metric_wa = f"{mc[0]}_wa" if mc else None

                if gc in grp_stats.columns and w_sum_col in grp_stats.columns and first_metric_wa in grp_stats.columns:
                    treemap_df = grp_stats[[gc, w_sum_col, first_metric_wa]].copy()
                    treemap_df.columns = ["group", "weight_sum", "metric_wa"]
                    treemap_df = treemap_df.dropna()

                    fig_treemap = px.treemap(
                        treemap_df,
                        path=["group"],
                        values="weight_sum",
                        color="metric_wa",
                        color_continuous_scale="RdYlGn",
                        title=f"Treemap: {gc} — размер={wc}, цвет={mc[0]} WA",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_treemap, use_container_width=True)
                else:
                    st.info("Нет данных для Treemap.")
            else:
                st.info("Добавьте группировку для отображения Treemap.")

        with col_r:
            if mc:
                first_mc = mc[0]
                box_df = df[[first_mc]].copy()
                box_df = box_df.dropna()

                if wa_result["group_cols"] and wa_result["group_cols"][0] in df.columns:
                    gc = wa_result["group_cols"][0]
                    n_groups = df[gc].nunique()
                    if n_groups <= 10:
                        box_df[gc] = df[gc]
                        fig_box = px.box(
                            box_df,
                            y=first_mc,
                            x=gc,
                            color=gc,
                            color_discrete_sequence=COLORS,
                            title=f"Распределение {first_mc} по {gc}",
                            template="plotly_white",
                        )
                    else:
                        fig_box = px.box(
                            box_df,
                            y=first_mc,
                            title=f"Распределение {first_mc}",
                            template="plotly_white",
                        )
                else:
                    fig_box = px.box(
                        box_df,
                        y=first_mc,
                        title=f"Распределение {first_mc}",
                        template="plotly_white",
                    )
                st.plotly_chart(fig_box, use_container_width=True)

        # Duration metrics
        if duration_result:
            section_header("Дюрация портфеля")
            d1, d2, d3 = st.columns(3)
            d1.metric(
                "Модифицированная дюрация",
                f"{duration_result['modified_duration']:.3f} лет",
            )
            dv01_val = duration_result.get("dv01")
            d2.metric(
                "DV01 (на 1 бп)",
                f"{dv01_val:.6f}" if dv01_val is not None else "N/A",
            )
            spread_val = duration_result.get("spread", 0.0)
            d3.metric(
                f"Спред (WAR - стоимость фондирования)",
                f"{spread_val:.2f}%",
                delta="положительный" if spread_val >= 0 else "отрицательный",
                delta_color="normal" if spread_val >= 0 else "inverse",
            )

            if dv01_val is not None:
                total_portfolio = df[weight_col].sum() if weight_col in df.columns else None
                if total_portfolio and total_portfolio > 0:
                    risk_1bp = abs(dv01_val) * total_portfolio
                    interpretation_box(
                        "Интерпретация дюрации и DV01",
                        f"**DV01 = {dv01_val:.6f}** означает, что при сдвиге процентных ставок на **1 б.п. (0.01%)** "
                        f"стоимость портфеля изменится на {abs(dv01_val)*100:.4f}%.\n\n"
                        f"При размере портфеля **{total_portfolio:,.0f}** — это **{risk_1bp:,.0f} руб.** на каждый базисный пункт.\n\n"
                        "**Модифицированная дюрация** показывает чувствительность: при росте ставки на 1% стоимость "
                        f"снижается примерно на **{abs(dv01_val)*100:.2f}%**.",
                        icon="📐",
                    )
                else:
                    interpretation_box(
                        "Интерпретация дюрации",
                        f"**DV01 = {dv01_val:.6f}** — изменение стоимости на 1 базисный пункт (0.01%) изменения ставки.",
                        icon="📐",
                    )

# ---------------------------------------------------------------------------
# Tab 2: Breakdown
# ---------------------------------------------------------------------------

with tab_breakdown:
    if wa_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    else:
        mc = wa_result["metric_cols"]
        wc = wa_result["weight_col"]
        group_cols_available = wa_result["group_cols"] or cat_cols

        if not group_cols_available:
            st.info("Нет доступных категориальных колонок для разбивки.")
        elif not mc:
            st.info("Выберите метрики в настройках выше.")
        else:
            selected_dim = st.selectbox(
                "Основной разрез",
                options=group_cols_available if group_cols_available else cat_cols,
                key="wa_breakdown_dim",
            )

            try:
                breakdown_stats = portfolio_weighted_averages(df, wc, mc, group_cols=[selected_dim])

                selected_metric = st.radio(
                    "Метрика для отображения",
                    options=mc,
                    horizontal=True,
                    key="wa_breakdown_metric",
                )

                if breakdown_stats is not None and not breakdown_stats.empty:
                    wa_col = f"{selected_metric}_wa"
                    if wa_col not in breakdown_stats.columns:
                        st.warning(f"Колонка {wa_col} не найдена.")
                    else:
                        breakdown_plot = breakdown_stats[[selected_dim, wa_col]].dropna()
                        breakdown_plot = breakdown_plot.sort_values(wa_col, ascending=True)

                        # Portfolio avg for reference line
                        portfolio_avg = float(wa_result["portfolio_stats"].get(wa_col, pd.Series([None])).iloc[0] or 0)

                        fig_breakdown = px.bar(
                            breakdown_plot,
                            x=wa_col,
                            y=selected_dim,
                            orientation="h",
                            title=f"WA {selected_metric} по {selected_dim}",
                            labels={wa_col: f"WA {selected_metric}", selected_dim: selected_dim},
                            color=wa_col,
                            color_continuous_scale="RdYlGn",
                            template="plotly_white",
                        )
                        fig_breakdown.add_vline(
                            x=portfolio_avg,
                            line_dash="dash",
                            line_color="navy",
                            annotation_text=f"Портфель: {portfolio_avg:.3f}",
                            annotation_position="top right",
                        )
                        st.plotly_chart(fig_breakdown, use_container_width=True)

                        # Full table with color coding
                        all_wa_cols = [f"{m}_wa" for m in mc if f"{m}_wa" in breakdown_stats.columns]
                        display_df = breakdown_stats[[selected_dim] + all_wa_cols].copy()
                        section_header("Таблица разбивки")

                        try:
                            styled = display_df.style.background_gradient(
                                subset=all_wa_cols,
                                cmap="RdYlGn",
                            )
                            st.dataframe(styled, use_container_width=True, hide_index=True)
                        except Exception:
                            st.dataframe(display_df.round(4), use_container_width=True, hide_index=True)

            except Exception as exc:
                st.error(f"Ошибка разбивки: {exc}")

# ---------------------------------------------------------------------------
# Tab 3: Comparison
# ---------------------------------------------------------------------------

with tab_compare:
    if wa_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    elif wa_result.get("ds_b_name") is None or wa_result.get("portfolio_stats_b") is None:
        st.info("Выберите датасет B в настройках выше для сравнения.")
    else:
        pstats_a = wa_result["portfolio_stats"]
        pstats_b = wa_result["portfolio_stats_b"]
        mc = wa_result["metric_cols"]
        wc = wa_result["weight_col"]
        ds_b_name = wa_result["ds_b_name"]
        df_b = get_dataset(ds_b_name)

        section_header(f"Сравнение: {chosen} vs {ds_b_name}")

        # Side-by-side metric cards
        cols_a = st.columns(len(mc))
        for i, metric in enumerate(mc):
            wa_key = f"{metric}_wa"
            val_a = float(pstats_a[wa_key].iloc[0]) if wa_key in pstats_a.columns and not pstats_a.empty else None
            val_b = float(pstats_b[wa_key].iloc[0]) if wa_key in pstats_b.columns and not pstats_b.empty else None

            val_a_str = f"{val_a:.4f}" if val_a is not None and not pd.isna(val_a) else "N/A"
            delta_val = None
            if val_a is not None and val_b is not None and not pd.isna(val_a) and not pd.isna(val_b):
                delta_val = round(val_a - val_b, 4)

            cols_a[i].metric(
                f"WA {metric}",
                f"A: {val_a_str}",
                delta=f"vs B: {delta_val:+.4f}" if delta_val is not None else None,
            )

        # Mix-rate decomposition if group cols available
        if wa_result["group_cols"] and mc and df_b is not None:
            try:
                gc = wa_result["group_cols"][0]
                first_metric = mc[0]
                if gc in df.columns and gc in df_b.columns and first_metric in df_b.columns and wc in df_b.columns:
                    decomp_df = mix_rate_decomposition(df, df_b, wc, first_metric, gc)

                    if not decomp_df.empty:
                        section_header(f"Декомпозиция: {first_metric} — эффект структуры и ставки")

                        # Waterfall chart
                        measures = []
                        x_vals = []
                        y_vals = []

                        for _, row in decomp_df.iterrows():
                            x_vals.append(f"{row['group']}\nструктура")
                            y_vals.append(float(row["mix_effect"]))
                            measures.append("relative")
                            x_vals.append(f"{row['group']}\nставка")
                            y_vals.append(float(row["rate_effect"]))
                            measures.append("relative")

                        x_vals.append("Итого")
                        y_vals.append(sum(decomp_df["total_effect"]))
                        measures.append("total")

                        fig_wf = go.Figure(go.Waterfall(
                            x=x_vals,
                            y=y_vals,
                            measure=measures,
                            increasing=dict(marker_color="#d62728"),
                            decreasing=dict(marker_color="#2ca02c"),
                            totals=dict(marker_color="#1f77b4"),
                            connector=dict(line=dict(color="grey", dash="dot")),
                        ))
                        fig_wf.update_layout(
                            title=f"Декомпозиция ΔWAR: эффект структуры и ставки",
                            xaxis_title="Компонент",
                            yaxis_title=f"Δ {first_metric}",
                            template="plotly_white",
                            showlegend=False,
                        )
                        st.plotly_chart(fig_wf, use_container_width=True)

                        section_header("Таблица декомпозиции")
                        st.dataframe(decomp_df.round(4), use_container_width=True, hide_index=True)

            except Exception as exc:
                st.warning(f"Декомпозиция недоступна: {exc}")

        # Grouped bar chart if group cols
        if wa_result["group_cols"] and mc and df_b is not None:
            try:
                gc = wa_result["group_cols"][0]
                first_metric = mc[0]
                if gc in df.columns and gc in df_b.columns:
                    stats_a_grp = portfolio_weighted_averages(df, wc, mc, group_cols=[gc])
                    stats_b_grp = portfolio_weighted_averages(df_b, wc, [m for m in mc if m in df_b.columns], group_cols=[gc])

                    wa_col = f"{first_metric}_wa"
                    if wa_col in stats_a_grp.columns:
                        comp_df = stats_a_grp[[gc, wa_col]].rename(columns={wa_col: "A"})
                        if wa_col in stats_b_grp.columns:
                            comp_b = stats_b_grp[[gc, wa_col]].rename(columns={wa_col: "B"})
                            comp_df = comp_df.merge(comp_b, on=gc, how="outer")
                            comp_melt = comp_df.melt(id_vars=gc, value_vars=["A", "B"], var_name="Портфель", value_name=f"WA {first_metric}")
                            fig_cmp = px.bar(
                                comp_melt,
                                x=gc,
                                y=f"WA {first_metric}",
                                color="Портфель",
                                barmode="group",
                                title=f"Сравнение WA {first_metric} по {gc}",
                                color_discrete_sequence=[COLORS[0], COLORS[1]],
                                template="plotly_white",
                            )
                            st.plotly_chart(fig_cmp, use_container_width=True)
            except Exception as exc:
                st.warning(f"График сравнения недоступен: {exc}")

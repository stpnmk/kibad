"""
pages/13_Cohort.py – Когортный анализ: удержание, отток, LTV.

Использует core/cohort.py для расчётов.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, dataset_selectbox
from app.styles import inject_all_css, page_header, section_header
from core.cohort import (
    build_cohort_table,
    retention_table,
    churn_rate_table,
    average_retention_curve,
    compute_clv,
)

st.set_page_config(page_title="KIBAD – Когортный анализ", layout="wide")
init_state()
inject_all_css()
page_header("Когортный анализ", "Удержание клиентов, отток и LTV по когортам", icon="👥")

# ── Выбор датасета ──────────────────────────────────────────────────────────
chosen = dataset_selectbox("cohort_ds")
if not chosen:
    st.info("Загрузите датасет на странице **Данные**.")
    st.stop()

df = st.session_state["datasets"][chosen]
st.caption(f"**{chosen}** — {len(df):,} строк × {len(df.columns)} столбцов")

# ── Настройка колонок ───────────────────────────────────────────────────────
section_header("Настройка параметров", "⚙️")

all_cols = df.columns.tolist()
dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()
str_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()

c1, c2, c3 = st.columns(3)
with c1:
    id_col = st.selectbox("ID клиента", all_cols, help="Уникальный идентификатор клиента")
with c2:
    act_col = st.selectbox("Дата активности", dt_cols if dt_cols else all_cols,
                           help="Дата транзакции / события")
with c3:
    acq_col_options = ["(авто — первая активность)"] + all_cols
    acq_choice = st.selectbox("Дата привлечения", acq_col_options,
                              help="Если не указана, берётся min(дата активности) по клиенту")
    acq_col = None if acq_choice.startswith("(авто") else acq_choice

c4, c5 = st.columns(2)
with c4:
    freq_map = {"Месяц": "MS", "Квартал": "QS"}
    freq_label = st.selectbox("Период когорты", list(freq_map.keys()))
    cohort_freq = freq_map[freq_label]
with c5:
    max_offset = st.slider("Макс. кол-во периодов", 3, 36, 12)

# ── Построить когорту ───────────────────────────────────────────────────────
if st.button("▶ Построить когортный анализ", type="primary"):
    try:
        cohort_counts = build_cohort_table(
            df, id_col, act_col,
            acquisition_date_col=acq_col,
            cohort_freq=cohort_freq,
            max_offset=max_offset,
        )
    except Exception as e:
        st.error(f"Ошибка построения когорты: {e}")
        st.stop()

    if cohort_counts.empty:
        st.warning("Не удалось построить когорты. Проверьте колонки с датами.")
        st.stop()

    st.session_state["_cohort_result"] = cohort_counts

# ── Результаты ──────────────────────────────────────────────────────────────
cohort_counts = st.session_state.get("_cohort_result")
if cohort_counts is None:
    st.info("Настройте параметры и нажмите кнопку выше.")
    st.stop()

ret = retention_table(cohort_counts)
churn = churn_rate_table(ret)
avg_ret = average_retention_curve(cohort_counts)

# ── Метрики ─────────────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
n_cohorts = len(cohort_counts)
total_customers = int(cohort_counts[0].sum()) if 0 in cohort_counts.columns else 0
avg_ret_1 = f"{avg_ret.get(1, 0):.0%}" if 1 in avg_ret.index else "—"
avg_ret_3 = f"{avg_ret.get(3, 0):.0%}" if 3 in avg_ret.index else "—"

m1.metric("Когорт", n_cohorts)
m2.metric("Всего клиентов", f"{total_customers:,}")
m3.metric("Удержание (период 1)", avg_ret_1)
m4.metric("Удержание (период 3)", avg_ret_3)

st.divider()

# ── Табы результатов ────────────────────────────────────────────────────────
tab_heat, tab_retention, tab_churn, tab_curve, tab_clv, tab_data = st.tabs([
    "🔥 Тепловая карта", "📊 Удержание (%)", "📉 Отток",
    "📈 Средняя кривая", "💰 CLV", "📋 Данные"
])

# Helper: format cohort index for display
def _fmt_cohort(idx):
    if hasattr(idx, "strftime"):
        return idx.strftime("%Y-%m")
    return str(idx)


with tab_heat:
    section_header("Тепловая карта удержания", "🔥")
    st.caption("Каждая строка — когорта (месяц привлечения). Значения — % оставшихся клиентов.")

    # Build display df
    display_ret = ret.copy()
    display_ret.index = [_fmt_cohort(c) for c in display_ret.index]

    fig_heat = go.Figure(data=go.Heatmap(
        z=display_ret.values * 100,
        x=[f"Период {c}" for c in display_ret.columns],
        y=display_ret.index,
        colorscale="Blues",
        text=np.round(display_ret.values * 100, 1),
        texttemplate="%{text:.0f}%",
        textfont=dict(size=10),
        hoverongaps=False,
        colorbar=dict(title="%"),
    ))
    fig_heat.update_layout(
        template="plotly_white",
        height=max(300, n_cohorts * 35 + 100),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=80, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


with tab_retention:
    section_header("Таблица удержания", "📊")
    styled_ret = (ret * 100).round(1)
    styled_ret.index = [_fmt_cohort(c) for c in styled_ret.index]
    st.dataframe(
        styled_ret.style.format("{:.1f}%").background_gradient(cmap="Blues", axis=None),
        use_container_width=True,
        height=min(500, n_cohorts * 40 + 60),
    )


with tab_churn:
    section_header("Отток по периодам", "📉")
    st.caption("Доля клиентов, ушедших между периодами.")

    display_churn = (churn * 100).round(1)
    display_churn.index = [_fmt_cohort(c) for c in display_churn.index]
    st.dataframe(
        display_churn.style.format("{:.1f}%", na_rep="—")
            .background_gradient(cmap="Reds", axis=None, vmin=0, vmax=50),
        use_container_width=True,
    )


with tab_curve:
    section_header("Средняя кривая удержания", "📈")
    st.caption("Взвешенная средняя кривая по всем когортам (вес = размер когорты).")

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=[f"Период {k}" for k in avg_ret.index],
        y=(avg_ret.values * 100).round(1),
        mode="lines+markers+text",
        text=[f"{v:.0f}%" for v in avg_ret.values * 100],
        textposition="top center",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=8),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.1)",
    ))
    fig_curve.update_layout(
        template="plotly_white",
        yaxis=dict(title="Удержание, %", range=[0, 105]),
        height=400,
        margin=dict(l=50, r=20, t=20, b=40),
    )
    st.plotly_chart(fig_curve, use_container_width=True)


with tab_clv:
    section_header("Расчёт CLV (Customer Lifetime Value)", "💰")

    clv_c1, clv_c2, clv_c3 = st.columns(3)
    with clv_c1:
        arpu = st.number_input("ARPU (выручка на клиента за период)", value=1000.0,
                               min_value=0.01, step=100.0, format="%.2f")
    with clv_c2:
        discount_rate = st.number_input("Годовая ставка дисконтирования, %", value=12.0,
                                        min_value=0.0, max_value=100.0, step=1.0)
    with clv_c3:
        horizon = st.number_input("Горизонт (месяцев)", value=12, min_value=1, max_value=60)

    if st.button("💰 Рассчитать CLV", type="primary"):
        clv = compute_clv(ret, arpu, discount_rate / 100.0, horizon)
        if clv.empty:
            st.warning("Не удалось рассчитать CLV.")
        else:
            clv_df = pd.DataFrame({
                "Когорта": [_fmt_cohort(c) for c in clv.index],
                "Размер когорты": [int(cohort_counts.loc[c, 0]) if c in cohort_counts.index and 0 in cohort_counts.columns else 0 for c in clv.index],
                "CLV": clv.values.round(2),
            })
            avg_clv = clv.mean()

            clv_m1, clv_m2, clv_m3 = st.columns(3)
            clv_m1.metric("Средний CLV", f"{avg_clv:,.0f}")
            clv_m2.metric("Макс CLV", f"{clv.max():,.0f}")
            clv_m3.metric("Мин CLV", f"{clv.min():,.0f}")

            fig_clv = go.Figure(go.Bar(
                x=clv_df["Когорта"],
                y=clv_df["CLV"],
                marker_color="#2563eb",
                text=clv_df["CLV"].apply(lambda v: f"{v:,.0f}"),
                textposition="outside",
            ))
            fig_clv.update_layout(
                template="plotly_white",
                yaxis_title="CLV",
                height=400,
            )
            st.plotly_chart(fig_clv, use_container_width=True)
            st.dataframe(clv_df, use_container_width=True, hide_index=True)


with tab_data:
    section_header("Исходные данные когорт", "📋")
    display_counts = cohort_counts.copy()
    display_counts.index = [_fmt_cohort(c) for c in display_counts.index]
    st.dataframe(display_counts, use_container_width=True)
    st.download_button(
        "📥 Скачать когортную таблицу (CSV)",
        display_counts.to_csv(),
        file_name="cohort_table.csv",
        mime="text/csv",
    )

"""
pages/15_ABC.py – ABC-XYZ анализ: классификация объектов по вкладу и стабильности.

Универсальный инструмент для любого бизнеса:
- ABC: сортировка по доле в общем объёме (Парето)
- XYZ: классификация по вариабельности (коэффициент вариации)
- Кросс-матрица ABC×XYZ
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.state import init_state, dataset_selectbox
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – ABC-XYZ анализ", layout="wide")
init_state()
inject_all_css()
page_header("ABC-XYZ анализ", "Классификация объектов по вкладу и стабильности", icon="🏷️")

# ── Методичка ───────────────────────────────────────────────────────────────
with st.expander("📖 Как работает ABC-XYZ анализ", expanded=False):
    st.markdown("""
**ABC-анализ** — классификация по принципу Парето (80/20):
- **A** — объекты, дающие 80% суммарного значения (ключевые)
- **B** — следующие 15% (средние)
- **C** — оставшиеся 5% (малозначимые)

**XYZ-анализ** — классификация по стабильности (коэффициент вариации CV):
- **X** — стабильные (CV < 10%), легко прогнозировать
- **Y** — умеренно нестабильные (10% ≤ CV < 25%)
- **Z** — нестабильные (CV ≥ 25%), трудно прогнозировать

**Кросс-матрица ABC×XYZ** даёт 9 сегментов с чёткими рекомендациями:
- **AX** → ключевые стабильные — точный прогноз, приоритетное управление
- **CZ** → малозначимые нестабильные — можно сократить или автоматизировать
""")

# ── Выбор датасета ──────────────────────────────────────────────────────────
chosen = dataset_selectbox("abc_ds")
if not chosen:
    st.info("Загрузите датасет на странице **Данные**.")
    st.stop()

df = st.session_state["datasets"][chosen]
st.caption(f"**{chosen}** — {len(df):,} строк × {len(df.columns)} столбцов")

# ── Настройка ───────────────────────────────────────────────────────────────
section_header("Настройка параметров", "⚙️")

all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

c1, c2 = st.columns(2)
with c1:
    item_col = st.selectbox("Объект (товар / клиент / SKU)", cat_cols if cat_cols else all_cols,
                            help="Колонка с наименованием объекта классификации")
with c2:
    value_col = st.selectbox("Значение (выручка / объём / сумма)", num_cols,
                             help="Числовая колонка для анализа")

# XYZ нуждается во временном измерении
has_time = bool(dt_cols)
use_xyz = st.checkbox("Включить XYZ-анализ (требуется временная колонка)", value=has_time)
time_col = None
if use_xyz:
    if dt_cols:
        time_col = st.selectbox("Временная колонка", dt_cols)
    else:
        st.warning("Нет колонок с датами — XYZ анализ невозможен.")
        use_xyz = False

# Пороги
with st.expander("⚙️ Настроить пороги", expanded=False):
    tc1, tc2 = st.columns(2)
    with tc1:
        abc_a = st.slider("Порог A (%)", 50, 95, 80, help="Кумулятивная доля для класса A")
        abc_b = st.slider("Порог A+B (%)", abc_a + 1, 99, 95, help="Кумулятивная доля для A+B")
    with tc2:
        xyz_x = st.slider("Порог X (CV, %)", 1, 30, 10, help="Макс. CV для класса X")
        xyz_y = st.slider("Порог Y (CV, %)", xyz_x + 1, 60, 25, help="Макс. CV для класса Y")

# ── Запуск ──────────────────────────────────────────────────────────────────
if not st.button("▶ Запустить ABC-XYZ анализ", type="primary"):
    st.info("Настройте параметры и нажмите кнопку.")
    st.stop()

# ── ABC расчёт ──────────────────────────────────────────────────────────────
agg = df.groupby(item_col, dropna=False)[value_col].sum().reset_index()
agg.columns = ["item", "total"]
agg = agg.sort_values("total", ascending=False).reset_index(drop=True)
agg["share"] = agg["total"] / agg["total"].sum() * 100
agg["cumshare"] = agg["share"].cumsum()

def _abc_class(cumshare):
    if cumshare <= abc_a:
        return "A"
    elif cumshare <= abc_b:
        return "B"
    return "C"

agg["ABC"] = agg["cumshare"].apply(_abc_class)

# ── XYZ расчёт ──────────────────────────────────────────────────────────────
if use_xyz and time_col:
    # Агрегируем по (item, period)
    work = df[[item_col, time_col, value_col]].copy()
    work[time_col] = pd.to_datetime(work[time_col], errors="coerce")
    work = work.dropna(subset=[time_col])
    work["_period"] = work[time_col].dt.to_period("M").dt.to_timestamp()

    period_agg = work.groupby([item_col, "_period"])[value_col].sum().reset_index()
    pivot = period_agg.pivot(index=item_col, columns="_period", values=value_col).fillna(0)

    cv_series = pivot.std(axis=1) / pivot.mean(axis=1).replace(0, np.nan) * 100
    cv_df = cv_series.reset_index()
    cv_df.columns = ["item", "CV"]
    cv_df["CV"] = cv_df["CV"].fillna(999)

    def _xyz_class(cv):
        if cv <= xyz_x:
            return "X"
        elif cv <= xyz_y:
            return "Y"
        return "Z"

    cv_df["XYZ"] = cv_df["CV"].apply(_xyz_class)
    agg = agg.merge(cv_df, on="item", how="left")
    agg["XYZ"] = agg["XYZ"].fillna("Z")
    agg["CV"] = agg["CV"].fillna(0).round(1)
    agg["Класс"] = agg["ABC"] + agg["XYZ"]
else:
    agg["Класс"] = agg["ABC"]

# Сохраняем результат
st.session_state["_abc_result"] = agg

# ── Метрики ─────────────────────────────────────────────────────────────────
n_items = len(agg)
n_a = (agg["ABC"] == "A").sum()
n_b = (agg["ABC"] == "B").sum()
n_c = (agg["ABC"] == "C").sum()
share_a = agg.loc[agg["ABC"] == "A", "total"].sum() / agg["total"].sum() * 100

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Всего объектов", n_items)
mc2.metric("Класс A", f"{n_a} ({n_a/n_items*100:.0f}%)")
mc3.metric("Класс B", f"{n_b} ({n_b/n_items*100:.0f}%)")
mc4.metric("Класс C", f"{n_c} ({n_c/n_items*100:.0f}%)")
mc5.metric("Доля A в выручке", f"{share_a:.1f}%")

st.divider()

# ── Табы ────────────────────────────────────────────────────────────────────
tabs = ["📊 ABC-график", "📋 Таблица"]
if use_xyz and time_col:
    tabs += ["🔀 Матрица ABC×XYZ", "📈 XYZ-график"]
tab_list = st.tabs(tabs)

# Tab 1: ABC Pareto chart
with tab_list[0]:
    section_header("ABC: кумулятивная кривая (Парето)", "📊")

    fig_abc = go.Figure()
    colors = agg["ABC"].map({"A": "#2563eb", "B": "#d97706", "C": "#94a3b8"})
    fig_abc.add_trace(go.Bar(
        x=agg["item"], y=agg["share"],
        marker_color=colors.tolist(),
        name="Доля, %",
        hovertemplate="%{x}: %{y:.1f}%<extra></extra>",
    ))
    fig_abc.add_trace(go.Scatter(
        x=agg["item"], y=agg["cumshare"],
        mode="lines+markers",
        name="Кумулятивная %",
        line=dict(color="#dc2626", width=2),
        marker=dict(size=3),
        yaxis="y2",
    ))
    # Threshold lines
    fig_abc.add_hline(y=abc_a, line_dash="dash", line_color="#2563eb",
                      annotation_text=f"A: {abc_a}%", yref="y2")
    fig_abc.add_hline(y=abc_b, line_dash="dash", line_color="#d97706",
                      annotation_text=f"B: {abc_b}%", yref="y2")

    fig_abc.update_layout(
        template="plotly_white",
        yaxis=dict(title="Доля, %"),
        yaxis2=dict(title="Кумулятивная доля, %", overlaying="y", side="right", range=[0, 105]),
        height=500,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=50, r=50, t=20, b=80),
    )
    if n_items > 30:
        fig_abc.update_xaxes(showticklabels=False)
    st.plotly_chart(fig_abc, use_container_width=True)

# Tab 2: Table
with tab_list[1]:
    section_header("Детальная таблица", "📋")

    display_cols = ["item", "total", "share", "cumshare", "ABC"]
    if "XYZ" in agg.columns:
        display_cols += ["CV", "XYZ", "Класс"]

    display_df = agg[display_cols].copy()
    display_df.columns = ["Объект", "Сумма", "Доля %", "Кумул. %", "ABC"] + (
        ["CV %", "XYZ", "Класс"] if "XYZ" in agg.columns else []
    )
    display_df["Сумма"] = display_df["Сумма"].round(2)
    display_df["Доля %"] = display_df["Доля %"].round(2)
    display_df["Кумул. %"] = display_df["Кумул. %"].round(2)

    def _color_abc(val):
        colors = {"A": "background-color: #dbeafe", "B": "background-color: #fef3c7",
                  "C": "background-color: #f1f5f9"}
        return colors.get(val, "")

    styled = display_df.style.map(_color_abc, subset=["ABC"])
    if "XYZ" in agg.columns:
        def _color_xyz(val):
            colors = {"X": "background-color: #d1fae5", "Y": "background-color: #fef3c7",
                      "Z": "background-color: #fee2e2"}
            return colors.get(val, "")
        styled = styled.map(_color_xyz, subset=["XYZ"])

    st.dataframe(styled, use_container_width=True, hide_index=True, height=500)

    st.download_button(
        "📥 Скачать CSV",
        display_df.to_csv(index=False),
        file_name="abc_xyz_analysis.csv",
        mime="text/csv",
    )

# Tab 3 & 4: XYZ-specific
if use_xyz and time_col and len(tab_list) > 2:
    with tab_list[2]:
        section_header("Кросс-матрица ABC × XYZ", "🔀")
        st.caption("Каждая ячейка — количество объектов в сегменте.")

        matrix = agg.groupby(["ABC", "XYZ"]).size().unstack(fill_value=0)
        # Ensure all categories present
        for c in ["X", "Y", "Z"]:
            if c not in matrix.columns:
                matrix[c] = 0
        for r in ["A", "B", "C"]:
            if r not in matrix.index:
                matrix.loc[r] = 0
        matrix = matrix.loc[["A", "B", "C"], ["X", "Y", "Z"]]

        # Heatmap
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=["X (стабильные)", "Y (умеренные)", "Z (нестабильные)"],
            y=["A (ключевые)", "B (средние)", "C (малозначимые)"],
            colorscale="Blues",
            text=matrix.values,
            texttemplate="%{text}",
            textfont=dict(size=16, color="black"),
        ))
        fig_matrix.update_layout(
            template="plotly_white",
            height=350,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=120, r=20, t=20, b=60),
        )
        st.plotly_chart(fig_matrix, use_container_width=True)

        # Рекомендации
        section_header("Рекомендации по сегментам", "💡")
        rec_data = {
            "AX": ("🟢", "Точный прогноз, поддержание запасов, приоритетное обслуживание"),
            "AY": ("🟡", "Мониторинг колебаний, буферный запас, анализ причин нестабильности"),
            "AZ": ("🔴", "Глубокий анализ причин нестабильности, ручное управление, страховой запас"),
            "BX": ("🟢", "Автоматическое управление, стандартные процессы"),
            "BY": ("🟡", "Периодический контроль, средний буфер"),
            "BZ": ("🟡", "Анализ по запросу, гибкие поставки"),
            "CX": ("⚪", "Минимальный контроль, автоматизация"),
            "CY": ("⚪", "Рассмотреть сокращение ассортимента"),
            "CZ": ("⚪", "Кандидаты на вывод / замену"),
        }

        for cls, (icon, rec) in rec_data.items():
            count = int(matrix.loc[cls[0], cls[1]]) if cls[0] in matrix.index and cls[1] in matrix.columns else 0
            if count > 0:
                st.markdown(f"**{icon} {cls}** ({count} объектов) — {rec}")

    with tab_list[3]:
        section_header("Распределение коэффициента вариации", "📈")

        fig_xyz = go.Figure()
        for cls, color in [("X", "#059669"), ("Y", "#d97706"), ("Z", "#dc2626")]:
            mask = agg["XYZ"] == cls
            if mask.any():
                fig_xyz.add_trace(go.Histogram(
                    x=agg.loc[mask, "CV"],
                    name=f"{cls} ({mask.sum()} шт.)",
                    marker_color=color,
                    opacity=0.7,
                ))
        fig_xyz.add_vline(x=xyz_x, line_dash="dash", line_color="#059669",
                          annotation_text=f"X<{xyz_x}%")
        fig_xyz.add_vline(x=xyz_y, line_dash="dash", line_color="#d97706",
                          annotation_text=f"Y<{xyz_y}%")
        fig_xyz.update_layout(
            template="plotly_white",
            xaxis_title="Коэффициент вариации, %",
            yaxis_title="Количество объектов",
            barmode="overlay",
            height=400,
        )
        st.plotly_chart(fig_xyz, use_container_width=True)

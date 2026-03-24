"""
pages/16_Funnel.py – Воронка конверсий: визуализация и анализ этапов.

Универсальный инструмент для любого бизнеса:
- Воронка продаж, маркетинга, HR, поддержки
- Drop-off между этапами
- Конверсия на каждом шаге
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

st.set_page_config(page_title="KIBAD – Воронка конверсий", layout="wide")
init_state()
inject_all_css()
page_header("Воронка конверсий", "Визуализация и анализ этапов процесса", icon="🔻")

# ── Метод ввода ─────────────────────────────────────────────────────────────
mode = st.radio("Способ ввода данных", ["📊 Из датасета", "✏️ Вручную"], horizontal=True)

funnel_data = None  # list of (stage, count)

if mode == "📊 Из датасета":
    chosen = dataset_selectbox("funnel_ds")
    if not chosen:
        st.info("Загрузите датасет на странице **Данные**.")
        st.stop()

    df = st.session_state["datasets"][chosen]
    st.caption(f"**{chosen}** — {len(df):,} строк × {len(df.columns)} столбцов")

    section_header("Настройка воронки из данных", "⚙️")

    input_mode = st.radio("Формат данных", [
        "Колонка этапов (каждая строка = событие)",
        "Агрегированные данные (этап + количество)",
    ], horizontal=True)

    if input_mode.startswith("Колонка этапов"):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        all_cols = df.columns.tolist()
        stage_col = st.selectbox("Колонка этапов", cat_cols if cat_cols else all_cols)

        # Get stage order
        stages_in_data = df[stage_col].dropna().unique().tolist()
        st.caption(f"Найдено {len(stages_in_data)} уникальных этапов")

        # Let user reorder
        stages_ordered = st.multiselect(
            "Порядок этапов (перетащите для сортировки)",
            stages_in_data,
            default=stages_in_data,
            help="Выберите этапы в правильном порядке: от первого к последнему",
        )

        if stages_ordered and st.button("▶ Построить воронку", type="primary"):
            counts = []
            for s in stages_ordered:
                counts.append((s, int((df[stage_col] == s).sum())))
            funnel_data = counts

    else:  # Агрегированные данные
        c1, c2 = st.columns(2)
        with c1:
            stage_col = st.selectbox("Колонка с названием этапа", df.columns.tolist())
        with c2:
            num_cols = df.select_dtypes(include="number").columns.tolist()
            count_col = st.selectbox("Колонка с количеством", num_cols)

        if st.button("▶ Построить воронку", type="primary"):
            agg = df[[stage_col, count_col]].dropna()
            funnel_data = list(zip(agg[stage_col].astype(str), agg[count_col].astype(int)))

else:  # Ручной ввод
    section_header("Ручной ввод этапов", "✏️")

    st.caption("Введите этапы воронки и количество на каждом.")

    n_stages = st.number_input("Количество этапов", min_value=2, max_value=20, value=5)

    stages = []
    defaults = [
        ("Визиты на сайт", 10000),
        ("Регистрация", 3500),
        ("Активация", 1800),
        ("Первая покупка", 900),
        ("Повторная покупка", 350),
    ]
    for i in range(n_stages):
        c1, c2 = st.columns([2, 1])
        with c1:
            default_name = defaults[i][0] if i < len(defaults) else f"Этап {i + 1}"
            name = st.text_input(f"Этап {i + 1}", default_name, key=f"funnel_stage_{i}")
        with c2:
            default_val = defaults[i][1] if i < len(defaults) else 100
            val = st.number_input(f"Количество", min_value=0, value=default_val, key=f"funnel_val_{i}")
        stages.append((name, val))

    if st.button("▶ Построить воронку", type="primary"):
        funnel_data = stages

# ── Результаты ──────────────────────────────────────────────────────────────
if funnel_data is None:
    st.stop()

if len(funnel_data) < 2:
    st.warning("Нужно минимум 2 этапа для воронки.")
    st.stop()

stages, counts = zip(*funnel_data)
counts = [int(c) for c in counts]

# ── Метрики ─────────────────────────────────────────────────────────────────
total_conversion = counts[-1] / counts[0] * 100 if counts[0] > 0 else 0
biggest_drop_idx = 0
biggest_drop_pct = 0
for i in range(1, len(counts)):
    if counts[i - 1] > 0:
        drop = (counts[i - 1] - counts[i]) / counts[i - 1] * 100
        if drop > biggest_drop_pct:
            biggest_drop_pct = drop
            biggest_drop_idx = i

m1, m2, m3, m4 = st.columns(4)
m1.metric("Вход в воронку", f"{counts[0]:,}")
m2.metric("Выход из воронки", f"{counts[-1]:,}")
m3.metric("Общая конверсия", f"{total_conversion:.1f}%")
m4.metric("Макс. потеря", f"{stages[biggest_drop_idx]} (−{biggest_drop_pct:.0f}%)")

st.divider()

# ── Табы ────────────────────────────────────────────────────────────────────
tab_funnel, tab_bar, tab_table = st.tabs(["🔻 Воронка", "📊 Потери по этапам", "📋 Таблица"])

with tab_funnel:
    section_header("Воронка конверсий", "🔻")

    fig = go.Figure(go.Funnel(
        y=list(stages),
        x=list(counts),
        textposition="inside",
        textinfo="value+percent initial",
        marker=dict(
            color=["#1F3864", "#2563eb", "#3b82f6", "#60a5fa", "#93c5fd",
                   "#bfdbfe", "#dbeafe", "#eff6ff", "#f0f9ff", "#f8fafc"][:len(stages)],
        ),
        connector=dict(line=dict(color="#cbd5e1", width=1)),
    ))
    fig.update_layout(
        template="plotly_white",
        height=max(400, len(stages) * 60),
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_bar:
    section_header("Потери между этапами", "📊")

    drop_stages = []
    drop_values = []
    drop_colors = []
    for i in range(1, len(counts)):
        drop = counts[i - 1] - counts[i]
        drop_pct = drop / counts[i - 1] * 100 if counts[i - 1] > 0 else 0
        drop_stages.append(f"{stages[i-1]} → {stages[i]}")
        drop_values.append(drop_pct)
        if drop_pct >= 50:
            drop_colors.append("#dc2626")
        elif drop_pct >= 30:
            drop_colors.append("#d97706")
        else:
            drop_colors.append("#059669")

    fig_drop = go.Figure(go.Bar(
        x=drop_stages,
        y=drop_values,
        marker_color=drop_colors,
        text=[f"−{v:.0f}%" for v in drop_values],
        textposition="outside",
    ))
    fig_drop.update_layout(
        template="plotly_white",
        yaxis_title="Потеря, %",
        height=400,
        margin=dict(l=50, r=20, t=20, b=80),
    )
    st.plotly_chart(fig_drop, use_container_width=True)

    # Интерпретация
    if biggest_drop_pct > 40:
        st.warning(
            f"⚠️ Критическая потеря на этапе **{stages[biggest_drop_idx-1]} → {stages[biggest_drop_idx]}**: "
            f"−{biggest_drop_pct:.0f}%. Рекомендуется детальный анализ причин."
        )

with tab_table:
    section_header("Детальная таблица", "📋")

    rows = []
    for i, (stage, count) in enumerate(zip(stages, counts)):
        conv_from_prev = (count / counts[i - 1] * 100) if i > 0 and counts[i - 1] > 0 else 100.0
        conv_from_start = count / counts[0] * 100 if counts[0] > 0 else 0
        drop = counts[i - 1] - count if i > 0 else 0
        rows.append({
            "Этап": stage,
            "Количество": count,
            "Конверсия от пред.": f"{conv_from_prev:.1f}%",
            "Конверсия от входа": f"{conv_from_start:.1f}%",
            "Потеря": f"−{drop:,}" if i > 0 else "—",
        })

    table_df = pd.DataFrame(rows)
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.download_button(
        "📥 Скачать CSV",
        table_df.to_csv(index=False),
        file_name="funnel_analysis.csv",
        mime="text/csv",
    )

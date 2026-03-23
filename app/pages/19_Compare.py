"""
pages/19_Compare.py – Side-by-side comparison of two periods or two segments.
Replaces Excel "this month vs last month" manual SUMIF tables for bank employees.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df
from core.audit import log_event
from app.components.ux import active_dataset_warnings
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Сравнение", layout="wide")
init_state()
inject_all_css()
active_dataset_warnings()

page_header("19. Сравнение", "Период A vs B, сегмент A vs B", "⚖️")

ds_name = dataset_selectbox("Датасет", key="cmp_ds")
if not ds_name:
    st.stop()

df = get_active_df()
if df is None or df.empty:
    st.info("Загрузите данные на странице «Данные».")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

if not num_cols:
    st.error("Нет числовых столбцов для сравнения.")
    st.stop()

# ---------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------
st.markdown("---")
mode = st.radio(
    "Режим сравнения:",
    ["📅 Период А vs Период Б", "🏷️ Сегмент А vs Сегмент Б", "📊 Фильтр А vs Фильтр Б"],
    horizontal=True,
    key="cmp_mode",
)

# ---------------------------------------------------------------------------
# Metrics to compare
# ---------------------------------------------------------------------------
with st.expander("📐 Выберите метрики и агрегат", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        selected_metrics = st.multiselect(
            "Показатели для сравнения:",
            num_cols,
            default=num_cols[:4],
            key="cmp_metrics",
        )
    with c2:
        agg_map = {
            "Сумма": "sum", "Среднее": "mean", "Медиана": "median",
            "Максимум": "max", "Минимум": "min", "Количество": "count",
        }
        agg_label = st.selectbox("Агрегация:", list(agg_map.keys()), key="cmp_agg")
        agg_func = agg_map[agg_label]

if not selected_metrics:
    st.info("Выберите хотя бы один показатель.")
    st.stop()


def aggregate_subset(sub_df: pd.DataFrame, metrics: list, func: str) -> dict:
    """Aggregate a subset of the dataframe."""
    result = {}
    for m in metrics:
        if m not in sub_df.columns:
            result[m] = None
            continue
        s = sub_df[m].dropna()
        if func == "sum":
            result[m] = s.sum()
        elif func == "mean":
            result[m] = s.mean()
        elif func == "median":
            result[m] = s.median()
        elif func == "max":
            result[m] = s.max()
        elif func == "min":
            result[m] = s.min()
        elif func == "count":
            result[m] = len(s)
        else:
            result[m] = s.sum()
    return result


# ---------------------------------------------------------------------------
# Define the two subsets based on mode
# ---------------------------------------------------------------------------
df_a = df_b = None
label_a = "А"
label_b = "Б"

if mode == "📅 Период А vs Период Б":
    if not date_cols:
        st.error("Нет столбцов с датами. Сначала распознайте даты на странице «Подготовка».")
        st.stop()

    date_col = st.selectbox("Столбец даты:", date_cols, key="cmp_date_col")
    min_date = df[date_col].min()
    max_date = df[date_col].max()

    st.markdown("**Период А (сравниваемый):**")
    pa1, pa2 = st.columns(2)
    with pa1:
        a_start = st.date_input("Начало А:", value=min_date, min_value=min_date, max_value=max_date, key="cmp_a_start")
    with pa2:
        mid_date = min_date + (max_date - min_date) / 2
        a_end = st.date_input("Конец А:", value=mid_date, min_value=min_date, max_value=max_date, key="cmp_a_end")

    st.markdown("**Период Б (базовый/план):**")
    pb1, pb2 = st.columns(2)
    with pb1:
        b_start = st.date_input("Начало Б:", value=mid_date, min_value=min_date, max_value=max_date, key="cmp_b_start")
    with pb2:
        b_end = st.date_input("Конец Б:", value=max_date, min_value=min_date, max_value=max_date, key="cmp_b_end")

    label_a = f"{a_start} – {a_end}"
    label_b = f"{b_start} – {b_end}"

    df_a = df[(df[date_col] >= pd.Timestamp(a_start)) & (df[date_col] <= pd.Timestamp(a_end))]
    df_b = df[(df[date_col] >= pd.Timestamp(b_start)) & (df[date_col] <= pd.Timestamp(b_end))]

elif mode == "🏷️ Сегмент А vs Сегмент Б":
    if not cat_cols:
        st.error("Нет категориальных столбцов для сегментации.")
        st.stop()

    seg_col = st.selectbox("Столбец сегмента:", cat_cols, key="cmp_seg_col")
    unique_vals = sorted(df[seg_col].dropna().unique().tolist())

    sc1, sc2 = st.columns(2)
    with sc1:
        seg_a_vals = st.multiselect("Значения для А:", unique_vals, default=unique_vals[:1], key="cmp_seg_a")
        label_a = ", ".join(str(v) for v in seg_a_vals[:3])
    with sc2:
        seg_b_vals = st.multiselect("Значения для Б:", unique_vals,
                                    default=unique_vals[1:2] if len(unique_vals) > 1 else unique_vals[:1],
                                    key="cmp_seg_b")
        label_b = ", ".join(str(v) for v in seg_b_vals[:3])

    df_a = df[df[seg_col].isin(seg_a_vals)] if seg_a_vals else df.iloc[:0]
    df_b = df[df[seg_col].isin(seg_b_vals)] if seg_b_vals else df.iloc[:0]

else:  # Custom filter mode
    st.markdown("**Условие для группы А:**")
    fa1, fa2, fa3 = st.columns(3)
    with fa1:
        fa_col = st.selectbox("Столбец А:", df.columns.tolist(), key="cmp_fa_col")
    with fa2:
        fa_op = st.selectbox("Оператор А:", [">", "<", ">=", "<=", "==", "!=", "содержит"], key="cmp_fa_op")
    with fa3:
        fa_val_str = st.text_input("Значение А:", key="cmp_fa_val")

    st.markdown("**Условие для группы Б:**")
    fb1, fb2, fb3 = st.columns(3)
    with fb1:
        fb_col = st.selectbox("Столбец Б:", df.columns.tolist(), key="cmp_fb_col")
    with fb2:
        fb_op = st.selectbox("Оператор Б:", [">", "<", ">=", "<=", "==", "!=", "содержит"], key="cmp_fb_op")
    with fb3:
        fb_val_str = st.text_input("Значение Б:", key="cmp_fb_val")

    def apply_filter(src_df, col, op, val_str):
        try:
            col_dtype = src_df[col].dtype
            if op == "содержит":
                return src_df[src_df[col].astype(str).str.contains(val_str, na=False)]
            val = float(val_str) if pd.api.types.is_numeric_dtype(col_dtype) else val_str
            if op == ">":   return src_df[src_df[col] > val]
            if op == "<":   return src_df[src_df[col] < val]
            if op == ">=":  return src_df[src_df[col] >= val]
            if op == "<=":  return src_df[src_df[col] <= val]
            if op == "==":  return src_df[src_df[col] == val]
            if op == "!=":  return src_df[src_df[col] != val]
        except Exception:
            return src_df.iloc[:0]
        return src_df

    label_a = f"{fa_col} {fa_op} {fa_val_str}"
    label_b = f"{fb_col} {fb_op} {fb_val_str}"
    df_a = apply_filter(df, fa_col, fa_op, fa_val_str) if fa_val_str else df.iloc[:0]
    df_b = apply_filter(df, fb_col, fb_op, fb_val_str) if fb_val_str else df.iloc[:0]

# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------
st.markdown("---")
if st.button("▶ Сравнить", type="primary", key="btn_compare"):
    if df_a is None or df_b is None:
        st.error("Не удалось определить группы для сравнения.")
        st.stop()
    if df_a.empty and df_b.empty:
        st.warning("Обе группы пусты. Проверьте условия фильтрации.")
        st.stop()

    vals_a = aggregate_subset(df_a, selected_metrics, agg_func)
    vals_b = aggregate_subset(df_b, selected_metrics, agg_func)

    # Build comparison table
    rows = []
    for m in selected_metrics:
        va = vals_a.get(m, 0) or 0
        vb = vals_b.get(m, 0) or 0
        delta = va - vb
        delta_pct = (delta / vb * 100) if vb != 0 else None
        rows.append({
            "Показатель": m,
            f"А: {label_a[:30]}": va,
            f"Б: {label_b[:30]}": vb,
            "Изменение": delta,
            "Изменение %": delta_pct,
        })
    cmp_df = pd.DataFrame(rows)

    st.session_state["_cmp_result"] = cmp_df
    st.session_state["_cmp_label_a"] = label_a
    st.session_state["_cmp_label_b"] = label_b
    st.session_state["_cmp_n_a"] = len(df_a)
    st.session_state["_cmp_n_b"] = len(df_b)

    log_event("analysis_run", {"type": "comparison", "dataset": ds_name, "mode": mode})


# Display stored results
if "_cmp_result" in st.session_state:
    cmp_df = st.session_state["_cmp_result"]
    label_a = st.session_state.get("_cmp_label_a", "А")
    label_b = st.session_state.get("_cmp_label_b", "Б")
    n_a = st.session_state.get("_cmp_n_a", 0)
    n_b = st.session_state.get("_cmp_n_b", 0)

    # Header info
    h1, h2 = st.columns(2)
    with h1:
        st.info(f"**Группа А:** {label_a}\n\n{n_a:,} строк")
    with h2:
        st.info(f"**Группа Б:** {label_b}\n\n{n_b:,} строк")

    res_tabs = st.tabs(["📋 Таблица", "📊 Диаграмма отклонений", "🌊 Водопад", "💾 Экспорт"])

    col_a_name = [c for c in cmp_df.columns if c.startswith("А:")][0]
    col_b_name = [c for c in cmp_df.columns if c.startswith("Б:")][0]

    with res_tabs[0]:
        # Styled display
        def color_delta(val):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return ""
            return "color: green; font-weight: bold" if val > 0 else ("color: red; font-weight: bold" if val < 0 else "")

        styled = cmp_df.style.applymap(color_delta, subset=["Изменение", "Изменение %"])
        st.dataframe(styled, use_container_width=True, height=350)

        # KPI summary cards
        st.markdown("#### Ключевые отклонения:")
        metric_cards = st.columns(min(len(selected_metrics), 4))
        for i, row in enumerate(cmp_df.itertuples()):
            col_idx = i % 4
            with metric_cards[col_idx]:
                val_a = getattr(row, col_a_name.replace(" ", "_").replace(":", "").replace("–", "").replace("-", "_"), 0)
                try:
                    val_a = cmp_df.iloc[i][col_a_name]
                    val_b = cmp_df.iloc[i][col_b_name]
                    delta_pct = cmp_df.iloc[i]["Изменение %"]
                    delta_str = f"{delta_pct:+.1f}%" if delta_pct is not None and not np.isnan(delta_pct) else "н/д"
                    st.metric(
                        label=cmp_df.iloc[i]["Показатель"],
                        value=f"{val_a:,.2f}".replace(",", " "),
                        delta=delta_str,
                    )
                except Exception:
                    pass

    with res_tabs[1]:
        # Bar chart comparing A vs B
        plot_data = []
        for _, row in cmp_df.iterrows():
            plot_data.append({"Показатель": row["Показатель"], "Значение": row[col_a_name], "Группа": f"А: {label_a[:20]}"})
            plot_data.append({"Показатель": row["Показатель"], "Значение": row[col_b_name], "Группа": f"Б: {label_b[:20]}"})
        plot_df = pd.DataFrame(plot_data)

        fig = px.bar(
            plot_df, x="Показатель", y="Значение", color="Группа",
            barmode="group", title="Сравнение А vs Б",
            template="plotly_white",
            color_discrete_map={
                f"А: {label_a[:20]}": "#636EFA",
                f"Б: {label_b[:20]}": "#EF553B",
            }
        )
        fig.update_layout(height=450, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

        # Delta % chart
        delta_vals = cmp_df["Изменение %"].fillna(0)
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in delta_vals]
        fig2 = go.Figure(go.Bar(
            x=cmp_df["Показатель"],
            y=delta_vals,
            marker_color=colors,
            text=[f"{v:+.1f}%" for v in delta_vals],
            textposition="outside",
        ))
        fig2.update_layout(
            title="Отклонение А vs Б (%)",
            yaxis_title="Изменение %",
            template="plotly_white",
            height=350,
        )
        st.plotly_chart(fig2, use_container_width=True)

    with res_tabs[2]:
        # Waterfall chart
        deltas = cmp_df["Изменение"].tolist()
        names = cmp_df["Показатель"].tolist()
        measure = ["relative"] * len(names)

        fig_wf = go.Figure(go.Waterfall(
            name="Отклонение",
            orientation="v",
            measure=measure,
            x=names,
            y=deltas,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2ecc71"}},
            decreasing={"marker": {"color": "#e74c3c"}},
            text=[f"{d:+,.0f}".replace(",", " ") for d in deltas],
            textposition="outside",
        ))
        fig_wf.update_layout(
            title="Вклад изменений по показателям (водопад)",
            template="plotly_white",
            height=450,
            showlegend=False,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        with st.expander("💡 Как читать водопадную диаграмму"):
            st.markdown(
                "Зелёные столбцы — показатели выросли в группе А по сравнению с Б. "
                "Красные — снизились. Высота столбца = величина изменения. "
                "Используйте для выявления ключевых факторов отклонения."
            )

    with res_tabs[3]:
        buf = cmp_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 Скачать таблицу (CSV)", data=buf, file_name=f"{ds_name}_comparison.csv", mime="text/csv")

        try:
            import io
            out = io.BytesIO()
            cmp_df.to_excel(out, index=False, sheet_name="Сравнение")
            out.seek(0)
            st.download_button("📥 Скачать таблицу (Excel)", data=out.read(),
                               file_name=f"{ds_name}_comparison.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception:
            pass

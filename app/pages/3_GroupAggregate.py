"""
pages/3_GroupAggregate.py – Group & Aggregate with pivot view, auto-charts and rich export.
Replaces Excel pivot tables + SUMIF/COUNTIF for bank employees.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df
from core.i18n import t
from core.audit import log_event
from app.components.ux import active_dataset_warnings
from app.styles import inject_all_css, page_header, section_header
from core.aggregate import (
    group_aggregate, pivot_view, available_agg_functions,
    to_csv_bytes, to_xlsx_bytes, to_parquet_bytes, TIME_BUCKET_MAP,
)

st.set_page_config(page_title="KIBAD – Группировка", layout="wide")
init_state()
inject_all_css()
active_dataset_warnings()

page_header("3. Группировка", "Сводные таблицы и агрегация данных", "📊")

ds_name = dataset_selectbox(label="Выберите датасет", key="ga_ds")
if not ds_name:
    st.stop()

df = get_active_df()
if df is None or df.empty:
    st.markdown('<div class="kibad-empty-state"><div class="kibad-empty-state-icon">📭</div><div class="kibad-empty-state-title">Нет данных</div><div class="kibad-empty-state-desc">Загрузите датасет на странице «Данные»</div></div>', unsafe_allow_html=True)
    st.stop()

_ga_c1, _ga_c2, _ga_c3 = st.columns([1, 1, 4])
_ga_c1.metric("Строк", f"{df.shape[0]:,}")
_ga_c2.metric("Столбцов", f"{df.shape[1]}")

all_cols = df.columns.tolist()
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in all_cols if c not in num_cols]
date_cols = [c for c in all_cols if pd.api.types.is_datetime64_any_dtype(df[c])]

# ---------------------------------------------------------------------------
# Configuration — main area (more discoverable than sidebar)
# ---------------------------------------------------------------------------
with st.expander("⚙️ Настройка группировки", expanded=True):
    col_g, col_m, col_a = st.columns(3)
    with col_g:
        st.markdown("**Группировать по:**")
        group_cols = st.multiselect(
            "Измерения (строки)",
            all_cols,
            default=cat_cols[:1] if cat_cols else [],
            key="ga_group",
            help="Аналог строк сводной таблицы Excel",
        )
    with col_m:
        st.markdown("**Числовые показатели:**")
        metric_cols = st.multiselect(
            "Метрики",
            num_cols,
            default=num_cols[:3] if num_cols else [],
            key="ga_metrics",
            help="Аналог значений сводной таблицы Excel",
        )
    with col_a:
        st.markdown("**Функции агрегации:**")
        agg_labels = {
            "sum": "Сумма", "mean": "Среднее", "count": "Количество",
            "median": "Медиана", "min": "Минимум", "max": "Максимум",
            "std": "Станд. откл.", "nunique": "Уникальных",
            "q25": "Квартиль 25%", "q75": "Квартиль 75%",
            "weighted_avg": "Взвешенное среднее",
        }
        agg_options = available_agg_functions()
        agg_display = [agg_labels.get(a, a) for a in agg_options]
        default_idxs = [i for i, a in enumerate(agg_options) if a in ("sum", "mean", "count")]
        chosen_display = st.multiselect(
            "Агрегаты",
            agg_display,
            default=[agg_display[i] for i in default_idxs],
            key="ga_aggs_display",
        )
        agg_rev = {v: k for k, v in agg_labels.items()}
        agg_funcs = [agg_rev.get(d, d) for d in chosen_display]

    # Second row: optional features
    opt1, opt2, opt3 = st.columns(3)
    with opt1:
        use_time_bucket = st.checkbox("🗓️ Разбивка по времени", value=bool(date_cols), key="ga_tb")
        date_col = None
        time_bucket = None
        if use_time_bucket and date_cols:
            date_col = st.selectbox("Столбец даты", date_cols, key="ga_date_col")
            bucket_labels = {
                "День": "День", "Неделя": "Неделя", "Месяц": "Месяц",
                "Квартал": "Квартал", "Год": "Год",
            }
            tb_display = list(TIME_BUCKET_MAP.keys())
            tb_display_ru = {
                "День": "День", "Неделя": "Неделя", "Месяц": "Месяц",
                "Квартал": "Квартал", "Год": "Год",
            }
            time_bucket = st.selectbox(
                "Период",
                tb_display,
                index=tb_display.index("Месяц") if "Месяц" in tb_display else 0,
                key="ga_tb_type",
            )
    with opt2:
        use_bins = st.checkbox("📊 Разбивка по числовым бинам", value=False, key="ga_bins")
        bin_col = None
        bin_edges = None
        n_quantiles = None
        if use_bins and num_cols:
            bin_col = st.selectbox("Колонка для бинов", num_cols, key="ga_bin_col")
            bin_mode = st.radio("Тип бинов", ["Квантили", "Ручные"], key="ga_bin_mode", horizontal=True)
            if bin_mode == "Квантили":
                n_quantiles = st.slider("Количество бинов", 2, 10, 4, key="ga_nq")
            else:
                edges_str = st.text_input("Границы бинов (через запятую)", "0,100,500,1000,5000", key="ga_edges")
                try:
                    bin_edges = [float(x.strip()) for x in edges_str.split(",")]
                except ValueError:
                    st.warning("Неверный формат границ бинов.")
    with opt3:
        weight_col = None
        if "weighted_avg" in agg_funcs:
            st.markdown("**Веса для взвешенного среднего:**")
            weight_col = st.selectbox("Столбец весов", num_cols, key="ga_weight")
        # Top N filter
        top_n = st.number_input("Top N строк в результате (0 = все)", min_value=0, max_value=10000, value=0, step=10, key="ga_topn")

# ---------------------------------------------------------------------------
# Run aggregation
# ---------------------------------------------------------------------------
run_col, hint_col = st.columns([1, 4])
with run_col:
    run_btn = st.button("▶ Рассчитать", key="btn_agg", type="primary", use_container_width=True)
with hint_col:
    if not group_cols:
        st.info("💡 Выберите хотя бы один столбец в «Группировать по»")
    elif not metric_cols:
        st.info("💡 Выберите хотя бы один числовой показатель")

if run_btn:
    if not group_cols:
        st.error("Выберите хотя бы одно измерение (столбец для группировки).")
        st.stop()
    if not metric_cols:
        st.error("Выберите хотя бы один числовой показатель.")
        st.stop()
    if not agg_funcs:
        st.error("Выберите хотя бы одну функцию агрегации.")
        st.stop()

    with st.spinner("Выполняем группировку..."):
        try:
            result = group_aggregate(
                df,
                group_cols=group_cols,
                metric_cols=metric_cols,
                agg_funcs=agg_funcs,
                date_col=date_col,
                time_bucket=time_bucket,
                numeric_bin_col=bin_col,
                numeric_bin_edges=bin_edges,
                numeric_n_quantiles=n_quantiles,
                weight_col=weight_col,
            )
            # Apply Top N filter
            if top_n and top_n > 0:
                result = result.head(int(top_n))

            st.session_state["aggregate_results"][ds_name] = result
            log_event("analysis_run", {
                "type": "group_aggregate",
                "dataset": ds_name,
                "group_cols": group_cols,
                "metric_cols": metric_cols,
                "agg_funcs": agg_funcs,
                "rows_result": len(result),
            })
            st.success(f"Готово! Получено {len(result):,} строк.")
        except Exception as e:
            st.error(f"Ошибка агрегации: {e}")
            st.stop()

# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
result_df = st.session_state.get("aggregate_results", {}).get(ds_name)
if result_df is not None and not result_df.empty:
    tabs = st.tabs(["📋 Таблица", "📊 Графики", "🔄 Сводная таблица", "💾 Экспорт"])

    # ---- TAB 1: Table ----
    with tabs[0]:
        # Sort control
        sort_col_options = result_df.columns.tolist()
        sc1, sc2 = st.columns([3, 1])
        with sc1:
            sort_col = st.selectbox("Сортировать по:", sort_col_options,
                                    index=len(sort_col_options) - 1, key="ga_sort_col")
        with sc2:
            sort_asc = st.radio("Порядок:", ["↓ По убыванию", "↑ По возрастанию"],
                                key="ga_sort_dir", horizontal=True) == "↑ По возрастанию"

        display_df = result_df.sort_values(sort_col, ascending=sort_asc)

        # Styled dataframe with totals row
        numeric_result_cols = display_df.select_dtypes(include="number").columns.tolist()
        totals = {c: display_df[c].sum() for c in numeric_result_cols}
        totals_row = pd.DataFrame([{**{c: "ИТОГО" for c in group_cols}, **totals}])

        st.dataframe(display_df, use_container_width=True, height=420)

        # Totals summary below table
        if numeric_result_cols:
            st.markdown("**Итого по числовым столбцам:**")
            tot_cols = st.columns(min(len(numeric_result_cols), 5))
            for i, col_name in enumerate(numeric_result_cols[:5]):
                val = display_df[col_name].sum()
                with tot_cols[i]:
                    st.metric(col_name, f"{val:,.2f}".replace(",", " "))

    # ---- TAB 2: Charts (auto-generated) ----
    with tabs[1]:
        section_header("Автоматическая визуализация", "📊")

        result_num_cols = result_df.select_dtypes(include="number").columns.tolist()

        if not result_num_cols:
            st.info("Нет числовых колонок для построения графика.")
        else:
            cv1, cv2, cv3 = st.columns(3)
            with cv1:
                chart_type = st.selectbox(
                    "Тип графика:",
                    ["Столбчатый", "Горизонтальный столбчатый", "Линейный", "Площадной", "Точечный"],
                    key="ga_chart_type",
                )
            with cv2:
                x_col = st.selectbox("Ось X:", result_df.columns.tolist(), key="ga_x")
            with cv3:
                y_col = st.selectbox("Ось Y (метрика):", result_num_cols,
                                     index=0, key="ga_y")

            color_col = None
            if len(group_cols) > 1:
                color_col_options = [c for c in group_cols if c != x_col]
                if color_col_options:
                    color_col = st.selectbox("Цвет (группировка):", ["— нет —"] + color_col_options, key="ga_color")
                    if color_col == "— нет —":
                        color_col = None

            # Sort for chart
            chart_df = result_df.sort_values(y_col, ascending=False)

            chart_kwargs = dict(
                x=x_col, y=y_col, color=color_col,
                title=f"{y_col} по {x_col}",
                labels={x_col: x_col, y_col: y_col},
                template="plotly_white",
            )

            try:
                if chart_type == "Столбчатый":
                    fig = px.bar(chart_df, **chart_kwargs, barmode="group")
                elif chart_type == "Горизонтальный столбчатый":
                    fig = px.bar(chart_df, x=y_col, y=x_col, color=color_col,
                                 orientation="h", title=f"{y_col} по {x_col}",
                                 template="plotly_white")
                elif chart_type == "Линейный":
                    fig = px.line(chart_df, **chart_kwargs, markers=True)
                elif chart_type == "Площадной":
                    fig = px.area(chart_df, **chart_kwargs)
                elif chart_type == "Точечный":
                    size_col = None
                    if len(result_num_cols) > 1:
                        size_col = [c for c in result_num_cols if c != y_col][0]
                    fig = px.scatter(chart_df, x=x_col, y=y_col, color=color_col,
                                     size=size_col, title=f"{y_col} по {x_col}",
                                     template="plotly_white")
                else:
                    fig = px.bar(chart_df, **chart_kwargs)

                fig.update_layout(height=500, margin=dict(t=50, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # Download chart
                try:
                    img_bytes = fig.to_image(format="png", width=1200, height=600)
                    st.download_button(
                        "📷 Скачать график (PNG)",
                        data=img_bytes,
                        file_name=f"{ds_name}_chart.png",
                        mime="image/png",
                    )
                except Exception:
                    pass  # kaleido not installed

            except Exception as e:
                st.error(f"Ошибка построения графика: {e}")

            # Auto-insight
            if not result_df.empty and y_col in result_df.columns:
                max_row = result_df.loc[result_df[y_col].idxmax()]
                min_row = result_df.loc[result_df[y_col].idxmin()]
                total_val = result_df[y_col].sum()
                max_label = str(max_row[x_col]) if x_col in max_row.index else "?"
                min_label = str(min_row[x_col]) if x_col in min_row.index else "?"

                with st.expander("💡 Автоматические выводы"):
                    st.markdown(
                        f"- **Максимум:** {max_label} — {max_row[y_col]:,.2f}".replace(",", " ")
                    )
                    st.markdown(
                        f"- **Минимум:** {min_label} — {min_row[y_col]:,.2f}".replace(",", " ")
                    )
                    if total_val != 0:
                        top_share = max_row[y_col] / total_val * 100
                        st.markdown(f"- **Доля лидера:** {top_share:.1f}% от общего итога")
                    n_groups = len(result_df)
                    st.markdown(f"- **Всего групп:** {n_groups}")

    # ---- TAB 3: Pivot View ----
    with tabs[2]:
        section_header("Сводная таблица (Pivot)", "🔄")
        st.caption("Разверните строки в столбцы — как в Excel «Сводная таблица»")
        pivot_cols = result_df.columns.tolist()

        if len(pivot_cols) < 3:
            st.info("Для сводной таблицы нужно минимум 3 колонки в результате (строки, столбцы, значения).")
        else:
            pc1, pc2, pc3 = st.columns(3)
            with pc1:
                pv_index = st.selectbox("Строки (Index):", pivot_cols, key="pv_idx",
                                        help="Что будет в строках сводной таблицы")
            with pc2:
                pv_columns = st.selectbox(
                    "Столбцы (Columns):",
                    [c for c in pivot_cols if c != pv_index],
                    key="pv_col",
                    help="Что будет в заголовках столбцов",
                )
            with pc3:
                val_cols = [c for c in pivot_cols if c not in [pv_index, pv_columns]]
                pv_values = st.selectbox("Значения:", val_cols, key="pv_val") if val_cols else None

            if pv_values and st.button("Построить сводную таблицу", key="btn_pivot", type="primary"):
                try:
                    pvt = pivot_view(result_df, pv_index, pv_columns, pv_values)
                    st.dataframe(pvt, use_container_width=True)

                    dl1, dl2 = st.columns(2)
                    with dl1:
                        st.download_button(
                            "📥 Скачать сводную таблицу (CSV)",
                            data=to_csv_bytes(pvt.reset_index()),
                            file_name=f"{ds_name}_pivot.csv",
                            mime="text/csv",
                        )
                    with dl2:
                        st.download_button(
                            "📥 Скачать сводную таблицу (Excel)",
                            data=to_xlsx_bytes(pvt.reset_index()),
                            file_name=f"{ds_name}_pivot.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                except Exception as e:
                    st.error(f"Ошибка построения сводной таблицы: {e}")

    # ---- TAB 4: Export ----
    with tabs[3]:
        section_header("Экспорт результатов", "💾")

        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.download_button(
                "📥 Скачать CSV",
                data=to_csv_bytes(result_df),
                file_name=f"{ds_name}_aggregated.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp_col2:
            st.download_button(
                "📥 Скачать Excel (.xlsx)",
                data=to_xlsx_bytes(result_df),
                file_name=f"{ds_name}_aggregated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with exp_col3:
            try:
                st.download_button(
                    "📥 Скачать Parquet",
                    data=to_parquet_bytes(result_df),
                    file_name=f"{ds_name}_aggregated.parquet",
                    mime="application/octet-stream",
                    use_container_width=True,
                )
            except Exception:
                st.info("Parquet недоступен (установите pyarrow).")

        # Save as new dataset
        st.divider()
        st.markdown("**Сохранить результат как новый датасет:**")
        new_ds_name = st.text_input("Название нового датасета:", value=f"{ds_name}_grouped", key="ga_new_ds_name")
        if st.button("💾 Сохранить в рабочую область", key="btn_save_agg"):
            from app.state import add_dataset
            add_dataset(new_ds_name, result_df.reset_index(drop=True))
            st.success(f"Датасет «{new_ds_name}» сохранён. Теперь его можно использовать на других страницах.")

elif run_btn:
    st.warning("Результат пустой. Проверьте настройки группировки.")

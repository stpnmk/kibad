"""
pages/4_Merge.py – Table merge, join, and concatenation with pitfall detection.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, list_dataset_names, get_dataset
from app.styles import inject_all_css, page_header, section_header
from core.merge import (
    merge_tables, concat_tables, analyze_key_cardinality,
    MergeWarning,
)
from core.audit import log_event

st.set_page_config(page_title="KIBAD – Merge", layout="wide")
init_state()
inject_all_css()

page_header("4. Объединение таблиц", "JOIN, UNION и конкатенация с диагностикой", "🔗")

all_ds = list_dataset_names()
if len(all_ds) < 1:
    st.markdown('<div class="kibad-empty-state"><div class="kibad-empty-state-icon">📭</div><div class="kibad-empty-state-title">Нет данных</div><div class="kibad-empty-state-desc">Загрузите хотя бы один датасет на странице «Данные»</div></div>', unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------------------
# Tabs: Join | Concat
# ---------------------------------------------------------------------------
tab_join, tab_concat = st.tabs(["🔗 Join (слияние по ключу)", "📋 Concat (добавление строк/столбцов)"])


# ===========================================================================
# TAB: JOIN
# ===========================================================================
with tab_join:
    section_header("Слияние по ключевым колонкам", "🔗")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Левая таблица (Left)**")
        left_ds = st.selectbox("Датасет (Left)", all_ds, key="merge_left_ds")
        left_df = get_dataset(left_ds)
        if left_df is not None:
            st.caption(f"{left_df.shape[0]:,} строк × {left_df.shape[1]} колонок")
            st.dataframe(left_df.head(3), use_container_width=True)

    with col_r:
        st.markdown("**Правая таблица (Right)**")
        default_right_idx = min(1, len(all_ds) - 1)
        right_ds = st.selectbox("Датасет (Right)", all_ds, index=default_right_idx, key="merge_right_ds")
        right_df = get_dataset(right_ds)
        if right_df is not None:
            st.caption(f"{right_df.shape[0]:,} строк × {right_df.shape[1]} колонок")
            st.dataframe(right_df.head(3), use_container_width=True)

    if left_df is None or right_df is None:
        st.error("Не удалось загрузить датасеты.")
        st.stop()

    st.divider()

    # Key column selection
    col_keys, col_opts = st.columns([2, 1])
    with col_keys:
        st.markdown("**Ключевые колонки**")
        key_mode = st.radio("Режим выбора ключей", ["Одна колонка", "Несколько колонок"], horizontal=True, key="merge_key_mode")
        if key_mode == "Одна колонка":
            left_key = st.selectbox("Левый ключ", left_df.columns.tolist(), key="merge_lk")
            right_key = st.selectbox("Правый ключ", right_df.columns.tolist(), key="merge_rk",
                                     index=right_df.columns.tolist().index(left_key)
                                     if left_key in right_df.columns else 0)
            left_keys = [left_key]
            right_keys = [right_key]
        else:
            left_keys = st.multiselect("Левые ключи", left_df.columns.tolist(), key="merge_lks")
            right_keys = st.multiselect("Правые ключи", right_df.columns.tolist(), key="merge_rks")

    with col_opts:
        st.markdown("**Параметры**")
        join_how = st.selectbox(
            "Тип объединения",
            ["left", "inner", "outer", "right", "cross"],
            key="merge_how",
            help="left — все строки левой; inner — только совпадения; outer — все строки обеих; right — все строки правой; cross — декартово произведение",
        )
        suffixes_x = st.text_input("Суффикс для дублей (left)", "_x", key="merge_sfx_x")
        suffixes_y = st.text_input("Суффикс для дублей (right)", "_y", key="merge_sfx_y")

    # Join type explanations
    _join_help = {
        "left": "**Left join** — все строки левой таблицы сохраняются. Правая присоединяется там, где ключ совпадает; остальные → NaN.",
        "inner": "**Inner join** — только строки с совпадающими ключами в обеих таблицах.",
        "outer": "**Full outer join** — все строки обеих таблиц. Несовпадения → NaN с обеих сторон.",
        "right": "**Right join** — все строки правой таблицы сохраняются. Симметрично left join.",
        "cross": "**Cross join** — декартово произведение: каждая строка левой × каждая строка правой. Результат = L × R строк!",
    }
    st.info(_join_help.get(join_how, ""))

    # Cardinality preview
    if left_keys and right_keys and len(left_keys) == len(right_keys):
        if all(k in left_df.columns for k in left_keys) and all(k in right_df.columns for k in right_keys):
            card = analyze_key_cardinality(left_df, right_df, left_keys, right_keys)
            join_type = card.get("join_type", "?")
            color_map = {"1:1": "success", "1:N": "info", "N:1": "info", "N:M": "warning"}
            level = color_map.get(join_type, "info")
            msg = f"**Кардинальность ключей: {join_type}** — {card.get('description', '')}"
            if level == "success":
                st.success(msg)
            elif level == "warning":
                st.warning(msg)
            else:
                st.info(msg)

            with st.expander("Детали кардинальности"):
                c1, c2 = st.columns(2)
                c1.metric("Уникальных ключей (left)", f"{card['left_unique_keys']:,} / {card['left_total_rows']:,}")
                c2.metric("Уникальных ключей (right)", f"{card['right_unique_keys']:,} / {card['right_total_rows']:,}")

    # Run merge
    st.divider()
    can_run = (
        len(left_keys) > 0 and len(right_keys) > 0 and
        len(left_keys) == len(right_keys)
    ) or join_how == "cross"

    if not can_run and join_how != "cross":
        st.warning("Выберите одинаковое количество левых и правых ключей.")

    if st.button("▶ Выполнить объединение", type="primary", key="btn_merge", disabled=(not can_run and join_how != "cross")):
        with st.spinner("Объединяем таблицы..."):
            try:
                lk = [] if join_how == "cross" else left_keys
                rk = [] if join_how == "cross" else right_keys
                result = merge_tables(
                    left_df, right_df, lk, rk,
                    how=join_how,
                    suffixes=(suffixes_x, suffixes_y),
                )
                st.session_state["merge_result"] = result

                log_event("merge_applied", {
                    "left_ds": left_ds, "right_ds": right_ds,
                    "how": join_how, "left_keys": lk, "right_keys": rk,
                    "result_rows": result.result_rows,
                })
            except Exception as e:
                st.error(f"Ошибка при объединении: {e}")

    # Show result
    merge_result = st.session_state.get("merge_result")
    if merge_result is not None:
        section_header("Результат объединения", "📋")

        # Stats row
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Строк до (left)", f"{merge_result.left_rows:,}")
        mc2.metric("Строк до (right)", f"{merge_result.right_rows:,}")
        mc3.metric("Строк после", f"{merge_result.result_rows:,}",
                   delta=f"{merge_result.result_rows - merge_result.left_rows:+,}",
                   delta_color="inverse" if merge_result.result_rows > merge_result.left_rows * 2 else "normal")
        mc4.metric("Колонок", f"{merge_result.result_cols}")

        # Explosion ratio
        if merge_result.explosion_ratio > 2:
            st.error(
                f"⚠️ **Взрыв строк!** Результат в {merge_result.explosion_ratio:.1f}× больше исходных таблиц. "
                "Убедитесь, что ключи уникальны или используйте агрегацию перед слиянием."
            )

        # Warnings
        for w in merge_result.warnings:
            if w.level == "error":
                st.error(f"🔴 **{w.code}**: {w.message}")
            elif w.level == "warning":
                st.warning(f"🟡 **{w.code}**: {w.message}")
            else:
                st.info(f"🔵 **{w.code}**: {w.message}")

        if not merge_result.warnings:
            st.success("✅ Предупреждений нет. Объединение выглядит корректно.")

        # Preview
        st.dataframe(merge_result.df.head(100), use_container_width=True)

        # Save to session
        col_save, col_dl = st.columns(2)
        with col_save:
            save_name = st.text_input("Имя нового датасета", f"{left_ds}_merged_{right_ds}", key="merge_save_name")
            if st.button("💾 Сохранить как датасет", key="btn_merge_save"):
                from app.state import add_dataset
                add_dataset(save_name, merge_result.df)
                st.success(f"Датасет «{save_name}» добавлен. Используйте его на других страницах.")
                st.rerun()

        with col_dl:
            csv_bytes = merge_result.df.to_csv(index=False).encode()
            st.download_button("⬇ Скачать CSV", csv_bytes, file_name="merged_result.csv", mime="text/csv")

        # Overlap analysis chart
        with st.expander("📊 Визуализация: ключевое перекрытие"):
            if left_keys and join_how != "cross":
                try:
                    left_key_vals = set(map(tuple, left_df[left_keys].dropna().values.tolist()))
                    right_key_vals = set(map(tuple, right_df[right_keys].dropna().values.tolist()))
                    only_left = len(left_key_vals - right_key_vals)
                    only_right = len(right_key_vals - left_key_vals)
                    both = len(left_key_vals & right_key_vals)

                    fig_venn = go.Figure(go.Bar(
                        x=["Только в left", "Совпадение", "Только в right"],
                        y=[only_left, both, only_right],
                        marker_color=["#e74c3c", "#2ecc71", "#3498db"],
                        text=[only_left, both, only_right],
                        textposition="auto",
                    ))
                    fig_venn.update_layout(
                        title="Уникальные ключи: перекрытие left ∩ right",
                        yaxis_title="Количество уникальных ключей",
                        template="plotly_white",
                    )
                    st.plotly_chart(fig_venn, use_container_width=True)
                except Exception:
                    pass


# ===========================================================================
# TAB: CONCAT
# ===========================================================================
with tab_concat:
    section_header("Конкатенация (вертикальная или горизонтальная)", "📋")

    concat_axis = st.radio(
        "Направление",
        ["По строкам (axis=0, добавить строки)", "По колонкам (axis=1, добавить колонки)"],
        horizontal=True,
        key="concat_axis",
        help="axis=0: добавляет строки (датасеты должны иметь одинаковые колонки). axis=1: добавляет колонки (датасеты должны иметь одинаковое число строк).",
    )
    axis = 0 if "0" in concat_axis else 1

    if axis == 0:
        st.info(
            "**По строкам** — добавляет строки из выбранных датасетов. "
            "Схема должна совпадать (одинаковые колонки). Несовпадающие колонки заполняются NaN."
        )
    else:
        st.info(
            "**По колонкам** — добавляет новые колонки справа. "
            "Количество строк должно совпадать, иначе Pandas выровняет по индексу."
        )

    concat_dfs = st.multiselect(
        "Выберите датасеты для конкатенации (минимум 2)",
        all_ds,
        default=all_ds[:min(2, len(all_ds))],
        key="concat_ds_select",
    )
    ignore_idx = st.checkbox("Сбросить индекс (ignore_index=True)", value=True, key="concat_ignore_idx", help="Если включено — индекс результата будет сброшен и пронумерован с нуля. Рекомендуется при конкатенации по строкам.")

    if len(concat_dfs) < 2:
        st.warning("Выберите минимум 2 датасета.")
    else:
        # Schema preview
        with st.expander("Сравнение схем выбранных датасетов"):
            schema_data = []
            for ds_name in concat_dfs:
                ds = get_dataset(ds_name)
                if ds is not None:
                    for col in ds.columns:
                        schema_data.append({"Датасет": ds_name, "Колонка": col, "Тип": str(ds[col].dtype)})
            if schema_data:
                schema_df = pd.DataFrame(schema_data)
                pivot_schema = schema_df.pivot_table(
                    index="Колонка", columns="Датасет", values="Тип", aggfunc="first"
                ).fillna("—")
                st.dataframe(pivot_schema, use_container_width=True)

        if st.button("▶ Выполнить конкатенацию", type="primary", key="btn_concat"):
            dfs_to_concat = [get_dataset(n) for n in concat_dfs if get_dataset(n) is not None]
            result_df, warns = concat_tables(dfs_to_concat, axis=axis, ignore_index=ignore_idx)
            st.session_state["concat_result"] = (result_df, warns)
            log_event("concat_applied", {
                "datasets": concat_dfs, "axis": axis,
                "result_rows": len(result_df), "result_cols": result_df.shape[1],
            })

        concat_state = st.session_state.get("concat_result")
        if concat_state is not None:
            result_df, warns = concat_state

            mc1, mc2 = st.columns(2)
            mc1.metric("Строк результата", f"{len(result_df):,}")
            mc2.metric("Колонок результата", f"{result_df.shape[1]}")

            for w in warns:
                if w.level == "error":
                    st.error(f"🔴 **{w.code}**: {w.message}")
                elif w.level == "warning":
                    st.warning(f"🟡 **{w.code}**: {w.message}")
                else:
                    st.info(f"🔵 **{w.code}**: {w.message}")

            if not warns:
                st.success("✅ Конкатенация выполнена без предупреждений.")

            st.dataframe(result_df.head(100), use_container_width=True)

            cc1, cc2 = st.columns(2)
            with cc1:
                csave_name = st.text_input("Имя нового датасета", "_".join(concat_dfs[:2]) + "_concat", key="concat_save_name")
                if st.button("💾 Сохранить как датасет", key="btn_concat_save"):
                    from app.state import add_dataset
                    add_dataset(csave_name, result_df)
                    st.success(f"Датасет «{csave_name}» добавлен.")
                    st.rerun()
            with cc2:
                csv_bytes = result_df.to_csv(index=False).encode()
                st.download_button("⬇ Скачать CSV", csv_bytes, file_name="concat_result.csv", mime="text/csv")

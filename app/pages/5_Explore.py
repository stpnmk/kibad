"""
pages/5_Explore.py – Exploratory analysis: auto-insights, time series, distributions,
correlation, pairplot, pivot, waterfall, STL, KPIs, data profiling.
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

from app.state import init_state, dataset_selectbox, get_active_df, store_prepared
from core.interpret import interpret_correlation
from app.components.ux import interpretation_box, recommendation_card, queue_recommendation_notification, show_pending_notification
from core.explore import (
    plot_timeseries, plot_histogram, plot_boxplot, plot_violin,
    plot_correlation_heatmap,
    build_pivot, plot_pivot_bar, plot_waterfall, plot_stl_decomposition, compute_kpi,
)
from core.insights import analyze_dataset, format_insights_markdown, get_chart_recommendation, score_data_quality
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Explore", layout="wide")
init_state()
inject_all_css()
page_header("5. Анализ данных", "Авто-инсайты, распределения, корреляции и профилирование", "🔎")

chosen = dataset_selectbox("Dataset", key="explore_ds_sel")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.error("No prepared data found. Go to Prepare first.")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()
dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
all_cols = list(df.columns)

show_pending_notification()
st.markdown(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} cols")

tab_auto, tab_quality, tab_ts, tab_dist, tab_corr, tab_pairplot, tab_pivot, tab_waterfall, tab_stl, tab_kpi, tab_profile = st.tabs([
    "🤖 Авто-анализ", "🏥 Качество данных",
    "⏱️ Временные ряды", "📊 Распределения", "🔗 Корреляции", "🔀 Попарные графики", "📋 Сводная таблица", "🌊 Водопад", "📈 STL-декомпозиция", "🎯 KPI-треккер", "📋 Профиль данных",
])

# ---------------------------------------------------------------------------
# Auto-Insights
# ---------------------------------------------------------------------------
with tab_auto:
    section_header("Автоматический анализ датасета")
    st.markdown(
        "Движок авто-инсайтов сканирует датасет и выявляет ключевые паттерны, "
        "аномалии и рекомендации без какой-либо настройки."
    )

    # Cache insights per dataset name
    if "auto_insights" not in st.session_state:
        st.session_state["auto_insights"] = {}

    ds_name = st.session_state.get("active_ds", "")
    cached = st.session_state["auto_insights"].get(ds_name)

    col_refresh, _ = st.columns([1, 6])
    force_refresh = col_refresh.button("🔄 Пересчитать", key="btn_auto_refresh")

    if cached is None or force_refresh:
        with st.spinner("Анализируем датасет..."):
            insights = analyze_dataset(df)
            st.session_state["auto_insights"][ds_name] = insights
    else:
        insights = cached

    # Main markdown output
    md = format_insights_markdown(insights)
    st.markdown(md)

    st.divider()

    # Expandable: Correlation heatmap
    with st.expander("Корреляционная матрица", expanded=False):
        _ai_num_cols = df.select_dtypes(include="number").columns.tolist()
        if len(_ai_num_cols) >= 2:
            fig_ai_corr = plot_correlation_heatmap(
                df,
                columns=_ai_num_cols[:min(12, len(_ai_num_cols))],
                title="Корреляционная матрица",
            )
            st.plotly_chart(fig_ai_corr, use_container_width=True)
        else:
            st.info("Недостаточно числовых столбцов для матрицы корреляций.")

    # Expandable: Distribution mini-histograms grid
    with st.expander("Распределения числовых переменных", expanded=False):
        _ai_num_cols2 = df.select_dtypes(include="number").columns.tolist()
        if _ai_num_cols2:
            _ai_grid_cols = min(3, len(_ai_num_cols2))
            _ai_n_rows = (len(_ai_num_cols2) + _ai_grid_cols - 1) // _ai_grid_cols
            for _ai_ri in range(_ai_n_rows):
                _ai_cols_g = st.columns(_ai_grid_cols)
                for _ai_ci in range(_ai_grid_cols):
                    _ai_idx = _ai_ri * _ai_grid_cols + _ai_ci
                    if _ai_idx >= len(_ai_num_cols2):
                        break
                    _ai_cname = _ai_num_cols2[_ai_idx]
                    with _ai_cols_g[_ai_ci]:
                        _ai_s = df[_ai_cname].dropna()
                        if len(_ai_s) > 0:
                            _ai_fig = px.histogram(
                                _ai_s,
                                nbins=30,
                                title=_ai_cname,
                                labels={"value": _ai_cname, "count": ""},
                            )
                            _ai_fig.add_vline(
                                x=float(_ai_s.median()),
                                line_dash="dash",
                                line_color="#e74c3c",
                                opacity=0.8,
                                annotation_text=f"med={_ai_s.median():.2f}",
                            )
                            _ai_fig.update_layout(
                                template="plotly_white",
                                height=220,
                                margin=dict(l=10, r=10, t=30, b=10),
                                showlegend=False,
                            )
                            st.plotly_chart(_ai_fig, use_container_width=True)
        else:
            st.info("Числовых столбцов не обнаружено.")

    # Expandable: Top anomalies table
    with st.expander("Топ аномалии (IQR)", expanded=False):
        _ai_anomalies = insights.get("anomalies", [])
        if _ai_anomalies:
            st.dataframe(
                pd.DataFrame(_ai_anomalies),
                use_container_width=True,
            )
        else:
            st.success("Значительных аномалий не обнаружено.")

    # Recommendations section
    _ai_recs = insights.get("recommendations", [])
    if _ai_recs:
        st.divider()
        section_header("Рекомендуемые следующие шаги")
        for _ai_i, _ai_rec in enumerate(_ai_recs):
            _atype = _ai_rec.get("action_type", "info")
            _can_apply = _atype in ("fill_na", "dedup")
            _clicked = recommendation_card(
                action_label=_ai_rec["action"],
                reason=_ai_rec["reason"],
                priority=_ai_rec["priority"],
                on_apply=True if _can_apply else None,
                key=f"ai_rec_{_ai_i}",
            )
            if _clicked:
                _df_before = df.copy()
                if _atype == "fill_na":
                    _df_new = df.copy()
                    for _col in _df_new.select_dtypes(include="number").columns:
                        _df_new[_col] = _df_new[_col].fillna(_df_new[_col].median())
                    for _col in _df_new.select_dtypes(include="object").columns:
                        _mv = _df_new[_col].mode()
                        if len(_mv) > 0:
                            _df_new[_col] = _df_new[_col].fillna(_mv[0])
                elif _atype == "dedup":
                    _df_new = df.drop_duplicates().reset_index(drop=True)
                else:
                    _df_new = df
                if _df_new is not df:
                    store_prepared(chosen, _df_new)
                    st.session_state.get("quality_scores", {}).pop(chosen, None)
                    st.session_state.get("auto_insights", {}).pop(chosen, None)
                    st.session_state.get("data_quality_reports", {}).pop(chosen, None)
                    queue_recommendation_notification(_ai_rec["action"], _df_before, _df_new, chosen)
                    st.rerun()

    # Smart page links based on what was found
    st.divider()
    section_header("Следующие шаги")
    _ai_summary = insights.get("summary", {})
    _ai_corrs = insights.get("correlations", [])
    _ai_next_cols = st.columns(1)
    st.markdown("📊 **Постройте график** → перейдите на страницу **20. Конструктор графиков**")
    if _ai_corrs:
        st.markdown("🔗 **Изучите корреляции подробнее** — смотрите вкладку Корреляции выше")
    if _ai_summary.get("n_datetime", 0) > 0:
        st.page_link("pages/7_TimeSeries.py", label="⏱️ Анализ временного ряда → 7. Временные ряды")
    if _ai_summary.get("n_categorical", 0) > 0:
        st.markdown("👥 **Сегментация** → перейдите на страницу **12. Кластеризация**")
    if _ai_summary.get("n_numeric", 0) >= 2:
        st.page_link("pages/6_Tests.py", label="🔬 Сравните группы → 6. Статистические тесты")

    # Download auto-analysis report
    st.divider()
    report_text = format_insights_markdown(insights)
    st.download_button(
        "⬇ Скачать отчёт (Markdown)",
        data=report_text.encode("utf-8"),
        file_name=f"auto_analysis_{ds_name}.md",
        mime="text/markdown",
    )


# ---------------------------------------------------------------------------
# Data Quality
# ---------------------------------------------------------------------------
with tab_quality:
    section_header("Качество данных")
    st.markdown(
        "Комплексная оценка качества датасета по четырём измерениям: "
        "полнота, уникальность, согласованность и общий балл."
    )

    # Cache quality scores per dataset
    if "quality_scores" not in st.session_state:
        st.session_state["quality_scores"] = {}

    _qs_ds_name = chosen  # use chosen (from selectbox), not active_ds, to avoid stale cache
    _qs_cached = st.session_state["quality_scores"].get(_qs_ds_name)

    _qs_col_refresh, _ = st.columns([1, 6])
    _qs_force = _qs_col_refresh.button("🔄 Пересчитать", key="btn_quality_refresh")

    if _qs_cached is None or _qs_force:
        with st.spinner("Оцениваем качество данных..."):
            _qs = score_data_quality(df)
            st.session_state["quality_scores"][_qs_ds_name] = _qs
    else:
        _qs = _qs_cached

    # Overall score with colored badge
    _qs_overall = _qs["overall"]
    if _qs_overall >= 80:
        _qs_color = "🟢"
        _qs_label = "Отлично"
    elif _qs_overall >= 60:
        _qs_color = "🟡"
        _qs_label = "Удовлетворительно"
    else:
        _qs_color = "🔴"
        _qs_label = "Требует внимания"

    st.metric(
        label=f"Общий балл качества  {_qs_color} {_qs_label}",
        value=f"{_qs_overall:.1f} / 100",
    )

    st.divider()

    # Sub-score cards
    _qs_c1, _qs_c2, _qs_c3 = st.columns(3)
    with _qs_c1:
        st.metric("Полнота", f"{_qs['completeness']:.1f}")
        st.caption("Доля заполненных значений")
    with _qs_c2:
        st.metric("Уникальность", f"{_qs['uniqueness']:.1f}")
        st.caption("Отсутствие дублирующихся строк")
    with _qs_c3:
        st.metric("Согласованность", f"{_qs['consistency']:.1f}")
        st.caption("Отсутствие константных и смешанных столбцов")

    # Issues list grouped by level
    _qs_issues = _qs.get("issues", [])
    if _qs_issues:
        st.divider()
        section_header("Обнаруженные проблемы")

        for _qs_level, _qs_icon, _qs_level_label in [
            ("error", "❌", "Ошибки"),
            ("warning", "⚠️", "Предупреждения"),
            ("info", "ℹ️", "Информация"),
        ]:
            _qs_level_issues = [i for i in _qs_issues if i["level"] == _qs_level]
            if not _qs_level_issues:
                continue
            st.markdown(f"**{_qs_icon} {_qs_level_label}**")
            for _qs_issue in _qs_level_issues:
                _qs_col_part = (
                    f" — столбец `{_qs_issue['col']}`" if _qs_issue.get("col") else ""
                )
                if _qs_level == "error":
                    st.error(f"{_qs_icon} {_qs_issue['message']}{_qs_col_part}")
                elif _qs_level == "warning":
                    st.warning(f"{_qs_icon} {_qs_issue['message']}{_qs_col_part}")
                else:
                    st.info(f"{_qs_icon} {_qs_issue['message']}{_qs_col_part}")
    else:
        st.success("Проблем с качеством данных не обнаружено.")

    # Auto-fix suggestions
    st.divider()
    section_header("🔧 Авто-исправления")
    _qs_has_missing = any(i["level"] in ("error", "warning") and "пропущен" in i["message"] for i in _qs_issues)
    _qs_has_dups = any("дублир" in i["message"] for i in _qs_issues)

    dupes_count = df.duplicated().sum()
    total_nulls = int(df.isnull().sum().sum())

    if dupes_count > 0:
        st.warning(f"Обнаружено {dupes_count:,} дублирующихся строк.")
        if st.button("⚡ Удалить дубликаты", key="qc_dedup"):
            df_before = df.copy()
            df_new = df.drop_duplicates().reset_index(drop=True)
            store_prepared(chosen, df_new)
            # invalidate quality and auto-analysis caches so next render is fresh
            st.session_state.get("quality_scores", {}).pop(chosen, None)
            st.session_state.get("auto_insights", {}).pop(chosen, None)
            queue_recommendation_notification("Удаление дубликатов", df_before, df_new, chosen)
            st.rerun()

    if total_nulls > 0:
        st.warning(f"Обнаружено {total_nulls:,} пропущенных значений.")
        if st.button("⚡ Заполнить пропуски (медиана/мода)", key="qc_fill_na"):
            df_before = df.copy()
            df_new = df.copy()
            for col in df_new.select_dtypes(include="number").columns:
                df_new[col] = df_new[col].fillna(df_new[col].median())
            for col in df_new.select_dtypes(include="object").columns:
                mode_val = df_new[col].mode()
                if len(mode_val) > 0:
                    df_new[col] = df_new[col].fillna(mode_val[0])
            store_prepared(chosen, df_new)
            # invalidate quality and auto-analysis caches
            st.session_state.get("quality_scores", {}).pop(chosen, None)
            st.session_state.get("auto_insights", {}).pop(chosen, None)
            queue_recommendation_notification("Заполнение пропусков", df_before, df_new, chosen)
            st.rerun()

    if _qs_has_missing:
        st.page_link("pages/2_Prepare.py", label="🩹 Заполнить пропуски → 2. Подготовка данных (Импутация)")
    if _qs_has_dups:
        st.page_link("pages/2_Prepare.py", label="🗑️ Удалить дубликаты → 2. Подготовка данных (Дедупликация)")
    if not _qs_has_missing and not _qs_has_dups and dupes_count == 0 and total_nulls == 0:
        st.info("Автоматических исправлений не требуется.")


# ---------------------------------------------------------------------------
# Time Series
# ---------------------------------------------------------------------------
with tab_ts:
    section_header("График временного ряда")
    if not dt_cols:
        st.warning("Столбцы с датами не найдены. Распознайте дату в разделе **Подготовка**.")
    else:
        ts_date = st.selectbox("Столбец даты", dt_cols, key="ts_date")
        ts_vals = st.multiselect("Столбцы значений", num_cols, default=num_cols[:1] if num_cols else [], key="ts_vals")
        ts_color = st.selectbox("Цвет / сегмент (опционально)", ["(none)"] + cat_cols, key="ts_color")
        color = None if ts_color == "(none)" else ts_color

        if ts_vals:
            # Smart chart hint for first value column vs date
            _ts_hint_chart, _ts_hint_reason = get_chart_recommendation(df, ts_date, ts_vals[0])
            st.info(f"💡 Рекомендуем: **{_ts_hint_chart}** — {_ts_hint_reason}")
            fig = plot_timeseries(df, ts_date, ts_vals, color_col=color,
                                  title=f"{', '.join(ts_vals)} over time")
            st.plotly_chart(fig, use_container_width=True)
            img_bytes = fig.to_image(format="png") if hasattr(fig, "to_image") else b""
            if img_bytes:
                st.download_button("Скачать PNG", img_bytes, file_name="timeseries.png", mime="image/png")
        else:
            st.info("Выберите хотя бы один столбец значений.")

# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------
with tab_dist:
    section_header("Распределения")
    dist_tab_hist, dist_tab_box, dist_tab_violin = st.tabs(["Гистограмма + KDE", "Ящик с усами", "Скрипичный график"])

    with dist_tab_hist:
        hist_col = st.selectbox("Столбец", num_cols, key="hist_col") if num_cols else None
        if hist_col:
            hist_color = st.selectbox("Цвет по", ["(none)"] + cat_cols, key="hist_color")
            _color_for_hint = None if hist_color == "(none)" else hist_color
            # Smart chart hint
            _hint_chart, _hint_reason = get_chart_recommendation(df, hist_col, _color_for_hint)
            st.info(f"💡 Рекомендуем: **{_hint_chart}** — {_hint_reason}")
            hist_bins = st.slider("Интервалы (bins)", 5, 100, 30, key="hist_bins")
            kde = st.checkbox("Показать KDE", value=True, key="hist_kde")
            color = _color_for_hint
            fig = plot_histogram(df, hist_col, bins=hist_bins, color_col=color, kde=kde)
            st.plotly_chart(fig, use_container_width=True)

    with dist_tab_box:
        box_col = st.selectbox("Столбец значений", num_cols, key="box_col") if num_cols else None
        if box_col:
            box_group = st.selectbox("Группировка", ["(none)"] + cat_cols, key="box_group")
            grp = None if box_group == "(none)" else box_group
            # Smart chart hint
            _box_hint_chart, _box_hint_reason = get_chart_recommendation(df, box_col, grp)
            st.info(f"💡 Рекомендуем: **{_box_hint_chart}** — {_box_hint_reason}")
            fig = plot_boxplot(df, box_col, group_col=grp)
            st.plotly_chart(fig, use_container_width=True)

    with dist_tab_violin:
        st.markdown("#### Скрипичный график (Violin)")
        st.caption("Показывает форму распределения — лучше ящика с усами при наличии нескольких мод")
        if not num_cols:
            st.warning("Нет числовых столбцов.")
        else:
            vv1, vv2 = st.columns(2)
            with vv1:
                violin_col = st.selectbox("Числовой столбец:", num_cols, key="violin_col")
            with vv2:
                violin_group = st.selectbox("Группировка (опционально):", ["— нет —"] + cat_cols, key="violin_group")
            if st.button("Построить Violin", key="btn_violin"):
                group_col = None if violin_group == "— нет —" else violin_group
                fig_violin = plot_violin(df, violin_col, group_col)
                st.plotly_chart(fig_violin, use_container_width=True)
                with st.expander("Как читать violin plot"):
                    st.markdown(
                        "Ширина скрипки = плотность данных. Широкое место = много значений. Узкое = мало. "
                        "Вложенный ящик с усами показывает медиану и квартили."
                    )

# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
with tab_corr:
    section_header("Тепловая карта корреляций")
    if len(num_cols) < 2:
        st.warning("Необходимо минимум 2 числовых столбца.")
    else:
        with st.expander("ℹ️ Как читать матрицу корреляций"):
            st.markdown("""
    - **r > 0.7** (тёмно-красный): Сильная положительная связь — оба показателя растут вместе.
    - **r < -0.7** (тёмно-синий): Сильная отрицательная связь — рост одного → падение другого.
    - **|r| < 0.3**: Слабая связь — можно не учитывать при анализе.
    - **Порог мультиколлинеарности:** |r| > 0.8 создаёт проблемы в регрессионных моделях.
    """)
        corr_cols = st.multiselect("Столбцы для включения", num_cols, default=num_cols[:min(8, len(num_cols))], key="corr_cols")
        corr_method = st.selectbox("Метод", ["pearson", "spearman", "kendall"], key="corr_method")
        if corr_cols and len(corr_cols) >= 2:
            fig = plot_correlation_heatmap(df, columns=corr_cols, method=corr_method)
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation of correlation matrix
            corr_matrix = df[corr_cols].corr(method=corr_method.lower())
            np.fill_diagonal(corr_matrix.values, 0)
            max_corr_idx = np.unravel_index(np.abs(corr_matrix.values).argmax(), corr_matrix.shape)
            r_max = corr_matrix.iloc[max_corr_idx[0], max_corr_idx[1]]
            c1_max = corr_matrix.columns[max_corr_idx[0]]
            c2_max = corr_matrix.columns[max_corr_idx[1]]

            strong_pairs = []
            for _i in range(len(corr_matrix.columns)):
                for _j in range(_i + 1, len(corr_matrix.columns)):
                    _r = corr_matrix.iloc[_i, _j]
                    if abs(_r) > 0.6:
                        strong_pairs.append((corr_matrix.columns[_i], corr_matrix.columns[_j], _r))

            interp_text = f"**Сильнейшая корреляция:** {c1_max} ↔ {c2_max} (r={r_max:.3f}).\n\n"
            interp_text += interpret_correlation(r_max) + "\n\n"
            if len(strong_pairs) > 1:
                interp_text += f"Всего сильных корреляций (|r| > 0.6): **{len(strong_pairs)}**.\n\n"
            if len(strong_pairs) > 0:
                interp_text += "⚠️ При высокой корреляции между признаками используйте Shapley или Ridge-регрессию для факторного анализа."

            interpretation_box("Что показывает матрица корреляций?", interp_text)

            # Lag correlation sub-tool
            with st.expander("Лаговая кросс-корреляция"):
                from core.tests import lag_correlation
                lcc_x = st.selectbox("X (опережающий)", num_cols, key="lcc_x")
                lcc_y = st.selectbox("Y (запаздывающий)", num_cols, index=min(1, len(num_cols)-1), key="lcc_y")
                lcc_max = st.slider("Макс. лаг", 1, 24, 12, key="lcc_max")
                lcc_method = st.selectbox("Метод", ["pearson", "spearman"], key="lcc_method")
                if st.button("Рассчитать лаговую корреляцию", key="btn_lcc"):
                    lcc_df = lag_correlation(df[lcc_x], df[lcc_y], max_lag=lcc_max, method=lcc_method)
                    import plotly.express as px
                    fig_lcc = px.bar(lcc_df, x="lag", y="correlation",
                                     color="p_value", color_continuous_scale="RdYlGn_r",
                                     title=f"Lag Correlation: {lcc_x} vs {lcc_y}")
                    st.plotly_chart(fig_lcc, use_container_width=True)
                    st.dataframe(lcc_df, use_container_width=True)
                    st.download_button("Скачать CSV", lcc_df.to_csv(index=False).encode(), file_name="lag_corr.csv")

# ---------------------------------------------------------------------------
# Pairplot / Scatter Matrix
# ---------------------------------------------------------------------------
with tab_pairplot:
    section_header("Pairplot — матрица диаграмм рассеивания")
    st.markdown(
        "Scatter matrix (pairplot) показывает попарные зависимости между числовыми переменными. "
        "На диагонали — гистограммы распределения, вне диагонали — точечные диаграммы."
    )
    if len(num_cols) < 2:
        st.warning("Нужно минимум 2 числовых колонки.")
    else:
        pp_cols = st.multiselect(
            "Колонки для pairplot",
            num_cols,
            default=num_cols[:min(5, len(num_cols))],
            key="pp_cols",
            help="Рекомендуется 3–7 колонок. Большее количество сильно замедляет рендеринг.",
        )
        pp_color = st.selectbox("Цвет по категории (опционально)", ["(нет)"] + cat_cols, key="pp_color")
        pp_diag = st.radio("Диагональ", ["histogram", "box"], horizontal=True, key="pp_diag")
        pp_opacity = st.slider("Прозрачность точек", 0.1, 1.0, 0.6, key="pp_opacity")

        if len(pp_cols) >= 2 and st.button("Построить Pairplot", type="primary", key="btn_pairplot"):
            import plotly.express as px
            color_col = None if pp_color == "(нет)" else pp_color
            try:
                plot_df = df[pp_cols + ([color_col] if color_col else [])].dropna()
                fig_pp = px.scatter_matrix(
                    plot_df,
                    dimensions=pp_cols,
                    color=color_col,
                    opacity=pp_opacity,
                    title=f"Pairplot: {', '.join(pp_cols)}",
                    labels={c: c for c in pp_cols},
                )
                fig_pp.update_traces(diagonal_visible=True, showupperhalf=True)

                if pp_diag == "histogram":
                    fig_pp.update_traces(diagonal_visible=True)

                fig_pp.update_layout(
                    template="plotly_white",
                    height=600 + 80 * max(0, len(pp_cols) - 3),
                )
                st.plotly_chart(fig_pp, use_container_width=True)

                # Strongest correlations table
                st.markdown("**Сильнейшие корреляции (|r| > 0.5)**")
                corr_matrix = plot_df[pp_cols].corr()
                pairs = []
                for i, c1 in enumerate(pp_cols):
                    for c2 in pp_cols[i+1:]:
                        r = corr_matrix.loc[c1, c2]
                        if abs(r) >= 0.5:
                            pairs.append({"Переменная 1": c1, "Переменная 2": c2,
                                          "Корреляция Пирсона": round(r, 3),
                                          "Сила": "Сильная" if abs(r) >= 0.7 else "Умеренная"})
                if pairs:
                    st.dataframe(pd.DataFrame(pairs).sort_values("Корреляция Пирсона", key=abs, ascending=False),
                                 use_container_width=True)
                else:
                    st.info("Нет пар с |r| > 0.5.")
            except Exception as e:
                st.error(f"Ошибка построения pairplot: {e}")

# ---------------------------------------------------------------------------
# Pivot aggregation builder
# ---------------------------------------------------------------------------
with tab_pivot:
    section_header("Построитель сводной таблицы")
    if not all_cols:
        st.warning("Нет данных.")
    else:
        piv_index = st.selectbox("Группировка по строкам (индекс)", all_cols, key="piv_index")
        piv_cols = st.selectbox("Группировка по столбцам (опционально)", ["(none)"] + all_cols, key="piv_cols")
        piv_value = st.selectbox("Столбец значений", num_cols, key="piv_value") if num_cols else None
        piv_agg = st.selectbox("Агрегация", ["sum", "mean", "count", "median", "min", "max"], key="piv_agg")

        if piv_value and st.button("Построить сводную таблицу", key="btn_pivot"):
            try:
                col_grp = None if piv_cols == "(none)" else piv_cols
                pivot_df = build_pivot(df, piv_index, col_grp, piv_value, agg_func=piv_agg)
                st.dataframe(pivot_df, use_container_width=True)
                csv_piv = pivot_df.to_csv(index=False).encode()
                st.download_button("Скачать сводную CSV", csv_piv, file_name="pivot.csv")

                # Bar chart
                val_cols_chart = [c for c in pivot_df.columns if c != piv_index]
                barmode = st.radio("Режим столбцов", ["group", "stack"], horizontal=True, key="piv_barmode")
                if val_cols_chart:
                    fig = plot_pivot_bar(pivot_df, piv_index, val_cols_chart, barmode=barmode)
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Ошибка сводной таблицы: {e}")

# ---------------------------------------------------------------------------
# Waterfall chart
# ---------------------------------------------------------------------------
with tab_waterfall:
    section_header("Водопад — вклад факторов")
    st.markdown(
        "Введите названия факторов и их числовые вклады (дельты). "
        "Положительный = рост, отрицательный = снижение."
    )
    n_factors = st.slider("Количество факторов", 2, 12, 4, key="wf_n")
    factor_names, factor_vals = [], []
    for i in range(n_factors):
        c1, c2 = st.columns(2)
        name = c1.text_input(f"Название фактора {i+1}", value=f"Фактор {i+1}", key=f"wf_name_{i}")
        val = c2.number_input(f"Значение фактора {i+1}", value=0.0, key=f"wf_val_{i}")
        factor_names.append(name)
        factor_vals.append(val)

    wf_base = st.text_input("Начальная метка", "Базис", key="wf_base")
    wf_end = st.text_input("Конечная метка", "Результат", key="wf_end")

    if st.button("Построить водопад", key="btn_waterfall"):
        fig = plot_waterfall(factor_names, factor_vals, base_label=wf_base, total_label=wf_end)
        st.plotly_chart(fig, use_container_width=True)

    # Auto waterfall from column deltas
    st.divider()
    st.markdown("**Авто-водопад** — вычислить дельты период-к-периоду по числовым столбцам.")
    if num_cols and dt_cols:
        aw_date = st.selectbox("Столбец даты", dt_cols, key="aw_date")
        aw_cols = st.multiselect("Столбцы значений", num_cols, default=num_cols[:min(4, len(num_cols))], key="aw_cols")
        if aw_cols and st.button("Авто-водопад (последний vs предыдущий период)", key="btn_auto_wf"):
            sorted_df = df.sort_values(aw_date).dropna(subset=aw_cols)
            if len(sorted_df) >= 2:
                last = sorted_df[aw_cols].iloc[-1]
                prev = sorted_df[aw_cols].iloc[-2]
                deltas = (last - prev).tolist()
                fig = plot_waterfall(aw_cols, deltas, title="Last Period vs Previous",
                                     base_label="Previous", total_label="Difference")
                st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# STL Decomposition
# ---------------------------------------------------------------------------
with tab_stl:
    section_header("STL-декомпозиция сезонности")
    if not dt_cols:
        st.warning("Столбцы с датами не найдены. Распознайте дату в разделе Подготовка.")
    elif not num_cols:
        st.warning("Числовых столбцов нет.")
    else:
        stl_date = st.selectbox("Столбец даты", dt_cols, key="stl_date")
        stl_target = st.selectbox("Целевой столбец", num_cols, key="stl_target")
        stl_period = st.slider("Сезонный период", 2, 52, 12, key="stl_period",
                               help="12 = ежемесячный годовой, 52 = еженедельный годовой, 4 = квартальный")
        if st.button("Декомпозировать", key="btn_stl"):
            with st.spinner("Выполняем STL..."):
                try:
                    sub = df[[stl_date, stl_target]].dropna().sort_values(stl_date)
                    fig = plot_stl_decomposition(sub, stl_date, stl_target, period=stl_period)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Ошибка STL: {e}")

# ---------------------------------------------------------------------------
# KPI Builder
# ---------------------------------------------------------------------------
with tab_kpi:
    section_header("KPI-конструктор")
    st.markdown(
        "Задавайте формулы KPI, используя имена столбцов как переменные. Пример: `revenue / sessions`"
    )
    kpi_defs = st.session_state.get("kpi_defs", [])
    with st.form("kpi_form"):
        kpi_label = st.text_input("Название KPI", "Мой KPI")
        kpi_formula = st.text_input("Формула (выражение Python)", "col_a / col_b")
        add_kpi = st.form_submit_button("Добавить KPI")
    if add_kpi:
        kpi_defs.append({"label": kpi_label, "formula": kpi_formula})
        st.session_state["kpi_defs"] = kpi_defs
        st.success(f"KPI '{kpi_label}' добавлен.")

    if kpi_defs:
        section_header("KPI-дашборд")
        kpi_cols = st.columns(min(len(kpi_defs), 4))
        for i, kd in enumerate(kpi_defs):
            try:
                res = compute_kpi(df, kd["formula"], label=kd["label"])
                with kpi_cols[i % len(kpi_cols)]:
                    st.metric(
                        label=res["label"],
                        value=f"{res['last']:,.3f}",
                        delta=f"{res['pct_change']:+.1f}%",
                    )
            except Exception as e:
                st.warning(f"Ошибка KPI '{kd['label']}': {e}")

        # Remove KPI
        rem_kpi = st.selectbox("Удалить KPI", ["—"] + [k["label"] for k in kpi_defs], key="rem_kpi")
        if rem_kpi != "—" and st.button("Удалить", key="btn_rem_kpi"):
            st.session_state["kpi_defs"] = [k for k in kpi_defs if k["label"] != rem_kpi]
            st.rerun()

# ---------------------------------------------------------------------------
# Data Profiling
# ---------------------------------------------------------------------------
with tab_profile:
    import plotly.express as px
    import numpy as np

    section_header("📋 Автоматический профиль данных")
    st.markdown(
        "Быстрый обзор качества и структуры датасета: пропуски, дубликаты, "
        "выбросы, распределения, уникальность — всё в одном месте."
    )

    if st.button("🔍 Сгенерировать профиль", type="primary", key="btn_profile"):
        with st.spinner("Анализируем данные..."):

            # --- Basic stats ---
            n_rows, n_cols = df.shape
            n_dup = df.duplicated().sum()
            n_null = df.isna().sum().sum()
            null_rate = n_null / (n_rows * n_cols) * 100 if n_rows * n_cols > 0 else 0

            pc1, pc2, pc3, pc4 = st.columns(4)
            pc1.metric("Строк", f"{n_rows:,}")
            pc2.metric("Колонок", f"{n_cols}")
            pc3.metric("Дублей строк", f"{n_dup:,}",
                       delta=f"{n_dup/n_rows*100:.1f}%" if n_rows > 0 else "0%",
                       delta_color="inverse" if n_dup > 0 else "off")
            pc4.metric("Пропусков (всего)", f"{n_null:,}",
                       delta=f"{null_rate:.1f}%", delta_color="inverse" if null_rate > 5 else "off")

            st.divider()

            # --- Per-column profile ---
            section_header("Профиль по колонкам")
            profile_rows = []
            for col in df.columns:
                s = df[col]
                null_c = s.isna().sum()
                unique_c = s.nunique()
                dtype = str(s.dtype)
                row = {
                    "Колонка": col,
                    "Тип": dtype,
                    "Пропусков": null_c,
                    "% пропусков": round(null_c / n_rows * 100, 1) if n_rows > 0 else 0,
                    "Уникальных": unique_c,
                    "% уникальных": round(unique_c / n_rows * 100, 1) if n_rows > 0 else 0,
                }
                if pd.api.types.is_numeric_dtype(s):
                    desc = s.describe()
                    row["Среднее"] = round(desc.get("mean", 0), 4)
                    row["Медиана"] = round(s.median(), 4)
                    row["Мин"] = round(desc.get("min", 0), 4)
                    row["Макс"] = round(desc.get("max", 0), 4)
                    row["Стд. откл."] = round(desc.get("std", 0), 4)
                    # IQR outliers
                    q1, q3 = s.quantile(0.25), s.quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
                    row["Выбросов (IQR)"] = int(outliers)
                profile_rows.append(row)

            profile_df = pd.DataFrame(profile_rows)

            # Color-code nulls and outliers
            def _highlight_nulls(val):
                if isinstance(val, float) and val > 20:
                    return "background-color: #fde8e8"
                if isinstance(val, float) and val > 5:
                    return "background-color: #fef9e7"
                return ""

            null_col = "% пропусков"
            styled = profile_df.style.applymap(_highlight_nulls, subset=[null_col])
            st.dataframe(styled, use_container_width=True)

            # Download profile
            st.download_button(
                "⬇ Скачать профиль CSV",
                profile_df.to_csv(index=False).encode(),
                file_name="data_profile.csv",
            )

            st.divider()

            # --- Missing values heatmap ---
            section_header("Тепловая карта пропущенных значений")
            if n_null > 0:
                missing_matrix = df.isna().astype(int)
                # Downsample if too large
                max_rows_display = 200
                if len(missing_matrix) > max_rows_display:
                    step = max(1, len(missing_matrix) // max_rows_display)
                    missing_matrix = missing_matrix.iloc[::step]
                    st.caption(f"(Показаны каждые {step} строк для производительности)")

                fig_miss = px.imshow(
                    missing_matrix.T,
                    color_continuous_scale=[[0, "#ecf0f1"], [1, "#e74c3c"]],
                    title="Пропущенные значения (красный = пропуск)",
                    labels={"x": "Индекс строки", "y": "Колонка", "color": "Пропуск"},
                    aspect="auto",
                )
                fig_miss.update_layout(template="plotly_white", coloraxis_showscale=False)
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ Пропущенных значений нет!")

            # --- Numeric distributions grid ---
            if num_cols:
                st.divider()
                section_header("Распределения числовых переменных")
                grid_cols = min(3, len(num_cols))
                rows_g = (len(num_cols) + grid_cols - 1) // grid_cols
                for row_i in range(rows_g):
                    cols_g = st.columns(grid_cols)
                    for col_j in range(grid_cols):
                        col_idx = row_i * grid_cols + col_j
                        if col_idx >= len(num_cols):
                            break
                        col_name = num_cols[col_idx]
                        with cols_g[col_j]:
                            s_col = df[col_name].dropna()
                            if len(s_col) > 0:
                                fig_mini = px.histogram(
                                    s_col, nbins=30,
                                    title=col_name,
                                    labels={"value": col_name, "count": ""},
                                )
                                # Add median line
                                fig_mini.add_vline(
                                    x=s_col.median(), line_dash="dash",
                                    line_color="#e74c3c", opacity=0.7,
                                    annotation_text=f"median={s_col.median():.2f}",
                                )
                                fig_mini.update_layout(
                                    template="plotly_white",
                                    height=220,
                                    margin=dict(l=10, r=10, t=30, b=10),
                                    showlegend=False,
                                )
                                st.plotly_chart(fig_mini, use_container_width=True)

            # --- Categorical value counts ---
            if cat_cols:
                st.divider()
                section_header("Топ значений категориальных переменных")
                for c in cat_cols[:5]:  # limit to 5 cat cols
                    vc = df[c].value_counts().head(10)
                    fig_vc = px.bar(
                        x=vc.index, y=vc.values,
                        title=f"Топ-10 значений: {c}",
                        labels={"x": c, "y": "Количество"},
                    )
                    fig_vc.update_layout(template="plotly_white", height=250,
                                         margin=dict(l=10, r=10, t=35, b=10))
                    st.plotly_chart(fig_vc, use_container_width=True)

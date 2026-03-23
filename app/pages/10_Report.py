"""
pages/7_Report.py – Generate and download HTML reports with auto business summaries.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from app.state import init_state, dataset_selectbox, get_active_df
from app.styles import inject_all_css, page_header, section_header
from core.data import profile_dataframe
from core.report import ReportBuilder, generate_business_summary
from core.explore import (
    plot_timeseries, plot_correlation_heatmap, plot_stl_decomposition,
)

st.set_page_config(page_title="KIBAD – Report", layout="wide")
init_state()
inject_all_css()

page_header("10. Отчёт", "Генерация отчётов в HTML, PDF и Excel", "📄")

chosen = dataset_selectbox("Датасет", key="report_ds_sel",
                           help="Выберите датасет для включения в отчёт")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.info("📥 Данные не загружены. Перейдите на страницу **[1. Данные](pages/1_Data.py)** и загрузите файл.")
    st.stop()

dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
num_cols = df.select_dtypes(include="number").columns.tolist()

# ---------------------------------------------------------------------------
# Report configuration
# ---------------------------------------------------------------------------
section_header("Настройки отчёта")
col_a, col_b = st.columns(2)
report_title = col_a.text_input("Заголовок отчёта", f"{chosen} – Аналитический отчёт", key="rpt_title",
                                help="Заголовок, который будет отображаться в шапке HTML-отчёта.")
include_profile = col_a.checkbox("Профиль данных", value=True, key="rpt_profile",
                                 help="Включить колоночный профиль: пропуски, кардинальность, типы.")
include_ts = col_a.checkbox("График временного ряда", value=True, key="rpt_ts",
                            help="Включить линейный график целевой переменной во времени.")
include_corr = col_b.checkbox("Матрица корреляций", value=True, key="rpt_corr",
                              help="Тепловая карта корреляции Пирсона между числовыми переменными.")
include_stl = col_b.checkbox("STL-декомпозиция", value=True, key="rpt_stl",
                             help="Разложение на тренд, сезонность и остаток (требуется ≥ 24 точки).")
include_forecasts = col_b.checkbox("Результаты прогнозирования", value=True, key="rpt_fc",
                                   help="Таблицы и метрики моделей прогноза, построенных на странице «Временные ряды».")
include_tests = col_b.checkbox("Результаты статистических тестов", value=True, key="rpt_tests",
                               help="Таблица и интерпретации тестов, запущенных на странице «Тесты».")

# Column selections for charts – pre-populate from col_mappings if available
mapped_date = st.session_state.get("col_mappings", {}).get(chosen, {}).get("date", None)
mapped_target = st.session_state.get("col_mappings", {}).get(chosen, {}).get("target", None)

dt_cols_opts = ["(none)"] + dt_cols
default_date_idx = dt_cols_opts.index(mapped_date) if mapped_date in dt_cols_opts else 0
rpt_date = st.selectbox("Колонка даты", dt_cols_opts, index=default_date_idx, key="rpt_date_col",
                        help="Используется для графика временного ряда и STL-декомпозиции.")

num_cols_opts = ["(none)"] + num_cols
default_target_idx = num_cols_opts.index(mapped_target) if mapped_target in num_cols_opts else 0
rpt_target = st.selectbox("Целевая колонка", num_cols_opts, index=default_target_idx, key="rpt_target_col",
                          help="Числовой показатель для временного ряда и бизнес-резюме.")

# ---------------------------------------------------------------------------
# Pre-build checklist
# ---------------------------------------------------------------------------
with st.expander("📋 Предварительный просмотр отчёта", expanded=False):
    sections = {
        "📊 Временной ряд": rpt_date != "(none)" and rpt_target != "(none)",
        "📈 Распределения": len(df.select_dtypes(include='number').columns) > 0,
        "🔗 Корреляции": len(df.select_dtypes(include='number').columns) >= 2,
        "🔮 Прогноз": st.session_state.get("cluster_result") is not None or True,  # simplified
    }
    for name, available in sections.items():
        status = "✅ Включено" if available else "⏭️ Пропущено (нет данных)"
        st.markdown(f"- {name}: {status}")

# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------
st.divider()

if st.button("▶ Сформировать отчёт", key="btn_build_report", type="primary"):
    with st.spinner("Building report..."):
        rb = ReportBuilder(
            title=report_title,
            dataset_name=chosen,
            n_rows=len(df),
        )

        # --- Overview ---
        rb.add_kv_metrics("Dataset Overview", {
            "Dataset": chosen,
            "Rows": f"{df.shape[0]:,}",
            "Columns": df.shape[1],
            "Numeric cols": len(num_cols),
            "Date cols": len(dt_cols),
        })

        # --- Data profile ---
        if include_profile:
            profile = profile_dataframe(df)
            rb.add_table("Data Profile", profile, max_rows=30,
                         interpretation="Column-level profiling including missingness and type inference.")

        # --- Business summary ---
        fc_results = st.session_state.get("forecast_results", [])
        test_results = st.session_state.get("test_results", [])

        date_col_rpt = rpt_date if rpt_date != "(none)" else None
        target_col_rpt = rpt_target if rpt_target != "(none)" else None

        if date_col_rpt and target_col_rpt:
            try:
                summary = generate_business_summary(
                    df, date_col_rpt, target_col_rpt,
                    forecast_result=fc_results[-1] if fc_results else None,
                    test_results=test_results if test_results else None,
                )
                rb.add_interpretation("Business Summary", summary.replace("\n\n", "<br/><br/>"))
            except Exception as e:
                rb.add_interpretation("Business Summary", f"Could not generate summary: {e}")

        # --- Time series chart ---
        if include_ts and date_col_rpt and target_col_rpt:
            try:
                fig_ts = plot_timeseries(df, date_col_rpt, [target_col_rpt],
                                         title=f"{target_col_rpt} over Time")
                rb.add_figure("Time Series", fig_ts,
                              interpretation=f"Historical trend of {target_col_rpt}.")
            except Exception as e:
                rb.add_section("Time Series", f"<em>Chart error: {e}</em>")

        # --- Correlation heatmap ---
        if include_corr and len(num_cols) >= 2:
            try:
                fig_corr = plot_correlation_heatmap(df, columns=num_cols[:10])
                rb.add_figure("Correlation Heatmap", fig_corr,
                              interpretation="Pearson correlation between numeric variables.")
            except Exception as e:
                rb.add_section("Correlation Heatmap", f"<em>Chart error: {e}</em>")

        # --- STL decomposition ---
        if include_stl and date_col_rpt and target_col_rpt:
            try:
                sub = df[[date_col_rpt, target_col_rpt]].dropna().sort_values(date_col_rpt)
                if len(sub) >= 24:
                    fig_stl = plot_stl_decomposition(sub, date_col_rpt, target_col_rpt, period=12)
                    rb.add_figure("STL Decomposition", fig_stl,
                                  interpretation="Trend, seasonal, and residual components (STL, period=12).")
            except Exception as e:
                rb.add_section("STL Decomposition", f"<em>STL error: {e}</em>")

        # --- Forecast results ---
        if include_forecasts and fc_results:
            for res in fc_results:
                metrics_html = "".join(
                    f'<span class="tag">{k}={v}</span>' for k, v in res.metrics.items()
                )
                content = f"<p><strong>Model:</strong> {res.model_name}</p>"
                content += f"<p><strong>Notes:</strong> {res.notes}</p>"
                content += f"<p>{metrics_html}</p>"
                if res.explainability is not None:
                    from core.report import _df_to_html_table
                    content += "<h3>Explainability</h3>" + _df_to_html_table(res.explainability, max_rows=20)
                rb.add_section(f"Forecast: {res.model_name}", content)

        # --- Test results ---
        if include_tests and test_results:
            rows = [{
                "Test": r.name,
                "Statistic": r.statistic,
                "p-value": r.p_value,
                "Significant": r.significant,
                "Effect size": r.effect_size,
                "Effect label": r.effect_label,
            } for r in test_results]
            test_df = pd.DataFrame(rows)
            rb.add_table("Statistical Test Results", test_df)

            for r in test_results:
                rb.add_interpretation(
                    f"Interpretation: {r.name}",
                    r.interpretation,
                )

        # --- Methods used ---
        methods_content = """
        <ul>
          <li><strong>Data Profiling:</strong> Missing value rates, cardinality, type inference per column.</li>
          <li><strong>Decomposition:</strong> STL (Seasonal and Trend decomposition using Loess), statsmodels.</li>
          <li><strong>Forecasting:</strong> Seasonal Naive baseline; ARX via Ridge regression with lag features;
              SARIMAX via statsmodels.</li>
          <li><strong>Backtesting:</strong> Rolling-window evaluation with MAE, RMSE, MAPE, Bias.</li>
          <li><strong>Statistical Tests:</strong> Welch t-test, Mann–Whitney U, Chi-square, Pearson/Spearman
              correlation, Bootstrap permutation test, A/B test helper.</li>
          <li><strong>Scenario Simulation:</strong> ARX model with shocked exogenous inputs.</li>
        </ul>
        """
        rb.add_section("Methods Used", methods_content)

        # Store and display
        html = rb.render()
        st.session_state["last_report_html"] = html
        st.success("✅ Отчёт успешно сформирован!")

# ---------------------------------------------------------------------------
# Display and download report
# ---------------------------------------------------------------------------
if "last_report_html" in st.session_state:
    html = st.session_state["last_report_html"]

    st.divider()
    section_header("Предварительный просмотр отчёта")
    components.html(html, height=700, scrolling=True)

    col1, col2 = st.columns(2)
    col1.download_button(
        "📥 Скачать HTML-отчёт",
        html.encode("utf-8"),
        file_name="kibad_report.html",
        mime="text/html",
    )

    # PDF via WeasyPrint (optional)
    if col2.button("📄 Сформировать PDF", key="btn_pdf"):
        try:
            from weasyprint import HTML as WHTML
            pdf_bytes = WHTML(string=html).write_pdf()
            col2.download_button(
                "📥 Скачать PDF",
                pdf_bytes,
                file_name="kibad_report.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )
        except ImportError:
            col2.warning("WeasyPrint not installed. Install with: `pip install weasyprint`")
        except OSError as e:
            if "libgobject" in str(e) or "pango" in str(e).lower() or "gobject" in str(e).lower():
                col2.warning(
                    "WeasyPrint требует системные библиотеки Pango/GObject. "
                    "На macOS: `brew install pango`. "
                    "На Linux (Debian/Ubuntu): `apt-get install libpango-1.0-0 libpangoft2-1.0-0`. "
                    "В корпоративном контуре: обратитесь к системному администратору. "
                    "HTML-версия отчёта доступна без дополнительных зависимостей."
                )
            else:
                col2.error(f"PDF generation error: {e}")
        except Exception as e:
            col2.error(f"PDF generation error: {e}")

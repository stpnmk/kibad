"""
pages/7_TimeSeries.py – Improved forecasting UX with method cards, Russian labels,
ACF/PACF, anomaly detection, and trigger alerts.
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
from plotly.subplots import make_subplots
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df
from app.styles import inject_all_css, page_header, section_header
from core.models import (
    run_naive_forecast, run_arx_forecast, run_sarimax_forecast,
    rolling_backtest, ForecastResult,
)

st.set_page_config(page_title="KIBAD – Прогнозирование", layout="wide")
init_state()
inject_all_css()

page_header("7. Временные ряды", "Прогнозирование и анализ динамики", "📈")

# ---------------------------------------------------------------------------
# Method guide
# ---------------------------------------------------------------------------
guide_seen = st.session_state.get("fc_guide_seen", False)
with st.expander("📖 Справочник методов прогнозирования — выберите подходящий метод", expanded=not guide_seen):
    st.session_state["fc_guide_seen"] = True
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🔁 Наивный / Сезонный наивный**

**Что делает:** Наивный = последнее значение. Сезонный = значение того же периода прошлого сезона.

**Когда использовать:**
- Быстрая базовая линия для сравнения
- Горизонт 1–3 периода
- Ряд без выраженного тренда

**Когда НЕ использовать:**
- Длинный горизонт прогноза
- Есть чёткий тренд или внешние факторы
""")
    with col2:
        st.markdown("""
**📈 ARX (авторегрессия + внешние факторы)**

**Что делает:** Прогноз = линейная комбинация прошлых значений (лаги) + внешние переменные. Ridge-регуляризация.

**Когда использовать:**
- Есть внешние факторы (цены, реклама, макро)
- Нужна интерпретация вклада каждого фактора
- Горизонт 3–24 периода

**Когда НЕ использовать:**
- Мало данных (< 30 точек)
- Сложная сезонность без внешних факторов
""")
    with col3:
        st.markdown("""
**🌊 SARIMAX**

**Что делает:** Статистическая модель с явным моделированием тренда (d), авторегрессии (p,q) и сезонности (P,D,Q,S). Даёт доверительные интервалы.

**Когда использовать:**
- Чёткая сезонность (год, квартал)
- Нужны доверительные интервалы
- Данных ≥ 50 наблюдений

**Когда НЕ использовать:**
- Мало данных (< 30) — модель не сойдётся
- Много внешних факторов (лучше ARX)
""")

st.divider()

# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
chosen = dataset_selectbox("Датасет", key="fc_ds_sel",
                           help="Выберите датасет с подготовленными временными рядами")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.info("📥 Данные не загружены. Перейдите на страницу **[1. Данные](pages/1_Data.py)** и загрузите файл.")
    st.stop()

dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
num_cols = df.select_dtypes(include="number").columns.tolist()

if not dt_cols:
    st.warning("⚠️ Нет колонок с датами. Преобразуйте дату в разделе **Prepare**.")
    st.stop()
if not num_cols:
    st.warning("⚠️ Нет числовых колонок для прогнозирования.")
    st.stop()

# Settings
with st.expander("⚙️ Основные настройки прогноза", expanded=True):
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    with col_s1:
        fc_date = st.selectbox("Колонка даты", dt_cols, key="fc_date",
                               help="Временна́я ось ряда. Колонка должна иметь тип datetime.")
    with col_s2:
        fc_target = st.selectbox("Целевая переменная", num_cols, key="fc_target",
                                 help="Числовой показатель, который необходимо спрогнозировать.")
    with col_s3:
        fc_horizon = st.slider("Горизонт прогноза (периодов)", 1, 60, 12, key="fc_horizon",
                               help="На сколько периодов вперёд строить прогноз.")
    with col_s4:
        fc_period = st.slider("Сезонный период", 1, 52, 12, key="fc_period",
                              help="12=месячный, 4=квартальный, 52=недельный")
    col_s5, col_s6, col_s7 = st.columns(3)
    with col_s5:
        fc_exog = st.multiselect("Внешние факторы (exog)", num_cols, key="fc_exog",
                                 help="Дополнительные числовые переменные-предикторы для ARX/SARIMAX.")
    with col_s6:
        fc_lags_str = st.text_input("Лаги AR (через запятую)", "1,2,3,12", key="fc_lags",
                                    help="Прошлые значения ряда, используемые как признаки. Пример: 1,2,3,12")
    with col_s7:
        sub = df[[fc_date, fc_target]].dropna().sort_values(fc_date)
        st.caption(f"Точек: {len(sub)}")
        if len(sub) > 0:
            st.caption(f"Период: {sub[fc_date].min().date()} — {sub[fc_date].max().date()}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _plot_forecast(result: ForecastResult, title: str) -> go.Figure:
    fd = result.forecast_df
    hist = fd[fd["actual"].notna()].copy()
    fut = fd[fd["actual"].isna() & fd["forecast"].notna()].copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["actual"],
        mode="lines", name="Факт (история)",
        line=dict(color="#2c3e50", width=2.5),
    ))
    if not hist["forecast"].dropna().empty:
        fig.add_trace(go.Scatter(
            x=hist["date"], y=hist["forecast"],
            mode="lines", name="Подгонка модели",
            line=dict(color="#3498db", width=1.5, dash="dot"), opacity=0.8,
        ))
    if not hist.empty and not fut.empty:
        fig.add_vline(
            x=hist["date"].iloc[-1].timestamp() * 1000,
            line_dash="dash", line_color="#95a5a6", opacity=0.6,
            annotation_text="Граница прогноза", annotation_position="top right",
        )
    if not fut.empty and "lower" in fut.columns and fut["lower"].notna().any():
        fig.add_traces([go.Scatter(
            x=pd.concat([fut["date"], fut["date"][::-1]]),
            y=pd.concat([fut["upper"], fut["lower"][::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% доверительный интервал", hoverinfo="skip",
        )])
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=fut["date"], y=fut["forecast"],
            mode="lines+markers", name="Прогноз",
            line=dict(color="#e74c3c", width=2.5), marker=dict(size=5),
        ))
    m = result.metrics
    metric_text = " | ".join([f"{k}: {v}" for k, v in m.items()])
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=16)),
        template="plotly_white", hovermode="x unified",
        legend=dict(orientation="h", y=-0.25, x=0),
        xaxis_title="Дата", yaxis_title=fc_target,
        annotations=[dict(
            xref="paper", yref="paper", x=0.01, y=1.03,
            text=f"📊 {metric_text}", showarrow=False,
            font=dict(size=11, color="#666"),
        )],
    )
    return fig


def _show_metrics(metrics: dict) -> None:
    hints = {
        "MAE": "Средняя абсолютная ошибка (в единицах ряда). Меньше = лучше.",
        "RMSE": "Среднеквадратичная ошибка. Штрафует большие выбросы сильнее MAE.",
        "MAPE": "Средняя процентная ошибка. < 10% = отлично, 10–20% = хорошо.",
        "Bias": "Систематическое смещение. Близко к 0 = модель не завышает и не занижает.",
    }
    cols = st.columns(len(metrics))
    for i, (k, v) in enumerate(metrics.items()):
        cols[i].metric(label=k, value=str(v), help=hints.get(k, ""))
    # Color-coded MAPE interpretation
    mape_val = None
    if "MAPE" in metrics:
        try:
            mape_val = float(metrics["MAPE"])
        except (TypeError, ValueError):
            mape_val = None
    if mape_val is not None:
        if mape_val < 10:
            st.success(f"✅ MAPE {mape_val:.1f}% — отличная точность (< 10%)")
        elif mape_val < 25:
            st.warning(f"⚠️ MAPE {mape_val:.1f}% — приемлемая точность (10–25%)")
        else:
            st.error(f"🔴 MAPE {mape_val:.1f}% — низкая точность (> 25%). Попробуйте другой метод.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_naive, tab_arx, tab_sarimax, tab_backtest, tab_compare, tab_acf, tab_anomaly = st.tabs([
    "🔁 Наивный", "📈 ARX", "🌊 SARIMAX",
    "🔄 Бэктестинг", "📊 Сравнение", "📉 ACF/PACF", "🚨 Аномалии",
])

# ===========================================================================
# Naive
# ===========================================================================
with tab_naive:
    section_header("Наивный прогноз / Сезонный наивный")
    st.info(
        "**Базовая линия.** Наивный: forecast(t) = actual(t-1). "
        "Сезонный наивный: forecast(t) = actual(t - period). "
        "Если другие методы не превосходят наивный — они бесполезны."
    )
    naive_type = st.radio(
        "Тип модели",
        ["Сезонный наивный", "Наивный (последнее значение)"],
        horizontal=True, key="naive_type",
    )
    if st.button("▶ Запустить наивный прогноз", type="primary", key="btn_naive"):
        with st.spinner("Строим прогноз..."):
            try:
                result = run_naive_forecast(
                    df, fc_date, fc_target,
                    horizon=fc_horizon,
                    seasonal=(naive_type == "Сезонный наивный"),
                    period=fc_period,
                )
                st.session_state["forecast_results"].append(result)
                fig = _plot_forecast(result, f"Наивный прогноз: {fc_target}")
                st.plotly_chart(fig, use_container_width=True)
                _show_metrics(result.metrics)
                with st.expander("📥 Данные прогноза"):
                    st.dataframe(result.forecast_df, use_container_width=True)
                    st.download_button("⬇ CSV", result.forecast_df.to_csv(index=False).encode(),
                                       file_name="naive_forecast.csv")
            except Exception as e:
                st.error(f"Ошибка: {e}")

# ===========================================================================
# ARX
# ===========================================================================
with tab_arx:
    section_header("ARX — авторегрессия с внешними факторами (Ridge)")
    st.info(
        "Модель вида: `y(t) = β₀ + β₁·y(t-1) + β₂·y(t-2) + ... + βₖ·exog(t)`. "
        "Коэффициенты показывают вклад каждого фактора. "
        "Ridge (α) предотвращает переобучение на коротких рядах."
    )
    arx_alpha = st.slider("Регуляризация Ridge (α)", 0.01, 100.0, 1.0, key="arx_alpha",
                          help="Больше α → проще модель, меньше переобучения")
    if st.button("▶ Запустить ARX", type="primary", key="btn_arx"):
        with st.spinner("Обучаем ARX..."):
            try:
                lags = [int(x.strip()) for x in fc_lags_str.split(",") if x.strip().isdigit()]
                if not lags:
                    st.warning("Укажите лаги (например: 1,2,3,12).")
                else:
                    result = run_arx_forecast(
                        df, fc_date, fc_target,
                        exog_cols=fc_exog if fc_exog else None,
                        lags=lags, horizon=fc_horizon, alpha=arx_alpha,
                    )
                    st.session_state["forecast_results"].append(result)
                    fig = _plot_forecast(result, f"ARX прогноз: {fc_target}")
                    st.plotly_chart(fig, use_container_width=True)
                    _show_metrics(result.metrics)
                    if result.explainability is not None:
                        with st.expander("📊 Важность факторов"):
                            coef_df = result.explainability.copy()
                            coef_df["abs"] = coef_df["coefficient"].abs()
                            coef_df = coef_df.sort_values("abs", ascending=False).drop("abs", axis=1)
                            fig_c = px.bar(
                                coef_df.head(15), x="coefficient", y="feature", orientation="h",
                                title="Коэффициенты ARX (топ-15)",
                                color="coefficient", color_continuous_scale="RdBu",
                                color_continuous_midpoint=0,
                                labels={"coefficient": "Коэффициент", "feature": "Фактор"},
                            )
                            fig_c.update_layout(template="plotly_white")
                            st.plotly_chart(fig_c, use_container_width=True)
                            st.dataframe(coef_df, use_container_width=True)
                    with st.expander("📥 Данные прогноза"):
                        st.download_button("⬇ CSV", result.forecast_df.to_csv(index=False).encode(),
                                           file_name="arx_forecast.csv")
            except Exception as e:
                st.error(f"Ошибка ARX: {e}")

# ===========================================================================
# SARIMAX
# ===========================================================================
with tab_sarimax:
    section_header("SARIMAX — сезонная ARIMA с внешними переменными")
    st.info(
        "SARIMAX(p,d,q)(P,D,Q,S): p/q — авторегрессия/скользящее среднее; "
        "d — дифференцирование для устранения тренда; P,D,Q,S — сезонные аналоги. "
        "Начальная рекомендация: (1,1,1)(1,0,1,S)."
    )
    acf_suggestion = st.session_state.get("acf_sarimax_suggestion")
    if acf_suggestion:
        if st.button(f"⚡ Применить рекомендацию ACF/PACF (p={acf_suggestion['p']}, q={acf_suggestion['q']})", key="btn_apply_acf"):
            st.session_state["sarimax_p"] = acf_suggestion["p"]
            st.session_state["sarimax_q"] = acf_suggestion["q"]
            st.rerun()

    with st.expander("⚙️ Параметры SARIMAX", expanded=True):
        col_ns, col_s = st.columns(2)
        with col_ns:
            st.markdown("**Несезонная часть (p, d, q)**")
            p = st.slider("p (AR)", 0, 5, value=st.session_state.get("sarimax_p", 1), key="sarimax_p", help="Авторегрессия: число прошлых значений")
            d = st.slider("d (дифф.)", 0, 2, 1, key="sarimax_d", help="1 = устранение тренда")
            q = st.slider("q (MA)", 0, 5, value=st.session_state.get("sarimax_q", 1), key="sarimax_q", help="Скользящее среднее")
        with col_s:
            st.markdown("**Сезонная часть (P, D, Q, S)**")
            P = st.slider("P", 0, 3, 1, key="sarimax_P")
            D = st.slider("D", 0, 2, 0, key="sarimax_D")
            Q = st.slider("Q", 0, 3, 1, key="sarimax_Q")
            S = st.number_input("S (период)", 1, 52, fc_period, key="sarimax_S",
                                help="12=месячный, 4=квартальный, 52=недельный")

    if st.button("▶ Запустить SARIMAX", type="primary", key="btn_sarimax"):
        with st.spinner("Подбираем SARIMAX (20–60 сек)..."):
            try:
                result = run_sarimax_forecast(
                    df, fc_date, fc_target,
                    exog_cols=fc_exog if fc_exog else None,
                    order=(p, d, q), seasonal_order=(P, D, Q, int(S)),
                    horizon=fc_horizon,
                )
                st.session_state["forecast_results"].append(result)
                if result.notes:
                    st.info(f"📝 {result.notes}")
                fig = _plot_forecast(result, f"SARIMAX({p},{d},{q})({P},{D},{Q},{S}): {fc_target}")
                st.plotly_chart(fig, use_container_width=True)
                _show_metrics(result.metrics)
                if result.explainability is not None:
                    with st.expander("📊 Параметры модели"):
                        st.dataframe(result.explainability, use_container_width=True)
                with st.expander("📥 Данные прогноза"):
                    st.download_button("⬇ CSV", result.forecast_df.to_csv(index=False).encode(),
                                       file_name="sarimax_forecast.csv")
            except Exception as e:
                st.error(f"Ошибка SARIMAX: {e}")

# ===========================================================================
# Backtesting
# ===========================================================================
with tab_backtest:
    section_header("Скользящее бэктестирование")
    st.info(
        "Честная оценка модели: обучение на первых N периодах → прогноз на K периодов → "
        "сдвиг окна → повтор. Данные будущего никогда не попадают в обучение."
    )
    bt_model = st.selectbox("Модель", ["Сезонный наивный", "ARX"], key="bt_model")
    c1, c2, c3 = st.columns(3)
    bt_folds = c1.slider("Фолдов", 2, 8, 3, key="bt_folds")
    bt_min_train = c2.slider("Мин. обучающих периодов", 12, 60, 24, key="bt_min_train")
    bt_horizon_bt = c3.slider("Горизонт фолда", 1, 24, 6, key="bt_horizon")

    if st.button("▶ Запустить бэктест", type="primary", key="btn_backtest"):
        with st.spinner("Запускаем скользящий бэктест..."):
            try:
                lags = [int(x.strip()) for x in fc_lags_str.split(",") if x.strip().isdigit()]
                fn = run_naive_forecast if bt_model == "Сезонный наивный" else run_arx_forecast
                kwargs = ({"seasonal": True, "period": fc_period}
                          if bt_model == "Сезонный наивный"
                          else {"lags": lags, "exog_cols": fc_exog if fc_exog else None})

                fold_results, summary_df = rolling_backtest(
                    df, fc_date, fc_target, model_fn=fn,
                    n_folds=bt_folds, min_train=bt_min_train, horizon=bt_horizon_bt, **kwargs,
                )

                if not summary_df.empty:
                    avg = summary_df[["MAE", "RMSE", "MAPE", "Bias"]].mean().round(4).to_dict()
                    st.markdown("**Средние метрики по всем фолдам:**")
                    _show_metrics(avg)

                    fig_bt = px.bar(
                        summary_df, x="fold", y="MAPE",
                        title=f"MAPE по фолдам — {bt_model}",
                        labels={"fold": "Фолд", "MAPE": "MAPE (%)"},
                        color="MAPE", color_continuous_scale="RdYlGn_r",
                        text="MAPE",
                    )
                    fig_bt.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
                    fig_bt.update_layout(template="plotly_white", showlegend=False)
                    st.plotly_chart(fig_bt, use_container_width=True)

                    st.dataframe(summary_df, use_container_width=True)
                    st.download_button("⬇ CSV", summary_df.to_csv(index=False).encode(),
                                       file_name="backtest_results.csv")
                else:
                    st.warning("Ни один фолд не завершился. Уменьшите мин. обучающих периодов.")
            except Exception as e:
                st.error(f"Ошибка бэктеста: {e}")

# ===========================================================================
# Compare
# ===========================================================================
with tab_compare:
    section_header("Сравнение моделей")
    fc_results = st.session_state.get("forecast_results", [])
    if not fc_results:
        st.info("Запустите хотя бы одну модель в других вкладках.")
    else:
        metrics_rows = [{"Модель": r.model_name, **r.metrics} for r in fc_results]
        metrics_df = pd.DataFrame(metrics_rows)
        st.dataframe(
            metrics_df.style.highlight_min(
                subset=[c for c in ["MAPE", "MAE", "RMSE"] if c in metrics_df.columns],
                color="#d4f0d4",
            ),
            use_container_width=True,
        )
        if "MAPE" in metrics_df.columns:
            best = metrics_df.loc[metrics_df["MAPE"].idxmin(), "Модель"]
            st.success(f"✅ Лучшая модель по MAPE: **{best}**")

        # Overlay chart
        fig_cmp = go.Figure()
        actual_added = False
        for r in fc_results:
            fd = r.forecast_df
            hist = fd[fd["actual"].notna()]
            fut = fd[fd["actual"].isna() & fd["forecast"].notna()]
            if not actual_added and not hist["actual"].dropna().empty:
                fig_cmp.add_trace(go.Scatter(
                    x=hist["date"], y=hist["actual"],
                    mode="lines", name="Факт", line=dict(color="#2c3e50", width=2.5),
                ))
                actual_added = True
            if not fut.empty:
                fig_cmp.add_trace(go.Scatter(
                    x=fut["date"], y=fut["forecast"],
                    mode="lines+markers", name=r.model_name,
                ))
        fig_cmp.update_layout(
            title="Наложение прогнозов всех моделей",
            template="plotly_white", hovermode="x unified",
            legend=dict(orientation="h", y=-0.25),
            xaxis_title="Дата", yaxis_title=fc_target,
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        if st.button("🗑 Очистить все прогнозы", key="btn_clear_fc"):
            st.session_state["forecast_results"] = []
            st.rerun()

# ===========================================================================
# ACF / PACF
# ===========================================================================
with tab_acf:
    section_header("ACF / PACF — автокорреляционный анализ")
    st.info(
        "**ACF** (автокорреляционная функция) → подсказывает порядок MA (q): где ACF «обрывается».\n\n"
        "**PACF** (частная АКФ) → подсказывает порядок AR (p): где PACF «обрывается».\n\n"
        "Красные столбцы вышли за пределы 95% доверительного интервала — значимые лаги."
    )
    acf_target = st.selectbox("Ряд", num_cols, key="acf_target")
    acf_nlags = st.slider("Лагов", 5, 60, 24, key="acf_nlags")
    acf_diff = st.checkbox("Применить разность (d=1) для устранения тренда", key="acf_diff")

    if st.button("📉 Построить ACF/PACF", type="primary", key="btn_acf"):
        try:
            from statsmodels.tsa.stattools import acf, pacf as pacf_fn

            series = df[acf_target].dropna()
            if acf_diff:
                series = series.diff().dropna()

            n = len(series)
            conf_bound = 1.96 / np.sqrt(n)
            nlags = min(acf_nlags, n - 2)

            acf_vals = acf(series, nlags=nlags, fft=True)
            pacf_vals = pacf_fn(series, nlags=min(nlags, n // 2 - 1))

            fig_ap = make_subplots(rows=2, cols=1,
                                   subplot_titles=["ACF — автокорреляция (→ порядок MA q)",
                                                   "PACF — частная автокорреляция (→ порядок AR p)"],
                                   vertical_spacing=0.15)

            lags_acf = list(range(len(acf_vals)))
            fig_ap.add_trace(go.Bar(
                x=lags_acf, y=acf_vals,
                marker_color=["#e74c3c" if abs(v) > conf_bound else "#3498db" for v in acf_vals],
                name="ACF", showlegend=False,
            ), row=1, col=1)
            fig_ap.add_hline(y=conf_bound, line_dash="dash", line_color="#95a5a6",
                             annotation_text=f"+{conf_bound:.3f}", row=1, col=1)
            fig_ap.add_hline(y=-conf_bound, line_dash="dash", line_color="#95a5a6",
                             annotation_text=f"-{conf_bound:.3f}", row=1, col=1)

            lags_pacf = list(range(len(pacf_vals)))
            fig_ap.add_trace(go.Bar(
                x=lags_pacf, y=pacf_vals,
                marker_color=["#e74c3c" if abs(v) > conf_bound else "#2ecc71" for v in pacf_vals],
                name="PACF", showlegend=False,
            ), row=2, col=1)
            fig_ap.add_hline(y=conf_bound, line_dash="dash", line_color="#95a5a6", row=2, col=1)
            fig_ap.add_hline(y=-conf_bound, line_dash="dash", line_color="#95a5a6", row=2, col=1)

            fig_ap.update_layout(template="plotly_white", height=520,
                                 xaxis_title="Лаг", xaxis2_title="Лаг",
                                 yaxis_title="Корреляция", yaxis2_title="Корреляция")
            st.plotly_chart(fig_ap, use_container_width=True)

            # Interpretation
            acf_cut = next((i for i, v in enumerate(acf_vals[1:], 1) if abs(v) <= conf_bound), None)
            pacf_cut = next((i for i, v in enumerate(pacf_vals[1:], 1) if abs(v) <= conf_bound), None)
            if acf_cut and pacf_cut:
                suggested_p = pacf_cut
                suggested_q = acf_cut
                st.session_state["acf_sarimax_suggestion"] = {"p": suggested_p, "q": suggested_q}
                st.success(
                    f"💡 **Рекомендация:** ACF обрывается на лаге {acf_cut} → q ≈ {acf_cut}. "
                    f"PACF обрывается на лаге {pacf_cut} → p ≈ {pacf_cut}. "
                    f"Попробуйте SARIMAX({pacf_cut},1,{acf_cut})(1,0,1,{fc_period})."
                )
            else:
                st.warning(
                    "Корреляции значимы на большинстве лагов — ряд возможно нестационарен. "
                    "Попробуйте включить дифференцирование (d=1) выше."
                )

        except ImportError:
            st.error("statsmodels не установлен. Выполните: `pip install statsmodels`")
        except Exception as e:
            st.error(f"Ошибка ACF/PACF: {e}")

# ===========================================================================
# Anomaly Detection + Triggers
# ===========================================================================
with tab_anomaly:
    section_header("Обнаружение аномалий")
    st.info(
        "**Z-score скользящего окна**: точка аномальна, если отклоняется от скользящего среднего "
        "более чем на threshold стандартных отклонений.\n\n"
        "**STL остатки**: аномалии ищутся в остатках после удаления тренда и сезонности."
    )

    an_target = st.selectbox("Ряд", num_cols, key="an_target")
    an_col1, an_col2, an_col3 = st.columns(3)
    an_method = an_col1.selectbox(
        "Метод", ["rolling_zscore", "stl_residual"], key="an_method",
        format_func=lambda x: "Z-score (окно)" if x == "rolling_zscore" else "STL остатки",
    )
    an_window = an_col2.slider("Окно (периодов)", 3, 30, 12, key="an_window")
    an_threshold = an_col3.slider("Порог (σ)", 1.5, 5.0, 3.0, step=0.1, key="an_threshold")

    if st.button("🔍 Найти аномалии", type="primary", key="btn_anomaly"):
        try:
            from core.models import detect_anomalies
            series = df[an_target].dropna()
            anomaly_df = detect_anomalies(
                series, method=an_method, window=an_window, threshold=an_threshold,
            )
            n_anom = anomaly_df["is_anomaly"].sum()
            n_anomalies = int(n_anom)
            series = anomaly_df["value"].dropna()
            st.metric("Обнаружено аномалий", n_anomalies,
                      delta=f"{n_anomalies/len(anomaly_df)*100:.1f}% от всех точек",
                      delta_color="inverse")

            if n_anomalies is not None and len(series) > 0:
                pct = n_anomalies / len(series) * 100
                expected_pct = 0.3  # ~0.3% for 3σ
                if pct > expected_pct * 2:
                    st.warning(f"📊 {pct:.2f}% точек помечены как аномалии. При пороге 3σ ожидается ~0.3%. Повышенный уровень аномалий может указывать на изменение режима данных или внешние шоки.")
                else:
                    st.info(f"📊 {pct:.2f}% точек помечены как аномалии — в пределах ожидаемого уровня (~0.3% при 3σ).")

            normal = anomaly_df[~anomaly_df["is_anomaly"]]
            anom = anomaly_df[anomaly_df["is_anomaly"]]
            fig_an = go.Figure()
            if "upper" in anomaly_df.columns and anomaly_df["upper"].notna().any():
                fig_an.add_trace(go.Scatter(
                    x=list(anomaly_df.index) + list(anomaly_df.index[::-1]),
                    y=list(anomaly_df["upper"]) + list(anomaly_df["lower"][::-1]),
                    fill="toself", fillcolor="rgba(52,152,219,0.1)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name=f"±{an_threshold}σ коридор", hoverinfo="skip",
                ))
            fig_an.add_trace(go.Scatter(
                x=normal.index, y=normal["value"],
                mode="lines", name="Норма",
                line=dict(color="#3498db", width=1.5),
            ))
            if not anom.empty:
                fig_an.add_trace(go.Scatter(
                    x=anom.index, y=anom["value"],
                    mode="markers", name="Аномалия ⚠️",
                    marker=dict(color="#e74c3c", size=12, symbol="x",
                                line=dict(width=2, color="#c0392b")),
                ))
            fig_an.update_layout(
                title=f"Аномалии: {an_target} (метод={an_method}, порог={an_threshold}σ)",
                template="plotly_white", hovermode="x unified",
                xaxis_title="Индекс", yaxis_title=an_target,
            )
            st.plotly_chart(fig_an, use_container_width=True)

            if n_anom > 0:
                st.dataframe(anomaly_df[anomaly_df["is_anomaly"]], use_container_width=True)
                st.download_button("⬇ Аномалии CSV",
                                   anomaly_df[anomaly_df["is_anomaly"]].to_csv().encode(),
                                   file_name="anomalies.csv")
            else:
                st.success("Аномалий не обнаружено при заданном пороге.")
        except ImportError:
            st.error("Модуль detect_anomalies недоступен.")
        except Exception as e:
            st.error(f"Ошибка: {e}")

    # Trigger Rules
    st.markdown("---")
    st.subheader("🚨 Правила-триггеры (алерты)")
    st.info("Задайте условия, при нарушении которых генерируется предупреждение.")

    with st.expander("➕ Добавить правило"):
        from core.triggers import TriggerRule
        tr_name = st.text_input("Название", "Превышение порога", key="tr_name")
        tr_type = st.selectbox("Тип", ["threshold_cross", "deviation_from_baseline", "slope_change"],
                               key="tr_type",
                               format_func=lambda x: {
                                   "threshold_cross": "Пересечение порога",
                                   "deviation_from_baseline": "Отклонение от базовой линии",
                                   "slope_change": "Изменение наклона (тренд)",
                               }.get(x, x))
        tr_col = st.selectbox("Колонка", num_cols, key="tr_col")
        tr_thresh = st.number_input("Порог", value=0.0, key="tr_thresh")
        tr_window = st.slider("Окно для baseline/slope", 2, 20, 5, key="tr_window")
        if st.button("Добавить правило", key="btn_add_rule"):
            rule = TriggerRule(
                name=tr_name, rule_type=tr_type, column=tr_col,
                params={"threshold": tr_thresh, "window": tr_window, "direction": "both"},
                active=True,
            )
            st.session_state["trigger_rules"].append(rule)
            st.success(f"Правило «{tr_name}» добавлено.")

    rules = st.session_state.get("trigger_rules", [])
    if rules:
        if st.button("🚨 Проверить правила", type="primary", key="btn_eval_triggers"):
            from core.triggers import evaluate_triggers, alerts_to_dataframe
            try:
                alerts = evaluate_triggers(df, rules)
                if alerts:
                    alert_df = alerts_to_dataframe(alerts)
                    st.warning(f"⚠️ {len(alerts)} срабатываний!")
                    st.dataframe(alert_df, use_container_width=True)
                    st.download_button("⬇ Алерты CSV", alert_df.to_csv(index=False).encode(),
                                       file_name="alerts.csv")
                else:
                    st.success("✅ Ни одно правило не сработало.")
            except Exception as e:
                st.error(f"Ошибка проверки: {e}")

        st.markdown("**Активные правила:**")
        for i, rule in enumerate(rules):
            c1r, c2r = st.columns([5, 1])
            c1r.markdown(f"`{rule.name}` — {rule.rule_type} на колонке `{rule.column}`")
            if c2r.button("❌", key=f"del_rule_{i}"):
                st.session_state["trigger_rules"].pop(i)
                st.rerun()

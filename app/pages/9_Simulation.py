"""
pages/6_Simulation.py – Scenario simulation: sliders, paths, component flows, export.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df
from app.styles import inject_all_css, page_header, section_header
from app.components.ux import interpretation_box
from core.audit import log_event
from core.simulation import (
    ScenarioPreset, run_scenario,
    plot_scenario_comparison, plot_component_flows, plot_scenario_delta,
)

st.set_page_config(page_title="KIBAD – Моделирование", layout="wide")
init_state()
inject_all_css()

page_header("9. Сценарное моделирование", "Что если? Анализ шоков и Монте-Карло", "🎲")

chosen = dataset_selectbox("Датасет", key="sim_ds_sel",
                           help="Выберите датасет с временны́ми рядами для сценарного моделирования")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.info("📥 Данные не загружены. Перейдите на страницу **[1. Данные](pages/1_Data.py)** и загрузите файл.")
    st.stop()

dt_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(include="object").columns.tolist()

if not dt_cols:
    st.warning("⚠️ Не найдено колонок с датой. Перейдите на страницу **[2. Подготовка](pages/2_Prepare.py)** → шаг «Парсинг дат» и преобразуйте нужную колонку.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_scenario, tab_mc = st.tabs(["🎯 Сценарный анализ", "🎲 Монте-Карло"])

with tab_scenario:
    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------
    with st.expander("⚙️ Настройки сценарного анализа", expanded=True):
        col_sc1, col_sc2, col_sc3 = st.columns(3)
        with col_sc1:
            sim_date = st.selectbox("Колонка даты", dt_cols, key="sim_date",
                                    help="Временна́я ось ряда. Должна иметь тип datetime.")
            sim_target = st.selectbox("Целевая переменная", num_cols, key="sim_target",
                                      help="Числовой показатель, по которому строятся базовый и сценарный прогнозы.")
        with col_sc2:
            sim_horizon = st.slider("Горизонт (периодов)", 1, 60, 12, key="sim_horizon",
                                    help="На сколько периодов вперёд строить прогноз.")
            sim_lags_str = st.text_input("Лаги AR", "1,2,3,12", key="sim_lags",
                                         help="Прошлые значения ряда в качестве признаков. Пример: 1,2,3,12")
        with col_sc3:
            sim_exog = st.multiselect("Экзогенные переменные", num_cols, key="sim_exog",
                                      help="Внешние факторы, к которым применяются шоки сценария.")
            sim_components = st.multiselect("Колонки компонентного потока (для 2-го графика)", num_cols,
                                            key="sim_comp_cols",
                                            help="Дополнительные колонки для отображения компонентного водопадного графика.")

    # ---------------------------------------------------------------------------
    # Scenario sliders
    # ---------------------------------------------------------------------------
    section_header("Шоки сценария")
    shock_type = st.radio("Тип шока", ["Относительный (%)", "Абсолютный (ед.)"], horizontal=True, key="sim_shock_type")
    st.markdown(
        "Используйте слайдеры для задания шоков по экзогенным переменным. "
        "Базовый сценарий = 0 (без изменений)."
    )

    shocks: dict[str, float] = {}
    if sim_exog:
        cols_per_row = 3
        rows = [sim_exog[i:i+cols_per_row] for i in range(0, len(sim_exog), cols_per_row)]
        for row in rows:
            scols = st.columns(len(row))
            for c, col in zip(row, scols):
                shock_label = f"{c} шок ({'%' if shock_type == 'Относительный (%)' else 'ед.'})"
                shock_pct = col.slider(
                    shock_label, -50, 100, 0, key=f"shock_{c}",
                    help=f"Задать шок для '{c}' на горизонте прогноза.",
                )
                shocks[c] = shock_pct / 100.0 if shock_type == "Относительный (%)" else float(shock_pct)
    else:
        st.info("Выберите экзогенные переменные в настройках выше, чтобы задать шоки по переменным. "
                "Без экзогенных переменных к прогнозу будет применён скалярный шок.")
        # Fallback: global output shock
        global_shock = st.slider("Глобальный шок на выходе (%)", -50, 100, 0, key="global_shock")
        shocks["_global"] = global_shock / 100.0

    scenario_name = st.text_input("Название сценария", "Мой сценарий", key="sim_name")

    # ---------------------------------------------------------------------------
    # Run simulation
    # ---------------------------------------------------------------------------
    if st.button("▶ Запустить симуляцию", key="btn_sim", type="primary"):
        with st.spinner("Запуск базового и сценарного прогноза..."):
            try:
                lags = [int(x.strip()) for x in sim_lags_str.split(",") if x.strip().isdigit()]
                preset = ScenarioPreset(
                    name=scenario_name,
                    shocks={k: v for k, v in shocks.items() if k != "_global"},
                    absolute_overrides={},
                    notes=f"Shock: {shocks}",
                )
                # Global fallback
                if not sim_exog and "_global" in shocks:
                    preset.shocks = {}
                    preset.absolute_overrides = {}
                    # handled inside run_scenario as scalar

                base_res, scen_res = run_scenario(
                    df,
                    date_col=sim_date,
                    target_col=sim_target,
                    exog_cols=sim_exog if sim_exog else None,
                    lags=lags,
                    horizon=sim_horizon,
                    preset=preset,
                    scenario_name=scenario_name,
                    component_cols=sim_components if sim_components else None,
                )

                combined_df = scen_res.forecast_df  # has date, baseline, scenario, delta

                st.session_state["last_sim_base"] = base_res
                st.session_state["last_sim_scen"] = scen_res
                st.session_state["last_sim_combined"] = combined_df
                st.session_state["last_sim_hist_df"] = df
                st.session_state["last_sim_config"] = {
                    "date": sim_date, "target": sim_target, "name": scenario_name,
                }

            except Exception as e:
                st.error(f"Ошибка симуляции: {e}")

    # ---------------------------------------------------------------------------
    # Charts
    # ---------------------------------------------------------------------------
    combined_df = st.session_state.get("last_sim_combined")
    hist_df_sim = st.session_state.get("last_sim_hist_df", df)
    sim_config = st.session_state.get("last_sim_config", {})
    scen_res = st.session_state.get("last_sim_scen")

    if combined_df is not None:
        date_col_cfg = sim_config.get("date", sim_date)
        target_col_cfg = sim_config.get("target", sim_target)
        scen_name_cfg = sim_config.get("name", "Scenario")

        st.divider()
        section_header("График 1: Факт vs Базовый vs Сценарий")
        fig1 = plot_scenario_comparison(
            combined_df, hist_df_sim, date_col_cfg, target_col_cfg,
            title=f"Scenario: {scen_name_cfg}",
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Post-simulation narrative
        base_res_obj = st.session_state.get("last_sim_base")
        scen_res_obj = st.session_state.get("last_sim_scen")
        if base_res_obj is not None and scen_res_obj is not None:
            try:
                baseline_df = base_res_obj.forecast_df
                scenario_df = scen_res_obj.forecast_df
                if baseline_df is not None and scenario_df is not None:
                    total_base = baseline_df.iloc[:, -1].sum() if hasattr(baseline_df, 'iloc') else 0
                    total_scen = scenario_df.iloc[:, -1].sum() if hasattr(scenario_df, 'iloc') else 0
                    delta = total_scen - total_base
                    delta_pct = (delta / abs(total_base) * 100) if total_base != 0 else 0
                    direction = "вырастет" if delta > 0 else "снизится"
                    interpretation_box(
                        "Итог сценарного анализа",
                        f"При заданных шоках прогнозируемый показатель **{direction}** на **{abs(delta_pct):.1f}%** "
                        f"относительно базового сценария.",
                        icon="📈" if delta > 0 else "📉",
                    )
            except Exception:
                pass

        section_header("График 2: Дельта (Сценарий – Базовый)")
        fig2 = plot_scenario_delta(combined_df, title=f"Delta: {scen_name_cfg} vs Baseline")
        st.plotly_chart(fig2, use_container_width=True)

        if scen_res and scen_res.components_df is not None:
            section_header("График 3: Компонентные потоки")
            fig3 = plot_component_flows(scen_res.components_df, title="Декомпозиция компонентных потоков")
            st.plotly_chart(fig3, use_container_width=True)

        # Экспорт
        st.divider()
        section_header("Экспорт результатов")
        csv_combined = combined_df.to_csv(index=False).encode()
        st.download_button(
            "📥 Скачать результаты сценария (CSV)", csv_combined,
            file_name=f"scenario_{scen_name_cfg}.csv", mime="text/csv",
        )

    # ---------------------------------------------------------------------------
    # Scenario preset management
    # ---------------------------------------------------------------------------
    st.divider()
    section_header("Пресеты сценариев")

    col_save, col_load = st.columns(2)
    with col_save:
        st.markdown("**Сохранить текущий сценарий как пресет**")
        preset_save_name = st.text_input("Название пресета", scenario_name, key="preset_save_name")
        if st.button("Сохранить пресет", key="btn_save_preset"):
            preset = ScenarioPreset(
                name=preset_save_name,
                shocks={k: v for k, v in shocks.items() if k != "_global"},
                notes=f"Сохранён: {scenario_name}",
            )
            st.session_state.setdefault("scenario_presets", {})[preset_save_name] = preset.to_json()
            st.success(f"Пресет «{preset_save_name}» сохранён.")

    with col_load:
        presets = st.session_state.get("scenario_presets", {})
        if presets:
            st.markdown("**Загрузить сохранённый пресет**")
            sel_preset = st.selectbox("Пресет", list(presets.keys()), key="preset_load_sel")
            if st.button("Загрузить пресет", key="btn_load_preset"):
                loaded = ScenarioPreset.from_json(presets[sel_preset])
                st.json({"shocks": loaded.shocks, "overrides": loaded.absolute_overrides})
                st.info("Пресет загружен. Настройте слайдеры и перезапустите симуляцию.")
        else:
            st.info("Пресеты ещё не сохранены.")

    # Show all presets as table
    if st.session_state.get("scenario_presets"):
        with st.expander("Все пресеты"):
            rows = []
            for pname, pjson in st.session_state["scenario_presets"].items():
                p = ScenarioPreset.from_json(pjson)
                rows.append({"name": pname, "shocks": str(p.shocks), "notes": p.notes})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            export_json = json.dumps(st.session_state["scenario_presets"], indent=2)
            st.download_button("⬇ Экспорт всех пресетов (JSON)", export_json.encode(),
                               file_name="scenario_presets.json", mime="application/json")

with tab_mc:
    section_header("Монте-Карло: Распределение риска", "🎲")

    with st.expander("📖 Как работает Монте-Карло", expanded=False):
        st.markdown("""
**Метод Монте-Карло** запускает N независимых симуляций будущего, каждая из которых использует случайные шоки,
распределённые нормально. Результат — **веер возможных сценариев** и вероятностные метрики риска.

| Параметр | Назначение |
|----------|------------|
| **Целевой показатель** | Числовая переменная, последнее наблюдение которой служит стартовым значением |
| **Количество симуляций (N)** | Чем больше, тем точнее распределение; рекомендуется ≥ 1 000 |
| **Горизонт (периодов)** | Количество шагов вперёд (периодов прогноза) |
| **Средний шок (%)** | Математическое ожидание случайного шока на каждом шаге (0 = нет тренда) |
| **Волатильность шока (%)** | Стандартное отклонение шока — чем выше, тем шире веер |
| **Сид случайности** | Фиксирует генератор псевдослучайных чисел для воспроизводимости; 0 = полностью случайный |
| **Перцентили** | Границы доверительных зон на веере и в итоговой таблице (P5 — стресс, P95 — лучший сценарий) |

**VaR (95%)** — максимальный убыток, который не будет превышен с вероятностью 95%.
**CVaR / ES** — средний убыток в худших 5% сценариев (более консервативная мера).
        """)

    mc_col1, mc_col2 = st.columns(2)
    with mc_col1:
        mc_target = st.selectbox("Целевой показатель", num_cols, key="mc_target",
                                  help="Последнее значение этой колонки — стартовая точка всех симуляций")
        mc_n_sims = st.slider("Количество симуляций (N)", 100, 5000, 1000, step=100, key="mc_n_sims",
                               help="Больше симуляций = точнее распределение, но дольше расчёт")
        mc_horizon = st.slider("Горизонт (периодов)", 1, 36, 12, key="mc_horizon",
                                help="На сколько шагов вперёд моделировать")
    with mc_col2:
        mc_shock_mean = st.number_input("Средний шок (%)", value=0.0, step=1.0, key="mc_shock_mean",
                                         help="Среднее изменение на каждом шаге. 0 = нет направленного тренда")
        mc_shock_std = st.number_input("Волатильность шока (%)", value=5.0, step=0.5, min_value=0.1, key="mc_shock_std",
                                        help="Стандартное отклонение шока. Чем выше — тем шире веер сценариев")
        mc_seed = st.number_input("Сид случайности (0 = случайный)", value=42, min_value=0, key="mc_seed",
                                   help="Фиксирует результат для воспроизводимости. 0 = новый случайный каждый раз")

    mc_method = st.radio(
        "Метод симуляции",
        ["Простые шоки", "GBM (геометрическое броуновское движение)"],
        horizontal=True, key="mc_method",
        help="GBM моделирует логнормальные доходности — реалистичнее для финансовых и бизнес-рядов.",
    )

    mc_percentiles = st.multiselect("Показать перцентили",
                                     [5, 10, 25, 50, 75, 90, 95],
                                     default=[5, 50, 95], key="mc_pct_sel",
                                     help="P5 = стресс-сценарий, P50 = медиана, P95 = оптимистичный")

    if st.button("▶ Запустить симуляции", type="primary", key="btn_mc"):
        series = df[mc_target].dropna().values
        if len(series) < 2:
            st.error("Недостаточно данных для симуляции.")
        else:
            base_value = float(series[-1])
            rng = np.random.default_rng(int(mc_seed) if mc_seed > 0 else None)

            all_paths = np.zeros((mc_n_sims, mc_horizon + 1))
            all_paths[:, 0] = base_value

            mu = mc_shock_mean / 100
            sigma = mc_shock_std / 100

            if mc_method == "GBM (геометрическое броуновское движение)":
                # S(t) = S(t-1) * exp((μ - σ²/2) + σ * Z)  where Z ~ N(0,1)
                for t in range(1, mc_horizon + 1):
                    z = rng.standard_normal(mc_n_sims)
                    all_paths[:, t] = all_paths[:, t-1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * z)
            else:
                # Simple multiplicative shocks
                for t in range(1, mc_horizon + 1):
                    period_shocks = rng.normal(mu, sigma, size=mc_n_sims)
                    all_paths[:, t] = all_paths[:, t-1] * (1 + period_shocks)

            st.session_state["mc_paths"] = all_paths
            st.session_state["mc_target_name"] = mc_target
            st.session_state["mc_base"] = base_value
            st.session_state["mc_pct_saved"] = sorted(mc_percentiles) if mc_percentiles else [5, 50, 95]
            log_event("analysis_run", {"type": "monte_carlo", "n_sims": mc_n_sims,
                                        "horizon": mc_horizon, "method": mc_method})
            st.success(f"✅ {mc_n_sims:,} симуляций выполнено ({mc_method})!")

    if "mc_paths" in st.session_state:
        all_paths = st.session_state["mc_paths"]
        target_name = st.session_state["mc_target_name"]
        base_val = st.session_state["mc_base"]
        pcts = st.session_state.get("mc_pct_saved", [5, 50, 95])

        n_sims, n_steps = all_paths.shape
        periods = list(range(n_steps))

        # Fan chart — percentile bands
        fig_fan = go.Figure()
        pct_colors = {5: "rgba(231,76,60,0.15)", 10: "rgba(231,76,60,0.2)",
                      25: "rgba(241,196,15,0.2)", 50: None,
                      75: "rgba(241,196,15,0.2)", 90: "rgba(46,204,113,0.2)",
                      95: "rgba(46,204,113,0.15)"}
        pct_data = {p: np.percentile(all_paths, p, axis=0) for p in pcts}

        # Draw bands between symmetric percentiles if present
        sorted_pcts = sorted(pcts)
        for i in range(len(sorted_pcts) // 2):
            lo_p = sorted_pcts[i]
            hi_p = sorted_pcts[-(i+1)]
            if lo_p != hi_p:
                color = pct_colors.get(lo_p, "rgba(100,100,200,0.1)")
                fig_fan.add_trace(go.Scatter(
                    x=periods + periods[::-1],
                    y=list(pct_data[lo_p]) + list(pct_data[hi_p])[::-1],
                    fill="toself", fillcolor=color,
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"P{lo_p}–P{hi_p}",
                    showlegend=True,
                ))

        # Draw percentile lines
        line_styles = {5: "dash", 10: "dot", 25: "dashdot", 50: "solid",
                       75: "dashdot", 90: "dot", 95: "dash"}
        line_colors = {5: "#e74c3c", 10: "#e74c3c", 25: "#f39c12", 50: "#2980b9",
                       75: "#f39c12", 90: "#27ae60", 95: "#27ae60"}
        for p in sorted_pcts:
            fig_fan.add_trace(go.Scatter(
                x=periods, y=pct_data[p],
                mode="lines",
                line=dict(color=line_colors.get(p, "#999"), dash=line_styles.get(p, "solid"),
                          width=2 if p == 50 else 1),
                name=f"P{p}",
            ))

        # Baseline
        fig_fan.add_hline(y=base_val, line_dash="dot", line_color="gray",
                          annotation_text="Базовый уровень")

        fig_fan.update_layout(
            title=f"Веер симуляций: {target_name} ({n_sims:,} путей)",
            xaxis_title="Период",
            yaxis_title=target_name,
            template="plotly_white",
            legend=dict(orientation="h", y=-0.15),
            height=450,
        )
        st.plotly_chart(fig_fan, use_container_width=True)

        # Final period histogram
        final_vals = all_paths[:, -1]
        fig_hist = px.histogram(
            x=final_vals, nbins=50,
            title=f"Распределение исходов (период {n_steps - 1})",
            labels={"x": target_name, "y": "Количество симуляций"},
            color_discrete_sequence=["#3498db"],
            template="plotly_white",
        )
        # Add percentile lines
        for p in sorted_pcts:
            pv = np.percentile(final_vals, p)
            fig_hist.add_vline(x=pv, line_dash="dash",
                               line_color=line_colors.get(p, "#999"),
                               annotation_text=f"P{p}: {pv:,.1f}",
                               annotation_position="top")
        st.plotly_chart(fig_hist, use_container_width=True)

        # Summary table
        section_header("Сводка по симуляциям", "📊")
        summary_rows = []
        for p in sorted_pcts:
            fv = np.percentile(final_vals, p)
            delta_pct = (fv - base_val) / abs(base_val) * 100 if base_val != 0 else 0
            summary_rows.append({
                "Перцентиль": f"P{p}",
                "Итоговое значение": f"{fv:,.2f}",
                "Изменение от базы": f"{delta_pct:+.1f}%",
                "Сценарий": "Стресс" if p <= 10 else "Пессимистичный" if p <= 25 else "Базовый" if p == 50 else "Оптимистичный" if p <= 90 else "Лучший",
            })
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        # Value at Risk
        var_95 = base_val - np.percentile(final_vals, 5)
        cvar_95 = base_val - np.mean(final_vals[final_vals <= np.percentile(final_vals, 5)])
        prob_loss = (final_vals < base_val).mean() * 100

        # Maximum Drawdown (median path)
        median_path = np.median(all_paths, axis=0)
        running_max = np.maximum.accumulate(median_path)
        drawdowns = (median_path - running_max) / np.where(running_max != 0, running_max, 1)
        max_drawdown_pct = float(drawdowns.min()) * 100  # most negative value

        risk_col1, risk_col2, risk_col3, risk_col4 = st.columns(4)
        risk_col1.metric("VaR (95%)", f"{var_95:,.1f}",
                          delta=f"{var_95/abs(base_val)*100:.1f}% от базы" if base_val != 0 else None,
                          delta_color="inverse")
        risk_col2.metric("CVaR / ES (95%)", f"{cvar_95:,.1f}",
                          delta="Ожидаемый убыток в худших 5%", delta_color="off")
        risk_col3.metric("Вероятность убытка", f"{prob_loss:.1f}%",
                          delta_color="inverse")
        risk_col4.metric("Макс. просадка (медиана)", f"{max_drawdown_pct:.1f}%",
                          delta="Пик → минимум медианного пути", delta_color="off")

        # Export
        paths_df = pd.DataFrame(all_paths, columns=[f"t{i}" for i in range(n_steps)])
        csv_bytes = paths_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        st.download_button("📥 Скачать все пути (CSV)", data=csv_bytes,
                           file_name="monte_carlo_paths.csv", mime="text/csv")

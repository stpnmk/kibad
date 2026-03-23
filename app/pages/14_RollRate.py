"""
pages/14_RollRate.py – Roll-rate / transition matrix page for KIBAD.

Transition matrix heatmap, n-period forward projection, rate dynamics,
and raw table download.
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
import streamlit as st

from app.state import init_state, dataset_selectbox, get_active_df
from core.audit import log_event
from app.components.ux import interpretation_box
from app.styles import inject_all_css, page_header, section_header
from core.rollrate import (
    auto_bucket,
    build_transition_matrix,
    matrix_power,
    steady_state,
    roll_forward_rates,
    cure_rates,
    transition_time_series,
    BUCKET_ORDER,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – Матрица миграций", layout="wide")
init_state()
inject_all_css()

page_header("14. Roll-Rate матрица", "Матрица миграции и Марковские цепи", "🔄")

COLORS = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Параметры")

    chosen = dataset_selectbox("Датасет", key="rollrate_ds_sel")
    if not chosen:
        st.stop()

    df = get_active_df()
    if df is None:
        st.error("Нет данных. Сначала загрузите датасет.")
        st.stop()

    all_cols = df.columns.tolist()

    loan_id_col = st.selectbox("ID займа", options=all_cols, key="rr_loan_id")

    date_candidates = df.select_dtypes(include=["datetime", "datetime64", "object"]).columns.tolist()
    period_col = st.selectbox("Период наблюдения (дата)", options=all_cols, key="rr_period")

    bucket_col = st.selectbox("Статус / DPD колонка", options=all_cols, key="rr_bucket")

    col_type = st.radio(
        "Тип колонки",
        ["Бакет (категория)", "DPD (число)"],
        key="rr_col_type",
    )

    edges_str = None
    if col_type == "DPD (число)":
        edges_str = st.text_input(
            "Границы бакетов",
            value="0,1,30,60,90,180",
            key="rr_edges",
        )

    use_weight = st.checkbox("Взвешивать по EAD/остатку", value=False, key="rr_use_weight")
    weight_col = None
    if use_weight:
        weight_col = st.selectbox(
            "Колонка остатка",
            options=all_cols,
            key="rr_weight_col",
        )

    forecast_horizon = st.number_input(
        "Горизонт прогноза (месяцев)",
        min_value=1,
        max_value=24,
        value=3,
        step=1,
        key="rr_forecast_horizon",
    )

    run_btn = st.button("▶ Рассчитать", type="primary", key="rr_run")

    st.divider()
    st.caption(f"Строк: **{len(df):,}**")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_matrix, tab_forecast, tab_dynamics, tab_table, tab_markov = st.tabs([
    "📊 Матрица переходов",
    "🔮 Прогноз (n-периодов)",
    "📈 Динамика ставок",
    "📋 Таблица",
    "🔗 Марковские цепи",
])

# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------

if run_btn:
    with st.spinner("Построение матрицы миграций…"):
        try:
            work_df = df.copy()

            # DPD → bucket if needed
            if col_type == "DPD (число)":
                edges = [int(e.strip()) for e in edges_str.split(",")]
                work_df["_bucket"] = auto_bucket(
                    pd.to_numeric(work_df[bucket_col], errors="coerce"),
                    edges=edges,
                )
                used_bucket_col = "_bucket"
            else:
                used_bucket_col = bucket_col

            count_m, rate_m = build_transition_matrix(
                work_df,
                loan_id_col=loan_id_col,
                period_col=period_col,
                bucket_col=used_bucket_col,
                weight_col=weight_col,
            )

            ts_df = transition_time_series(
                work_df,
                loan_id_col=loan_id_col,
                period_col=period_col,
                bucket_col=used_bucket_col,
            )

            st.session_state["rr_result"] = {
                "count_m": count_m,
                "rate_m": rate_m,
                "ts_df": ts_df,
                "work_df": work_df,
                "used_bucket_col": used_bucket_col,
            }
            log_event("rollrate_run", {
                "loan_id_col": loan_id_col,
                "period_col": period_col,
                "bucket_col": bucket_col,
                "col_type": col_type,
                "use_weight": use_weight,
            })
            st.success("Матрица миграций построена.")
        except Exception as exc:
            st.error(f"Ошибка: {exc}")

rr_result = st.session_state.get("rr_result")

# ---------------------------------------------------------------------------
# Tab 1: Transition matrix
# ---------------------------------------------------------------------------

with tab_matrix:
    if rr_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    else:
        count_m = rr_result["count_m"]
        rate_m = rr_result["rate_m"]

        # Filter to buckets with any activity
        active_buckets = [b for b in BUCKET_ORDER if b in rate_m.index and rate_m.loc[b].sum() > 0]
        if not active_buckets:
            active_buckets = [b for b in BUCKET_ORDER if b in rate_m.index]

        rate_display = rate_m.loc[active_buckets, active_buckets] if active_buckets else rate_m

        st.caption("**Строки** = текущий статус | **Столбцы** = следующий статус. "
                   "Читайте построчно: из статуса X, Y% переходят в статус Z.")

        col_left, col_right = st.columns([60, 40])

        with col_left:
            z_vals = rate_display.values * 100

            text_vals = []
            for i in range(len(z_vals)):
                row_text = []
                for j in range(len(z_vals[i])):
                    v = z_vals[i, j]
                    row_text.append(f"{v:.1f}%")
                text_vals.append(row_text)

            fig_matrix = go.Figure(data=go.Heatmap(
                z=z_vals,
                x=active_buckets,
                y=active_buckets,
                colorscale="RdBu_r",
                zmin=0,
                zmax=100,
                text=text_vals,
                texttemplate="%{text}",
                hovertemplate="Из: %{y}<br>В: %{x}<br>Ставка: %{z:.1f}%<extra></extra>",
                colorbar=dict(title="%"),
            ))
            fig_matrix.update_layout(
                title="Матрица переходов (% строки)",
                xaxis_title="Статус в следующем периоде",
                yaxis_title="Статус в текущем периоде",
                template="plotly_white",
            )
            st.plotly_chart(fig_matrix, use_container_width=True)

        with col_right:
            rf = roll_forward_rates(rate_m)
            cr = cure_rates(rate_m)

            # Average rates across non-absorbing buckets
            non_absorbing = [b for b in active_buckets if b not in ("Списан", "Закрыт")]

            avg_rf = float(rf[non_absorbing].mean()) * 100 if non_absorbing else 0.0
            avg_cr = float(cr[non_absorbing].mean()) * 100 if non_absorbing else 0.0

            # Write-off inflow: 90+ → Списан rate
            wo_rate = 0.0
            if "90+" in rate_m.index and "Списан" in rate_m.columns:
                wo_rate = float(rate_m.loc["90+", "Списан"]) * 100

            net_flow = avg_rf - avg_cr

            section_header("Ключевые метрики")
            st.metric("Средняя ставка ухудшения", f"{avg_rf:.1f}%")
            st.metric("Средняя ставка улучшения (cure)", f"{avg_cr:.1f}%")
            st.metric("Приток в списание (90+ → Списан)", f"{wo_rate:.1f}%")
            delta_color = "inverse" if net_flow > 0 else "normal"
            st.metric(
                "Чистый поток портфеля",
                f"{net_flow:+.1f}%",
                delta="ухудшение" if net_flow > 0 else "улучшение",
                delta_color="inverse" if net_flow > 0 else "normal",
            )

        # Raw matrices
        section_header("Матрицы (абсолютные и ставки)")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Количество переходов**")
            st.dataframe(count_m.loc[active_buckets, active_buckets].astype(int), use_container_width=True)
        with c2:
            st.markdown("**Ставки переходов (%)**")
            st.dataframe((rate_display * 100).round(1), use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Forecast
# ---------------------------------------------------------------------------

with tab_forecast:
    if rr_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    else:
        rate_m = rr_result["rate_m"]
        work_df = rr_result["work_df"]
        used_bucket_col = rr_result["used_bucket_col"]

        horizon = st.slider(
            "Горизонт (периодов)",
            min_value=1,
            max_value=24,
            value=int(forecast_horizon),
            key="rr_forecast_slider",
        )

        # Current distribution from last period
        try:
            last_period = work_df[period_col].max()
            last_df = work_df[work_df[period_col] == last_period]
            current_dist = last_df[used_bucket_col].value_counts()
            current_dist_norm = {b: float(current_dist.get(b, 0)) for b in BUCKET_ORDER if b in rate_m.index}
            total_loans = sum(current_dist_norm.values())
            if total_loans > 0:
                current_dist_norm = {k: v / total_loans for k, v in current_dist_norm.items()}
            else:
                current_dist_norm = {b: 1.0 / len(rate_m.index) for b in rate_m.index}

            # Build starting vector
            active_buckets = [b for b in BUCKET_ORDER if b in rate_m.index]
            start_vec = np.array([current_dist_norm.get(b, 0) for b in active_buckets])

            rate_m_active = rate_m.loc[active_buckets, active_buckets]

            projections = {}
            vec = start_vec.copy()
            projections[0] = {b: v for b, v in zip(active_buckets, vec)}
            for t in range(1, horizon + 1):
                T_n = matrix_power(rate_m_active, t)
                proj_vec = start_vec @ T_n.values
                projections[t] = {b: float(proj_vec[i]) for i, b in enumerate(active_buckets)}

            # Stacked area chart
            proj_df = pd.DataFrame(projections).T
            proj_df.index.name = "Период"
            proj_df = proj_df * 100

            fig_proj = go.Figure()
            for i, bucket in enumerate(active_buckets):
                if bucket in proj_df.columns:
                    fig_proj.add_trace(go.Scatter(
                        x=proj_df.index,
                        y=proj_df[bucket],
                        name=bucket,
                        stackgroup="one",
                        line=dict(color=COLORS[i % len(COLORS)]),
                        hovertemplate=f"{bucket}<br>Период: %{{x}}<br>Доля: %{{y:.1f}}%<extra></extra>",
                    ))

            # Steady state
            try:
                ss = steady_state(rate_m_active)
                for i, bucket in enumerate(active_buckets):
                    if bucket in ss.index:
                        fig_proj.add_hline(
                            y=float(ss[bucket]) * 100,
                            line_dash="dash",
                            line_color=COLORS[i % len(COLORS)],
                            opacity=0.4,
                        )
            except Exception:
                pass

            fig_proj.update_layout(
                title=f"Прогноз распределения портфеля на {horizon} периодов",
                xaxis_title="Период",
                yaxis_title="Доля портфеля (%)",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_proj, use_container_width=True)
            st.caption("Пунктирные линии — стационарное распределение.")

            interpretation_box(
                "Стационарное распределение",
                "**Пунктирные линии** показывают долгосрочное равновесие портфеля — распределение, к которому "
                "он асимптотически стремится при стабильной матрице переходов.\n\n"
                "Если текущее распределение далеко от пунктирных линий — портфель находится в переходном состоянии. "
                "Чем быстрее сходимость, тем стабильнее процесс миграции.",
                icon="📉",
            )

            section_header("Таблица проекции")
            st.dataframe(proj_df.round(2), use_container_width=True)

        except Exception as exc:
            st.error(f"Ошибка прогноза: {exc}")

# ---------------------------------------------------------------------------
# Tab 3: Rate dynamics
# ---------------------------------------------------------------------------

with tab_dynamics:
    if rr_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    else:
        ts_df = rr_result["ts_df"]

        if ts_df.empty or ts_df["period"].nunique() < 2:
            st.info("Нужно 2+ периода для анализа динамики.")
        else:
            show_metric = st.radio(
                "Показать",
                ["Roll-forward", "Cure", "Write-off"],
                horizontal=True,
                key="rr_metric_radio",
            )

            buckets_available = [b for b in BUCKET_ORDER if b in ts_df["source_bucket"].unique()]

            fig_dyn = go.Figure()
            for i, bucket in enumerate(buckets_available):
                bdf = ts_df[ts_df["source_bucket"] == bucket].sort_values("period")
                if show_metric == "Roll-forward":
                    y_vals = bdf["roll_forward_rate"] * 100
                    y_title = "Ставка ухудшения (%)"
                elif show_metric == "Cure":
                    y_vals = bdf["cure_rate"] * 100
                    y_title = "Ставка улучшения (%)"
                else:
                    # Write-off: only 90+ bucket
                    if bucket != "90+":
                        continue
                    y_vals = bdf["roll_forward_rate"] * 100
                    y_title = "Ставка списания (%)"

                fig_dyn.add_trace(go.Scatter(
                    x=bdf["period"].astype(str),
                    y=y_vals,
                    name=bucket,
                    mode="lines+markers",
                    line=dict(color=COLORS[i % len(COLORS)]),
                ))

            fig_dyn.update_layout(
                title=f"Динамика ставок: {show_metric}",
                xaxis_title="Период",
                yaxis_title=y_title if "y_title" in dir() else "%",
                template="plotly_white",
                hovermode="x unified",
            )
            st.plotly_chart(fig_dyn, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4: Table
# ---------------------------------------------------------------------------

with tab_table:
    if rr_result is None:
        st.info("Настройте параметры и нажмите **▶ Рассчитать**.")
    else:
        count_m = rr_result["count_m"]
        rate_m = rr_result["rate_m"]

        section_header("Матрица переходов: количество")
        st.dataframe(count_m.astype(int), use_container_width=True)

        section_header("Матрица переходов: ставки (%)")
        st.dataframe((rate_m * 100).round(2), use_container_width=True)

        csv_bytes = rate_m.to_csv().encode("utf-8")
        st.download_button(
            "⬇️ Скачать матрицу ставок (CSV)",
            data=csv_bytes,
            file_name="rollrate_matrix.csv",
            mime="text/csv",
            key="rr_download",
        )

# ---------------------------------------------------------------------------
# Tab 5: Markov chain (absorbing states)
# ---------------------------------------------------------------------------

with tab_markov:
    section_header("Марковские цепи: Поглощающие состояния", "🔗")
    st.caption(
        "Анализирует вероятность достижения 'дефолта/списания' из каждого DPD-бакета. "
        "Рассчитывает фундаментальную матрицу и ожидаемое время до поглощения."
    )

    if rr_result is None:
        st.info("ℹ️ Сначала рассчитайте матрицу миграций на вкладке «Матрица переходов».")
    else:
        mat_df = rr_result["rate_m"]  # DataFrame with state labels as index+columns
        # Keep only states that have at least some activity (row sum > 0)
        active_mask = mat_df.sum(axis=1) > 0
        mat_df = mat_df.loc[active_mask, active_mask]
        states = list(mat_df.index)

        st.markdown("**Выберите поглощающие состояния** (из которых нет возврата):")
        # Auto-detect absorbing states: diagonal value >= 0.99
        auto_absorbing = [s for s in states if mat_df.loc[s, s] >= 0.99]
        absorbing_states = st.multiselect(
            "Поглощающие состояния:",
            options=states,
            default=auto_absorbing or ([states[-1]] if states else []),
            key="markov_absorbing",
            help="Обычно это 'Дефолт', 'Списан' — состояния с самопоглощением ~100%",
        )

        if not absorbing_states:
            st.warning("Выберите хотя бы одно поглощающее состояние.")
        elif len(absorbing_states) >= len(states):
            st.warning("Должно быть хотя бы одно нестабильное (транзитное) состояние.")
        else:
            transient_states = [s for s in states if s not in absorbing_states]

            # Build sub-matrices
            Q = mat_df.loc[transient_states, transient_states].values.astype(float)
            R = mat_df.loc[transient_states, absorbing_states].values.astype(float)

            # Fundamental matrix: N = (I - Q)^{-1}
            try:
                I_mat = np.eye(len(transient_states))
                N = np.linalg.inv(I_mat - Q)
                N_df = pd.DataFrame(N, index=transient_states, columns=transient_states)

                # Expected steps to absorption (row sums of N)
                t_steps = N.sum(axis=1)
                t_df = pd.DataFrame({
                    "Состояние": transient_states,
                    "Ожид. шагов до поглощения": t_steps.round(2),
                })

                # Absorption probabilities
                B = N @ R
                B_df = pd.DataFrame(B, index=transient_states, columns=absorbing_states)
                B_df = B_df.round(4)

                # Display
                mc_tab1, mc_tab2, mc_tab3 = st.tabs([
                    "📊 Вероятности поглощения",
                    "⏱️ Время до поглощения",
                    "🔢 Фундаментальная матрица",
                ])

                with mc_tab1:
                    st.markdown("**Вероятность достичь каждого поглощающего состояния из каждого транзитного:**")
                    fig_abs = px.imshow(
                        B_df,
                        text_auto=".2%",
                        color_continuous_scale="RdYlGn_r",
                        aspect="auto",
                        title="Вероятности поглощения (Lifetime PD по бакетам)",
                        labels=dict(color="Вероятность"),
                    )
                    fig_abs.update_layout(
                        template="plotly_white",
                        height=max(250, len(transient_states) * 60 + 100),
                    )
                    st.plotly_chart(fig_abs, use_container_width=True)

                    if absorbing_states:
                        default_col = absorbing_states[-1]
                        worst_state = B_df[default_col].idxmax()
                        best_state = B_df[default_col].idxmin()
                        prob_worst = B_df.loc[worst_state, default_col]
                        prob_best = B_df.loc[best_state, default_col]
                        st.info(
                            f"📌 Из состояния **{worst_state}** вероятность попасть в **{default_col}**: "
                            f"**{prob_worst:.1%}**  \n"
                            f"📌 Из состояния **{best_state}** вероятность попасть в **{default_col}**: "
                            f"**{prob_best:.1%}**"
                        )

                    st.dataframe(
                        B_df.style.background_gradient(cmap="RdYlGn_r"),
                        use_container_width=True,
                    )

                    csv_b = B_df.to_csv(encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        "📥 Скачать (CSV)",
                        data=csv_b,
                        file_name="absorption_probs.csv",
                        mime="text/csv",
                        key="markov_download_b",
                    )

                with mc_tab2:
                    st.markdown("**Ожидаемое количество шагов (периодов) до поглощения:**")

                    fig_time = px.bar(
                        t_df,
                        x="Состояние",
                        y="Ожид. шагов до поглощения",
                        color="Ожид. шагов до поглощения",
                        color_continuous_scale="RdYlGn_r",
                        title="Ожидаемое время до дефолта/списания (в периодах)",
                        text_auto=".1f",
                        template="plotly_white",
                    )
                    fig_time.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig_time, use_container_width=True)

                    st.dataframe(t_df, use_container_width=True, hide_index=True)

                    fastest = t_df.loc[t_df["Ожид. шагов до поглощения"].idxmin(), "Состояние"]
                    slowest = t_df.loc[t_df["Ожид. шагов до поглощения"].idxmax(), "Состояние"]
                    st.info(
                        f"⚡ Быстрее всего поглощается состояние **{fastest}** "
                        f"({t_df['Ожид. шагов до поглощения'].min():.1f} периодов)  \n"
                        f"🐢 Дольше всего — **{slowest}** "
                        f"({t_df['Ожид. шагов до поглощения'].max():.1f} периодов)"
                    )

                with mc_tab3:
                    st.markdown("**Фундаментальная матрица N = (I - Q)⁻¹**")
                    st.caption(
                        "N[i,j] = ожидаемое число посещений состояния j при старте из i до поглощения"
                    )
                    st.dataframe(N_df.round(3), use_container_width=True)

            except np.linalg.LinAlgError:
                st.error(
                    "Матрица (I - Q) вырождена — невозможно вычислить обратную. "
                    "Проверьте, что транзитные состояния не образуют замкнутый цикл."
                )

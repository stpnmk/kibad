"""
pages/12_Clustering.py – Clustering and segmentation page for KIBAD.

Provides K-Means clustering, elbow-method analysis, cluster profiling,
PCA visualisation, and labelled-data export — all in Russian.
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
from app.styles import inject_all_css, page_header, section_header
from core.audit import log_event
from core.cluster import run_kmeans, run_elbow, cluster_profiles, pca_transform

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – Кластеризация", layout="wide")
init_state()
inject_all_css()

page_header("12. Кластеризация", "Сегментация данных методами K-Means и иерархической кластеризации", "🔵")

# ---------------------------------------------------------------------------
# Dataset & controls
# ---------------------------------------------------------------------------

chosen = dataset_selectbox("Датасет", key="cluster_ds_sel",
                          help="Выберите датасет с числовыми признаками для кластеризации.")
if not chosen:
    st.stop()

df = get_active_df()
if df is None:
    st.info("📥 Данные не загружены. Перейдите на страницу **[1. Данные](pages/1_Data.py)** и загрузите файл.")
    st.stop()

num_cols = df.select_dtypes(include="number").columns.tolist()
if not num_cols:
    st.error("Датасет не содержит числовых колонок.")
    st.stop()

with st.expander("⚙️ Настройки кластеризации", expanded=True):
    col_c1, col_c2, col_c3, col_c4 = st.columns([3, 2, 2, 2])
    with col_c1:
        feature_cols = st.multiselect(
            "Признаки для кластеризации",
            options=num_cols,
            default=num_cols[:min(4, len(num_cols))],
            key="cluster_features",
            help="Числовые колонки, по которым будет выполнена кластеризация. Рекомендуется 2–10 признаков.",
        )
    with col_c2:
        n_clusters = st.number_input(
            "Количество кластеров (K)",
            min_value=2,
            max_value=15,
            value=4,
            step=1,
            key="cluster_k",
            help="Задайте K вручную или используйте вкладку «Метод локтя» для автоматического подбора.",
        )
    with col_c3:
        scale_features = st.checkbox("Нормализовать признаки (StandardScaler)", value=True, key="cluster_scale",
                                     help="Рекомендуется включить, если признаки имеют разные единицы измерения.")
        random_state = st.slider(
            "Случайное состояние",
            min_value=0,
            max_value=100,
            value=42,
            key="cluster_rs",
            help="Фиксирует случайность для воспроизводимости результатов.",
        )
    with col_c4:
        st.caption(f"Строк в датасете: **{len(df):,}**")
        st.caption(f"Числовых колонок: **{len(num_cols)}**")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_km, tab_elbow, tab_profiles, tab_pca, tab_results = st.tabs([
    "🎯 K-Means",
    "📐 Метод локтя",
    "📊 Профили кластеров",
    "🔵 PCA-визуализация",
    "📋 Результаты",
])

COLORS = px.colors.qualitative.Plotly

# ---------------------------------------------------------------------------
# Tab 1 – K-Means
# ---------------------------------------------------------------------------

with tab_km:
    section_header("K-Means кластеризация")

    if not feature_cols:
        st.warning("Выберите хотя бы один признак в настройках выше.")
    else:
        if st.button("▶ Кластеризовать", key="btn_kmeans", type="primary"):
            with st.spinner("Запуск K-Means…"):
                try:
                    result = run_kmeans(
                        df,
                        columns=feature_cols,
                        n_clusters=int(n_clusters),
                        random_state=int(random_state),
                        scale=scale_features,
                    )
                    st.session_state["cluster_result"] = result
                    log_event(
                        "clustering_run",
                        {
                            "n_clusters": int(n_clusters),
                            "features": feature_cols,
                            "scale": scale_features,
                            "silhouette": result.silhouette,
                            "inertia": result.inertia,
                        },
                    )
                    st.success("Кластеризация выполнена успешно.")
                except Exception as exc:
                    st.error(f"Ошибка кластеризации: {exc}")

        result = st.session_state.get("cluster_result")
        if result is not None:
            # --- Metrics row ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Кластеров (K)", result.n_clusters)
            sil_val = result.silhouette
            sil_str = f"{sil_val:.3f}" if not np.isnan(sil_val) else "н/д"
            c2.metric("Силуэтный коэффициент", sil_str)
            c3.metric("Инерция (SSE)", f"{result.inertia:,.1f}")
            c4.metric("Всего точек", f"{len(result.labels):,}")

            # --- Cluster size bar chart ---
            section_header("Размер кластеров")
            size_df = (
                pd.Series(result.labels, name="cluster")
                .value_counts()
                .sort_index()
                .reset_index()
            )
            size_df.columns = ["cluster", "count"]
            size_df["cluster"] = size_df["cluster"].astype(str)

            fig_size = px.bar(
                size_df,
                x="cluster",
                y="count",
                color="cluster",
                color_discrete_sequence=COLORS,
                labels={"cluster": "Кластер", "count": "Количество точек"},
                title="Распределение точек по кластерам",
                template="plotly_white",
            )
            fig_size.update_layout(showlegend=False, xaxis_title="Кластер", yaxis_title="Количество")
            st.plotly_chart(fig_size, use_container_width=True)

            # --- Silhouette explanation ---
            if not np.isnan(sil_val):
                if sil_val > 0.5:
                    quality = "хорошее"
                    icon = "✅"
                elif sil_val > 0.25:
                    quality = "приемлемое"
                    icon = "⚠️"
                else:
                    quality = "слабое"
                    icon = "❌"
                st.info(
                    f"{icon} **Силуэтный коэффициент: {sil_val:.3f}** — {quality} разделение кластеров.\n\n"
                    "- **> 0.5** — кластеры хорошо разделены\n"
                    "- **0.25–0.5** — разделение приемлемо, кластеры перекрываются\n"
                    "- **< 0.25** — кластеры слабо выражены, попробуйте другое K или другие признаки"
                )

# ---------------------------------------------------------------------------
# Tab 2 – Elbow method
# ---------------------------------------------------------------------------

with tab_elbow:
    section_header("Метод локтя")

    if not feature_cols:
        st.warning("Выберите хотя бы один признак в настройках выше.")
    else:
        max_k = st.slider(
            "Максимальное K для перебора",
            min_value=3,
            max_value=20,
            value=10,
            key="elbow_max_k",
            help="Алгоритм перебирает K от 2 до этого значения и строит кривую инерции и силуэта.",
        )

        if st.button("📐 Построить кривую локтя", key="btn_elbow", type="primary"):
            with st.spinner("Вычисление кривой локтя…"):
                try:
                    elbow_df = run_elbow(
                        df,
                        columns=feature_cols,
                        k_range=range(2, max_k + 1),
                        random_state=int(random_state),
                        scale=scale_features,
                    )
                    st.session_state["elbow_df"] = elbow_df
                    log_event("elbow_run", {"max_k": max_k, "features": feature_cols})
                except Exception as exc:
                    st.error(f"Ошибка: {exc}")

        elbow_df = st.session_state.get("elbow_df")
        if elbow_df is not None and not elbow_df.empty:
            best_k_row = elbow_df.loc[elbow_df["silhouette"].idxmax()]
            best_k = int(best_k_row["k"])

            # Dual-axis chart: inertia (left) + silhouette (right)
            fig_elbow = go.Figure()

            fig_elbow.add_trace(
                go.Scatter(
                    x=elbow_df["k"],
                    y=elbow_df["inertia"],
                    name="Инерция (SSE)",
                    line=dict(color="#636EFA", width=2),
                    mode="lines+markers",
                    yaxis="y1",
                )
            )

            fig_elbow.add_trace(
                go.Scatter(
                    x=elbow_df["k"],
                    y=elbow_df["silhouette"],
                    name="Силуэтный коэффициент",
                    line=dict(color="#EF553B", width=2, dash="dot"),
                    mode="lines+markers",
                    yaxis="y2",
                )
            )

            # Annotate best K
            fig_elbow.add_vline(
                x=best_k,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Рекомендуемое K={best_k}",
                annotation_position="top right",
            )

            fig_elbow.update_layout(
                title="Кривая локтя: инерция и силуэтный коэффициент",
                xaxis=dict(title="Количество кластеров (K)", dtick=1),
                yaxis=dict(title="Инерция (SSE)", title_font=dict(color="#636EFA")),
                yaxis2=dict(
                    title="Силуэтный коэффициент",
                    title_font=dict(color="#EF553B"),
                    overlaying="y",
                    side="right",
                ),
                legend=dict(x=0.01, y=0.99),
                template="plotly_white",
                hovermode="x unified",
            )

            st.plotly_chart(fig_elbow, use_container_width=True)

            st.success(f"Рекомендуемое K по силуэтному коэффициенту: **{best_k}**")

            st.markdown(
                """
**Как читать кривую локтя:**

- **Инерция (SSE)** — сумма квадратов расстояний от точек до центроидов их кластеров.
  Чем меньше, тем лучше. Ищите точку «локтя», где снижение инерции резко замедляется —
  это обычно оптимальное K.
- **Силуэтный коэффициент** — мера компактности и разделённости кластеров (диапазон −1…1).
  Выбирайте K с **максимальным** значением.
- Если два критерия дают разные K — ориентируйтесь на смысловую интерпретацию кластеров.
"""
            )

# ---------------------------------------------------------------------------
# Tab 3 – Cluster profiles
# ---------------------------------------------------------------------------

with tab_profiles:
    section_header("Профили кластеров")

    result = st.session_state.get("cluster_result")
    if result is None:
        st.info("Сначала запустите кластеризацию на вкладке **🎯 K-Means**.")
    else:
        profile_df = cluster_profiles(result)

        section_header("Сводная таблица профилей", "📊")
        mean_cols = [c for c in profile_df.columns if c.endswith("_mean")]

        # Styled dataframe – highlight max per mean column
        def highlight_max(s: pd.Series) -> list[str]:
            is_max = s == s.max()
            return ["background-color: #d4edda" if v else "" for v in is_max]

        styled = profile_df.style.apply(highlight_max, subset=mean_cols)
        st.dataframe(styled, use_container_width=True)

        # --- Radar chart ---
        section_header("Радарный график профилей кластеров", "🕸️")
        mean_df = profile_df[mean_cols].copy()
        mean_df.columns = [c.replace("_mean", "") for c in mean_df.columns]

        # Normalize 0–1 per feature for radar
        norm_df = mean_df.copy()
        for col in norm_df.columns:
            col_min, col_max = norm_df[col].min(), norm_df[col].max()
            if col_max > col_min:
                norm_df[col] = (norm_df[col] - col_min) / (col_max - col_min)
            else:
                norm_df[col] = 0.5

        categories = list(norm_df.columns)
        fig_radar = go.Figure()
        for idx, cluster_id in enumerate(norm_df.index):
            values = norm_df.loc[cluster_id].tolist()
            values_closed = values + [values[0]]  # close the polygon
            cats_closed = categories + [categories[0]]
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=values_closed,
                    theta=cats_closed,
                    fill="toself",
                    name=f"Кластер {cluster_id}",
                    line=dict(color=COLORS[idx % len(COLORS)]),
                    opacity=0.7,
                )
            )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Нормализованные профили кластеров (0–1)",
            template="plotly_white",
            legend=dict(x=1.05, y=0.5),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

        # --- Centroid heatmap ---
        section_header("Тепловая карта центроидов кластеров", "🌡️")
        centers = result.centers_df.copy()
        centers.index = [f"Кластер {i}" for i in centers.index]

        fig_heat = go.Figure(
            data=go.Heatmap(
                z=centers.values,
                x=centers.columns.tolist(),
                y=centers.index.tolist(),
                colorscale="RdBu_r",
                text=np.round(centers.values, 2),
                texttemplate="%{text}",
                hovertemplate="Кластер: %{y}<br>Признак: %{x}<br>Значение: %{z:.3f}<extra></extra>",
            )
        )
        fig_heat.update_layout(
            title="Центроиды кластеров (оригинальный масштаб)",
            xaxis_title="Признак",
            yaxis_title="Кластер",
            template="plotly_white",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 4 – PCA visualisation
# ---------------------------------------------------------------------------

with tab_pca:
    section_header("PCA-визуализация")

    result = st.session_state.get("cluster_result")
    if result is None:
        st.info("Сначала запустите кластеризацию на вкладке **🎯 K-Means**.")
    elif len(result.feature_cols) < 2:
        st.warning("PCA требует минимум 2 признака.")
    else:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        color_option = st.selectbox(
            "Раскрасить дополнительно по (необязательно)",
            options=["(кластер)"] + cat_cols,
            key="pca_color_col",
            help="Выберите категориальный признак для дополнительного цветового кодирования точек на PCA-графике.",
        )

        try:
            pca_df, explained = pca_transform(
                result.df_with_labels,
                columns=result.feature_cols,
                n_components=2,
                scale=result.scaled,
            )
        except Exception as exc:
            st.error(f"Ошибка PCA: {exc}")
            st.stop()

        # Align pca_df with df_with_labels
        plot_df = result.df_with_labels.loc[pca_df.index].copy()
        plot_df["pca_1"] = pca_df["pca_1"].values
        plot_df["pca_2"] = pca_df["pca_2"].values
        plot_df["cluster_str"] = plot_df["cluster"].astype("Int64").astype(str)

        pc1_pct = explained[0] * 100 if len(explained) > 0 else 0.0
        pc2_pct = explained[1] * 100 if len(explained) > 1 else 0.0

        if color_option == "(кластер)":
            color_col = "cluster_str"
            color_label = "Кластер"
            color_seq = COLORS
            color_discrete = {str(i): COLORS[i % len(COLORS)] for i in range(result.n_clusters)}
        else:
            color_col = color_option
            color_label = color_option
            color_seq = COLORS
            color_discrete = None

        fig_pca = px.scatter(
            plot_df.dropna(subset=["pca_1", "pca_2", "cluster_str"]),
            x="pca_1",
            y="pca_2",
            color=color_col,
            color_discrete_sequence=color_seq,
            color_discrete_map=color_discrete,
            labels={
                "pca_1": f"PC1 ({pc1_pct:.1f}%)",
                "pca_2": f"PC2 ({pc2_pct:.1f}%)",
                color_col: color_label,
            },
            title=(
                f"PCA-проекция кластеров "
                f"(PC1: {pc1_pct:.1f}%, PC2: {pc2_pct:.1f}%)"
            ),
            template="plotly_white",
            opacity=0.75,
        )
        fig_pca.update_traces(marker=dict(size=6))
        fig_pca.update_layout(legend_title_text=color_label)
        st.plotly_chart(fig_pca, use_container_width=True)

        total_explained = (pc1_pct + pc2_pct)
        st.caption(
            f"Суммарная объяснённая дисперсия (2 компоненты): **{total_explained:.1f}%**"
        )

# ---------------------------------------------------------------------------
# Tab 5 – Results
# ---------------------------------------------------------------------------

with tab_results:
    section_header("Результаты кластеризации")

    result = st.session_state.get("cluster_result")
    if result is None:
        st.info("Сначала запустите кластеризацию на вкладке **🎯 K-Means**.")
    else:
        # Preview of labelled data
        section_header("Данные с метками кластеров (первые 50 строк)", "📋")
        st.dataframe(result.df_with_labels.head(50), use_container_width=True)

        # Per-cluster statistics table
        section_header("Статистика по кластерам (среднее и стд. откл.)", "📈")
        try:
            valid = result.df_with_labels.dropna(subset=["cluster"])
            stats_mean = (
                valid.groupby("cluster")[result.feature_cols]
                .mean()
                .add_suffix(" (среднее)")
            )
            stats_std = (
                valid.groupby("cluster")[result.feature_cols]
                .std(ddof=1)
                .add_suffix(" (стд)")
            )
            stats_df = pd.concat([stats_mean, stats_std], axis=1).sort_index(axis=1)
            st.dataframe(stats_df.round(3), use_container_width=True)
        except Exception as exc:
            st.warning(f"Не удалось построить статистику: {exc}")

        # Download button
        st.divider()
        csv_bytes = result.df_with_labels.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Скачать CSV с кластерами",
            data=csv_bytes,
            file_name="kibad_clusters.csv",
            mime="text/csv",
            key="download_clusters",
        )

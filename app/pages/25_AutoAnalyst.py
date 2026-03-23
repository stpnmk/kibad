"""
pages/25_AutoAnalyst.py – One-click automatic full dataset analysis.
Generates a structured narrative report with charts, statistics, and actionable recommendations.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy import stats as scipy_stats
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False

from app.state import init_state, dataset_selectbox, get_active_df, store_prepared
from app.styles import inject_all_css, page_header, section_header
from app.components.ux import apply_recommendation_notification

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – Авто-аналитик", layout="wide")
init_state()
inject_all_css()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _boolean_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="bool").columns.tolist()


def _datetime_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _safe_skew(s: pd.Series) -> float:
    try:
        vals = pd.to_numeric(s, errors="coerce").dropna().values
        if len(vals) < 3:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if _SCIPY_OK:
                return float(scipy_stats.skew(vals))
            return float(pd.Series(vals).skew())
    except Exception:
        return 0.0


def _safe_kurtosis(s: pd.Series) -> float:
    try:
        vals = pd.to_numeric(s, errors="coerce").dropna().values
        if len(vals) < 4:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if _SCIPY_OK:
                return float(scipy_stats.kurtosis(vals))
            return float(pd.Series(vals).kurtosis())
    except Exception:
        return 0.0


def _iqr_outlier_mask(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=s.index)
    return (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)


def _zscore_outlier_count(s: pd.Series, threshold: float = 3.0) -> int:
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 4:
        return 0
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        return 0
    return int(((vals - mu).abs() > threshold * sigma).sum())


def _quality_badge(score: float) -> str:
    if score >= 80:
        return "Хорошее"
    if score >= 50:
        return "Требует внимания"
    return "Проблемное"


def _badge_color(score: float) -> str:
    if score >= 80:
        return "#28a745"
    if score >= 50:
        return "#ffc107"
    return "#dc3545"


def _card_html(text: str, color: str = "#17a2b8") -> str:
    return (
        f'<div style="background:{color}18;border-left:4px solid {color};'
        f'padding:10px 14px;border-radius:6px;margin-bottom:8px;">'
        f"{text}</div>"
    )


# ---------------------------------------------------------------------------
# Analysis cache key
# ---------------------------------------------------------------------------

def _cache_key(ds_name: str, df: pd.DataFrame) -> str:
    return f"{ds_name}__{df.shape[0]}__{df.shape[1]}"


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, depth: str, focus: list[str]) -> dict:
    """Run the full automated analysis and return structured results."""
    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    dt_cols = _datetime_cols(df)
    bool_cols = _boolean_cols(df)

    n_rows, n_cols = df.shape
    total_cells = n_rows * n_cols
    non_null_cells = int(df.notna().sum().sum())
    completeness = round(non_null_cells / total_cells * 100, 1) if total_cells > 0 else 100.0

    missing_per_col = df.isnull().mean() * 100
    n_dupes = int(df.duplicated().sum())
    pct_dupes = round(n_dupes / n_rows * 100, 1) if n_rows > 0 else 0.0

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
    high_card_cats = [
        c for c in cat_cols
        if df[c].nunique(dropna=True) > 0
        and df[c].nunique(dropna=True) / max(df[c].notna().sum(), 1) > 0.95
        and df[c].nunique(dropna=True) > 10
    ]

    # --- Outlier counts per numeric col (IQR) ---
    outlier_counts: dict[str, int] = {}
    for c in num_cols:
        mask = _iqr_outlier_mask(df[c])
        outlier_counts[c] = int(mask.sum())

    total_outlier_cells = sum(outlier_counts.values())
    pct_outliers = round(total_outlier_cells / max(n_rows * max(len(num_cols), 1), 1) * 100, 1)

    # --- Quality score ---
    pct_missing_overall = round(100 - completeness, 1)
    quality_score = max(0.0, min(100.0,
        100 - (pct_missing_overall * 0.4 + pct_dupes * 0.3 + pct_outliers * 0.3)
    ))

    # --- Numeric summary stats ---
    num_stats: list[dict] = []
    for c in num_cols:
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 2:
            continue
        num_stats.append({
            "column": c,
            "mean": round(float(s.mean()), 4),
            "median": round(float(s.median()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "max": round(float(s.max()), 4),
            "skew": round(_safe_skew(s), 3),
            "kurtosis": round(_safe_kurtosis(s), 3),
        })

    # --- Correlation matrix ---
    corr_matrix = None
    top_corr_pairs: list[tuple[str, str, float]] = []
    if len(num_cols) >= 2:
        sample = df[num_cols].dropna(how="all")
        if len(sample) > 50_000:
            sample = sample.sample(50_000, random_state=42)
        try:
            corr_matrix = sample.corr()
            pairs: list[tuple[str, str, float]] = []
            cols_list = corr_matrix.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    r = corr_matrix.iloc[i, j]
                    if pd.notna(r):
                        pairs.append((cols_list[i], cols_list[j], float(r)))
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            top_corr_pairs = pairs[:3]
        except Exception:
            pass

    # --- Anomaly top rows ---
    anomaly_rows: list[dict] = []
    if num_cols:
        scores: list[tuple[float, int, str, float, str]] = []
        for c in num_cols[:20]:
            s = pd.to_numeric(df[c], errors="coerce").dropna()
            if len(s) < 10:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                continue
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (s < lo) | (s > hi)
            for idx in s[mask].index:
                val = float(s[idx])
                score = abs(val - float(s.median())) / iqr
                scores.append((score, int(idx), c, round(val, 4), f"[{round(lo,2)}, {round(hi,2)}]"))
        scores.sort(key=lambda x: x[0], reverse=True)
        seen: set[int] = set()
        for sc, idx, col, val, rng in scores:
            if idx not in seen:
                anomaly_rows.append({"row": idx, "column": col, "value": val, "expected": rng})
                seen.add(idx)
            if len(anomaly_rows) >= 10:
                break

    # --- Z-score counts per col ---
    zscore_counts: dict[str, int] = {}
    for c in num_cols:
        zscore_counts[c] = _zscore_outlier_count(df[c])

    # --- Category info ---
    cat_info: dict[str, dict] = {}
    for c in cat_cols[:6]:
        vc = df[c].value_counts(dropna=True).head(10)
        n_unique = int(df[c].nunique(dropna=True))
        top_pct = round(float(vc.iloc[0]) / max(int(df[c].notna().sum()), 1) * 100, 1) if len(vc) > 0 else 0.0
        cat_info[c] = {
            "value_counts": vc,
            "n_unique": n_unique,
            "top_pct": top_pct,
        }

    # --- Time spike detection ---
    time_spike: dict | None = None
    if dt_cols and num_cols:
        try:
            dc = dt_cols[0]
            vc_col = num_cols[0]
            tmp = df[[dc, vc_col]].dropna().copy()
            tmp = tmp.sort_values(dc)
            tmp["_period"] = tmp[dc].dt.to_period("M")
            grp = tmp.groupby("_period")[vc_col].mean()
            if len(grp) >= 3:
                rolling_mean = grp.rolling(3, min_periods=1).mean()
                diff = (grp - rolling_mean).abs()
                spike_idx = diff.idxmax()
                spike_val = float(diff.max())
                mean_diff = float(diff.mean())
                if mean_diff > 0 and spike_val / mean_diff > 3:
                    time_spike = {
                        "period": str(spike_idx),
                        "col": vc_col,
                        "deviation": round(spike_val, 3),
                    }
        except Exception:
            pass

    return {
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dt_cols": dt_cols,
        "bool_cols": bool_cols,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "completeness": completeness,
        "missing_per_col": missing_per_col,
        "n_dupes": n_dupes,
        "pct_dupes": pct_dupes,
        "constant_cols": constant_cols,
        "high_card_cats": high_card_cats,
        "outlier_counts": outlier_counts,
        "quality_score": quality_score,
        "num_stats": num_stats,
        "corr_matrix": corr_matrix,
        "top_corr_pairs": top_corr_pairs,
        "cat_info": cat_info,
        "anomaly_rows": anomaly_rows,
        "zscore_counts": zscore_counts,
        "time_spike": time_spike,
        "pct_missing_overall": pct_missing_overall,
        "pct_dupes": pct_dupes,
        "pct_outliers": pct_outliers,
    }


# ---------------------------------------------------------------------------
# Rendering sections
# ---------------------------------------------------------------------------

def _render_section1(df: pd.DataFrame, res: dict) -> None:
    """Раздел 1: Обзор датасета"""
    with st.expander("1. Обзор датасета", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Строк", f"{res['n_rows']:,}")
        c2.metric("Столбцов", f"{res['n_cols']}")
        c3.metric("Полнота данных", f"{res['completeness']:.1f}%")
        mem = df.memory_usage(deep=True).sum()
        c4.metric("Память", f"{mem / 1024:.1f} KB" if mem < 1024**2 else f"{mem/1024**2:.2f} MB")

        st.markdown("**Полнота данных (% непустых ячеек)**")
        score = res["completeness"]
        bar_color = "#28a745" if score >= 80 else ("#ffc107" if score >= 50 else "#dc3545")
        st.markdown(
            f'<div style="background:#e9ecef;border-radius:6px;height:22px;width:100%;">'
            f'<div style="background:{bar_color};width:{score:.1f}%;height:100%;border-radius:6px;'
            f'display:flex;align-items:center;padding-left:8px;color:white;font-size:13px;font-weight:600;">'
            f'{score:.1f}%</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        nc = len(res["num_cols"])
        cc = len(res["cat_cols"])
        dc = len(res["dt_cols"])
        bc = len(res["bool_cols"])
        other = res["n_cols"] - nc - cc - dc - bc

        type_labels, type_vals = [], []
        if nc: type_labels.append("Числовые"); type_vals.append(nc)
        if cc: type_labels.append("Категориальные"); type_vals.append(cc)
        if dc: type_labels.append("Дата/время"); type_vals.append(dc)
        if bc: type_labels.append("Булевы"); type_vals.append(bc)
        if other > 0: type_labels.append("Прочие"); type_vals.append(other)

        if type_labels:
            col_left, col_right = st.columns([1, 2])
            with col_left:
                fig_pie = go.Figure(go.Pie(
                    labels=type_labels,
                    values=type_vals,
                    hole=0.4,
                    textinfo="label+value",
                    textfont_size=12,
                ))
                fig_pie.update_layout(
                    margin=dict(t=10, b=10, l=10, r=10),
                    height=220,
                    showlegend=False,
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_right:
                st.markdown("**Первые 5 строк датасета**")
                st.dataframe(df.head(5), use_container_width=True)


def _render_section2(df: pd.DataFrame, res: dict) -> None:
    """Раздел 2: Качество данных"""
    with st.expander("2. Качество данных", expanded=True):
        score = res["quality_score"]
        badge = _quality_badge(score)
        color = _badge_color(score)
        st.markdown(
            f'<div style="display:inline-block;background:{color};color:white;'
            f'font-weight:700;font-size:1.1rem;padding:6px 18px;border-radius:20px;margin-bottom:12px;">'
            f'Качество: {badge} ({score:.0f}/100)</div>',
            unsafe_allow_html=True,
        )

        # Missing values chart
        missing = res["missing_per_col"]
        missing_nonzero = missing[missing > 0].sort_values(ascending=False)
        if not missing_nonzero.empty:
            st.markdown("**Пропуски по столбцам (%)**")
            fig_miss = px.bar(
                x=missing_nonzero.values,
                y=missing_nonzero.index,
                orientation="h",
                color=missing_nonzero.values,
                color_continuous_scale=["#28a745", "#ffc107", "#dc3545"],
                labels={"x": "% пропусков", "y": "Столбец"},
            )
            fig_miss.update_layout(
                height=max(200, len(missing_nonzero) * 28),
                margin=dict(t=10, b=10, l=0, r=0),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("Пропуски отсутствуют")

        c1, c2 = st.columns(2)
        c1.metric("Дублирующихся строк", f"{res['n_dupes']:,} ({res['pct_dupes']:.1f}%)")
        c2.metric("Выбросов (IQR, числовые)", f"{res['pct_outliers']:.1f}% от ячеек")

        if res["constant_cols"]:
            st.warning(
                f"Константные столбцы (1 уникальное значение): "
                f"**{', '.join(res['constant_cols'])}** — рекомендуется удалить"
            )

        if res["high_card_cats"]:
            st.info(
                f"Вероятные идентификаторы (>95% уникальных): "
                f"**{', '.join(res['high_card_cats'])}**"
            )

        if res["outlier_counts"]:
            nonzero_out = {k: v for k, v in res["outlier_counts"].items() if v > 0}
            if nonzero_out:
                st.markdown("**Количество выбросов (IQR) по числовым столбцам**")
                fig_out = px.bar(
                    x=list(nonzero_out.keys()),
                    y=list(nonzero_out.values()),
                    labels={"x": "Столбец", "y": "Кол-во выбросов"},
                    color=list(nonzero_out.values()),
                    color_continuous_scale=["#ffc107", "#dc3545"],
                )
                fig_out.update_layout(
                    height=250,
                    margin=dict(t=10, b=10),
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_out, use_container_width=True)


def _render_section3(df: pd.DataFrame, res: dict) -> None:
    """Раздел 3: Числовые переменные"""
    if not res["num_cols"]:
        return
    with st.expander("3. Числовые переменные", expanded=True):
        if res["num_stats"]:
            stats_df = pd.DataFrame(res["num_stats"]).set_index("column")
            st.dataframe(stats_df, use_container_width=True)

        # Top skewed columns
        skewed = sorted(res["num_stats"], key=lambda d: abs(d["skew"]), reverse=True)[:3]
        for s in skewed:
            if abs(s["skew"]) > 1:
                direction = "правосторонний" if s["skew"] > 0 else "левосторонний"
                st.info(
                    f"Колонка **{s['column']}**: перекос {s['skew']:+.2f} ({direction}) — "
                    f"рассмотрите {'log' if s['skew']>0 else 'sqrt'}-трансформацию"
                )

        # Distribution grid (up to 8 cols)
        num_cols_plot = res["num_cols"][:8]
        if num_cols_plot:
            n_plots = len(num_cols_plot)
            n_cols_grid = 2
            n_rows_grid = (n_plots + 1) // 2
            fig_hist = make_subplots(
                rows=n_rows_grid,
                cols=n_cols_grid,
                subplot_titles=num_cols_plot,
            )
            for i, col in enumerate(num_cols_plot):
                row = i // 2 + 1
                col_idx = i % 2 + 1
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                fig_hist.add_trace(
                    go.Histogram(x=vals, nbinsx=30, name=col, showlegend=False,
                                 marker_color="#4a90d9"),
                    row=row, col=col_idx,
                )
            fig_hist.update_layout(
                height=200 * n_rows_grid,
                margin=dict(t=30, b=10),
                title_text="Распределения числовых переменных",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Correlation heatmap
        if res["corr_matrix"] is not None and len(res["num_cols"]) >= 2:
            st.markdown("**Матрица корреляций**")
            cm = res["corr_matrix"]
            fig_corr = go.Figure(go.Heatmap(
                z=cm.values,
                x=cm.columns.tolist(),
                y=cm.index.tolist(),
                colorscale="RdBu",
                zmid=0,
                zmin=-1,
                zmax=1,
                text=cm.round(2).values,
                texttemplate="%{text}",
                textfont_size=10,
            ))
            fig_corr.update_layout(
                height=max(300, len(cm) * 40 + 80),
                margin=dict(t=20, b=10),
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            if res["top_corr_pairs"]:
                st.markdown("**Топ-3 корреляции:**")
                for col_a, col_b, r in res["top_corr_pairs"]:
                    direction = "положительная" if r >= 0 else "отрицательная"
                    st.markdown(f"- `{col_a}` ↔ `{col_b}`: r = **{r:.3f}** ({direction})")


def _render_section4(df: pd.DataFrame, res: dict) -> None:
    """Раздел 4: Категориальные переменные"""
    if not res["cat_cols"]:
        return
    with st.expander("4. Категориальные переменные", expanded=True):
        for col, info in res["cat_info"].items():
            vc = info["value_counts"]
            top_pct = info["top_pct"]
            n_unique = info["n_unique"]
            st.markdown(f"**{col}** — {n_unique} уникальных значений")
            if top_pct > 80:
                st.warning(f"Низкая вариативность: «{vc.index[0]}» занимает {top_pct:.1f}%")
            col_l, col_r = st.columns([3, 1])
            with col_l:
                fig_bar = px.bar(
                    x=vc.values,
                    y=vc.index.astype(str),
                    orientation="h",
                    labels={"x": "Кол-во", "y": col},
                    color=vc.values,
                    color_continuous_scale=["#4a90d9", "#003d8f"],
                )
                fig_bar.update_layout(
                    height=max(180, len(vc) * 25),
                    margin=dict(t=10, b=10),
                    coloraxis_showscale=False,
                    yaxis={"autorange": "reversed"},
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            with col_r:
                st.metric("Топ-значение", f"{vc.index[0]}")
                st.metric("Доля топ-1", f"{top_pct:.1f}%")


def _render_section5(df: pd.DataFrame, res: dict) -> None:
    """Раздел 5: Аномалии"""
    with st.expander("5. Аномалии", expanded=True):
        if res["anomaly_rows"]:
            st.markdown("**Топ аномальных строк (IQR)**")
            anomaly_df = pd.DataFrame(res["anomaly_rows"])
            st.dataframe(anomaly_df, use_container_width=True)
        else:
            st.success("Аномальных строк не обнаружено")

        nonzero_z = {k: v for k, v in res["zscore_counts"].items() if v > 0}
        if nonzero_z:
            st.markdown("**Экстремальные значения (|z| > 3) по столбцам:**")
            for col, cnt in sorted(nonzero_z.items(), key=lambda x: x[1], reverse=True):
                st.markdown(f"- `{col}`: **{cnt}** значений")
        else:
            st.info("Экстремальных значений (|z|>3) не обнаружено")

        if res["time_spike"]:
            sp = res["time_spike"]
            st.warning(
                f"Аномальный всплеск в «{sp['col']}» в период **{sp['period']}** "
                f"(отклонение от скользящего среднего: {sp['deviation']:.2f})"
            )


def _render_section6(df: pd.DataFrame, res: dict, ds_name: str = "") -> None:
    """Раздел 6: Умные рекомендации"""
    with st.expander("6. Умные рекомендации", expanded=True):
        recommendations: list[tuple[str, str, str]] = []  # (text, color, type)

        missing = res["missing_per_col"]
        has_missing = (missing > 5).any()
        if has_missing:
            cols_miss = missing[missing > 5].index.tolist()
            recommendations.append((
                f"Заполните пропуски (>5%) в столбцах **{', '.join(cols_miss[:3])}**"
                f"{'...' if len(cols_miss) > 3 else ''} — страница **2. Подготовка** → Заполнение пропусков",
                "#dc3545", "warning",
            ))

        has_dupes = res["n_dupes"] > 0
        if has_dupes:
            recommendations.append((
                f"Удалите **{res['n_dupes']}** дублирующихся строк — страница **2. Подготовка** → Дедупликация",
                "#dc3545", "warning",
            ))

        for col_a, col_b, r in res["top_corr_pairs"]:
            if abs(r) > 0.85:
                recommendations.append((
                    f"Высокая корреляция между **{col_a}** и **{col_b}** (r={r:.2f}) — "
                    "возможна мультиколлинеарность, рассмотрите исключение одного столбца",
                    "#856404", "info",
                ))

        for stat in res["num_stats"]:
            if abs(stat["skew"]) > 2:
                tr = "log" if stat["skew"] > 0 else "sqrt"
                recommendations.append((
                    f"Рассмотрите **{tr}-трансформацию** для колонки **{stat['column']}** "
                    f"(перекос {stat['skew']:+.2f})",
                    "#0c5460", "info",
                ))

        if len(res["num_cols"]) >= 2 and len(res["cat_cols"]) >= 1:
            recommendations.append((
                "Хорошая структура для группового анализа — перейдите на **3. Группировка**",
                "#155724", "success",
            ))

        if res["dt_cols"] and res["num_cols"]:
            recommendations.append((
                f"Временной ряд обнаружен (столбец **{res['dt_cols'][0]}**) — "
                "перейдите на **7. Временные ряды**",
                "#155724", "success",
            ))

        recommendations.append((
            "Постройте визуализации на **20. Конструктор графиков**",
            "#155724", "success",
        ))

        color_map = {
            "warning": "#dc3545",
            "info": "#17a2b8",
            "success": "#28a745",
        }

        if not recommendations:
            st.success("Датасет выглядит чисто — рекомендаций нет")
            return

        for text, color, rtype in recommendations:
            bg = color_map.get(rtype, "#17a2b8")
            st.markdown(_card_html(text, bg), unsafe_allow_html=True)

        # Auto-apply buttons for actionable recommendations
        if has_missing and ds_name:
            if st.button("⚡ Заполнить пропуски", key="aa_fill_na"):
                df_before = df.copy()
                df_new = df.copy()
                for col in df_new.select_dtypes(include="number").columns:
                    df_new[col] = df_new[col].fillna(df_new[col].median())
                for col in df_new.select_dtypes(include="object").columns:
                    mode_val = df_new[col].mode()
                    if len(mode_val) > 0:
                        df_new[col] = df_new[col].fillna(mode_val[0])
                store_prepared(ds_name, df_new)
                apply_recommendation_notification("Заполнение пропусков", df_before, df_new, ds_name)
                cache_key = f"{ds_name}__{len(df_before)}__{len(df_before.columns)}"
                st.session_state.get("auto_analysis", {}).pop(cache_key, None)
                st.rerun()

        if has_dupes and ds_name:
            if st.button("⚡ Удалить дубликаты", key="aa_dedup"):
                df_before = df.copy()
                df_new = df.drop_duplicates().reset_index(drop=True)
                store_prepared(ds_name, df_new)
                apply_recommendation_notification("Удаление дубликатов", df_before, df_new, ds_name)
                cache_key = f"{ds_name}__{len(df_before)}__{len(df_before.columns)}"
                st.session_state.get("auto_analysis", {}).pop(cache_key, None)
                st.rerun()


def _render_section7(df: pd.DataFrame, res: dict) -> None:
    """Раздел 7: Экспорт"""
    with st.expander("7. Экспорт", expanded=False):
        if res["num_stats"]:
            stats_df = pd.DataFrame(res["num_stats"])
            csv_buf = io.StringIO()
            stats_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="Скачать отчёт (CSV)",
                data=csv_buf.getvalue().encode("utf-8"),
                file_name="auto_analysis_summary.csv",
                mime="text/csv",
            )

        # Build recommendations text
        lines: list[str] = ["Рекомендации Авто-аналитика KIBAD", "=" * 40, ""]
        missing = res["missing_per_col"]
        if (missing > 5).any():
            cols_miss = missing[missing > 5].index.tolist()
            lines.append(f"[ВНИМАНИЕ] Пропуски >5% в: {', '.join(cols_miss)}")
        if res["n_dupes"] > 0:
            lines.append(f"[ВНИМАНИЕ] Удалите {res['n_dupes']} дублирующихся строк")
        for col_a, col_b, r in res["top_corr_pairs"]:
            if abs(r) > 0.85:
                lines.append(f"[КОРРЕЛЯЦИЯ] {col_a} ↔ {col_b}: r={r:.2f} — возможна мультиколлинеарность")
        for stat in res["num_stats"]:
            if abs(stat["skew"]) > 2:
                lines.append(f"[ТРАНСФОРМАЦИЯ] Колонка {stat['column']}: перекос {stat['skew']:+.2f}")
        if res["dt_cols"] and res["num_cols"]:
            lines.append(f"[ТРЕНД] Временной ряд: {res['dt_cols'][0]} — откройте страницу 7. Временные ряды")
        lines.append("[ВИЗУАЛИЗАЦИЯ] Постройте графики на странице 20. Конструктор графиков")

        txt_content = "\n".join(lines)
        st.download_button(
            label="Скачать список рекомендаций (TXT)",
            data=txt_content.encode("utf-8"),
            file_name="auto_analysis_recommendations.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

page_header("25. Авто-аналитик", "Полный анализ данных в один клик — без настройки", "🤖")

# Dataset selector
ds_name = dataset_selectbox("Датасет", key="auto_analyst_ds")

if ds_name is None:
    st.stop()

df_active = get_active_df()
if df_active is None or df_active.empty:
    st.warning("Выбранный датасет пуст или недоступен.")
    st.stop()

# Config
col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
with col_cfg1:
    depth = st.radio(
        "Глубина анализа",
        ["Быстрый (30 сек)", "Стандартный (1–2 мин)", "Глубокий (3–5 мин)"],
        index=1,
        key="aa_depth",
    )
with col_cfg2:
    focus = st.multiselect(
        "Фокус",
        ["Качество данных", "Распределения", "Корреляции", "Аномалии", "Тренды", "Сегменты"],
        default=["Качество данных", "Распределения", "Корреляции", "Аномалии"],
        key="aa_focus",
    )
with col_cfg3:
    lang = st.radio("Язык отчёта", ["Русский", "English"], index=0, key="aa_lang")

# Run button (full-width trick)
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    run_btn = st.button("▶ Запустить авто-анализ", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Cache key check
# ---------------------------------------------------------------------------

if "auto_analysis" not in st.session_state:
    st.session_state["auto_analysis"] = {}

cache_key = _cache_key(ds_name, df_active)
cached = st.session_state["auto_analysis"].get(cache_key)

# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------

if run_btn or cached is not None:
    if run_btn or cached is None:
        t0 = time.perf_counter()
        with st.spinner("Выполняется авто-анализ…"):
            result = run_analysis(df_active, depth=depth, focus=focus)
        elapsed = time.perf_counter() - t0
        st.session_state["auto_analysis"][cache_key] = {
            "result": result,
            "elapsed": elapsed,
        }
        cached = st.session_state["auto_analysis"][cache_key]

    result = cached["result"]
    elapsed = cached["elapsed"]

    st.success(f"Анализ завершён за {elapsed:.1f} сек")

    _render_section1(df_active, result)
    _render_section2(df_active, result)
    _render_section3(df_active, result)
    _render_section4(df_active, result)
    _render_section5(df_active, result)
    _render_section6(df_active, result, ds_name=ds_name)
    _render_section7(df_active, result)

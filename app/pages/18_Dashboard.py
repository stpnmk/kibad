"""
pages/18_Dashboard.py – KPI Dashboard page for KIBAD.

Non-technical managers can see key metrics at a glance — aggregated KPI cards
with period-over-period deltas, traffic-light (RAG) thresholds, trend charts,
and a one-click summary CSV export.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
from typing import Literal

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.state import init_state, dataset_selectbox, get_dataset
from app.components.ux import active_dataset_warnings
from app.styles import inject_all_css, page_header, section_header
from app.theme import RAG_GREEN, RAG_YELLOW, RAG_RED, RAG_GRAY

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – KPI Дашборд", layout="wide")
init_state()
inject_all_css()
active_dataset_warnings()

page_header("18. KPI Дашборд", "Мониторинг метрик с трендами и светофором", "📊")

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

AGG_LABELS = {
    "Сумма": "sum",
    "Среднее": "mean",
    "Медиана": "median",
    "Максимум": "max",
    "Минимум": "min",
}

PERIOD_FREQ: dict[str, str] = {
    "День": "D",
    "Неделя": "W",
    "Месяц": "ME",
    "Квартал": "QE",
    "Год": "YE",
}

RAG_COLORS = {
    "green": RAG_GREEN,
    "yellow": RAG_YELLOW,
    "red": RAG_RED,
    "none": RAG_GRAY,
}

RAG_LABELS = {
    "green": "Зелёный",
    "yellow": "Жёлтый",
    "red": "Красный",
    "none": "—",
}


def fmt_num(v: float | None) -> str:
    """Format number with Russian-style thousands separator."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:,.2f}".replace(",", "\u202f")  # narrow no-break space


def detect_date_cols(df: pd.DataFrame) -> list[str]:
    """Return columns that look like dates."""
    date_kws = {"date", "дата", "period", "период", "time", "время", "month", "месяц", "year", "год", "week"}
    found: list[str] = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(col)
        elif any(kw in col.lower() for kw in date_kws):
            found.append(col)
    return found


def apply_agg(series: pd.Series, agg: str) -> float:
    """Apply aggregation function name to a series."""
    fn = {"sum": series.sum, "mean": series.mean, "median": series.median,
          "max": series.max, "min": series.min}
    return float(fn.get(agg, series.sum)())


def rag_status(value: float, thresholds: dict) -> str:
    """Return 'green' / 'yellow' / 'red' / 'none' based on thresholds."""
    if not thresholds or thresholds.get("direction") is None:
        return "none"
    direction = thresholds["direction"]  # "lower_is_better" | "higher_is_better"
    tg = thresholds.get("green")
    ty = thresholds.get("yellow")
    tr = thresholds.get("red")

    if direction == "lower_is_better":
        if tg is not None and value <= tg:
            return "green"
        if ty is not None and value <= ty:
            return "yellow"
        if tr is not None:
            return "red"
    else:  # higher_is_better
        if tg is not None and value >= tg:
            return "green"
        if ty is not None and value >= ty:
            return "yellow"
        if tr is not None:
            return "red"
    return "none"


def metric_card_html(label: str, value_str: str, delta_str: str | None,
                     delta_positive: bool | None, border_color: str) -> str:
    """Build a styled HTML metric card."""
    delta_html = ""
    if delta_str:
        arrow = "▲" if delta_positive else "▼"
        color = RAG_GREEN if delta_positive else RAG_RED
        delta_html = f'<div style="font-size:0.85rem;color:{color};margin-top:2px">{arrow} {delta_str}</div>'

    return (
        f'<div style="border-left:5px solid {border_color};background:#fafafa;'
        f'border-radius:8px;padding:14px 18px;box-shadow:0 1px 4px rgba(0,0,0,.08);height:100%">'
        f'<div style="font-size:0.78rem;color:#666;text-transform:uppercase;letter-spacing:.04em">{label}</div>'
        f'<div style="font-size:1.6rem;font-weight:700;color:#1a1a2e;line-height:1.2">{value_str}</div>'
        f'{delta_html}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Dataset & Setup
# ---------------------------------------------------------------------------

chosen = dataset_selectbox("Датасет", key="dash_ds_sel")
if not chosen:
    st.stop()

df_raw = get_dataset(chosen)
if df_raw is None or df_raw.empty:
    st.info("📥 Выберите датасет или загрузите новый на странице **[1. Данные](pages/1_Data.py)**")
    st.stop()

df = df_raw.copy()
num_cols = df.select_dtypes(include="number").columns.tolist()

if not num_cols:
    st.error("❌ Датасет не содержит числовых колонок. Перейдите на страницу **[2. Подготовка](pages/2_Prepare.py)** → шаг «Типы данных» и приведите нужные колонки к числовому типу.")
    st.stop()

with st.expander("⚙️ Настройки дашборда", expanded=True):
    # --- Row 1: Date, period, aggregation ---
    date_candidates = detect_date_cols(df)
    date_options = ["(нет)"] + date_candidates + [c for c in df.columns if c not in date_candidates]

    col_d1, col_d2, col_d3, col_d4 = st.columns(4)
    with col_d1:
        date_col_sel = st.selectbox("Дата / Период", options=date_options, key="dash_date_col")
    date_col: str | None = None if date_col_sel == "(нет)" else date_col_sel

    # Try to parse as datetime if not already
    if date_col and not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        try:
            df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
        except Exception:
            st.warning(f"Не удалось распознать «{date_col}» как дату. Дата не используется.")
            date_col = None

    period_label = "Месяц"
    period_n_current = 3
    period_n_prev = 3
    if date_col:
        with col_d2:
            period_label = st.selectbox(
                "Гранулярность периода",
                options=list(PERIOD_FREQ.keys()),
                index=2,
                key="dash_period_gran",
            )
        with col_d3:
            period_n_current = st.number_input("Текущий период (N)", min_value=1, max_value=36, value=1, step=1, key="dash_cur_n")
        with col_d4:
            period_n_prev = st.number_input("Предыдущий период (N)", min_value=1, max_value=36, value=1, step=1, key="dash_prev_n")

    # --- Row 2: Metrics, aggregation ---
    col_m1, col_m2, col_m3 = st.columns([4, 2, 2])
    with col_m1:
        selected_metrics = st.multiselect(
            "Числовые колонки (метрики)",
            options=num_cols,
            default=num_cols[:min(8, len(num_cols))],
            key="dash_metrics",
        )
    with col_m2:
        agg_label = st.selectbox(
            "Агрегация",
            options=list(AGG_LABELS.keys()),
            index=0,
            key="dash_agg",
        )
        agg_fn = AGG_LABELS[agg_label]
    with col_m3:
        st.caption(f"Строк: **{len(df):,}** · Колонок: **{len(df.columns)}**")

# ---------------------------------------------------------------------------
# Guard: no metrics selected
# ---------------------------------------------------------------------------

if not selected_metrics:
    st.info("Выберите хотя бы одну числовую колонку в настройках выше.")
    st.stop()

# ---------------------------------------------------------------------------
# Compute aggregates
# ---------------------------------------------------------------------------

freq = PERIOD_FREQ.get(period_label, "ME")

# Build period-sliced dataframes
cur_df: pd.DataFrame | None = None
prev_df: pd.DataFrame | None = None
timeline_df: pd.DataFrame | None = None

if date_col and date_col in df.columns:
    df_dated = df.dropna(subset=[date_col]).copy()
    df_dated = df_dated.sort_values(date_col)

    if not df_dated.empty:
        # Build period index
        df_dated["_period"] = df_dated[date_col].dt.to_period(freq)
        all_periods = sorted(df_dated["_period"].unique())

        if len(all_periods) >= 1:
            n_cur = int(period_n_current)
            n_prev = int(period_n_prev)

            cur_periods = all_periods[-n_cur:]
            prev_end_idx = len(all_periods) - n_cur - 1
            prev_start_idx = max(0, prev_end_idx - n_prev + 1)
            prev_periods = all_periods[prev_start_idx: prev_end_idx + 1] if prev_end_idx >= 0 else []

            cur_df = df_dated[df_dated["_period"].isin(cur_periods)]
            prev_df = df_dated[df_dated["_period"].isin(prev_periods)] if prev_periods else None

        # Timeline aggregation for trend charts
        agg_dict = {m: agg_fn for m in selected_metrics if m in df_dated.columns}
        if agg_dict:
            try:
                timeline_df = (
                    df_dated.set_index(date_col)
                    .groupby(pd.Grouper(freq=freq))[list(agg_dict.keys())]
                    .agg(agg_fn)
                    .reset_index()
                )
                timeline_df = timeline_df[timeline_df[list(agg_dict.keys())].notna().any(axis=1)]
            except Exception:
                timeline_df = None

if cur_df is None:
    cur_df = df  # fallback: use entire dataset

# Compute scalar values per metric
results: dict[str, dict] = {}
for m in selected_metrics:
    if m not in cur_df.columns:
        continue
    cur_val = apply_agg(cur_df[m].dropna(), agg_fn)
    prev_val: float | None = None
    delta_pct: float | None = None

    if prev_df is not None and m in prev_df.columns and not prev_df.empty:
        prev_val = apply_agg(prev_df[m].dropna(), agg_fn)
        if prev_val != 0 and not np.isnan(prev_val):
            delta_pct = (cur_val - prev_val) / abs(prev_val) * 100
        else:
            delta_pct = None

    results[m] = {
        "cur": cur_val,
        "prev": prev_val,
        "delta_pct": delta_pct,
    }

# ---------------------------------------------------------------------------
# RAG thresholds — stored in session state
# ---------------------------------------------------------------------------

if "dash_thresholds" not in st.session_state:
    st.session_state["dash_thresholds"] = {}

thresholds: dict[str, dict] = st.session_state["dash_thresholds"]

# ---------------------------------------------------------------------------
# Section: Traffic Lights Configuration
# ---------------------------------------------------------------------------

with st.expander("🚦 Настройка порогов (RAG)", expanded=False):
    st.markdown(
        "Задайте пороги для каждой метрики. Карточки и сводная таблица будут окрашены "
        "в зелёный / жёлтый / красный в зависимости от текущего значения."
    )
    for m in selected_metrics:
        if m not in results:
            continue
        st.markdown(f"---\n**{m}**  (текущее: `{fmt_num(results[m]['cur'])}`)")
        t_cols = st.columns([2, 1, 1, 1, 1])
        with t_cols[0]:
            direction = st.radio(
                "Направление",
                options=["higher_is_better", "lower_is_better"],
                format_func=lambda x: "Больше — лучше" if x == "higher_is_better" else "Меньше — лучше",
                index=0,
                key=f"dash_dir_{m}",
                horizontal=True,
            )
        with t_cols[1]:
            tg = st.number_input("Порог Зелёный", value=thresholds.get(m, {}).get("green") or 0.0,
                                 key=f"dash_tg_{m}", format="%.4f")
        with t_cols[2]:
            ty = st.number_input("Порог Жёлтый", value=thresholds.get(m, {}).get("yellow") or 0.0,
                                 key=f"dash_ty_{m}", format="%.4f")
        with t_cols[3]:
            tr = st.number_input("Порог Красный", value=thresholds.get(m, {}).get("red") or 0.0,
                                 key=f"dash_tr_{m}", format="%.4f")
        with t_cols[4]:
            enabled = st.checkbox("Включить", value=thresholds.get(m, {}).get("enabled", False),
                                  key=f"dash_en_{m}")

        thresholds[m] = {
            "direction": direction,
            "green": float(tg),
            "yellow": float(ty),
            "red": float(tr),
            "enabled": enabled,
        }

    st.session_state["dash_thresholds"] = thresholds

# ---------------------------------------------------------------------------
# Compute RAG for all metrics
# ---------------------------------------------------------------------------

rag_map: dict[str, str] = {}
for m in selected_metrics:
    if m not in results:
        rag_map[m] = "none"
        continue
    t = thresholds.get(m, {})
    if t.get("enabled"):
        rag_map[m] = rag_status(results[m]["cur"], t)
    else:
        rag_map[m] = "none"

# ---------------------------------------------------------------------------
# Section: KPI Cards
# ---------------------------------------------------------------------------

st.markdown("### 📊 KPI Карточки")

CARDS_PER_ROW = 4
metric_list = [m for m in selected_metrics if m in results]
n_metrics = len(metric_list)

for row_start in range(0, n_metrics, CARDS_PER_ROW):
    row_metrics = metric_list[row_start: row_start + CARDS_PER_ROW]
    cols = st.columns(len(row_metrics))
    for col, m in zip(cols, row_metrics):
        r = results[m]
        cur_val = r["cur"]
        delta_pct = r["delta_pct"]
        border_color = RAG_COLORS.get(rag_map.get(m, "none"), RAG_COLORS["none"])

        delta_str: str | None = None
        delta_positive: bool | None = None
        if delta_pct is not None:
            sign = "+" if delta_pct >= 0 else ""
            delta_str = f"{sign}{delta_pct:.1f}% к пред. периоду"
            delta_positive = delta_pct >= 0

        card_html = metric_card_html(
            label=f"{m} ({agg_label})",
            value_str=fmt_num(cur_val),
            delta_str=delta_str,
            delta_positive=delta_positive,
            border_color=border_color,
        )
        with col:
            st.markdown(card_html, unsafe_allow_html=True)

st.markdown("")  # spacing

# ---------------------------------------------------------------------------
# Section: RAG Summary Table
# ---------------------------------------------------------------------------

rag_rows = []
for m in metric_list:
    r = results[m]
    status = rag_map.get(m, "none")
    rag_rows.append({
        "Метрика": m,
        "Текущее значение": fmt_num(r["cur"]),
        "Предыдущее значение": fmt_num(r["prev"]) if r["prev"] is not None else "—",
        "Δ %": (f"{'+' if (r['delta_pct'] or 0) >= 0 else ''}{r['delta_pct']:.1f}%"
                if r["delta_pct"] is not None else "—"),
        "Статус": RAG_LABELS.get(status, "—"),
        "_status": status,
    })

if any(row["_status"] != "none" for row in rag_rows):
    st.markdown("### 🚦 Сводная таблица RAG")

    def _style_rag_row(row: pd.Series) -> list[str]:
        bg_map = {"green": "#d4edda", "yellow": "#fff3cd", "red": "#ffd7d7", "none": ""}
        bg = bg_map.get(row["_status"], "")
        return [f"background-color: {bg}"] * len(row)

    rag_display = pd.DataFrame(rag_rows).drop(columns=["_status"])
    try:
        styled_rag = pd.DataFrame(rag_rows).style.apply(_style_rag_row, axis=1)
        styled_rag = styled_rag.hide(axis="columns", subset=["_status"])  # type: ignore[arg-type]
        st.dataframe(styled_rag, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(rag_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Section: Trend Charts
# ---------------------------------------------------------------------------

if timeline_df is not None and not timeline_df.empty and date_col:
    st.markdown("### 📈 Динамика по периодам")

    # Determine boundary between previous and current periods
    boundary_date: pd.Timestamp | None = None
    if cur_df is not None and "_period" in cur_df.columns and not cur_df.empty:
        try:
            first_cur_period = cur_df["_period"].min()
            boundary_date = first_cur_period.to_timestamp()
        except Exception:
            boundary_date = None

    CHARTS_PER_ROW = 2
    chart_metrics = [m for m in metric_list if m in timeline_df.columns]

    for row_start in range(0, len(chart_metrics), CHARTS_PER_ROW):
        row_m = chart_metrics[row_start: row_start + CHARTS_PER_ROW]
        chart_cols = st.columns(len(row_m))
        for col, m in zip(chart_cols, row_m):
            with col:
                plot_df = timeline_df[[date_col, m]].dropna()
                if plot_df.empty:
                    st.caption(f"Нет данных для «{m}»")
                    continue

                fig = px.bar(
                    plot_df,
                    x=date_col,
                    y=m,
                    title=f"{m} · {agg_label}",
                    labels={date_col: "Период", m: agg_label},
                    template="plotly_white",
                    color_discrete_sequence=["#4c8bf5"],
                )

                # Add trend line
                try:
                    fig.add_scatter(
                        x=plot_df[date_col],
                        y=plot_df[m],
                        mode="lines",
                        line=dict(color="#ff7043", width=2),
                        name="Тренд",
                    )
                except Exception:
                    pass

                # Add vertical boundary line
                if boundary_date is not None:
                    try:
                        fig.add_vline(
                            x=boundary_date.value / 1e6,  # plotly expects ms for datetime
                            line_dash="dash",
                            line_color="#666",
                            annotation_text="Текущий период",
                            annotation_position="top left",
                            annotation_font_size=10,
                        )
                    except Exception:
                        try:
                            fig.add_vline(
                                x=str(boundary_date),
                                line_dash="dash",
                                line_color="#666",
                                annotation_text="Текущий период",
                                annotation_position="top left",
                                annotation_font_size=10,
                            )
                        except Exception:
                            pass

                fig.update_layout(
                    height=220,
                    margin=dict(l=10, r=10, t=40, b=10),
                    showlegend=False,
                    xaxis_title="",
                    yaxis_title="",
                    title_font_size=13,
                )
                st.plotly_chart(fig, use_container_width=True)

elif not date_col:
    st.info("Для отображения трендов выберите колонку с датой в настройках выше.")

# ---------------------------------------------------------------------------
# Section: Download
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 💾 Экспорт")

export_rows = []
for m in metric_list:
    r = results[m]
    status = rag_map.get(m, "none")
    export_rows.append({
        "metric": m,
        "aggregation": agg_label,
        "current_value": r["cur"],
        "previous_value": r["prev"] if r["prev"] is not None else "",
        "delta_pct": (f"{r['delta_pct']:.2f}" if r["delta_pct"] is not None else ""),
        "rag_status": RAG_LABELS.get(status, "—"),
    })

export_df = pd.DataFrame(export_rows)
csv_bytes = export_df.to_csv(index=False).encode("utf-8-sig")  # BOM for Excel

st.download_button(
    label="⬇️ Скачать сводку (CSV)",
    data=csv_bytes,
    file_name="kibad_kpi_summary.csv",
    mime="text/csv",
    key="dash_download",
)

st.caption(
    f"Датасет: **{chosen}** · Агрегация: **{agg_label}** · "
    + (f"Период: **{period_label}**, текущий N={period_n_current}, пред. N={period_n_prev}" if date_col else "Без разбивки по периодам")
)

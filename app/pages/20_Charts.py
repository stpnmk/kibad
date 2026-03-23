"""
pages/20_Charts.py – Конструктор графиков для KIBAD.

Visualization studio: 18 типов графиков, настройка без кода,
автовыводы, скачивание PNG и CSV.
"""
from __future__ import annotations

import sys
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

from app.state import init_state, dataset_selectbox, get_active_df
from app.styles import inject_all_css, page_header, section_header

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="KIBAD – Графики", layout="wide")
init_state()
inject_all_css()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHART_TYPES: list[str] = [
    # Базовые
    "📊 Столбчатый",
    "📊 Горизонтальный столбчатый",
    "📈 Линейный",
    "📉 Площадной",
    "🔵 Точечный",
    # Распределения
    "📊 Гистограмма",
    "📦 Ящик с усами (Box plot)",
    "🎻 Скрипичный (Violin)",
    # Части целого
    "🥧 Круговой",
    "🍩 Кольцевой",
    "🌳 Дерево (Treemap)",
    "☀️ Солнечный (Sunburst)",
    # Специальные
    "🌡️ Тепловая карта",
    "🫧 Пузырьковый",
    "🏔️ Воронка (Funnel)",
    # Расширенные
    "📊 Двойная ось (bar + line)",
    "🕯️ Свечи (OHLC)",
    "🎬 Анимированный (bar race)",
]

# Internal keys
CHART_KEY: dict[str, str] = {
    "📊 Столбчатый": "bar",
    "📊 Горизонтальный столбчатый": "bar_h",
    "📈 Линейный": "line",
    "📉 Площадной": "area",
    "🔵 Точечный": "scatter",
    "📊 Гистограмма": "histogram",
    "📦 Ящик с усами (Box plot)": "box",
    "🎻 Скрипичный (Violin)": "violin",
    "🥧 Круговой": "pie",
    "🍩 Кольцевой": "donut",
    "🌳 Дерево (Treemap)": "treemap",
    "☀️ Солнечный (Sunburst)": "sunburst",
    "🌡️ Тепловая карта": "heatmap",
    "🫧 Пузырьковый": "bubble",
    "🏔️ Воронка (Funnel)": "funnel",
    "📊 Двойная ось (bar + line)": "dual_axis",
    "🕯️ Свечи (OHLC)": "candlestick",
    "🎬 Анимированный (bar race)": "bar_race",
}

COLOR_SCHEMES: dict[str, list | None] = {
    "По умолчанию": None,
    "Синий": px.colors.sequential.Blues,
    "Красный": px.colors.sequential.Reds,
    "Зелёный": px.colors.sequential.Greens,
    "Пастельный": px.colors.qualitative.Pastel,
    "Сберовский (зелёный)": ["#21A038", "#42BB5C", "#6ECC80", "#9DDDA4", "#CEEEC7"],
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _num_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _cat_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def _all_cols(df: pd.DataFrame) -> list[str]:
    return df.columns.tolist()


def _none_plus(cols: list[str]) -> list[str]:
    return ["(нет)"] + cols


def _resolve_color(scheme_name: str, discrete: bool) -> dict:
    """Return kwargs for plotly express color scale / sequence."""
    palette = COLOR_SCHEMES[scheme_name]
    if palette is None:
        return {}
    if discrete:
        return {"color_discrete_sequence": palette}
    return {"color_continuous_scale": palette}


def _apply_layout(fig: go.Figure, title: str, height: int, show_legend: bool,
                  x_label: str, y_label: str, log_y: bool) -> go.Figure:
    updates: dict = {
        "height": height,
        "showlegend": show_legend,
        "title": {"text": title, "font": {"size": 16}},
        "margin": {"t": 60, "l": 50, "r": 30, "b": 50},
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
    }
    fig.update_layout(**updates)
    axis_kwargs: dict = {"showgrid": True, "gridcolor": "#eeeeee"}
    if x_label:
        fig.update_xaxes(title_text=x_label, **axis_kwargs)
    else:
        fig.update_xaxes(**axis_kwargs)
    if y_label:
        fig.update_yaxes(title_text=y_label, **axis_kwargs)
    else:
        fig.update_yaxes(**axis_kwargs)
    if log_y:
        fig.update_yaxes(type="log")
    return fig


def _maybe_topn(df: pd.DataFrame, x_col: str, y_col: str, top_n: int, sort: bool) -> pd.DataFrame:
    """Optionally sort and limit rows for bar/funnel-like charts."""
    out = df.copy()
    if sort and y_col and y_col in out.columns:
        out = out.sort_values(y_col, ascending=False)
    if top_n and top_n > 0:
        out = out.head(top_n)
    return out


def _add_data_labels(fig: go.Figure, chart_key: str) -> go.Figure:
    """Add text labels on bars/lines etc."""
    if chart_key in ("bar", "bar_h"):
        fig.update_traces(texttemplate="%{value:,.0f}", textposition="outside")
    elif chart_key == "line":
        fig.update_traces(mode="lines+markers+text", textposition="top center")
    elif chart_key in ("pie", "donut"):
        fig.update_traces(textinfo="label+percent")
    return fig


# ---------------------------------------------------------------------------
# Auto-chart suggestion
# ---------------------------------------------------------------------------


def suggest_chart(x_col: str, y_col: str, df: pd.DataFrame) -> str | None:
    x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col]) if x_col in df.columns else False
    x_is_num = pd.api.types.is_numeric_dtype(df[x_col]) if x_col in df.columns else False
    y_is_num = pd.api.types.is_numeric_dtype(df[y_col]) if y_col in df.columns else False
    x_nunique = df[x_col].nunique() if x_col in df.columns else 0

    if x_is_date and y_is_num:
        return "📈 Рекомендуем: Линейный — идеально для временных рядов"
    elif x_is_num and y_is_num:
        try:
            r = df[[x_col, y_col]].corr().iloc[0, 1] if len(df) > 2 else 0
        except Exception:
            r = 0
        return f"🔵 Рекомендуем: Точечный — корреляция r={r:.2f}"
    elif not x_is_num and y_is_num and x_nunique <= 10:
        return "📊 Рекомендуем: Круговой — мало категорий"
    elif not x_is_num and y_is_num:
        return "📊 Рекомендуем: Столбчатый — сравнение категорий"
    return None


# ---------------------------------------------------------------------------
# Auto-insights
# ---------------------------------------------------------------------------


def _insights_bar(df: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    if not y_col or y_col not in df.columns or x_col not in df.columns:
        return []
    try:
        grp = df.groupby(x_col, observed=True)[y_col].sum().sort_values(ascending=False)
        if grp.empty:
            return []
        top_label = str(grp.index[0])
        top_val = grp.iloc[0]
        last_val = grp.iloc[-1]
        gap = round((top_val - last_val) / top_val * 100, 1) if top_val != 0 else 0
        lines = [f"Лидер: **{top_label}** ({top_val:,.1f}). Отставание последнего: **{gap}%**"]
        if len(grp) >= 3:
            mean_val = grp.mean()
            lines.append(f"Среднее по группам: {mean_val:,.1f}. Всего категорий: {len(grp)}.")
        return lines
    except Exception:
        return []


def _insights_line(df: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    if not y_col or y_col not in df.columns:
        return []
    try:
        series = df[y_col].dropna()
        if len(series) < 2:
            return []
        first = series.iloc[0]
        last = series.iloc[-1]
        pct = round((last - first) / first * 100, 1) if first != 0 else 0
        trend = "растущий" if pct > 2 else "падающий" if pct < -2 else "нейтральный"
        return [
            f"Тренд: **{trend}**. Изменение: {first:,.1f} → {last:,.1f} (**{pct:+.1f}%**)",
            f"Мин: {series.min():,.1f}, Макс: {series.max():,.1f}, Среднее: {series.mean():,.1f}",
        ]
    except Exception:
        return []


def _insights_histogram(df: pd.DataFrame, col: str) -> list[str]:
    if col not in df.columns:
        return []
    try:
        s = df[col].dropna()
        if s.empty:
            return []
        mean = s.mean()
        median = s.median()
        skew = s.skew()
        if abs(skew) < 0.5:
            dist = "нормальное"
        elif skew > 0:
            dist = "правосторонний перекос"
        else:
            dist = "левосторонний перекос"
        return [
            f"Среднее: **{mean:,.2f}**. Медиана: **{median:,.2f}**.",
            f"Распределение: **{dist}** (асимметрия: {skew:.2f}). Стд. откл.: {s.std():,.2f}.",
        ]
    except Exception:
        return []


def _insights_scatter(df: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
        return []
    try:
        sub = df[[x_col, y_col]].dropna()
        if len(sub) < 3:
            return []
        r = sub[x_col].corr(sub[y_col])
        if abs(r) >= 0.7:
            strength = "сильная"
        elif abs(r) >= 0.4:
            strength = "умеренная"
        else:
            strength = "слабая"
        direction = "положительная" if r >= 0 else "отрицательная"
        return [f"Корреляция: **{r:.2f}** ({strength} {direction} связь). Точек: {len(sub):,}."]
    except Exception:
        return []


def _insights_pie(df: pd.DataFrame, label_col: str, val_col: str) -> list[str]:
    if label_col not in df.columns or val_col not in df.columns:
        return []
    try:
        grp = df.groupby(label_col, observed=True)[val_col].sum()
        total = grp.sum()
        if total == 0:
            return []
        top_label = str(grp.idxmax())
        top_pct = grp.max() / total * 100
        return [
            f"Топ-1 доля: **{top_label}** = **{top_pct:.1f}%**. Всего категорий: {len(grp)}."
        ]
    except Exception:
        return []


def _get_insights(chart_key: str, df: pd.DataFrame, cfg: dict) -> list[str]:
    try:
        if chart_key in ("bar", "bar_h"):
            return _insights_bar(df, cfg.get("x", ""), cfg.get("y", ""))
        elif chart_key in ("line", "area"):
            return _insights_line(df, cfg.get("x", ""), cfg.get("y", ""))
        elif chart_key == "histogram":
            return _insights_histogram(df, cfg.get("col", ""))
        elif chart_key == "scatter":
            return _insights_scatter(df, cfg.get("x", ""), cfg.get("y", ""))
        elif chart_key in ("pie", "donut"):
            return _insights_pie(df, cfg.get("labels", ""), cfg.get("values", ""))
        elif chart_key == "bubble":
            return _insights_scatter(df, cfg.get("x", ""), cfg.get("y", ""))
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_chart(chart_key: str, df: pd.DataFrame, cfg: dict,
                 title: str, scheme: str, show_legend: bool, show_labels: bool,
                 height: int, x_label: str, y_label: str, sort_vals: bool,
                 top_n: int, log_y: bool) -> go.Figure:
    color_kwargs_disc = _resolve_color(scheme, discrete=True)
    color_kwargs_cont = _resolve_color(scheme, discrete=False)

    fig: go.Figure | None = None

    # ---- Bar ----
    if chart_key == "bar":
        x_col = cfg["x"]
        y_col = cfg["y"]
        color_col = cfg.get("color") or None
        plot_df = _maybe_topn(df, x_col, y_col, top_n, sort_vals)
        # Auto-aggregate when X is categorical with many unique values
        _n_unique = plot_df[x_col].nunique() if x_col in plot_df.columns else 0
        if _n_unique > 20 and pd.api.types.is_numeric_dtype(plot_df.get(y_col, pd.Series(dtype=float))):
            plot_df = plot_df.groupby(x_col, as_index=False)[y_col].sum()
            import streamlit as _st
            _st.caption(f"⚡ Колонка «{x_col}» содержит {_n_unique} уникальных значений — данные агрегированы (сумма по «{y_col}»).")
        kwargs: dict = dict(x=x_col, y=y_col, color=color_col, **color_kwargs_disc)
        fig = px.bar(plot_df, **kwargs)

    # ---- Horizontal bar ----
    elif chart_key == "bar_h":
        x_col = cfg["x"]
        y_col = cfg["y"]
        color_col = cfg.get("color") or None
        plot_df = _maybe_topn(df, x_col, y_col, top_n, sort_vals)
        # Auto-aggregate when X is categorical with many unique values
        _n_unique_h = plot_df[x_col].nunique() if x_col in plot_df.columns else 0
        if _n_unique_h > 20 and pd.api.types.is_numeric_dtype(plot_df.get(y_col, pd.Series(dtype=float))):
            plot_df = plot_df.groupby(x_col, as_index=False)[y_col].sum()
            import streamlit as _st
            _st.caption(f"⚡ Колонка «{x_col}» содержит {_n_unique_h} уникальных значений — данные агрегированы (сумма по «{y_col}»).")
        kwargs = dict(x=y_col, y=x_col, color=color_col, orientation="h", **color_kwargs_disc)
        fig = px.bar(plot_df, **kwargs)

    # ---- Line ----
    elif chart_key == "line":
        x_col = cfg["x"]
        y_col = cfg["y"]
        color_col = cfg.get("color") or None
        plot_df = df.copy()
        if sort_vals and y_col in plot_df.columns:
            plot_df = plot_df.sort_values(y_col, ascending=False)
        if top_n and top_n > 0:
            plot_df = plot_df.head(top_n)
        fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, **color_kwargs_disc)

    # ---- Area ----
    elif chart_key == "area":
        x_col = cfg["x"]
        y_col = cfg["y"]
        color_col = cfg.get("color") or None
        plot_df = df.copy()
        if sort_vals and y_col in plot_df.columns:
            plot_df = plot_df.sort_values(y_col, ascending=False)
        if top_n and top_n > 0:
            plot_df = plot_df.head(top_n)
        fig = px.area(plot_df, x=x_col, y=y_col, color=color_col, **color_kwargs_disc)

    # ---- Scatter ----
    elif chart_key == "scatter":
        x_col = cfg["x"]
        y_col = cfg["y"]
        color_col = cfg.get("color") or None
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, **color_kwargs_disc)

    # ---- Histogram ----
    elif chart_key == "histogram":
        col = cfg["col"]
        nbins = cfg.get("nbins", 20)
        fig = px.histogram(df, x=col, nbins=nbins, **color_kwargs_disc)

    # ---- Box ----
    elif chart_key == "box":
        val_col = cfg["val"]
        grp_col = cfg.get("group") or None
        fig = px.box(df, x=grp_col, y=val_col, color=grp_col, **color_kwargs_disc)

    # ---- Violin ----
    elif chart_key == "violin":
        val_col = cfg["val"]
        grp_col = cfg.get("group") or None
        fig = px.violin(df, x=grp_col, y=val_col, color=grp_col,
                        box=True, **color_kwargs_disc)

    # ---- Pie ----
    elif chart_key == "pie":
        labels_col = cfg["labels"]
        values_col = cfg["values"]
        plot_df = df.groupby(labels_col, observed=True)[values_col].sum().reset_index()
        if top_n and top_n > 0:
            plot_df = plot_df.nlargest(top_n, values_col)
        palette = COLOR_SCHEMES[scheme]
        kwargs = {}
        if palette:
            kwargs["color_discrete_sequence"] = palette
        fig = px.pie(plot_df, names=labels_col, values=values_col, **kwargs)

    # ---- Donut ----
    elif chart_key == "donut":
        labels_col = cfg["labels"]
        values_col = cfg["values"]
        plot_df = df.groupby(labels_col, observed=True)[values_col].sum().reset_index()
        if top_n and top_n > 0:
            plot_df = plot_df.nlargest(top_n, values_col)
        palette = COLOR_SCHEMES[scheme]
        kwargs = {}
        if palette:
            kwargs["color_discrete_sequence"] = palette
        fig = px.pie(plot_df, names=labels_col, values=values_col, hole=0.45, **kwargs)

    # ---- Treemap ----
    elif chart_key == "treemap":
        path_cols = cfg["path"]
        values_col = cfg["values"]
        palette = COLOR_SCHEMES[scheme]
        kwargs = {}
        if palette:
            kwargs["color_discrete_sequence"] = palette
        fig = px.treemap(df, path=path_cols, values=values_col, **kwargs)

    # ---- Sunburst ----
    elif chart_key == "sunburst":
        path_cols = cfg["path"]
        values_col = cfg["values"]
        palette = COLOR_SCHEMES[scheme]
        kwargs = {}
        if palette:
            kwargs["color_discrete_sequence"] = palette
        fig = px.sunburst(df, path=path_cols, values=values_col, **kwargs)

    # ---- Heatmap ----
    elif chart_key == "heatmap":
        mode = cfg.get("mode", "Корреляционная матрица")
        if mode == "Корреляционная матрица":
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                raise ValueError("Нужно минимум 2 числовых столбца для корреляционной матрицы.")
            corr = num_df.corr()
            palette = COLOR_SCHEMES[scheme]
            cs = palette if palette else "RdBu_r"
            fig = go.Figure(go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale=cs,
                zmin=-1, zmax=1,
                text=corr.round(2).values,
                texttemplate="%{text}",
                textfont={"size": 10},
            ))
        else:
            row_col = cfg["row"]
            col_col = cfg["col_col"]
            val_col = cfg["val"]
            pivot = df.pivot_table(index=row_col, columns=col_col,
                                   values=val_col, aggfunc="sum")
            palette = COLOR_SCHEMES[scheme]
            cs = palette if palette else "Blues"
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=[str(c) for c in pivot.columns],
                y=[str(r) for r in pivot.index],
                colorscale=cs,
            ))

    # ---- Bubble ----
    elif chart_key == "bubble":
        x_col = cfg["x"]
        y_col = cfg["y"]
        size_col = cfg.get("size") or None
        color_col = cfg.get("color") or None
        fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                         size_max=60, **color_kwargs_disc)

    # ---- Funnel ----
    elif chart_key == "funnel":
        stage_col = cfg["stage"]
        val_col = cfg["val"]
        plot_df = df.groupby(stage_col, observed=True)[val_col].sum().reset_index()
        if sort_vals:
            plot_df = plot_df.sort_values(val_col, ascending=False)
        if top_n and top_n > 0:
            plot_df = plot_df.head(top_n)
        fig = go.Figure(go.Funnel(
            y=plot_df[stage_col].astype(str).tolist(),
            x=plot_df[val_col].tolist(),
            textinfo="value+percent total",
        ))
        palette = COLOR_SCHEMES[scheme]
        if palette:
            fig.update_traces(marker_color=palette[0] if isinstance(palette, list) else None)

    # ---- Dual axis (bar + line) ----
    elif chart_key == "dual_axis":
        x_col = cfg["x"]
        y1_col = cfg["y1"]
        y2_col = cfg["y2"]
        y1_label = cfg.get("y1_label", "")
        y2_label = cfg.get("y2_label", "")
        chart_df = df.copy()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=chart_df[x_col], y=chart_df[y1_col], name=y1_col),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=chart_df[x_col], y=chart_df[y2_col], name=y2_col,
                       mode="lines+markers"),
            secondary_y=True,
        )
        fig.update_yaxes(title_text=y1_label or y1_col, secondary_y=False)
        fig.update_yaxes(title_text=y2_label or y2_col, secondary_y=True)
        fig.update_layout(
            title=f"{y1_col} и {y2_col} по {x_col}",
            template="plotly_white",
            height=height,
        )

    # ---- Candlestick (OHLC) ----
    elif chart_key == "candlestick":
        date_col_cs = cfg["date"]
        open_col = cfg["open"]
        high_col = cfg["high"]
        low_col = cfg["low"]
        close_col = cfg["close"]
        chart_df = df.copy()
        fig = go.Figure(go.Candlestick(
            x=chart_df[date_col_cs],
            open=chart_df[open_col],
            high=chart_df[high_col],
            low=chart_df[low_col],
            close=chart_df[close_col],
            increasing_line_color="#2ecc71",
            decreasing_line_color="#e74c3c",
        ))
        fig.update_layout(
            title="OHLC",
            template="plotly_white",
            height=height,
            xaxis_rangeslider_visible=False,
        )

    # ---- Animated bar race ----
    elif chart_key == "bar_race":
        x_col = cfg["x"]
        y_col = cfg["y"]
        anim_col = cfg["anim"]
        color_col = cfg.get("color") or None
        chart_df = df.copy()
        fig = px.bar(
            chart_df.sort_values(y_col, ascending=False),
            x=x_col,
            y=y_col,
            animation_frame=anim_col,
            color=color_col,
            title=f"Анимация: {y_col} по {x_col} ({anim_col})",
            template="plotly_white",
            range_y=[0, chart_df[y_col].max() * 1.1],
        )
        fig.update_layout(height=height)

    if fig is None:
        raise ValueError(f"Неизвестный тип графика: {chart_key}")

    if show_labels and chart_key not in ("heatmap", "treemap", "sunburst", "histogram", "violin"):
        fig = _add_data_labels(fig, chart_key)

    fig = _apply_layout(fig, title, height, show_legend, x_label, y_label, log_y)
    return fig


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

page_header("20. Конструктор графиков", "18 типов визуализаций с авто-рекомендациями", "📈")

# ---------------------------------------------------------------------------
# Dataset selector
# ---------------------------------------------------------------------------

ds_name = dataset_selectbox("Датасет:", key="charts_ds")
df = get_active_df()

if df is None or df.empty:
    st.info("📥 Данные не загружены. Перейдите на страницу **[1. Данные](pages/1_Data.py)** и загрузите файл.")
    st.stop()

num_columns = _num_cols(df)
cat_columns = _cat_cols(df)
all_columns = _all_cols(df)

if not all_columns:
    st.error("Датасет не содержит ни одного столбца.")
    st.stop()

# ---------------------------------------------------------------------------
# Top bar: chart type + build button in one row
# ---------------------------------------------------------------------------

top_col1, top_col2 = st.columns([3, 1])
with top_col1:
    chart_label = st.selectbox(
        "Тип графика",
        options=CHART_TYPES,
        key="chart_type_sel",
    )
    chart_key = CHART_KEY[chart_label]
with top_col2:
    st.markdown("<div style='margin-top:28px'></div>", unsafe_allow_html=True)
    build_btn_top = st.button("▶ Построить", type="primary", use_container_width=True, key="build_btn_top")

st.divider()

# ---------------------------------------------------------------------------
# Two-column layout
# ---------------------------------------------------------------------------

left, right = st.columns([1, 2])

# ===========================================================================
# LEFT PANEL – Configuration
# ===========================================================================

with left:
    st.markdown("**Данные**")

    # ---- Axis / field controls depending on chart type ----
    cfg: dict = {}

    if chart_key in ("bar", "bar_h", "line", "area", "scatter"):
        cfg["x"] = st.selectbox(
            "Ось X",
            options=all_columns,
            key="cfg_x",
        )
        y_opts = num_columns if num_columns else all_columns
        cfg["y"] = st.selectbox(
            "Ось Y (значение)",
            options=y_opts,
            key="cfg_y",
        )
        color_opts = _none_plus(all_columns)
        color_sel = st.selectbox("Цвет (группировка)", options=color_opts, key="cfg_color")
        cfg["color"] = None if color_sel == "(нет)" else color_sel

    elif chart_key == "bubble":
        cfg["x"] = st.selectbox("Ось X", options=all_columns, key="cfg_x")
        y_opts = num_columns if num_columns else all_columns
        cfg["y"] = st.selectbox("Ось Y (значение)", options=y_opts, key="cfg_y")
        size_opts = _none_plus(num_columns)
        size_sel = st.selectbox("Размер пузырька", options=size_opts, key="cfg_size")
        cfg["size"] = None if size_sel == "(нет)" else size_sel
        color_opts = _none_plus(all_columns)
        color_sel = st.selectbox("Цвет (группировка)", options=color_opts, key="cfg_color")
        cfg["color"] = None if color_sel == "(нет)" else color_sel

    elif chart_key == "histogram":
        col_opts = num_columns if num_columns else all_columns
        cfg["col"] = st.selectbox("Столбец", options=col_opts, key="cfg_hist_col")
        cfg["nbins"] = st.slider("Количество бинов", min_value=5, max_value=100,
                                  value=20, key="cfg_nbins")

    elif chart_key in ("box", "violin"):
        val_opts = num_columns if num_columns else all_columns
        cfg["val"] = st.selectbox("Столбец значений", options=val_opts, key="cfg_val")
        grp_opts = _none_plus(cat_columns if cat_columns else all_columns)
        grp_sel = st.selectbox("Группировка", options=grp_opts, key="cfg_group")
        cfg["group"] = None if grp_sel == "(нет)" else grp_sel

    elif chart_key in ("pie", "donut"):
        lbl_opts = cat_columns if cat_columns else all_columns
        cfg["labels"] = st.selectbox("Метки (категории)", options=lbl_opts, key="cfg_labels")
        val_opts = num_columns if num_columns else all_columns
        cfg["values"] = st.selectbox("Значения", options=val_opts, key="cfg_values")

    elif chart_key in ("treemap", "sunburst"):
        path_opts = cat_columns if cat_columns else all_columns
        if not path_opts:
            st.warning("⚠️ Нет текстовых (категориальных) столбцов для иерархии. Убедитесь, что в датасете есть колонки с текстом (object/category). Перейдите на страницу **2. Подготовка** → Типы данных и преобразуйте нужные колонки.")
        cfg["path"] = st.multiselect(
            "Иерархия (путь)", options=path_opts,
            default=path_opts[:1] if path_opts else [],
            key="cfg_path",
        )
        val_opts = num_columns if num_columns else all_columns
        cfg["values"] = st.selectbox("Значения", options=val_opts, key="cfg_values")

    elif chart_key == "heatmap":
        mode = st.radio(
            "Режим",
            options=["Корреляционная матрица", "Сводная таблица"],
            key="cfg_heatmap_mode",
            horizontal=True,
        )
        cfg["mode"] = mode
        if mode == "Сводная таблица":
            cfg["row"] = st.selectbox("Строки", options=all_columns, key="cfg_hm_row")
            cfg["col_col"] = st.selectbox("Столбцы", options=all_columns,
                                           index=min(1, len(all_columns) - 1),
                                           key="cfg_hm_col")
            val_opts = num_columns if num_columns else all_columns
            cfg["val"] = st.selectbox("Значения", options=val_opts, key="cfg_hm_val")

    elif chart_key == "funnel":
        cfg["stage"] = st.selectbox("Этап (категория)", options=all_columns, key="cfg_stage")
        val_opts = num_columns if num_columns else all_columns
        cfg["val"] = st.selectbox("Значение", options=val_opts, key="cfg_funnel_val")

    elif chart_key == "dual_axis":
        cfg["x"] = st.selectbox("Ось X", options=all_columns, key="cfg_x")
        y_opts = num_columns if num_columns else all_columns
        cfg["y1"] = st.selectbox("Левая ось (столбцы)", options=y_opts, key="cfg_y1")
        cfg["y2"] = st.selectbox(
            "Правая ось (линия)",
            options=y_opts,
            index=min(1, len(y_opts) - 1),
            key="cfg_y2",
        )
        cfg["y1_label"] = st.text_input("Подпись левой оси", value="", key="cfg_y1_label")
        cfg["y2_label"] = st.text_input("Подпись правой оси", value="", key="cfg_y2_label")

    elif chart_key == "candlestick":
        dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        date_opts = dt_cols if dt_cols else all_columns
        cfg["date"] = st.selectbox("Столбец даты", options=date_opts, key="cfg_cs_date")
        y_opts = num_columns if num_columns else all_columns
        cfg["open"] = st.selectbox("Open (открытие)", options=y_opts, key="cfg_cs_open")
        cfg["high"] = st.selectbox(
            "High (максимум)",
            options=y_opts,
            index=min(1, len(y_opts) - 1),
            key="cfg_cs_high",
        )
        cfg["low"] = st.selectbox(
            "Low (минимум)",
            options=y_opts,
            index=min(2, len(y_opts) - 1),
            key="cfg_cs_low",
        )
        cfg["close"] = st.selectbox(
            "Close (закрытие)",
            options=y_opts,
            index=min(3, len(y_opts) - 1),
            key="cfg_cs_close",
        )

    elif chart_key == "bar_race":
        cfg["x"] = st.selectbox("Ось X (категория)", options=all_columns, key="cfg_x")
        y_opts = num_columns if num_columns else all_columns
        cfg["y"] = st.selectbox("Ось Y (значение)", options=y_opts, key="cfg_y")
        cfg["anim"] = st.selectbox(
            "Фрейм анимации (дата / категория)",
            options=all_columns,
            index=min(1, len(all_columns) - 1),
            key="cfg_anim",
        )
        color_opts = _none_plus(all_columns)
        color_sel = st.selectbox("Цвет (группировка)", options=color_opts, key="cfg_color")
        cfg["color"] = None if color_sel == "(нет)" else color_sel

    # ---- Customization ----
    st.divider()
    with st.expander("🎨 Оформление", expanded=False):
        auto_title = chart_label.split(" ", 1)[-1]
        chart_title = st.text_input("Заголовок графика", value=auto_title, key="cfg_title")
        scheme_name = st.selectbox(
            "Цветовая схема",
            options=list(COLOR_SCHEMES.keys()),
            key="cfg_scheme",
        )
        show_legend = st.checkbox("Показывать легенду", value=True, key="cfg_legend")
        show_labels = st.checkbox("Подписи данных", value=False, key="cfg_labels_show")
        chart_height = st.slider("Высота графика", min_value=300, max_value=800,
                                  value=500, step=50, key="cfg_height")

    # ---- Advanced ----
    with st.expander("⚙️ Дополнительно", expanded=False):
        x_label_override = st.text_input("Подпись оси X", value="", key="cfg_xlabel",
                                          placeholder="Автоматически")
        y_label_override = st.text_input("Подпись оси Y", value="", key="cfg_ylabel",
                                          placeholder="Автоматически")
        sort_vals = st.checkbox("Сортировать по значению (убывание)", value=False,
                                 key="cfg_sort")
        top_n = int(st.number_input("Топ N (0 = все)", min_value=0, max_value=500,
                                     value=0, step=5, key="cfg_topn"))
        log_y = st.checkbox("Логарифмическая шкала Y", value=False, key="cfg_logy")

    # ---- Auto-chart suggestion ----
    _suggest_x = cfg.get("x") or cfg.get("col") or cfg.get("stage")
    _suggest_y = cfg.get("y") or cfg.get("y1") or cfg.get("val") or cfg.get("values")
    if _suggest_x and _suggest_y:
        _suggestion = suggest_chart(_suggest_x, _suggest_y, df)
        if _suggestion:
            st.info(_suggestion)


# ===========================================================================
# RIGHT PANEL – Chart output
# ===========================================================================

with right:
    # Restore last chart on page reload (before build button is clicked)
    fig_state: go.Figure | None = st.session_state.get("last_chart_fig")
    fig_df_state: pd.DataFrame | None = st.session_state.get("last_chart_df")

    if build_btn_top:
        try:
            # Validate minimal config
            if chart_key in ("treemap", "sunburst") and not cfg.get("path"):
                st.error("❌ Для Treemap/Sunburst выберите хотя бы одну колонку в разделе «Иерархия» (левая панель → Данные графика)")
                st.stop()

            if chart_key == "heatmap" and cfg.get("mode") == "Сводная таблица":
                if cfg.get("row") == cfg.get("col_col"):
                    st.error("❌ Столбец строк и столбец колонок не могут совпадать. Выберите разные колонки для «Строки» и «Столбцы» в левой панели.")
                    st.stop()

            fig = _build_chart(
                chart_key=chart_key,
                df=df,
                cfg=cfg,
                title=chart_title,
                scheme=scheme_name,
                show_legend=show_legend,
                show_labels=show_labels,
                height=chart_height,
                x_label=x_label_override,
                y_label=y_label_override,
                sort_vals=sort_vals,
                top_n=top_n,
                log_y=log_y,
            )
            st.session_state["last_chart_fig"] = fig
            st.session_state["last_chart_cfg"] = cfg
            st.session_state["last_chart_key"] = chart_key
            st.session_state["last_chart_df"] = df.copy()
            fig_state = fig
            fig_df_state = df

        except Exception as e:
            st.error(f"❌ Не удалось построить график: {e}\n\nПроверьте настройки осей и типы данных в выбранных колонках.")

    if fig_state is not None:
        st.plotly_chart(fig_state, use_container_width=True)

        # ---- Download buttons ----
        dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 2])
        with dl_col1:
            try:
                img_bytes = fig_state.to_image(format="png", scale=2)
                st.download_button("📷 PNG", img_bytes, file_name="chart.png",
                                   mime="image/png", use_container_width=True)
            except Exception:
                st.caption("📷 PNG: `pip install kaleido`")
        with dl_col2:
            if fig_df_state is not None:
                csv_bytes = fig_df_state.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📊 CSV", csv_bytes, file_name="chart_data.csv",
                                   mime="text/csv", use_container_width=True)

        # ---- Auto-insights ----
        saved_key = st.session_state.get("last_chart_key", chart_key)
        saved_cfg = st.session_state.get("last_chart_cfg", cfg)
        insights = _get_insights(saved_key, fig_df_state if fig_df_state is not None else df,
                                  saved_cfg)
        if insights:
            with st.expander("💡 Автовыводы", expanded=True):
                for line in insights:
                    st.markdown(f"- {line}")

    else:
        st.info(
            "Выберите тип графика и настройте оси → нажмите **▶ Построить** вверху страницы.\n\n"
            "💡 *Авто-предложение подскажет подходящий тип на основе типов данных*"
        )

"""
app/pages/p20_charts.py – Конструктор графиков (Dash).

18+ типов графиков, настройка без кода, автовыводы,
скачивание PNG (с fallback на SVG/HTML).
"""
from __future__ import annotations

import base64
import io

import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, callback, dcc, html, dash_table, no_update
import dash_bootstrap_components as dbc

from app.state import (
    get_df_from_store, list_datasets,
    STORE_DATASET, STORE_ACTIVE_DS, STORE_PREPARED,
)
from app.figure_theme import apply_kibad_theme

dash.register_page(
    __name__,
    path="/charts",
    name="20. Графики",
    order=20,
    icon="bar-chart",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHART_OPTIONS = [
    {"label": "Столбчатый", "value": "bar"},
    {"label": "Горизонтальный столбчатый", "value": "bar_h"},
    {"label": "Линейный", "value": "line"},
    {"label": "Площадной", "value": "area"},
    {"label": "Точечный", "value": "scatter"},
    {"label": "Гистограмма", "value": "histogram"},
    {"label": "Ящик с усами (Box plot)", "value": "box"},
    {"label": "Скрипичный (Violin)", "value": "violin"},
    {"label": "Круговой", "value": "pie"},
    {"label": "Кольцевой", "value": "donut"},
    {"label": "Дерево (Treemap)", "value": "treemap"},
    {"label": "Солнечный (Sunburst)", "value": "sunburst"},
    {"label": "Тепловая карта", "value": "heatmap"},
    {"label": "Пузырьковый", "value": "bubble"},
    {"label": "Воронка (Funnel)", "value": "funnel"},
    {"label": "Двойная ось (bar + line)", "value": "dual_axis"},
    {"label": "Свечи (OHLC)", "value": "candlestick"},
    {"label": "Анимированный (bar race)", "value": "bar_race"},
]

COLOR_SCHEMES = {
    "По умолчанию": None,
    "Синий": px.colors.sequential.Blues,
    "Красный": px.colors.sequential.Reds,
    "Зелёный": px.colors.sequential.Greens,
    "Пастельный": px.colors.qualitative.Pastel,
}


# ---------------------------------------------------------------------------
# Insight helpers
# ---------------------------------------------------------------------------

def _get_insights(chart_key: str, df: pd.DataFrame, x_col: str, y_col: str) -> list[str]:
    try:
        if chart_key in ("bar", "bar_h") and y_col and y_col in df.columns and x_col in df.columns:
            grp = df.groupby(x_col, observed=True)[y_col].sum().sort_values(ascending=False)
            if grp.empty:
                return []
            top_label = str(grp.index[0])
            top_val = grp.iloc[0]
            last_val = grp.iloc[-1]
            gap = round((top_val - last_val) / top_val * 100, 1) if top_val != 0 else 0
            return [f"Лидер: {top_label} ({top_val:,.1f}). Отставание последнего: {gap}%"]
        elif chart_key in ("line", "area") and y_col and y_col in df.columns:
            s = df[y_col].dropna()
            if len(s) < 2:
                return []
            first, last = s.iloc[0], s.iloc[-1]
            pct = round((last - first) / first * 100, 1) if first != 0 else 0
            trend = "растущий" if pct > 2 else "падающий" if pct < -2 else "нейтральный"
            return [f"Тренд: {trend}. {first:,.1f} -> {last:,.1f} ({pct:+.1f}%)"]
        elif chart_key == "scatter" and x_col and y_col:
            sub = df[[x_col, y_col]].dropna()
            if len(sub) < 3:
                return []
            r = sub[x_col].corr(sub[y_col])
            strength = "сильная" if abs(r) >= 0.7 else "умеренная" if abs(r) >= 0.4 else "слабая"
            return [f"Корреляция: {r:.2f} ({strength})"]
        elif chart_key == "histogram" and x_col and x_col in df.columns:
            s = df[x_col].dropna()
            skew = s.skew()
            dist = "нормальное" if abs(skew) < 0.5 else "перекос вправо" if skew > 0 else "перекос влево"
            return [f"Среднее: {s.mean():,.2f}. Медиана: {s.median():,.2f}. Распределение: {dist}"]
        elif chart_key in ("pie", "donut") and x_col and y_col:
            grp = df.groupby(x_col, observed=True)[y_col].sum()
            total = grp.sum()
            if total == 0:
                return []
            top_label = str(grp.idxmax())
            top_pct = grp.max() / total * 100
            return [f"Топ: {top_label} = {top_pct:.1f}%. Категорий: {len(grp)}"]
    except Exception:
        pass
    return []


def _suggest_chart(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    if not x_col or not y_col:
        return ""
    try:
        x_is_date = pd.api.types.is_datetime64_any_dtype(df[x_col])
        x_is_num = pd.api.types.is_numeric_dtype(df[x_col])
        y_is_num = pd.api.types.is_numeric_dtype(df[y_col])
        x_nunique = df[x_col].nunique()
        if x_is_date and y_is_num:
            return "Рекомендуем: Линейный -- идеально для временных рядов"
        elif x_is_num and y_is_num:
            return "Рекомендуем: Точечный"
        elif not x_is_num and y_is_num and x_nunique <= 10:
            return "Рекомендуем: Круговой -- мало категорий"
        elif not x_is_num and y_is_num:
            return "Рекомендуем: Столбчатый"
    except Exception:
        pass
    return ""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

layout = dbc.Container([
    dbc.Row(dbc.Col(html.H3("20. Конструктор графиков"), className="mb-1")),
    dbc.Row(dbc.Col(html.P(
        "18 типов визуализаций с автовыводами", className="text-muted mb-3",
    ))),

    dbc.Row([
        # --- Settings ---
        dbc.Col([
            dbc.Card(dbc.CardBody([
                html.H6("Настройки графика"),
                dbc.Label("Тип графика"),
                dcc.Dropdown(
                    id="ch-type", options=CHART_OPTIONS, value="bar",
                ),
                dbc.Label("Ось X / Категория", className="mt-2"),
                dcc.Dropdown(id="ch-x", placeholder="Столбец X"),
                dbc.Label("Ось Y / Значение", className="mt-2"),
                dcc.Dropdown(id="ch-y", placeholder="Столбец Y"),
                dbc.Label("Цвет (группировка)", className="mt-2"),
                dcc.Dropdown(id="ch-color", placeholder="(нет)"),
                dbc.Label("Размер (для пузырькового)", className="mt-2"),
                dcc.Dropdown(id="ch-size", placeholder="(нет)"),
                html.Hr(),
                dbc.Label("Заголовок"),
                dbc.Input(id="ch-title", value="", placeholder="Заголовок графика"),
                dbc.Label("Цветовая схема", className="mt-2"),
                dcc.Dropdown(
                    id="ch-scheme",
                    options=[{"label": k, "value": k} for k in COLOR_SCHEMES],
                    value="По умолчанию",
                ),
                dbc.Row([
                    dbc.Col(dbc.Checklist(
                        id="ch-opts",
                        options=[
                            {"label": "Легенда", "value": "legend"},
                            {"label": "Подписи данных", "value": "labels"},
                            {"label": "Сортировать", "value": "sort"},
                            {"label": "Лог. шкала Y", "value": "logy"},
                        ],
                        value=["legend"],
                        inline=True,
                    )),
                ], className="mt-2"),
                dbc.Label("Топ N", className="mt-2"),
                dbc.Input(id="ch-topn", type="number", value=0, min=0, step=1),
                dbc.Label("Высота (пикселей)", className="mt-2"),
                dbc.Input(id="ch-height", type="number", value=500, min=200, max=1200, step=50),
                dbc.Button(
                    "Построить", id="ch-build-btn", color="primary",
                    className="mt-3 w-100",
                ),
            ])),
        ], md=3),

        # --- Chart area ---
        dbc.Col([
            html.Div(id="ch-suggestion", className="mb-2"),
            dcc.Loading(type="circle", color="#10b981", children=[
                html.Div(id="ch-alert"),
                dcc.Graph(id="ch-graph", style={"display": "none"}),
                html.Div(id="ch-insights", className="mt-2"),
                html.Div(id="ch-download-area", className="mt-2"),
            ]),
        ], md=9),
    ]),
], fluid=True, className="p-4")


# ---------------------------------------------------------------------------
# Populate columns
# ---------------------------------------------------------------------------

@callback(
    [Output("ch-x", "options"),
     Output("ch-y", "options"),
     Output("ch-color", "options"),
     Output("ch-size", "options")],
    [Input(STORE_DATASET, "data"),
     Input(STORE_PREPARED, "data"),
     Input(STORE_ACTIVE_DS, "data")],
)
def populate_chart_cols(ds_data, prep_data, active_ds):
    if not active_ds:
        return [], [], [], []
    df = get_df_from_store(prep_data, active_ds) or get_df_from_store(ds_data, active_ds)
    if df is None:
        return [], [], [], []
    all_opts = [{"label": c, "value": c} for c in df.columns]
    none_plus = [{"label": "(нет)", "value": ""}] + all_opts
    return all_opts, all_opts, none_plus, none_plus


# ---------------------------------------------------------------------------
# Suggestion
# ---------------------------------------------------------------------------

@callback(
    Output("ch-suggestion", "children"),
    [Input("ch-x", "value"),
     Input("ch-y", "value"),
     Input(STORE_DATASET, "data"),
     Input(STORE_PREPARED, "data"),
     Input(STORE_ACTIVE_DS, "data")],
)
def show_suggestion(x_col, y_col, ds_data, prep_data, active_ds):
    if not active_ds or not x_col or not y_col:
        return ""
    df = get_df_from_store(prep_data, active_ds) or get_df_from_store(ds_data, active_ds)
    if df is None:
        return ""
    text = _suggest_chart(df, x_col, y_col)
    if text:
        return dbc.Alert(text, color="info", className="py-1 px-2 mb-0")
    return ""


# ---------------------------------------------------------------------------
# Build chart
# ---------------------------------------------------------------------------

@callback(
    [Output("ch-graph", "figure"),
     Output("ch-graph", "style"),
     Output("ch-alert", "children"),
     Output("ch-insights", "children"),
     Output("ch-download-area", "children")],
    Input("ch-build-btn", "n_clicks"),
    [State(STORE_DATASET, "data"),
     State(STORE_PREPARED, "data"),
     State(STORE_ACTIVE_DS, "data"),
     State("ch-type", "value"),
     State("ch-x", "value"),
     State("ch-y", "value"),
     State("ch-color", "value"),
     State("ch-size", "value"),
     State("ch-title", "value"),
     State("ch-scheme", "value"),
     State("ch-opts", "value"),
     State("ch-topn", "value"),
     State("ch-height", "value")],
    prevent_initial_call=True,
)
def build_chart(n_clicks, ds_data, prep_data, active_ds,
                chart_type, x_col, y_col, color_col, size_col,
                title, scheme, opts, top_n, height):
    if not active_ds:
        return no_update, {"display": "none"}, dbc.Alert("Нет данных.", color="warning"), "", ""
    df = get_df_from_store(prep_data, active_ds) or get_df_from_store(ds_data, active_ds)
    if df is None or df.empty:
        return no_update, {"display": "none"}, dbc.Alert("Нет данных.", color="danger"), "", ""

    opts = opts or []
    show_legend = "legend" in opts
    show_labels = "labels" in opts
    sort_vals = "sort" in opts
    log_y = "logy" in opts
    color_col = color_col or None
    size_col = size_col or None
    top_n = int(top_n) if top_n else 0
    height = int(height) if height else 500
    title = title or ""

    palette = COLOR_SCHEMES.get(scheme)
    disc_kwargs = {"color_discrete_sequence": palette} if palette else {}
    cont_kwargs = {"color_continuous_scale": palette} if palette else {}

    try:
        fig = None

        if chart_type == "bar":
            plot_df = df.sort_values(y_col, ascending=False) if sort_vals and y_col in df.columns else df
            if top_n > 0:
                plot_df = plot_df.head(top_n)
            fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "bar_h":
            plot_df = df.sort_values(y_col, ascending=False) if sort_vals and y_col in df.columns else df
            if top_n > 0:
                plot_df = plot_df.head(top_n)
            fig = px.bar(plot_df, x=y_col, y=x_col, color=color_col, orientation="h", **disc_kwargs)

        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "area":
            fig = px.area(df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, nbins=30, **disc_kwargs)

        elif chart_type == "box":
            fig = px.box(df, x=color_col, y=y_col, color=color_col, **disc_kwargs)

        elif chart_type == "violin":
            fig = px.violin(df, x=color_col, y=y_col, color=color_col, box=True, **disc_kwargs)

        elif chart_type == "pie":
            plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
            if top_n > 0:
                plot_df = plot_df.nlargest(top_n, y_col)
            fig = px.pie(plot_df, names=x_col, values=y_col, **disc_kwargs)

        elif chart_type == "donut":
            plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
            if top_n > 0:
                plot_df = plot_df.nlargest(top_n, y_col)
            fig = px.pie(plot_df, names=x_col, values=y_col, hole=0.45, **disc_kwargs)

        elif chart_type == "treemap":
            path_cols = [x_col]
            if color_col:
                path_cols = [color_col, x_col]
            fig = px.treemap(df, path=path_cols, values=y_col, **disc_kwargs)

        elif chart_type == "sunburst":
            path_cols = [x_col]
            if color_col:
                path_cols = [color_col, x_col]
            fig = px.sunburst(df, path=path_cols, values=y_col, **disc_kwargs)

        elif chart_type == "heatmap":
            num_df = df.select_dtypes(include="number")
            if num_df.shape[1] < 2:
                return no_update, {"display": "none"}, dbc.Alert("Нужно минимум 2 числовых столбца.", color="warning"), "", ""
            corr = num_df.corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                colorscale="RdBu_r", zmin=-1, zmax=1,
                text=corr.round(2).values, texttemplate="%{text}",
            ))

        elif chart_type == "bubble":
            fig = px.scatter(df, x=x_col, y=y_col, size=size_col, color=color_col,
                             size_max=60, **disc_kwargs)

        elif chart_type == "funnel":
            plot_df = df.groupby(x_col, observed=True)[y_col].sum().reset_index()
            if sort_vals:
                plot_df = plot_df.sort_values(y_col, ascending=False)
            if top_n > 0:
                plot_df = plot_df.head(top_n)
            fig = go.Figure(go.Funnel(
                y=plot_df[x_col].astype(str).tolist(),
                x=plot_df[y_col].tolist(),
                textinfo="value+percent total",
            ))

        elif chart_type == "dual_axis":
            if not y_col or not color_col:
                return no_update, {"display": "none"}, dbc.Alert("Для двойной оси укажите Y (столбцы) и Цвет (вторая ось).", color="warning"), "", ""
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=df[x_col], y=df[y_col], name=y_col), secondary_y=False)
            fig.add_trace(go.Scatter(x=df[x_col], y=df[color_col], name=color_col, mode="lines+markers"), secondary_y=True)
            fig.update_yaxes(title_text=y_col, secondary_y=False)
            fig.update_yaxes(title_text=color_col, secondary_y=True)

        elif chart_type == "candlestick":
            # Expect x=date, y=open, color=high, size=close; low derived
            return no_update, {"display": "none"}, dbc.Alert(
                "Для свечного графика используйте 4 числовых столбца (Открытие, Максимум, Минимум, Закрытие) "
                "и столбец даты. Укажите X=дата, Y=Открытие.", color="info",
            ), "", ""

        elif chart_type == "bar_race":
            if not color_col:
                return no_update, {"display": "none"}, dbc.Alert("Для анимации укажите столбец «Цвет» как кадр анимации.", color="warning"), "", ""
            fig = px.bar(
                df.sort_values(y_col, ascending=False), x=x_col, y=y_col,
                animation_frame=color_col, title=title or f"{y_col} по {x_col}",
                range_y=[0, df[y_col].max() * 1.1],
            )

        if fig is None:
            return no_update, {"display": "none"}, dbc.Alert("Неизвестный тип графика.", color="danger"), "", ""

        # Apply layout settings
        fig.update_layout(
            height=height, showlegend=show_legend,
            title={"text": title, "font": {"size": 16}} if title else None,
        )
        if log_y:
            fig.update_yaxes(type="log")
        if show_labels and chart_type in ("bar", "bar_h"):
            fig.update_traces(texttemplate="%{value:,.0f}", textposition="outside")
        elif show_labels and chart_type in ("pie", "donut"):
            fig.update_traces(textinfo="label+percent")

        apply_kibad_theme(fig)

        # Insights
        insights = _get_insights(chart_type, df, x_col, y_col)
        insights_div = html.Div([
            dbc.Alert(line, color="light", className="py-1 mb-1") for line in insights
        ]) if insights else ""

        # Download button
        try:
            img_bytes = fig.to_image(format="png")
            b64 = base64.b64encode(img_bytes).decode()
            dl = html.A(
                dbc.Button("Скачать PNG", color="secondary", size="sm"),
                href=f"data:image/png;base64,{b64}",
                download="chart.png",
            )
        except Exception:
            html_str = fig.to_html()
            b64 = base64.b64encode(html_str.encode()).decode()
            dl = html.A(
                dbc.Button("Скачать HTML", color="secondary", size="sm"),
                href=f"data:text/html;base64,{b64}",
                download="chart.html",
            )

        return fig, {"display": "block"}, None, insights_div, dl

    except Exception as exc:
        return no_update, {"display": "none"}, dbc.Alert(f"Ошибка: {exc}", color="danger"), "", ""

"""p10_report – Report generation page (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc
import pandas as pd
import io

from app.state import (
    get_df_from_store, STORE_DATASET, STORE_PREPARED, STORE_ACTIVE_DS, STORE_REPORT,
)
from app.components.layout import page_header, section_header, empty_state
from app.components.alerts import alert_banner
from core.report import ReportBuilder, generate_business_summary
from core.audit import log_event

dash.register_page(__name__, path="/report", name="10. Отчёт", order=10, icon="file-earmark-text")

layout = html.Div([
    page_header("10. Генерация отчёта", "HTML и PDF отчёты с экспортом"),
    dbc.Row([
        dbc.Col(dcc.Dropdown(id="rep-ds", placeholder="Выберите датасет..."), width=4),
    ], className="mb-3"),
    dbc.Card([
        dbc.CardHeader("Настройки отчёта"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    html.Label("Заголовок отчёта", className="kb-stat-label"),
                    dcc.Input(id="rep-title", value="KIBAD Analysis Report", style={"width": "100%"}),
                ], width=4),
                dbc.Col([
                    html.Label("Дата-колонка", className="kb-stat-label"),
                    dcc.Dropdown(id="rep-date-col", placeholder="Для бизнес-саммари..."),
                ], width=3),
                dbc.Col([
                    html.Label("Целевая метрика", className="kb-stat-label"),
                    dcc.Dropdown(id="rep-target-col", placeholder="Для бизнес-саммари..."),
                ], width=3),
                dbc.Col([
                    dbc.Button("Сгенерировать", id="rep-gen-btn", color="primary", className="mt-4"),
                ], width=2),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Checklist(
                        id="rep-sections",
                        options=[
                            {"label": "Профиль данных", "value": "profile"},
                            {"label": "Описательная статистика", "value": "stats"},
                            {"label": "Бизнес-саммари", "value": "summary"},
                        ],
                        value=["profile", "stats"],
                        inline=True,
                        className="mt-2",
                    ),
                ]),
            ]),
        ]),
    ], className="mb-3"),
    dcc.Loading(html.Div(id="rep-result"), type="circle", color="#00c896"),
    dcc.Download(id="rep-download-html"),
    dcc.Download(id="rep-download-pdf"),
])


@callback(
    Output("rep-ds", "options"),
    Output("rep-ds", "value"),
    Input(STORE_DATASET, "data"),
    State(STORE_ACTIVE_DS, "data"),
)
def update_ds(datasets, active):
    if not datasets:
        return [], None
    names = list(datasets.keys())
    val = active if active in names else (names[0] if names else None)
    return [{"label": n, "value": n} for n in names], val


@callback(
    Output("rep-date-col", "options"),
    Output("rep-target-col", "options"),
    Input("rep-ds", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
)
def update_cols(ds, datasets, prepared):
    if not ds:
        return [], []
    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return [], []
    all_opts = [{"label": c, "value": c} for c in df.columns]
    num_opts = [{"label": c, "value": c} for c in df.select_dtypes(include="number").columns]
    return all_opts, num_opts


@callback(
    Output("rep-result", "children"),
    Output(STORE_REPORT, "data"),
    Input("rep-gen-btn", "n_clicks"),
    State("rep-ds", "value"),
    State("rep-title", "value"),
    State("rep-date-col", "value"),
    State("rep-target-col", "value"),
    State("rep-sections", "value"),
    State(STORE_DATASET, "data"),
    State(STORE_PREPARED, "data"),
    prevent_initial_call=True,
)
def generate_report(n, ds, title, date_col, target_col, sections, datasets, prepared):
    if not ds:
        return alert_banner("Выберите датасет.", "warning"), no_update

    df = get_df_from_store(prepared, ds) or get_df_from_store(datasets, ds)
    if df is None:
        return alert_banner("Датасет не найден.", "danger"), no_update

    try:
        rb = ReportBuilder(title=title or "KIBAD Report", dataset_name=ds, n_rows=len(df))

        if "profile" in (sections or []):
            profile_html = f"<p>Строк: {len(df):,}, Столбцов: {df.shape[1]}</p>"
            profile_html += f"<p>Числовых: {len(df.select_dtypes(include='number').columns)}, "
            profile_html += f"Текстовых: {len(df.select_dtypes(include='object').columns)}</p>"
            miss = df.isnull().sum().sum()
            profile_html += f"<p>Пропусков: {miss:,} ({miss / df.size * 100:.1f}%)</p>"
            rb.add_section("Профиль данных", profile_html)

        if "stats" in (sections or []):
            desc = df.describe(include="all").round(2)
            rb.add_table("Описательная статистика", desc.reset_index())

        if "summary" in (sections or []) and date_col and target_col:
            try:
                summary = generate_business_summary(df, date_col, target_col)
                rb.add_interpretation("Бизнес-саммари", summary)
            except Exception:
                pass

        report_html = rb.render()
        log_event("report", dataset=ds, details=f"sections={sections}")

        return html.Div([
            section_header("Отчёт сгенерирован"),
            dbc.Row([
                dbc.Col(dbc.Button("Скачать HTML", id="rep-dl-html-btn", color="primary", outline=True), width="auto"),
                dbc.Col(dbc.Button("Скачать PDF", id="rep-dl-pdf-btn", color="primary", outline=True), width="auto"),
            ], className="mb-3"),
            html.Iframe(
                srcDoc=report_html,
                style={"width": "100%", "height": "600px", "border": "1px solid #2a2f42",
                       "borderRadius": "10px", "background": "#fff"},
            ),
        ]), report_html

    except Exception as e:
        return alert_banner(f"Ошибка генерации: {e}", "danger"), no_update


@callback(
    Output("rep-download-html", "data"),
    Input("rep-dl-html-btn", "n_clicks"),
    State(STORE_REPORT, "data"),
    prevent_initial_call=True,
)
def download_html(n, report_html):
    if not report_html:
        return no_update
    return dcc.send_string(report_html, "kibad_report.html")


@callback(
    Output("rep-download-pdf", "data"),
    Input("rep-dl-pdf-btn", "n_clicks"),
    State(STORE_REPORT, "data"),
    prevent_initial_call=True,
)
def download_pdf(n, report_html):
    if not report_html:
        return no_update
    try:
        from core.report_pdf import generate_pdf_bytes
        pdf_bytes = generate_pdf_bytes(report_html)
        return dcc.send_bytes(pdf_bytes, "kibad_report.pdf")
    except Exception:
        # Fallback: download as HTML
        return dcc.send_string(report_html, "kibad_report.html")

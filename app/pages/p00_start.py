"""
pages/p00_start.py -- Landing / guided-workflow page for KIBAD (Dash).
"""
from __future__ import annotations

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

from app.state import (
    STORE_DATASET, STORE_FORECAST, STORE_TEST_RESULTS,
    list_datasets,
)

dash.register_page(
    __name__,
    path="/",
    name="Старт",
    order=0,
    icon="house",
)

# ---------------------------------------------------------------------------
# Constants -- module grid and workflow data
# ---------------------------------------------------------------------------

# Each module: (letter_icon, accent_color, title, description, href)
_MODULES_PRIMARY = [
    ("Д", "#4b9eff", "Загрузить данные",
     "CSV, Excel, Parquet или PostgreSQL", "/data"),
    ("П", "#9b59b6", "Подготовить данные",
     "Очистка, формулы, нормализация", "/prepare"),
    ("Г", "#00c896", "Группировка",
     "Сводные таблицы и агрегация", "/group"),
    ("О", "#4b9eff", "Объединение",
     "JOIN / UNION с диагностикой", "/merge"),
    ("И", "#e8a020", "Исследование",
     "Корреляции, распределения, профиль", "/explore"),
    ("Т", "#e05252", "Стат. тесты",
     "t-тест, хи-квадрат, AB-тесты", "/tests"),
]

_MODULES_ANALYTICS = [
    ("ВР", "#1abc9c", "Временные ряды",
     "Прогнозирование, ACF/PACF, аномалии", "/timeseries"),
    ("Ф", "#f39c12", "Факторная атрибуция",
     "Декомпозиция и вклад факторов", "/attribution"),
    ("М", "#3498db", "Моделирование",
     "Сценарный анализ и симуляция", "/simulation"),
    ("К", "#e05252", "Кластеризация",
     "K-Means, DBSCAN, иерархическая", "/cluster"),
    ("Ср", "#9b59b6", "Сравнение групп",
     "Сопоставимость и мэтчинг", "/matching"),
    ("Ст", "#4b9eff", "Сравнение периодов",
     "Динамика показателей по периодам", "/compare"),
]

_MODULES_TOOLS = [
    ("Гр", "#00c896", "Графики",
     "Интерактивные визуализации", "/charts"),
    ("Тк", "#e8a020", "Текст",
     "Анализ текстовых данных", "/text"),
    ("АА", "#9b59b6", "Автоанализ",
     "Автоматический разведочный анализ", "/autoanalyst"),
    ("Шб", "#3498db", "Шаблоны",
     "Готовые аналитические пайплайны", "/templates"),
    ("Кв", "#1abc9c", "Конвейер",
     "Цепочки трансформаций данных", "/pipeline"),
    ("О", "#f39c12", "Отчёты",
     "Экспорт PDF, HTML, CSV, Excel", "/report"),
]

_STEPS = [
    ("1", "Загрузите CSV / Excel на странице Данные",
     "#4b9eff"),
    ("2", "Очистите данные на странице Подготовка",
     "#9b59b6"),
    ("3", "Запустите анализ на любой аналитической странице",
     "#00c896"),
    ("4", "Сформируйте отчёт и экспортируйте результат",
     "#e8a020"),
]


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _hero_section() -> html.Div:
    """Gradient hero banner -- dark theme."""
    pills = [
        html.Span(
            label,
            style={
                "background": "rgba(0,200,150,0.12)",
                "border": "1px solid rgba(0,200,150,0.25)",
                "borderRadius": "50px",
                "padding": "5px 15px",
                "fontSize": "0.78rem",
                "fontWeight": "600",
                "color": "#00c896",
            },
        )
        for label in ["Без Python", "Без SQL", "Без Excel"]
    ]

    return html.Div(
        [
            html.Div(
                "АНАЛИТИКА БЕЗ КОДА",
                style={
                    "fontSize": "0.68rem", "fontWeight": "700",
                    "textTransform": "uppercase", "letterSpacing": "0.14em",
                    "color": "#00c896", "marginBottom": "14px",
                },
            ),
            html.Div(
                "KIBAD — Аналитическая студия",
                style={
                    "fontSize": "2rem", "fontWeight": "800",
                    "color": "#e8eaf0",
                    "lineHeight": "1.15", "letterSpacing": "-0.03em",
                    "marginBottom": "10px",
                },
            ),
            html.P(
                "Загрузите данные и получите готовый анализ за 5 минут. "
                "21 инструмент для работы с данными в одном приложении.",
                style={
                    "fontSize": "0.95rem", "color": "#8b92a8",
                    "marginBottom": "24px", "lineHeight": "1.6",
                    "maxWidth": "600px",
                },
            ),
            html.Div(pills, style={
                "display": "flex", "gap": "10px", "flexWrap": "wrap",
            }),
        ],
        style={
            "background": "linear-gradient(135deg, #141720 0%, #1c2030 50%, #141720 100%)",
            "border": "1px solid #2a2f42",
            "borderRadius": "16px",
            "padding": "44px 48px",
            "marginBottom": "32px",
            "position": "relative",
            "overflow": "hidden",
        },
    )


def _stat_cards() -> html.Div:
    """Row of 3 stat cards -- values filled by callback."""
    def _card(label: str, value_id: str, accent: str) -> html.Div:
        return html.Div(
            html.Div(
                [
                    html.Div(label, className="kb-stat-label"),
                    html.Div("--", id=value_id, className="kb-stat-value"),
                ],
                className="kb-stat-card",
            ),
            className="kb-start-stat",
        )

    return html.Div(
        [
            _card("Датасетов", "start-stat-datasets", "#4b9eff"),
            _card("Прогнозов", "start-stat-forecasts", "#00c896"),
            _card("Тестов", "start-stat-tests", "#e8a020"),
        ],
        className="kb-stats-grid",
        style={"marginBottom": "32px"},
    )


def _module_card(icon: str, accent: str, title: str, desc: str, href: str) -> dbc.Col:
    """Single module card for the grid."""
    return dbc.Col(
        dcc.Link(
            html.Div(
                [
                    html.Div(
                        icon,
                        className="kb-module-icon",
                        style={
                            "background": f"{accent}18",
                            "color": accent,
                            "border": f"1px solid {accent}30",
                        },
                    ),
                    html.Div(title, className="kb-module-title"),
                    html.Div(desc, className="kb-module-desc"),
                ],
                className="kb-module-card",
            ),
            href=href,
            style={"textDecoration": "none"},
        ),
        lg=4, md=6, sm=6, xs=12, className="mb-3",
    )


def _module_section(title: str, modules: list) -> html.Div:
    """Section of module cards with a heading."""
    cards = [
        _module_card(icon, accent, t, desc, href)
        for icon, accent, t, desc, href in modules
    ]
    return html.Div([
        html.Div(
            title,
            className="kb-start-section-title",
        ),
        dbc.Row(cards),
    ], style={"marginBottom": "28px"})


def _how_it_works() -> html.Div:
    """4-step quick-start flow."""
    steps = []
    for num, text, color in _STEPS:
        step = html.Div(
            [
                html.Div(
                    num,
                    className="kb-start-step-num",
                    style={"background": color},
                ),
                html.Span(text, style={
                    "fontSize": "0.86rem", "color": "#8b92a8",
                }),
            ],
            className="kb-start-step",
        )
        steps.append(step)

    return html.Div(
        [
            html.Div(
                "Как начать работу",
                className="kb-start-section-title",
            ),
            html.Div(
                steps,
                style={
                    "background": "#141720",
                    "border": "1px solid #2a2f42",
                    "borderRadius": "12px",
                    "padding": "20px 24px",
                },
            ),
        ],
        style={"marginBottom": "28px"},
    )


def _loaded_datasets_notice() -> html.Div:
    """Container updated by callback showing loaded dataset names."""
    return html.Div(id="start-loaded-notice", className="mt-3")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        _hero_section(),
        _stat_cards(),
        _module_section("Работа с данными", _MODULES_PRIMARY),
        _module_section("Аналитика и моделирование", _MODULES_ANALYTICS),
        _module_section("Инструменты и экспорт", _MODULES_TOOLS),
        _how_it_works(),
        _loaded_datasets_notice(),
    ],
    style={"maxWidth": "1100px", "margin": "0 auto", "padding": "24px 16px"},
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("start-stat-datasets", "children"),
    Output("start-stat-forecasts", "children"),
    Output("start-stat-tests", "children"),
    Output("start-loaded-notice", "children"),
    Input(STORE_DATASET, "data"),
    Input(STORE_FORECAST, "data"),
    Input(STORE_TEST_RESULTS, "data"),
)
def update_stats(ds_store, forecast_store, test_store):
    ds_names = list_datasets(ds_store)
    n_ds = len(ds_names)
    n_fc = len(forecast_store) if forecast_store else 0
    n_ts = len(test_store) if test_store else 0

    notice = ""
    if ds_names:
        names_str = ", ".join(ds_names)
        notice = dbc.Alert(
            f"Загружены датасеты: {names_str}. Перейдите к нужному инструменту.",
            color="success",
            className="mt-3",
        )

    return str(n_ds), str(n_fc), str(n_ts), notice or ""

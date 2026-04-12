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
    ("Д", "#3b82f6", "Загрузить данные",
     "CSV, Excel, Parquet или PostgreSQL", "/data"),
    ("П", "#8b5cf6", "Подготовить данные",
     "Очистка, формулы, нормализация", "/prepare"),
    ("Г", "#10b981", "Группировка",
     "Сводные таблицы и агрегация", "/group"),
    ("О", "#06b6d4", "Объединение",
     "JOIN / UNION с диагностикой", "/merge"),
    ("И", "#f59e0b", "Исследование",
     "Корреляции, распределения, профиль", "/explore"),
    ("Т", "#ef4444", "Стат. тесты",
     "t-тест, хи-квадрат, AB-тесты", "/tests"),
]

_MODULES_ANALYTICS = [
    ("ВР", "#06b6d4", "Временные ряды",
     "Прогнозирование, ACF/PACF, аномалии", "/timeseries"),
    ("Ф", "#f97316", "Факторная атрибуция",
     "Декомпозиция и вклад факторов", "/attribution"),
    ("М", "#3b82f6", "Моделирование",
     "Сценарный анализ и симуляция", "/simulation"),
    ("К", "#ef4444", "Кластеризация",
     "K-Means, DBSCAN, иерархическая", "/cluster"),
    ("Ср", "#8b5cf6", "Сравнение групп",
     "Сопоставимость и мэтчинг", "/matching"),
    ("Ст", "#06b6d4", "Сравнение периодов",
     "Динамика показателей по периодам", "/compare"),
]

_MODULES_TOOLS = [
    ("Гр", "#10b981", "Графики",
     "Интерактивные визуализации", "/charts"),
    ("Тк", "#f59e0b", "Текст",
     "Анализ текстовых данных", "/text"),
    ("АА", "#8b5cf6", "Автоанализ",
     "Автоматический разведочный анализ", "/autoanalyst"),
    ("Шб", "#3b82f6", "Шаблоны",
     "Готовые аналитические пайплайны", "/templates"),
    ("Кв", "#06b6d4", "Конвейер",
     "Цепочки трансформаций данных", "/pipeline"),
    ("О", "#f97316", "Отчёты",
     "Экспорт PDF, HTML, CSV, Excel", "/report"),
]

_STEPS = [
    ("1", "Загрузите CSV / Excel на странице Данные", "#3b82f6"),
    ("2", "Очистите данные на странице Подготовка", "#8b5cf6"),
    ("3", "Запустите анализ на любой аналитической странице", "#10b981"),
    ("4", "Сформируйте отчёт и экспортируйте результат", "#f59e0b"),
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
                "background": "rgba(16,185,129,0.10)",
                "border": "1px solid rgba(16,185,129,0.20)",
                "borderRadius": "50px",
                "padding": "5px 15px",
                "fontSize": "0.76rem",
                "fontWeight": "600",
                "color": "#10b981",
            },
        )
        for label in ["Без Python", "Без SQL", "Без Excel"]
    ]

    return html.Div(
        [
            html.Div(
                "АНАЛИТИКА БЕЗ КОДА",
                style={
                    "fontSize": "0.72rem", "fontWeight": "700",
                    "textTransform": "uppercase", "letterSpacing": "0.14em",
                    "color": "#10b981", "marginBottom": "14px",
                },
            ),
            html.Div(
                "KIBAD — Аналитическая студия",
                style={
                    "fontSize": "1.9rem", "fontWeight": "800",
                    "color": "#e4e7ee",
                    "lineHeight": "1.15", "letterSpacing": "-0.03em",
                    "marginBottom": "10px",
                },
            ),
            html.P(
                "Загрузите данные и получите готовый анализ за 5 минут. "
                "21 инструмент для работы с данными в одном приложении.",
                style={
                    "fontSize": "0.92rem", "color": "#9ba3b8",
                    "marginBottom": "24px", "lineHeight": "1.6",
                    "maxWidth": "580px",
                },
            ),
            html.Div(pills, style={
                "display": "flex", "gap": "10px", "flexWrap": "wrap",
            }),
        ],
        style={
            "background": "linear-gradient(135deg, #111318 0%, #191c24 50%, #111318 100%)",
            "border": "1px solid #1e2232",
            "borderRadius": "14px",
            "padding": "44px 48px",
            "marginBottom": "32px",
            "position": "relative",
            "overflow": "hidden",
        },
    )


def _stat_cards() -> html.Div:
    """Row of 3 stat cards -- values filled by callback."""
    def _card(label: str, value_id: str) -> html.Div:
        return html.Div(
            html.Div(
                [
                    html.Div(label, className="kb-stat-label"),
                    html.Div("--", id=value_id, className="kb-stat-value"),
                ],
                className="kb-stat-card",
            ),
        )

    return html.Div(
        [
            _card("Датасетов", "start-stat-datasets"),
            _card("Прогнозов", "start-stat-forecasts"),
            _card("Тестов", "start-stat-tests"),
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
                            "background": f"{accent}14",
                            "color": accent,
                            "border": f"1px solid {accent}25",
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
                    "fontSize": "0.84rem", "color": "#9ba3b8",
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
                    "background": "#111318",
                    "border": "1px solid #1e2232",
                    "borderRadius": "10px",
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

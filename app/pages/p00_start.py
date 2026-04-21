"""p00_start — Landing / overview page.

Design port of Slide 2 «Обзор платформы». Shows:
- HERO card with overline + headline + subtitle + success-chip pills,
  plus a decorative dot-grid backdrop on the right half.
- Three KPI tiles driven by the in-session stores (datasets, forecasts,
  tests).
- Three sections of module cards — «Работа с данными», «Аналитика
  и моделирование», «Инструменты и экспорт» — each as a three-column
  grid of icon-tile cards.
"""
from __future__ import annotations

import dash
from dash import Input, Output, callback, dcc, html

from app.components.cards import kpi
from app.components.icons import icon
from app.state import (
    STORE_DATASET, STORE_FORECAST, STORE_TEST_RESULTS,
    list_datasets,
)

dash.register_page(__name__, path="/", name="Старт", order=0, icon="house")


# ---------------------------------------------------------------------------
# Module catalogue — (icon, title, description, href, accent)
#   accent=True → "primary" tile (accent-900 bg / accent-300 icon)
#   accent=False → "neutral" tile (surface-2 bg / text-secondary icon)
# ---------------------------------------------------------------------------
_DATA_TOOLS = [
    ("upload",   "Загрузить данные",   "CSV, Excel, Parquet или PostgreSQL", "/data",    True),
    ("settings", "Подготовить данные", "Очистка, формулы, нормализация",     "/prepare", True),
    ("table",    "Группировка",        "Сводные таблицы и агрегация",        "/group",   True),
    ("link",     "Объединение",        "JOIN и UNION с диагностикой",        "/merge",   True),
    ("search",   "Исследование",       "Корреляции, распределения, профиль", "/explore", True),
    ("trend",    "Стат. тесты",        "t-тест, хи-квадрат, A/B-тесты",      "/tests",   True),
]

_ANALYTICS_TOOLS = [
    ("trend",     "Временные ряды",       "Прогнозирование, ACF/PACF, аномалии", "/timeseries",  False),
    ("bar-chart", "Факторная атрибуция",  "Декомпозиция и вклад факторов",       "/attribution", False),
    ("grid",      "Моделирование",        "Сценарный анализ и симуляция",        "/simulation",  False),
    ("grid",      "Кластеризация",        "K-means, DBSCAN, профили",            "/cluster",     False),
    ("percent",   "Roll Rate",            "Миграционные матрицы",                "/rollrate",    False),
    ("users",     "Сопоставление групп",  "Matching по ковариатам",              "/matching",    False),
]

_PRODUCTIVITY_TOOLS = [
    ("chart",     "Графики",              "Интерактивные визуализации",          "/charts",      False),
    ("file-text", "Текст",                "Анализ текстовых полей",              "/text",        False),
    ("robot",     "Автоанализ",           "Автоматический разведочный анализ",   "/autoanalyst", False),
    ("layers",    "Шаблоны",              "Готовые аналитические пайплайны",     "/templates",   False),
    ("wrench",    "Конвейер",             "Цепочки трансформаций",               "/pipeline",    False),
    ("download",  "Отчёты",               "Экспорт PDF / HTML / CSV / Excel",    "/report",      False),
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _hero() -> html.Div:
    """Hero card — overline + headline + paragraph + 3 success chips.

    The right half carries a decorative dot-grid overlay (subtle white
    dots) in an SVG pattern, matching the handoff mockup.
    """
    # Inline SVG dot-grid used as a background element. Anchored right and
    # sized to ~420px wide; blends over `var(--surface-1)` in the card.
    dot_bg = html.Div(
        dcc.Markdown(
            """
<svg xmlns="http://www.w3.org/2000/svg" width="100%" height="100%" viewBox="0 0 420 200" preserveAspectRatio="xMidYMid slice">
  <defs>
    <pattern id="hero-dots" width="16" height="16" patternUnits="userSpaceOnUse">
      <circle cx="1" cy="1" r="0.9" fill="rgba(255,255,255,0.06)"/>
    </pattern>
    <radialGradient id="hero-spot" cx="78%" cy="20%" r="55%">
      <stop offset="0%"   stop-color="rgba(33,160,102,0.10)"/>
      <stop offset="100%" stop-color="rgba(33,160,102,0)"/>
    </radialGradient>
  </defs>
  <rect width="420" height="200" fill="url(#hero-dots)"/>
  <rect width="420" height="200" fill="url(#hero-spot)"/>
</svg>
            """.strip(),
            dangerously_allow_html=True,
        ),
        className="kb-start-hero__bg",
    )

    return html.Div(
        [
            dot_bg,
            html.Div(
                [
                    html.Div("Аналитика без кода", className="kb-overline"),
                    html.H1("Аналитическая студия", className="kb-h1 kb-start-hero__title"),
                    html.P(
                        "Загрузите данные и получите готовый анализ за 5 минут. "
                        "21 инструмент для работы с данными в одном приложении.",
                        className="kb-body-l kb-start-hero__lead",
                    ),
                    html.Div(
                        [
                            html.Span(
                                [icon("check", 12), html.Span("Без Python")],
                                className="kb-chip kb-chip--success",
                            ),
                            html.Span(
                                [icon("check", 12), html.Span("Без SQL")],
                                className="kb-chip kb-chip--success",
                            ),
                            html.Span(
                                [icon("check", 12), html.Span("Без Excel")],
                                className="kb-chip kb-chip--success",
                            ),
                        ],
                        className="kb-start-hero__chips",
                    ),
                ],
                className="kb-start-hero__body",
            ),
        ],
        className="kb-card kb-start-hero",
    )


def _kpi_row() -> html.Div:
    """Three KPI tiles — values populated by the callback below."""
    return html.Div(
        [
            html.Div(
                [
                    html.Div("Датасетов", className="kb-stat-label"),
                    html.Div(
                        [
                            html.Div("—", id="start-stat-datasets", className="kb-stat-value"),
                            html.Div(id="start-stat-datasets-delta"),
                        ],
                        className="kb-stat-row",
                    ),
                ],
                className="kb-stat-card",
            ),
            html.Div(
                [
                    html.Div("Прогнозов", className="kb-stat-label"),
                    html.Div(
                        [
                            html.Div("—", id="start-stat-forecasts", className="kb-stat-value"),
                            html.Div(id="start-stat-forecasts-delta"),
                        ],
                        className="kb-stat-row",
                    ),
                ],
                className="kb-stat-card",
            ),
            html.Div(
                [
                    html.Div("Тестов", className="kb-stat-label"),
                    html.Div(
                        [
                            html.Div("—", id="start-stat-tests", className="kb-stat-value"),
                            html.Div(id="start-stat-tests-delta"),
                        ],
                        className="kb-stat-row",
                    ),
                ],
                className="kb-stat-card",
            ),
        ],
        className="kb-start-kpis",
    )


def _module_card(ic: str, title: str, desc: str, href: str, accent: bool) -> dcc.Link:
    """Single clickable module card — icon tile + title + chevron + description."""
    tile_cls = "kb-start-card__icon" + (" is-accent" if accent else "")
    return dcc.Link(
        html.Div(
            [
                html.Div(
                    [
                        html.Div(icon(ic, 16), className=tile_cls),
                        html.Div(title, className="kb-start-card__title"),
                        html.Div(
                            icon("chevron-right", 14),
                            className="kb-start-card__chev",
                        ),
                    ],
                    className="kb-start-card__head",
                ),
                html.Div(desc, className="kb-start-card__desc"),
            ],
            className="kb-start-card",
        ),
        href=href,
        className="kb-start-card-link",
    )


def _section(title: str, tools: list[tuple]) -> html.Div:
    """Section heading (overline) + 3-column card grid."""
    return html.Div(
        [
            html.Div(title, className="kb-overline kb-start-section__title"),
            html.Div(
                [_module_card(*t) for t in tools],
                className="kb-start-grid",
            ),
        ],
        className="kb-start-section",
    )


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
layout = html.Div(
    [
        _hero(),
        _kpi_row(),
        _section("Работа с данными", _DATA_TOOLS),
        _section("Аналитика и моделирование", _ANALYTICS_TOOLS),
        _section("Инструменты и экспорт", _PRODUCTIVITY_TOOLS),
        html.Div(id="start-loaded-notice"),
    ],
    className="kb-page kb-page-start",
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _delta_chip(n: int) -> html.Span | str:
    """Show an accent-coloured trend chip when a positive count is present."""
    if not n:
        return ""
    return html.Span(
        f"+{n}", className="kb-stat-delta kb-stat-delta--positive",
    )


@callback(
    Output("start-stat-datasets", "children"),
    Output("start-stat-forecasts", "children"),
    Output("start-stat-tests", "children"),
    Output("start-stat-datasets-delta", "children"),
    Output("start-stat-forecasts-delta", "children"),
    Output("start-stat-tests-delta", "children"),
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
        chips = [
            html.Span(n, className="kb-chip kb-chip--neutral kb-start-notice__chip")
            for n in ds_names[:6]
        ]
        if len(ds_names) > 6:
            chips.append(
                html.Span(f"+{len(ds_names) - 6}", className="kb-chip kb-chip--neutral")
            )
        notice = html.Div(
            [
                html.Div(
                    [icon("database", 14), html.Span("Загруженные датасеты")],
                    className="kb-start-notice__head",
                ),
                html.Div(chips, className="kb-start-notice__chips"),
            ],
            className="kb-start-notice",
        )

    return (
        str(n_ds), str(n_fc), str(n_ts),
        _delta_chip(n_ds), _delta_chip(n_fc), _delta_chip(n_ts),
        notice,
    )

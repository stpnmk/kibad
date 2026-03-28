"""p24_templates – Pre-built analysis scenario walkthroughs (Dash)."""
import dash
from dash import html, dcc, callback, Input, Output, State, no_update
import dash_bootstrap_components as dbc

from app.components.layout import page_header, section_header

dash.register_page(__name__, path="/templates", name="24. Шаблоны", order=24, icon="grid")

_TEMPLATES = [
    {
        "id": "sales",
        "icon": "П",
        "title": "Анализ продаж",
        "desc": "Загрузка → подготовка → группировка по периодам → тренд → прогноз → отчёт",
        "steps": [
            "1. Загрузите CSV/Excel с колонками: дата, сумма, категория",
            "2. Перейдите в Подготовка → парсинг дат, заполнение пропусков",
            "3. Группировка по месяцу + категории → сумма",
            "4. Исследование → временные ряды",
            "5. Временные ряды → SARIMAX прогноз",
            "6. Отчёт → PDF",
        ],
    },
    {
        "id": "ab_test",
        "icon": "AB",
        "title": "A/B-тестирование",
        "desc": "Загрузка → разделение на группы → статистический тест → интерпретация",
        "steps": [
            "1. Загрузите данные с колонками: группа (A/B), метрика",
            "2. Тесты → A/B-тест",
            "3. Выберите колонку группы и метрику",
            "4. Оцените p-value и размер эффекта",
            "5. Используйте интерпретацию для принятия решения",
        ],
    },
    {
        "id": "segmentation",
        "icon": "С",
        "title": "Сегментация клиентов",
        "desc": "Загрузка → подготовка → кластеризация → профили → визуализация",
        "steps": [
            "1. Загрузите данные с числовыми признаками клиентов",
            "2. Подготовка → нормализация, заполнение пропусков",
            "3. Кластеризация → Elbow для выбора K → K-Means",
            "4. Анализ профилей кластеров",
            "5. PCA-визуализация для понимания структуры",
        ],
    },
    {
        "id": "attribution",
        "icon": "Ф",
        "title": "Факторная декомпозиция",
        "desc": "Загрузка парных данных → выбор метода → водопадный график",
        "steps": [
            "1. Подготовьте данные: текущие и предыдущие значения для каждого фактора",
            "2. Атрибуция → выберите метод (аддитивный / Шэпли)",
            "3. Укажите целевой показатель и факторы-драйверы",
            "4. Проанализируйте водопадный график вкладов",
        ],
    },
    {
        "id": "risk",
        "icon": "Р",
        "title": "Оценка рисков (Монте-Карло)",
        "desc": "Загрузка → временной ряд → симуляция → VaR/CVaR",
        "steps": [
            "1. Загрузите исторические данные метрики",
            "2. Симуляция → Монте-Карло",
            "3. Настройте количество симуляций и горизонт",
            "4. Оцените VaR и CVaR на выходе",
        ],
    },
    {
        "id": "compare",
        "icon": "Ср",
        "title": "Сравнение периодов",
        "desc": "Загрузка → фильтрация по периодам → сравнительный анализ → водопад",
        "steps": [
            "1. Загрузите данные с датами и метриками",
            "2. Сравнение → выберите период A и период B",
            "3. Проанализируйте таблицу отклонений",
            "4. Водопадный график покажет источники изменений",
        ],
    },
]


def _template_card(t):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Span(t["icon"], style={"fontSize": "2rem", "marginRight": "12px"}),
                html.Div([
                    html.H4(t["title"], style={"margin": "0", "color": "#e8eaf0"}),
                    html.P(t["desc"], className="kb-text-secondary kb-text-sm", style={"margin": "4px 0 0 0"}),
                ]),
            ], style={"display": "flex", "alignItems": "flex-start"}),
        ]),
    ], className="mb-2", style={"cursor": "pointer"}, id=f"tmpl-card-{t['id']}")


layout = html.Div([
    page_header("24. Шаблоны анализа", "Готовые сценарии для типовых задач"),

    html.P("Выберите шаблон для пошагового руководства:", className="kb-text-secondary mb-3"),

    dbc.Row([
        dbc.Col(_template_card(t), width=6)
        for t in _TEMPLATES
    ]),

    html.Hr(),
    html.Div(id="tmpl-detail"),
])


# Create callbacks for each template card
for tmpl in _TEMPLATES:
    @callback(
        Output("tmpl-detail", "children", allow_duplicate=True),
        Input(f"tmpl-card-{tmpl['id']}", "n_clicks"),
        prevent_initial_call=True,
    )
    def show_template(n, _t=tmpl):
        if not n:
            return no_update
        return html.Div([
            section_header(f"Шаблон: {_t['title']}"),
            html.Ol([
                html.Li(step, className="kb-text-secondary", style={"marginBottom": "8px"})
                for step in _t["steps"]
            ], style={"paddingLeft": "20px"}),
        ])

# Архитектура KIBAD

## Обзор

KIBAD — многостраничное Streamlit-приложение для интерактивного анализа данных. Архитектура построена на разделении ответственности: пять слоёв изолируют логику ввода/вывода, аналитические вычисления, визуализацию, пользовательский интерфейс и хранение данных.

## Разделение слоёв

### 1. Слой данных (IO, схема, валидация, преобразования)

Отвечает за загрузку, валидацию и трансформацию данных. Не содержит зависимостей от UI — только чистые функции Python, которые можно тестировать независимо.

| Модуль | Ответственность |
|--------|----------------|
| `core/data.py` | Загрузка файлов: CSV, Excel (xlsx/xls), Parquet, PostgreSQL |
| `core/validate.py` | Проверка схемы, вывод типов, контроль ограничений |
| `core/prepare.py` | Очистка, приведение типов, обработка пропусков |
| `core/aggregate.py` | Группировка, сводные таблицы, скользящие окна |
| `core/merge.py` | Объединение датасетов: JOIN, UNION, конкатенация с диагностикой pitfalls |

### 2. Аналитический слой (статистика, модели, атрибуция)

Реализует все аналитические методы. Каждый модуль предоставляет чистые функции, принимающие DataFrame и возвращающие результат в виде словаря или DataFrame.

| Модуль | Ответственность |
|--------|----------------|
| `core/tests.py` | Статистические тесты: t-тест, Манна-Уитни, хи-квадрат, Bootstrap, Power Analysis |
| `core/models.py` | Временные ряды: STL, SARIMAX, ARX, обнаружение аномалий |
| `core/attribution.py` | Факторная атрибуция: аддитивная, мультипликативная, Shapley Values |
| `core/simulation.py` | Монте-Карло, сценарный анализ «что если», VaR/CVaR |
| `core/triggers.py` | Правила алертов: пороговые значения, отклонение от базы, изменение тренда |

### 3. Слой визуализации (графики Plotly)

Все графики строятся на Plotly и возвращаются как объекты `go.Figure`. Функции построения графиков находятся внутри модулей страниц и в `core/explore.py`.

### 4. Слой UI (страницы Streamlit)

Двадцать страниц Streamlit формируют пользовательский интерфейс. Каждая страница импортирует только из модулей `core/` и `services/`, но никогда из других страниц.

### 5. Слой хранения (состояние сессии, аудит-лог)

| Модуль | Ответственность |
|--------|----------------|
| `core/audit.py` | Append-only аудит-лог всех операций пользователя |
| `core/i18n.py` | Локализация строк интерфейса (русский / английский) |
| `core/report.py` | Генерация PDF/HTML отчётов через Jinja2 и WeasyPrint |

Состояние сессии хранится в `st.session_state` и является единственным источником данных для текущей аналитической сессии. Управление состоянием централизовано через `app/state.py`.

## Схема модулей

```
app/main.py                              (точка входа)
  |
  +-- app/pages/
  |     0_Start.py           -----> (навигация)
  |     1_Data.py            -----> core/data.py, core/validate.py
  |     2_Prepare.py         -----> core/prepare.py
  |     3_GroupAggregate.py  -----> core/aggregate.py
  |     4_Merge.py           -----> core/merge.py
  |     5_Explore.py         -----> core/explore.py
  |     6_Tests.py           -----> core/tests.py
  |     7_TimeSeries.py      -----> core/models.py
  |     8_Attribution.py     -----> core/attribution.py
  |     9_Simulation.py      -----> core/simulation.py
  |     10_Report.py         -----> core/report.py, core/audit.py
  |     11_Help.py           -----> core/i18n.py
  |     12_Clustering.py     -----> core/explore.py, scikit-learn
  |     14_RollRate.py       -----> core/simulation.py
  |     19_Compare.py        -----> core/explore.py
  |     20_Charts.py         -----> plotly
  |     21_Pipeline.py       -----> core/* (оркестратор)
  |     22_TextAnalytics.py  -----> (текстовый анализ)
  |     24_Templates.py      -----> core/* (пошаговые сценарии)
  |     25_AutoAnalyst.py    -----> core/explore.py, core/tests.py
  |
  +-- core/
  |     data.py           Адаптеры ввода/вывода (CSV, Excel, Parquet, SQL)
  |     validate.py       Валидация схемы, проверка типов
  |     prepare.py        Очистка, парсинг, импутация
  |     aggregate.py      GroupBy, сводные таблицы, скользящие статистики
  |     merge.py          Объединение датасетов, диагностика pitfalls
  |     explore.py        EDA, распределения, корреляции
  |     tests.py          Статистические тесты
  |     models.py         Модели временных рядов, обнаружение аномалий
  |     attribution.py    Факторная декомпозиция
  |     simulation.py     Монте-Карло, сценарное моделирование
  |     triggers.py       Правила и движок алертов
  |     audit.py          Аудит-след операций
  |     i18n.py           Локализация (RU/EN)
  |     report.py         Генерация PDF/HTML отчётов
  |     __init__.py       Публичный API (реэкспорт)
  |
  +-- services/
  |     db.py             Подключение к PostgreSQL (SQLAlchemy)
  |
  +-- sample_data/        Тестовые датасеты (CSV)
  +-- tests/              Набор тестов pytest (343 теста)
```

## Поток данных

Основной поток данных следует линейному конвейеру:

```
  Загрузка         Валидация          Очистка/Подготовка
(CSV/XLSX/    -->  (схема,      -->   (приведение типов,
 Parquet/SQL)      типы,               пропуски,
                   ограничения)        выбросы)
      |                                     |
      v                                     v
  Анализ        <--  Итерации  <--   Трансформация
 (тесты,                             (агрегация,
  модели,                             сводные таблицы,
  атрибуция)                          фильтры)
      |
      v
   Экспорт
 (PDF/HTML отчёт,
  CSV, аудит-лог)
```

### Пошаговое описание

1. **Загрузка** — пользователь загружает данные через файловый загрузчик или подключение к PostgreSQL (`1_Data.py` → `core/data.py`).
2. **Валидация** — автоматическое определение схемы; пользователь проверяет выведенные типы и ограничения (`core/validate.py`).
3. **Очистка/Подготовка** — приведение типов, парсинг дат, импутация пропусков, удаление дублей (`2_Prepare.py` → `core/prepare.py`).
4. **Трансформация** — агрегация по группам, сводные таблицы, вычисляемые колонки (`3_GroupAggregate.py` → `core/aggregate.py`).
5. **Анализ** — EDA, статистические тесты, модели временных рядов, факторная атрибуция, моделирование (`5_Explore` — `9_Simulation`).
6. **Экспорт** — генерация PDF/HTML отчёта, скачивание обработанных данных, просмотр аудит-лога (`10_Report.py` → `core/report.py`).

## Страницы приложения

| № | Страница | Назначение |
|---|----------|------------|
| 0 | Старт | Приветственный экран, выбор сценария |
| 1 | Данные | Загрузка файлов, импорт из PostgreSQL, предпросмотр |
| 2 | Подготовка | Приведение типов, очистка, обработка пропусков |
| 3 | Группировка | Агрегация по группам, сводные таблицы, скользящие статистики |
| 4 | Объединение | JOIN, UNION, конкатенация, диагностика pitfalls |
| 5 | Исследование | EDA: распределения, корреляции, профиль данных |
| 6 | Тесты | Статистическое тестирование гипотез, Power Analysis |
| 7 | Временные ряды | Декомпозиция, прогнозирование, обнаружение аномалий |
| 8 | Атрибуция | Факторный анализ |
| 9 | Моделирование | Монте-Карло, сценарный анализ |
| 10 | Отчёт | Генерация PDF/HTML отчётов |
| 11 | Справка | Руководство пользователя, справочник методологий |
| 12 | Кластеризация | K-means, иерархическая кластеризация, DBSCAN |
| 14 | Roll-Rate | Матрица переходов, Марковские цепи |
| 19 | Сравнение | Сравнение периодов и датасетов |
| 20 | Графики | 18 типов интерактивных графиков |
| 21 | Пайплайн | Визуальный редактор автоматизации |
| 22 | Текстовая аналитика | Анализ текстов, облако слов, тональность |
| 24 | Шаблоны | Пошаговые сценарии с трекером прогресса |
| 25 | Авто-аналитик | Полный анализ одним кликом |

## Ключевые архитектурные решения

- **Нет перекрёстных импортов между страницами**: страницы импортируют только из `core/`. Это предотвращает циклические зависимости и обеспечивает независимое тестирование каждой страницы.
- **Чистые функции в core**: все модули ядра не содержат состояния. Состояние хранится исключительно в `st.session_state` и управляется через `app/state.py`.
- **Plotly для всех графиков**: единообразная интерактивная визуализация на всех страницах с экспортом в PNG/SVG через kaleido.
- **Офлайн-режим**: нет внешних API-вызовов, нет телеметрии. Приложение работает полностью на локальной машине.
- **Дизайн-система**: токены дизайна централизованы в `app/assets/theme.css`; компоненты — в `app/components/`.
- **Локализация**: все строки интерфейса управляются через `core/i18n.py`. Основной язык — русский, резервный — английский.
- **Аудит**: все значимые операции логируются в `core/audit.py` с отметкой времени.

---

## Dash Migration (v5.0)

### Обоснование перехода Streamlit → Dash

Фреймворк заменён со Streamlit на **Dash 2.17.x** + **dash-bootstrap-components 1.6.x** по следующим причинам:

1. **Pure Python** — отсутствует build-шаг, не требуется JS-тулчейн (npm, webpack).
2. **Нативный рендеринг Plotly** — фигуры из `core/` (объекты `go.Figure` / `px.*`) передаются напрямую в `dcc.Graph` без промежуточного слоя.
3. **Локальный Bootstrap CSS** — DBC включает Bootstrap 5 CSS в pip-пакете (без CDN).
4. **Session state** — `dcc.Store` с `storage_type='session'` заменяет `st.session_state`.
5. **Multi-page plugin** — `dash.page_registry` заменяет директорную конвенцию Streamlit `pages/`.

### Структура после миграции

```
app/
├── main.py              # Dash app factory + WSGI server entry point
├── state.py             # dcc.Store schema definitions (constants + helpers)
├── figure_theme.py      # apply_kibad_theme() — KIBAD dark theme for Plotly
├── assets/
│   ├── theme.css        # CSS token system + global dark theme styles
│   ├── layout.css       # Grid/flex layout helpers
│   └── fonts/           # IBM Plex Sans/Mono WOFF2 files (local, no CDN)
├── components/
│   ├── cards.py         # stat_card(label, value, delta)
│   ├── layout.py        # page_header, section_header, empty_state
│   ├── table.py         # data_table(df, id) — styled DataTable
│   ├── nav.py           # sidebar_nav, tab_bar
│   ├── form.py          # select_input, number_input, text_input, slider_input
│   ├── alerts.py        # alert_banner(msg, level)
│   └── upload.py        # upload_zone(id)
└── pages/               # Dash pages registered via dash.register_page()
    ├── p00_start.py     # path='/'
    ├── p01_data.py      # path='/data'
    ├── p02_prepare.py   # path='/prepare'
    ├── p03_group.py     # path='/group'
    ├── p04_merge.py     # path='/merge'
    ├── p05_explore.py   # path='/explore'
    ├── p06_tests.py     # path='/tests'
    ├── p07_timeseries.py# path='/timeseries'
    ├── p08_attribution.py# path='/attribution'
    ├── p09_simulation.py# path='/simulation'
    ├── p10_report.py    # path='/report'
    ├── p11_help.py      # path='/help'
    ├── p12_cluster.py   # path='/cluster'
    ├── p14_rollrate.py  # path='/rollrate'
    ├── p19_compare.py   # path='/compare'
    ├── p20_charts.py    # path='/charts'
    ├── p21_pipeline.py  # path='/pipeline'
    ├── p22_text.py      # path='/text'
    ├── p24_templates.py # path='/templates'
    └── p25_autoanalyst.py# path='/auto'
```

### Ключевые изменения

| Streamlit | Dash |
|-----------|------|
| `st.session_state` | `dcc.Store(storage_type='session')` |
| `st.dataframe(df)` | `dash_table.DataTable` via `data_table(df, id)` |
| `st.plotly_chart(fig)` | `dcc.Graph(figure=apply_kibad_theme(fig))` |
| `st.selectbox` | `dcc.Dropdown` via `select_input()` |
| `st.button` + `st.spinner` | `dbc.Button` + `dcc.Loading` |
| `st.file_uploader` | `dcc.Upload` via `upload_zone()` |
| `st.tabs` | `dbc.Tabs` / `dbc.Tab` |
| `st.expander` | `dbc.Accordion` / `dbc.AccordionItem` |
| `st.download_button` | `dcc.Download` |
| `st.success/error/warning` | `alert_banner(msg, level)` |
| `weasyprint` (PDF) | `reportlab` via `core/report_pdf.py` |

### Sberbank enterprise constraints

- NO external CDN URLs — all JS/CSS served from pip-installed packages.
- NO Node.js / npm / webpack — deployment is `pip install && python app/main.py`.
- NO cloud API calls at runtime.
- `psycopg2` (not `-binary`) — requires `libpq-dev` on host.
- `reportlab` replaces `weasyprint` (no system lib deps).
- `kaleido` is OPTIONAL — image export wrapped in try/except with SVG fallback.
- IBM Plex fonts served from local `app/assets/fonts/` directory.

# KIBAD Dependencies / Зависимости KIBAD

## Dependency Table / Таблица зависимостей

| Package            | Version     | Purpose / Назначение                              |
|--------------------|-------------|---------------------------------------------------|
| streamlit          | >= 1.32.0   | Web UI framework / Веб-интерфейс                  |
| pandas             | >= 2.1.0    | DataFrames, IO / Работа с таблицами               |
| numpy              | >= 1.26.0   | Numeric operations / Числовые операции             |
| plotly             | >= 5.18.0   | Interactive charts / Интерактивные графики         |
| scipy              | >= 1.11.0   | Statistical tests / Статистические тесты          |
| statsmodels        | >= 0.14.1   | Time series, regression / Временные ряды          |
| scikit-learn       | >= 1.4.0    | ML utilities / Утилиты машинного обучения         |
| openpyxl           | >= 3.1.2    | Excel xlsx read/write / Чтение/запись xlsx        |
| xlrd               | >= 2.0.1    | Legacy xls read / Чтение старых xls               |
| pyarrow            | >= 14.0.0   | Parquet support / Поддержка Parquet               |
| psycopg2-binary    | >= 2.9.9    | PostgreSQL driver / Драйвер PostgreSQL            |
| sqlalchemy         | >= 2.0.0    | SQL ORM / SQL-запросы через ORM                   |
| weasyprint         | >= 61.0     | PDF generation / Генерация PDF                    |
| jinja2             | >= 3.1.2    | HTML templates / HTML-шаблоны                     |
| ruptures           | >= 1.1.8    | Change-point detection / Обнаружение точек разрыва|
| kaleido            | >= 0.2.1    | Plotly static export / Экспорт графиков в PNG     |
| pytest             | >= 7.4.0    | Test runner / Запуск тестов                       |
| pytest-cov         | >= 4.1.0    | Coverage reports / Отчёты о покрытии              |

## Vendoring Strategy / Стратегия вендоринга

### Download wheels / Скачивание пакетов

Create a local wheel cache for offline installation:

```bash
# Download all dependencies into ./wheels
# Скачать все зависимости в ./wheels
pip download -d ./wheels -r requirements.txt

# For a specific platform (e.g., Linux x86_64)
# Для конкретной платформы (например, Linux x86_64)
pip download -d ./wheels -r requirements.txt \
    --platform manylinux2014_x86_64 \
    --python-version 3.11 \
    --only-binary=:all:
```

### Offline Install / Установка без интернета

```bash
# Install from local wheel cache
# Установка из локального кэша
pip install --no-index --find-links=./wheels -r requirements.txt
```

### Verifying the cache / Проверка кэша

```bash
# Check all requirements are satisfied from the cache
# Проверить, что все зависимости доступны в кэше
pip install --no-index --find-links=./wheels --dry-run -r requirements.txt
```

## Graceful Degradation / Деградация при отсутствии библиотек

If a dependency is unavailable, the following features degrade:

| Missing Package     | Affected Feature / Затронутый функционал       | Behavior / Поведение                          |
|---------------------|------------------------------------------------|-----------------------------------------------|
| weasyprint          | PDF report export / Экспорт PDF                | Falls back to HTML-only export                |
| kaleido             | Static chart images / Статические изображения  | Charts render interactively only              |
| psycopg2-binary     | PostgreSQL import / Импорт из PostgreSQL       | SQL tab is hidden; file upload still works    |
| xlrd                | Legacy .xls files / Старые файлы .xls          | Only .xlsx and .csv supported                 |
| pyarrow             | Parquet file support / Поддержка Parquet        | Parquet option hidden in upload dialog        |
| ruptures            | Change-point detection / Точки разрыва          | Feature disabled on TimeSeries page           |
| scikit-learn        | Regression attribution / Регрессионная атрибуция| Only additive/multiplicative methods available|

Each optional import is wrapped in a try/except block with a warning shown to
the user when a feature is unavailable.

## Telemetry / Телеметрия

KIBAD makes **zero external network calls**. Streamlit telemetry is explicitly
disabled.

**KIBAD не выполняет никаких внешних сетевых запросов.** Телеметрия Streamlit
явно отключена.

Configuration in `.streamlit/config.toml`:

```toml
[browser]
gatherUsageStats = false

[server]
enableStaticServing = false
```

## Python Version / Версия Python

KIBAD requires Python 3.10 or later. Recommended: Python 3.11.

KIBAD требует Python 3.10 или новее. Рекомендуется: Python 3.11.

## Virtual Environment Setup / Настройка виртуального окружения

```bash
# Create and activate / Создать и активировать
python3.11 -m venv .venv
source .venv/bin/activate

# Online install / Установка с интернетом
pip install -r requirements.txt

# Offline install / Установка без интернета
pip install --no-index --find-links=./wheels -r requirements.txt
```

## System Dependencies / Системные зависимости

WeasyPrint requires system-level libraries for PDF rendering:

WeasyPrint требует системных библиотек для рендеринга PDF:

```bash
# macOS
brew install pango libffi

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0

# RHEL/CentOS
sudo yum install pango gdk-pixbuf2
```

## Updating Dependencies / Обновление зависимостей

```bash
# Check for outdated packages / Проверить устаревшие пакеты
pip list --outdated

# Update requirements.txt after testing / Обновить после тестирования
pip freeze > requirements.lock
```

Always run the full test suite after updating any dependency.

Всегда запускайте полный набор тестов после обновления любой зависимости.

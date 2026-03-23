# Зависимости KIBAD

## Таблица зависимостей

| Пакет | Версия | Назначение |
|-------|--------|------------|
| streamlit | >= 1.32.0 | Веб-интерфейс приложения |
| pandas | >= 2.1.0 | Работа с таблицами, ввод/вывод данных |
| numpy | >= 1.26.0 | Числовые операции |
| plotly | >= 5.18.0 | Интерактивные графики |
| scipy | >= 1.11.0 | Статистические тесты |
| statsmodels | >= 0.14.1 | Временные ряды, регрессия |
| scikit-learn | >= 1.4.0 | Утилиты машинного обучения, кластеризация |
| openpyxl | >= 3.1.2 | Чтение и запись файлов xlsx |
| xlrd | >= 2.0.1 | Чтение устаревших файлов xls |
| pyarrow | >= 14.0.0 | Поддержка формата Parquet |
| psycopg2-binary | >= 2.9.9 | Драйвер PostgreSQL |
| sqlalchemy | >= 2.0.0 | SQL-запросы через ORM |
| weasyprint | >= 61.0 | Генерация PDF-отчётов |
| jinja2 | >= 3.1.2 | HTML-шаблоны для отчётов |
| ruptures | >= 1.1.8 | Обнаружение точек изменения тренда |
| kaleido | >= 0.2.1 | Экспорт графиков Plotly в PNG/SVG |
| pytest | >= 7.4.0 | Запуск тестов |
| pytest-cov | >= 4.1.0 | Отчёты о покрытии кода тестами |

---

## Версия Python

KIBAD требует Python 3.10 или новее. Рекомендуется Python 3.11.

---

## Настройка виртуального окружения

```bash
# Создать и активировать виртуальное окружение
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# или
.venv\Scripts\activate          # Windows

# Установка с доступом к интернету
pip install -r requirements.txt

# Установка без интернета (из локального кэша)
pip install --no-index --find-links=./wheels -r requirements.txt
```

---

## Системные зависимости

WeasyPrint требует системных библиотек для рендеринга PDF:

```bash
# macOS
brew install pango libffi

# Ubuntu/Debian
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0

# RHEL/CentOS
sudo yum install pango gdk-pixbuf2
```

При отсутствии системных библиотек экспорт PDF недоступен; приложение автоматически откатывается к экспорту в HTML.

---

## Офлайн-установка

### Скачивание пакетов в локальный кэш

```bash
# Скачать все зависимости в ./wheels
pip download -d ./wheels -r requirements.txt

# Для конкретной платформы (например, Linux x86_64)
pip download -d ./wheels -r requirements.txt \
    --platform manylinux2014_x86_64 \
    --python-version 3.11 \
    --only-binary=:all:
```

### Установка из локального кэша

```bash
pip install --no-index --find-links=./wheels -r requirements.txt
```

### Проверка кэша

```bash
pip install --no-index --find-links=./wheels --dry-run -r requirements.txt
```

---

## Деградация при отсутствии библиотек

Если зависимость недоступна, соответствующая функциональность деградирует:

| Отсутствующий пакет | Затронутый функционал | Поведение |
|---------------------|----------------------|-----------|
| weasyprint | Экспорт PDF | Откат к экспорту только в HTML |
| kaleido | Статические изображения графиков | Графики остаются только интерактивными |
| psycopg2-binary | Импорт из PostgreSQL | Вкладка SQL скрыта; загрузка файлов работает |
| xlrd | Устаревшие файлы .xls | Поддерживаются только .xlsx и .csv |
| pyarrow | Поддержка Parquet | Опция Parquet скрыта в диалоге загрузки |
| ruptures | Обнаружение точек разрыва | Функция отключена на странице временных рядов |
| scikit-learn | Регрессионная атрибуция, кластеризация | Доступны только аддитивный и мультипликативный методы |

Каждый опциональный импорт обёрнут в блок try/except; при недоступности функции пользователю отображается предупреждение.

---

## Телеметрия

KIBAD не выполняет никаких внешних сетевых запросов. Телеметрия Streamlit явно отключена.

Конфигурация в `.streamlit/config.toml`:

```toml
[browser]
gatherUsageStats = false

[server]
enableStaticServing = false
```

---

## Обновление зависимостей

```bash
# Проверить устаревшие пакеты
pip list --outdated

# Обновить файл зависимостей после тестирования
pip freeze > requirements.lock
```

Всегда запускайте полный набор тестов после обновления любой зависимости:

```bash
python -m pytest tests/ -v
```

"""
core/i18n.py – Lightweight internationalisation for KIBAD.

Usage::

    from core.i18n import t
    st.header(t("page_data_title"))

The active language is stored in ``st.session_state["lang"]`` (default ``"ru"``).
"""
from __future__ import annotations

from typing import Any

_STRINGS: dict[str, dict[str, str]] = {
    # ── Global / Navigation ──────────────────────────────────────────────
    "app_title": {
        "ru": "KIBAD — Интерактивная бизнес-аналитика и прогнозирование",
        "en": "KIBAD — Interactive Business Analytics & Forecasting",
    },
    "app_welcome": {
        "ru": "Добро пожаловать в **KIBAD** — ваш инструмент для комплексного анализа данных.",
        "en": "Welcome to **KIBAD**, your end-to-end analysis studio.",
    },
    "nav_data": {"ru": "Загрузка данных", "en": "Data"},
    "nav_prepare": {"ru": "Подготовка", "en": "Prepare"},
    "nav_group_agg": {"ru": "Группировка и агрегация", "en": "Group & Aggregate"},
    "nav_explore": {"ru": "Исследование", "en": "Explore"},
    "nav_tests": {"ru": "Статистические тесты", "en": "Tests"},
    "nav_timeseries": {"ru": "Временные ряды", "en": "Time Series"},
    "nav_attribution": {"ru": "Факторный анализ", "en": "Factor Attribution"},
    "nav_simulation": {"ru": "Симуляция", "en": "Simulation"},
    "nav_report": {"ru": "Отчёт", "en": "Report"},
    "nav_help": {"ru": "Справка", "en": "Help"},
    "lang_label": {"ru": "Язык / Language", "en": "Language / Язык"},

    # ── Common UI labels ─────────────────────────────────────────────────
    "select_dataset": {"ru": "Выберите датасет", "en": "Select dataset"},
    "no_datasets": {
        "ru": "Датасеты не загружены. Перейдите на страницу **Загрузка данных**.",
        "en": "No datasets loaded yet. Go to the **Data** page to upload one.",
    },
    "upload_files": {"ru": "Загрузить файлы", "en": "Upload files"},
    "download": {"ru": "Скачать", "en": "Download"},
    "export_csv": {"ru": "Экспорт CSV", "en": "Export CSV"},
    "export_xlsx": {"ru": "Экспорт XLSX", "en": "Export XLSX"},
    "export_parquet": {"ru": "Экспорт Parquet", "en": "Export Parquet"},
    "preview": {"ru": "Предпросмотр", "en": "Preview"},
    "apply": {"ru": "Применить", "en": "Apply"},
    "reset": {"ru": "Сбросить", "en": "Reset"},
    "rows": {"ru": "Строки", "en": "Rows"},
    "columns": {"ru": "Столбцы", "en": "Columns"},
    "settings": {"ru": "Настройки", "en": "Settings"},
    "run": {"ru": "Запустить", "en": "Run"},
    "results": {"ru": "Результаты", "en": "Results"},
    "warnings": {"ru": "Предупреждения", "en": "Warnings"},
    "errors": {"ru": "Ошибки", "en": "Errors"},

    # ── Data page ────────────────────────────────────────────────────────
    "page_data_title": {"ru": "Загрузка данных", "en": "Data Ingestion"},
    "page_data_upload_label": {
        "ru": "Загрузите CSV или XLSX файлы (можно несколько)",
        "en": "Upload CSV or XLSX files (multiple allowed)",
    },
    "page_data_separator": {"ru": "Разделитель CSV", "en": "CSV separator"},
    "page_data_encoding": {"ru": "Кодировка", "en": "Encoding"},
    "page_data_schema": {"ru": "Схема данных", "en": "Data Schema"},
    "page_data_profile": {"ru": "Профиль данных", "en": "Data Profile"},
    "page_data_db_section": {"ru": "Подключение к БД", "en": "Database Connection"},
    "datasets_loaded": {"ru": "Загружено датасетов", "en": "Datasets loaded"},
    "forecast_models_run": {"ru": "Моделей прогноза", "en": "Forecast models run"},
    "tests_run": {"ru": "Тестов выполнено", "en": "Tests run"},
    "active_dataset": {"ru": "Активный датасет", "en": "Active dataset"},

    # ── Prepare page ─────────────────────────────────────────────────────
    "page_prepare_title": {"ru": "Подготовка данных", "en": "Data Preparation"},
    "prepare_col_mapping": {"ru": "Маппинг столбцов", "en": "Column Mapping"},
    "prepare_type_overrides": {"ru": "Приведение типов", "en": "Type Overrides"},
    "prepare_date_parsing": {"ru": "Парсинг дат", "en": "Date Parsing"},
    "prepare_numeric_parsing": {"ru": "Парсинг чисел", "en": "Numeric Parsing"},
    "prepare_imputation": {"ru": "Заполнение пропусков", "en": "Missing Value Imputation"},
    "prepare_outliers": {"ru": "Выбросы", "en": "Outlier Removal"},
    "prepare_dedup": {"ru": "Дедупликация", "en": "Deduplication"},
    "prepare_resample": {"ru": "Ресемплирование", "en": "Resampling"},
    "prepare_features": {"ru": "Конструирование признаков", "en": "Feature Engineering"},
    "prepare_validation": {"ru": "Правила валидации", "en": "Validation Rules"},
    "prepare_history": {"ru": "История трансформаций", "en": "Transformation History"},
    "prepare_thousands_sep": {"ru": "Разделитель тысяч", "en": "Thousands separator"},
    "prepare_decimal_sep": {"ru": "Десятичный разделитель", "en": "Decimal separator"},

    # ── Group & Aggregate page ───────────────────────────────────────────
    "page_group_agg_title": {"ru": "Группировка и агрегация", "en": "Group & Aggregate"},
    "group_by_cols": {"ru": "Столбцы группировки", "en": "Group-by columns"},
    "metric_cols": {"ru": "Метрики (числовые столбцы)", "en": "Metric columns (numeric)"},
    "agg_functions": {"ru": "Функции агрегации", "en": "Aggregation functions"},
    "time_bucket": {"ru": "Временной период", "en": "Time bucket"},
    "time_bucket_none": {"ru": "Без группировки по времени", "en": "No time bucket"},
    "numeric_bins": {"ru": "Числовые корзины", "en": "Numeric bins"},
    "pivot_view": {"ru": "Сводная таблица", "en": "Pivot View"},
    "agg_result": {"ru": "Результат агрегации", "en": "Aggregation Result"},
    "weighted_avg_weight_col": {"ru": "Столбец весов (для взвешенного среднего)", "en": "Weight column (for weighted avg)"},

    # ── Explore page ─────────────────────────────────────────────────────
    "page_explore_title": {"ru": "Исследовательский анализ", "en": "Exploratory Analysis"},
    "explore_timeseries": {"ru": "Временной ряд", "en": "Time Series"},
    "explore_distributions": {"ru": "Распределения", "en": "Distributions"},
    "explore_correlation": {"ru": "Корреляция", "en": "Correlation"},
    "explore_missingness": {"ru": "Карта пропусков", "en": "Missingness Map"},
    "explore_outliers": {"ru": "Детекция выбросов", "en": "Outlier Detection"},

    # ── Tests page ───────────────────────────────────────────────────────
    "page_tests_title": {"ru": "Статистические тесты", "en": "Statistical Tests"},
    "test_ttest": {"ru": "t-тест (Уэлча)", "en": "Welch's t-test"},
    "test_mann_whitney": {"ru": "Манна-Уитни", "en": "Mann-Whitney U"},
    "test_chi_square": {"ru": "Хи-квадрат", "en": "Chi-Square"},
    "test_correlation": {"ru": "Тест корреляции", "en": "Correlation Test"},
    "test_bootstrap": {"ru": "Бутстреп-тест", "en": "Bootstrap Test"},
    "test_permutation": {"ru": "Пермутационный тест", "en": "Permutation Test"},
    "test_ab": {"ru": "A/B-тест", "en": "A/B Test"},
    "test_batch": {"ru": "Пакетное тестирование (BH коррекция)", "en": "Batch Testing (BH correction)"},
    "test_significant": {"ru": "Статистически значимо", "en": "Statistically significant"},
    "test_not_significant": {"ru": "Не значимо", "en": "Not significant"},
    "test_effect_size": {"ru": "Размер эффекта", "en": "Effect size"},

    # ── Time Series page ─────────────────────────────────────────────────
    "page_timeseries_title": {"ru": "Временные ряды и прогнозирование", "en": "Time Series & Forecasting"},
    "ts_forecast": {"ru": "Прогноз", "en": "Forecast"},
    "ts_acf_pacf": {"ru": "ACF / PACF", "en": "ACF / PACF"},
    "ts_anomalies": {"ru": "Обнаружение аномалий", "en": "Anomaly Detection"},
    "ts_triggers": {"ru": "Правила оповещений", "en": "Trigger Rules"},
    "ts_decomposition": {"ru": "Декомпозиция", "en": "Decomposition"},
    "ts_model_naive": {"ru": "Сезонный наивный", "en": "Seasonal Naive"},
    "ts_model_arx": {"ru": "ARX (Ridge)", "en": "ARX (Ridge)"},
    "ts_model_sarimax": {"ru": "SARIMAX", "en": "SARIMAX"},

    # ── Factor Attribution page ──────────────────────────────────────────
    "page_attribution_title": {"ru": "Факторный анализ / декомпозиция", "en": "Factor Attribution / Decomposition"},
    "attr_target_col": {"ru": "Целевой показатель", "en": "Target metric"},
    "attr_target_prev": {"ru": "Предыдущее значение целевого показателя", "en": "Previous target value column"},
    "attr_drivers": {"ru": "Факторы-драйверы", "en": "Driver columns"},
    "attr_drivers_prev": {"ru": "Предыдущие значения факторов (суффикс _last)", "en": "Previous driver columns (_last suffix)"},
    "attr_method": {"ru": "Метод декомпозиции", "en": "Decomposition method"},
    "attr_method_additive": {"ru": "Аддитивный (линейный)", "en": "Additive (linear)"},
    "attr_method_ratio": {"ru": "Мультипликативный (ratio)", "en": "Multiplicative (ratio)"},
    "attr_method_regression": {"ru": "Регрессионный", "en": "Regression-based"},
    "attr_method_shapley": {"ru": "Шэпли-аппроксимация", "en": "Shapley approximation"},
    "attr_waterfall": {"ru": "Водопадная диаграмма", "en": "Waterfall Chart"},
    "attr_segment_drilldown": {"ru": "Детализация по сегментам", "en": "Segment Drill-down"},
    "attr_contributions": {"ru": "Вклады факторов", "en": "Factor Contributions"},
    "attr_residual": {"ru": "Остаток", "en": "Residual"},

    # ── Simulation page ──────────────────────────────────────────────────
    "page_simulation_title": {"ru": "Сценарное моделирование", "en": "Scenario Simulation"},

    # ── Report page ──────────────────────────────────────────────────────
    "page_report_title": {"ru": "Генерация отчёта", "en": "Report Generation"},

    # ── Help page ────────────────────────────────────────────────────────
    "page_help_title": {"ru": "Справка по KIBAD", "en": "KIBAD Help"},

    # ── Diagnostics / warnings ───────────────────────────────────────────
    "warn_few_observations": {
        "ru": "Слишком мало наблюдений (n={n}). Результаты могут быть ненадёжными.",
        "en": "Too few observations (n={n}). Results may be unreliable.",
    },
    "warn_zero_variance": {
        "ru": "Столбец «{col}» имеет нулевую дисперсию.",
        "en": "Column '{col}' has zero variance.",
    },
    "warn_missing_critical_col": {
        "ru": "Отсутствует обязательный столбец: {col}",
        "en": "Missing required column: {col}",
    },
    "warn_nan_metrics": {
        "ru": "Метрика содержит NaN — проверьте входные данные.",
        "en": "Metric contains NaN — check your input data.",
    },
    "warn_time_gaps": {
        "ru": "Обнаружены пропуски во временном ряду: {gaps} периодов.",
        "en": "Time gaps detected: {gaps} missing periods.",
    },
    "warn_duplicates_found": {
        "ru": "Найдено {n} дублирующихся строк.",
        "en": "Found {n} duplicate rows.",
    },

    # ── Audit log labels ─────────────────────────────────────────────────
    "audit_file_loaded": {"ru": "Файл загружен", "en": "File loaded"},
    "audit_transform": {"ru": "Трансформация применена", "en": "Transform applied"},
    "audit_export": {"ru": "Экспорт создан", "en": "Export created"},
    "audit_analysis": {"ru": "Анализ выполнен", "en": "Analysis run"},
}


def t(key: str, lang: str | None = None, **kwargs: Any) -> str:
    """Translate a message key.

    Parameters
    ----------
    key:
        Key in the ``_STRINGS`` dictionary.
    lang:
        Language code (``"ru"`` or ``"en"``).  Defaults to
        ``st.session_state["lang"]`` if available, otherwise ``"ru"``.
    **kwargs:
        Format arguments for the string (e.g. ``n=5``).

    Returns
    -------
    str
    """
    if lang is None:
        try:
            import streamlit as st
            lang = st.session_state.get("lang", "ru")
        except Exception:
            lang = "ru"

    entry = _STRINGS.get(key)
    if entry is None:
        return key  # fallback: return the key itself

    text = entry.get(lang, entry.get("ru", key))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


def available_keys() -> list[str]:
    """Return all registered translation keys (useful for testing)."""
    return list(_STRINGS.keys())


def available_languages() -> list[str]:
    """Return supported language codes."""
    return ["ru", "en"]


def register(key: str, translations: dict[str, str]) -> None:
    """Register or update a translation entry at runtime.

    Parameters
    ----------
    key:
        Translation key.
    translations:
        ``{"ru": "...", "en": "..."}`` mapping.
    """
    _STRINGS[key] = translations

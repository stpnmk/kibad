"""
pages/24_Templates.py – Guided analysis templates for KIBAD.

One-click templates that guide users through multi-page analytical workflows.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.state import init_state, list_dataset_names
from core.audit import log_event
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Шаблоны анализа", layout="wide")
init_state()
inject_all_css()

page_header("24. Шаблоны анализа", "Готовые пошаговые сценарии для типовых задач", "🗂️")

# ---------------------------------------------------------------------------
# Template definitions
# ---------------------------------------------------------------------------
TEMPLATES = {
    "🔍 Разведочный анализ нового датасета": {
        "description": "Быстрый обзор любых данных: структура, пропуски, выбросы, корреляции, распределения.",
        "use_case": "Когда нужно: понять незнакомые данные перед началом полноценного анализа.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите файл CSV или Excel. KIBAD автоматически проверит качество данных и выдаст список проблем."),
            ("2_Prepare", "🧹 Первичная очистка", "Просмотрите проблемы: дубликаты, пропуски, выбросы. Примените рекомендуемые исправления одним кликом."),
            ("5_Explore", "🤖 Авто-анализ", "Перейдите на вкладку «Авто-анализ» — KIBAD автоматически найдёт корреляции, аномалии, тренды и выдаст рекомендации."),
            ("5_Explore", "📊 Распределения", "Изучите гистограммы, box plot и violin plot для ключевых колонок."),
            ("3_GroupAggregate", "📋 Сводные таблицы", "Сгруппируйте данные по ключевым измерениям, постройте агрегаты."),
            ("20_Charts", "🎨 Визуализация", "Постройте нужные графики. Авто-предложение подскажет подходящий тип графика по типам колонок."),
        ],
        "tags": ["EDA", "разведка", "первичный анализ"],
        "est_time": "15–30 мин",
    },
    "🧹 Очистка и стандартизация данных": {
        "description": "Полный цикл качества данных: дубликаты, пропуски, типы, выбросы, нормализация.",
        "use_case": "Когда нужно: привести «грязные» данные к пригодному виду перед любым анализом.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите файл. В разделе «Качество данных» увидите автоматический отчёт: пропуски, дубликаты, подозрительные типы."),
            ("2_Prepare", "🔎 Диагностика", "Откройте шаг «Качество данных» — изучите список проблем по каждой колонке с рекомендациями."),
            ("2_Prepare", "🗑️ Дубликаты", "Шаг «Дубликаты» — выберите ключевые колонки, удалите полные или частичные дубли."),
            ("2_Prepare", "❓ Пропуски", "Шаг «Пропущенные значения» — заполните медианой / модой / константой или удалите строки/колонки."),
            ("2_Prepare", "🔢 Типы данных", "Шаг «Типы колонок» — приведите даты, числа, категории к нужному типу."),
            ("2_Prepare", "📐 Нормализация", "Шаг «Вычисляемые колонки» → режим Текст или Арифметика — нормализуйте / стандартизируйте значения при необходимости."),
            ("1_Data", "💾 Экспорт", "Сохраните очищенный датасет через кнопку «Экспорт» в CSV или Excel."),
        ],
        "tags": ["качество данных", "очистка", "ETL"],
        "est_time": "20–45 мин",
    },
    "📈 A/B-тест: сравнение двух групп": {
        "description": "Статистически корректное сравнение двух групп: конверсия, средний показатель, распределения.",
        "use_case": "Когда нужно: оценить эффект нового подхода, продукта, изменения — с расчётом значимости.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите файл с данными по двум группам. Необходима колонка группы (A/B) и числовая метрика."),
            ("2_Prepare", "🧹 Подготовка", "Удалите выбросы, проверьте баланс групп — A и B должны быть сопоставимы по размеру."),
            ("6_Tests", "🔬 Статистические тесты", "Запустите t-тест или Манна-Уитни. Проверьте поправку Бенджамини-Хохберга если метрик несколько. Оцените размер эффекта (Cohen d)."),
            ("19_Compare", "⚖️ Сравнение", "Используйте режим «Сегмент А vs Б» — сравните KPI, посмотрите водопадный график отклонений."),
            ("20_Charts", "📉 Визуализация", "Постройте Box Plot или Violin Plot для наглядного сравнения распределений."),
            ("10_Report", "📄 Отчёт", "Экспортируйте результаты в Excel или PDF с таблицей значимости и графиками."),
        ],
        "tags": ["A/B тест", "статистика", "сравнение"],
        "est_time": "20–40 мин",
    },
    "💰 Прогноз временного ряда": {
        "description": "Прогноз любого числового показателя во времени: тренд, сезонность, доверительный интервал.",
        "use_case": "Когда нужно: предсказать продажи, трафик, объём, нагрузку на N периодов вперёд.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите временной ряд. Нужны колонки: дата и числовой показатель."),
            ("2_Prepare", "🧹 Подготовка", "Убедитесь, что дата распознана как datetime. Заполните пропущенные периоды."),
            ("7_TimeSeries", "⏱️ Прогноз", "Выберите метод: Prophet (сезонность), ARIMA или ETS. Настройте горизонт прогноза. Изучите ACF/PACF для подбора параметров."),
            ("9_Simulation", "🎲 Сценарии + Монте-Карло", "Добавьте шоковые сценарии (оптимистичный / пессимистичный). Запустите Монте-Карло для оценки диапазона неопределённости (VaR, CVaR)."),
            ("10_Report", "📄 Отчёт", "Экспортируйте прогноз с графиками в Excel или PDF."),
        ],
        "tags": ["прогноз", "временные ряды", "планирование"],
        "est_time": "30–50 мин",
    },
    "🔗 Корреляции и регрессионный анализ": {
        "description": "Найдите взаимосвязи между переменными, постройте регрессию, оцените значимость факторов.",
        "use_case": "Когда нужно: понять, что влияет на ключевой показатель, построить объяснительную модель.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите таблицу с числовыми колонками. Определите зависимую переменную (Y) и факторы (X)."),
            ("2_Prepare", "🧹 Подготовка", "Удалите выбросы, проверьте типы колонок. При необходимости создайте лаговые или логарифмические колонки через Формулы."),
            ("5_Explore", "🔬 Корреляционная матрица", "Перейдите на вкладку «Корреляции» — изучите тепловую карту. Найдите сильные связи и мультиколлинеарность."),
            ("5_Explore", "📊 Парные диаграммы", "Вкладка «Pairplot» — постройте матрицу рассеяния для выбранных переменных."),
            ("6_Tests", "📐 Регрессия", "Запустите линейную или множественную регрессию. Изучите R², p-values, коэффициенты. Проверьте остатки на нормальность."),
            ("8_Attribution", "🏗️ Факторный вклад", "Оцените вклад каждого фактора через декомпозицию (Shapley, регрессионные коэффициенты)."),
            ("10_Report", "📄 Отчёт", "Экспортируйте результаты с таблицами коэффициентов и графиками."),
        ],
        "tags": ["корреляции", "регрессия", "взаимосвязи", "статистика"],
        "est_time": "30–60 мин",
    },
    "🎯 Анализ выбросов и аномалий": {
        "description": "Найдите и обработайте нетипичные значения: IQR, z-score, визуальный контроль.",
        "use_case": "Когда нужно: выявить ошибки данных, мошенничество, нетипичные события или очистить данные для моделирования.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите файл. Авто-проверка качества выдаст первые сигналы об аномалиях."),
            ("5_Explore", "🤖 Авто-анализ", "Вкладка «Авто-анализ» — раздел «Аномалии» покажет топ-5 нетипичных строк по методу IQR."),
            ("5_Explore", "📦 Box Plot", "Вкладка «Распределения» → Box Plot — визуально увидите выбросы за усами Тьюки."),
            ("20_Charts", "🎨 Scatter Plot", "Постройте диаграмму рассеяния по двум ключевым показателям — выбросы видны как изолированные точки."),
            ("2_Prepare", "✂️ Обработка", "Шаг «Выбросы» — задайте метод (IQR / z-score), выберите действие: удалить / заменить / отметить флагом."),
            ("6_Tests", "🔬 Тест на нормальность", "Запустите тест Шапиро-Уилка или Колмогорова-Смирнова — убедитесь, что данные после очистки соответствуют ожиданиям."),
            ("10_Report", "📄 Отчёт", "Экспортируйте список найденных аномалий с обоснованием."),
        ],
        "tags": ["аномалии", "выбросы", "качество данных"],
        "est_time": "20–35 мин",
    },
    "📊 Построение KPI-дашборда": {
        "description": "Создайте живой дашборд с ключевыми метриками, трендами и сравнением с предыдущим периодом.",
        "use_case": "Когда нужно: регулярный мониторинг показателей, замена ручного Excel-отчёта.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите файл с историческими данными. Нужны: дата, числовые KPI, опционально — категориальный срез."),
            ("2_Prepare", "🧹 Подготовка", "Убедитесь, что дата распознана как datetime. Создайте нужные расчётные колонки (например: маржа = доход / выручка)."),
            ("18_Dashboard", "📊 KPI-карточки", "Настройте метрики: выберите колонки, единицы, пороги RAG (красный/жёлтый/зелёный). Задайте период сравнения."),
            ("19_Compare", "⚖️ Сравнение периодов", "Используйте «Период А vs Б» — сравните текущий период с предыдущим, посмотрите дельты и водопадный график."),
            ("20_Charts", "📈 Трендовые графики", "Постройте линейные графики динамики KPI. Добавьте двойную ось если метрики разного масштаба."),
            ("3_GroupAggregate", "📋 Разрезы", "Добавьте сводные таблицы по ключевым срезам (продукт, регион, канал)."),
            ("10_Report", "📄 Отчёт", "Сгенерируйте HTML/PDF отчёт — можно отправить коллегам без доступа к KIBAD."),
        ],
        "tags": ["дашборд", "KPI", "мониторинг", "отчётность"],
        "est_time": "30–60 мин",
    },
    "🧩 Сегментация и кластеризация": {
        "description": "Разбейте данные на однородные группы: K-Means, иерархическая кластеризация, PCA-визуализация.",
        "use_case": "Когда нужно: найти естественные группы в данных, персонализировать подход, приоритизировать объекты.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Загрузите таблицу с числовыми признаками для кластеризации."),
            ("2_Prepare", "🧹 Подготовка", "Нормализуйте числовые колонки (min-max или z-score) через Вычисляемые колонки. Удалите выбросы."),
            ("5_Explore", "🔬 Корреляции", "Изучите корреляционную матрицу — устраните мультиколлинеарность, уберите сильно коррелирующие признаки."),
            ("12_Clustering", "🧩 Кластеризация", "Запустите K-Means или иерархическую. Используйте метод локтя для выбора числа кластеров. Изучите PCA-визуализацию."),
            ("3_GroupAggregate", "📊 Профили групп", "Сгруппируйте по полученному кластеру — рассчитайте средние, медианы, доли для каждой группы."),
            ("20_Charts", "📉 Визуализация", "Treemap или Pie chart по размеру сегментов. Bar chart по ключевым показателям каждой группы."),
            ("10_Report", "📄 Отчёт", "Экспортируйте описание сегментов в Excel."),
        ],
        "tags": ["сегментация", "кластеры", "машинное обучение"],
        "est_time": "45–75 мин",
    },
    "📉 Декомпозиция изменения показателя": {
        "description": "Почему изменился показатель? Вклад каждого фактора. Водопадный график отклонений.",
        "use_case": "Когда нужно: объяснить рост или падение метрики между двумя периодами или группами.",
        "steps": [
            ("1_Data", "📥 Загрузка данных", "Нужны данные за два периода или две группы. Можно два отдельных файла — объедините через Merge."),
            ("4_Merge", "🔗 Объединение", "Если данные в двух файлах — объедините по ключевой колонке. Убедитесь, что нет взрыва строк."),
            ("2_Prepare", "🧹 Подготовка", "Создайте колонку изменения: delta = текущее − предыдущее. При необходимости нормализуйте."),
            ("8_Attribution", "🏗️ Факторный анализ", "Выберите метод декомпозиции. Настройте факторы-драйверы. Изучите водопадный график вкладов."),
            ("19_Compare", "⚖️ Сравнение", "Используйте «Период А vs Б» для параллельной верификации результатов."),
            ("10_Report", "📄 Отчёт", "Экспортируйте водопадный график и таблицу вкладов факторов."),
        ],
        "tags": ["факторный анализ", "декомпозиция", "сравнение"],
        "est_time": "30–50 мин",
    },
    "🤖 Автоматизация повторяющегося анализа": {
        "description": "Создайте пайплайн-макрос для автоматического повторения шагов обработки данных.",
        "use_case": "Когда нужно: еженедельный/ежемесячный отчёт, стандартная обработка однотипных выгрузок.",
        "steps": [
            ("1_Data", "📥 Загрузка шаблонного файла", "Загрузите пример файла с нужной структурой колонок."),
            ("2_Prepare", "🧹 Настройте обработку вручную", "Выполните нужные преобразования: очистка, формулы, фильтры — один раз руками."),
            ("21_Pipeline", "⚙️ Создайте пайплайн", "В Пайплайне добавьте шаги: фильтры, расчётные колонки, группировки, переименование. Сохраните пайплайн в JSON-файл."),
            ("21_Pipeline", "▶ Запускайте на новых данных", "В следующий раз: загрузите новый файл → загрузите JSON-пайплайн → запустите → экспортируйте результат."),
            ("10_Report", "📄 Отчёт", "После запуска пайплайна сгенерируйте отчёт одним нажатием."),
        ],
        "tags": ["автоматизация", "пайплайн", "макросы", "повторяемость"],
        "est_time": "20–40 мин (настройка), затем 5 мин/запуск",
    },
}


# ---------------------------------------------------------------------------
# Auto-detection of completed steps based on session state
# ---------------------------------------------------------------------------
def _auto_detect_completion(steps, progress_key):
    """Check session state for evidence that certain steps are already done.

    Returns a set of step indices that were auto-detected as complete.
    """
    auto_detected = set()

    for i, (page_id, step_title, _step_desc) in enumerate(steps):
        title_lower = step_title.lower()

        # Data loading
        if page_id == "1_Data" or "загрузка" in title_lower:
            datasets = st.session_state.get("datasets")
            if datasets and len(datasets) > 0:
                auto_detected.add(i)

        # Prepare / cleaning
        if page_id == "2_Prepare" or any(
            kw in title_lower for kw in ("подготовка", "очистка", "диагностика")
        ):
            transform_logs = st.session_state.get("transform_logs")
            if transform_logs and len(transform_logs) > 0:
                auto_detected.add(i)

        # Statistical tests
        if page_id == "6_Tests" or "тест" in title_lower:
            test_results = st.session_state.get("test_results")
            if test_results and len(test_results) > 0:
                auto_detected.add(i)

        # Forecasting
        if page_id == "7_TimeSeries" or "прогноз" in title_lower:
            forecast_results = st.session_state.get("forecast_results")
            if forecast_results and len(forecast_results) > 0:
                auto_detected.add(i)

        # Group / aggregate
        if page_id == "3_GroupAggregate" or any(
            kw in title_lower for kw in ("сводные", "разрезы")
        ):
            aggregate_results = st.session_state.get("aggregate_results")
            if aggregate_results and len(aggregate_results) > 0:
                auto_detected.add(i)

    # Merge auto-detected into the progress set (don't remove manual marks)
    if progress_key in st.session_state:
        st.session_state[progress_key] |= auto_detected
    else:
        st.session_state[progress_key] = auto_detected

    return auto_detected


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
names = list_dataset_names()

# Template selector
st.markdown("### Выберите сценарий анализа")

# Display as card grid
template_names = list(TEMPLATES.keys())
cols_per_row = 3
rows = [template_names[i:i+cols_per_row] for i in range(0, len(template_names), cols_per_row)]

selected_template = st.session_state.get("selected_template")

for row in rows:
    cols = st.columns(cols_per_row)
    for col, tname in zip(cols, row):
        tmpl = TEMPLATES[tname]
        is_selected = tname == selected_template
        border_color = "#1F3864" if is_selected else "#dee2e6"
        bg_color = "#eef2f9" if is_selected else "#ffffff"
        selected_badge = (
            "<br><span style='background:#1F3864;color:white;padding:2px 10px;"
            "border-radius:50px;font-size:0.72rem;font-weight:700;"
            "letter-spacing:0.05em'>✓ ВЫБРАН</span>"
            if is_selected else ""
        )
        col.markdown(
            f"""<div style="border: 2px solid {border_color}; border-radius: 10px; padding: 14px;
            background: {bg_color}; margin-bottom: 8px; min-height: 130px; cursor: pointer;">
            <b style="font-size:15px">{tname}</b><br>
            <small style="color:#555">{tmpl['description'][:90]}...</small><br><br>
            <small>⏱ {tmpl['est_time']}</small>
            {selected_badge}
            </div>""",
            unsafe_allow_html=True,
        )
        btn_label = "✓ Выбран" if is_selected else "Выбрать"
        if col.button(btn_label, key=f"tmpl_{tname}", use_container_width=True,
                      type="primary" if is_selected else "secondary"):
            st.session_state["selected_template"] = tname
            log_event("template_selected", {"template": tname})
            st.rerun()

# ---------------------------------------------------------------------------
# Selected template detail
# ---------------------------------------------------------------------------
if selected_template and selected_template in TEMPLATES:
    st.divider()
    tmpl = TEMPLATES[selected_template]

    st.markdown(f"## {selected_template}")

    info_col1, info_col2 = st.columns([3, 1])
    with info_col1:
        st.markdown(f"**Описание:** {tmpl['description']}")
        st.markdown(f"**Когда использовать:** {tmpl['use_case']}")
    with info_col2:
        st.metric("⏱ Время", tmpl["est_time"])
        tags_display = " · ".join(tmpl["tags"])
        st.caption(f"🏷️ {tags_display}")

    if not names:
        st.warning("⚠️ Для начала работы загрузите датасет на странице **1. Данные**.")

    # Track step completion in session state
    progress_key = f"tmpl_progress_{selected_template}"
    if progress_key not in st.session_state:
        st.session_state[progress_key] = set()

    # Auto-detect completion from session state
    auto_detected_steps = _auto_detect_completion(tmpl["steps"], progress_key)
    completed_steps = st.session_state[progress_key]

    n_done = len(completed_steps)
    n_total = len(tmpl["steps"])
    all_complete = n_done == n_total

    # Find current step (first uncompleted)
    current_step_idx = None
    for _ci in range(n_total):
        if _ci not in completed_steps:
            current_step_idx = _ci
            break

    # ---- "Start execution" button ----
    st.markdown("### 📋 Пошаговый план")

    if not all_complete:
        if st.button("▶ Начать выполнение", key="start_execution", type="primary",
                      use_container_width=True):
            st.session_state["active_template"] = selected_template
            first_uncompleted = 0
            for _fi in range(n_total):
                if _fi not in completed_steps:
                    first_uncompleted = _fi
                    break
            st.session_state["active_template_step"] = first_uncompleted
            st.success(
                f"Шаблон активирован! Начинайте с шага {first_uncompleted + 1}. "
                "Открывайте страницы по ссылкам и отмечайте шаги по мере выполнения."
            )

    # ---- Step list ----
    for i, (page_id, step_title, step_desc) in enumerate(tmpl["steps"]):
        step_num = i + 1
        is_done = i in completed_steps
        is_current = i == current_step_idx
        is_auto = i in auto_detected_steps and is_done

        step_col1, step_col2, step_col3 = st.columns([0.5, 5, 1.5])

        with step_col1:
            if is_done:
                st.markdown(
                    "<div style='text-align:center;font-size:22px'>✅</div>",
                    unsafe_allow_html=True,
                )
            elif is_current:
                st.markdown(
                    "<div style='text-align:center;font-size:22px'>🔵</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div style='text-align:center;font-size:18px;color:#999;"
                    f"font-weight:bold'>{step_num}</div>",
                    unsafe_allow_html=True,
                )

        with step_col2:
            # Color coding: green = done, blue = current, grey = future
            if is_done:
                bg = "#f0fff0"
                border = "#28a745"
                text_color = "#555"
            elif is_current:
                bg = "#e8f0fe"
                border = "#1a73e8"
                text_color = "#1a73e8"
            else:
                bg = "#f5f5f5"
                border = "#ccc"
                text_color = "#999"

            # Status label
            if is_auto:
                status_label = (
                    "<span style='background:#28a745;color:white;padding:1px 8px;"
                    "border-radius:10px;font-size:0.7rem;margin-left:8px'>"
                    "✅ Автоопределено</span>"
                )
            elif is_current:
                status_label = (
                    "<span style='background:#1a73e8;color:white;padding:1px 8px;"
                    "border-radius:10px;font-size:0.7rem;margin-left:8px'>"
                    "🔵 Текущий шаг</span>"
                )
            else:
                status_label = ""

            st.markdown(
                f"""<div style="background:{bg};border-left:4px solid {border};border-radius:6px;
                padding:10px 14px;margin-bottom:6px;">
                <b style="color:{text_color}">{step_title}</b>{status_label}<br>
                <small style="color:#666">{step_desc}</small>
                </div>""",
                unsafe_allow_html=True,
            )

        with step_col3:
            btn_col1, btn_col2 = st.columns(2)
            with btn_col1:
                st.page_link(f"pages/{page_id}.py", label="Открыть →")
            with btn_col2:
                toggle_label = "↩ Отменить" if is_done else "✓ Готово"
                if st.button(toggle_label, key=f"done_{selected_template}_{i}",
                             use_container_width=True,
                             type="secondary"):
                    if is_done:
                        st.session_state[progress_key].discard(i)
                    else:
                        st.session_state[progress_key].add(i)
                        # Auto-advance: find next uncompleted step
                        next_step = None
                        for j in range(i + 1, n_total):
                            if j not in st.session_state[progress_key]:
                                next_step = j
                                break
                        st.session_state["active_template_step"] = (
                            next_step if next_step is not None else i
                        )
                    st.rerun()

        # Show "Go to next step" button right after marking a step done
        active_step = st.session_state.get("active_template_step")
        if (
            is_done
            and not is_auto
            and active_step is not None
            and active_step == i + 1
            and active_step < n_total
            and active_step not in completed_steps
        ):
            next_page_id, next_title, _ = tmpl["steps"][active_step]
            st.page_link(
                f"pages/{next_page_id}.py",
                label=f"▶ Перейти к следующему шагу → {next_title}",
                icon="➡️",
            )

    # Progress bar
    n_done = len(completed_steps)
    pct = n_done / n_total if n_total > 0 else 0

    st.divider()
    prog_col1, prog_col2 = st.columns([3, 1])
    with prog_col1:
        st.progress(pct, text=f"Выполнено {n_done} из {n_total} шагов")
    with prog_col2:
        if n_done == n_total:
            st.success("🎉🎊 Анализ завершён! Поздравляем!")
        elif n_done > 0:
            st.info(f"⏳ Осталось: {n_total - n_done} шагов")
        else:
            st.caption("Отмечайте шаги по мере выполнения")

    # ---- Completion / reset actions ----
    if n_done == n_total:
        st.balloons()
        st.markdown(
            "<div style='background:#e8f5e9;border:2px solid #4caf50;border-radius:10px;"
            "padding:20px;text-align:center;margin:10px 0'>"
            "<h3 style='color:#2e7d32'>🎊 Все шаги выполнены!</h3>"
            "<p>Шаблон успешно завершён.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        reset_col1, reset_col2 = st.columns(2)
        with reset_col1:
            if st.button("🔄 Начать заново", key="restart_template",
                         use_container_width=True, type="primary"):
                st.session_state[progress_key] = set()
                st.session_state.pop("active_template", None)
                st.session_state.pop("active_template_step", None)
                st.rerun()
        with reset_col2:
            if st.button("📋 Выбрать другой шаблон", key="choose_another",
                         use_container_width=True, type="secondary"):
                st.session_state.pop("selected_template", None)
                st.session_state.pop("active_template", None)
                st.session_state.pop("active_template_step", None)
                st.rerun()
    elif n_done > 0:
        if st.button("🔄 Сбросить прогресс", key="reset_tmpl_progress"):
            st.session_state[progress_key] = set()
            st.session_state.pop("active_template_step", None)
            st.rerun()

    # Related templates
    st.divider()
    st.markdown("### 💡 Похожие шаблоны")
    current_tags = set(tmpl["tags"])
    related = []
    for tname, t in TEMPLATES.items():
        if tname == selected_template:
            continue
        overlap = len(current_tags & set(t["tags"]))
        if overlap > 0:
            related.append((overlap, tname))
    related.sort(reverse=True)

    if related:
        rel_cols = st.columns(min(3, len(related)))
        for col, (_, tname) in zip(rel_cols, related[:3]):
            t = TEMPLATES[tname]
            col.markdown(f"**{tname}**")
            col.caption(t["description"][:80] + "...")
            if col.button("Выбрать →", key=f"related_{tname}", use_container_width=True):
                st.session_state["selected_template"] = tname
                st.rerun()

else:
    # No template selected yet — show tips
    st.divider()
    st.markdown("### 💡 Как использовать шаблоны")
    tip_col1, tip_col2, tip_col3 = st.columns(3)
    with tip_col1:
        st.markdown("""
        <div style="border:1px solid #dee2e6;border-radius:8px;padding:14px;text-align:center">
        <h3>1️⃣</h3><b>Выберите сценарий</b><br>
        <small>Найдите шаблон, соответствующий вашей бизнес-задаче</small>
        </div>
        """, unsafe_allow_html=True)
    with tip_col2:
        st.markdown("""
        <div style="border:1px solid #dee2e6;border-radius:8px;padding:14px;text-align:center">
        <h3>2️⃣</h3><b>Следуйте шагам</b><br>
        <small>Открывайте страницы по ссылкам, отмечайте выполненные шаги</small>
        </div>
        """, unsafe_allow_html=True)
    with tip_col3:
        st.markdown("""
        <div style="border:1px solid #dee2e6;border-radius:8px;padding:14px;text-align:center">
        <h3>3️⃣</h3><b>Экспортируйте</b><br>
        <small>Каждый шаблон заканчивается генерацией отчёта</small>
        </div>
        """, unsafe_allow_html=True)

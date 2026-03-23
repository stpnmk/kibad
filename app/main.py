"""
app/main.py – KIBAD main entry point.

Run with:
    streamlit run app/main.py
"""
import sys
from pathlib import Path

# Make project root importable when running from any directory
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.state import init_state, list_dataset_names
from core.i18n import t
from app.styles import inject_all_css

st.set_page_config(
    page_title="KIBAD – Analytics Studio",
    page_icon="\U0001F4CA",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_state()
inject_all_css()

# Language selector in sidebar
with st.sidebar:
    lang = st.selectbox(
        t("lang_label"),
        ["ru", "en"],
        index=0 if st.session_state.get("lang", "ru") == "ru" else 1,
        key="lang_selector",
    )
    st.session_state["lang"] = lang
    st.divider()
    st.caption("v3.0 · Аналитика без кода")

# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown("""
<div class="kibad-hero">
  <div style="position:relative;z-index:1">
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px">
      <div style="font-size:2.4rem;line-height:1">📊</div>
      <div>
        <div style="font-size:2rem;font-weight:900;color:#ffffff;letter-spacing:-0.03em;
             line-height:1;font-family:Inter,sans-serif">KIBAD</div>
        <div style="font-size:0.75rem;font-weight:600;color:#93c5fd;text-transform:uppercase;
             letter-spacing:0.12em;font-family:Inter,sans-serif">Analytics Studio v3</div>
      </div>
    </div>
    <div style="font-size:1.35rem;font-weight:700;color:#ffffff;margin-bottom:6px;
         font-family:Inter,sans-serif;letter-spacing:-0.01em">
      Профессиональная аналитика без кода
    </div>
    <div style="font-size:0.92rem;color:#bfdbfe;margin-bottom:20px;
         font-family:Inter,sans-serif;font-weight:400">
      Загрузите данные и получите готовый анализ за 5 минут
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);
           border-radius:50px;padding:5px 14px;font-size:0.83rem;font-weight:600;
           color:#ffffff;font-family:Inter,sans-serif;backdrop-filter:blur(4px)">
        ✓ Без Python
      </span>
      <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);
           border-radius:50px;padding:5px 14px;font-size:0.83rem;font-weight:600;
           color:#ffffff;font-family:Inter,sans-serif;backdrop-filter:blur(4px)">
        ✓ Без SQL
      </span>
      <span style="background:rgba(255,255,255,0.15);border:1px solid rgba(255,255,255,0.25);
           border-radius:50px;padding:5px 14px;font-size:0.83rem;font-weight:600;
           color:#ffffff;font-family:Inter,sans-serif;backdrop-filter:blur(4px)">
        ✓ Без Excel
      </span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Stats bar – 5 metric cards
# ---------------------------------------------------------------------------
names = list_dataset_names()
fc_results = st.session_state.get("forecast_results", [])
test_results = st.session_state.get("test_results", [])
agg_results = st.session_state.get("aggregate_results", {})
pipeline_runs = st.session_state.get("pipeline_run_count", 0)

_STAT_ITEMS = [
    ("📂", "#2563eb", "Датасетов", len(names)),
    ("📈", "#059669", "Прогнозов", len(fc_results)),
    ("🔬", "#7c3aed", "Тестов", len(test_results)),
    ("📊", "#d97706", "Группировок", len(agg_results)),
    ("⚙️", "#0369a1", "Пайплайнов", pipeline_runs),
]

stat_cols = st.columns(5)
for col, (emoji, color, label, value) in zip(stat_cols, _STAT_ITEMS):
    with col:
        st.markdown(f"""
<div class="kibad-stat-card" style="border-top-color:{color}">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
    <div style="width:36px;height:36px;border-radius:8px;background:{color}18;
         display:flex;align-items:center;justify-content:center;font-size:1.1rem">
      {emoji}
    </div>
    <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
         letter-spacing:0.08em;color:#94a3b8;font-family:Inter,sans-serif">{label}</div>
  </div>
  <div style="font-size:2rem;font-weight:900;color:#1e293b;line-height:1;
       letter-spacing:-0.03em;font-family:Inter,sans-serif">{value}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Active dataset callout
# ---------------------------------------------------------------------------
if names:
    active = st.session_state.get("active_ds")
    ds = st.session_state.get("datasets", {}).get(active)
    rows_str = f"{len(ds):,}" if ds is not None else "—"
    cols_str = str(len(ds.columns)) if ds is not None else "—"
    others = [n for n in names if n != active]
    others_str = (", ".join(others)) if others else "только этот"
    st.markdown(f"""
<div style="background:#f0fdf4;border-radius:12px;padding:16px 20px;margin-bottom:20px;
     border:1px solid #bbf7d0;display:flex;align-items:center;gap:16px;
     box-shadow:0 2px 8px rgba(5,150,105,0.1)">
  <div style="width:42px;height:42px;min-width:42px;border-radius:10px;
       background:linear-gradient(135deg,#059669 0%,#10b981 100%);
       display:flex;align-items:center;justify-content:center;font-size:1.3rem">✅</div>
  <div style="flex:1">
    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
         letter-spacing:0.08em;color:#059669;font-family:Inter,sans-serif;margin-bottom:3px">
      Активный датасет
    </div>
    <div style="font-size:1rem;font-weight:700;color:#065f46;font-family:Inter,sans-serif">
      {active}
      <span style="font-size:0.8rem;font-weight:500;color:#047857;margin-left:12px">
        {rows_str} строк · {cols_str} столбцов
      </span>
    </div>
    <div style="font-size:0.8rem;color:#6ee7b7;margin-top:2px;font-family:Inter,sans-serif">
      Все датасеты: {others_str}
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
else:
    st.markdown("""
<div style="background:#eff6ff;border-radius:12px;padding:16px 20px;margin-bottom:20px;
     border:1px solid #bfdbfe;display:flex;align-items:center;gap:16px;
     box-shadow:0 2px 8px rgba(37,99,235,0.08)">
  <div style="width:42px;height:42px;min-width:42px;border-radius:10px;
       background:linear-gradient(135deg,#1F3864 0%,#2563eb 100%);
       display:flex;align-items:center;justify-content:center;font-size:1.3rem">👆</div>
  <div style="flex:1">
    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;
         letter-spacing:0.08em;color:#2563eb;font-family:Inter,sans-serif;margin-bottom:3px">
      Данные не загружены
    </div>
    <div style="font-size:0.92rem;font-weight:600;color:#1e40af;font-family:Inter,sans-serif">
      Перейдите на страницу <b>1. Данные</b>, чтобы загрузить датасет и начать работу
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.page_link("pages/24_Templates.py", label="🗂️ Или выберите готовый шаблон анализа →")

# ---------------------------------------------------------------------------
# Module grid — 3-column card grid, grouped by category
# ---------------------------------------------------------------------------
st.markdown("""
<div style="font-size:1.25rem;font-weight:800;color:#1e293b;margin:32px 0 4px 0;
     letter-spacing:-0.02em;font-family:Inter,sans-serif">Модули аналитики</div>
<div style="font-size:0.88rem;color:#64748b;margin-bottom:20px;font-family:Inter,sans-serif">
  Выберите инструмент для вашей задачи
</div>
""", unsafe_allow_html=True)

# Category: Данные и подготовка
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
  <div style="width:3px;height:20px;background:linear-gradient(135deg,#1F3864 0%,#2563eb 100%);
       border-radius:2px"></div>
  <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
       color:#94a3b8;font-family:Inter,sans-serif">Данные и подготовка</span>
</div>
""", unsafe_allow_html=True)

prep_cols = st.columns(3)
_PREP_MODULES = [
    ("📥", "#2563eb", "Данные", "Загрузка CSV, Excel, JSON. Подключение к PostgreSQL с визуальным конструктором запросов — без SQL.", "pages/1_Data.py"),
    ("🔧", "#7c3aed", "Подготовка", "Очистка, удаление выбросов, заполнение пропусков, нормализация. Формулы как в Excel. Банковские пресеты: PAR, NPL, DPD.", "pages/2_Prepare.py"),
    ("🔗", "#0369a1", "Объединение", "JOIN, UNION таблиц с автодиагностикой: взрыв строк, коллинеарность, дублирование ключей.", "pages/4_Merge.py"),
]
for col, (emoji, color, title, desc, page) in zip(prep_cols, _PREP_MODULES):
    with col:
        st.markdown(f"""
<div class="kibad-feature-card">
  <div class="kibad-feature-card-icon" style="background:{color}18;color:{color}">
    {emoji}
  </div>
  <div class="kibad-feature-card-title">{title}</div>
  <div class="kibad-feature-card-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)
        st.page_link(page, label=f"→ Открыть {title.lower()}", use_container_width=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Category: Анализ
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
  <div style="width:3px;height:20px;background:linear-gradient(135deg,#059669 0%,#10b981 100%);
       border-radius:2px"></div>
  <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
       color:#94a3b8;font-family:Inter,sans-serif">Анализ</span>
</div>
""", unsafe_allow_html=True)

_ANALYSIS_MODULES = [
    ("📋", "#059669", "Группировка", "Аналог сводной таблицы Excel. Группируйте по любым измерениям, считайте любые агрегаты. Автоматические графики.", "pages/3_GroupAggregate.py"),
    ("🔍", "#d97706", "Разведочный анализ", "Корреляции, распределения, выбросы, Pairplot. Профиль данных с тепловой картой пропусков.", "pages/5_Explore.py"),
    ("🧪", "#dc2626", "Статистические тесты", "t-тест, Манна-Уитни, ANOVA, хи-квадрат. A/B тест с поправкой Бенджамини-Хохберга. Размер эффекта.", "pages/6_Tests.py"),
    ("📈", "#2563eb", "Временные ряды", "Prophet, ARIMA, ETS. ACF/PACF. Детекция аномалий. Настройка алертов.", "pages/7_TimeSeries.py"),
    ("⚖️", "#7c3aed", "Факторный анализ", "Декомпозиция: вклад каждого фактора в изменение показателя. Водопадный график, сегментация.", "pages/8_Attribution.py"),
    ("🎲", "#0369a1", "Сценарное моделирование", "Что если? Шоковые сценарии по ключевым параметрам. Расчёт воздействия на целевые метрики.", "pages/9_Simulation.py"),
]

for row_start in range(0, len(_ANALYSIS_MODULES), 3):
    row_cols = st.columns(3)
    for col, (emoji, color, title, desc, page) in zip(row_cols, _ANALYSIS_MODULES[row_start:row_start + 3]):
        with col:
            st.markdown(f"""
<div class="kibad-feature-card">
  <div class="kibad-feature-card-icon" style="background:{color}18;color:{color}">
    {emoji}
  </div>
  <div class="kibad-feature-card-title">{title}</div>
  <div class="kibad-feature-card-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)
            st.page_link(page, label=f"→ {title}", use_container_width=True)
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# Category: Визуализация и отчёты
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:14px">
  <div style="width:3px;height:20px;background:linear-gradient(135deg,#d97706 0%,#fbbf24 100%);
       border-radius:2px"></div>
  <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
       color:#94a3b8;font-family:Inter,sans-serif">Визуализация и отчёты</span>
</div>
""", unsafe_allow_html=True)

_REPORT_MODULES = [
    ("🎨", "#2563eb", "Конструктор графиков", "18 типов: Bar, Line, Scatter, Pie, Heatmap, Treemap, Funnel, Свечи. Автовыводы. Скачать PNG.", "pages/20_Charts.py"),
    ("⚖️", "#059669", "Сравнение", "Сравнение двух периодов или двух сегментов. Водопадный график отклонений по метрикам.", "pages/19_Compare.py"),
    ("📄", "#d97706", "Отчёт и экспорт", "Генерация отчёта в HTML и PDF. Экспорт в Excel с форматированием.", "pages/10_Report.py"),
]

report_cols = st.columns(3)
for col, (emoji, color, title, desc, page) in zip(report_cols, _REPORT_MODULES):
    with col:
        st.markdown(f"""
<div class="kibad-feature-card">
  <div class="kibad-feature-card-icon" style="background:{color}18;color:{color}">
    {emoji}
  </div>
  <div class="kibad-feature-card-title">{title}</div>
  <div class="kibad-feature-card-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)
        st.page_link(page, label=f"→ {title}", use_container_width=True)

st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# How it works — 3-step horizontal flow
# ---------------------------------------------------------------------------
st.markdown("""
<div style="font-size:1.1rem;font-weight:800;color:#1e293b;margin-bottom:16px;
     letter-spacing:-0.02em;font-family:Inter,sans-serif">Как это работает</div>
<div style="display:flex;align-items:stretch;gap:0;margin-bottom:32px">

  <div style="flex:1;background:#ffffff;border-radius:12px 0 0 12px;padding:24px 22px;
       box-shadow:0 2px 12px rgba(31,56,100,0.08);border-right:1px solid #f1f5f9">
    <div style="width:44px;height:44px;border-radius:10px;
         background:linear-gradient(135deg,#1F3864 0%,#2563eb 100%);
         display:flex;align-items:center;justify-content:center;
         font-size:1.2rem;margin-bottom:14px;color:#fff;font-weight:800;
         font-family:Inter,sans-serif">1</div>
    <div style="font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:6px;
         font-family:Inter,sans-serif">Загрузить</div>
    <div style="font-size:0.82rem;color:#64748b;line-height:1.5;font-family:Inter,sans-serif">
      CSV, Excel, JSON или подключение к PostgreSQL. Несколько датасетов одновременно.
    </div>
  </div>

  <div style="display:flex;align-items:center;padding:0 2px;background:#f0f4f8;z-index:1">
    <div style="font-size:1.2rem;color:#94a3b8">→</div>
  </div>

  <div style="flex:1;background:#ffffff;padding:24px 22px;
       box-shadow:0 2px 12px rgba(31,56,100,0.08);border-right:1px solid #f1f5f9;
       border-left:1px solid #f1f5f9">
    <div style="width:44px;height:44px;border-radius:10px;
         background:linear-gradient(135deg,#7c3aed 0%,#a78bfa 100%);
         display:flex;align-items:center;justify-content:center;
         font-size:1.2rem;margin-bottom:14px;color:#fff;font-weight:800;
         font-family:Inter,sans-serif">2</div>
    <div style="font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:6px;
         font-family:Inter,sans-serif">Подготовить</div>
    <div style="font-size:0.82rem;color:#64748b;line-height:1.5;font-family:Inter,sans-serif">
      Очистка, формулы, фильтры, объединение таблиц — без единой строки кода.
    </div>
  </div>

  <div style="display:flex;align-items:center;padding:0 2px;background:#f0f4f8;z-index:1">
    <div style="font-size:1.2rem;color:#94a3b8">→</div>
  </div>

  <div style="flex:1;background:#ffffff;border-radius:0 12px 12px 0;padding:24px 22px;
       box-shadow:0 2px 12px rgba(31,56,100,0.08);border-left:1px solid #f1f5f9">
    <div style="width:44px;height:44px;border-radius:10px;
         background:linear-gradient(135deg,#059669 0%,#10b981 100%);
         display:flex;align-items:center;justify-content:center;
         font-size:1.2rem;margin-bottom:14px;color:#fff;font-weight:800;
         font-family:Inter,sans-serif">3</div>
    <div style="font-size:0.95rem;font-weight:700;color:#1e293b;margin-bottom:6px;
         font-family:Inter,sans-serif">Анализировать</div>
    <div style="font-size:0.82rem;color:#64748b;line-height:1.5;font-family:Inter,sans-serif">
      Графики, модели, тесты, отчёты в PDF/Excel — всё прямо в браузере.
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("""
<div style="border-top:1px solid #e2e8f0;padding-top:16px;margin-top:8px;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
  <div style="font-size:0.78rem;color:#94a3b8;font-family:Inter,sans-serif">
    <span style="font-weight:700;color:#475569">KIBAD Analytics Studio</span>
    &nbsp;·&nbsp; v3.0
    &nbsp;·&nbsp; Профессиональная аналитика без кода
  </div>
  <div style="font-size:0.75rem;color:#cbd5e1;font-family:Inter,sans-serif">
    Аналитика · Визуализация · Прогнозирование
  </div>
</div>
""", unsafe_allow_html=True)

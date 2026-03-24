"""
pages/0_Start.py – Guided Workflow landing page for KIBAD.

Entry point for non-technical users. Explains what KIBAD does,
shows quick-start workflow tiles, and links directly to relevant pages.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st
from app.state import init_state, list_dataset_names, store_dataset
from app.styles import inject_all_css

st.set_page_config(page_title="KIBAD – Старт", layout="wide")
init_state()
inject_all_css()


# ---------------------------------------------------------------------------
# Onboarding banner for first-time users
# ---------------------------------------------------------------------------
if not st.session_state.get("onboarding_done", False) and not list_dataset_names():
    st.markdown("""
<div class="kibad-onboarding">
  <div style="font-size:1.05rem;font-weight:800;color:#1e293b;margin-bottom:4px;font-family:Inter,sans-serif">
    👋 Добро пожаловать в KIBAD!
  </div>
  <div style="font-size:0.88rem;color:#475569;margin-bottom:16px;font-family:Inter,sans-serif">
    Три простых шага для первого анализа:
  </div>
  <div class="kibad-onboarding-step">
    <div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#1F3864 0%,#2563eb 100%);
         color:white;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;flex-shrink:0">1</div>
    <div><b>Загрузите данные</b> — CSV или Excel-файл на странице <b>1. Данные</b></div>
  </div>
  <div class="kibad-onboarding-step">
    <div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#7c3aed 0%,#a78bfa 100%);
         color:white;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;flex-shrink:0">2</div>
    <div><b>Очистите данные</b> — удалите дубликаты, заполните пропуски на странице <b>2. Подготовка</b></div>
  </div>
  <div class="kibad-onboarding-step">
    <div style="width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#059669 0%,#10b981 100%);
         color:white;display:flex;align-items:center;justify-content:center;font-size:0.8rem;font-weight:700;flex-shrink:0">3</div>
    <div><b>Запустите анализ</b> — одним кликом на странице <b>25. Авто-аналитик</b></div>
  </div>
</div>
""", unsafe_allow_html=True)

    _ob_c1, _ob_c2, _ob_c3 = st.columns([1, 1, 3])
    with _ob_c1:
        if st.button("📥 Загрузить данные", use_container_width=True, type="primary"):
            st.switch_page("pages/1_Data.py")
    with _ob_c2:
        if st.button("🎲 Попробовать демо-данные", use_container_width=True):
            # Generate a small sales demo dataset inline
            _t = np.arange(24)
            _df_demo = pd.DataFrame({
                "date": pd.date_range("2022-01-01", periods=24, freq="MS"),
                "revenue": np.round(1000 + 12 * _t + 200 * np.sin(2 * np.pi * _t / 12) + np.random.normal(0, 50, 24), 2),
                "units": np.round(80 + 0.5 * _t + 15 * np.sin(2 * np.pi * _t / 12) + np.random.normal(0, 5, 24)).astype(int),
                "segment": np.where(_t < 12, "Старый", "Новый"),
            })
            store_dataset("demo_sales", _df_demo, source="demo")
            st.session_state["onboarding_done"] = True
            st.success("✅ Демо-датасет «demo_sales» загружен (24 строки). Перейдите к анализу!")
            st.rerun()
    with _ob_c3:
        if st.button("✕ Закрыть", use_container_width=False, help="Скрыть онбординг"):
            st.session_state["onboarding_done"] = True
            st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Hero with gradient background and animated dots
# ---------------------------------------------------------------------------
st.markdown("""
<div style="background:linear-gradient(135deg,#1F3864 0%,#2563eb 100%);border-radius:16px;
     padding:44px 48px;margin-bottom:32px;position:relative;overflow:hidden">

  <!-- Animated floating dots -->
  <div class="hero-dot" style="width:80px;height:80px;top:10%;right:8%;animation-delay:0s"></div>
  <div class="hero-dot" style="width:50px;height:50px;top:55%;right:18%;animation-delay:1.5s"></div>
  <div class="hero-dot" style="width:30px;height:30px;top:20%;right:28%;animation-delay:0.8s"></div>
  <div class="hero-dot" style="width:120px;height:120px;bottom:-30px;right:4%;animation-delay:2.2s;opacity:0.08"></div>

  <div style="position:relative;z-index:1;max-width:600px">
    <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.12em;
         color:#93c5fd;margin-bottom:12px;font-family:Inter,sans-serif">
      Аналитика без кода
    </div>
    <div style="font-size:2.2rem;font-weight:900;color:#ffffff;line-height:1.1;
         letter-spacing:-0.03em;font-family:Inter,sans-serif;margin-bottom:10px">
      KIBAD — Старт
    </div>
    <div style="font-size:1.05rem;color:#bfdbfe;margin-bottom:24px;
         font-family:Inter,sans-serif;font-weight:400;line-height:1.5">
      Загрузите данные и получите готовый анализ за 5 минут — без единой строки кода
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <span style="background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.3);
           border-radius:50px;padding:5px 15px;font-size:0.82rem;font-weight:600;
           color:#ffffff;font-family:Inter,sans-serif">✓ Без Python</span>
      <span style="background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.3);
           border-radius:50px;padding:5px 15px;font-size:0.82rem;font-weight:600;
           color:#ffffff;font-family:Inter,sans-serif">✓ Без SQL</span>
      <span style="background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.3);
           border-radius:50px;padding:5px 15px;font-size:0.82rem;font-weight:600;
           color:#ffffff;font-family:Inter,sans-serif">✓ Без Excel</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# CTA buttons
cta1, cta2, cta3, _spacer = st.columns([1, 1, 1, 2])
with cta1:
    st.page_link("pages/1_Data.py", label="📥 Загрузить данные", use_container_width=True)
with cta2:
    st.page_link("pages/24_Templates.py", label="🗂️ Шаблоны анализа", use_container_width=True)
with cta3:
    st.page_link("pages/11_Help.py", label="📖 Справка", use_container_width=True)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Что вы хотите сделать? — workflow tiles
# ---------------------------------------------------------------------------
st.markdown("""
<div style="font-size:1.1rem;font-weight:800;color:#1e293b;margin-bottom:6px;
     letter-spacing:-0.02em;font-family:Inter,sans-serif">Что вы хотите сделать?</div>
<div style="font-size:0.85rem;color:#64748b;margin-bottom:18px;font-family:Inter,sans-serif">
  Выберите задачу — инструмент откроется сразу
</div>
""", unsafe_allow_html=True)

# Category divider helper
def _cat_header(label: str, color: str) -> None:
    st.markdown(f"""
<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;margin-top:20px">
  <div style="width:3px;height:18px;background:{color};border-radius:2px"></div>
  <span style="font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.1em;
       color:#94a3b8;font-family:Inter,sans-serif">{label}</span>
</div>
""", unsafe_allow_html=True)

# Tile helper
def _tile(icon: str, color: str, title: str, desc: str, page_path: str) -> None:
    st.markdown(f"""
<div class="start-tile">
  <div class="start-tile-icon" style="background:{color}18">
    {icon}
  </div>
  <div class="start-tile-title">{title}</div>
  <div class="start-tile-desc">{desc}</div>
</div>
""", unsafe_allow_html=True)
    st.page_link(page_path, label=f"→ {title}", use_container_width=True)
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# --- Подготовка данных ---
_cat_header("Подготовка данных", "linear-gradient(135deg,#1F3864 0%,#2563eb 100%)")
prep_cols = st.columns(3)
_PREP = [
    ("📥", "#2563eb", "Загрузить данные", "CSV, Excel, JSON или PostgreSQL. Несколько датасетов одновременно.", "pages/1_Data.py"),
    ("🔧", "#7c3aed", "Подготовить данные", "Очистка, формулы, фильтры, нормализация. Банковские пресеты: PAR, NPL, DPD.", "pages/2_Prepare.py"),
    ("🔗", "#0369a1", "Объединить таблицы", "JOIN, UNION с автодиагностикой взрыва строк и дублирования ключей.", "pages/4_Merge.py"),
]
for col, args in zip(prep_cols, _PREP):
    with col:
        _tile(*args)

# --- Анализ ---
_cat_header("Анализ", "linear-gradient(135deg,#059669 0%,#10b981 100%)")
analysis_cols = st.columns(3)
_ANALYSIS = [
    ("🎨", "#2563eb", "Конструктор графиков", "18 типов: Bar, Line, Scatter, Pie, Heatmap, Treemap, Funnel, Свечи. Автовыводы.", "pages/20_Charts.py"),
    ("👥", "#059669", "Сегментация клиентов", "K-Means, иерархическая кластеризация, PCA. Визуализация сегментов.", "pages/12_Clustering.py"),
    ("📈", "#d97706", "Прогноз показателей", "Prophet, ARIMA, ETS. Детекция аномалий. Настройка алертов.", "pages/7_TimeSeries.py"),
]
for col, args in zip(analysis_cols, _ANALYSIS):
    with col:
        _tile(*args)

analysis2_cols = st.columns(3)
_ANALYSIS2 = [
    ("🔍", "#dc2626", "Разведочный анализ", "Корреляции, распределения, выбросы, Pairplot. Профиль данных.", "pages/5_Explore.py"),
    ("🧪", "#7c3aed", "Статистические тесты", "t-тест, Mann-Whitney, ANOVA. A/B тест с поправкой Бенджамини-Хохберга.", "pages/6_Tests.py"),
    ("⚙️", "#475569", "Автоматизация (Пайплайн)", "Создайте макрос из операций. Сохраните в JSON, запускайте на новых данных.", "pages/21_Pipeline.py"),
]
for col, args in zip(analysis2_cols, _ANALYSIS2):
    with col:
        _tile(*args)

analysis3_cols = st.columns(3)
_ANALYSIS3 = [
    ("👥", "#059669", "Когортный анализ", "Удержание, отток, LTV. Тепловая карта ретеншена по когортам.", "pages/13_Cohort.py"),
    ("🏷️", "#7c3aed", "ABC-XYZ анализ", "Классификация по вкладу и стабильности. Кросс-матрица 9 сегментов.", "pages/15_ABC.py"),
    ("🔻", "#dc2626", "Воронка конверсий", "Этапы процесса: конверсия, потери, узкие места. Из данных или вручную.", "pages/16_Funnel.py"),
]
for col, args in zip(analysis3_cols, _ANALYSIS3):
    with col:
        _tile(*args)

# --- Отчёты ---
_cat_header("Отчёты и экспорт", "linear-gradient(135deg,#d97706 0%,#fbbf24 100%)")
report_cols = st.columns(3)
_REPORTS = [
    ("⚖️", "#059669", "Сравнение показателей", "Сравните периоды и сегменты: дельты, водопадный график, детализация.", "pages/19_Compare.py"),
    ("📋", "#2563eb", "Группировка и агрегация", "Сгруппируйте по любым измерениям. Аналог сводной таблицы Excel.", "pages/3_GroupAggregate.py"),
    ("📄", "#d97706", "Отчёт и экспорт", "Генерация отчёта в PDF/HTML с таблицами и графиками. Экспорт в Excel.", "pages/10_Report.py"),
]
for col, args in zip(report_cols, _REPORTS):
    with col:
        _tile(*args)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Что нового в v3 — changelog timeline
# ---------------------------------------------------------------------------
st.markdown("""
<div style="font-size:1.05rem;font-weight:800;color:#1e293b;margin-bottom:14px;
     letter-spacing:-0.02em;font-family:Inter,sans-serif">Что нового в v3</div>
""", unsafe_allow_html=True)

_CHANGELOG = [
    ("#2563eb", "Модуль Merge", "Объединение таблиц (JOIN/UNION) с автодиагностикой: взрыв строк, коллинеарность, дублирование ключей, анализ мощности ключей."),
    ("#059669", "Pairplot и Data Profile", "На странице Explore: матрица scatter-графиков (Pairplot) + автопрофиль данных с тепловой картой пропусков и распределением колонок."),
    ("#7c3aed", "Прогнозирование v2", "Улучшенный UX: карточки методов на русском, улучшенные графики с граничной линией, бэндами доверия и метрическими карточками."),
    ("#d97706", "Новый дизайн", "Полностью переработанный UI: Inter шрифт, градиентные акценты, премиум карточки с тенями, underline-табы, улучшенные метрики."),
    ("#dc2626", "PDF экспорт", "Graceful error для отсутствующих системных библиотек (WeasyPrint). Понятное сообщение вместо crash."),
]

timeline_html = '<div style="background:#ffffff;border-radius:12px;padding:18px 24px;box-shadow:0 2px 12px rgba(31,56,100,0.08)">'
for i, (color, title, desc) in enumerate(_CHANGELOG):
    border = "border-bottom:1px solid #f1f5f9;" if i < len(_CHANGELOG) - 1 else ""
    timeline_html += f"""
<div style="display:flex;gap:14px;padding:14px 0;{border}align-items:flex-start">
  <div style="width:10px;height:10px;min-width:10px;border-radius:50%;
       background:{color};margin-top:5px"></div>
  <div>
    <div style="font-size:0.88rem;font-weight:700;color:#1e293b;margin-bottom:3px;
         font-family:Inter,sans-serif">{title}</div>
    <div style="font-size:0.8rem;color:#64748b;line-height:1.5;font-family:Inter,sans-serif">{desc}</div>
  </div>
</div>"""
timeline_html += "</div>"
st.markdown(timeline_html, unsafe_allow_html=True)

st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Популярные рабочие процессы
# ---------------------------------------------------------------------------
st.markdown("""
<div style="font-size:1.05rem;font-weight:800;color:#1e293b;margin-bottom:14px;
     letter-spacing:-0.02em;font-family:Inter,sans-serif">Популярные рабочие процессы</div>
""", unsafe_allow_html=True)

_WORKFLOWS = [
    (
        "🔬",
        "A/B тест",
        ["📥 Загрузить данные", "🔧 Подготовить (группа A/B)", "🧪 Тест Манна-Уитни", "📄 Экспорт отчёта"],
        "#7c3aed",
        "pages/6_Tests.py",
    ),
    (
        "👥",
        "Сегментация клиентов",
        ["📥 Загрузить данные", "🔧 Нормализовать", "👥 K-Means кластеризация", "🎨 Визуализация PCA"],
        "#059669",
        "pages/12_Clustering.py",
    ),
    (
        "📈",
        "Прогноз выручки",
        ["📥 Временной ряд", "🔍 ACF/PACF анализ", "📈 Prophet прогноз", "📊 KPI дашборд"],
        "#2563eb",
        "pages/7_TimeSeries.py",
    ),
]

wf_cols = st.columns(3)
for col, (emoji, title, steps, color, page) in zip(wf_cols, _WORKFLOWS):
    with col:
        steps_html = "".join(
            f'<div style="font-size:0.78rem;color:#475569;padding:4px 0;border-bottom:1px solid #f1f5f9;'
            f'font-family:Inter,sans-serif">{s}</div>'
            for s in steps
        )
        st.markdown(f"""
<div class="workflow-card">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px">
    <div style="width:36px;height:36px;border-radius:8px;background:{color};
         display:flex;align-items:center;justify-content:center;font-size:1.1rem">{emoji}</div>
    <div style="font-size:0.92rem;font-weight:700;color:#1e293b;font-family:Inter,sans-serif">{title}</div>
  </div>
  {steps_html}
</div>
""", unsafe_allow_html=True)
        st.page_link(page, label=f"→ Начать: {title}", use_container_width=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Loaded datasets notice
# ---------------------------------------------------------------------------
loaded = list_dataset_names()
if loaded:
    names_str = ", ".join(f"<b>{n}</b>" for n in loaded)
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:12px;
        padding:14px 20px;font-size:0.88rem;color:#047857;font-family:Inter,sans-serif;
        box-shadow:0 2px 8px rgba(5,150,105,0.08)">
        ✅ У вас загружены датасеты: {names_str}.
        Перейдите к нужному инструменту выше или продолжите работу.
        </div>""",
        unsafe_allow_html=True,
    )

"""
pages/8_Help.py – Usage guide and documentation.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from app.state import init_state
from app.styles import inject_all_css, page_header, section_header

st.set_page_config(page_title="KIBAD – Help", layout="wide")
init_state()
inject_all_css()

page_header("11. Справка", "Руководство пользователя и FAQ", "❓")

section_header("Что такое KIBAD?")

st.markdown("""
**KIBAD** (KPI Interactive Business Analytics & Data studio) — платформа сквозного анализа данных,
которая позволяет аналитикам загружать датасеты, исследовать данные, проводить статистические тесты,
строить прогнозы временных рядов и моделировать бизнес-сценарии — без написания кода.
""")

st.divider()

section_header("Быстрый старт — 7 шагов")

st.markdown("""
| Шаг | Страница | Что делать |
|-----|----------|------------|
| 1 | **Данные** | Загрузите CSV/XLSX или подключитесь к PostgreSQL. |
| 2 | **Подготовка** | Задайте маппинг колонок (дата, цель, сегмент), очистите данные, ресемплируйте. |
| 3 | **Исследование** | Стройте графики, распределения, корреляции; создавайте KPI. |
| 4 | **Тесты** | Запускайте t-тест, Манна–Уитни, хи-квадрат, корреляцию или A/B-тест. |
| 5 | **Временные ряды** | Выберите модель (Наивный / ARX / SARIMAX), задайте горизонт, запустите прогноз. |
| 6 | **Моделирование** | Задавайте шоки слайдерами и сравнивайте сценарии. |
| 7 | **Отчёт** | Сгенерируйте и скачайте HTML или PDF отчёт. |
""")

st.divider()

section_header("Описание страниц")

col1, col2 = st.columns(2)

with col1:
    with st.expander("Страница 1 – Данные", expanded=False):
        st.markdown("""
**Upload tab:**
- Drag and drop a CSV or Excel file.
- Choose a dataset name and CSV separator (`,`, `;`, `\\t`, `|`).
- Click **Load File** to parse and store.

**PostgreSQL tab:**
- Enter host, port, database, user, password.
- Write any SELECT query; rows are streamed for large tables.
- The result is stored as a named dataset.

**Dataset Catalog:**
- Preview top-N rows.
- See column profiling (missingness, unique counts, sample values).
- Inspect numeric descriptive statistics.
- Set per-column type overrides (datetime / numeric / categorical / boolean).
        """)

    with st.expander("Страница 2 – Подготовка", expanded=False):
        st.markdown("""
**Column Mapping:**
Map real column names to logical roles:
- `date` → used for time series and resampling
- `target` → the main metric to forecast
- `segment` → optional categorical grouping

**Cleaning steps (apply in any order):**
1. **Type overrides** – apply date/numeric/boolean casts.
2. **Date parsing** – converts strings to datetime64; strips timezone.
3. **Missing value imputation** – median, mean, mode, ffill, bfill, or zero-fill.
4. **Outlier removal** – IQR fence or z-score threshold; reports rows removed.
5. **Deduplication** – drops exact duplicates on selected columns.

**Resampling:**
- Supports Daily, Weekly (W-MON), Monthly Start (MS), Monthly End (ME), Quarterly.
- Aggregation: sum, mean, median, last, min, max.
- Optional group-by for segment-level resampling.

**Feature Engineering:**
- **Lags** – shift columns backward (1, 2, 3, 12 periods etc.)
- **Rolling** – rolling mean, std, sum with configurable window
- **EMA** – exponentially-weighted moving average
- **Buckets** – quantile bins or custom bin edges
- **Normalize** – z-score or min-max scaling
- **Interaction** – multiply, divide, add, subtract two columns
        """)

    with st.expander("Страница 3 – Исследование", expanded=False):
        st.markdown("""
**Time Series:** Multi-line chart with optional segment color. Download as PNG.

**Distributions:**
- Histogram with optional KDE (kernel density estimate) overlay.
- Box plots grouped by a categorical column.

**Correlation:**
- Pearson / Spearman / Kendall heatmap.
- Lag cross-correlation tool: find leading/lagging relationships.

**Pivot Aggregation Builder:**
- Choose row dimension, optional column dimension, metric, and aggregation.
- Result shown as table + bar chart; export to CSV.

**Waterfall Chart:**
- Manual: enter factor names and delta values.
- Auto: period-over-period delta per selected column.

**STL Decomposition:**
- Decomposes a time series into Trend + Seasonal + Residual.
- Adjust seasonal period (12 = monthly annual, 52 = weekly).

**KPI Builder:**
- Write Python expressions (e.g., `revenue / sessions`).
- KPI cards show last value and period-over-period % change.
        """)

with col2:
    with st.expander("Страница 4 – Тесты", expanded=False):
        st.markdown("""
All tests report:
- **Test statistic** and **p-value**.
- **Effect size** with a plain-English magnitude label (negligible / small / medium / large).
- **Business interpretation** in non-technical language.

| Test | When to use |
|------|-------------|
| **t-Test** | Compare means of two groups (normally distributed) |
| **Mann–Whitney** | Compare medians of two groups (non-parametric) |
| **Chi-Square** | Test association between two categorical variables |
| **Correlation** | Measure and test linear/rank relationship between two numeric columns |
| **Bootstrap** | Robust mean/median comparison without normality assumption |
| **A/B Test** | Full suite: runs all three tests + effect size + lift % |

Set the significance level **α** (0.01–0.10) in the sidebar.
Test history is accumulated and exportable as CSV.
        """)

    with st.expander("Страница 5 – Прогнозирование", expanded=False):
        st.markdown("""
**Models available:**

| Model | Best for | Notes |
|-------|----------|-------|
| **Seasonal Naive** | Stable seasonal business data | Repeats last seasonal cycle; unbeatable baseline |
| **ARX (Ridge)** | Interpretable regression; any series | Lag features + exogenous variables; coefficient table provided |
| **SARIMAX** | Complex seasonality + exog | Full ARIMA with seasonal orders; AIC/BIC and parameter table |

**Backtesting:**
- Rolling window evaluation.
- Select model, number of folds, minimum training size, horizon per fold.
- Reports per-fold MAE, RMSE, MAPE, Bias.

**Forecast chart** shows:
- Black line = actual history
- Dotted blue = in-sample fitted values
- Orange solid = forecast
- Shaded band = 95% confidence interval

**Compare tab:** overlay all models side-by-side.
        """)

    with st.expander("Страница 6 – Сценарное моделирование", expanded=False):
        st.markdown("""
**Workflow:**
1. Configure date, target, exogenous variables, and horizon in the sidebar.
2. Use **sliders** to set percentage shocks per exogenous variable (e.g., +10% rate, -5% volume).
3. Click **Run Simulation** — baseline and shocked forecasts are computed.

**Charts:**
- **Chart 1** – Actual history + baseline + scenario paths on the same timeline.
  The actual line ends at the last available date; no forward extension.
- **Chart 2** – Delta bar chart (scenario minus baseline per period).
- **Chart 3** – Component flow stacked bars (if component columns are selected).

**Export:** Download scenario results as CSV.

**Presets:** Save named scenario configurations (shock parameters) and reload them later.
The preset library can be exported as JSON for sharing.
        """)

    with st.expander("Страница 7 – Отчёт", expanded=False):
        st.markdown("""
**What's included:**
- Dataset overview card (rows, columns, missingness).
- Data profile table.
- Auto business summary (trend direction, momentum, volatility, forecast narrative, caveats).
- Time series chart, correlation heatmap, STL decomposition.
- Forecast model results and explainability tables.
- Statistical test results with interpretations.
- Methods section describing all techniques used.

**Output formats:**
- **HTML** – interactive charts, embedded via Plotly div.
- **PDF** – requires WeasyPrint (`pip install weasyprint`).

Both formats are downloadable directly from the page.
        """)

st.divider()

section_header("Частые проблемы и решения")

with st.expander("Таблица типичных ошибок", expanded=False):
    st.markdown("""
| Problem | Solution |
|---------|----------|
| **"No datetime columns"** | Go to **Prepare** → Date Parsing and parse your date column. |
| **"Need at least 2 numeric columns"** | Ensure your columns are parsed as numeric (type overrides in Data). |
| **SARIMAX takes too long** | Reduce seasonal order or series length; use ARX instead. |
| **Outlier removal removes too many rows** | Increase IQR multiplier (e.g., 2.5 or 3.0). |
| **STL fails with "not enough data"** | Need ≥ 2× the seasonal period of non-null data points. |
| **Forecast CI not showing** | Ensure there are future dates (horizon ≥ 1). |
| **PostgreSQL connection fails** | Check host/port/credentials; ensure psycopg2-binary is installed. |
| **WeasyPrint PDF fails** | Install system dependencies (Cairo, Pango) per WeasyPrint docs. |
    """)

with st.expander("Примеры датасетов", expanded=False):
    st.markdown("""
Located in the `data/` folder of the project:

| File | Description |
|------|-------------|
| `sample_monthly_sales.csv` | Monthly sales, costs, units — ideal for forecasting demos |
| `sample_ab_test.csv` | Two-group experiment data — use for A/B test page |
| `sample_multivariate.csv` | Multi-segment monthly KPIs with exogenous variables |

**Load a sample:**
1. Go to **Data** → Upload tab.
2. Click the uploader and navigate to the `data/` folder.
3. Choose a file and click **Load File**.
    """)

with st.expander("Советы по работе", expanded=False):
    st.markdown("""
- Every chart has a **Download PNG** button (hover over the chart → camera icon in Plotly toolbar).
- Every table has a **Download CSV** button below it.
- Use **sidebar** on Forecast and Simulation pages to configure parameters globally.
- Session state persists within the current browser session. Refresh = data loss.
  Download your prepared CSV from the **Prepare** page to avoid re-work.
- Multiple datasets can be loaded simultaneously and switched via the selectbox on each page.
    """)

st.info("По вопросам и предложениям — создайте issue в репозитории проекта.")

st.divider()
section_header("Банковский глоссарий")

glossary = {
    "PD (Probability of Default)": "Вероятность дефолта — вероятность того, что заёмщик не выполнит свои обязательства в течение определённого периода (обычно 12 месяцев для Stage 1, LT для Stage 2).",
    "LGD (Loss Given Default)": "Потери при дефолте — доля кредита, которая не будет возмещена в случае дефолта (диапазон: 0–1 или 0–100%).",
    "EAD (Exposure at Default)": "Величина под риском на момент дефолта — сумма обязательств заёмщика на момент дефолта.",
    "EL (Expected Loss)": "Ожидаемые потери: EL = PD × LGD × EAD.",
    "ECL (Expected Credit Loss)": "Ожидаемые кредитные убытки по МСФО 9 — включают Stage 1 (12М), Stage 2 (LTL) и Stage 3 (обесценение).",
    "DPD (Days Past Due)": "Просроченность в днях — количество дней просрочки платежа.",
    "MOB (Month on Book)": "Время жизни кредита в месяцах с момента выдачи.",
    "CDR (Cumulative Default Rate)": "Накопленный уровень дефолтов — доля дефолтов, накопленных к данному MOB.",
    "NPL (Non-Performing Loan)": "Неработающий кредит — кредит с просрочкой > 90 дней.",
    "HHI (Herfindahl-Hirschman Index)": "Индекс концентрации портфеля: сумма квадратов долей. HHI < 0.10 — низкая концентрация, > 0.18 — высокая.",
    "WAR (Weighted Average Rate)": "Средневзвешенная ставка — средняя процентная ставка, взвешенная по размеру кредита.",
    "WAM (Weighted Average Maturity)": "Средневзвешенная срочность — средний срок до погашения, взвешенный по EAD.",
    "DV01": "Dollar Value of 1 basis point — изменение стоимости портфеля при сдвиге ставки на 1 б.п. (0.01%).",
    "VaR (Value at Risk)": "Стоимость под риском — максимально ожидаемые потери с заданной вероятностью (напр., 99% за 10 дней).",
    "RAROC (Risk-Adjusted Return on Capital)": "Доходность на капитал с учётом риска: (Выручка - EL) / Экономический капитал.",
}

for term, definition in glossary.items():
    with st.expander(f"**{term}**"):
        st.markdown(definition)

st.divider()
section_header("Быстрый старт с демо-данными")

import pandas as _pd_help
ROOT_HELP = Path(__file__).parent.parent.parent

demo_files = list((ROOT_HELP / "data").glob("*.csv")) if (ROOT_HELP / "data").exists() else []
if demo_files:
    from app.state import store_dataset
    demo_choice = st.selectbox("Выберите демо-датасет", [f.name for f in demo_files], key="help_demo_sel")
    if st.button("📂 Загрузить демо-датасет", key="btn_help_demo"):
        chosen_file = ROOT_HELP / "data" / demo_choice
        df_demo = _pd_help.read_csv(chosen_file)
        ds_name_demo = demo_choice.rsplit(".", 1)[0]
        store_dataset(ds_name_demo, df_demo, source="demo")
        st.success(f"✅ Загружен «{ds_name_demo}»: {df_demo.shape[0]:,} × {df_demo.shape[1]}. Перейдите на страницу **1. Данные**.")
else:
    st.info("Демо-файлы не найдены в папке `data/`. Загрузите свои данные на странице **1. Данные**.")

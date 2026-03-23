# KIBAD Architecture

## Overview

KIBAD (KPI-Based Analytical Dashboard) is a multi-page Streamlit application for
interactive data analysis. It follows a layered architecture that separates
concerns into five distinct layers: Core, Analytics, Visualization, UI, and
Storage.

## Layer Separation

### 1. Core Layer (IO, Schema, Validation, Transforms)

Handles all data ingestion, validation, and transformation logic. No UI
dependencies -- pure Python functions that can be tested independently.

| Module              | Responsibility                                        |
|---------------------|-------------------------------------------------------|
| `core/data.py`      | File I/O: CSV, Excel (xlsx/xls), Parquet, PostgreSQL  |
| `core/validate.py`  | Schema checks, type inference, constraint enforcement  |
| `core/prepare.py`   | Cleaning, type coercion, missing value handling        |
| `core/aggregate.py` | Group-by operations, pivot tables, rolling windows     |

### 2. Analytics Layer (Statistical Tests, Models, Attribution)

Implements all analytical methods. Each module exposes pure functions that accept
DataFrames and return results as dicts or DataFrames.

| Module                | Responsibility                                      |
|-----------------------|-----------------------------------------------------|
| `core/tests.py`       | Hypothesis testing: t-test, Mann-Whitney, Chi-sq    |
| `core/models.py`      | Time series: STL, SARIMAX, ARX, anomaly detection   |
| `core/attribution.py` | Factor attribution: additive, multiplicative, SHAP  |
| `core/simulation.py`  | Monte Carlo, scenario what-if analysis              |
| `core/triggers.py`    | Alert rules: threshold, deviation, slope triggers   |

### 3. Visualization Layer (Plotly Charts)

All charts are built with Plotly and returned as `go.Figure` objects. Chart
functions live inside page modules and in `core/explore.py`.

### 4. UI Layer (Streamlit Pages)

Ten Streamlit pages provide the user interface. Each page imports from the Core
and Analytics layers, never from other pages.

### 5. Storage Layer (Session State, Audit Logs)

| Module            | Responsibility                                        |
|-------------------|-------------------------------------------------------|
| `core/audit.py`   | Append-only audit log of all user operations          |
| `core/i18n.py`    | Internationalization strings (Russian / English)      |
| `core/report.py`  | PDF/HTML report generation via Jinja2 + WeasyPrint    |

Session state is managed through `st.session_state` and acts as an in-memory
store for the current analysis session.

## Module Diagram

```
app/main.py                          (entry point)
  |
  +-- app/pages/
  |     1_Data.py        -----> core/data.py, core/validate.py
  |     2_Prepare.py     -----> core/prepare.py
  |     3_GroupAggregate.py ---> core/aggregate.py
  |     4_Explore.py     -----> core/explore.py
  |     5_Tests.py       -----> core/tests.py
  |     6_TimeSeries.py  -----> core/models.py
  |     7_Attribution.py -----> core/attribution.py
  |     8_Simulation.py  -----> core/simulation.py
  |     9_Report.py      -----> core/report.py
  |    10_Help.py        -----> core/i18n.py
  |
  +-- core/
  |     data.py           IO adapters (CSV, Excel, Parquet, SQL)
  |     validate.py       Schema validation, type checks
  |     prepare.py        Cleaning, parsing, imputation
  |     aggregate.py      GroupBy, pivot, rolling stats
  |     explore.py        EDA utilities, distribution plots
  |     tests.py          Statistical hypothesis tests
  |     models.py         Time series models, anomaly detection
  |     attribution.py    Factor decomposition
  |     simulation.py     Monte Carlo, what-if scenarios
  |     triggers.py       Alert / trigger rule engine
  |     audit.py          Operation audit trail
  |     i18n.py           Localization (RU/EN)
  |     report.py         PDF/HTML report builder
  |     __init__.py       Public API re-exports
  |
  +-- data/               Sample datasets (CSV)
  +-- tests/              pytest test suite
  +-- services/           Background / helper services
```

## Data Flow

The primary data flow follows a linear pipeline:

```
  Upload          Validate          Clean/Prepare
 (CSV/XLSX/  -->  (schema,    -->   (type coercion,
  Parquet/SQL)     types,            missing values,
                   constraints)      outlier handling)
       |                                  |
       v                                  v
   Analyze         <--  Iterate  <--  Transform
  (tests,                             (aggregate,
   models,                             pivot,
   attribution)                        filter)
       |
       v
    Export
  (PDF report,
   CSV download,
   audit log)
```

### Step-by-step

1. **Upload** -- User loads data via file upload or PostgreSQL connection
   (`1_Data.py` -> `core/data.py`).
2. **Validate** -- Automatic schema detection; user reviews inferred types and
   constraints (`core/validate.py`).
3. **Clean/Prepare** -- Type coercion, date parsing, missing value imputation,
   duplicate removal (`2_Prepare.py` -> `core/prepare.py`).
4. **Transform** -- Group-by aggregation, pivot tables, calculated columns
   (`3_GroupAggregate.py` -> `core/aggregate.py`).
5. **Analyze** -- EDA, hypothesis tests, time series models, factor attribution,
   simulation (`4_Explore` through `8_Simulation`).
6. **Export** -- Generate PDF/HTML report, download processed data, review audit
   log (`9_Report.py` -> `core/report.py`).

## Streamlit Pages

| #  | Page             | Purpose                                      |
|----|------------------|----------------------------------------------|
| 1  | Data             | File upload, PostgreSQL import, preview       |
| 2  | Prepare          | Type casting, cleaning, missing values        |
| 3  | GroupAggregate    | Group-by, pivot tables, rolling aggregations  |
| 4  | Explore          | EDA: distributions, correlations, scatter     |
| 5  | Tests            | Statistical hypothesis testing                |
| 6  | TimeSeries       | Decomposition, forecasting, anomaly detection |
| 7  | Attribution      | Factor attribution analysis                   |
| 8  | Simulation       | Monte Carlo, what-if scenarios                |
| 9  | Report           | PDF/HTML report generation                    |
| 10 | Help             | User guide, methodology reference             |

## Key Design Decisions

- **No cross-page imports**: pages only import from `core/`. This prevents
  circular dependencies and keeps pages independently testable.
- **Pure functions in core**: all core modules are stateless. State lives
  exclusively in `st.session_state`.
- **Plotly for all charts**: consistent interactive charting across all pages
  with export to PNG/SVG via kaleido.
- **Offline-first**: no external API calls, no telemetry. The application runs
  entirely on the local machine.

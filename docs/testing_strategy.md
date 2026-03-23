# KIBAD Testing Strategy

## Overview

KIBAD uses a multi-layered testing approach to ensure correctness of data
transformations, statistical computations, and UI workflows. The target is
150+ tests covering all core modules.

## Test Categories

### 1. Unit Tests

Each `core/` module has a corresponding test file. Unit tests verify individual
functions in isolation with known inputs and expected outputs.

| Core Module         | Test File                  | What Is Tested                            |
|---------------------|----------------------------|-------------------------------------------|
| `core/data.py`      | `tests/test_data.py`       | CSV/Excel/Parquet loading, SQL queries     |
| `core/validate.py`  | `tests/test_validate.py`   | Type inference, constraint checks          |
| `core/prepare.py`   | `tests/test_prepare.py`    | Date parsing, numeric coercion, cleaning   |
| `core/aggregate.py` | `tests/test_aggregate.py`  | GroupBy, pivot, rolling window results     |
| `core/explore.py`   | `tests/test_explore.py`    | Distribution stats, correlation matrices   |
| `core/tests.py`     | `tests/test_tests.py`      | t-test, Mann-Whitney, Chi-sq, bootstrap    |
| `core/models.py`    | `tests/test_models.py`     | STL decomposition, SARIMAX, anomaly detect |
| `core/attribution.py`| `tests/test_attribution.py`| Additive, multiplicative, Shapley values   |
| `core/simulation.py`| `tests/test_simulation.py` | Monte Carlo, scenario parameter sweeps     |
| `core/triggers.py`  | `tests/test_triggers.py`   | Threshold, deviation, slope triggers       |
| `core/audit.py`     | `tests/test_audit.py`      | Log append, log retrieval, log format      |
| `core/i18n.py`      | `tests/test_i18n.py`       | Translation keys, fallback behavior        |
| `core/report.py`    | `tests/test_report.py`     | HTML rendering, PDF generation             |

#### Key unit test areas

- **Date parsing**: ISO formats, DD.MM.YYYY, MM/DD/YYYY, ambiguous dates,
  invalid strings, timezone handling.
- **Numeric parsing**: comma decimals ("1.234,56"), currency symbols, percentage
  strings, scientific notation, empty strings, NaN handling.
- **Aggregation**: sum, mean, median, count, min, max on grouped data;
  multi-level grouping; null group keys.
- **Attribution**: verify that factor contributions sum to total delta (additive)
  or multiply to ratio (multiplicative).
- **Triggers**: threshold crossed, not crossed, edge cases at boundary values.
- **Validation**: missing required columns, wrong types, duplicate rows,
  out-of-range values.

### 2. Integration Tests

Integration tests verify the end-to-end pipeline using sample data files from
the `data/` directory.

```
Load CSV -> Validate schema -> Clean columns -> Aggregate -> Run tests -> Export
```

Key integration scenarios:

- **Multi-file merge**: load two CSVs, join on a key column, verify merged shape.
- **Full pipeline**: import raw data, apply type coercion, filter rows, group-by
  aggregate, run a t-test, generate a PDF report.
- **SQL round-trip**: load from CSV, write to in-memory SQLite, read back,
  compare DataFrames.
- **Report generation**: run analysis, generate HTML report, verify all sections
  are present.

### 3. Golden Tests

Golden tests compare current output against saved "golden" reference files to
detect regressions.

- Golden outputs are stored in `tests/golden/`.
- Each golden test loads a fixed dataset, runs an analysis, and compares the
  result to the saved reference (CSV or JSON).
- If a change is intentional, update the golden file with
  `pytest --update-golden`.

Example golden tests:

| Test                     | Dataset              | Output Verified                |
|--------------------------|----------------------|--------------------------------|
| STL decomposition        | monthly_sales.csv    | Trend + seasonal components    |
| Attribution (additive)   | multivariate.csv     | Factor contribution table      |
| Bootstrap CI             | ab_test.csv          | Confidence interval bounds     |

### 4. Diagnostic Warnings

Tests verify that the system produces appropriate warnings for edge cases:

| Condition                        | Expected Warning                           |
|----------------------------------|--------------------------------------------|
| Fewer than 5 observations        | "Too few observations for reliable test"   |
| Zero variance in a column        | "Column has zero variance"                 |
| Missing required column          | "Required column not found: {name}"        |
| NaN in metric column             | "Metric contains {n} missing values"       |
| Date column outside bounds       | "Dates outside expected range detected"    |
| Group with single observation    | "Group {name} has only 1 row"              |
| High multicollinearity (VIF > 10)| "Multicollinearity detected among factors" |

## Running Tests

### Full test suite

```bash
python -m pytest tests/ -v
```

### With coverage report

```bash
python -m pytest tests/ -v --cov=core --cov-report=term-missing
```

### Smoke test (quick sanity check)

```bash
python smoke_test.py
```

The smoke test loads sample data, runs each core module function, and verifies
no exceptions are raised. It completes in under 10 seconds.

### Run a specific test file

```bash
python -m pytest tests/test_tests.py -v
```

### Run tests matching a pattern

```bash
python -m pytest tests/ -v -k "attribution"
```

## Test Data

Sample datasets in `data/`:

| File                 | Rows  | Description                              |
|----------------------|-------|------------------------------------------|
| monthly_sales.csv    | ~120  | Monthly revenue with seasonality         |
| ab_test.csv          | ~1000 | A/B test with control and treatment      |
| multivariate.csv     | ~200  | Multi-factor dataset for attribution     |

Tests should never depend on external data sources. All test data is committed
to the repository.

## Coverage Target

| Module           | Target Coverage |
|------------------|-----------------|
| core/data.py     | 85%+            |
| core/validate.py | 90%+            |
| core/prepare.py  | 90%+            |
| core/aggregate.py| 85%+            |
| core/tests.py    | 90%+            |
| core/models.py   | 80%+            |
| core/attribution.py | 85%+        |
| core/simulation.py  | 80%+        |
| core/triggers.py | 90%+            |
| **Overall**      | **85%+**        |

## Continuous Testing Workflow

1. Write or modify code in `core/`.
2. Run the relevant test file: `pytest tests/test_<module>.py -v`.
3. Run the full suite before committing: `pytest tests/ -v`.
4. Check coverage does not drop: `pytest --cov=core`.
5. Run `smoke_test.py` as a final sanity check.

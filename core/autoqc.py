"""
core/autoqc.py – Automatic data quality checks and preprocessing recommendations.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any


def check_upload(df: pd.DataFrame) -> dict[str, Any]:
    """Run quick quality checks on a freshly uploaded DataFrame.

    Returns dict with keys:
    - n_rows, n_cols
    - duplicate_rows (int)
    - null_columns (list[str]) — columns with 100% nulls
    - missing_pct (dict[str, float]) — col → % missing
    - overall_missing_pct (float) — avg across all cols
    - high_missing_cols (list[str]) — cols with >30% missing
    - numeric_cols (list[str])
    - categorical_cols (list[str])
    - datetime_cols (list[str])
    - constant_cols (list[str]) — cols with single unique value
    - high_cardinality_cols (list[str]) — categoricals with >100 unique values
    - outlier_cols (dict[str, float]) — col → outlier rate via IQR
    - type_conflicts (list[dict]) — cols that look numeric but typed as object
    - severity: 'ok' | 'warning' | 'error'
    """
    result: dict[str, Any] = {}
    result["n_rows"] = len(df)
    result["n_cols"] = len(df.columns)

    # Duplicates
    result["duplicate_rows"] = int(df.duplicated().sum())

    # Missing values
    missing = df.isnull().mean()
    result["null_columns"] = [c for c in df.columns if missing[c] == 1.0]
    result["missing_pct"] = {c: round(float(missing[c]) * 100, 1) for c in df.columns if missing[c] > 0}
    result["overall_missing_pct"] = round(float(missing.mean()) * 100, 1)
    result["high_missing_cols"] = [c for c in df.columns if missing[c] > 0.3]

    # Column types
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dt_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
    result["numeric_cols"] = num_cols
    result["categorical_cols"] = cat_cols
    result["datetime_cols"] = dt_cols

    # Constant columns
    result["constant_cols"] = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    # High cardinality categoricals
    result["high_cardinality_cols"] = [
        c for c in cat_cols if df[c].nunique(dropna=True) > 100
    ]

    # Outliers via IQR on numeric cols
    outlier_cols: dict[str, float] = {}
    for col in num_cols[:20]:  # cap at 20 for performance
        s = df[col].dropna()
        if len(s) < 10:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        outlier_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
        rate = float(outlier_mask.mean())
        if rate > 0.01:
            outlier_cols[col] = round(rate * 100, 1)
    result["outlier_cols"] = outlier_cols

    # Type conflicts: object cols that look numeric
    type_conflicts = []
    for col in cat_cols:
        sample = df[col].dropna().head(100)
        if len(sample) == 0:
            continue
        try:
            pd.to_numeric(sample, errors="raise")
            type_conflicts.append({"column": col, "suggested_type": "numeric"})
        except (ValueError, TypeError):
            pass
    result["type_conflicts"] = type_conflicts

    # Severity
    has_error = (
        len(result["null_columns"]) > 0
        or result["overall_missing_pct"] > 50
        or result["duplicate_rows"] > len(df) * 0.5
    )
    has_warning = (
        len(result["high_missing_cols"]) > 0
        or result["duplicate_rows"] > 0
        or len(outlier_cols) > 0
        or len(type_conflicts) > 0
    )
    result["severity"] = "error" if has_error else ("warning" if has_warning else "ok")

    return result


def recommend_preprocessing(df: pd.DataFrame, qc: dict | None = None) -> list[dict]:
    """Generate ordered preprocessing recommendations based on data profile.

    Returns list of dicts: {action, column, reason, priority, auto_params}
    priority: 'high' | 'medium' | 'low'
    action: 'drop_nullcol' | 'impute' | 'drop_duplicates' | 'outlier_cap' |
            'type_cast' | 'drop_constant' | 'parse_dates'
    """
    if qc is None:
        qc = check_upload(df)

    recs: list[dict] = []

    # Drop null columns
    for col in qc.get("null_columns", []):
        recs.append({
            "action": "drop_nullcol",
            "column": col,
            "reason": f"Колонка «{col}» полностью пустая (100% пропусков).",
            "priority": "high",
            "auto_params": {"columns": [col]},
        })

    # Drop duplicates
    dups = qc.get("duplicate_rows", 0)
    if dups > 0:
        recs.append({
            "action": "drop_duplicates",
            "column": None,
            "reason": f"Обнаружено {dups:,} дублирующихся строк ({dups/len(df)*100:.1f}%).",
            "priority": "high" if dups > len(df) * 0.1 else "medium",
            "auto_params": {},
        })

    # Impute high-missing columns
    for col in qc.get("high_missing_cols", []):
        miss_pct = qc["missing_pct"].get(col, 0)
        is_num = col in qc.get("numeric_cols", [])
        method = "median" if is_num else "mode"
        recs.append({
            "action": "impute",
            "column": col,
            "reason": f"«{col}»: {miss_pct}% пропусков. Рекомендуемый метод: {method}.",
            "priority": "high" if miss_pct > 50 else "medium",
            "auto_params": {"column": col, "method": method},
        })

    # Impute moderate-missing columns
    for col, pct in qc.get("missing_pct", {}).items():
        if col in qc.get("null_columns", []) or col in qc.get("high_missing_cols", []):
            continue
        if pct > 5:
            is_num = col in qc.get("numeric_cols", [])
            method = "median" if is_num else "mode"
            recs.append({
                "action": "impute",
                "column": col,
                "reason": f"«{col}»: {pct}% пропусков. Рекомендуемый метод: {method}.",
                "priority": "low",
                "auto_params": {"column": col, "method": method},
            })

    # Outlier capping
    for col, rate in qc.get("outlier_cols", {}).items():
        recs.append({
            "action": "outlier_cap",
            "column": col,
            "reason": f"«{col}»: {rate}% выбросов (метод IQR). Рекомендуется: clip на 1.5×IQR.",
            "priority": "medium",
            "auto_params": {"column": col, "method": "iqr", "threshold": 1.5},
        })

    # Type conflicts
    for conflict in qc.get("type_conflicts", []):
        col = conflict["column"]
        recs.append({
            "action": "type_cast",
            "column": col,
            "reason": f"«{col}» хранится как текст, но содержит числовые значения.",
            "priority": "medium",
            "auto_params": {"column": col, "target_type": "numeric"},
        })

    # Drop constant columns
    for col in qc.get("constant_cols", []):
        recs.append({
            "action": "drop_constant",
            "column": col,
            "reason": f"«{col}» имеет только одно уникальное значение — не несёт информации.",
            "priority": "low",
            "auto_params": {"columns": [col]},
        })

    # Suggest date parsing if no datetime cols detected
    if not qc.get("datetime_cols") and not qc.get("numeric_cols") == []:
        cat_cols = qc.get("categorical_cols", [])
        date_candidates = [c for c in cat_cols if any(kw in c.lower() for kw in
                           ("date", "time", "dt", "period", "month", "year", "дата", "период", "месяц"))]
        for col in date_candidates[:2]:
            recs.append({
                "action": "parse_dates",
                "column": col,
                "reason": f"«{col}» похоже на дату. Парсинг откроет временные анализы.",
                "priority": "low",
                "auto_params": {"column": col},
            })

    # Sort by priority
    order = {"high": 0, "medium": 1, "low": 2}
    recs.sort(key=lambda r: order[r["priority"]])

    return recs

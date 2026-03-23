"""
core/validate.py – Data validation rules for KIBAD.

Provides composable checks that return structured ``ValidationResult`` objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


Severity = Literal["error", "warning", "info"]


@dataclass
class ValidationResult:
    """One validation check result.

    Attributes
    ----------
    rule : str
        Short name of the rule (e.g. ``"required_columns"``).
    severity : Severity
        ``"error"``, ``"warning"``, or ``"info"``.
    passed : bool
        Whether the check passed.
    message : str
        Human-readable description.
    details : dict
        Extra context (counts, sample values, etc.).
    """
    rule: str
    severity: Severity
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_required_columns(
    df: pd.DataFrame,
    required: list[str],
) -> ValidationResult:
    """Verify that all *required* columns are present in *df*.

    Returns
    -------
    ValidationResult
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        return ValidationResult(
            rule="required_columns",
            severity="error",
            passed=False,
            message=f"Missing required columns: {', '.join(missing)}",
            details={"missing": missing},
        )
    return ValidationResult(
        rule="required_columns",
        severity="info",
        passed=True,
        message="All required columns present.",
        details={"required": required},
    )


def check_no_nulls(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    threshold: float = 0.0,
) -> ValidationResult:
    """Check that null fraction in *columns* does not exceed *threshold*.

    Parameters
    ----------
    df:
        DataFrame to check.
    columns:
        Columns to inspect; defaults to all.
    threshold:
        Maximum allowed fraction of nulls (0.0 = no nulls, 0.05 = 5 %).

    Returns
    -------
    ValidationResult
    """
    cols = columns or df.columns.tolist()
    cols = [c for c in cols if c in df.columns]
    report: dict[str, float] = {}
    for c in cols:
        frac = float(df[c].isna().mean())
        if frac > threshold:
            report[c] = round(frac, 4)

    if report:
        return ValidationResult(
            rule="no_nulls",
            severity="warning",
            passed=False,
            message=f"Columns exceed null threshold ({threshold:.0%}): {', '.join(report.keys())}",
            details={"null_fractions": report, "threshold": threshold},
        )
    return ValidationResult(
        rule="no_nulls",
        severity="info",
        passed=True,
        message=f"All columns within null threshold ({threshold:.0%}).",
    )


def check_ranges(
    df: pd.DataFrame,
    col: str,
    min_val: float | None = None,
    max_val: float | None = None,
) -> ValidationResult:
    """Check that values in *col* fall within [min_val, max_val].

    Returns
    -------
    ValidationResult
    """
    if col not in df.columns:
        return ValidationResult(
            rule="range_check",
            severity="error",
            passed=False,
            message=f"Column '{col}' not found.",
        )

    s = pd.to_numeric(df[col], errors="coerce")
    violations = 0
    if min_val is not None:
        violations += int((s < min_val).sum())
    if max_val is not None:
        violations += int((s > max_val).sum())

    if violations > 0:
        return ValidationResult(
            rule="range_check",
            severity="warning",
            passed=False,
            message=f"Column '{col}': {violations} values outside [{min_val}, {max_val}].",
            details={"column": col, "violations": violations, "min": min_val, "max": max_val},
        )
    return ValidationResult(
        rule="range_check",
        severity="info",
        passed=True,
        message=f"Column '{col}' values within [{min_val}, {max_val}].",
    )


def check_uniqueness(
    df: pd.DataFrame,
    columns: list[str],
) -> ValidationResult:
    """Check that the combination of *columns* is unique across rows.

    Returns
    -------
    ValidationResult
    """
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return ValidationResult(
            rule="uniqueness",
            severity="error",
            passed=False,
            message="None of the specified columns exist.",
            details={"requested": columns},
        )

    dup_mask = df.duplicated(subset=existing, keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup > 0:
        sample = df[dup_mask].head(5)[existing].to_dict("records")
        return ValidationResult(
            rule="uniqueness",
            severity="warning",
            passed=False,
            message=f"{n_dup} rows violate uniqueness on columns {existing}.",
            details={"duplicate_count": n_dup, "sample": sample},
        )
    return ValidationResult(
        rule="uniqueness",
        severity="info",
        passed=True,
        message=f"All rows are unique on columns {existing}.",
    )


def check_zero_variance(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> ValidationResult:
    """Flag numeric columns with zero variance.

    Returns
    -------
    ValidationResult
    """
    num_df = df.select_dtypes(include="number")
    if columns:
        num_df = num_df[[c for c in columns if c in num_df.columns]]

    zero_var = [c for c in num_df.columns if num_df[c].std() == 0]
    if zero_var:
        return ValidationResult(
            rule="zero_variance",
            severity="warning",
            passed=False,
            message=f"Zero-variance columns: {', '.join(zero_var)}",
            details={"columns": zero_var},
        )
    return ValidationResult(
        rule="zero_variance",
        severity="info",
        passed=True,
        message="No zero-variance numeric columns found.",
    )


def check_time_gaps(
    df: pd.DataFrame,
    date_col: str,
    expected_freq: str = "MS",
) -> ValidationResult:
    """Detect missing periods in a time series column.

    Parameters
    ----------
    df:
        DataFrame (must be sorted by *date_col*).
    date_col:
        Datetime column.
    expected_freq:
        Expected pandas frequency alias (``"MS"``, ``"D"``, ``"W-MON"``, etc.).

    Returns
    -------
    ValidationResult
    """
    if date_col not in df.columns:
        return ValidationResult(
            rule="time_gaps",
            severity="error",
            passed=False,
            message=f"Column '{date_col}' not found.",
        )

    dates = pd.to_datetime(df[date_col], errors="coerce").dropna().sort_values()
    if len(dates) < 2:
        return ValidationResult(
            rule="time_gaps",
            severity="info",
            passed=True,
            message="Not enough dates to check gaps.",
        )

    full_range = pd.date_range(dates.iloc[0], dates.iloc[-1], freq=expected_freq)
    missing = full_range.difference(dates)
    n_missing = len(missing)

    if n_missing > 0:
        return ValidationResult(
            rule="time_gaps",
            severity="warning",
            passed=False,
            message=f"{n_missing} missing period(s) detected in '{date_col}'.",
            details={"missing_count": n_missing, "sample_missing": [str(d) for d in missing[:5]]},
        )
    return ValidationResult(
        rule="time_gaps",
        severity="info",
        passed=True,
        message="No time gaps detected.",
    )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_all_checks(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
    range_rules: list[dict[str, Any]] | None = None,
    uniqueness_columns: list[str] | None = None,
    date_col: str | None = None,
    expected_freq: str = "MS",
) -> list[ValidationResult]:
    """Run a standard set of validation checks.

    Parameters
    ----------
    df:
        The DataFrame to validate.
    required_columns:
        Columns that must be present.
    range_rules:
        List of ``{"col": ..., "min": ..., "max": ...}`` dicts.
    uniqueness_columns:
        Columns whose combined values should be unique.
    date_col:
        If given, check for time gaps.
    expected_freq:
        Expected frequency for time gap check.

    Returns
    -------
    list[ValidationResult]
    """
    results: list[ValidationResult] = []

    if required_columns:
        results.append(check_required_columns(df, required_columns))

    results.append(check_no_nulls(df))
    results.append(check_zero_variance(df))

    if range_rules:
        for rule in range_rules:
            results.append(check_ranges(
                df, rule["col"],
                min_val=rule.get("min"),
                max_val=rule.get("max"),
            ))

    if uniqueness_columns:
        results.append(check_uniqueness(df, uniqueness_columns))

    if date_col:
        results.append(check_time_gaps(df, date_col, expected_freq))

    return results

"""
core/merge.py – Table merge, join, and concatenation with pitfall detection.

Pitfalls detected:
- Row explosion (many-to-many join produces more rows than either input)
- Collinearity risk (duplicate non-key columns after join)
- Key nulls (NaN values in join keys)
- Key type mismatch (left key dtype != right key dtype)
- Duplicate rows after join
- Low join match rate (many rows lost in inner join)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

JoinHow = Literal["inner", "left", "right", "outer", "cross"]


@dataclass
class MergeWarning:
    level: Literal["error", "warning", "info"]
    code: str
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class MergeResult:
    df: pd.DataFrame
    warnings: list[MergeWarning]
    left_rows: int
    right_rows: int
    result_rows: int
    left_cols: int
    right_cols: int
    result_cols: int
    match_rate: float  # fraction of left rows matched (inner join reference)

    @property
    def explosion_ratio(self) -> float:
        """How many times bigger is result vs max(left, right)."""
        denom = max(self.left_rows, self.right_rows)
        return self.result_rows / denom if denom > 0 else 0.0

    @property
    def has_errors(self) -> bool:
        return any(w.level == "error" for w in self.warnings)


# ---------------------------------------------------------------------------
# Key validation helpers
# ---------------------------------------------------------------------------

def _check_key_nulls(df: pd.DataFrame, keys: list[str], side: str) -> list[MergeWarning]:
    warnings = []
    for k in keys:
        if k not in df.columns:
            continue
        null_count = df[k].isna().sum()
        if null_count > 0:
            warnings.append(MergeWarning(
                level="warning",
                code="KEY_NULLS",
                message=f"Key '{k}' in {side} table has {null_count} null value(s). "
                        "Null keys will not match and rows may be dropped.",
                details={"key": k, "side": side, "null_count": int(null_count)},
            ))
    return warnings


def _check_key_type_mismatch(
    left: pd.DataFrame, right: pd.DataFrame,
    left_keys: list[str], right_keys: list[str],
) -> list[MergeWarning]:
    warnings = []
    for lk, rk in zip(left_keys, right_keys):
        if lk not in left.columns or rk not in right.columns:
            continue
        lt = str(left[lk].dtype)
        rt = str(right[rk].dtype)
        if lt != rt:
            warnings.append(MergeWarning(
                level="warning",
                code="KEY_TYPE_MISMATCH",
                message=f"Key dtype mismatch: left '{lk}' is {lt}, right '{rk}' is {rt}. "
                        "Join may silently drop rows or fail.",
                details={"left_key": lk, "left_dtype": lt, "right_key": rk, "right_dtype": rt},
            ))
    return warnings


def _check_row_explosion(
    left_rows: int, right_rows: int, result_rows: int, how: str,
) -> list[MergeWarning]:
    warnings = []
    expected_max = max(left_rows, right_rows)
    if result_rows > expected_max * 2:
        ratio = result_rows / expected_max
        warnings.append(MergeWarning(
            level="warning" if ratio < 10 else "error",
            code="ROW_EXPLOSION",
            message=f"Row explosion detected: result has {result_rows:,} rows "
                    f"({ratio:.1f}× more than the larger input with {expected_max:,} rows). "
                    "This happens with many-to-many joins — each left key matches multiple "
                    "right rows and vice versa. Check for duplicate keys.",
            details={
                "left_rows": left_rows, "right_rows": right_rows,
                "result_rows": result_rows, "explosion_ratio": round(ratio, 2),
            },
        ))
    return warnings


def _check_collinearity(
    left: pd.DataFrame, right: pd.DataFrame, left_keys: list[str], right_keys: list[str],
) -> list[MergeWarning]:
    warnings = []
    left_non_keys = set(left.columns) - set(left_keys)
    right_non_keys = set(right.columns) - set(right_keys)
    duplicates = left_non_keys & right_non_keys
    if duplicates:
        warnings.append(MergeWarning(
            level="warning",
            code="DUPLICATE_COLUMNS",
            message=f"Columns {sorted(duplicates)} exist in both tables (non-key). "
                    "Pandas will suffix them (_x, _y). These may be collinear — "
                    "consider dropping one before joining.",
            details={"duplicate_cols": sorted(duplicates)},
        ))
    return warnings


def _check_duplicates_in_result(result: pd.DataFrame, keys: list[str]) -> list[MergeWarning]:
    warnings = []
    if keys and all(k in result.columns for k in keys):
        dup_count = result.duplicated(subset=keys).sum()
        if dup_count > 0:
            warnings.append(MergeWarning(
                level="info",
                code="DUPLICATE_KEY_ROWS",
                message=f"{dup_count:,} rows share duplicate key values in the result. "
                        "This is expected for one-to-many joins but may indicate data issues.",
                details={"duplicate_key_rows": int(dup_count)},
            ))
    return warnings


def _check_match_rate(
    left_rows: int, right_rows: int, result_rows: int, how: str,
) -> tuple[float, list[MergeWarning]]:
    """Estimate match rate for inner join semantics."""
    warnings = []
    if left_rows == 0:
        return 0.0, warnings
    if how == "inner":
        match_rate = result_rows / left_rows
    elif how == "left":
        # Count non-null right-side rows (approx: rows that matched)
        match_rate = min(result_rows / left_rows, 1.0)
    else:
        match_rate = 1.0  # outer/right — harder to estimate

    if how == "inner" and match_rate < 0.5:
        warnings.append(MergeWarning(
            level="warning",
            code="LOW_MATCH_RATE",
            message=f"Only {match_rate:.0%} of left rows matched in inner join "
                    f"({result_rows:,} of {left_rows:,} rows kept). "
                    "Consider using left join to preserve all left rows.",
            details={"match_rate": round(match_rate, 4), "left_rows": left_rows, "result_rows": result_rows},
        ))
    return match_rate, warnings


# ---------------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------------

def merge_tables(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
    how: JoinHow = "left",
    suffixes: tuple[str, str] = ("_x", "_y"),
) -> MergeResult:
    """
    Merge two DataFrames with pitfall detection.

    Parameters
    ----------
    left, right : DataFrames to join
    left_keys, right_keys : key columns to join on (parallel lists)
    how : join type (inner / left / right / outer / cross)
    suffixes : column suffixes for overlapping non-key columns

    Returns
    -------
    MergeResult with joined DataFrame and list of warnings.
    """
    warnings: list[MergeWarning] = []

    # Pre-merge checks
    if how != "cross":
        warnings += _check_key_nulls(left, left_keys, "left")
        warnings += _check_key_nulls(right, right_keys, "right")
        warnings += _check_key_type_mismatch(left, right, left_keys, right_keys)
        warnings += _check_collinearity(left, right, left_keys, right_keys)

    left_rows, right_rows = len(left), len(right)
    left_cols, right_cols = left.shape[1], right.shape[1]

    # Perform merge
    try:
        if how == "cross":
            result = left.merge(right, how="cross", suffixes=suffixes)
        else:
            result = left.merge(
                right,
                left_on=left_keys, right_on=right_keys,
                how=how,
                suffixes=suffixes,
            )
    except ValueError as exc:
        # Type mismatch that pandas cannot coerce → add error warning, return empty result
        warnings.append(MergeWarning(
            level="error",
            code="MERGE_FAILED",
            message=f"Объединение не удалось: {exc}. Проверьте типы ключевых колонок.",
            details={"error": str(exc)},
        ))
        empty = pd.DataFrame(columns=list(left.columns) + list(right.columns))
        return MergeResult(
            df=empty, warnings=warnings,
            left_rows=left_rows, right_rows=right_rows,
            result_rows=0, left_cols=left_cols, right_cols=right_cols,
            result_cols=0, match_rate=0.0,
        )

    result_rows = len(result)

    # Post-merge checks
    if how != "cross":
        warnings += _check_row_explosion(left_rows, right_rows, result_rows, how)
        warnings += _check_duplicates_in_result(result, left_keys)

    match_rate, rate_warnings = _check_match_rate(left_rows, right_rows, result_rows, how)
    warnings += rate_warnings

    return MergeResult(
        df=result,
        warnings=warnings,
        left_rows=left_rows,
        right_rows=right_rows,
        result_rows=result_rows,
        left_cols=left_cols,
        right_cols=right_cols,
        result_cols=result.shape[1],
        match_rate=match_rate,
    )


# ---------------------------------------------------------------------------
# Concat function
# ---------------------------------------------------------------------------

def concat_tables(
    dfs: list[pd.DataFrame],
    axis: Literal[0, 1] = 0,
    ignore_index: bool = True,
) -> tuple[pd.DataFrame, list[MergeWarning]]:
    """
    Concatenate a list of DataFrames with schema warnings.

    Returns (result_df, warnings).
    """
    warnings: list[MergeWarning] = []

    if not dfs:
        return pd.DataFrame(), []

    if axis == 0:
        # Check schema consistency
        first_cols = set(dfs[0].columns)
        for i, df in enumerate(dfs[1:], start=2):
            missing = first_cols - set(df.columns)
            extra = set(df.columns) - first_cols
            if missing:
                warnings.append(MergeWarning(
                    level="warning",
                    code="SCHEMA_MISMATCH_MISSING",
                    message=f"Table {i} is missing columns {sorted(missing)} from table 1. "
                            "Missing values will be filled with NaN.",
                    details={"table_index": i, "missing_cols": sorted(missing)},
                ))
            if extra:
                warnings.append(MergeWarning(
                    level="info",
                    code="SCHEMA_MISMATCH_EXTRA",
                    message=f"Table {i} has extra columns {sorted(extra)} not in table 1. "
                            "They will appear as NaN for all other tables.",
                    details={"table_index": i, "extra_cols": sorted(extra)},
                ))

        # Check dtype consistency per column
        for col in first_cols:
            dtypes = set()
            for df in dfs:
                if col in df.columns:
                    dtypes.add(str(df[col].dtype))
            if len(dtypes) > 1:
                warnings.append(MergeWarning(
                    level="info",
                    code="DTYPE_MISMATCH",
                    message=f"Column '{col}' has mixed dtypes across tables: {dtypes}. "
                            "Pandas will upcast automatically.",
                    details={"col": col, "dtypes": sorted(dtypes)},
                ))

        total_rows = sum(len(d) for d in dfs)
        result = pd.concat(dfs, axis=0, ignore_index=ignore_index)
        dup_count = result.duplicated().sum()
        if dup_count > 0:
            warnings.append(MergeWarning(
                level="info",
                code="DUPLICATE_ROWS",
                message=f"{dup_count:,} exact duplicate rows found after concat. "
                        "Consider deduplicating.",
                details={"duplicate_rows": int(dup_count), "total_rows": total_rows},
            ))

    else:  # axis=1 (column concat)
        all_cols: list[str] = []
        for df in dfs:
            all_cols.extend(df.columns.tolist())
        dup_cols = [c for c in set(all_cols) if all_cols.count(c) > 1]
        if dup_cols:
            warnings.append(MergeWarning(
                level="error",
                code="DUPLICATE_COLUMN_NAMES",
                message=f"Duplicate column names after column-wise concat: {dup_cols}. "
                        "This will cause ambiguous column access.",
                details={"duplicate_cols": dup_cols},
            ))
        # Check row count consistency
        row_counts = [len(d) for d in dfs]
        if len(set(row_counts)) > 1:
            warnings.append(MergeWarning(
                level="warning",
                code="ROW_COUNT_MISMATCH",
                message=f"Tables have different row counts: {row_counts}. "
                        "Pandas will align by index — unmatched rows become NaN.",
                details={"row_counts": row_counts},
            ))
        result = pd.concat(dfs, axis=1, ignore_index=ignore_index)

    return result, warnings


# ---------------------------------------------------------------------------
# Utility: key cardinality analysis
# ---------------------------------------------------------------------------

def analyze_key_cardinality(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_keys: list[str],
    right_keys: list[str],
) -> dict:
    """
    Analyse key cardinality to predict join type (1:1, 1:N, N:1, N:M).
    Returns dict with cardinality info and join type prediction.
    """
    if not left_keys or not all(k in left.columns for k in left_keys):
        return {"join_type": "unknown", "reason": "Invalid left keys"}
    if not right_keys or not all(k in right.columns for k in right_keys):
        return {"join_type": "unknown", "reason": "Invalid right keys"}

    left_unique = left[left_keys].drop_duplicates()
    right_unique = right[right_keys].drop_duplicates()

    left_is_unique = len(left_unique) == len(left)
    right_is_unique = len(right_unique) == len(right)

    if left_is_unique and right_is_unique:
        join_type = "1:1"
        description = "One-to-one join. Each left key matches at most one right row. Safe."
    elif left_is_unique and not right_is_unique:
        join_type = "1:N"
        description = "One-to-many join. Each left row may match multiple right rows. Result will have more rows than left."
    elif not left_is_unique and right_is_unique:
        join_type = "N:1"
        description = "Many-to-one join. Multiple left rows match one right row. Standard lookup join."
    else:
        join_type = "N:M"
        description = "Many-to-many join. Row explosion risk! Each non-unique left key will match multiple right rows."

    return {
        "join_type": join_type,
        "description": description,
        "left_unique_keys": len(left_unique),
        "left_total_rows": len(left),
        "right_unique_keys": len(right_unique),
        "right_total_rows": len(right),
        "left_is_unique": left_is_unique,
        "right_is_unique": right_is_unique,
    }

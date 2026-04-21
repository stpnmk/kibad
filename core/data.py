"""
core/data.py – Dataset loading, profiling, and type inference.
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sniffing — encoding + dialect auto-detection
# ---------------------------------------------------------------------------

_ENCODING_FALLBACKS: list[str] = [
    "utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1", "cp866", "koi8-r",
]
_CANDIDATE_SEPS: list[str] = [",", ";", "\t", "|"]


def _read_bytes(source: Any) -> bytes:
    """Read up to ~256 KB from *source* for sniffing purposes.

    ``source`` may be a path, bytes, or a file-like object. File-like
    objects are rewound to the start after reading.
    """
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[: 256 * 1024])
    if isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            return f.read(256 * 1024)
    # file-like
    pos = None
    try:
        pos = source.tell()
    except Exception:
        pass
    data = source.read(256 * 1024)
    if pos is not None:
        try:
            source.seek(pos)
        except Exception:
            pass
    return data if isinstance(data, (bytes, bytearray)) else data.encode("utf-8")


def _detect_encoding(sample: bytes) -> str:
    """Pick a plausible encoding for *sample* bytes.

    Tries ``charset_normalizer`` first; falls back to probing a short list
    of common encodings. Always returns *something* decodable.
    """
    if not sample:
        return "utf-8"
    # BOM shortcuts
    if sample.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if sample.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16"
    try:
        from charset_normalizer import from_bytes  # lazy import
        best = from_bytes(sample).best()
        if best is not None and best.encoding:
            enc = best.encoding.lower().replace("_", "-")
            # normalise common aliases
            if enc in ("windows-1251", "cp1251"):
                return "cp1251"
            if enc in ("windows-1252", "cp1252"):
                return "cp1252"
            return enc
    except Exception:
        pass
    for enc in _ENCODING_FALLBACKS:
        try:
            sample.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"  # last resort — pd.read_csv will use errors="replace"


def _detect_delimiter(text_sample: str) -> str:
    """Pick the most likely CSV delimiter from a decoded *text_sample*.

    Uses ``csv.Sniffer`` when possible; otherwise falls back to counting
    candidate characters across the first ~30 non-blank lines.
    """
    if not text_sample:
        return ","
    # Try stdlib sniffer first
    try:
        dialect = csv.Sniffer().sniff(text_sample[:8192], delimiters=",;\t|")
        if dialect.delimiter in _CANDIDATE_SEPS:
            return dialect.delimiter
    except Exception:
        pass
    # Fallback — pick the candidate with the most consistent per-line count
    lines = [ln for ln in text_sample.splitlines()[:30] if ln.strip()]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for sep in _CANDIDATE_SEPS:
        counts = [ln.count(sep) for ln in lines]
        avg = sum(counts) / len(counts)
        if avg < 1:
            continue
        # reward consistency (low variance) + high count
        var = sum((c - avg) ** 2 for c in counts) / len(counts)
        score = avg - var
        if score > best_score:
            best_score, best = score, sep
    return best


def _detect_decimal(text_sample: str, sep: str) -> str:
    """Return ``","`` if numbers look like ``1,23`` under *sep*, else ``"."``.

    If the delimiter is already ``","`` we can't use ``","`` as decimal, so
    we short-circuit to ``"."``.
    """
    if sep == ",":
        return "."
    # Look at a handful of non-header lines
    lines = [ln for ln in text_sample.splitlines()[1:50] if ln.strip()]
    if not lines:
        return "."
    comma_dec = dot_dec = 0
    # Match *entire* token — "1,23" or "1.23" or "-1234,5" — but NOT
    # "05.01.2024" (two separators) or "2024-01-05" (dash).
    pat = re.compile(r"^-?\d+[,.]\d+$")
    for ln in lines:
        for raw in ln.split(sep):
            tok = raw.strip().strip('"').strip("'")
            if not pat.fullmatch(tok):
                continue
            if "," in tok:
                comma_dec += 1
            else:
                dot_dec += 1
    return "," if comma_dec > dot_dec else "."


def sniff_csv(source: Any) -> dict[str, Any]:
    """Detect CSV dialect parameters for *source*.

    Parameters
    ----------
    source:
        Path, raw bytes, or file-like — same shapes accepted by
        :func:`load_csv`.

    Returns
    -------
    dict
        With keys ``encoding``, ``sep``, ``decimal`` (always present).
    """
    sample = _read_bytes(source)
    encoding = _detect_encoding(sample)
    try:
        text = sample.decode(encoding, errors="replace")
    except LookupError:
        text = sample.decode("utf-8", errors="replace")
        encoding = "utf-8"
    sep = _detect_delimiter(text)
    decimal = _detect_decimal(text, sep)
    return {"encoding": encoding, "sep": sep, "decimal": decimal}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(
    source: Any,
    *,
    sep: str | None = None,
    encoding: str | None = None,
    decimal: str | None = None,
    _sniff_result: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Load CSV from a file path, file-like object, or bytes.

    When *sep* / *encoding* / *decimal* are ``None`` they are auto-detected
    via :func:`sniff_csv`. Pass explicit values (or any truthy string) to
    disable sniffing for that parameter.

    Parameters
    ----------
    source:
        Path string, pathlib.Path, BytesIO / UploadedFile, or raw bytes.
    sep, encoding, decimal:
        Explicit CSV dialect params. ``None`` enables auto-detect.
    _sniff_result:
        Internal — pre-computed sniff dict, to avoid sniffing twice when
        :func:`load_file` already did it.
    **kwargs:
        Extra kwargs forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
    """
    need_sniff = sep is None or encoding is None or decimal is None
    if need_sniff:
        sniffed = _sniff_result or sniff_csv(source)
        if sep is None:
            sep = sniffed["sep"]
        if encoding is None:
            encoding = sniffed["encoding"]
        if decimal is None:
            decimal = sniffed["decimal"]

    kwargs.setdefault("sep", sep)
    kwargs.setdefault("encoding", encoding)
    kwargs.setdefault("decimal", decimal)
    # Robustness — ignore bad lines rather than failing the whole upload
    kwargs.setdefault("on_bad_lines", "skip")

    if isinstance(source, (str, Path)):
        return pd.read_csv(source, **kwargs)
    if isinstance(source, (bytes, bytearray)):
        return pd.read_csv(io.BytesIO(source), **kwargs)
    # file-like (Streamlit UploadedFile, BytesIO, etc.)
    return pd.read_csv(source, **kwargs)


def load_excel(source: Any, sheet_name: int | str = 0, **kwargs) -> pd.DataFrame:
    """Load an Excel file (xls/xlsx).

    Parameters
    ----------
    source:
        Path string, pathlib.Path, BytesIO / UploadedFile, or raw bytes.
    sheet_name:
        Sheet index or name.
    **kwargs:
        Extra kwargs forwarded to ``pd.read_excel``.

    Returns
    -------
    pd.DataFrame
    """
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    return pd.read_excel(source, sheet_name=sheet_name, **kwargs)


def load_file(
    source: Any,
    filename: str = "",
    *,
    return_meta: bool = False,
    **kwargs,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """Auto-detect CSV vs Excel by filename extension and load.

    Parameters
    ----------
    source:
        Raw bytes, file-like object, or path.
    filename:
        Original filename used to detect extension.
    return_meta:
        If True, return ``(df, meta)`` where ``meta`` describes the
        effective loader parameters (``kind``, and for CSV also
        ``encoding``, ``sep``, ``decimal``). Useful for surfacing
        auto-detected values back to the UI.
    **kwargs:
        Extra kwargs forwarded to the underlying loader. Pass
        ``sep``/``encoding``/``decimal`` explicitly to override
        auto-detection.

    Returns
    -------
    pd.DataFrame, or (pd.DataFrame, dict) if ``return_meta`` is True.
    """
    name = filename.lower() if filename else ""
    if name.endswith((".xls", ".xlsx")):
        df = load_excel(source, **kwargs)
        return (df, {"kind": "excel"}) if return_meta else df

    # CSV path — sniff upfront so we can report what was used
    sniffed = sniff_csv(source)
    df = load_csv(source, _sniff_result=sniffed, **kwargs)
    if not return_meta:
        return df
    meta = {
        "kind": "csv",
        "encoding": kwargs.get("encoding") or sniffed["encoding"],
        "sep":      kwargs.get("sep")      or sniffed["sep"],
        "decimal":  kwargs.get("decimal")  or sniffed["decimal"],
    }
    return df, meta


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    r"\d{4}-\d{2}-\d{2}",          # 2023-01-15
    r"\d{2}/\d{2}/\d{4}",          # 15/01/2023
    r"\d{2}\.\d{2}\.\d{4}",        # 15.01.2023
    r"\d{4}/\d{2}/\d{2}",          # 2023/01/15
    r"\d{1,2} \w+ \d{4}",          # 5 Jan 2023
]
_DATE_RE = re.compile("|".join(_DATE_PATTERNS))


def infer_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Infer logical types for each column.

    Returns a dict mapping column name → inferred type:
    ``"datetime"``, ``"numeric"``, ``"boolean"``, or ``"categorical"``.

    Parameters
    ----------
    df:
        The DataFrame to inspect.

    Returns
    -------
    dict[str, str]
    """
    result: dict[str, str] = {}
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            result[col] = "categorical"
            continue

        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            result[col] = "datetime"
            continue

        # Numeric
        if pd.api.types.is_numeric_dtype(series):
            result[col] = "numeric"
            continue

        # Boolean-like
        unique_lower = set(str(v).strip().lower() for v in series.unique())
        if unique_lower <= {"true", "false", "yes", "no", "0", "1", "t", "f"}:
            result[col] = "boolean"
            continue

        # Try to parse as datetime (sample first 200 non-null values)
        sample = series.astype(str).head(200)
        matched = sample.str.contains(_DATE_RE, na=False).sum()
        if matched >= min(5, len(sample) * 0.6):
            # Attempt actual parsing
            try:
                pd.to_datetime(sample, errors="raise")
                result[col] = "datetime"
                continue
            except Exception:
                pass

        result[col] = "categorical"
    return result


def cast_column(series: pd.Series, target_type: str) -> pd.Series:
    """Cast a column to the target logical type.

    Parameters
    ----------
    series:
        The column to cast.
    target_type:
        One of ``"datetime"``, ``"numeric"``, ``"boolean"``, ``"categorical"``.

    Returns
    -------
    pd.Series
    """
    if target_type == "datetime":
        return pd.to_datetime(series, errors="coerce", utc=False)
    if target_type == "numeric":
        return pd.to_numeric(series, errors="coerce")
    if target_type == "boolean":
        bool_map = {
            "true": True, "false": False, "yes": True, "no": False,
            "1": True, "0": False, "t": True, "f": False,
        }
        return series.astype(str).str.strip().str.lower().map(bool_map)
    # categorical / string
    return series.astype(str)


def apply_type_overrides(
    df: pd.DataFrame,
    overrides: dict[str, str],
) -> pd.DataFrame:
    """Apply user-specified type overrides to a DataFrame.

    Parameters
    ----------
    df:
        The DataFrame to modify.
    overrides:
        Mapping of column name → desired type string.

    Returns
    -------
    pd.DataFrame
        A copy with casts applied.
    """
    df = df.copy()
    for col, ttype in overrides.items():
        if col in df.columns:
            df[col] = cast_column(df[col], ttype)
    return df


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a per-column profile summary.

    Returns a DataFrame with columns:
    ``dtype, inferred_type, n_missing, pct_missing, n_unique, sample_values``.

    Parameters
    ----------
    df:
        The DataFrame to profile.

    Returns
    -------
    pd.DataFrame
    """
    inferred = infer_column_types(df)
    rows = []
    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = round(n_missing / len(series) * 100, 1) if len(series) else 0.0
        n_unique = int(series.nunique(dropna=True))
        sample = series.dropna().unique()[:5]
        sample_str = ", ".join(str(v) for v in sample)
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "inferred_type": inferred.get(col, "categorical"),
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": n_unique,
            "sample_values": sample_str,
        })
    return pd.DataFrame(rows)


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Extended numeric summary (adds IQR and skewness).

    Parameters
    ----------
    df:
        The DataFrame to describe.

    Returns
    -------
    pd.DataFrame
    """
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.DataFrame()
    desc = df[numeric_cols].describe().T
    desc["iqr"] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
    desc["skewness"] = df[numeric_cols].skew()
    return desc.round(4)


# ---------------------------------------------------------------------------
# Dataset catalog helpers
# ---------------------------------------------------------------------------

class DatasetCatalog:
    """In-memory catalog of named DataFrames with metadata.

    Attributes
    ----------
    _store : dict[str, dict]
        Internal mapping of name → {"df": pd.DataFrame, "source": str, ...}
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    def add(
        self,
        name: str,
        df: pd.DataFrame,
        source: str = "upload",
        notes: str = "",
    ) -> None:
        """Add or replace a dataset in the catalog.

        Parameters
        ----------
        name:
            Logical name for the dataset.
        df:
            The DataFrame to store.
        source:
            Where the data came from (``"upload"``, ``"postgres"``, etc.).
        notes:
            Free-text notes.
        """
        self._store[name] = {
            "df": df,
            "source": source,
            "notes": notes,
            "shape": df.shape,
            "columns": list(df.columns),
        }

    def get(self, name: str) -> pd.DataFrame:
        """Retrieve a DataFrame by name.

        Parameters
        ----------
        name:
            Dataset name.

        Returns
        -------
        pd.DataFrame

        Raises
        ------
        KeyError
            If the name is not in the catalog.
        """
        return self._store[name]["df"]

    def list_datasets(self) -> list[dict[str, Any]]:
        """Return a list of metadata dicts for all stored datasets.

        Returns
        -------
        list[dict]
        """
        return [
            {
                "name": k,
                "rows": v["shape"][0],
                "cols": v["shape"][1],
                "source": v["source"],
                "notes": v["notes"],
            }
            for k, v in self._store.items()
        ]

    def remove(self, name: str) -> None:
        """Remove a dataset from the catalog.

        Parameters
        ----------
        name:
            Dataset name to remove.
        """
        self._store.pop(name, None)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __len__(self) -> int:
        return len(self._store)

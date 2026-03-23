"""
core/insights.py – Auto-insights engine.

Automatically surfaces key findings from any DataFrame:
correlations, distributions, trends, category concentration,
anomalies, and actionable recommendations — all in Russian.
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def _categorical_cols(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def _datetime_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _safe_skew(s: pd.Series) -> float:
    try:
        from scipy.stats import skew
        vals = pd.to_numeric(s, errors="coerce").dropna().values
        if len(vals) < 3:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(skew(vals))
    except Exception:
        return 0.0


def _safe_kurtosis(s: pd.Series) -> float:
    try:
        from scipy.stats import kurtosis
        vals = pd.to_numeric(s, errors="coerce").dropna().values
        if len(vals) < 4:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(kurtosis(vals))
    except Exception:
        return 0.0


def _outlier_pct_iqr(s: pd.Series) -> float:
    """Fraction of values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR], in percent."""
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 4:
        return 0.0
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0.0
    mask = (vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)
    return round(float(mask.mean()) * 100, 1)


def _outlier_pct_3sigma(s: pd.Series) -> float:
    """Fraction of values beyond 3 standard deviations from mean, in percent."""
    vals = pd.to_numeric(s, errors="coerce").dropna()
    if len(vals) < 4:
        return 0.0
    mu, sigma = vals.mean(), vals.std()
    if sigma == 0:
        return 0.0
    mask = (vals - mu).abs() > 3 * sigma
    return round(float(mask.mean()) * 100, 1)


def _corr_strength_label(r: float) -> str:
    abs_r = abs(r)
    if abs_r >= 0.8:
        return "очень сильная"
    if abs_r >= 0.6:
        return "сильная"
    if abs_r >= 0.4:
        return "умеренная"
    if abs_r >= 0.2:
        return "слабая"
    return "очень слабая"


def _corr_direction_label(r: float) -> str:
    return "положительная" if r >= 0 else "отрицательная"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_dataset(df: pd.DataFrame) -> dict[str, Any]:
    """Automatically analyze a DataFrame and return structured insights.

    Parameters
    ----------
    df:
        Any pandas DataFrame.

    Returns
    -------
    dict with keys: summary, correlations, distributions, trends,
    top_values, anomalies, recommendations.
    """
    if df is None or df.empty:
        return {
            "summary": {},
            "correlations": [],
            "distributions": [],
            "trends": [],
            "top_values": {},
            "anomalies": [],
            "recommendations": [],
        }

    num_cols = _numeric_cols(df)
    cat_cols = _categorical_cols(df)
    dt_cols = _datetime_cols(df)

    n_rows, n_cols = df.shape
    missing_pct = round(float(df.isnull().mean().mean()) * 100, 1)
    n_dup = int(df.duplicated().sum())
    duplicate_pct = round(n_dup / n_rows * 100, 1) if n_rows > 0 else 0.0

    # ---- Summary ----
    summary = {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "n_numeric": len(num_cols),
        "n_categorical": len(cat_cols),
        "n_datetime": len(dt_cols),
        "missing_pct": missing_pct,
        "duplicate_pct": duplicate_pct,
        "n_duplicates": n_dup,
    }

    # ---- Correlations ----
    correlations: list[dict] = []
    if len(num_cols) >= 2:
        # sample for speed on very large datasets
        sample_df = df[num_cols].dropna(how="all")
        if len(sample_df) > 50_000:
            sample_df = sample_df.sample(50_000, random_state=42)
        try:
            corr_matrix = sample_df.corr()
            # upper triangle, excluding diagonal
            pairs: list[tuple[str, str, float]] = []
            cols_list = corr_matrix.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    r = corr_matrix.iloc[i, j]
                    if pd.notna(r):
                        pairs.append((cols_list[i], cols_list[j], float(r)))
            # sort by absolute correlation, take top 5
            pairs.sort(key=lambda x: abs(x[2]), reverse=True)
            for col_a, col_b, r in pairs[:5]:
                strength = _corr_strength_label(r)
                direction = _corr_direction_label(r)
                if r > 0:
                    insight_text = (
                        f"Рост «{col_a}» сопровождается ростом «{col_b}» — "
                        f"{strength} {direction} связь (r = {r:.2f})"
                    )
                else:
                    insight_text = (
                        f"Рост «{col_a}» сопровождается снижением «{col_b}» — "
                        f"{strength} {direction} связь (r = {r:.2f})"
                    )
                correlations.append({
                    "col_a": col_a,
                    "col_b": col_b,
                    "r": round(r, 3),
                    "strength": strength,
                    "direction": direction,
                    "insight_text": insight_text,
                })
        except Exception:
            pass

    # ---- Distributions ----
    distributions: list[dict] = []
    for col in num_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 3:
            continue
        mean_val = float(s.mean())
        median_val = float(s.median())
        std_val = float(s.std())
        skewness = _safe_skew(s)
        kurt = _safe_kurtosis(s)
        outlier_pct = _outlier_pct_3sigma(s)

        if skewness > 1:
            skew_text = (
                "Правосторонний перекос — большинство значений сосредоточено слева, "
                "есть крупные выбросы"
            )
        elif skewness < -1:
            skew_text = (
                "Левосторонний перекос — большинство значений сосредоточено справа, "
                "нижние значения ненормально малы"
            )
        else:
            skew_text = "Распределение близко к нормальному"

        if outlier_pct > 5:
            outlier_text = f"Много выбросов — {outlier_pct:.1f}% значений выходит за 3σ"
        else:
            outlier_text = ""

        insight_text = skew_text
        if outlier_text:
            insight_text += f". {outlier_text}"

        distributions.append({
            "col": col,
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4),
            "skewness": round(skewness, 3),
            "kurtosis": round(kurt, 3),
            "outlier_pct": outlier_pct,
            "insight_text": insight_text,
        })

    # ---- Trends (datetime + numeric) ----
    trends: list[dict] = []
    if dt_cols and num_cols:
        date_col = dt_cols[0]
        sorted_df = df.sort_values(date_col)
        for val_col in num_cols[:5]:  # limit to 5 numeric cols
            series = pd.to_numeric(sorted_df[val_col], errors="coerce").dropna()
            valid_dates = sorted_df.loc[series.index, date_col]
            if len(series) < 2:
                continue
            first_val = float(series.iloc[0])
            last_val = float(series.iloc[-1])
            if first_val == 0:
                pct_change = None
            else:
                pct_change = round((last_val - first_val) / abs(first_val) * 100, 1)

            if pct_change is None:
                direction = "нет данных"
                insight_text = f"Начальное значение равно нулю — изменение в % не вычислено"
            elif pct_change > 5:
                direction = "рост"
                first_date = valid_dates.iloc[0]
                last_date = valid_dates.iloc[-1]
                insight_text = (
                    f"Рост +{pct_change:.1f}% с "
                    f"{first_date.strftime('%d.%m.%Y') if hasattr(first_date, 'strftime') else str(first_date)} "
                    f"по {last_date.strftime('%d.%m.%Y') if hasattr(last_date, 'strftime') else str(last_date)}"
                )
            elif pct_change < -5:
                direction = "снижение"
                first_date = valid_dates.iloc[0]
                last_date = valid_dates.iloc[-1]
                insight_text = (
                    f"Снижение {pct_change:.1f}% с "
                    f"{first_date.strftime('%d.%m.%Y') if hasattr(first_date, 'strftime') else str(first_date)} "
                    f"по {last_date.strftime('%d.%m.%Y') if hasattr(last_date, 'strftime') else str(last_date)}"
                )
            else:
                direction = "стабильно"
                insight_text = f"Значение стабильно (изменение {pct_change:+.1f}%)"

            trends.append({
                "col_date": date_col,
                "col_value": val_col,
                "direction": direction,
                "pct_change": pct_change,
                "insight_text": insight_text,
            })

    # ---- Top values (categorical concentration) ----
    top_values: dict[str, dict] = {}
    for col in cat_cols:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 30:
            continue
        vc = df[col].value_counts(dropna=True)
        if vc.empty:
            continue
        top_value = vc.index[0]
        top_count = int(vc.iloc[0])
        total_non_null = int(df[col].notna().sum())
        top_pct = round(top_count / total_non_null * 100, 1) if total_non_null > 0 else 0.0

        if top_pct > 50:
            concentration_insight = (
                f"Высокая концентрация: «{top_value}» составляет {top_pct:.1f}%"
            )
        elif top_pct > 30:
            concentration_insight = (
                f"Умеренная концентрация: «{top_value}» составляет {top_pct:.1f}%"
            )
        else:
            concentration_insight = f"Равномерное распределение по {n_unique} категориям"

        top_values[col] = {
            "col": col,
            "top_value": str(top_value),
            "top_pct": top_pct,
            "n_unique": n_unique,
            "concentration_insight": concentration_insight,
        }

    # ---- Anomalies (IQR-based, top 5 most extreme) ----
    anomalies: list[dict] = []
    anomaly_scores: list[tuple[float, dict]] = []

    for col in num_cols[:20]:  # cap for performance
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 10:
            continue
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = (s < lower) | (s > upper)
        for idx in s[outlier_mask].index:
            val = float(s[idx])
            score = abs(val - s.median()) / (iqr if iqr > 0 else 1)
            anomaly_scores.append((score, {
                "row_idx": int(idx),
                "col": col,
                "value": round(val, 4),
                "expected_range": f"[{round(lower, 2)}, {round(upper, 2)}]",
            }))

    anomaly_scores.sort(key=lambda x: x[0], reverse=True)
    # deduplicate by row_idx — keep most anomalous col per row
    seen_rows: set[int] = set()
    for _, info in anomaly_scores:
        if info["row_idx"] not in seen_rows:
            anomalies.append(info)
            seen_rows.add(info["row_idx"])
        if len(anomalies) >= 5:
            break

    # ---- Recommendations ----
    recommendations: list[dict] = []

    # High missing data
    n_missing_cols = int((df.isnull().mean() > 0.05).sum())
    if missing_pct > 10:
        recommendations.append({
            "priority": "high",
            "action": f"Заполните пропуски в {n_missing_cols} столбцах",
            "reason": f"Общий уровень пропусков {missing_pct:.1f}% может исказить анализ",
            "action_type": "fill_na",
        })

    # Duplicates
    if n_dup > 0:
        recommendations.append({
            "priority": "high",
            "action": f"Удалите {n_dup} дублирующихся строк перед анализом",
            "reason": f"Дублей: {duplicate_pct:.1f}% от объёма — влияют на статистику",
            "action_type": "dedup",
        })

    # Time series forecast opportunity
    if dt_cols and num_cols:
        recommendations.append({
            "priority": "medium",
            "action": "Постройте прогноз временного ряда",
            "reason": (
                f"Обнаружен столбец дат «{dt_cols[0]}» и {len(num_cols)} числовых метрик — "
                "данные подходят для прогнозирования"
            ),
        })

    # Clustering opportunity
    low_missing_num = [
        c for c in num_cols if df[c].isnull().mean() < 0.1
    ]
    if len(low_missing_num) >= 5:
        recommendations.append({
            "priority": "medium",
            "action": "Проведите кластеризацию",
            "reason": (
                f"Доступно {len(low_missing_num)} числовых столбцов с низким уровнем пропусков — "
                "хорошая база для кластерного анализа (k-means, DBSCAN)"
            ),
        })

    # Strong correlation — recommend significance test
    strong_corr_pairs = [c for c in correlations if abs(c["r"]) > 0.7]
    if strong_corr_pairs:
        pair = strong_corr_pairs[0]
        recommendations.append({
            "priority": "medium",
            "action": (
                f"Проверьте статистическую значимость корреляции "
                f"«{pair['col_a']}» ↔ «{pair['col_b']}»"
            ),
            "reason": f"r = {pair['r']:.2f} — сильная связь, стоит исключить случайность (тест Пирсона/Спирмена)",
        })

    # Skewed columns — recommend normalization
    skewed_cols = [d["col"] for d in distributions if abs(d["skewness"]) > 1]
    for col in skewed_cols[:2]:
        recommendations.append({
            "priority": "low",
            "action": f"Нормализуйте / логарифмируйте «{col}»",
            "reason": (
                f"Перекос распределения = {next(d['skewness'] for d in distributions if d['col'] == col):.2f} — "
                "логарифм или Box-Cox улучшат качество моделей"
            ),
        })

    return {
        "summary": summary,
        "correlations": correlations,
        "distributions": distributions,
        "trends": trends,
        "top_values": top_values,
        "anomalies": anomalies,
        "recommendations": recommendations,
    }


def format_insights_markdown(insights: dict) -> str:
    """Convert an insights dict (from analyze_dataset) into a rich markdown string.

    Parameters
    ----------
    insights:
        Dict returned by :func:`analyze_dataset`.

    Returns
    -------
    str — markdown ready for st.markdown().
    """
    summary = insights.get("summary", {})
    correlations = insights.get("correlations", [])
    distributions = insights.get("distributions", [])
    trends = insights.get("trends", [])
    top_values = insights.get("top_values", {})
    recommendations = insights.get("recommendations", [])

    n_rows = summary.get("n_rows", 0)
    n_cols = summary.get("n_cols", 0)
    missing_pct = summary.get("missing_pct", 0)
    n_dup = summary.get("n_duplicates", 0)
    n_numeric = summary.get("n_numeric", 0)
    n_cat = summary.get("n_categorical", 0)
    n_dt = summary.get("n_datetime", 0)

    lines: list[str] = []

    # Header
    lines.append("## Автоматический анализ датасета")
    lines.append("")
    lines.append(
        f"**Размер:** {n_rows:,} строк × {n_cols} столбцов &nbsp;|&nbsp; "
        f"**Типы:** {n_numeric} числовых, {n_cat} категориальных, {n_dt} дат &nbsp;|&nbsp; "
        f"**Пропуски:** {missing_pct:.1f}% &nbsp;|&nbsp; "
        f"**Дублей:** {n_dup:,}"
    )
    lines.append("")

    # --- Correlations ---
    if correlations:
        lines.append("### Корреляции")
        for c in correlations:
            r = c["r"]
            abs_r = abs(r)
            if abs_r >= 0.8:
                icon = "🔗"
            elif abs_r >= 0.6:
                icon = "🔗"
            elif abs_r >= 0.4:
                icon = "〰️"
            else:
                icon = "➖"
            strength_cap = c["strength"].capitalize()
            direction_cap = c["direction"].capitalize()
            lines.append(
                f"- {icon} **{strength_cap} {c['direction']}**: "
                f"`{c['col_a']}` ↔ `{c['col_b']}` (r = {r:.2f}) — {c['insight_text']}"
            )
        lines.append("")
    else:
        lines.append("### Корреляции")
        lines.append("- Недостаточно числовых столбцов для расчёта корреляций.")
        lines.append("")

    # --- Distributions ---
    if distributions:
        lines.append("### Распределения")
        for d in distributions:
            col = d["col"]
            mean_v = d["mean"]
            median_v = d["median"]
            insight = d["insight_text"]
            lines.append(
                f"- 📊 **{col}**: Среднее {mean_v:,.3f}, медиана {median_v:,.3f} — {insight}"
            )
        lines.append("")

    # --- Trends ---
    if trends:
        lines.append("### Тренды")
        for t in trends:
            direction = t["direction"]
            if direction == "рост":
                icon = "📈"
            elif direction == "снижение":
                icon = "📉"
            else:
                icon = "➡️"
            lines.append(f"- {icon} **{t['col_value']}**: {t['insight_text']}")
        lines.append("")

    # --- Category concentration ---
    if top_values:
        lines.append("### Концентрация категорий")
        for col, info in top_values.items():
            top_pct = info["top_pct"]
            if top_pct > 50:
                icon = "🏷️"
            elif top_pct > 30:
                icon = "🏷️"
            else:
                icon = "📋"
            lines.append(
                f"- {icon} **{col}**: {info['concentration_insight']} "
                f"(всего значений: {info['n_unique']})"
            )
        lines.append("")

    # --- Recommendations ---
    if recommendations:
        lines.append("### Рекомендации")
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
        for rec in recommendations:
            icon = priority_icon.get(rec["priority"], "⚪")
            lines.append(f"- {icon} {rec['action']} — *{rec['reason']}*")
        lines.append("")

    return "\n".join(lines)


def score_data_quality(df: pd.DataFrame) -> dict:
    """
    Compute a comprehensive data quality score (0-100) with sub-scores.
    Returns dict with: overall, completeness, uniqueness, consistency, issues list.
    """
    if df is None or df.empty:
        return {
            "overall": 0.0,
            "completeness": 0.0,
            "uniqueness": 0.0,
            "consistency": 0.0,
            "issues": [{"level": "error", "message": "Датасет пуст или не загружен.", "col": None}],
        }

    n_rows = len(df)
    issues: list[dict] = []

    # ------------------------------------------------------------------
    # Completeness (0-100)
    # ------------------------------------------------------------------
    col_missing = df.isnull().mean()
    completeness = float(100.0 * (1.0 - col_missing.mean()))

    for col in df.columns:
        mp = float(col_missing[col]) * 100
        if mp > 30:
            issues.append({
                "level": "error",
                "message": f"Более {mp:.1f}% пропущенных значений",
                "col": col,
            })
        elif mp > 5:
            issues.append({
                "level": "warning",
                "message": f"{mp:.1f}% пропущенных значений — стоит заполнить",
                "col": col,
            })

    # ------------------------------------------------------------------
    # Uniqueness (0-100)
    # ------------------------------------------------------------------
    n_dup = int(df.duplicated().sum())
    dup_pct = n_dup / max(n_rows, 1) * 100
    uniqueness = float(100.0 * (1.0 - n_dup / max(n_rows, 1)))

    if dup_pct > 5:
        issues.append({
            "level": "error",
            "message": f"Найдено {n_dup} дублирующихся строк ({dup_pct:.1f}%)",
            "col": None,
        })
    elif n_dup > 0:
        issues.append({
            "level": "warning",
            "message": f"Найдено {n_dup} дублирующихся строк ({dup_pct:.1f}%) — рекомендуется удалить",
            "col": None,
        })

    # ------------------------------------------------------------------
    # Consistency (0-100)
    # ------------------------------------------------------------------
    penalty = 0.0

    # Constant columns
    for col in df.columns:
        if df[col].nunique(dropna=False) == 1:
            penalty += 10
            issues.append({
                "level": "error",
                "message": "Столбец константный — все значения одинаковые",
                "col": col,
            })

    # Mixed-type object columns
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        n_numeric = pd.to_numeric(s, errors="coerce").notna().sum()
        ratio = n_numeric / len(s)
        if 0.1 < ratio < 0.9:
            penalty += 5
            issues.append({
                "level": "warning",
                "message": (
                    f"Смешанный тип: {ratio*100:.0f}% значений похожи на числа, "
                    "остальные — текст"
                ),
                "col": col,
            })

    consistency = max(0.0, 100.0 - penalty)

    # ------------------------------------------------------------------
    # Extra warnings / info
    # ------------------------------------------------------------------
    # High cardinality categoricals
    for col in df.select_dtypes(include=["object", "category"]).columns:
        n_unique = df[col].nunique(dropna=True)
        if n_unique > 50 and n_unique / max(n_rows, 1) > 0.5:
            issues.append({
                "level": "warning",
                "message": f"Высокая кардинальность: {n_unique} уникальных значений",
                "col": col,
            })

    # High skew numerics
    for col in _numeric_cols(df):
        sk = _safe_skew(df[col])
        if abs(sk) > 2:
            issues.append({
                "level": "warning",
                "message": f"Сильный перекос распределения (skew = {sk:.2f}) — рекомендуется нормализация",
                "col": col,
            })
            issues.append({
                "level": "info",
                "message": f"Попробуйте логарифмирование или Box-Cox для нормализации столбца",
                "col": col,
            })

    # Suggest date parsing for object cols that look like dates
    for col in df.select_dtypes(include="object").columns:
        s = df[col].dropna().head(50)
        if len(s) == 0:
            continue
        try:
            parsed = pd.to_datetime(s, errors="coerce")
            if parsed.notna().mean() > 0.8:
                issues.append({
                    "level": "info",
                    "message": "Столбец похож на дату — рекомендуется преобразовать в datetime",
                    "col": col,
                })
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Overall weighted score
    # ------------------------------------------------------------------
    overall = round(0.45 * completeness + 0.35 * uniqueness + 0.20 * consistency, 1)

    return {
        "overall": overall,
        "completeness": round(completeness, 1),
        "uniqueness": round(uniqueness, 1),
        "consistency": round(consistency, 1),
        "issues": issues,
    }


def get_chart_recommendation(
    df: pd.DataFrame,
    col_x: str,
    col_y: str | None = None,
) -> tuple[str, str]:
    """Recommend the best chart type for the given column combination.

    Parameters
    ----------
    df:
        Source DataFrame.
    col_x:
        Primary column (x-axis / single column).
    col_y:
        Optional second column (y-axis).

    Returns
    -------
    tuple[str, str]
        (chart_type, reason_russian)
        chart_type is one of: "scatter", "bar", "line", "pie", "histogram"
    """
    if col_x not in df.columns:
        return "histogram", "Столбец не найден в датасете."

    x_is_numeric = pd.api.types.is_numeric_dtype(df[col_x])
    x_is_datetime = pd.api.types.is_datetime64_any_dtype(df[col_x])
    x_is_cat = not x_is_numeric and not x_is_datetime

    if col_y is None or col_y not in df.columns:
        # Single column
        if x_is_numeric:
            return "histogram", (
                f"«{col_x}» — числовой столбец. "
                "Гистограмма покажет распределение значений."
            )
        if x_is_datetime:
            return "line", (
                f"«{col_x}» — временной столбец. "
                "Линейный график подходит для отображения дат."
            )
        # Categorical
        n_unique = df[col_x].nunique()
        if n_unique <= 8:
            return "pie", (
                f"«{col_x}» имеет {n_unique} уникальных значений. "
                "Круговая диаграмма наглядно покажет доли."
            )
        return "bar", (
            f"«{col_x}» имеет {n_unique} категорий — слишком много для круговой. "
            "Столбчатая диаграмма лучше."
        )

    # Two columns
    y_is_numeric = pd.api.types.is_numeric_dtype(df[col_y])
    y_is_datetime = pd.api.types.is_datetime64_any_dtype(df[col_y])

    if x_is_numeric and y_is_numeric:
        # Check correlation for insight
        try:
            r = float(df[[col_x, col_y]].dropna().corr().iloc[0, 1])
            r_str = f"r = {r:.2f}"
            if abs(r) >= 0.6:
                reason = (
                    f"Оба столбца числовые. Точечная диаграмма покажет связь "
                    f"({r_str} — {'сильная' if abs(r) >= 0.7 else 'умеренная'} корреляция)."
                )
            else:
                reason = (
                    f"Оба столбца числовые ({r_str}). "
                    "Точечная диаграмма выявит паттерны или кластеры."
                )
        except Exception:
            reason = "Оба столбца числовые — точечная диаграмма покажет взаимосвязь."
        return "scatter", reason

    if (x_is_datetime and y_is_numeric) or (y_is_datetime and x_is_numeric):
        date_col = col_x if x_is_datetime else col_y
        val_col = col_y if x_is_datetime else col_x
        return "line", (
            f"«{date_col}» — дата, «{val_col}» — числовой. "
            "Линейный график идеален для временных рядов."
        )

    if x_is_cat and y_is_numeric:
        return "bar", (
            f"«{col_x}» — категория, «{col_y}» — числовой. "
            "Столбчатая диаграмма сравнит значения по группам."
        )

    if x_is_numeric and not y_is_numeric and not y_is_datetime:
        return "bar", (
            f"«{col_y}» — категория, «{col_x}» — числовой. "
            "Столбчатая диаграмма сравнит значения по группам."
        )

    # Fallback
    return "bar", "Столбчатая диаграмма подходит для большинства сочетаний столбцов."

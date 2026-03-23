"""
core/interpret.py – Plain-language interpretation helpers for analytics outputs.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def interpret_pvalue(p: float, alpha: float = 0.05) -> str:
    if p < alpha:
        strength = "крайне значимо" if p < 0.001 else ("значимо" if p < 0.01 else "значимо")
        return (
            f"Результат статистически **{strength}** (p={p:.4f} < α={alpha}). "
            "Разница между группами, вероятно, не случайна."
        )
    return (
        f"Результат **не** статистически значим (p={p:.4f} ≥ α={alpha}). "
        "Недостаточно доказательств для отклонения нулевой гипотезы."
    )


def interpret_effect_size(d: float, kind: str = "cohen_d") -> str:
    """kind: 'cohen_d' | 'eta_squared' | 'cliff_delta'"""
    abs_d = abs(d)
    if kind == "cohen_d":
        if abs_d < 0.2:
            label, pct = "пренебрежимо малый", "<9%"
        elif abs_d < 0.5:
            label, pct = "малый", "≈9–25%"
        elif abs_d < 0.8:
            label, pct = "средний", "≈25–64%"
        else:
            label, pct = "большой", ">64%"
        return f"Величина эффекта: **{label}** (d={d:.3f}). Перекрытие распределений: {pct}."
    elif kind == "eta_squared":
        if abs_d < 0.01:
            label = "пренебрежимо малый"
        elif abs_d < 0.06:
            label = "малый"
        elif abs_d < 0.14:
            label = "средний"
        else:
            label = "большой"
        return f"Объяснённая дисперсия: η²={d:.3f} ({label} эффект)."
    elif kind == "cliff_delta":
        if abs_d < 0.147:
            label = "пренебрежимо малый"
        elif abs_d < 0.33:
            label = "малый"
        elif abs_d < 0.474:
            label = "средний"
        else:
            label = "большой"
        return f"Cliff's delta: δ={d:.3f} ({label} эффект). Вероятность того, что значение из группы 1 > группы 2: {(d+1)/2*100:.0f}%."
    return f"Величина эффекта: {d:.3f}."


def interpret_correlation(r: float, p: float | None = None) -> str:
    abs_r = abs(r)
    direction = "положительная" if r > 0 else "отрицательная"
    if abs_r < 0.2:
        strength = "очень слабая"
    elif abs_r < 0.4:
        strength = "слабая"
    elif abs_r < 0.6:
        strength = "умеренная"
    elif abs_r < 0.8:
        strength = "сильная"
    else:
        strength = "очень сильная"

    text = f"{strength.capitalize()} {direction} корреляция (r={r:.3f})."
    if p is not None:
        sig = "статистически значима" if p < 0.05 else "статистически незначима"
        text += f" Корреляция {sig} (p={p:.4f})."
    if abs_r > 0.7:
        text += " ⚠️ Возможна мультиколлинеарность при построении регрессионных моделей."
    return text


def interpret_trend(series: pd.Series) -> dict:
    if len(series) < 3:
        return {"direction": "unknown", "strength": "unknown", "description": "Недостаточно данных."}

    x = np.arange(len(series))
    y = series.dropna().values
    if len(y) < 3:
        return {"direction": "unknown", "strength": "unknown", "description": "Недостаточно данных."}

    coef = np.polyfit(x[:len(y)], y, 1)
    slope = coef[0]
    relative_slope = slope / (abs(y.mean()) + 1e-9)

    if abs(relative_slope) < 0.005:
        direction, strength = "flat", "stable"
        desc = "Ряд стабилен — выраженного тренда не обнаружено."
    elif relative_slope > 0:
        direction = "up"
        strength = "strong" if relative_slope > 0.05 else "weak"
        desc = f"Восходящий тренд: +{relative_slope*100:.1f}% в среднем за период."
    else:
        direction = "down"
        strength = "strong" if relative_slope < -0.05 else "weak"
        desc = f"Нисходящий тренд: {relative_slope*100:.1f}% в среднем за период."

    return {"direction": direction, "strength": strength, "description": desc}


def interpret_distribution(series: pd.Series) -> str:
    from scipy import stats as sp_stats
    s = series.dropna()
    if len(s) < 8:
        return "Недостаточно данных для анализа распределения."

    skew = float(s.skew())
    kurt = float(s.kurtosis())

    if abs(skew) < 0.5 and abs(kurt) < 1:
        shape = "нормальное"
    elif skew > 1:
        shape = "правостороннее (длинный правый хвост)"
    elif skew < -1:
        shape = "левостороннее (длинный левый хвост)"
    elif kurt > 3:
        shape = "лептокуртозное (острый пик, тяжёлые хвосты)"
    elif kurt < -1:
        shape = "платикуртозное (плоский пик)"
    else:
        shape = "умеренно асимметричное"

    desc = f"Распределение: **{shape}** (skewness={skew:.2f}, kurtosis={kurt:.2f})."

    if abs(skew) > 1:
        desc += " Рекомендуется лог-преобразование для линейных моделей."

    # Normality test
    if len(s) <= 5000:
        try:
            _, p_sw = sp_stats.shapiro(s.sample(min(len(s), 500), random_state=42))
            if p_sw < 0.05:
                desc += " Тест Шапиро–Уилка: **не нормальное** (p={:.4f}).".format(p_sw)
            else:
                desc += " Тест Шапиро–Уилка: нормальное (p={:.4f}).".format(p_sw)
        except Exception:
            pass

    return desc


def interpret_missing(col: str, pct: float) -> str:
    if pct == 0:
        return f"«{col}»: пропусков нет ✅"
    elif pct < 5:
        return f"«{col}»: {pct:.1f}% пропусков — незначительно, безопасно удалить строки."
    elif pct < 30:
        return f"«{col}»: {pct:.1f}% пропусков — рекомендуется медианная/модальная импутация."
    elif pct < 70:
        return f"«{col}»: {pct:.1f}% пропусков — высокий уровень; рассмотрите удаление колонки или KNN-импутацию."
    else:
        return f"«{col}»: {pct:.1f}% пропусков — критически высокий уровень; рекомендуется удалить колонку."


def cdr_risk_label(cdr_pct: float) -> tuple[str, str]:
    """Return (label, color) for CDR level."""
    if cdr_pct < 3:
        return "Низкий риск", "green"
    elif cdr_pct < 7:
        return "Средний риск", "orange"
    else:
        return "Высокий риск", "red"


def hhi_interpretation(hhi: float) -> str:
    if hhi < 0.1:
        return "Низкая концентрация (HHI < 0.10): портфель хорошо диверсифицирован."
    elif hhi < 0.18:
        return "Умеренная концентрация (0.10 ≤ HHI < 0.18): допустимо, мониторьте топ-заёмщиков."
    else:
        return f"Высокая концентрация (HHI = {hhi:.3f} ≥ 0.18): риск концентрации значителен. Требуется диверсификация."

"""
core/timeseries_auto.py — Слой автоматики поверх core/models.py.

Назначение: дать нетехническому пользователю «прогноз в один клик».
Все функции pure, с понятными русскими подсказками в результате.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

from core.models import (
    ForecastResult,
    run_arx_forecast,
    run_naive_forecast,
    run_sarimax_forecast,
    run_stl_decomposition,
    _coerce_datetime_series,
)


_FREQ_TO_PERIOD = {
    "M": 12, "ME": 12, "MS": 12, "BM": 12, "BME": 12,
    "Q": 4, "QE": 4, "QS": 4, "BQ": 4, "BQE": 4,
    "A": 1, "Y": 1, "YE": 1, "AS": 1, "YS": 1,
    "W": 52, "W-MON": 52, "W-SUN": 52,
    "D": 7, "B": 5,
    "H": 24,
}


@dataclass
class AutoForecastResult:
    """Результат авто-прогноза с человеко-читаемыми пояснениями."""

    forecast: ForecastResult
    decisions: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Период сезонности
# ---------------------------------------------------------------------------

def detect_period(series: pd.Series, dates: pd.Series | None = None) -> int:
    """Определить сезонный период по ACF (пик на лагах 2..52) либо по pd.infer_freq.

    Возвращает 1, если сезонность не обнаружена.
    """
    if dates is not None:
        try:
            d = _coerce_datetime_series(dates).dropna().sort_values().reset_index(drop=True)
            if len(d) >= 3:
                freq = pd.infer_freq(d)
                if freq:
                    base = freq.split("-")[0]
                    if base in _FREQ_TO_PERIOD:
                        return _FREQ_TO_PERIOD[base]
                    if freq in _FREQ_TO_PERIOD:
                        return _FREQ_TO_PERIOD[freq]
        except Exception:
            pass

    s = pd.Series(series).dropna().astype(float).values
    n = len(s)
    if n < 8:
        return 1

    try:
        from statsmodels.tsa.stattools import acf
        max_lag = min(52, n // 2 - 1)
        if max_lag < 2:
            return 1
        # Снимаем линейный тренд, чтобы ACF не «прилипала» к малым лагам.
        x = np.arange(n)
        slope, intercept = np.polyfit(x, s, 1)
        s_detrended = s - (slope * x + intercept)
        acf_vals = acf(s_detrended, nlags=max_lag, fft=True)
        # Локальные максимумы по лагам ≥ 2 со значением > 0.3.
        peaks = []
        for k in range(2, max_lag):
            if (
                acf_vals[k] > 0.3
                and acf_vals[k] >= acf_vals[k - 1]
                and acf_vals[k] >= acf_vals[k + 1]
            ):
                peaks.append((k, acf_vals[k]))
        if peaks:
            # Берём наибольший по значению ACF; при равенстве — наименьший лаг.
            peaks.sort(key=lambda kv: (-kv[1], kv[0]))
            return int(peaks[0][0])
    except Exception:
        pass

    return 1


# ---------------------------------------------------------------------------
# Стационарность (ADF)
# ---------------------------------------------------------------------------

def adf_stationarity(series: pd.Series) -> dict[str, Any]:
    """ADF-тест стационарности. Возвращает {stationary, pvalue, hint}."""
    s = pd.Series(series).dropna().astype(float).values
    if len(s) < 10:
        return {
            "stationary": None,
            "pvalue": float("nan"),
            "hint": "Слишком мало точек для теста стационарности.",
        }
    try:
        from statsmodels.tsa.stattools import adfuller
        pvalue = float(adfuller(s, autolag="AIC")[1])
    except Exception as exc:
        return {
            "stationary": None,
            "pvalue": float("nan"),
            "hint": f"Не удалось выполнить тест: {exc}",
        }

    stationary = pvalue < 0.05
    if stationary:
        hint = f"Ряд стационарен (ADF p = {pvalue:.3f})."
    else:
        hint = (
            f"Ряд нестационарен (ADF p = {pvalue:.3f}). "
            "Рассмотрите дифференцирование (d=1) перед интерпретацией ACF/PACF."
        )
    return {"stationary": stationary, "pvalue": pvalue, "hint": hint}


# ---------------------------------------------------------------------------
# Сила сезонности (через STL)
# ---------------------------------------------------------------------------

def seasonality_strength(
    df: pd.DataFrame, date_col: str, target_col: str, period: int
) -> dict[str, Any]:
    """Оценка силы сезонности Fs через STL. Возвращает {fs, label}."""
    if period < 2:
        return {"fs": 0.0, "label": "Нет сезонности"}
    try:
        result = run_stl_decomposition(
            df, date_col, target_col, period=period, robust=True, multiplicative=False
        )
        fs = float(result.seasonality_strength)
    except Exception:
        return {"fs": 0.0, "label": "Не удалось оценить"}

    if fs >= 0.6:
        label = "Сильная сезонность"
    elif fs >= 0.3:
        label = "Умеренная сезонность"
    else:
        label = "Слабая сезонность"
    return {"fs": fs, "label": label}


# ---------------------------------------------------------------------------
# Рекомендатор модели
# ---------------------------------------------------------------------------

def recommend_model(
    n_obs: int,
    exog_cols: Iterable[str] | None,
    period: int,
    seasonality_label: str,
) -> dict[str, Any]:
    """Простые правила: вернуть имя модели, причину и уровень уверенности.

    Имена: 'naive_seasonal', 'naive_last', 'arx', 'sarimax'.
    """
    has_exog = bool(exog_cols)

    if n_obs < 12:
        return {
            "model": "naive_last",
            "reason": "Очень мало наблюдений (< 12). Используем последнее значение.",
            "confidence": "low",
        }
    if n_obs < 24:
        if period >= 2:
            return {
                "model": "naive_seasonal",
                "reason": "Мало данных для сложных моделей — берём сезонный наивный прогноз.",
                "confidence": "low",
            }
        return {
            "model": "naive_last",
            "reason": "Мало данных и сезонности нет — берём последнее значение.",
            "confidence": "low",
        }

    if seasonality_label == "Сильная сезонность" and n_obs >= 50:
        return {
            "model": "sarimax",
            "reason": (
                "Сильная сезонность и достаточно данных — выбираем SARIMAX, "
                "он явно моделирует сезонную и несезонную компоненты."
            ),
            "confidence": "high",
        }
    if has_exog and n_obs >= 30:
        return {
            "model": "arx",
            "reason": (
                "Заданы внешние факторы — используем ARX (Ridge с лагами), "
                "это позволит учесть их вклад."
            ),
            "confidence": "high",
        }
    if seasonality_label == "Умеренная сезонность" and n_obs >= 50:
        return {
            "model": "sarimax",
            "reason": "Умеренная сезонность и достаточно данных — пробуем SARIMAX.",
            "confidence": "medium",
        }
    if period >= 2:
        return {
            "model": "naive_seasonal",
            "reason": "Сезонный наивный прогноз как надёжный базовый вариант.",
            "confidence": "medium",
        }
    return {
        "model": "naive_last",
        "reason": "Без сезонности — берём наивный прогноз по последнему значению.",
        "confidence": "low",
    }


# ---------------------------------------------------------------------------
# Подбор лагов и order
# ---------------------------------------------------------------------------

def suggest_arx_lags(series: pd.Series, period: int, max_lag: int = 24) -> list[int]:
    """Топ-3 пика PACF плюс сезонный лаг."""
    s = pd.Series(series).dropna().astype(float).values
    if len(s) < max_lag + 2:
        max_lag = max(2, len(s) // 3)

    chosen: list[int] = [1]
    try:
        from statsmodels.tsa.stattools import pacf
        nlags = min(max_lag, len(s) // 2 - 1)
        if nlags >= 2:
            pacf_vals = pacf(s, nlags=nlags, method="ywm")
            ranked = sorted(range(2, nlags + 1), key=lambda k: -abs(pacf_vals[k]))
            for lag in ranked[:3]:
                if lag not in chosen:
                    chosen.append(int(lag))
    except Exception:
        chosen = [1, 2, 3]

    if period >= 2 and period not in chosen and period <= max_lag:
        chosen.append(int(period))
    return sorted(set(chosen))


def suggest_sarimax_order(
    series: pd.Series, period: int, adf_pvalue: float | None
) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    """Простые эвристики: d по ADF, D=1 при сезонности, p,q,P,Q ∈ {0,1}."""
    if adf_pvalue is None or pd.isna(adf_pvalue) or adf_pvalue >= 0.05:
        d = 1
    else:
        d = 0
    seasonal_period = period if period >= 2 else 0
    D = 1 if seasonal_period >= 2 else 0
    order = (1, d, 1)
    seasonal_order = (1, D, 1, seasonal_period) if seasonal_period >= 2 else (0, 0, 0, 0)
    return order, seasonal_order


# ---------------------------------------------------------------------------
# Интерпретация метрик
# ---------------------------------------------------------------------------

def interpret_metrics(metrics: dict[str, float]) -> dict[str, str]:
    """Вернуть {label, color, message} по MAPE/sMAPE."""
    pct: float | None = None
    name: str | None = None
    for key in ("MAPE", "sMAPE"):
        if key in metrics and not pd.isna(metrics[key]):
            pct = float(metrics[key])
            name = key
            break

    if pct is None:
        return {
            "label": "Нет оценки",
            "color": "secondary",
            "message": "Метрики качества не рассчитаны.",
        }

    if pct < 10:
        return {
            "label": "Хорошее качество",
            "color": "success",
            "message": f"{name} {pct:.1f}% — модель прогнозирует хорошо.",
        }
    if pct < 20:
        return {
            "label": "Удовлетворительно",
            "color": "warning",
            "message": f"{name} {pct:.1f}% — приемлемо, но есть запас улучшения.",
        }
    return {
        "label": "Плохое качество",
        "color": "danger",
        "message": (
            f"{name} {pct:.1f}% — большая ошибка прогноза. "
            "Проверьте данные на пропуски, выбросы, смену режима."
        ),
    }


# ---------------------------------------------------------------------------
# Оркестратор «авто-прогноз»
# ---------------------------------------------------------------------------

def run_auto_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str] | None = None,
    horizon: int | None = None,
) -> AutoForecastResult:
    """Полный авто-прогноз: детект периода, ADF, выбор модели, подбор гиперпараметров."""
    if df is None or df.empty:
        raise ValueError("Пустой DataFrame.")
    if date_col not in df.columns or target_col not in df.columns:
        raise ValueError("Колонки даты или цели нет в датасете.")

    work = df[[date_col, target_col] + list(exog_cols or [])].copy()
    work = work.dropna(subset=[date_col, target_col])
    work = work.sort_values(date_col).reset_index(drop=True)
    n_obs = len(work)
    if n_obs < 4:
        raise ValueError("Слишком мало наблюдений для прогноза (< 4).")

    series = work[target_col].astype(float)
    period = detect_period(series, work[date_col])
    adf = adf_stationarity(series)
    season = seasonality_strength(work, date_col, target_col, period)
    rec = recommend_model(n_obs, exog_cols, period, season["label"])

    if horizon is None:
        horizon = max(period if period >= 2 else 1, max(6, n_obs // 4))
        horizon = int(min(horizon, 60))

    decisions: dict[str, Any] = {
        "n_obs": n_obs,
        "period": period,
        "horizon": horizon,
        "stationarity": adf,
        "seasonality": season,
        "model": rec,
        "exog_cols": list(exog_cols or []),
    }
    notes: list[str] = []
    notes.append(
        f"Наблюдений: {n_obs}. Сезонный период: {period if period >= 2 else 'не обнаружен'}."
    )
    notes.append(adf["hint"])
    if season["label"] != "Нет сезонности":
        notes.append(f"{season['label']} (Fs = {season['fs']:.2f}).")
    notes.append(rec["reason"])

    model_name = rec["model"]
    if model_name == "naive_last":
        forecast = run_naive_forecast(
            work, date_col, target_col,
            horizon=horizon, seasonal=False, period=max(period, 1),
        )
    elif model_name == "naive_seasonal":
        forecast = run_naive_forecast(
            work, date_col, target_col,
            horizon=horizon, seasonal=True, period=max(period, 2),
        )
    elif model_name == "arx":
        lags = suggest_arx_lags(series, period)
        decisions["arx_lags"] = lags
        notes.append(f"Подобраны лаги AR: {lags}.")
        forecast = run_arx_forecast(
            work, date_col, target_col,
            exog_cols=list(exog_cols or []),
            lags=lags, horizon=horizon, alpha=1.0,
        )
    elif model_name == "sarimax":
        order, seasonal_order = suggest_sarimax_order(series, period, adf["pvalue"])
        decisions["sarimax_order"] = order
        decisions["sarimax_seasonal_order"] = seasonal_order
        notes.append(
            f"SARIMAX order={order}, seasonal_order={seasonal_order}."
        )
        try:
            forecast = run_sarimax_forecast(
                work, date_col, target_col,
                exog_cols=list(exog_cols or []),
                order=order, seasonal_order=seasonal_order, horizon=horizon,
            )
        except Exception as exc:
            notes.append(f"SARIMAX не сошёлся ({exc}). Падаем в сезонный наивный.")
            decisions["model"] = {
                "model": "naive_seasonal",
                "reason": "Fallback после ошибки SARIMAX.",
                "confidence": "low",
            }
            forecast = run_naive_forecast(
                work, date_col, target_col,
                horizon=horizon, seasonal=period >= 2, period=max(period, 2),
            )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    decisions["metrics_interpretation"] = interpret_metrics(forecast.metrics)

    return AutoForecastResult(forecast=forecast, decisions=decisions, notes=notes)

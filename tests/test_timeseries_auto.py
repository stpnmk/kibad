"""
tests/test_timeseries_auto.py — Тесты слоя автоматики временных рядов.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.timeseries_auto import (
    adf_stationarity,
    detect_period,
    interpret_metrics,
    recommend_model,
    run_auto_forecast,
    seasonality_strength,
    suggest_arx_lags,
    suggest_sarimax_order,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_seasonal_ts() -> pd.DataFrame:
    """48 месяцев: тренд + явная годовая сезонность."""
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=48, freq="MS")
    trend = np.arange(48) * 5 + 1000
    seasonal = 100 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = np.random.normal(0, 15, 48)
    return pd.DataFrame({"date": dates, "value": trend + seasonal + noise})


@pytest.fixture
def short_ts() -> pd.DataFrame:
    """8 наблюдений — слишком мало для сложных моделей."""
    dates = pd.date_range("2024-01-01", periods=8, freq="MS")
    return pd.DataFrame({"date": dates, "value": np.arange(8) * 2.0 + 100})


@pytest.fixture
def random_walk_ts() -> pd.DataFrame:
    """Случайное блуждание — нестационарно."""
    np.random.seed(7)
    dates = pd.date_range("2018-01-01", periods=100, freq="MS")
    values = np.cumsum(np.random.normal(0, 1, 100)) + 100
    return pd.DataFrame({"date": dates, "value": values})


# ---------------------------------------------------------------------------
# detect_period
# ---------------------------------------------------------------------------

def test_detect_period_monthly_via_freq(monthly_seasonal_ts):
    period = detect_period(monthly_seasonal_ts["value"], monthly_seasonal_ts["date"])
    assert period == 12


def test_detect_period_weekly_via_freq():
    dates = pd.date_range("2024-01-01", periods=20, freq="W")
    s = pd.Series(np.arange(20))
    assert detect_period(s, dates) == 52


def test_detect_period_no_dates_uses_acf(monthly_seasonal_ts):
    period = detect_period(monthly_seasonal_ts["value"])
    assert period == 12


def test_detect_period_short_returns_one():
    s = pd.Series([1.0, 2.0, 3.0])
    assert detect_period(s) == 1


# ---------------------------------------------------------------------------
# adf_stationarity
# ---------------------------------------------------------------------------

def test_adf_random_walk_is_nonstationary(random_walk_ts):
    res = adf_stationarity(random_walk_ts["value"])
    assert res["stationary"] is False
    assert "нестационарен" in res["hint"]


def test_adf_white_noise_is_stationary():
    np.random.seed(0)
    s = pd.Series(np.random.normal(0, 1, 200))
    res = adf_stationarity(s)
    assert res["stationary"] is True


def test_adf_too_short():
    res = adf_stationarity(pd.Series([1.0, 2.0, 3.0]))
    assert res["stationary"] is None
    assert np.isnan(res["pvalue"])


# ---------------------------------------------------------------------------
# seasonality_strength
# ---------------------------------------------------------------------------

def test_seasonality_strength_strong(monthly_seasonal_ts):
    res = seasonality_strength(monthly_seasonal_ts, "date", "value", period=12)
    assert res["fs"] > 0.5
    assert res["label"] in ("Сильная сезонность", "Умеренная сезонность")


def test_seasonality_strength_no_period():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=24, freq="MS"),
                       "value": np.arange(24, dtype=float)})
    res = seasonality_strength(df, "date", "value", period=1)
    assert res["label"] == "Нет сезонности"


# ---------------------------------------------------------------------------
# recommend_model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_obs,exog,period,seas_label,expected", [
    (8,  None, 1,  "Нет сезонности",        "naive_last"),
    (10, None, 12, "Сильная сезонность",    "naive_last"),
    (18, None, 12, "Сильная сезонность",    "naive_seasonal"),
    (60, None, 12, "Сильная сезонность",    "sarimax"),
    (60, ["x"], 12, "Слабая сезонность",    "arx"),
    (40, ["x"], 12, "Слабая сезонность",    "arx"),
    (60, None, 12, "Умеренная сезонность",  "sarimax"),
    (60, None, 1,  "Нет сезонности",        "naive_last"),
])
def test_recommend_model_rules(n_obs, exog, period, seas_label, expected):
    rec = recommend_model(n_obs, exog, period, seas_label)
    assert rec["model"] == expected
    assert rec["reason"]
    assert rec["confidence"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# suggest_*
# ---------------------------------------------------------------------------

def test_suggest_arx_lags_includes_seasonal(monthly_seasonal_ts):
    lags = suggest_arx_lags(monthly_seasonal_ts["value"], period=12)
    assert 1 in lags
    assert 12 in lags
    assert all(isinstance(l, int) for l in lags)


def test_suggest_sarimax_order_nonstationary():
    order, seasonal = suggest_sarimax_order(pd.Series([1.0] * 50), period=12, adf_pvalue=0.5)
    assert order[1] == 1  # d=1
    assert seasonal[3] == 12


def test_suggest_sarimax_order_no_seasonality():
    order, seasonal = suggest_sarimax_order(pd.Series([1.0] * 50), period=1, adf_pvalue=0.01)
    assert seasonal == (0, 0, 0, 0)
    assert order[1] == 0


# ---------------------------------------------------------------------------
# interpret_metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mape_val,color", [
    (5.0, "success"),
    (15.0, "warning"),
    (40.0, "danger"),
])
def test_interpret_metrics_mape(mape_val, color):
    out = interpret_metrics({"MAPE": mape_val})
    assert out["color"] == color


def test_interpret_metrics_smape_priority():
    out = interpret_metrics({"sMAPE": 7.0})
    assert out["color"] == "success"
    assert "sMAPE" in out["message"]


def test_interpret_metrics_missing():
    out = interpret_metrics({})
    assert out["color"] == "secondary"


# ---------------------------------------------------------------------------
# run_auto_forecast — end-to-end
# ---------------------------------------------------------------------------

def test_auto_forecast_seasonal(monthly_seasonal_ts):
    result = run_auto_forecast(monthly_seasonal_ts, "date", "value")
    assert result.forecast.forecast_df is not None
    assert not result.forecast.forecast_df.empty
    assert "model" in result.decisions
    assert result.decisions["period"] == 12
    assert len(result.notes) >= 2


def test_auto_forecast_short_series_picks_naive(short_ts):
    result = run_auto_forecast(short_ts, "date", "value")
    assert result.decisions["model"]["model"] in ("naive_last", "naive_seasonal")


def test_auto_forecast_with_exog():
    np.random.seed(1)
    dates = pd.date_range("2018-01-01", periods=60, freq="MS")
    x = np.random.normal(0, 1, 60)
    y = 50 + 2.5 * x + np.random.normal(0, 1, 60)
    df = pd.DataFrame({"date": dates, "y": y, "x": x})
    result = run_auto_forecast(df, "date", "y", exog_cols=["x"])
    assert result.decisions["model"]["model"] in ("arx", "sarimax")


def test_auto_forecast_empty_raises():
    with pytest.raises(ValueError):
        run_auto_forecast(pd.DataFrame({"date": [], "value": []}), "date", "value")


def test_auto_forecast_horizon_default(monthly_seasonal_ts):
    result = run_auto_forecast(monthly_seasonal_ts, "date", "value")
    h = result.decisions["horizon"]
    assert 6 <= h <= 60

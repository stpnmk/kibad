"""
tests/test_models.py – Unit tests for core/models.py
"""
import pytest
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    mae, rmse, mape, bias, compute_all_metrics,
    NaiveForecast, run_naive_forecast, run_arx_forecast,
    ForecastResult, rolling_backtest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def monthly_ts():
    np.random.seed(42)
    dates = pd.date_range("2018-01-01", periods=48, freq="MS")
    trend = np.arange(48) * 5 + 1000
    seasonal = 100 * np.sin(2 * np.pi * np.arange(48) / 12)
    noise = np.random.normal(0, 20, 48)
    values = trend + seasonal + noise
    return pd.DataFrame({"date": dates, "value": values})


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_mae():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.5, 2.5, 3.5])
    assert mae(a, p) == pytest.approx(0.5)


def test_rmse():
    a = np.array([0.0, 0.0, 0.0])
    p = np.array([1.0, 1.0, 1.0])
    assert rmse(a, p) == pytest.approx(1.0)


def test_mape():
    a = np.array([100.0, 200.0, 300.0])
    p = np.array([110.0, 190.0, 300.0])
    expected = np.mean([10 / 100, 10 / 200, 0]) * 100
    assert mape(a, p) == pytest.approx(expected, rel=1e-5)


def test_mape_skips_zeros():
    a = np.array([0.0, 100.0])
    p = np.array([5.0, 110.0])
    # Should only use the non-zero actual
    assert not np.isnan(mape(a, p))


def test_bias():
    a = np.array([10.0, 10.0, 10.0])
    p = np.array([12.0, 12.0, 12.0])
    assert bias(a, p) == pytest.approx(2.0)


def test_compute_all_metrics():
    a = np.array([100.0, 200.0, 150.0])
    p = np.array([110.0, 190.0, 155.0])
    m = compute_all_metrics(a, p)
    assert set(m.keys()) == {"MAE", "RMSE", "MAPE", "Bias"}
    assert all(isinstance(v, float) for v in m.values())


def test_compute_all_metrics_ignores_nans():
    a = np.array([100.0, np.nan, 150.0])
    p = np.array([110.0, 200.0, 155.0])
    m = compute_all_metrics(a, p)
    assert not np.isnan(m["MAE"])


# ---------------------------------------------------------------------------
# NaiveForecast
# ---------------------------------------------------------------------------

def test_naive_requires_fit():
    model = NaiveForecast()
    with pytest.raises(RuntimeError):
        model.predict(3)


def test_naive_plain_repeats_last():
    model = NaiveForecast(seasonal=False)
    model.fit(np.array([1, 2, 3, 10.0]))
    preds = model.predict(3)
    assert np.all(preds == 10.0)


def test_naive_seasonal_repeats_cycle():
    history = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    model = NaiveForecast(seasonal=True, period=12)
    model.fit(history)
    preds = model.predict(12)
    np.testing.assert_array_equal(preds, history)


def test_naive_forecast_short_history_falls_back():
    model = NaiveForecast(seasonal=True, period=12)
    model.fit(np.array([5.0, 6.0, 7.0]))  # shorter than period
    preds = model.predict(3)
    assert np.all(preds == 7.0)


# ---------------------------------------------------------------------------
# run_naive_forecast
# ---------------------------------------------------------------------------

def test_run_naive_forecast_shape(monthly_ts):
    result = run_naive_forecast(monthly_ts, "date", "value", horizon=6, period=12)
    assert isinstance(result, ForecastResult)
    fc = result.forecast_df
    assert "date" in fc.columns
    assert "actual" in fc.columns
    assert "forecast" in fc.columns
    # Future rows
    fut = fc[fc["actual"].isna() & fc["forecast"].notna()]
    assert len(fut) == 6


def test_run_naive_forecast_fact_not_extended(monthly_ts):
    """Fact line must not extend past last actual date."""
    result = run_naive_forecast(monthly_ts, "date", "value", horizon=12)
    fc = result.forecast_df
    last_actual_date = fc[fc["actual"].notna()]["date"].max()
    last_data_date = monthly_ts["date"].max()
    assert last_actual_date == last_data_date


def test_run_naive_forecast_metrics(monthly_ts):
    result = run_naive_forecast(monthly_ts, "date", "value", horizon=6)
    assert "MAE" in result.metrics
    assert not np.isnan(result.metrics["MAE"])


# ---------------------------------------------------------------------------
# run_arx_forecast
# ---------------------------------------------------------------------------

def test_run_arx_forecast_basic(monthly_ts):
    result = run_arx_forecast(monthly_ts, "date", "value", lags=[1, 2, 12], horizon=6)
    assert isinstance(result, ForecastResult)
    assert "ARX" in result.model_name
    fut = result.forecast_df[result.forecast_df["actual"].isna() & result.forecast_df["forecast"].notna()]
    assert len(fut) == 6


def test_run_arx_forecast_coef_table(monthly_ts):
    result = run_arx_forecast(monthly_ts, "date", "value", lags=[1, 2], horizon=3)
    assert result.explainability is not None
    assert "feature" in result.explainability.columns
    assert "coefficient" in result.explainability.columns


def test_run_arx_forecast_with_exog(monthly_ts):
    monthly_ts["exog"] = np.random.normal(0, 1, len(monthly_ts))
    result = run_arx_forecast(monthly_ts, "date", "value",
                              exog_cols=["exog"], lags=[1, 2], horizon=3)
    assert isinstance(result, ForecastResult)


def test_run_arx_forecast_metrics_not_nan(monthly_ts):
    result = run_arx_forecast(monthly_ts, "date", "value", lags=[1, 12], horizon=6)
    for v in result.metrics.values():
        assert not np.isnan(v), f"Metric is NaN: {result.metrics}"


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def test_rolling_backtest_runs(monthly_ts):
    fold_results, summary = rolling_backtest(
        monthly_ts, "date", "value",
        model_fn=run_naive_forecast,
        n_folds=2, min_train=12, horizon=3,
        seasonal=True, period=12,
    )
    assert len(fold_results) >= 1
    assert not summary.empty
    assert "MAE" in summary.columns


def test_rolling_backtest_arx(monthly_ts):
    fold_results, summary = rolling_backtest(
        monthly_ts, "date", "value",
        model_fn=run_arx_forecast,
        n_folds=2, min_train=18, horizon=3,
        lags=[1, 2, 12],
    )
    assert not summary.empty


# ---------------------------------------------------------------------------
# ForecastResult dataclass
# ---------------------------------------------------------------------------

def test_forecast_result_defaults():
    fc_df = pd.DataFrame({"date": [], "actual": [], "forecast": []})
    res = ForecastResult(model_name="Test", forecast_df=fc_df)
    assert res.model_name == "Test"
    assert res.metrics == {}
    assert res.backtest_details == []
    assert res.explainability is None

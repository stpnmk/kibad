"""
core/models.py – Time series forecasting models with backtesting and explainability.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (skips zero actuals)."""
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Bias (signed mean error)."""
    return float(np.mean(predicted - actual))


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (avoids MAPE blow-up near zero)."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    mask = denom > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def compute_all_metrics(
    actual: np.ndarray | pd.Series,
    predicted: np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute MAE, RMSE, MAPE or sMAPE, Bias for a forecast.

    Uses sMAPE instead of MAPE when min(|actual|) < 5% of mean(|actual|)
    to avoid distortion from near-zero values.

    Parameters
    ----------
    actual, predicted:
        Arrays of equal length.

    Returns
    -------
    dict[str, float]
    """
    a = np.array(actual, dtype=float)
    p = np.array(predicted, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(p))
    a, p = a[mask], p[mask]
    mean_abs = np.abs(a).mean() if len(a) > 0 else 0
    use_smape = mean_abs > 0 and np.abs(a).min() < 0.05 * mean_abs
    pct_metric_name = "sMAPE" if use_smape else "MAPE"
    pct_metric_val = smape(a, p) if use_smape else mape(a, p)
    return {
        "MAE": round(mae(a, p), 4),
        "RMSE": round(rmse(a, p), 4),
        pct_metric_name: round(pct_metric_val, 4),
        "Bias": round(bias(a, p), 4),
    }


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ForecastResult:
    """Container for a forecast run.

    Attributes
    ----------
    model_name : str
    forecast_df : pd.DataFrame
        Columns: ``date``, ``actual`` (NaN for future), ``forecast``,
        ``lower``, ``upper``.
    metrics : dict[str, float]
        In-sample or backtest metrics.
    explainability : pd.DataFrame | None
        Coefficient/feature-importance table.
    backtest_details : list[dict]
        Per-fold backtest metrics.
    notes : str
    """
    model_name: str
    forecast_df: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)
    explainability: pd.DataFrame | None = None
    backtest_details: list[dict] = field(default_factory=list)
    notes: str = ""


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

class NaiveForecast:
    """Seasonal naive or plain naive forecast.

    Parameters
    ----------
    seasonal : bool
        If True, repeats last seasonal cycle; otherwise repeats last value.
    period : int
        Seasonal period (e.g. 12 for monthly, 52 for weekly).
    """

    def __init__(self, seasonal: bool = True, period: int = 12) -> None:
        self.seasonal = seasonal
        self.period = period
        self._history: np.ndarray | None = None

    def fit(self, y: np.ndarray) -> "NaiveForecast":
        self._history = np.array(y, dtype=float)
        return self

    def predict(self, steps: int) -> np.ndarray:
        if self._history is None:
            raise RuntimeError("Call fit() first.")
        if self.seasonal and len(self._history) >= self.period:
            tail = self._history[-self.period:]
            preds = np.tile(tail, int(np.ceil(steps / self.period)))[:steps]
        else:
            preds = np.full(steps, self._history[-1])
        return preds


def _coerce_datetime_series(dates) -> pd.Series:
    """Coerce a date-like input to ``datetime64[ns]`` **preserving length & order**.

    Use this in forecast functions where the resulting series must stay
    aligned with ``y``. Returns NaT for unparseable values rather than
    dropping them. PeriodDtype is converted via ``to_timestamp()`` (since
    ``pd.to_datetime`` errors on PeriodDtype).
    """
    if isinstance(dates, pd.DatetimeIndex):
        s = pd.Series(dates)
    elif isinstance(dates, pd.Series):
        s = dates
    else:
        s = pd.Series(list(dates))

    if isinstance(s.dtype, pd.PeriodDtype):
        s = s.dt.to_timestamp()
    else:
        s = pd.to_datetime(s, errors="coerce")

    if isinstance(s.dtype, pd.DatetimeTZDtype):
        s = s.dt.tz_localize(None)

    return s


def _ensure_datetime(dates) -> pd.Series:
    """Like ``_coerce_datetime_series`` but also drops NaT and sorts.

    Use only when length/order need not stay aligned with another array
    (e.g. inside ``_future_dates``).
    """
    return _coerce_datetime_series(dates).dropna().sort_values().reset_index(drop=True)


def _future_dates(dates, horizon: int) -> pd.DatetimeIndex:
    """Generate ``horizon`` future timestamps after the last date.

    Always uses ``Timedelta``- or ``DateOffset``-based arithmetic — never
    ``Timestamp + int`` (deprecated since pandas 2.x:
    "Addition/subtraction of integers and integer-arrays with Timestamp is no
    longer supported. Instead of adding/subtracting `n`, use `n * obj.freq`").

    Strategy:
      1. ``pd.infer_freq`` on a regular series → ``pd.date_range`` with the
         alias-mapped frequency (handles ME/QE/YE/W/D/H/min etc.).
      2. Irregular series → step from ``Timedelta`` median of observed gaps.
      3. Defensive fallback: 1-day step if median gap is NaT / zero / missing.
    """
    horizon = int(horizon)
    if horizon <= 0:
        return pd.DatetimeIndex([])

    dates = _ensure_datetime(dates)
    if len(dates) == 0:
        # No usable dates; return a placeholder daily index from "now".
        return pd.date_range(pd.Timestamp.now().normalize(),
                             periods=horizon, freq="D")

    last = pd.Timestamp(dates.iloc[-1])

    # ── (1) regular cadence via inferred frequency ──────────────────────────
    freq = pd.infer_freq(dates) if len(dates) >= 3 else None
    if freq:
        # Map deprecated aliases to modern equivalents (pandas ≥ 2.2).
        _alias_map = {"M": "ME", "Q": "QE", "A": "YE", "Y": "YE",
                      "BM": "BME", "BQ": "BQE", "BA": "BYE", "BY": "BYE"}
        freq_mapped = _alias_map.get(freq, freq)
        try:
            return pd.date_range(last, periods=horizon + 1, freq=freq_mapped)[1:]
        except Exception:
            pass  # fall through to timedelta approach

    # ── (2/3) irregular cadence: median Timedelta step ──────────────────────
    if len(dates) >= 2:
        diffs = dates.diff().dropna()
        try:
            median_step = diffs.median() if not diffs.empty else pd.NaT
        except Exception:
            median_step = pd.NaT
    else:
        median_step = pd.NaT

    if (
        not isinstance(median_step, pd.Timedelta)
        or pd.isna(median_step)
        or median_step.total_seconds() == 0
    ):
        median_step = pd.Timedelta(days=1)

    # Build via Timedelta arithmetic only (Timestamp + Timedelta is supported).
    return pd.DatetimeIndex(
        [last + median_step * int(i) for i in range(1, horizon + 1)]
    )


def run_naive_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    horizon: int = 12,
    seasonal: bool = True,
    period: int = 12,
    confidence: float = 0.95,
) -> ForecastResult:
    """Run a naive/seasonal-naive forecast.

    Parameters
    ----------
    df:
        Time series DataFrame (sorted by date).
    date_col:
        Name of the datetime column.
    target_col:
        Column to forecast.
    horizon:
        Number of future periods to predict.
    seasonal:
        Use seasonal naive instead of plain naive.
    period:
        Seasonal period.
    confidence:
        CI coverage.

    Returns
    -------
    ForecastResult
    """
    df = df.sort_values(date_col).dropna(subset=[target_col])
    y = df[target_col].values.astype(float)
    dates = _coerce_datetime_series(df[date_col])

    model = NaiveForecast(seasonal=seasonal, period=period)
    model.fit(y)
    preds = model.predict(horizon)

    # residual std for CI (exclude NaN predictions to avoid bias)
    in_sample = model.predict(len(y))[-len(y):]
    valid_mask = ~np.isnan(in_sample)
    if valid_mask.any():
        resid = y[valid_mask] - in_sample[valid_mask]
    else:
        resid = np.zeros(len(y))  # no valid in-sample: use zero residuals → wide CI
    resid_std = float(np.nanstd(resid))
    z = float(_norm.ppf(1 - (1 - confidence) / 2))
    # Widen CI by sqrt(h) for each forecast step to account for error accumulation
    h_factors = np.sqrt(np.arange(1, horizon + 1))
    lower = preds - z * resid_std * h_factors
    upper = preds + z * resid_std * h_factors

    future_dates = _future_dates(dates, horizon)

    hist_df = pd.DataFrame({
        "date": dates.values,
        "actual": y,
        "forecast": np.nan,
        "lower": np.nan,
        "upper": np.nan,
    })
    fut_df = pd.DataFrame({
        "date": future_dates,
        "actual": np.nan,
        "forecast": preds,
        "lower": lower,
        "upper": upper,
    })
    forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)
    metrics = compute_all_metrics(y, model.predict(len(y)))

    return ForecastResult(
        model_name="Seasonal Naive" if seasonal else "Naive",
        forecast_df=forecast_df,
        metrics=metrics,
        notes=f"Period={period}, Horizon={horizon}",
    )


# ---------------------------------------------------------------------------
# ARX (Ridge regression with lags + exogenous variables)
# ---------------------------------------------------------------------------

def _build_arx_features(
    y: pd.Series,
    exog: pd.DataFrame | None,
    lags: list[int],
    date_col: str | None = None,
    add_time_features: bool = True,
) -> pd.DataFrame:
    """Build a feature matrix for ARX regression.

    Parameters
    ----------
    y:
        Target series (indexed by integer position, aligned).
    exog:
        Optional exogenous DataFrame (same length as y).
    lags:
        Lag periods to include.
    date_col:
        If provided and dates are available, add calendar features.
    add_time_features:
        Whether to add month/quarter dummies.

    Returns
    -------
    pd.DataFrame
    """
    features: dict[str, Any] = {}

    # Lag features
    for lag in lags:
        features[f"lag_{lag}"] = y.shift(lag).values

    # Exogenous features
    if exog is not None:
        for col in exog.columns:
            features[f"exog_{col}"] = exog[col].values

    return pd.DataFrame(features, index=y.index)


class ARXForecaster:
    """ARX model: ridge regression with lag features + exogenous variables.

    Parameters
    ----------
    lags:
        List of autoregressive lag periods.
    alpha:
        Ridge regularisation strength.
    """

    def __init__(self, lags: list[int] | None = None, alpha: float = 1.0) -> None:
        self.lags = lags or [1, 2, 3, 12]
        self.alpha = alpha
        self._model: Ridge | None = None
        self._feature_names: list[str] = []
        self._y_train: pd.Series | None = None
        self._exog_train: pd.DataFrame | None = None
        self._min_lag = max(self.lags)

    def fit(
        self,
        y: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> "ARXForecaster":
        """Fit the model.

        Parameters
        ----------
        y:
            Target time series.
        exog:
            Exogenous features aligned with y.

        Returns
        -------
        ARXForecaster
        """
        self._y_train = y.reset_index(drop=True)
        self._exog_train = exog.reset_index(drop=True) if exog is not None else None

        feat_df = _build_arx_features(self._y_train, self._exog_train, self.lags)
        valid_mask = ~feat_df.isnull().any(axis=1)
        X = feat_df[valid_mask].values
        Y = self._y_train[valid_mask].values
        self._feature_names = feat_df.columns.tolist()

        self._model = Ridge(alpha=self.alpha)
        self._model.fit(X, Y)
        return self

    def predict_in_sample(self) -> np.ndarray:
        """Return in-sample fitted values."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        feat_df = _build_arx_features(self._y_train, self._exog_train, self.lags)
        preds = np.full(len(self._y_train), np.nan)
        valid_mask = ~feat_df.isnull().any(axis=1)
        X = feat_df[valid_mask].values
        preds[valid_mask] = self._model.predict(X)
        return preds

    def predict(
        self,
        steps: int,
        future_exog: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Forecast future steps recursively.

        Parameters
        ----------
        steps:
            Number of future periods.
        future_exog:
            Exogenous features for the forecast horizon (length == steps).

        Returns
        -------
        np.ndarray of length ``steps``.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        history = list(self._y_train.values)
        preds = []

        for i in range(steps):
            row: dict[str, float] = {}
            for lag in self.lags:
                idx = len(history) - lag
                row[f"lag_{lag}"] = history[idx] if idx >= 0 else np.nan

            if future_exog is not None:
                for col in future_exog.columns:
                    row[f"exog_{col}"] = float(future_exog.iloc[i][col])

            # Align feature order
            x = np.array([row.get(f, np.nan) for f in self._feature_names]).reshape(1, -1)
            if np.isnan(x).any():
                pred = float(np.nanmean(history[-self._min_lag:])) if history else 0.0
            else:
                pred = float(self._model.predict(x)[0])
            history.append(pred)
            preds.append(pred)
        return np.array(preds)

    @property
    def coef_table(self) -> pd.DataFrame:
        """Return a DataFrame of feature names and their coefficients."""
        if self._model is None:
            return pd.DataFrame()
        return pd.DataFrame({
            "feature": self._feature_names,
            "coefficient": self._model.coef_.round(4),
        }).sort_values("coefficient", key=np.abs, ascending=False)


def run_arx_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str] | None = None,
    lags: list[int] | None = None,
    horizon: int = 12,
    alpha: float = 1.0,
    confidence: float = 0.95,
) -> ForecastResult:
    """Run ARX forecast.

    Parameters
    ----------
    df:
        Sorted time series DataFrame.
    date_col:
        Date column name.
    target_col:
        Target column name.
    exog_cols:
        Optional exogenous column names.
    lags:
        Lag list; defaults to [1, 2, 3, 12].
    horizon:
        Number of future periods.
    alpha:
        Ridge regularisation.
    confidence:
        CI coverage.

    Returns
    -------
    ForecastResult
    """
    df = df.sort_values(date_col).dropna(subset=[target_col])
    y = df[target_col].reset_index(drop=True).astype(float)
    dates = _coerce_datetime_series(df[date_col])

    exog = None
    future_exog = None
    if exog_cols:
        exog_cols_valid = [c for c in exog_cols if c in df.columns]
        if exog_cols_valid:
            exog = df[exog_cols_valid].reset_index(drop=True)
            # For future exog, use last known values (constant extrapolation)
            last_row = exog.iloc[-1:].values
            future_exog_arr = np.tile(last_row, (horizon, 1))
            future_exog = pd.DataFrame(future_exog_arr, columns=exog_cols_valid)

    lag_list = lags or [1, 2, 3, 12]
    model = ARXForecaster(lags=lag_list, alpha=alpha)
    model.fit(y, exog=exog)

    in_sample_preds = model.predict_in_sample()
    future_preds = model.predict(horizon, future_exog=future_exog)

    valid_mask = ~np.isnan(in_sample_preds)
    resid = y.values[valid_mask] - in_sample_preds[valid_mask] if valid_mask.any() else y.values - in_sample_preds
    resid_std = float(np.nanstd(resid))
    z = float(_norm.ppf(1 - (1 - confidence) / 2))

    future_dates = _future_dates(dates, horizon)

    hist_df = pd.DataFrame({
        "date": dates.values,
        "actual": y.values,
        "forecast": in_sample_preds,
        "lower": in_sample_preds - z * resid_std,
        "upper": in_sample_preds + z * resid_std,
    })
    # CI grows with sqrt(h) — each step accumulates forecast error
    h_factors = np.sqrt(np.arange(1, horizon + 1))
    fut_df = pd.DataFrame({
        "date": future_dates,
        "actual": np.nan,
        "forecast": future_preds,
        "lower": future_preds - z * resid_std * h_factors,
        "upper": future_preds + z * resid_std * h_factors,
    })
    forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)

    valid_mask = ~np.isnan(in_sample_preds)
    metrics = compute_all_metrics(y.values[valid_mask], in_sample_preds[valid_mask])

    return ForecastResult(
        model_name="ARX (Ridge)",
        forecast_df=forecast_df,
        metrics=metrics,
        explainability=model.coef_table,
        notes=f"Lags={lag_list}, Exog={exog_cols or []}, Horizon={horizon}",
    )


# ---------------------------------------------------------------------------
# SARIMAX
# ---------------------------------------------------------------------------

def run_sarimax_forecast(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    exog_cols: list[str] | None = None,
    order: tuple[int, int, int] = (1, 1, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 0, 1, 12),
    horizon: int = 12,
    confidence: float = 0.95,
) -> ForecastResult:
    """Run SARIMAX forecast via statsmodels.

    Parameters
    ----------
    df:
        Sorted time series DataFrame.
    date_col:
        Date column name.
    target_col:
        Target column name.
    exog_cols:
        Optional exogenous columns.
    order:
        (p, d, q) ARIMA order.
    seasonal_order:
        (P, D, Q, s) seasonal order.
    horizon:
        Number of future periods.
    confidence:
        CI coverage.

    Returns
    -------
    ForecastResult
    """
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        raise ImportError("statsmodels is required for SARIMAX.")

    df = df.sort_values(date_col).dropna(subset=[target_col])
    y = df[target_col].astype(float).values
    dates = _coerce_datetime_series(df[date_col])

    exog_train = None
    future_exog = None
    if exog_cols:
        valid_exog = [c for c in exog_cols if c in df.columns]
        if valid_exog:
            # Encode categorical columns safely via one-hot encoding
            exog_df = df[valid_exog].copy()
            cat_cols = [c for c in exog_df.columns
                        if exog_df[c].dtype == object or str(exog_df[c].dtype) == "category"]
            num_exog_cols = [c for c in exog_df.columns if c not in cat_cols]
            # One-hot encode categoricals; drop first to avoid multicollinearity
            if cat_cols:
                exog_df[cat_cols] = exog_df[cat_cols].astype(str).fillna("__missing__")
                exog_df = pd.get_dummies(exog_df, columns=cat_cols, drop_first=True,
                                         dtype=float)
            # Fill numeric NaNs with 0
            exog_df = exog_df.fillna(0).astype(float)
            exog_train = exog_df.values
            # For future, use last known row
            last_row = exog_df.iloc[-1:].values
            future_exog = np.tile(last_row, (horizon, 1))

    try:
        model = SARIMAX(
            y,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=200)
    except Exception as exc:
        raise RuntimeError(f"SARIMAX fitting failed: {exc}") from exc

    fitted = res.fittedvalues
    alpha_ci = 1.0 - confidence
    forecast_res = res.get_forecast(steps=horizon, exog=future_exog)
    mean_forecast = forecast_res.predicted_mean
    ci = forecast_res.conf_int(alpha=alpha_ci)

    future_dates = _future_dates(dates, horizon)

    hist_df = pd.DataFrame({
        "date": dates.values,
        "actual": y,
        "forecast": fitted,
        "lower": np.nan,
        "upper": np.nan,
    })
    fut_df = pd.DataFrame({
        "date": future_dates,
        "actual": np.nan,
        "forecast": mean_forecast,
        "lower": ci[:, 0] if isinstance(ci, np.ndarray) else ci.iloc[:, 0].values,
        "upper": ci[:, 1] if isinstance(ci, np.ndarray) else ci.iloc[:, 1].values,
    })
    forecast_df = pd.concat([hist_df, fut_df], ignore_index=True)

    valid_mask = ~np.isnan(fitted)
    metrics = compute_all_metrics(y[valid_mask], fitted[valid_mask])

    # AIC/BIC and parameter summary
    try:
        param_df = pd.DataFrame({
            "param": res.param_names,
            "coefficient": res.params.round(4),
            "std_err": res.bse.round(4),
            "p_value": res.pvalues.round(4),
        })
    except Exception:
        param_df = None

    return ForecastResult(
        model_name="SARIMAX",
        forecast_df=forecast_df,
        metrics=metrics,
        explainability=param_df,
        notes=f"order={order}, seasonal={seasonal_order}, AIC={res.aic:.1f}",
    )


# ---------------------------------------------------------------------------
# Rolling backtest
# ---------------------------------------------------------------------------

def rolling_backtest(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    model_fn,
    n_folds: int = 3,
    min_train: int = 24,
    horizon: int = 6,
    **model_kwargs: Any,
) -> tuple[list[ForecastResult], pd.DataFrame]:
    """Rolling-window backtest for any model function.

    Parameters
    ----------
    df:
        Full time series DataFrame sorted by date.
    date_col:
        Date column.
    target_col:
        Target column.
    model_fn:
        A callable like ``run_arx_forecast`` or ``run_naive_forecast``.
    n_folds:
        Number of rolling folds to evaluate.
    min_train:
        Minimum training periods.
    horizon:
        Forecast horizon per fold.
    **model_kwargs:
        Extra kwargs forwarded to ``model_fn``.

    Returns
    -------
    tuple[list[ForecastResult], pd.DataFrame]
        (list of fold results, summary metrics DataFrame)
    """
    df = df.sort_values(date_col).dropna(subset=[target_col]).reset_index(drop=True)
    n = len(df)
    results = []
    all_metrics = []

    step = max(1, (n - min_train - horizon) // max(1, n_folds))
    fold_starts = range(min_train, n - horizon, step)

    for fold_idx, split_idx in enumerate(fold_starts):
        if fold_idx >= n_folds:
            break
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:split_idx + horizon]

        if len(test_df) == 0:
            continue

        try:
            res = model_fn(train_df, date_col=date_col, target_col=target_col,
                           horizon=len(test_df), **model_kwargs)
        except Exception as exc:
            continue

        fut = res.forecast_df[res.forecast_df["actual"].isna()].iloc[:len(test_df)]
        actual_vals = test_df[target_col].values
        pred_vals = fut["forecast"].values[:len(actual_vals)]

        if len(actual_vals) > 0 and len(pred_vals) > 0:
            m = compute_all_metrics(actual_vals, pred_vals)
            m["fold"] = fold_idx + 1
            m["train_size"] = split_idx
            m["test_size"] = len(test_df)
            all_metrics.append(m)
            results.append(res)

    summary = pd.DataFrame(all_metrics)
    return results, summary


# ---------------------------------------------------------------------------
# Forecast comparison chart helper
# ---------------------------------------------------------------------------

def forecast_chart_data(results: list[ForecastResult]) -> dict[str, pd.DataFrame]:
    """Extract per-model forecast DataFrames for plotting.

    Parameters
    ----------
    results:
        List of ForecastResult objects.

    Returns
    -------
    dict mapping model name → forecast_df.
    """
    return {r.model_name: r.forecast_df for r in results}


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(
    series: "pd.Series",
    method: str = "rolling_zscore",
    window: int = 12,
    threshold: float = 3.0,
    period: int = 12,
) -> "pd.DataFrame":
    """
    Detect anomalies in a time series.

    Parameters
    ----------
    series : pd.Series with numeric values
    method : 'rolling_zscore' or 'stl_residual'
    window : rolling window size (for rolling_zscore)
    threshold : number of std deviations to flag as anomaly
    period : seasonal period (for stl_residual)

    Returns
    -------
    pd.DataFrame with columns: value, rolling_mean, rolling_std,
        z_score, is_anomaly, upper, lower
    """
    import pandas as pd
    import numpy as np

    s = series.reset_index(drop=True)
    result = pd.DataFrame({"value": s})

    if method == "rolling_zscore":
        rolling_mean = s.rolling(window=window, center=True, min_periods=1).mean()
        rolling_std = s.rolling(window=window, center=True, min_periods=1).std().fillna(1.0)
        z_score = (s - rolling_mean) / rolling_std.replace(0, 1)
        result["rolling_mean"] = rolling_mean
        result["rolling_std"] = rolling_std
        result["z_score"] = z_score
        result["is_anomaly"] = z_score.abs() > threshold
        result["upper"] = rolling_mean + threshold * rolling_std
        result["lower"] = rolling_mean - threshold * rolling_std
        result["severity"] = z_score.abs() / threshold

    elif method == "stl_residual":
        try:
            from statsmodels.tsa.seasonal import STL
            stl = STL(s, period=period, robust=True)
            stl_fit = stl.fit()
            residuals = pd.Series(stl_fit.resid, index=s.index)
            res_std = residuals.std()
            res_mean = residuals.mean()
            z_score = (residuals - res_mean) / (res_std if res_std > 0 else 1.0)
            result["trend"] = stl_fit.trend
            result["seasonal"] = stl_fit.seasonal
            result["residual"] = residuals
            result["z_score"] = z_score
            result["is_anomaly"] = z_score.abs() > threshold
            result["upper"] = stl_fit.trend + stl_fit.seasonal + res_mean + threshold * res_std
            result["lower"] = stl_fit.trend + stl_fit.seasonal + res_mean - threshold * res_std
            result["severity"] = z_score.abs() / threshold
        except Exception:
            # Fallback to rolling_zscore
            return detect_anomalies(series, method="rolling_zscore",
                                    window=window, threshold=threshold)
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'rolling_zscore' or 'stl_residual'.")

    return result


# ---------------------------------------------------------------------------
# STL Decomposition
# ---------------------------------------------------------------------------

@dataclass
class STLDecompResult:
    """Container for STL decomposition output.

    Attributes
    ----------
    dates : pd.DatetimeIndex
    observed : np.ndarray
    trend : np.ndarray
    seasonal : np.ndarray
    residual : np.ndarray
    seasonality_strength : float
        Fs = max(0, 1 - Var(R) / Var(S+R)). >0.6 = strong seasonality.
    model_type : str
        'additive' or 'multiplicative'.
    """
    dates: pd.DatetimeIndex
    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    seasonality_strength: float
    model_type: str


def run_stl_decomposition(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    period: int = 12,
    robust: bool = True,
    multiplicative: bool = False,
) -> STLDecompResult:
    """Decompose a time series using STL (Seasonal and Trend decomposition using Loess).

    Parameters
    ----------
    df : pd.DataFrame
        Time series data.
    date_col : str
        Date column name.
    target_col : str
        Target column name.
    period : int
        Seasonal period (12=monthly, 4=quarterly, 52=weekly).
    robust : bool
        Use robust LOESS fitting (reduces outlier influence).
    multiplicative : bool
        If True, apply log-transform before STL for multiplicative decomposition.
        Requires all y > 0.

    Returns
    -------
    STLDecompResult
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        raise ImportError("statsmodels required for STL decomposition.")

    df = df.sort_values(date_col).dropna(subset=[target_col])
    y = df[target_col].astype(float).values
    dates = _coerce_datetime_series(df[date_col])

    if multiplicative:
        if np.any(y <= 0):
            raise ValueError(
                "Мультипликативная модель требует y > 0 для log-преобразования. "
                "Обнаружены нулевые или отрицательные значения."
            )
        y_fit = np.log(y)
    else:
        y_fit = y

    stl = STL(y_fit, period=period, robust=robust)
    res = stl.fit()

    # Seasonality strength on transformed scale (per vault formula)
    var_r = np.var(res.resid)
    var_sr = np.var(res.seasonal + res.resid)
    fs = float(max(0.0, 1.0 - var_r / var_sr)) if var_sr > 0 else 0.0

    if multiplicative:
        # Back-transform trend and seasonal; recompute residual on original scale
        trend = np.exp(res.trend)
        seasonal = np.exp(res.seasonal)
        residual = y - trend * seasonal
    else:
        trend = res.trend
        seasonal = res.seasonal
        residual = res.resid

    return STLDecompResult(
        dates=dates,
        observed=y,
        trend=trend,
        seasonal=seasonal,
        residual=residual,
        seasonality_strength=fs,
        model_type="multiplicative" if multiplicative else "additive",
    )


# ---------------------------------------------------------------------------
# Residual Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticsResult:
    """Residual diagnostics for a fitted forecast model.

    Attributes
    ----------
    model_name : str
    residuals : np.ndarray
    fitted : np.ndarray
    ljung_box : pd.DataFrame
        Columns lb_stat, lb_pvalue for lags [10, 20].
    acf_residuals : np.ndarray
    ci_bound : float
        95% significance boundary = 1.96 / sqrt(n).
    """
    model_name: str
    residuals: np.ndarray
    fitted: np.ndarray
    ljung_box: pd.DataFrame
    acf_residuals: np.ndarray
    ci_bound: float


def compute_residual_diagnostics(result: ForecastResult) -> DiagnosticsResult:
    """Compute residual diagnostics for a ForecastResult.

    Runs Ljung-Box test and ACF of residuals.

    Parameters
    ----------
    result : ForecastResult
        A fitted forecast (from run_naive_forecast, run_arx_forecast, etc.).

    Returns
    -------
    DiagnosticsResult
    """
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        from statsmodels.tsa.stattools import acf
    except ImportError:
        raise ImportError("statsmodels required for residual diagnostics.")

    fd = result.forecast_df
    hist = fd[fd["actual"].notna()].copy()

    actual = hist["actual"].values.astype(float)
    fitted_vals = hist["forecast"].values.astype(float)

    valid = ~(np.isnan(actual) | np.isnan(fitted_vals))
    actual = actual[valid]
    fitted_vals = fitted_vals[valid]
    residuals = actual - fitted_vals

    n = len(residuals)
    max_lag = min(20, n // 2 - 1)
    lags = [l for l in [10, 20] if l <= max_lag]
    if not lags:
        lags = [max(1, max_lag)]

    lb_result = acorr_ljungbox(residuals, lags=lags, return_df=True)

    nlags = min(30, n // 2 - 1)
    acf_vals = acf(residuals, nlags=max(1, nlags), fft=True)
    ci_bound = 1.96 / np.sqrt(n) if n > 0 else 0.0

    return DiagnosticsResult(
        model_name=result.model_name,
        residuals=residuals,
        fitted=fitted_vals,
        ljung_box=lb_result,
        acf_residuals=acf_vals,
        ci_bound=ci_bound,
    )

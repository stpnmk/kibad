"""
core/explore.py – Exploratory analysis and Plotly chart builders.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
PALETTE = px.colors.qualitative.Plotly


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

def plot_timeseries(
    df: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    color_col: str | None = None,
    title: str = "Time Series",
    yaxis_title: str = "Value",
) -> go.Figure:
    """Line chart for one or more time-series columns.

    Parameters
    ----------
    df:
        DataFrame with the data.
    date_col:
        Column name for x-axis.
    value_cols:
        Columns to plot as separate lines.
    color_col:
        If provided, each unique value gets its own line (long-format).
    title:
        Chart title.
    yaxis_title:
        Y-axis label.

    Returns
    -------
    go.Figure
    """
    fig = go.Figure()

    if color_col and color_col in df.columns:
        for i, group in enumerate(df[color_col].unique()):
            sub = df[df[color_col] == group].sort_values(date_col)
            for col in value_cols:
                fig.add_trace(go.Scatter(
                    x=sub[date_col], y=sub[col],
                    mode="lines+markers",
                    name=f"{group} – {col}",
                    line=dict(color=PALETTE[i % len(PALETTE)]),
                ))
    else:
        for i, col in enumerate(value_cols):
            sub = df.sort_values(date_col)
            fig.add_trace(go.Scatter(
                x=sub[date_col], y=sub[col],
                mode="lines+markers",
                name=col,
                line=dict(color=PALETTE[i % len(PALETTE)]),
            ))

    fig.update_layout(
        title=title,
        xaxis_title=date_col,
        yaxis_title=yaxis_title,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------

def plot_histogram(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    color_col: str | None = None,
    kde: bool = True,
    title: str | None = None,
) -> go.Figure:
    """Histogram with optional KDE overlay.

    Parameters
    ----------
    df:
        DataFrame.
    col:
        Numeric column to plot.
    bins:
        Number of bins.
    color_col:
        Optional grouping column for faceted histograms.
    kde:
        Whether to overlay a KDE curve.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if color_col and color_col in df.columns:
        fig = px.histogram(
            df, x=col, color=color_col, nbins=bins,
            barmode="overlay", opacity=0.7,
            title=title or f"Distribution of {col}",
        )
    else:
        fig = px.histogram(
            df, x=col, nbins=bins,
            title=title or f"Distribution of {col}",
        )

    if kde:
        vals = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(vals) >= 5:
            from scipy.stats import gaussian_kde
            kde_func = gaussian_kde(vals, bw_method="scott")
            x_range = np.linspace(vals.min(), vals.max(), 200)
            # scale KDE to histogram counts
            n_bins = bins
            bin_width = (vals.max() - vals.min()) / n_bins if n_bins > 0 else 1
            scale = len(vals) * bin_width
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_func(x_range) * scale,
                mode="lines", name="KDE",
                line=dict(color="crimson", width=2, dash="dash"),
            ))
    fig.update_layout(template="plotly_white")
    return fig


def plot_boxplot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Box-and-whisker plot, optionally grouped.

    Parameters
    ----------
    df:
        DataFrame.
    value_col:
        Numeric column.
    group_col:
        Optional categorical column for grouping.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if group_col and group_col in df.columns:
        fig = px.box(
            df, x=group_col, y=value_col,
            title=title or f"{value_col} by {group_col}",
            color=group_col,
        )
    else:
        fig = px.box(
            df, y=value_col,
            title=title or f"Box Plot: {value_col}",
        )
    fig.update_layout(template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Correlation heatmap
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    method: str = "pearson",
    title: str = "Correlation Heatmap",
) -> go.Figure:
    """Correlation heatmap for numeric columns.

    Parameters
    ----------
    df:
        DataFrame.
    columns:
        Subset of numeric columns; defaults to all numeric columns.
    method:
        ``"pearson"``, ``"spearman"``, or ``"kendall"``.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    num_df = df.select_dtypes(include="number")
    if columns:
        num_df = num_df[[c for c in columns if c in num_df.columns]]
    if num_df.empty or num_df.shape[1] < 2:
        fig = go.Figure()
        fig.update_layout(title="Not enough numeric columns for correlation.")
        return fig

    corr = num_df.corr(method=method).round(3)
    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        colorbar=dict(title="r"),
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_tickangle=-45,
    )
    return fig


# ---------------------------------------------------------------------------
# Pivot aggregation builder
# ---------------------------------------------------------------------------

def build_pivot(
    df: pd.DataFrame,
    index_col: str,
    columns_col: str | None,
    value_col: str,
    agg_func: str = "sum",
) -> pd.DataFrame:
    """Build a pivot table.

    Parameters
    ----------
    df:
        Source DataFrame.
    index_col:
        Row grouping column.
    columns_col:
        Optional column grouping; if None, aggregates only by index.
    value_col:
        Column to aggregate.
    agg_func:
        Aggregation function name.

    Returns
    -------
    pd.DataFrame
    """
    if columns_col and columns_col in df.columns:
        pivot = pd.pivot_table(
            df,
            index=index_col,
            columns=columns_col,
            values=value_col,
            aggfunc=agg_func,
            fill_value=0,
        )
        pivot.columns = [str(c) for c in pivot.columns]
        return pivot.reset_index()
    else:
        grouped = df.groupby(index_col, as_index=False)[value_col].agg(agg_func)
        return grouped


def plot_pivot_bar(
    pivot_df: pd.DataFrame,
    index_col: str,
    value_cols: list[str] | None = None,
    title: str = "Aggregated View",
    barmode: str = "group",
) -> go.Figure:
    """Bar chart from a pivot DataFrame.

    Parameters
    ----------
    pivot_df:
        Result of :func:`build_pivot`.
    index_col:
        Column to use as x-axis.
    value_cols:
        Columns to plot as bars; defaults to all non-index columns.
    title:
        Chart title.
    barmode:
        ``"group"`` or ``"stack"``.

    Returns
    -------
    go.Figure
    """
    cols = value_cols or [c for c in pivot_df.columns if c != index_col]
    fig = go.Figure()
    for i, col in enumerate(cols):
        fig.add_trace(go.Bar(
            x=pivot_df[index_col],
            y=pivot_df[col],
            name=col,
            marker_color=PALETTE[i % len(PALETTE)],
        ))
    fig.update_layout(
        title=title,
        barmode=barmode,
        template="plotly_white",
        xaxis_tickangle=-30,
        legend=dict(orientation="h", y=-0.3),
    )
    return fig


# ---------------------------------------------------------------------------
# Waterfall chart
# ---------------------------------------------------------------------------

def plot_waterfall(
    categories: list[str],
    values: list[float],
    title: str = "Factor Contributions",
    base_label: str = "Start",
    total_label: str = "End",
) -> go.Figure:
    """Waterfall chart showing factual delta contributions.

    Parameters
    ----------
    categories:
        Names of contributing factors.
    values:
        Numeric contribution of each factor (positive or negative).
    title:
        Chart title.
    base_label:
        Label for the starting total bar.
    total_label:
        Label for the ending total bar.

    Returns
    -------
    go.Figure
    """
    measures = ["relative"] * len(values)
    all_x = [base_label] + categories + [total_label]
    all_y = [0.0] + list(values) + [0.0]
    all_m = ["absolute"] + measures + ["total"]

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=all_m,
        x=all_x,
        y=all_y,
        connector=dict(line=dict(color="lightgrey")),
        increasing=dict(marker_color="seagreen"),
        decreasing=dict(marker_color="tomato"),
        totals=dict(marker_color="steelblue"),
    ))
    fig.update_layout(
        title=title,
        template="plotly_white",
        showlegend=False,
        yaxis_title="Delta",
    )
    return fig


# ---------------------------------------------------------------------------
# STL decomposition plot
# ---------------------------------------------------------------------------

def plot_stl_decomposition(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    period: int = 12,
    title: str | None = None,
) -> go.Figure:
    """STL decomposition chart (trend, seasonal, residual).

    Parameters
    ----------
    df:
        Sorted time series DataFrame.
    date_col:
        Date column.
    value_col:
        Target column.
    period:
        Seasonal period (12 for monthly, 52 for weekly).
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        fig = go.Figure()
        fig.update_layout(title="statsmodels not available.")
        return fig

    sub = df[[date_col, value_col]].dropna().sort_values(date_col)
    series = sub.set_index(date_col)[value_col].asfreq(None)

    if len(series) < 2 * period:
        fig = go.Figure()
        fig.update_layout(title=f"Not enough data for STL (need ≥ {2*period} points).")
        return fig

    stl = STL(series, period=period, robust=True)
    res = stl.fit()

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=[
            title or f"STL: {value_col}",
            "Trend", "Seasonal", "Residual",
        ],
        vertical_spacing=0.08,
    )
    dates = sub[date_col]
    fig.add_trace(go.Scatter(x=dates, y=series.values, name="Observed", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=res.trend, name="Trend", line=dict(color="steelblue")), row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=res.seasonal, name="Seasonal", line=dict(color="seagreen")), row=3, col=1)
    fig.add_trace(go.Scatter(x=dates, y=res.resid, name="Residual", mode="markers",
                             marker=dict(size=3, color="grey")), row=4, col=1)
    fig.update_layout(
        height=700, template="plotly_white", showlegend=False,
        hovermode="x unified",
    )
    return fig


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def plot_violin(
    df: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    title: str | None = None,
) -> go.Figure:
    """Violin plot, optionally grouped.

    Parameters
    ----------
    df:
        DataFrame.
    value_col:
        Numeric column.
    group_col:
        Optional categorical column for grouping.
    title:
        Chart title.

    Returns
    -------
    go.Figure
    """
    if group_col and group_col in df.columns:
        fig = px.violin(
            df, x=group_col, y=value_col, box=True, points="outliers",
            color=group_col,
            title=title or f"{value_col} by {group_col}",
        )
    else:
        fig = px.violin(
            df, y=value_col, box=True, points="outliers",
            title=title or f"Violin: {value_col}",
        )
    fig.update_layout(template="plotly_white")
    return fig


# ---------------------------------------------------------------------------
# Missingness map
# ---------------------------------------------------------------------------

def plot_missingness_map(
    df: pd.DataFrame,
    title: str = "Missingness Map",
    max_rows: int = 500,
) -> go.Figure:
    """Heatmap showing NaN positions in the DataFrame.

    Parameters
    ----------
    df:
        Source DataFrame.
    title:
        Chart title.
    max_rows:
        If df has more rows, sample for readability.

    Returns
    -------
    go.Figure
    """
    sample = df.head(max_rows) if len(df) > max_rows else df
    mask = sample.isna().astype(int)

    fig = go.Figure(go.Heatmap(
        z=mask.values,
        x=mask.columns.tolist(),
        y=list(range(len(mask))),
        colorscale=[[0, "steelblue"], [1, "tomato"]],
        showscale=False,
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Column",
        yaxis_title="Row index",
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=max(300, min(len(mask) * 2, 700)),
    )
    return fig


# ---------------------------------------------------------------------------
# Isolation Forest outlier detection
# ---------------------------------------------------------------------------

def detect_outliers_isolation_forest(
    df: pd.DataFrame,
    columns: list[str],
    contamination: float = 0.05,
    random_state: int = 42,
) -> pd.Series:
    """Detect outliers using Isolation Forest.

    Parameters
    ----------
    df:
        Source DataFrame.
    columns:
        Numeric columns to use for detection.
    contamination:
        Expected fraction of outliers (0.01–0.5).
    random_state:
        RNG seed.

    Returns
    -------
    pd.Series[bool] — True where the row is an outlier.
    """
    num = df[columns].apply(pd.to_numeric, errors="coerce")
    mask = num.notna().all(axis=1)

    result = pd.Series(False, index=df.index)
    if mask.sum() < 10:
        return result

    iso = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    preds = iso.fit_predict(num[mask].values)
    result.loc[mask] = preds == -1
    return result


# ---------------------------------------------------------------------------
# ACF / PACF plots
# ---------------------------------------------------------------------------

def plot_acf_pacf(
    series: pd.Series,
    nlags: int = 24,
    title: str = "ACF / PACF",
) -> go.Figure:
    """Plot Autocorrelation (ACF) and Partial Autocorrelation (PACF).

    Parameters
    ----------
    series:
        Numeric time series (should be stationary or differenced).
    nlags:
        Number of lags to compute.
    title:
        Chart title.

    Returns
    -------
    go.Figure (2-row subplot)
    """
    from statsmodels.tsa.stattools import acf, pacf

    s = pd.to_numeric(series, errors="coerce").dropna().values
    if len(s) < 4:
        raise ValueError("Ряд слишком короткий для ACF/PACF (минимум 4 наблюдения)")
    if len(s) < nlags + 1:
        nlags = max(1, len(s) // 2)

    acf_vals = acf(s, nlags=nlags, fft=True)
    try:
        pacf_vals = pacf(s, nlags=nlags)
    except Exception:
        pacf_vals = np.zeros(nlags + 1)

    n = len(s)
    ci = 1.96 / np.sqrt(n)

    fig = make_subplots(rows=2, cols=1, subplot_titles=["ACF", "PACF"], vertical_spacing=0.15)

    lags_x = list(range(len(acf_vals)))
    # ACF
    fig.add_trace(go.Bar(x=lags_x, y=acf_vals, marker_color="steelblue", name="ACF", showlegend=False), row=1, col=1)
    fig.add_hline(y=ci, line_dash="dash", line_color="grey", row=1, col=1)
    fig.add_hline(y=-ci, line_dash="dash", line_color="grey", row=1, col=1)

    # PACF
    fig.add_trace(go.Bar(x=lags_x[:len(pacf_vals)], y=pacf_vals, marker_color="seagreen", name="PACF", showlegend=False), row=2, col=1)
    fig.add_hline(y=ci, line_dash="dash", line_color="grey", row=2, col=1)
    fig.add_hline(y=-ci, line_dash="dash", line_color="grey", row=2, col=1)

    fig.update_layout(title=title, template="plotly_white", height=500)
    return fig


# ---------------------------------------------------------------------------
# Anomaly detection (rolling z-score / STL residual)
# ---------------------------------------------------------------------------

def detect_anomalies(
    series: pd.Series,
    method: str = "rolling_zscore",
    window: int = 12,
    threshold: float = 2.5,
    period: int = 12,
) -> pd.Series:
    """Detect anomalies in a time series.

    Parameters
    ----------
    series:
        Numeric series.
    method:
        ``"rolling_zscore"`` or ``"stl_residual"``.
    window:
        Rolling window size (for rolling_zscore).
    threshold:
        Z-score threshold.
    period:
        Seasonal period (for STL method).

    Returns
    -------
    pd.Series[bool] — True where anomaly detected.
    """
    s = pd.to_numeric(series, errors="coerce")
    result = pd.Series(False, index=s.index)

    if method == "rolling_zscore":
        roll_mean = s.rolling(window, min_periods=max(3, window // 2)).mean()
        roll_std = s.rolling(window, min_periods=max(3, window // 2)).std().replace(0, np.nan)
        z_scores = ((s - roll_mean) / roll_std).abs()
        result = z_scores > threshold
        result = result.fillna(False)

    elif method == "stl_residual":
        try:
            from statsmodels.tsa.seasonal import STL
            clean = s.dropna()
            if len(clean) >= 2 * period:
                stl = STL(clean, period=period, robust=True)
                res = stl.fit()
                resid = res.resid
                resid_std = resid.std()
                if resid_std > 0:
                    z = (resid / resid_std).abs()
                    anomaly_mask = z > threshold
                    result.loc[clean.index] = anomaly_mask
        except ImportError:
            pass

    return result.astype(bool)


# ---------------------------------------------------------------------------
# KPI sparkline card data
# ---------------------------------------------------------------------------

def compute_kpi(
    df: pd.DataFrame,
    formula: str,
    label: str = "KPI",
) -> dict[str, Any]:
    """Evaluate a KPI formula on the DataFrame.

    The formula is a Python expression evaluated in a namespace where
    each column is available by name as a pandas Series.

    Parameters
    ----------
    df:
        Source DataFrame.
    formula:
        Python expression (e.g. ``"revenue / sessions"``).
    label:
        KPI name.

    Returns
    -------
    dict with keys: ``label``, ``series`` (pd.Series), ``last``, ``mean``, ``pct_change``.
    """
    namespace = {col: df[col] for col in df.columns if pd.api.types.is_numeric_dtype(df[col])}
    try:
        result = eval(formula, {"__builtins__": {}}, namespace)  # noqa: S307
    except Exception as exc:
        raise ValueError(f"KPI formula error: {exc}") from exc

    series = pd.to_numeric(result, errors="coerce")
    last_val = float(series.dropna().iloc[-1]) if not series.dropna().empty else float("nan")
    mean_val = float(series.mean())
    pct_chg = float(series.pct_change().iloc[-1]) * 100 if len(series) > 1 else float("nan")
    return {
        "label": label,
        "series": series,
        "last": round(last_val, 4),
        "mean": round(mean_val, 4),
        "pct_change": round(pct_chg, 2),
    }

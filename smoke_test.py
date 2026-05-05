"""
smoke_test.py – Minimal end-to-end smoke test for KIBAD core modules.

Verifies: load sample → profile → prepare → chart → stat test → forecast
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

print("=" * 60)
print("KIBAD Smoke Test")
print("=" * 60)

# ── 1. Load sample dataset ────────────────────────────────────
from core.data import load_csv, profile_dataframe, infer_column_types

df = load_csv("data/sample_monthly_sales.csv")
print(f"\n[1] Loaded sample_monthly_sales.csv: {df.shape[0]} rows × {df.shape[1]} cols")
assert df.shape == (60, 8), f"Unexpected shape: {df.shape}"

profile = profile_dataframe(df)
print(f"    Profile shape: {profile.shape}")
assert len(profile) == 8

types = infer_column_types(df)
print(f"    Inferred types: { {k: v for k,v in list(types.items())[:4]} } ...")

# ── 2. Data preparation ───────────────────────────────────────
from core.prepare import parse_dates, resample_timeseries, impute_missing

df = parse_dates(df, "date")
assert pd.api.types.is_datetime64_any_dtype(df["date"]), "Date parse failed"
print("\n[2] Date column parsed successfully")

df_resampled = resample_timeseries(df, "date", ["revenue", "units", "margin"],
                                   freq="MS", agg_func="sum")
print(f"    Resampled (MS sum): {df_resampled.shape}")
assert "revenue" in df_resampled.columns

# Introduce a missing value and impute
df_with_na = df.copy()
df_with_na.loc[0, "revenue"] = None
df_imputed = impute_missing(df_with_na, ["revenue"], method="median")
assert df_imputed["revenue"].isna().sum() == 0
print("    Imputation: OK (1 missing → filled)")

# ── 3. Explore – produce a chart ─────────────────────────────
from core.explore import plot_timeseries, compute_kpi

fig = plot_timeseries(df, "date", ["revenue", "units"],
                      title="Smoke Test: Revenue & Units")
assert fig is not None
print("\n[3] Time series chart created (Plotly Figure)")

kpi = compute_kpi(df, "revenue / units", label="Revenue per Unit")
assert isinstance(kpi["last"], float)
print(f"    KPI 'Revenue per Unit' last value: {kpi['last']:.4f}")

# ── 4. Statistical test ───────────────────────────────────────
from core.tests import ttest_independent, mann_whitney

segments = df["segment"].unique()
a = df[df["segment"] == segments[0]]["revenue"]
b = df[df["segment"] == segments[1]]["revenue"]

t_result = ttest_independent(a, b, label_a=segments[0], label_b=segments[1])
print(f"\n[4] t-test: statistic={t_result.statistic}, p={t_result.p_value}, "
      f"significant={t_result.significant}")
assert isinstance(t_result.p_value, float)
assert len(t_result.interpretation) > 20

mw_result = mann_whitney(a, b)
print(f"    Mann-Whitney: U={mw_result.statistic}, p={mw_result.p_value}")

# ── 5. Forecasting ────────────────────────────────────────────
from core.models import run_naive_forecast, run_arx_forecast, compute_all_metrics

naive_res = run_naive_forecast(df, "date", "revenue", horizon=6, period=12)
assert len(naive_res.forecast_df[naive_res.forecast_df["actual"].isna()]) == 6
print(f"\n[5] Seasonal Naive: MAE={naive_res.metrics['MAE']}, "
      f"MAPE={naive_res.metrics['MAPE']}%")

arx_res = run_arx_forecast(df, "date", "revenue", lags=[1, 2, 3, 12],
                            exog_cols=["fx_rate"], horizon=6)
assert arx_res.explainability is not None
print(f"    ARX (Ridge): MAE={arx_res.metrics['MAE']}, MAPE={arx_res.metrics['MAPE']}%")
print(f"    Top features: {arx_res.explainability['feature'].head(3).tolist()}")

# Auto-forecast (новый слой автоматики)
from core.timeseries_auto import run_auto_forecast
auto_res = run_auto_forecast(df, "date", "revenue")
assert auto_res.forecast.forecast_df is not None
assert not auto_res.forecast.forecast_df.empty
assert auto_res.decisions["period"] >= 1
assert len(auto_res.notes) >= 2
print(f"    Auto-forecast: model={auto_res.decisions['model']['model']}, "
      f"period={auto_res.decisions['period']}, "
      f"horizon={auto_res.decisions['horizon']}")

# Verify fact line not extended beyond last date
last_actual = naive_res.forecast_df[naive_res.forecast_df["actual"].notna()]["date"].max()
last_data = df["date"].max()
assert last_actual == last_data, "Fact line extends past last data date!"
print("    Fact line alignment: OK (no forward extension of actuals)")

# ── 6. Simulation ─────────────────────────────────────────────
from core.simulation import ScenarioPreset, run_scenario

preset = ScenarioPreset(name="FX+10%", shocks={"fx_rate": 0.10})
base, scen = run_scenario(
    df, date_col="date", target_col="revenue",
    exog_cols=["fx_rate"], lags=[1, 2, 12],
    horizon=6, preset=preset, scenario_name="FX+10%",
)
assert "baseline" in base.forecast_df.columns or "forecast" in base.forecast_df.columns
assert scen.forecast_df is not None
print(f"\n[6] Scenario simulation: baseline and scenario computed")

# ── 7. Report ─────────────────────────────────────────────────
from core.report import ReportBuilder, generate_business_summary

rb = ReportBuilder(title="Smoke Test Report", dataset_name="sample_monthly_sales", n_rows=60)
rb.add_kv_metrics("Overview", {"rows": 60, "cols": 8})
rb.add_table("Profile", profile, max_rows=5)
summary_text = generate_business_summary(df, "date", "revenue", forecast_result=naive_res)
rb.add_interpretation("Business Summary", summary_text[:300])
html = rb.render()
assert "<!DOCTYPE html>" in html
assert "Smoke Test Report" in html
print(f"\n[7] HTML report built: {len(html):,} chars")
print(f"    Business summary excerpt: {summary_text[:120]}...")

# ── All done ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED ✓")
print("=" * 60)
print("\nTo launch the app:")
print("  pip install streamlit")
print("  streamlit run app/main.py")

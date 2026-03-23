"""
data/generate_samples.py – Generate synthetic sample datasets for KIBAD demos.

Run: python data/generate_samples.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).parent
np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. Monthly sales time series (with trend, seasonality, exogenous FX rate)
# ---------------------------------------------------------------------------

dates = pd.date_range("2019-01-01", periods=60, freq="MS")
n = len(dates)
t = np.arange(n)

# Trend + seasonality + noise
trend = 1000 + 12 * t
seasonality = 200 * np.sin(2 * np.pi * t / 12)
noise = np.random.normal(0, 50, n)
units = np.round(trend + seasonality + noise).astype(int).clip(100)

fx_rate = np.round(1.1 + 0.05 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.02, n), 4)
price_per_unit = np.round(15 + 0.1 * t + np.random.normal(0, 0.5, n), 2)
revenue = np.round(units * price_per_unit * fx_rate, 2)
cost = np.round(revenue * (0.6 + np.random.normal(0, 0.02, n)), 2)
margin = np.round(revenue - cost, 2)

df_sales = pd.DataFrame({
    "date": dates,
    "units": units,
    "revenue": revenue,
    "cost": cost,
    "margin": margin,
    "fx_rate": fx_rate,
    "price_per_unit": price_per_unit,
    "segment": np.where(t < 30, "Legacy", "New"),
})
df_sales.to_csv(OUT / "sample_monthly_sales.csv", index=False)
print(f"Created sample_monthly_sales.csv  shape={df_sales.shape}")


# ---------------------------------------------------------------------------
# 2. A/B test dataset
# ---------------------------------------------------------------------------

n_ctrl = 500
n_trt = 480

control_revenue = np.round(np.random.normal(45, 12, n_ctrl).clip(1), 2)
treatment_revenue = np.round(np.random.normal(49, 13, n_trt).clip(1), 2)

control_converted = np.random.binomial(1, 0.12, n_ctrl)
treatment_converted = np.random.binomial(1, 0.155, n_trt)

df_ab = pd.DataFrame({
    "user_id": range(1, n_ctrl + n_trt + 1),
    "group": ["control"] * n_ctrl + ["treatment"] * n_trt,
    "revenue": np.concatenate([control_revenue, treatment_revenue]),
    "converted": np.concatenate([control_converted, treatment_converted]),
    "session_length_min": np.round(
        np.concatenate([
            np.random.exponential(5, n_ctrl),
            np.random.exponential(6, n_trt),
        ]), 1
    ),
    "device": np.random.choice(["mobile", "desktop", "tablet"],
                                n_ctrl + n_trt, p=[0.55, 0.35, 0.1]),
    "country": np.random.choice(["US", "UK", "DE", "FR", "AU"],
                                 n_ctrl + n_trt, p=[0.4, 0.2, 0.15, 0.15, 0.1]),
})
df_ab.to_csv(OUT / "sample_ab_test.csv", index=False)
print(f"Created sample_ab_test.csv  shape={df_ab.shape}")


# ---------------------------------------------------------------------------
# 3. Multi-segment multivariate monthly KPI dataset
# ---------------------------------------------------------------------------

segments = ["Alpha", "Beta", "Gamma"]
dates_mv = pd.date_range("2020-01-01", periods=48, freq="MS")
rows = []

for seg in segments:
    t_s = np.arange(len(dates_mv))
    base = {"Alpha": 500, "Beta": 300, "Gamma": 700}[seg]
    trend_s = base + 8 * t_s
    seas_s = 80 * np.sin(2 * np.pi * t_s / 12 + {"Alpha": 0, "Beta": 1, "Gamma": 2}[seg])
    noise_s = np.random.normal(0, 30, len(dates_mv))
    target = np.round(trend_s + seas_s + noise_s).clip(50)

    rate = np.round(0.05 + 0.01 * np.sin(2 * np.pi * t_s / 12) + np.random.normal(0, 0.003, len(dates_mv)), 4)
    new_customers = np.random.poisson(target * 0.05)
    churned = np.random.poisson(target * 0.03)
    renewed = np.round(target - new_customers + churned)

    for i, d in enumerate(dates_mv):
        rows.append({
            "date": d,
            "segment": seg,
            "volume": int(target[i]),
            "new": int(new_customers[i]),
            "renewed": int(renewed[i]),
            "churned": int(churned[i]),
            "rate": rate[i],
            "region": np.random.choice(["EMEA", "AMER", "APAC"]),
        })

df_mv = pd.DataFrame(rows)
df_mv.to_csv(OUT / "sample_multivariate.csv", index=False)
print(f"Created sample_multivariate.csv  shape={df_mv.shape}")


# ---------------------------------------------------------------------------
# 4. Factor attribution dataset (segment-level portfolio with drivers)
# ---------------------------------------------------------------------------

segments_fa = ["Retail", "Corporate", "SME"]
rows_fa = []

config_fa = {
    "Retail":    {"vol_range": (1600, 3800), "rate_range": (0.055, 0.10), "nc_range": (12, 42), "churn_range": (8, 22), "n": 9},
    "Corporate": {"vol_range": (2950, 4850), "rate_range": (0.08, 0.12), "nc_range": (15, 48), "churn_range": (14, 35), "n": 9},
    "SME":       {"vol_range": (1300, 2800), "rate_range": (0.05, 0.10), "nc_range": (10, 35), "churn_range": (6, 16), "n": 9},
}

for seg, cfg in config_fa.items():
    for _ in range(cfg["n"]):
        volume = int(np.random.uniform(*cfg["vol_range"]))
        rate = round(np.random.uniform(*cfg["rate_range"]), 3)
        new_clients = int(np.random.randint(cfg["nc_range"][0], cfg["nc_range"][1] + 1))
        churn = round(np.random.uniform(*cfg["churn_range"]), 1)

        # almIndex ~ volume * rate + new_clients - churn + noise
        almIndex = round(volume * rate + new_clients - churn + np.random.normal(0, 10), 1)
        almIndex = max(almIndex, 100.0)

        # _last values: 5-20% perturbation
        def perturb(val, lo=0.85, hi=1.15):
            return val * np.random.uniform(lo, hi)

        volume_last = int(perturb(volume, 0.9, 1.1))
        rate_last = round(perturb(rate, 0.92, 1.08), 3)
        new_clients_last = max(1, int(perturb(new_clients, 0.8, 1.2)))
        churn_last = round(perturb(churn, 0.85, 1.15), 1)
        almIndex_last = round(volume_last * rate_last + new_clients_last - churn_last + np.random.normal(0, 10), 1)
        almIndex_last = max(almIndex_last, 100.0)

        rows_fa.append({
            "segment": seg,
            "almIndex": almIndex,
            "almIndex_last": almIndex_last,
            "volume": volume,
            "volume_last": volume_last,
            "rate": rate,
            "rate_last": rate_last,
            "new_clients": new_clients,
            "new_clients_last": new_clients_last,
            "churn": churn,
            "churn_last": churn_last,
        })

df_fa = pd.DataFrame(rows_fa)
df_fa.to_csv(OUT / "sample_factor_attribution.csv", index=False)
print(f"Created sample_factor_attribution.csv  shape={df_fa.shape}")


print("\nAll sample datasets generated successfully.")

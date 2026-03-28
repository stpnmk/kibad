"""Тесты для core/matching.py — сопоставление групп."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.matching import (
    MatchResult,
    balance_summary,
    coarsened_exact_match,
    exact_match,
    nearest_neighbor_match,
    propensity_score_match,
    standardized_mean_diff,
)


# ---------------------------------------------------------------------------
# Fixture: synthetic dataset with known imbalance
# ---------------------------------------------------------------------------

@pytest.fixture()
def imbalanced_df() -> pd.DataFrame:
    """Treatment group has higher age and income on average."""
    rng = np.random.default_rng(42)
    n = 400
    treatment = np.array([1] * 150 + [0] * 250)
    age = np.where(treatment == 1, rng.normal(45, 8, n), rng.normal(35, 10, n))
    income = np.where(treatment == 1, rng.normal(70000, 15000, n), rng.normal(50000, 20000, n))
    gender = rng.choice(["M", "F"], size=n)
    region = rng.choice(["A", "B", "C"], size=n)
    return pd.DataFrame({
        "group": treatment,
        "age": age,
        "income": income,
        "gender": gender,
        "region": region,
    })


@pytest.fixture()
def balanced_df() -> pd.DataFrame:
    """Groups are already balanced."""
    rng = np.random.default_rng(7)
    n = 200
    treatment = np.array([1] * 100 + [0] * 100)
    x = rng.normal(50, 10, n)
    y = rng.normal(100, 20, n)
    return pd.DataFrame({"group": treatment, "x": x, "y": y})


# ---------------------------------------------------------------------------
# standardized_mean_diff
# ---------------------------------------------------------------------------

class TestSMD:
    def test_basic(self, imbalanced_df: pd.DataFrame):
        smd_df = standardized_mean_diff(imbalanced_df, "group", ["age", "income"])
        assert len(smd_df) == 2
        assert "smd" in smd_df.columns
        assert "abs_smd" in smd_df.columns
        # Age should show imbalance (SMD > 0.5)
        age_smd = smd_df.loc[smd_df["covariate"] == "age", "abs_smd"].iloc[0]
        assert age_smd > 0.5

    def test_balanced(self, balanced_df: pd.DataFrame):
        smd_df = standardized_mean_diff(balanced_df, "group", ["x", "y"])
        # Should have low SMD for balanced groups
        assert smd_df["abs_smd"].max() < 0.5

    def test_single_covariate(self, imbalanced_df: pd.DataFrame):
        smd_df = standardized_mean_diff(imbalanced_df, "group", ["age"])
        assert len(smd_df) == 1

    def test_empty_covariates(self, imbalanced_df: pd.DataFrame):
        smd_df = standardized_mean_diff(imbalanced_df, "group", [])
        assert len(smd_df) == 0


class TestBalanceSummary:
    def test_summary(self, imbalanced_df: pd.DataFrame):
        smd_df = standardized_mean_diff(imbalanced_df, "group", ["age", "income"])
        summary = balance_summary(smd_df)
        assert "mean_abs_smd" in summary
        assert "max_abs_smd" in summary
        assert "pct_below_01" in summary
        assert "n_covariates" in summary
        assert summary["n_covariates"] == 2


# ---------------------------------------------------------------------------
# PSM
# ---------------------------------------------------------------------------

class TestPSM:
    def test_basic(self, imbalanced_df: pd.DataFrame):
        result = propensity_score_match(
            imbalanced_df, "group", ["age", "income"],
            caliper=0.25, ratio=1,
        )
        assert isinstance(result, MatchResult)
        assert result.method == "PSM"
        assert result.n_matched_treatment > 0
        assert result.n_matched_control > 0
        assert result.propensity_scores is not None
        assert result.common_support is not None
        assert len(result.matched_df) > 0

    def test_improves_balance(self, imbalanced_df: pd.DataFrame):
        result = propensity_score_match(
            imbalanced_df, "group", ["age", "income"],
            caliper=0.2, ratio=1,
        )
        before = balance_summary(result.balance_before)
        after = balance_summary(result.balance_after)
        # PSM should reduce mean |SMD|
        assert after["mean_abs_smd"] < before["mean_abs_smd"]

    def test_ratio_2(self, imbalanced_df: pd.DataFrame):
        result = propensity_score_match(
            imbalanced_df, "group", ["age", "income"],
            caliper=0.3, ratio=2,
        )
        assert result.n_matched_control >= result.n_matched_treatment

    def test_tight_caliper_fewer_matches(self, imbalanced_df: pd.DataFrame):
        r_wide = propensity_score_match(
            imbalanced_df, "group", ["age", "income"], caliper=0.5,
        )
        r_tight = propensity_score_match(
            imbalanced_df, "group", ["age", "income"], caliper=0.05,
        )
        assert r_tight.n_matched_treatment <= r_wide.n_matched_treatment


# ---------------------------------------------------------------------------
# Exact matching
# ---------------------------------------------------------------------------

class TestExactMatch:
    def test_basic(self, imbalanced_df: pd.DataFrame):
        result = exact_match(
            imbalanced_df, "group", ["gender", "region"],
            covariates=["age", "income"],
        )
        assert isinstance(result, MatchResult)
        assert result.method == "Exact"
        assert result.n_matched_treatment > 0
        assert result.n_matched_treatment == result.n_matched_control

    def test_equal_groups(self, imbalanced_df: pd.DataFrame):
        result = exact_match(
            imbalanced_df, "group", ["gender"],
            covariates=["age"],
        )
        mdf = result.matched_df
        n_t = (mdf["group"] == 1).sum()
        n_c = (mdf["group"] == 0).sum()
        assert n_t == n_c


# ---------------------------------------------------------------------------
# NN matching
# ---------------------------------------------------------------------------

class TestNNMatch:
    def test_basic(self, imbalanced_df: pd.DataFrame):
        result = nearest_neighbor_match(
            imbalanced_df, "group", ["age", "income"],
            n_neighbors=1, metric="euclidean",
        )
        assert isinstance(result, MatchResult)
        assert result.method == "NN"
        assert result.n_matched_treatment > 0

    def test_mahalanobis(self, imbalanced_df: pd.DataFrame):
        result = nearest_neighbor_match(
            imbalanced_df, "group", ["age", "income"],
            n_neighbors=1, metric="mahalanobis",
        )
        assert result.n_matched_treatment > 0

    def test_improves_balance(self, imbalanced_df: pd.DataFrame):
        result = nearest_neighbor_match(
            imbalanced_df, "group", ["age", "income"],
        )
        before = balance_summary(result.balance_before)
        after = balance_summary(result.balance_after)
        assert after["mean_abs_smd"] < before["mean_abs_smd"]


# ---------------------------------------------------------------------------
# CEM
# ---------------------------------------------------------------------------

class TestCEM:
    def test_basic(self, imbalanced_df: pd.DataFrame):
        result = coarsened_exact_match(
            imbalanced_df, "group", ["age", "income"], n_bins=4,
        )
        assert isinstance(result, MatchResult)
        assert result.method == "CEM"
        assert result.n_matched_treatment > 0
        assert "_cem_weight" in result.matched_df.columns

    def test_more_bins_fewer_matches(self, imbalanced_df: pd.DataFrame):
        r_few = coarsened_exact_match(
            imbalanced_df, "group", ["age", "income"], n_bins=3,
        )
        r_many = coarsened_exact_match(
            imbalanced_df, "group", ["age", "income"], n_bins=10,
        )
        # More bins = stricter = potentially fewer matches
        assert r_many.n_matched_treatment <= r_few.n_matched_treatment + 50  # tolerance


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_nan_handling(self):
        df = pd.DataFrame({
            "group": [1, 1, 0, 0, 1, 0],
            "x": [1.0, 2.0, np.nan, 1.5, 2.5, 1.0],
            "y": [10, 20, 15, np.nan, 25, 12],
        })
        result = propensity_score_match(df, "group", ["x", "y"])
        assert isinstance(result, MatchResult)

    def test_small_dataset(self):
        df = pd.DataFrame({
            "group": [1, 1, 0, 0],
            "x": [10, 20, 11, 19],
        })
        result = nearest_neighbor_match(df, "group", ["x"], n_neighbors=1)
        assert result.n_matched_treatment <= 2

    def test_single_covariate_cem(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            "group": [1] * 50 + [0] * 50,
            "val": rng.normal(0, 1, 100),
        })
        result = coarsened_exact_match(df, "group", ["val"], n_bins=3)
        assert isinstance(result, MatchResult)

"""
core/tests.py – Statistical tests with business-readable interpretations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class TestResult:
    """Container for a statistical test result.

    Attributes
    ----------
    name : str
        Name of the test.
    statistic : float
        Test statistic value.
    p_value : float
        P-value.
    alpha : float
        Significance level used.
    significant : bool
        Whether the result is statistically significant.
    effect_size : float | None
        Effect size measure (Cohen's d, Cramér's V, r, etc.).
    effect_label : str
        Textual magnitude label (``"negligible"``, ``"small"``, etc.).
    ci : tuple[float, float] | None
        95% confidence interval where applicable.
    details : dict[str, Any]
        Extra numeric details (group stats, etc.).
    interpretation : str
        Human-readable business language summary.
    """
    name: str
    statistic: float
    p_value: float
    alpha: float = 0.05
    significant: bool = False
    effect_size: float | None = None
    effect_label: str = ""
    ci: tuple[float, float] | None = None
    details: dict[str, Any] = field(default_factory=dict)
    interpretation: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent groups."""
    pooled_std = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def _cramers_v(chi2: float, n: int, k: int, r: int) -> float:
    """Cramér's V effect size for chi-square."""
    return float(np.sqrt(chi2 / (n * (min(k, r) - 1)))) if n > 0 else 0.0


def _cramer_label(v: float) -> str:
    if v < 0.1:
        return "negligible"
    if v < 0.3:
        return "small"
    if v < 0.5:
        return "medium"
    return "large"


def _r_label(r: float) -> str:
    ar = abs(r)
    if ar < 0.1:
        return "negligible"
    if ar < 0.3:
        return "small"
    if ar < 0.5:
        return "medium"
    return "large"


def _sig_text(significant: bool, p: float, alpha: float) -> str:
    if significant:
        return f"statistically significant (p={p:.4f} < α={alpha})"
    return f"NOT statistically significant (p={p:.4f} ≥ α={alpha})"


def _cliff_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


# ---------------------------------------------------------------------------
# t-test (independent samples)
# ---------------------------------------------------------------------------

def ttest_independent(
    group_a: pd.Series | np.ndarray,
    group_b: pd.Series | np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = False,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> TestResult:
    """Welch's (or Student's) two-sample t-test.

    Parameters
    ----------
    group_a, group_b:
        Numeric samples to compare.
    alpha:
        Significance level.
    equal_var:
        If True, use Student's t-test (equal variances assumed).
    label_a, label_b:
        Human-readable group names.

    Returns
    -------
    TestResult
    """
    a = np.array(pd.to_numeric(group_a, errors="coerce").dropna(), dtype=float)
    b = np.array(pd.to_numeric(group_b, errors="coerce").dropna(), dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group must have at least 2 non-null observations.")

    stat, p = stats.ttest_ind(a, b, equal_var=equal_var)
    sig = bool(p < alpha)
    d = _cohens_d(a, b)

    # 95% CI on the difference in means (from t-distribution)
    diff = np.mean(a) - np.mean(b)
    se_diff = np.sqrt(np.var(a, ddof=1) / len(a) + np.var(b, ddof=1) / len(b))
    df_val = len(a) + len(b) - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df=df_val)
    ci = (round(diff - t_crit * se_diff, 4), round(diff + t_crit * se_diff, 4))

    direction = "higher" if np.mean(a) > np.mean(b) else "lower"
    interp = (
        f"The mean of {label_a} ({np.mean(a):.4f}) is {direction} than {label_b} ({np.mean(b):.4f}). "
        f"The difference is {_sig_text(sig, p, alpha)}. "
        f"Effect size (Cohen's d = {d:.3f}) is {_d_label(d)}. "
        f"95% CI for the difference: [{ci[0]}, {ci[1]}]."
    )

    return TestResult(
        name="Independent t-test (Welch)" if not equal_var else "Independent t-test (Student)",
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        alpha=alpha,
        significant=sig,
        effect_size=round(d, 4),
        effect_label=_d_label(d),
        ci=ci,
        details={
            f"n_{label_a}": len(a),
            f"mean_{label_a}": round(np.mean(a), 4),
            f"std_{label_a}": round(np.std(a, ddof=1), 4),
            f"n_{label_b}": len(b),
            f"mean_{label_b}": round(np.mean(b), 4),
            f"std_{label_b}": round(np.std(b, ddof=1), 4),
        },
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Mann-Whitney U
# ---------------------------------------------------------------------------

def mann_whitney(
    group_a: pd.Series | np.ndarray,
    group_b: pd.Series | np.ndarray,
    alpha: float = 0.05,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> TestResult:
    """Non-parametric Mann–Whitney U test for location difference.

    Parameters
    ----------
    group_a, group_b:
        Numeric samples.
    alpha:
        Significance level.
    label_a, label_b:
        Human-readable group names.

    Returns
    -------
    TestResult
    """
    a = np.array(pd.to_numeric(group_a, errors="coerce").dropna(), dtype=float)
    b = np.array(pd.to_numeric(group_b, errors="coerce").dropna(), dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group must have at least 2 non-null observations.")

    stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    sig = bool(p < alpha)

    # Rank-biserial correlation as effect size
    n1, n2 = len(a), len(b)
    r = 1 - (2 * stat) / (n1 * n2)

    direction = "higher" if np.median(a) > np.median(b) else "lower"
    interp = (
        f"The median of {label_a} ({np.median(a):.4f}) is {direction} than {label_b} ({np.median(b):.4f}). "
        f"The difference is {_sig_text(sig, p, alpha)}. "
        f"Effect size (rank-biserial r = {r:.3f}) is {_r_label(r)}."
    )

    return TestResult(
        name="Mann–Whitney U test",
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        alpha=alpha,
        significant=sig,
        effect_size=round(float(r), 4),
        effect_label=_r_label(r),
        details={
            f"n_{label_a}": n1,
            f"median_{label_a}": round(float(np.median(a)), 4),
            f"n_{label_b}": n2,
            f"median_{label_b}": round(float(np.median(b)), 4),
        },
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Chi-square test of independence
# ---------------------------------------------------------------------------

def chi_square_independence(
    df: pd.DataFrame,
    col_a: str,
    col_b: str,
    alpha: float = 0.05,
) -> TestResult:
    """Chi-square test of independence between two categorical columns.

    Parameters
    ----------
    df:
        DataFrame containing the columns.
    col_a, col_b:
        Categorical columns to test.
    alpha:
        Significance level.

    Returns
    -------
    TestResult
    """
    ct = pd.crosstab(df[col_a].astype(str), df[col_b].astype(str))
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        raise ValueError("Each column must have at least 2 unique non-null values.")

    chi2, p, dof, expected = stats.chi2_contingency(ct)
    sig = bool(p < alpha)
    n = int(ct.values.sum())
    v = _cramers_v(chi2, n, ct.shape[1], ct.shape[0])

    interp = (
        f"The association between '{col_a}' and '{col_b}' is {_sig_text(sig, p, alpha)}. "
        f"Chi²({dof}) = {chi2:.4f}. "
        f"Effect size: Cramér's V = {v:.3f} ({_cramer_label(v)})."
    )

    return TestResult(
        name="Chi-square test of independence",
        statistic=round(float(chi2), 4),
        p_value=round(float(p), 6),
        alpha=alpha,
        significant=sig,
        effect_size=round(v, 4),
        effect_label=_cramer_label(v),
        details={"dof": dof, "n": n, "expected_min": round(float(expected.min()), 2)},
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Pearson / Spearman correlation significance
# ---------------------------------------------------------------------------

def correlation_test(
    x: pd.Series | np.ndarray,
    y: pd.Series | np.ndarray,
    method: str = "pearson",
    alpha: float = 0.05,
    label_x: str = "X",
    label_y: str = "Y",
) -> TestResult:
    """Correlation coefficient with significance test.

    Parameters
    ----------
    x, y:
        Numeric series to correlate.
    method:
        ``"pearson"`` or ``"spearman"``.
    alpha:
        Significance level.
    label_x, label_y:
        Variable names.

    Returns
    -------
    TestResult
    """
    x_arr = np.array(pd.to_numeric(x, errors="coerce").dropna(), dtype=float)
    y_arr = np.array(pd.to_numeric(y, errors="coerce").dropna(), dtype=float)
    n = min(len(x_arr), len(y_arr))
    x_arr, y_arr = x_arr[:n], y_arr[:n]

    if n < 3:
        raise ValueError("Need at least 3 paired observations.")

    if method == "spearman":
        r, p = stats.spearmanr(x_arr, y_arr)
        test_name = "Spearman rank correlation"
    else:
        r, p = stats.pearsonr(x_arr, y_arr)
        test_name = "Pearson correlation"

    sig = bool(p < alpha)
    direction = "positive" if r > 0 else "negative"
    interp = (
        f"There is a {_r_label(abs(r))} {direction} {method} correlation between "
        f"'{label_x}' and '{label_y}' (r = {r:.4f}, {_sig_text(sig, p, alpha)})."
    )

    return TestResult(
        name=test_name,
        statistic=round(float(r), 4),
        p_value=round(float(p), 6),
        alpha=alpha,
        significant=sig,
        effect_size=round(float(r), 4),
        effect_label=_r_label(abs(r)),
        details={"n": n},
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Bootstrap test (mean / median difference)
# ---------------------------------------------------------------------------

def bootstrap_test(
    group_a: pd.Series | np.ndarray,
    group_b: pd.Series | np.ndarray,
    statistic: str = "mean",
    n_bootstrap: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> TestResult:
    """Bootstrap permutation test for mean or median difference.

    Parameters
    ----------
    group_a, group_b:
        Numeric samples.
    statistic:
        ``"mean"`` or ``"median"``.
    n_bootstrap:
        Number of bootstrap resamples.
    alpha:
        Significance level.
    seed:
        RNG seed for reproducibility.
    label_a, label_b:
        Group names.

    Returns
    -------
    TestResult
    """
    a = np.array(pd.to_numeric(group_a, errors="coerce").dropna(), dtype=float)
    b = np.array(pd.to_numeric(group_b, errors="coerce").dropna(), dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group must have at least 2 observations.")

    rng = np.random.default_rng(seed)
    stat_fn = np.mean if statistic == "mean" else np.median
    observed_diff = float(stat_fn(a) - stat_fn(b))

    combined = np.concatenate([a, b])
    n_a = len(a)
    boot_diffs = np.array([
        stat_fn(rng.choice(combined, size=n_a, replace=True)) -
        stat_fn(rng.choice(combined, size=len(b), replace=True))
        for _ in range(n_bootstrap)
    ])
    p_value = float(np.mean(np.abs(boot_diffs) >= abs(observed_diff)))
    sig = bool(p_value < alpha)

    ci_low = float(np.percentile(boot_diffs, (alpha / 2) * 100))
    ci_high = float(np.percentile(boot_diffs, (1 - alpha / 2) * 100))

    interp = (
        f"Bootstrap test ({n_bootstrap} resamples) for {statistic} difference: "
        f"{label_a} {statistic} = {stat_fn(a):.4f}, {label_b} {statistic} = {stat_fn(b):.4f}. "
        f"Observed difference = {observed_diff:.4f}. "
        f"The difference is {_sig_text(sig, p_value, alpha)}. "
        f"{int((1-alpha)*100)}% CI of null distribution: [{ci_low:.4f}, {ci_high:.4f}]."
    )

    return TestResult(
        name=f"Bootstrap permutation test ({statistic})",
        statistic=round(observed_diff, 4),
        p_value=round(p_value, 6),
        alpha=alpha,
        significant=sig,
        ci=(round(ci_low, 4), round(ci_high, 4)),
        details={
            f"n_{label_a}": len(a),
            f"{statistic}_{label_a}": round(float(stat_fn(a)), 4),
            f"n_{label_b}": len(b),
            f"{statistic}_{label_b}": round(float(stat_fn(b)), 4),
            "n_bootstrap": n_bootstrap,
        },
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# A/B test helper (with effect size + CI)
# ---------------------------------------------------------------------------

def ab_test(
    control: pd.Series | np.ndarray,
    treatment: pd.Series | np.ndarray,
    alpha: float = 0.05,
    label_ctrl: str = "Control",
    label_trt: str = "Treatment",
) -> dict[str, Any]:
    """Comprehensive A/B test helper.

    Runs a t-test, Mann-Whitney, and bootstrap; reports effect size and CI.

    Parameters
    ----------
    control, treatment:
        Numeric samples for each group.
    alpha:
        Significance level.
    label_ctrl, label_trt:
        Group names.

    Returns
    -------
    dict with keys: ``ttest``, ``mann_whitney``, ``bootstrap``, ``summary``.
    """
    ttest = ttest_independent(control, treatment, alpha=alpha,
                              label_a=label_ctrl, label_b=label_trt)
    mw = mann_whitney(control, treatment, alpha=alpha,
                      label_a=label_ctrl, label_b=label_trt)
    boot = bootstrap_test(control, treatment, statistic="mean", alpha=alpha,
                          label_a=label_ctrl, label_b=label_trt)

    ctrl_arr = np.array(pd.to_numeric(pd.Series(control), errors="coerce").dropna())
    trt_arr = np.array(pd.to_numeric(pd.Series(treatment), errors="coerce").dropna())
    lift = (np.mean(trt_arr) - np.mean(ctrl_arr)) / abs(np.mean(ctrl_arr)) * 100 if np.mean(ctrl_arr) != 0 else float("nan")
    any_sig = ttest.significant or mw.significant

    summary = (
        f"A/B Test Summary — {label_ctrl} vs {label_trt}:\n"
        f"  Lift: {lift:+.2f}%  |  Cohen's d: {ttest.effect_size} ({ttest.effect_label})\n"
        f"  t-test p={ttest.p_value}  |  Mann-Whitney p={mw.p_value}  |  Bootstrap p={boot.p_value}\n"
        f"  Overall verdict: {'SIGNIFICANT' if any_sig else 'NOT SIGNIFICANT'} at α={alpha}."
    )

    return {"ttest": ttest, "mann_whitney": mw, "bootstrap": boot, "summary": summary, "lift_pct": round(lift, 4)}


# ---------------------------------------------------------------------------
# Lag correlation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cliff's delta effect size
# ---------------------------------------------------------------------------

def cliffs_delta(
    group_a: pd.Series | np.ndarray,
    group_b: pd.Series | np.ndarray,
) -> tuple[float, str]:
    """Cliff's delta non-parametric effect size.

    Cliff's delta is the proportion of (a_i, b_j) pairs where a > b
    minus the proportion where a < b.  Range: [-1, 1].

    Parameters
    ----------
    group_a, group_b:
        Numeric samples.

    Returns
    -------
    tuple[float, str]
        (delta value, magnitude label).
    """
    a = np.array(pd.to_numeric(pd.Series(group_a), errors="coerce").dropna(), dtype=float)
    b = np.array(pd.to_numeric(pd.Series(group_b), errors="coerce").dropna(), dtype=float)

    if len(a) == 0 or len(b) == 0:
        return 0.0, "negligible"

    # Vectorised comparison
    n_a, n_b = len(a), len(b)
    more = 0
    less = 0
    for ai in a:
        more += int((ai > b).sum())
        less += int((ai < b).sum())
    d = (more - less) / (n_a * n_b)
    return round(d, 4), _cliff_label(d)


# ---------------------------------------------------------------------------
# Permutation test
# ---------------------------------------------------------------------------

def permutation_test(
    group_a: pd.Series | np.ndarray,
    group_b: pd.Series | np.ndarray,
    n_perm: int = 10000,
    stat: str = "mean",
    alpha: float = 0.05,
    seed: int = 42,
    label_a: str = "Group A",
    label_b: str = "Group B",
) -> TestResult:
    """Exact permutation test for difference in mean or median.

    Unlike the bootstrap test, the permutation test shuffles group labels
    to construct the null distribution of the test statistic.

    Parameters
    ----------
    group_a, group_b:
        Numeric samples.
    n_perm:
        Number of permutations.
    stat:
        ``"mean"`` or ``"median"``.
    alpha:
        Significance level.
    seed:
        RNG seed.
    label_a, label_b:
        Group names.

    Returns
    -------
    TestResult
    """
    a = np.array(pd.to_numeric(pd.Series(group_a), errors="coerce").dropna(), dtype=float)
    b = np.array(pd.to_numeric(pd.Series(group_b), errors="coerce").dropna(), dtype=float)

    if len(a) < 2 or len(b) < 2:
        raise ValueError("Each group must have at least 2 observations.")

    rng = np.random.default_rng(seed)
    stat_fn = np.mean if stat == "mean" else np.median
    observed = float(stat_fn(a) - stat_fn(b))

    combined = np.concatenate([a, b])
    n_a = len(a)
    count_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_diff = stat_fn(combined[:n_a]) - stat_fn(combined[n_a:])
        if abs(perm_diff) >= abs(observed):
            count_extreme += 1

    p_value = float(count_extreme / n_perm)
    sig = bool(p_value < alpha)

    interp = (
        f"Permutation test ({n_perm} permutations) for {stat} difference: "
        f"{label_a} = {stat_fn(a):.4f}, {label_b} = {stat_fn(b):.4f}. "
        f"Observed difference = {observed:.4f}. "
        f"The difference is {_sig_text(sig, p_value, alpha)}."
    )

    return TestResult(
        name=f"Permutation test ({stat})",
        statistic=round(observed, 4),
        p_value=round(p_value, 6),
        alpha=alpha,
        significant=sig,
        details={
            f"n_{label_a}": len(a),
            f"{stat}_{label_a}": round(float(stat_fn(a)), 4),
            f"n_{label_b}": len(b),
            f"{stat}_{label_b}": round(float(stat_fn(b)), 4),
            "n_permutations": n_perm,
        },
        interpretation=interp,
    )


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def bh_correction(p_values: list[float], alpha: float = 0.05) -> list[dict[str, Any]]:
    """Benjamini-Hochberg procedure for multiple testing correction.

    Parameters
    ----------
    p_values:
        Raw p-values from multiple tests.
    alpha:
        Family-wise significance level.

    Returns
    -------
    list[dict] with keys ``original_index``, ``p_value``, ``rank``,
    ``bh_threshold``, ``significant``, ``adjusted_p``.
    """
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results: list[dict[str, Any]] = [{} for _ in range(n)]

    # Compute adjusted p-values (step-up)
    prev_adj = 0.0
    adjusted = [0.0] * n
    for rank_idx, (orig_idx, p) in enumerate(indexed):
        rank = rank_idx + 1
        adj_p = min(p * n / rank, 1.0)
        adjusted[rank_idx] = adj_p

    # Enforce monotonicity: adjusted p-values must be non-decreasing by rank
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    for i in range(1, n):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])

    for rank_idx, (orig_idx, p) in enumerate(indexed):
        rank = rank_idx + 1
        bh_thresh = alpha * rank / n
        results[orig_idx] = {
            "original_index": orig_idx,
            "p_value": round(p, 6),
            "rank": rank,
            "bh_threshold": round(bh_thresh, 6),
            "significant": bool(p <= bh_thresh),
            "adjusted_p": round(adjusted[rank_idx], 6),
        }

    return results


# ---------------------------------------------------------------------------
# Normality & variance diagnostics
# ---------------------------------------------------------------------------

@dataclass
class NormalityResult:
    """Result of a normality test for a single sample."""
    test_name: str
    statistic: float
    p_value: float
    alpha: float
    is_normal: bool
    n: int
    skewness: float
    kurtosis: float
    skew_label: str
    kurt_label: str
    interpretation: str


def normality_test(
    series: "pd.Series | np.ndarray",
    alpha: float = 0.05,
    label: str = "Выборка",
) -> NormalityResult:
    """Test normality of a sample, auto-selecting the appropriate test by n.

    - n < 50  → Shapiro-Wilk (most powerful for small samples)
    - 50 ≤ n < 5000 → D'Agostino-Pearson K²
    - n ≥ 5000 → Kolmogorov-Smirnov vs fitted normal
    """
    arr = np.array(pd.to_numeric(pd.Series(series), errors="coerce").dropna(), dtype=float)
    n = len(arr)
    if n < 3:
        raise ValueError("Нужно минимум 3 наблюдения для теста нормальности.")

    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))  # excess kurtosis

    if n < 50:
        stat, p = stats.shapiro(arr)
        test_name = "Шапиро-Уилк"
    elif n < 5000:
        stat, p = stats.normaltest(arr)  # D'Agostino-Pearson K²
        test_name = "Д'Агостино-Пирсон K²"
    else:
        mu, sigma = float(arr.mean()), float(arr.std())
        stat, p = stats.kstest(arr, "norm", args=(mu, sigma))
        test_name = "Колмогоров-Смирнов"

    is_normal = bool(p >= alpha)

    abs_skew = abs(skew)
    if abs_skew < 0.5:
        skew_label = "симметричное"
    elif abs_skew < 1.0:
        skew_label = "умеренно асимметричное"
    else:
        skew_label = "сильно асимметричное"

    if kurt < -1:
        kurt_label = "платикуртное (плоское)"
    elif kurt > 1:
        kurt_label = "лептокуртное (остроконечное)"
    else:
        kurt_label = "мезокуртное (норма)"

    verdict = (
        f"нет оснований отвергнуть нормальность (p={p:.4f} ≥ α={alpha})"
        if is_normal
        else f"нормальность отвергается (p={p:.4f} < α={alpha})"
    )
    interp = (
        f"{test_name} для «{label}» (n={n}): W={stat:.4f}, p={p:.4f}. "
        f"Вывод: {verdict}. "
        f"Асимметрия={skew:.3f} ({skew_label}), эксцесс={kurt:.3f} ({kurt_label})."
    )
    return NormalityResult(
        test_name=test_name,
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        alpha=alpha,
        is_normal=is_normal,
        n=n,
        skewness=round(skew, 4),
        kurtosis=round(kurt, 4),
        skew_label=skew_label,
        kurt_label=kurt_label,
        interpretation=interp,
    )


def levene_test(
    *groups: "pd.Series | np.ndarray",
    alpha: float = 0.05,
    labels: "list[str] | None" = None,
) -> TestResult:
    """Levene's test for equality of variances across groups."""
    arrays = [
        np.array(pd.to_numeric(pd.Series(g), errors="coerce").dropna(), dtype=float)
        for g in groups
    ]
    if len(arrays) < 2:
        raise ValueError("Нужно минимум 2 группы.")
    stat, p = stats.levene(*arrays)
    sig = bool(p < alpha)
    if labels is None:
        labels = [f"Группа {i+1}" for i in range(len(arrays))]
    verdict = (
        "дисперсии значимо различаются — гетероскедастичность"
        if sig
        else "дисперсии не различаются значимо — гомоскедастичность"
    )
    vars_str = ", ".join(
        f"{lb}: σ²={np.var(arr, ddof=1):.4f}" for lb, arr in zip(labels, arrays)
    )
    interp = (
        f"Тест Левена: W={stat:.4f}, p={p:.4f}. При α={alpha}: {verdict}. {vars_str}."
    )
    return TestResult(
        name="Тест Левена (однородность дисперсий)",
        statistic=round(float(stat), 4),
        p_value=round(float(p), 6),
        alpha=alpha,
        significant=sig,
        interpretation=interp,
    )


def diagnose_groups(
    group_a: "pd.Series | np.ndarray",
    group_b: "pd.Series | np.ndarray",
    alpha: float = 0.05,
    label_a: str = "Группа A",
    label_b: str = "Группа B",
) -> dict:
    """Full pre-test diagnostic for a two-group comparison.

    Returns normality results, Levene test, recommended test key + reason,
    and a list of user-facing warnings.
    """
    a = np.array(pd.to_numeric(pd.Series(group_a), errors="coerce").dropna(), dtype=float)
    b = np.array(pd.to_numeric(pd.Series(group_b), errors="coerce").dropna(), dtype=float)

    norm_a = normality_test(a, alpha=alpha, label=label_a)
    norm_b = normality_test(b, alpha=alpha, label=label_b)
    lev = levene_test(a, b, alpha=alpha, labels=[label_a, label_b])

    n_min = min(norm_a.n, norm_b.n)
    both_normal = norm_a.is_normal and norm_b.is_normal
    equal_var = not lev.significant

    warnings: list[str] = []

    if n_min < 5:
        rec_test = "bootstrap"
        rec_name = "Бутстрап / Перестановочный тест"
        rec_reason = f"Очень малая выборка (n_min={n_min}). Параметрические тесты ненадёжны."
        warnings.append("⚠️ Слишком мало данных — результаты могут быть ненадёжными.")
    elif n_min < 30 and not both_normal:
        rec_test = "mann_whitney"
        rec_name = "Манна-Уитни U"
        rec_reason = (
            f"Малая выборка (n_min={n_min} < 30) и ненормальное распределение — "
            "используйте непараметрический тест."
        )
    elif not both_normal:
        rec_test = "mann_whitney"
        rec_name = "Манна-Уитни U"
        rec_reason = "Распределение отклоняется от нормального — предпочтителен непараметрический тест."
        if n_min >= 100:
            warnings.append(
                "ℹ️ При n ≥ 100 t-тест тоже устойчив благодаря ЦПТ — допустимо использовать оба."
            )
    elif both_normal and equal_var:
        rec_test = "ttest_student"
        rec_name = "t-тест Стьюдента"
        rec_reason = (
            "Оба распределения нормальные, дисперсии однородны — "
            "классический t-тест Стьюдента оптимален."
        )
    else:
        rec_test = "ttest_welch"
        rec_name = "t-тест Уэлча"
        rec_reason = (
            "Распределения нормальные, но дисперсии различаются — "
            "t-тест Уэлча корректнее Стьюдента."
        )

    if abs(norm_a.skewness) > 1 or abs(norm_b.skewness) > 1:
        warnings.append(
            "⚠️ Сильная асимметрия — среднее может не отражать типичное значение. "
            "Рассмотрите использование медианы."
        )
    if norm_a.n > 0 and norm_b.n > 0:
        ratio = max(norm_a.n, norm_b.n) / min(norm_a.n, norm_b.n)
        if ratio > 3:
            warnings.append(
                f"⚠️ Группы несбалансированы (соотношение {ratio:.1f}:1) — "
                "интерпретируйте с осторожностью."
            )

    return {
        "norm_a": norm_a,
        "norm_b": norm_b,
        "levene": lev,
        "rec_test": rec_test,
        "rec_name": rec_name,
        "rec_reason": rec_reason,
        "warnings": warnings,
        "both_normal": both_normal,
        "equal_var": equal_var,
        "n_a": norm_a.n,
        "n_b": norm_b.n,
    }


# ---------------------------------------------------------------------------
# Lag correlation
# ---------------------------------------------------------------------------

def lag_correlation(
    x: pd.Series,
    y: pd.Series,
    max_lag: int = 12,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute cross-correlation between x and lagged y.

    Parameters
    ----------
    x:
        Leading series.
    y:
        Lagging series.
    max_lag:
        Maximum number of periods to lag.
    method:
        ``"pearson"`` or ``"spearman"``.

    Returns
    -------
    pd.DataFrame with columns ``lag``, ``correlation``, ``p_value``.
    """
    results = []
    for lag in range(0, max_lag + 1):
        x_arr = x.iloc[:len(x) - lag].reset_index(drop=True) if lag > 0 else x.reset_index(drop=True)
        y_arr = y.iloc[lag:].reset_index(drop=True)
        n = min(len(x_arr), len(y_arr))
        x_s = pd.to_numeric(x_arr[:n], errors="coerce").dropna()
        y_s = pd.to_numeric(y_arr[:n], errors="coerce").dropna()
        m = min(len(x_s), len(y_s))
        if m < 3:
            results.append({"lag": lag, "correlation": float("nan"), "p_value": float("nan")})
            continue
        if method == "spearman":
            r, p = stats.spearmanr(x_s[:m].values, y_s[:m].values)
        else:
            r, p = stats.pearsonr(x_s[:m].values, y_s[:m].values)
        results.append({"lag": lag, "correlation": round(float(r), 4), "p_value": round(float(p), 6)})
    return pd.DataFrame(results)

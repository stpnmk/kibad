# KIBAD Methodology

## Statistical Hypothesis Tests

### Welch's t-test

**What it does**: Compares means of two independent samples without assuming
equal variances.

**Formula**:

```
t = (X1_bar - X2_bar) / sqrt(s1^2/n1 + s2^2/n2)

df = (s1^2/n1 + s2^2/n2)^2 / ((s1^2/n1)^2/(n1-1) + (s2^2/n2)^2/(n2-1))
```

**Assumptions**: Independent observations, approximately normal distributions
(robust for n > 30 by CLT).

**When to use**: Comparing a metric (revenue, conversion) between two groups
(control vs treatment, region A vs region B).

**When NOT to use**: Ordinal or heavily skewed data with small samples -- use
Mann-Whitney U instead. Paired observations -- use paired t-test.

**Pitfalls**: Significant p-value does not imply practical significance. Always
report effect size alongside the test.

### Mann-Whitney U Test

**What it does**: Non-parametric test comparing the rank distributions of two
independent samples.

**Formula**:

```
U = n1*n2 + n1*(n1+1)/2 - R1

where R1 = sum of ranks in sample 1
```

**Assumptions**: Independent observations, ordinal or continuous outcome.

**When to use**: Skewed distributions, ordinal data, small samples where
normality is questionable.

**When NOT to use**: Paired data. When you need to compare means specifically
(this tests stochastic dominance, not means).

**Pitfalls**: Less powerful than t-test when normality holds. Ties reduce power.

### Chi-Square Test of Independence

**What it does**: Tests whether two categorical variables are independent.

**Formula**:

```
chi2 = sum((O_ij - E_ij)^2 / E_ij)

E_ij = (row_i_total * col_j_total) / grand_total
```

**Assumptions**: Expected frequency >= 5 in each cell (use Fisher's exact test
otherwise).

**When to use**: Comparing proportions across categories (conversion rates by
segment).

**When NOT to use**: Continuous variables. Small expected counts (< 5).

### Correlation (Pearson and Spearman)

**Pearson r**: Linear relationship strength between two continuous variables.

```
r = cov(X, Y) / (std(X) * std(Y))
```

**Spearman rho**: Rank-based correlation, measures monotonic relationship.

**When NOT to use**: Pearson is misleading for non-linear relationships. Always
visualize the scatter plot first.

### Bootstrap Confidence Interval

**What it does**: Estimates the sampling distribution of a statistic by
resampling with replacement.

**Procedure**:

1. Draw B bootstrap samples (default B = 10,000) from the data.
2. Compute the statistic of interest for each sample.
3. Use the percentile method: CI = [q(alpha/2), q(1 - alpha/2)].

**When to use**: When parametric assumptions are uncertain, or for statistics
without known closed-form CIs (e.g., median, ratio of means).

**Pitfalls**: Bias-corrected and accelerated (BCa) intervals are preferred for
skewed distributions but are more computationally expensive.

### Permutation Test

**What it does**: Tests whether the observed difference between groups could
arise by chance, by permuting group labels.

**Procedure**:

1. Compute the observed test statistic (e.g., difference in means).
2. Randomly permute group labels N times (default N = 10,000).
3. p-value = proportion of permuted statistics >= observed.

**When to use**: Small samples, non-standard test statistics, when parametric
assumptions fail.

### Cliff's Delta (Effect Size)

**What it does**: Non-parametric effect size measuring the probability that a
randomly chosen observation from one group is larger than one from another.

```
delta = (count(x_i > y_j) - count(x_i < y_j)) / (n1 * n2)
```

**Interpretation**: |d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else large.

### Benjamini-Hochberg Correction

**What it does**: Controls the false discovery rate (FDR) when running multiple
tests simultaneously.

**Procedure**:

1. Sort p-values in ascending order: p(1) <= p(2) <= ... <= p(m).
2. For each p(i), compute the BH threshold: (i/m) * alpha.
3. Find the largest i where p(i) <= threshold. Reject all hypotheses 1..i.

**When to use**: Always apply when running more than one hypothesis test in the
same analysis (e.g., testing multiple metrics or segments).

## Time Series Methods

### STL Decomposition

**What it does**: Decomposes a time series into trend, seasonal, and residual
components using LOESS smoothing.

```
Y(t) = T(t) + S(t) + R(t)
```

**When to use**: Understanding underlying patterns in monthly/weekly data.

**When NOT to use**: Irregular time series with missing observations or
non-periodic data.

### ACF / PACF

**ACF (Autocorrelation Function)**: Correlation of the series with its own
lagged values. Identifies MA order.

**PACF (Partial ACF)**: Correlation after removing the effect of intermediate
lags. Identifies AR order.

### SARIMAX

**What it does**: Seasonal ARIMA with exogenous regressors.

```
ARIMA(p, d, q) x (P, D, Q, s) + X*beta
```

- p, d, q: non-seasonal AR, differencing, MA orders.
- P, D, Q, s: seasonal orders and period.
- X: exogenous variables.

**When to use**: Forecasting time series with trend, seasonality, and external
drivers.

**Pitfalls**: Requires stationary data (apply differencing). Sensitive to
outliers. AIC/BIC for model selection.

### ARX (Autoregressive with Exogenous Variables)

Simplified model: `Y(t) = a1*Y(t-1) + ... + ap*Y(t-p) + b*X(t) + e(t)`.

**When to use**: When SARIMAX is overkill and a simple AR model with covariates
suffices.

### Naive Forecast

Baseline model: `Y(t+h) = Y(t)` (last value) or `Y(t+h) = Y(t-s)` (seasonal
naive).

**When to use**: Always compute as a baseline. If your model cannot beat naive,
it has no value.

### Anomaly Detection

**Rolling z-score**: Flag points where `|x - rolling_mean| / rolling_std > threshold`.

**STL residual**: Flag points where the residual component exceeds a threshold
(e.g., 3 standard deviations of residuals).

**When to use**: Monitoring KPIs for unexpected deviations.

## Factor Attribution

### Additive Attribution

**What it does**: Decomposes the total change in a metric into additive
contributions of each factor.

```
delta_Y = sum(delta_factor_i)
```

Each factor's contribution is estimated by varying it while holding others at
their base values.

### Multiplicative Attribution (Log-Ratio)

**What it does**: Decomposes a ratio change into multiplicative factor
contributions using logarithms.

```
ln(Y1/Y0) = sum(ln(factor_i_1 / factor_i_0))
```

**When to use**: When the metric is a product of factors (e.g.,
Revenue = Users * Conversion * ARPU).

### Regression-Based Attribution

Uses OLS regression coefficients to attribute change:

```
delta_Y = sum(beta_i * delta_X_i)
```

**When to use**: When factors are not strictly multiplicative and interact.

**Pitfalls**: Multicollinearity inflates coefficient uncertainty. Check VIF.

### Shapley Value Approximation

**What it does**: Fair allocation of total change among factors based on their
marginal contributions across all possible orderings.

```
phi_i = (1/|N|!) * sum over permutations(marginal contribution of i)
```

KIBAD uses a sampling approximation (1000 permutations) for computational
efficiency.

**When to use**: When you need a theoretically fair attribution that accounts
for factor interactions.

**Pitfalls**: Computationally expensive for many factors. Approximation
introduces variance.

## Trigger Rules

### Threshold Crossing

Fires when a metric crosses an absolute threshold:

```
ALERT if metric > upper_threshold OR metric < lower_threshold
```

### Deviation from Baseline

Fires when a metric deviates from a rolling baseline by more than k standard
deviations:

```
ALERT if |metric - rolling_mean| > k * rolling_std
```

### Slope Change

Fires when the slope of a metric over a recent window differs significantly
from the historical slope:

```
slope_recent = linear_regression(metric, window=w_recent)
slope_history = linear_regression(metric, window=w_history)
ALERT if |slope_recent - slope_history| > threshold
```

**When to use**: Detecting trend shifts that threshold rules would miss.

**Pitfalls**: Sensitive to window size selection. Short windows produce noisy
alerts.

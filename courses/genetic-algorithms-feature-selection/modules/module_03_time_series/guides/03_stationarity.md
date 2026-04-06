# Stationarity Requirements for GA Feature Selection

> **Reading time:** ~18 min | **Module:** 3 — Time Series | **Prerequisites:** 02 Lag Features

## In Brief

Stationarity means that a time series' statistical properties (mean, variance, autocorrelation) do not change over time. For genetic algorithm feature selection, non-stationary data can cause selected features to perform well during training but fail on future data when the underlying relationships shift.

<div class="callout-key">

**Key Concept Summary:** Stationarity determines whether the patterns your GA discovers will persist into the future. A GA trained on non-stationary data selects features that describe *what has been happening recently* (spurious trends, temporary correlations). A GA trained on stationarized data selects features with *persistent predictive power* that generalizes across time periods and market regimes.

</div>

<div class="callout-insight">

Feature selection on non-stationary data finds spurious relationships that are time-specific rather than persistent. Differencing, detrending, or feature engineering transforms can induce stationarity, allowing GA to discover truly robust feature combinations that generalize to future time periods.

</div>


![Walk-Forward Timeline](./walk_forward_timeline.svg)

## Formal Definition

### Strict Stationarity

A time series $\{X_t\}$ is strictly stationary if the joint distribution is invariant under time shifts:

$$P(X_{t_1}, X_{t_2}, ..., X_{t_k}) = P(X_{t_1+\tau}, X_{t_2+\tau}, ..., X_{t_k+\tau})$$

for all $t_1, ..., t_k, \tau$

### Weak (Covariance) Stationarity

A time series is weakly stationary if:

1. **Constant mean**: $E[X_t] = \mu$ for all $t$
2. **Constant variance**: $\text{Var}(X_t) = \sigma^2$ for all $t$
3. **Time-invariant autocovariance**: $\text{Cov}(X_t, X_{t+k}) = \gamma(k)$ depends only on lag $k$, not time $t$

### Augmented Dickey-Fuller (ADF) Test

Tests the null hypothesis: series has a unit root (non-stationary)

Test statistic:
$$\Delta y_t = \alpha + \beta t + \gamma y_{t-1} + \delta_1 \Delta y_{t-1} + ... + \delta_p \Delta y_{t-p} + \epsilon_t$$

If $\gamma = 0$ (unit root), series is non-stationary.

p-value < 0.05 → reject null → series is stationary

### Common Non-Stationary Patterns

1. **Trend**: $X_t = \mu_t + \epsilon_t$ where $\mu_t$ changes over time
2. **Seasonality**: $X_t = S_t + \epsilon_t$ where $S_t$ has periodic pattern
3. **Structural breaks**: Mean/variance changes at specific points
4. **Heteroscedasticity**: Variance changes over time

## Intuitive Explanation

Imagine trying to predict house prices using a dataset spanning 1970-2020. If you include "year" as a feature and find it strongly predicts prices (upward trend), your GA might select year-related features. But this relationship isn't useful for predicting 2021 prices—the trend might have changed due to a recession, pandemic, or policy shift.

**Stationary data** behaves the same way throughout time. Patterns you discover in 1980 data still work in 2020 data. The autocorrelation structure, mean level, and variance remain consistent.

**Non-stationary data** has shifting patterns. A feature that predicts well during a bull market might fail during a recession. GA trained on non-stationary data learns time-specific patterns, not generalizable relationships.

**Solutions**:
1. **Differencing**: Predict changes ($\Delta y_t = y_t - y_{t-1}$) instead of levels ($y_t$)
2. **Detrending**: Remove trend component before feature selection
3. **Seasonal adjustment**: Remove seasonal patterns
4. **Log transformation**: Stabilize variance
5. **Feature engineering**: Create ratio/growth rate features that are naturally stationary

## Why Stationarity Matters for GA Feature Selection Specifically

Stationarity is important for all time series modeling, but it creates a *specific and acute* problem for GA feature selection. The GA's optimization process amplifies the damage from non-stationarity in a way that a single model fit does not.

**Worked example -- GA on non-stationary vs. stationarized data:**

Suppose you have 50 candidate features for predicting monthly commodity returns. Five of these features have genuine, persistent predictive power (e.g., inventory-to-consumption ratios, carry signals). Ten others are non-stationary (e.g., raw price levels, cumulative volume, GDP in dollars) and happen to correlate with the target during your training period because both share an upward trend.

- **GA on non-stationary data:** The GA discovers that the 10 trending features produce excellent fitness scores during training -- the shared trend makes them appear strongly predictive. The GA converges on a chromosome that selects mostly trending features and only 1-2 of the genuinely predictive ones. When deployed on new data where the trend reverses (e.g., a recession), the model's accuracy collapses.

- **GA on stationarized data:** After differencing the features and the target, the shared trend is removed. The trending features now appear as what they are -- noise with no predictive content for differenced returns. The GA converges on the 5 genuinely predictive features, and the model generalizes across different market regimes.

<div class="callout-insight">

The GA is an optimization machine -- it will find and exploit any shortcut to high fitness, including spurious correlations from shared trends. Stationarity testing removes these shortcuts, forcing the GA to discover genuine predictive relationships.

</div>

**The compounding problem:** Unlike fitting a single model, the GA evaluates thousands of feature combinations across many generations. Each generation, it preferentially selects and breeds chromosomes that exploit the spurious trend correlations. By generation 50, the population has converged on feature sets dominated by non-stationary features -- the GA has efficiently optimized for the wrong thing. A single Lasso fit might also select some trending features, but the GA's iterative search amplifies the problem because it has many more opportunities to find and lock onto spurious patterns.

## The Embargo/Gap: How Autocorrelation Leaks and How to Prevent It

Even with walk-forward validation, a subtle form of information leakage persists when time series data has strong autocorrelation. Understanding this mechanism is essential for proper GA fitness evaluation.

**How autocorrelation leaks information:** When the last training observation is at time $t$ and the first test observation is at time $t+1$, autocorrelation means these two observations are highly correlated. The model effectively "knows" the test target because the training target at time $t$ is nearly the same value (for AR(1) with coefficient 0.95, the correlation between consecutive observations is 0.95). This is not the same as future leakage -- the model does not train on future data -- but it creates unrealistically easy test predictions at the boundary.

```
Train:  ─────────────[last training obs]
Test:                                    [first test obs]─────────
                                    ^
                                    |
                        These two are highly correlated
                        (autocorrelation = 0.95)
```

**How the gap (embargo) prevents it:** Adding a gap of $g$ observations between training and test sets means the nearest test observation is $g$ steps away from the training boundary. The autocorrelation decays with distance: for an AR(1) process, correlation at lag $g$ is $\phi^g$. With $\phi = 0.95$ and $g = 10$, the correlation drops from 0.95 to $0.95^{10} = 0.60$. With $g = 50$, it drops to $0.95^{50} = 0.08$ -- essentially independent.

```
Train:  ─────────────[last]
Gap:                       [skip g observations]
Test:                                            [first test]─────
                                    ^
                                    |
                        Autocorrelation has decayed
```

**Rule of thumb for gap size:** Set the gap to the lag where the ACF drops below the significance threshold (typically $1.96/\sqrt{n}$). For daily financial data with autocorrelation half-life of 5 days, a gap of 5-10 trading days is standard. For monthly macroeconomic data with slow decay, gaps of 3-6 months may be needed.

<div class="callout-warning">

**Warning:** The gap trades information leakage for fewer test samples. A gap of 50 on a 500-sample dataset with 5 walk-forward splits wastes 250 observations. Balance the gap size against available data -- use the minimum gap that reduces autocorrelation below the significance threshold.

</div>

## Code Implementation

### Stationarity Testing


<span class="filename">__str__.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class StationarityTestResult:
    """Results from stationarity test."""
    test_statistic: float
    p_value: float
    critical_values: Dict[str, float]
    is_stationary: bool
    lags_used: int
    test_name: str

    def __str__(self):
        status = "STATIONARY" if self.is_stationary else "NON-STATIONARY"
        return (
            f"{self.test_name} Test Results:\n"
            f"  Test Statistic: {self.test_statistic:.4f}\n"
            f"  P-value: {self.p_value:.4f}\n"
            f"  Critical Values: {self.critical_values}\n"
            f"  Conclusion: {status}"
        )


def adf_test(
    series: np.ndarray,
    maxlag: Optional[int] = None,
    regression: str = 'c',
    alpha: float = 0.05
) -> StationarityTestResult:
    """
    Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    series : np.ndarray
        Time series to test
    maxlag : int, optional
        Maximum lag to use in test
        If None, uses int(12 * (n/100)^(1/4))
    regression : str
        'c': constant only
        'ct': constant and trend
        'ctt': constant, linear and quadratic trend
        'n': no constant, no trend
    alpha : float
        Significance level

    Returns
    -------
    StationarityTestResult
        Test results

    Examples
    --------
    >>> series = np.random.randn(100).cumsum()  # Random walk (non-stationary)
    >>> result = adf_test(series)
    >>> print(result.is_stationary)  # False
    """
    from statsmodels.tsa.stattools import adfuller

    # Run ADF test
    result = adfuller(series, maxlag=maxlag, regression=regression)

    test_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    critical_values = result[4]

    # Determine if stationary (reject null hypothesis)
    is_stationary = p_value < alpha

    return StationarityTestResult(
        test_statistic=test_stat,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        lags_used=lags_used,
        test_name="Augmented Dickey-Fuller"
    )


def kpss_test(
    series: np.ndarray,
    regression: str = 'c',
    nlags: str = 'auto',
    alpha: float = 0.05
) -> StationarityTestResult:
    """
    KPSS test for stationarity.

    Note: Null hypothesis is STATIONARITY (opposite of ADF)

    Parameters
    ----------
    series : np.ndarray
        Time series to test
    regression : str
        'c': level stationarity
        'ct': trend stationarity
    nlags : str or int
        'auto' or specific number of lags
    alpha : float
        Significance level

    Returns
    -------
    StationarityTestResult
        Test results

    Notes
    -----
    KPSS null hypothesis: series is stationary
    p-value < alpha → reject null → series is NON-stationary
    """
    from statsmodels.tsa.stattools import kpss

    result = kpss(series, regression=regression, nlags=nlags)

    test_stat = result[0]
    p_value = result[1]
    lags_used = result[2]
    critical_values = result[3]

    # KPSS: reject null means NON-stationary
    is_stationary = p_value >= alpha

    return StationarityTestResult(
        test_statistic=test_stat,
        p_value=p_value,
        critical_values=critical_values,
        is_stationary=is_stationary,
        lags_used=lags_used,
        test_name="KPSS"
    )


def combined_stationarity_test(
    series: np.ndarray,
    alpha: float = 0.05
) -> Dict[str, StationarityTestResult]:
    """
    Run both ADF and KPSS tests.

    Best practice: series should pass both tests.

    Returns
    -------
    Dict[str, StationarityTestResult]
        Results from both tests

    Interpretation
    --------------
    ADF stationary + KPSS stationary → Stationary
    ADF non-stat + KPSS non-stat → Non-stationary
    ADF stationary + KPSS non-stat → Difference-stationary
    ADF non-stat + KPSS stationary → Trend-stationary
    """
    adf_result = adf_test(series, alpha=alpha)
    kpss_result = kpss_test(series, alpha=alpha)

    return {
        'adf': adf_result,
        'kpss': kpss_result
    }


def test_stationarity_multiple_series(
    data: pd.DataFrame,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Test stationarity for all columns in DataFrame.

    Returns
    -------
    pd.DataFrame
        Summary of stationarity tests for each column
    """
    results = []

    for col in data.columns:
        series = data[col].dropna().values

        adf = adf_test(series, alpha=alpha)
        kpss = kpss_test(series, alpha=alpha)

        results.append({
            'column': col,
            'adf_statistic': adf.test_statistic,
            'adf_pvalue': adf.p_value,
            'adf_stationary': adf.is_stationary,
            'kpss_statistic': kpss.test_statistic,
            'kpss_pvalue': kpss.p_value,
            'kpss_stationary': kpss.is_stationary,
            'both_stationary': adf.is_stationary and kpss.is_stationary
        })

    return pd.DataFrame(results)
```

</div>
</div>


### Transformation to Stationarity


<span class="filename">stationarity_transforms.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def make_stationary(
    series: np.ndarray,
    method: str = 'diff',
    **kwargs
) -> Tuple[np.ndarray, Dict]:
    """
    Transform series to achieve stationarity.

    Parameters
    ----------
    series : np.ndarray
        Non-stationary time series
    method : str
        'diff': differencing
        'log_diff': log + differencing
        'detrend': remove linear trend
        'seasonal_diff': seasonal differencing
        'box_cox': Box-Cox transformation + differencing

    Returns
    -------
    transformed : np.ndarray
        Stationary series
    transform_info : Dict
        Information needed to invert transformation

    Examples
    --------
    >>> # Random walk (non-stationary)
    >>> series = np.random.randn(100).cumsum()
    >>> stationary, info = make_stationary(series, method='diff')
    >>> # Now stationary (returns of random walk are white noise)
    """
    if method == 'diff':
        order = kwargs.get('order', 1)
        transformed = np.diff(series, n=order)
        transform_info = {'method': 'diff', 'order': order}

    elif method == 'log_diff':
        order = kwargs.get('order', 1)
        # Ensure positive values
        series_positive = series - series.min() + 1e-8
        log_series = np.log(series_positive)
        transformed = np.diff(log_series, n=order)
        transform_info = {
            'method': 'log_diff',
            'order': order,
            'min_value': series.min()
        }

    elif method == 'detrend':
        # Remove linear trend
        t = np.arange(len(series))
        coeffs = np.polyfit(t, series, deg=1)
        trend = np.polyval(coeffs, t)
        transformed = series - trend
        transform_info = {
            'method': 'detrend',
            'coefficients': coeffs
        }

    elif method == 'seasonal_diff':
        period = kwargs.get('period', 12)
        transformed = series[period:] - series[:-period]
        transform_info = {
            'method': 'seasonal_diff',
            'period': period
        }

    elif method == 'box_cox':
        from scipy.stats import boxcox
        # Ensure positive values
        series_positive = series - series.min() + 1e-8
        transformed_bc, lambda_param = boxcox(series_positive)
        # Then difference
        order = kwargs.get('order', 1)
        transformed = np.diff(transformed_bc, n=order)
        transform_info = {
            'method': 'box_cox',
            'lambda': lambda_param,
            'min_value': series.min(),
            'diff_order': order
        }

    else:
        raise ValueError(f"Unknown method: {method}")

    return transformed, transform_info


def invert_transformation(
    transformed: np.ndarray,
    transform_info: Dict,
    original_values: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Invert stationarity transformation.

    Parameters
    ----------
    transformed : np.ndarray
        Stationary series
    transform_info : Dict
        Information from make_stationary
    original_values : np.ndarray, optional
        Original series (needed for differencing inverse)

    Returns
    -------
    np.ndarray
        Original scale series
    """
    method = transform_info['method']

    if method == 'diff':
        order = transform_info['order']
        if original_values is None:
            raise ValueError("Need original_values to invert differencing")

        # Cumulative sum to invert differencing
        inverted = np.concatenate([original_values[:order], transformed])
        for _ in range(order):
            inverted = np.cumsum(inverted)
        return inverted

    elif method == 'log_diff':
        # Invert differencing, then exp
        order = transform_info['order']
        min_val = transform_info['min_value']

        if original_values is None:
            raise ValueError("Need original_values to invert log differencing")

        log_original = np.log(original_values - min_val + 1e-8)
        log_inverted = np.concatenate([log_original[:order], transformed])

        for _ in range(order):
            log_inverted = np.cumsum(log_inverted)

        inverted = np.exp(log_inverted) + min_val - 1e-8
        return inverted

    elif method == 'detrend':
        # Add trend back
        coeffs = transform_info['coefficients']
        t = np.arange(len(transformed))
        trend = np.polyval(coeffs, t)
        return transformed + trend

    elif method == 'seasonal_diff':
        period = transform_info['period']
        if original_values is None:
            raise ValueError("Need original_values to invert seasonal differencing")

        inverted = np.zeros(len(transformed) + period)
        inverted[:period] = original_values[:period]
        inverted[period:] = transformed + original_values[:-period]
        return inverted

    else:
        raise ValueError(f"Inversion not implemented for method: {method}")


def auto_make_stationary(
    series: np.ndarray,
    max_diff: int = 2,
    alpha: float = 0.05
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Automatically determine and apply transformations for stationarity.

    Parameters
    ----------
    series : np.ndarray
        Time series
    max_diff : int
        Maximum differencing order to try
    alpha : float
        Significance level for stationarity tests

    Returns
    -------
    stationary_series : np.ndarray
        Transformed stationary series
    transformations : List[Dict]
        Sequence of transformations applied

    Algorithm
    ---------
    1. Test for stationarity
    2. If non-stationary, try transformations:
       - Differencing
       - Log + differencing
       - Seasonal differencing (if seasonality detected)
    3. Return first transformation that achieves stationarity
    """
    transformations = []
    current = series.copy()

    # Check initial stationarity
    adf = adf_test(current, alpha=alpha)

    if adf.is_stationary:
        return current, transformations

    # Try differencing
    for order in range(1, max_diff + 1):
        current_diff, info = make_stationary(current, method='diff', order=order)
        adf = adf_test(current_diff, alpha=alpha)

        transformations.append(info)

        if adf.is_stationary:
            return current_diff, transformations

        current = current_diff

    # If still not stationary, try log transformation
    if not adf.is_stationary and np.all(series > 0):
        current_log_diff, info = make_stationary(series, method='log_diff', order=1)
        adf = adf_test(current_log_diff, alpha=alpha)

        if adf.is_stationary:
            return current_log_diff, [info]

    # Return best attempt
    return current, transformations
```

</div>
</div>

### GA Feature Selection with Stationarity Handling


<span class="filename">stationary_fitness.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def stationary_feature_selection_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn,
    cv_splits: List,
    check_stationarity: bool = True,
    alpha_stationarity: float = 0.1
) -> float:
    """
    Fitness function that penalizes selection of non-stationary features.

    Parameters
    ----------
    check_stationarity : bool
        Whether to test selected features for stationarity
    alpha_stationarity : float
        Penalty weight for non-stationary features

    Returns
    -------
    float
        Fitness (lower is better)
    """
    from sklearn.metrics import mean_squared_error

    # Select features
    selected_features = np.where(chromosome == 1)[0]

    if len(selected_features) == 0:
        return float('inf')

    X_selected = X[:, selected_features]

    # Evaluate on cross-validation
    fold_scores = []

    for fold in cv_splits:
        X_train = X_selected[fold.train_indices]
        y_train = y[fold.train_indices]
        X_test = X_selected[fold.test_indices]
        y_test = y[fold.test_indices]

        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        fold_scores.append(mse)

    avg_mse = np.mean(fold_scores)

    # Stationarity penalty
    stationarity_penalty = 0.0

    if check_stationarity:
        n_nonstationary = 0

        for idx in selected_features:
            feature_series = X[:, idx]

            # Test stationarity
            try:
                adf = adf_test(feature_series, alpha=0.05)
                if not adf.is_stationary:
                    n_nonstationary += 1
            except:
                # If test fails, assume non-stationary
                n_nonstationary += 1

        stationarity_penalty = alpha_stationarity * n_nonstationary

    fitness = avg_mse + stationarity_penalty

    return fitness


def prepare_stationary_features(
    X: pd.DataFrame,
    methods: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Transform all features to stationary versions.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix
    methods : Dict[str, str], optional
        Mapping from column name to transformation method
        If None, auto-determines method for each column

    Returns
    -------
    X_stationary : pd.DataFrame
        Transformed features
    transform_info : Dict
        Information about transformations applied

    Examples
    --------
    >>> X_stat, info = prepare_stationary_features(X)
    >>> # All features now stationary
    """
    X_stationary = pd.DataFrame(index=X.index)
    transform_info = {}

    for col in X.columns:
        series = X[col].values

        # Determine method
        if methods and col in methods:
            method = methods[col]
            transformed, info = make_stationary(series, method=method)
        else:
            # Auto-determine
            transformed, transformations = auto_make_stationary(series)
            info = {'auto_transforms': transformations}

        # Handle length mismatch from differencing
        if len(transformed) < len(series):
            # Pad with NaN at the beginning
            padded = np.full(len(series), np.nan)
            padded[-len(transformed):] = transformed
            transformed = padded

        X_stationary[col] = transformed
        transform_info[col] = info

    # Drop NaN rows
    X_stationary = X_stationary.dropna()

    return X_stationary, transform_info
```

</div>
</div>

### Complete Example

```python
def stationarity_example():
    """
    Demonstrate impact of stationarity on feature selection.
    """
    np.random.seed(42)
    n = 500

    print("Stationarity in Feature Selection")
    print("=" * 70)

    # Create non-stationary features
    # Feature 1: Random walk (integrated I(1))
    f1 = np.random.randn(n).cumsum()

    # Feature 2: Trend
    f2 = np.arange(n) * 0.5 + np.random.randn(n)

    # Feature 3: Stationary AR(1)
    f3 = np.zeros(n)
    f3[0] = np.random.randn()
    for t in range(1, n):
        f3[t] = 0.7 * f3[t-1] + np.random.randn()

    # Feature 4: Seasonal + noise
    f4 = 3 * np.sin(2 * np.pi * np.arange(n) / 50) + np.random.randn(n)

    # Feature 5: White noise (stationary)
    f5 = np.random.randn(n)

    # Target: combination of stationary components
    y = 2 * f3 + 1.5 * f4 + 0.5 * np.random.randn(n)

    # Create feature matrix
    X = np.column_stack([f1, f2, f3, f4, f5])
    feature_names = ['Random_Walk', 'Trend', 'AR1', 'Seasonal', 'White_Noise']

    # Test stationarity of each feature
    print("\nStep 1: Test stationarity of features")
    print("-" * 70)

    for i, name in enumerate(feature_names):
        result = adf_test(X[:, i])
        status = "STATIONARY" if result.is_stationary else "NON-STATIONARY"
        print(f"{name:15s}: {status:15s} (p={result.p_value:.4f})")

    print(f"\nTarget:         STATIONARY     (by construction)")

    # Show that non-stationary features can appear useful due to spurious correlation
    print("\n\nStep 2: Correlations with target")
    print("-" * 70)

    for i, name in enumerate(feature_names):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"{name:15s}: correlation = {corr:7.4f}")

    print("\nNote: Random_Walk and Trend show correlation despite being irrelevant!")
    print("This is spurious correlation from non-stationarity.")

    # Transform to stationarity
    print("\n\nStep 3: Transform to stationary features")
    print("-" * 70)

    X_df = pd.DataFrame(X, columns=feature_names)
    X_stat, transform_info = prepare_stationary_features(X_df)

    print(f"Original features: {X.shape}")
    print(f"Stationary features: {X_stat.shape}")
    print(f"\nTransformations applied:")
    for col, info in transform_info.items():
        if 'auto_transforms' in info:
            transforms = info['auto_transforms']
            if transforms:
                print(f"  {col}: {[t['method'] for t in transforms]}")
            else:
                print(f"  {col}: already stationary")

    # Test correlations after transformation
    print("\n\nStep 4: Correlations after stationarity transformation")
    print("-" * 70)

    y_aligned = y[-len(X_stat):]  # Align with transformed features

    for col in X_stat.columns:
        corr = np.corrcoef(X_stat[col].values, y_aligned)[0, 1]
        print(f"{col:15s}: correlation = {corr:7.4f}")

    print("\nNote: Spurious correlations reduced after transformation!")

    # Visualize one non-stationary series and its transformation
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Random walk
    axes[0, 0].plot(f1)
    axes[0, 0].set_title('Random Walk (Non-Stationary)')
    axes[0, 0].set_xlabel('Time')

    axes[0, 1].plot(np.diff(f1))
    axes[0, 1].set_title('Random Walk - First Difference (Stationary)')
    axes[0, 1].set_xlabel('Time')

    # Trend
    axes[1, 0].plot(f2)
    axes[1, 0].set_title('Trend (Non-Stationary)')
    axes[1, 0].set_xlabel('Time')

    axes[1, 1].plot(np.diff(f2))
    axes[1, 1].set_title('Trend - First Difference (Stationary)')
    axes[1, 1].set_xlabel('Time')

    plt.tight_layout()
    plt.savefig('stationarity_transformations.png', dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'X': X,
        'X_stationary': X_stat,
        'y': y,
        'transform_info': transform_info
    }


if __name__ == "__main__":
    results = stationarity_example()
```

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> Regression on non-stationary data produces spurious correlations. Two unrelated random walks will show a correlation of 0.5-0.9 purely by chance. If your GA selects non-stationary features, you are likely finding artifacts, not real predictive relationships.

</div>

### 1. Ignoring Non-Stationarity

**Problem**: Selected features work in training but fail in production.

```python
# Bad - no stationarity check
fitness = mean_squared_error(y_true, y_pred)

# Good - penalize non-stationary features
fitness = mse + alpha * n_nonstationary_features
```

### 2. Over-Differencing

**Problem**: Differencing stationary data introduces unnecessary complexity.

```python
# Bad - blindly difference everything
X_diff = np.diff(X, axis=0)

# Good - test first, only transform if needed
X_stat, info = auto_make_stationary(X)
```

<div class="callout-warning">

<strong>Warning:</strong> Over-differencing a stationary series introduces artificial negative autocorrelation at lag 1. Always test stationarity before differencing -- if the ADF test already rejects the unit root null, do not difference.

</div>

### 3. Mixing Stationary and Non-Stationary Features

**Problem**: Regression on mixed stationarity can give spurious results.

```python
# Bad - mix of levels and differences
X_mixed = np.column_stack([price, np.diff(volume)])

# Good - all features same stationarity
X_all_diff = np.column_stack([np.diff(price), np.diff(volume)])
```

### 4. Not Accounting for Structural Breaks

**Problem**: Series appears non-stationary due to regime change.

```python
# Solution: Test for structural breaks
from statsmodels.stats.diagnostic import breaks_cusumolsresid

# Or: use rolling window stationarity tests
def rolling_stationarity_test(series, window=100):
    """Test stationarity in rolling windows."""
    pass
```

## Connections

<div class="callout-info">

ℹ️ **How this connects to the rest of the course:**

</div>

### Prerequisites
- Time series basics
- Hypothesis testing
- Autocorrelation concepts

### Leads To
- Cointegration (non-stationary series with stationary linear combination)
- Error correction models
- Unit root tests (ADF, PP, KPSS)
- Seasonal adjustment methods

### Related Concepts
- Spurious regression
- Granger causality
- ARIMA modeling
- Regime-switching models

## Practice Problems

### Problem 1: Seasonal Stationarity

Implement test for seasonal stationarity.

```python
def seasonal_stationarity_test(
    series: np.ndarray,
    period: int = 12
) -> StationarityTestResult:
    """
    Test stationarity within each season.

    Example: For monthly data, test if January values are stationary,
    February values are stationary, etc.
    """
    pass
```

### Problem 2: Structural Break Detection

Implement structural break detection before stationarity testing.

```python
def detect_structural_breaks(
    series: np.ndarray,
    max_breaks: int = 3
) -> List[int]:
    """
    Detect structural breaks (regime changes).

    Returns
    -------
    List[int]
        Indices where breaks occur

    Use: Bai-Perron test or CUSUM test
    """
    pass
```

### Problem 3: Cointegration-Based Feature Selection

Implement GA that selects cointegrated feature sets.

```python
def cointegrated_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    alpha_coint: float = 0.05
) -> np.ndarray:
    """
    Select features that are cointegrated with target.

    Even if X and y are non-stationary individually,
    they might have stationary linear combination.

    Use: Engle-Granger test or Johansen test
    """
    pass
```

### Problem 4: Adaptive Stationarity Monitoring

Implement online monitoring of stationarity.

```python
class StationarityMonitor:
    """
    Monitor stationarity of features in production.

    Alert when feature becomes non-stationary (regime change).
    """

    def update(self, new_data: np.ndarray):
        """Add new data and test stationarity."""
        pass

    def is_stationary(self, window: int = 100) -> bool:
        """Test recent window for stationarity."""
        pass
```

### Problem 5: Multi-Level Differencing Strategy

Implement intelligent differencing strategy.

```python
def optimal_differencing_strategy(
    series: np.ndarray,
    max_order: int = 2,
    max_seasonal: int = 1,
    seasonal_period: Optional[int] = None
) -> Dict:
    """
    Determine optimal combination of:
    - Regular differencing order (d)
    - Seasonal differencing order (D)
    - Seasonal period (m)

    Returns ARIMA(p,d,q)(P,D,Q)[m] structure

    Use AIC/BIC to select best combination.
    """
    pass
```

### Problem 6: Conceptual — Spurious Feature Selection

**Task:** You run a GA on raw (non-stationary) commodity price data and it selects "US Dollar Index level" and "Cumulative global production" as the top features with fitness 0.92. You then difference all features and the target, re-run the GA, and it selects "change in inventory-to-use ratio" and "change in shipping rates" with fitness 0.58. Explain why the first GA found higher fitness but worse features. Which set of features would you deploy in production, and why?

### Problem 7: Conceptual — Gap Size Tradeoff

**Task:** Your daily financial time series has an ACF that drops below the significance threshold at lag 15. You have 750 data points and want to run walk-forward validation with 5 splits. Explain the tradeoff you face in setting the gap size. What happens if you use gap=0? What happens if you use gap=50? What gap would you recommend and why?

## Further Reading

### Academic Papers

- Dickey, D. A., & Fuller, W. A. (1979). "Distribution of the estimators for autoregressive time series with a unit root". Journal of the American Statistical Association, 74(366a), 427-431.
  - Original ADF test paper

- Kwiatkowski, D., et al. (1992). "Testing the null hypothesis of stationarity against the alternative of a unit root". Journal of Econometrics, 54(1-3), 159-178.
  - KPSS test (complementary to ADF)

- Granger, C. W. J., & Newbold, P. (1974). "Spurious regressions in econometrics". Journal of Econometrics, 2(2), 111-120.
  - Dangers of regression on non-stationary data

### Books

- Hamilton, J. D. (1994). "Time Series Analysis"
  - Comprehensive treatment of stationarity and unit roots

- Enders, W. (2014). "Applied Econometric Time Series" (4th ed.)
  - Practical guide to stationarity testing and transformation

### Online Resources

- Penn State: STAT 510 - https://online.stat.psu.edu/stat510/
  - Excellent course notes on stationarity

- Statsmodels Documentation: https://www.statsmodels.org/stable/tsa.html
  - Implementation of all major stationarity tests

### Key Insights

1. **Always test stationarity** before feature selection on time series
2. **Use both ADF and KPSS** - they test different hypotheses
3. **Differencing usually sufficient** - first difference makes most series stationary
4. **Watch for over-differencing** - can introduce spurious autocorrelation
5. **Cointegration matters** - non-stationary features can be useful if cointegrated with target
---

**Next:** [Companion Slides](./03_stationarity_slides.md) | [Notebook](../notebooks/02_lag_selection.ipynb)

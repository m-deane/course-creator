# Lag Feature Selection for Time Series

> **Reading time:** ~18 min | **Module:** 3 — Time Series | **Prerequisites:** 01 Walk-Forward Validation

## In Brief

Lag features capture temporal dependencies by including past values of variables as predictors. Genetic algorithms for lag feature selection must balance autocorrelation benefits against multicollinearity risks while respecting the temporal ordering constraint that predictions cannot use future information.

<div class="callout-key">

**Key Concept Summary:** Lag features are the bridge between raw time series and supervised learning: they encode "what happened K steps ago" as predictor variables. ACF and PACF analysis identifies which lags carry predictive information, and the GA then searches for the optimal *combination* of lags -- something statistical tests alone cannot determine because they evaluate lags independently.

</div>

<div class="callout-insight">

Not all lags are equally informative. The optimal lag set depends on the autocorrelation structure (ACF/PACF), seasonal patterns, and forecast horizon. GA feature selection can automatically discover these patterns, but must penalize redundant highly-correlated lags to avoid overfitting and numerical instability.

</div>


![Walk-Forward Timeline](./walk_forward_timeline.svg)

## Formal Definition

### Lag Feature Construction

Given time series $\{y_1, y_2, ..., y_T\}$, a lag-$k$ feature is:

$$x_t^{(k)} = y_{t-k}$$

For multivariate series with $p$ variables $\{y_t^{(1)}, ..., y_t^{(p)}\}$:

$$X_t = [y_{t-1}^{(1)}, ..., y_{t-L}^{(1)}, y_{t-1}^{(2)}, ..., y_{t-L}^{(2)}, ..., y_{t-1}^{(p)}, ..., y_{t-L}^{(p)}]$$

Total features: $n = p \times L$ (variables × max lag)

### Autocorrelation Function (ACF)

$$\rho(k) = \frac{\text{Cov}(y_t, y_{t-k})}{\text{Var}(y_t)} = \frac{\sum_{t=k+1}^{T}(y_t - \bar{y})(y_{t-k} - \bar{y})}{\sum_{t=1}^{T}(y_t - \bar{y})^2}$$

Significant lags: $|\rho(k)| > \frac{1.96}{\sqrt{T}}$ (95% confidence)

### Partial Autocorrelation Function (PACF)

$$\phi_{kk} = \text{Corr}(y_t, y_{t-k} | y_{t-1}, ..., y_{t-k+1})$$

Measures correlation at lag $k$ after removing effects of intermediate lags.

### Multicollinearity Measure

Variance Inflation Factor for feature $j$:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

where $R_j^2$ is the $R^2$ from regressing $x_j$ on all other features.

Rule of thumb: $\text{VIF} > 10$ indicates problematic multicollinearity.

## Intuitive Explanation

Imagine predicting tomorrow's temperature. Yesterday's temperature is obviously useful (strong lag-1 autocorrelation). But what about the temperature from 2, 3, or 365 days ago?

**Lag-365** (one year ago) might be very useful due to seasonality—Christmas temperatures tend to be similar year-to-year. But **lag-364** is probably redundant if you already have lag-365, because the temperatures from 364 and 365 days ago are nearly identical.

**The challenge**:
- ACF tells you which individual lags correlate with the target
- PACF tells you which lags add new information beyond shorter lags
- But neither accounts for feature interactions or the specific model you're using

**Genetic algorithms** can discover useful lag combinations that:
- Capture different types of patterns (trend, seasonality, cycles)
- Avoid redundant highly-correlated lags
- Work well with your specific model architecture

**Common patterns discovered**:
- **Daily data**: lags {1, 7} (yesterday + last week)
- **Hourly data**: lags {1, 24, 168} (last hour, yesterday same time, last week same time)
- **Financial data**: lags {1, 5, 20} (previous day, week, month)

## Code Implementation

### Lag Feature Engineering


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">create_lag_features.py</span>
</div>

```python
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union, Dict
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt


def create_lag_features(
    data: Union[pd.DataFrame, np.ndarray],
    lags: List[int],
    columns: Optional[List[str]] = None,
    dropna: bool = True
) -> pd.DataFrame:
    """
    Create lag features from time series data.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        Time series data (n_samples, n_variables)
    lags : List[int]
        List of lag values to create
    columns : List[str], optional
        Column names (for numpy arrays)
    dropna : bool
        Whether to drop rows with NaN values

    Returns
    -------
    pd.DataFrame
        DataFrame with original data and lag features

    Examples
    --------
    >>> data = pd.DataFrame({'y': [1, 2, 3, 4, 5]})
    >>> df_lags = create_lag_features(data, lags=[1, 2])
    >>> # Creates columns: y, y_lag1, y_lag2
    """
    # Convert to DataFrame if needed
    if isinstance(data, np.ndarray):
        if columns is None:
            columns = [f'var_{i}' for i in range(data.shape[1])]
        data = pd.DataFrame(data, columns=columns)

    df = data.copy()

    # Create lag features
    for col in data.columns:
        for lag in lags:
            lag_col_name = f'{col}_lag{lag}'
            df[lag_col_name] = df[col].shift(lag)

    if dropna:
        df = df.dropna()

    return df


def create_seasonal_lags(
    data: pd.DataFrame,
    seasonality: int,
    n_seasons: int = 2,
    include_recent: bool = True
) -> pd.DataFrame:
    """
    Create seasonal lag features.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data
    seasonality : int
        Seasonal period (e.g., 7 for weekly, 12 for monthly)
    n_seasons : int
        Number of seasonal lags to include
    include_recent : bool
        Whether to include recent lags (1, 2, ..., seasonality-1)

    Examples
    --------
    >>> # Daily data with weekly seasonality
    >>> df_lags = create_seasonal_lags(data, seasonality=7, n_seasons=4)
    >>> # Creates lags: 7, 14, 21, 28 (and 1-6 if include_recent=True)
    """
    # Seasonal lags
    seasonal_lags = [seasonality * i for i in range(1, n_seasons + 1)]

    # Recent lags (within one season)
    if include_recent:
        recent_lags = list(range(1, seasonality))
        all_lags = recent_lags + seasonal_lags
    else:
        all_lags = seasonal_lags

    return create_lag_features(data, lags=all_lags, dropna=True)


def compute_acf_pacf(
    series: np.ndarray,
    max_lag: int = 40,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute autocorrelation and partial autocorrelation functions.

    Parameters
    ----------
    series : np.ndarray
        Time series values
    max_lag : int
        Maximum lag to compute
    alpha : float
        Significance level for confidence intervals

    Returns
    -------
    acf_values : np.ndarray
        Autocorrelation values for lags 0 to max_lag
    pacf_values : np.ndarray
        Partial autocorrelation values for lags 0 to max_lag
    confidence_interval : float
        Critical value for significance testing

    Examples
    --------
    >>> acf, pacf, ci = compute_acf_pacf(series, max_lag=20)
    >>> significant_lags = np.where(np.abs(acf[1:]) > ci)[0] + 1
    """
    from statsmodels.tsa.stattools import acf, pacf as pacf_func

    # Compute ACF
    acf_values = acf(series, nlags=max_lag, fft=True)

    # Compute PACF
    pacf_values = pacf_func(series, nlags=max_lag, method='ywmle')

    # Confidence interval (Bartlett's formula for ACF)
    n = len(series)
    confidence_interval = stats.norm.ppf(1 - alpha/2) / np.sqrt(n)

    return acf_values, pacf_values, confidence_interval


def plot_acf_pacf(
    series: np.ndarray,
    max_lag: int = 40,
    figsize: Tuple[int, int] = (14, 5)
):
    """
    Plot ACF and PACF with significance bands.
    """
    acf_vals, pacf_vals, ci = compute_acf_pacf(series, max_lag=max_lag)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # ACF plot
    lags = np.arange(len(acf_vals))
    ax1.stem(lags, acf_vals, basefmt=' ')
    ax1.axhline(y=0, color='black', linewidth=0.8)
    ax1.axhline(y=ci, color='red', linestyle='--', linewidth=1, label=f'95% CI')
    ax1.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('Autocorrelation')
    ax1.set_title('Autocorrelation Function (ACF)')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # PACF plot
    ax2.stem(lags, pacf_vals, basefmt=' ')
    ax2.axhline(y=0, color='black', linewidth=0.8)
    ax2.axhline(y=ci, color='red', linestyle='--', linewidth=1, label=f'95% CI')
    ax2.axhline(y=-ci, color='red', linestyle='--', linewidth=1)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Partial Autocorrelation')
    ax2.set_title('Partial Autocorrelation Function (PACF)')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    return fig


def identify_significant_lags(
    series: np.ndarray,
    max_lag: int = 40,
    method: str = 'both',
    alpha: float = 0.05
) -> Dict[str, List[int]]:
    """
    Identify statistically significant lags.

    Parameters
    ----------
    series : np.ndarray
        Time series
    max_lag : int
        Maximum lag to consider
    method : str
        'acf', 'pacf', or 'both'
    alpha : float
        Significance level

    Returns
    -------
    Dict[str, List[int]]
        Dictionary with 'acf' and/or 'pacf' significant lags
    """
    acf_vals, pacf_vals, ci = compute_acf_pacf(series, max_lag, alpha)

    results = {}

    if method in ['acf', 'both']:
        # Significant ACF lags (excluding lag 0)
        significant_acf = np.where(np.abs(acf_vals[1:]) > ci)[0] + 1
        results['acf'] = significant_acf.tolist()

    if method in ['pacf', 'both']:
        # Significant PACF lags (excluding lag 0)
        significant_pacf = np.where(np.abs(pacf_vals[1:]) > ci)[0] + 1
        results['pacf'] = significant_pacf.tolist()

    return results
```

</div>
</div>


### Multicollinearity Detection and Handling


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">multicollinearity.py</span>
</div>

```python
def compute_vif(X: pd.DataFrame) -> pd.Series:
    """
    Compute Variance Inflation Factor for each feature.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix

    Returns
    -------
    pd.Series
        VIF values for each feature

    Notes
    -----
    VIF > 10: high multicollinearity
    VIF > 5: moderate multicollinearity
    """
    from sklearn.linear_model import LinearRegression

    vif_data = {}

    for i, col in enumerate(X.columns):
        # Regress feature i on all other features
        X_others = X.drop(columns=[col])
        y_target = X[col]

        model = LinearRegression()
        model.fit(X_others, y_target)
        r_squared = model.score(X_others, y_target)

        # VIF = 1 / (1 - R²)
        vif = 1 / (1 - r_squared) if r_squared < 0.9999 else float('inf')
        vif_data[col] = vif

    return pd.Series(vif_data).sort_values(ascending=False)


def correlation_matrix_lags(
    data: pd.DataFrame,
    lags: List[int],
    target_col: str = 'y'
) -> pd.DataFrame:
    """
    Compute correlation matrix for lag features.

    Helps visualize multicollinearity among lags.
    """
    df_lags = create_lag_features(data[[target_col]], lags=lags)
    corr_matrix = df_lags.corr()
    return corr_matrix


def remove_redundant_lags(
    X: pd.DataFrame,
    threshold: float = 0.95
) -> pd.DataFrame:
    """
    Remove highly correlated lag features.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with lag features
    threshold : float
        Correlation threshold (default: 0.95)

    Returns
    -------
    pd.DataFrame
        Feature matrix with redundant lags removed

    Notes
    -----
    For each pair of features with |correlation| > threshold,
    removes the one with higher average correlation to other features.
    """
    corr_matrix = X.corr().abs()

    # Find highly correlated pairs
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()

    for column in upper_triangle.columns:
        # Find features highly correlated with this one
        high_corr_features = upper_triangle.index[upper_triangle[column] > threshold].tolist()

        if high_corr_features:
            # Keep feature with lower average correlation to others
            avg_corr_current = corr_matrix[column].mean()

            for other_feature in high_corr_features:
                avg_corr_other = corr_matrix[other_feature].mean()

                # Drop the one with higher average correlation
                if avg_corr_current > avg_corr_other:
                    to_drop.add(column)
                else:
                    to_drop.add(other_feature)

    # Remove identified features
    X_reduced = X.drop(columns=list(to_drop))

    print(f"Removed {len(to_drop)} redundant features: {sorted(to_drop)}")
    print(f"Remaining features: {X_reduced.shape[1]}")

    return X_reduced
```

</div>
</div>

## From ACF/PACF to GA Configuration

The statistical analysis above (ACF, PACF, VIF) produces useful diagnostics, but translating those results into GA configuration requires deliberate decisions. This section bridges the gap between "I ran ACF/PACF" and "I configured my GA."

### 1. PACF Significant Lags Inform `max_lag`

The PACF identifies which specific lags contribute new information beyond shorter lags. If PACF is significant at lags 1, 2, 7, and 50 but insignificant beyond lag 50, set `max_lag = 50`. There is no value in letting the GA search over lags 51-200 -- the PACF tells you those lags add no direct predictive information. This reduces the chromosome length from 200 bits to 50 bits, making the GA search dramatically more efficient.

<div class="callout-insight">

Think of PACF as a pre-filter: it tells the GA which lags are *candidates*. The GA then determines which *combinations* of those candidates work best together -- something PACF cannot answer because it tests each lag in isolation.

</div>

### 2. ACF Decay Rate Informs Population Seeding

The ACF decay pattern reveals the series' memory structure:

- **Slow ACF decay** (significant out to lag 30+): The series has long memory. Seed the initial population with chromosomes that include a mix of short, medium, and long lags. A random initialization might miss the long lags entirely if the chromosome is sparse.
- **Fast ACF decay** (insignificant by lag 5): The series has short memory. Most useful information is in the first few lags. You can use smaller chromosomes and standard random initialization will work fine.
- **ACF with periodic spikes** (e.g., significant at lags 7, 14, 21): The series has seasonality. Seed some individuals with seasonal lag patterns (e.g., every 7th lag) to give the GA a head start on discovering periodic structure.

### 3. Lag Groups Guide Mutation Operator Design

Consecutive lags (lag-1, lag-2, lag-3) are highly correlated with each other. If the GA flips one bit at a time (standard bit-flip mutation), it frequently creates chromosomes with redundant adjacent lags -- wasting capacity without improving prediction.

Design lag-group-aware mutation that operates on meaningful clusters:

- **Recent lags group**: lags 1-5 (short-term dynamics)
- **Medium-term group**: lags 6-20 (weekly/business-cycle patterns)
- **Seasonal lag group**: lags at the seasonal period (e.g., 7, 14, 21 for daily data with weekly seasonality)

The `lag_aware_mutation` function in the code below implements this idea. When it mutates, it flips an entire group on or off, maintaining coherent lag patterns rather than creating random bit noise.

<div class="callout-warning">

**Warning:** Do not skip ACF/PACF analysis and go straight to the GA. Without it, you are asking the GA to search blindly over potentially hundreds of lag features. ACF/PACF analysis takes 5 minutes and can reduce your search space by 10x.

</div>

### GA Fitness Function for Lag Selection


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">lag_selection_fitness.py</span>
</div>

```python
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from typing import Callable


def lag_selection_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable[[], BaseEstimator],
    cv_splits: List,
    alpha_complexity: float = 0.01,
    alpha_vif: float = 0.05,
    vif_threshold: float = 10.0
) -> float:
    """
    Fitness function for lag feature selection with multicollinearity penalty.

    Parameters
    ----------
    chromosome : np.ndarray
        Binary chromosome indicating selected lag features
    X : np.ndarray
        Feature matrix with all possible lag features
    y : np.ndarray
        Target values
    model_fn : Callable
        Function that returns model instance
    cv_splits : List
        Time series cross-validation splits
    alpha_complexity : float
        Penalty weight for number of features
    alpha_vif : float
        Penalty weight for multicollinearity (VIF)
    vif_threshold : float
        VIF threshold above which penalty is applied

    Returns
    -------
    float
        Fitness value (lower is better)

    Notes
    -----
    Fitness = MSE + alpha_complexity * n_features + alpha_vif * VIF_penalty
    """
    # Select features
    selected_features = np.where(chromosome == 1)[0]

    if len(selected_features) == 0:
        return float('inf')

    X_selected = X[:, selected_features]

    # Evaluate on cross-validation folds
    fold_scores = []

    for fold in cv_splits:
        X_train = X_selected[fold.train_indices]
        y_train = y[fold.train_indices]
        X_test = X_selected[fold.test_indices]
        y_test = y[fold.test_indices]

        # Train and evaluate
        model = model_fn()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        fold_scores.append(mse)

    avg_mse = np.mean(fold_scores)

    # Complexity penalty
    n_features = len(selected_features)
    complexity_penalty = alpha_complexity * n_features

    # Multicollinearity penalty (VIF)
    vif_penalty = 0.0

    if len(selected_features) > 1:
        # Convert to DataFrame for VIF computation
        X_df = pd.DataFrame(X_selected)

        try:
            vif_values = compute_vif(X_df)
            # Penalize VIF values above threshold
            high_vif = vif_values[vif_values > vif_threshold]
            if len(high_vif) > 0:
                vif_penalty = alpha_vif * high_vif.sum()
        except:
            # If VIF computation fails (perfect collinearity), heavy penalty
            vif_penalty = alpha_vif * 1000

    # Total fitness
    fitness = avg_mse + complexity_penalty + vif_penalty

    return fitness


def lag_aware_mutation(
    chromosome: np.ndarray,
    lag_groups: List[List[int]],
    mutation_rate: float = 0.1
) -> np.ndarray:
    """
    Lag-aware mutation that operates on lag groups.

    Parameters
    ----------
    chromosome : np.ndarray
        Binary chromosome
    lag_groups : List[List[int]]
        Groups of related lag features
        Example: [[0,1,2], [3,4,5]] for two variables with 3 lags each
    mutation_rate : float
        Probability of mutating each group

    Returns
    -------
    np.ndarray
        Mutated chromosome

    Notes
    -----
    Instead of flipping individual bits, this mutation:
    - Selects or deselects entire lag groups together
    - Helps maintain coherent lag patterns
    """
    mutant = chromosome.copy()

    for group in lag_groups:
        if np.random.random() < mutation_rate:
            # Flip entire group
            current_state = mutant[group[0]]
            new_state = 1 - current_state
            mutant[group] = new_state

    return mutant
```

</div>
</div>

### Complete Example

```python
def lag_selection_example():
    """
    Complete example: GA-based lag feature selection for time series.
    """
    # Generate synthetic time series
    np.random.seed(42)
    n_samples = 500

    # Create time series with known lag structure
    # AR(2) process: y_t = 0.6*y_{t-1} + 0.3*y_{t-2} + noise
    y = np.zeros(n_samples)
    y[0] = np.random.randn()
    y[1] = 0.6 * y[0] + np.random.randn()

    for t in range(2, n_samples):
        y[t] = 0.6 * y[t-1] + 0.3 * y[t-2] + 0.1 * np.random.randn()

    # Add seasonal component
    seasonal = 2 * np.sin(2 * np.pi * np.arange(n_samples) / 50)
    y = y + seasonal

    print("Lag Feature Selection Example")
    print("=" * 70)
    print(f"Data: AR(2) + Seasonal(50)")
    print(f"Samples: {n_samples}")
    print(f"True important lags: 1, 2, 50, 100 (AR + seasonal)")
    print()

    # Analyze autocorrelation structure
    print("Step 1: Analyze ACF/PACF")
    acf_vals, pacf_vals, ci = compute_acf_pacf(y, max_lag=100)

    sig_lags = identify_significant_lags(y, max_lag=100, method='both')
    print(f"Significant ACF lags: {sig_lags['acf'][:10]}...")  # First 10
    print(f"Significant PACF lags: {sig_lags['pacf'][:10]}...")
    print()

    # Create lag features (lags 1-100)
    max_lag = 100
    df = pd.DataFrame({'y': y})
    df_lags = create_lag_features(df, lags=list(range(1, max_lag + 1)))

    # Prepare for GA
    X = df_lags.drop(columns=['y']).values
    y_target = df_lags['y'].values

    print(f"Step 2: Create feature matrix")
    print(f"Features: {X.shape[1]} (lags 1-{max_lag})")
    print(f"Samples after lag creation: {X.shape[0]}")
    print()

    # Check multicollinearity
    print("Step 3: Check multicollinearity")
    X_df = pd.DataFrame(X, columns=[f'lag{i}' for i in range(1, max_lag + 1)])
    vif = compute_vif(X_df)
    print(f"Features with VIF > 10: {len(vif[vif > 10])}")
    print(f"Max VIF: {vif.max():.2f}")
    print()

    # Visualize ACF/PACF
    fig = plot_acf_pacf(y, max_lag=100)
    plt.savefig('acf_pacf_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Test different feature selection approaches
    from sklearn.linear_model import Ridge

    # 1. All lags
    print("Step 4: Compare feature selection strategies")
    print("-" * 70)

    # Setup walk-forward validation
    from genetic_algorithms_feature_selection.modules.module_03_time_series.guides.walk_forward import (
        WalkForwardValidator
    )

    validator = WalkForwardValidator(n_splits=5, test_size=50, expanding=True)
    cv_splits = validator.split(X)

    def model_fn():
        return Ridge(alpha=1.0)

    # Test 1: All lags
    chrom_all = np.ones(X.shape[1], dtype=int)
    fitness_all = lag_selection_fitness(
        chrom_all, X, y_target, model_fn, cv_splits,
        alpha_complexity=0.001, alpha_vif=0.01
    )
    print(f"All lags ({X.shape[1]}):     Fitness = {fitness_all:.4f}")

    # Test 2: Only significant PACF lags
    chrom_pacf = np.zeros(X.shape[1], dtype=int)
    pacf_lags = [lag-1 for lag in sig_lags['pacf'] if lag <= max_lag]  # 0-indexed
    if pacf_lags:
        chrom_pacf[pacf_lags] = 1
        fitness_pacf = lag_selection_fitness(
            chrom_pacf, X, y_target, model_fn, cv_splits,
            alpha_complexity=0.001, alpha_vif=0.01
        )
        print(f"PACF lags ({chrom_pacf.sum()}):    Fitness = {fitness_pacf:.4f}")

    # Test 3: True lags (1, 2, 50, 100) - if we knew them
    chrom_true = np.zeros(X.shape[1], dtype=int)
    true_lags = [0, 1, 49, 99]  # 0-indexed (lags 1, 2, 50, 100)
    chrom_true[true_lags] = 1
    fitness_true = lag_selection_fitness(
        chrom_true, X, y_target, model_fn, cv_splits,
        alpha_complexity=0.001, alpha_vif=0.01
    )
    print(f"True lags (4):      Fitness = {fitness_true:.4f}")

    # Test 4: Random selection
    chrom_random = (np.random.random(X.shape[1]) < 0.1).astype(int)
    chrom_random[chrom_random.sum() == 0] = 1  # Ensure at least one
    fitness_random = lag_selection_fitness(
        chrom_random, X, y_target, model_fn, cv_splits,
        alpha_complexity=0.001, alpha_vif=0.01
    )
    print(f"Random ({chrom_random.sum()}):       Fitness = {fitness_random:.4f}")

    print()
    print("Note: True lags should have lowest fitness (if penalties calibrated well)")

    return {
        'X': X,
        'y': y_target,
        'significant_lags': sig_lags,
        'vif': vif
    }


if __name__ == "__main__":
    results = lag_selection_example()
```

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> Lag features with consecutive indices (lag-1, lag-2, lag-3, ...) are almost always highly correlated (VIF > 100). Including them all causes numerical instability in linear models and wastes capacity in tree models. Always check VIF before using lag features.


### 1. Including Too Many Lags

**Problem**: Overfitting and multicollinearity from redundant lags.

```python

# Bad - 100 highly correlated lag features
df_lags = create_lag_features(data, lags=list(range(1, 101)))

# Good - selective lags based on ACF/PACF
sig_lags = identify_significant_lags(data['y'].values, max_lag=100)
df_lags = create_lag_features(data, lags=sig_lags['pacf'])
```

### 2. Ignoring Multicollinearity

**Problem**: Numerical instability and inflated standard errors.

```python

# Bad - no multicollinearity check
X_selected = X[:, chromosome == 1]

# Good - penalize high VIF in fitness
fitness = mse + alpha_vif * vif_penalty
```

### 3. Wrong Lag Interpretation

**Problem**: Confusing lag-k with k-step-ahead forecast.

```python

# lag-1 feature: uses y_{t-1} to predict y_t (1-step ahead)

# lag-k feature: uses y_{t-k} to predict y_t (still 1-step ahead!)

# For k-step ahead forecast, you need y_{t-k-1}, not y_{t-1}
```

<div class="callout-warning">

<strong>Warning:</strong> Confusing lag-k features with k-step-ahead forecasting is a common mistake. A lag-k feature uses y(t-k) to predict y(t) -- this is still a 1-step-ahead forecast. For true k-step-ahead, you need lags starting at k, not 1.


### 4. Not Accounting for Seasonality

**Problem**: Missing periodic patterns.

```python

# Bad - only recent lags
lags = list(range(1, 8))

# Good - include seasonal lags
lags = list(range(1, 8)) + [30, 60, 90, 365]  # Daily data
```



## Connections

<div class="callout-info">

ℹ️ **How this connects to the rest of the course:**


### Prerequisites
- Autocorrelation concepts (ACF, PACF)
- Walk-forward validation
- Multicollinearity understanding

### Leads To
- Stationarity requirements
- Multi-step ahead forecasting
- Feature engineering for time series
- ARIMA/SARIMA models

### Related Concepts
- Granger causality
- Vector autoregression (VAR)
- Distributed lag models
- Impulse response functions



## Practice Problems

### Problem 1: Optimal Lag Selection

Implement information criterion-based lag selection.

```python
def select_lags_by_aic(
    series: np.ndarray,
    max_lag: int = 20,
    criterion: str = 'aic'
) -> List[int]:
    """
    Select optimal lags using AIC/BIC.

    Fit AR(p) models for p=1,...,max_lag and select best.

    Returns
    -------
    List[int]
        Optimal lag set
    """
    pass
```

### Problem 2: Cross-Series Lag Features

Implement lag feature creation for multivariate time series.

```python
def create_cross_series_lags(
    data: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    max_lag: int = 10
) -> pd.DataFrame:
    """
    Create lags of predictor columns to predict target.

    Example: Predict sales using lags of: price, advertising, weather
    """
    pass
```

### Problem 3: Grouped Lag Selection

Implement GA that selects lag groups together.

```python
def grouped_lag_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    n_vars: int,
    lags_per_var: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover that keeps lags for each variable together.

    If selecting lags for 3 variables × 10 lags each:
    - Chromosome: [v1_lag1, ..., v1_lag10, v2_lag1, ..., v2_lag10, v3_lag1, ..., v3_lag10]
    - Crossover swaps entire variable blocks, not individual lags
    """
    pass
```

### Problem 4: Dynamic Lag Selection

Implement online lag selection that adapts over time.

```python
class AdaptiveLagSelector:
    """
    Maintains and updates lag selection as new data arrives.

    Re-runs GA periodically to adapt to changing patterns.
    """

    def update(self, new_data: np.ndarray):
        """
        Incorporate new data and potentially update lag selection.
        """
        pass

    def predict(self, X_recent: np.ndarray) -> float:
        """
        Predict using current lag selection.
        """
        pass
```

### Problem 5: Lag Feature Importance

Implement method to measure importance of selected lags.

```python
def analyze_lag_importance(
    X: np.ndarray,
    y: np.ndarray,
    selected_lags: List[int],
    model: BaseEstimator
) -> pd.DataFrame:
    """
    Analyze importance of selected lag features.

    Returns
    -------
    pd.DataFrame
        DataFrame with: lag, coefficient/importance, p-value, VIF
    """
    pass
```

### Problem 6: Conceptual — Why PACF, Not ACF, for Lag Selection

**Task:** Your PACF shows significant values at lags 1, 2, and 7. Your ACF shows significant values at lags 1 through 12. Explain why you should use the PACF results (not the ACF) to inform your GA's candidate lag set. What would go wrong if you included all 12 ACF-significant lags as candidates? How does this relate to multicollinearity?

### Problem 7: Conceptual — Lag Groups and Domain Knowledge

**Task:** You are predicting daily retail sales. You know the business has strong weekly patterns (weekday vs. weekend) and monthly patterns (payday effects). Describe how you would design lag groups for the GA mutation operator. What specific lags would go in each group, and why would group-aware mutation outperform standard bit-flip mutation for this problem?

## Further Reading

### Academic Papers

- Box, G. E. P., & Jenkins, G. M. (1976). "Time Series Analysis: Forecasting and Control". Holden-Day.
  - Classic reference on lag selection for ARIMA models

- Akaike, H. (1974). "A new look at the statistical model identification". IEEE Transactions on Automatic Control, 19(6), 716-723.
  - AIC for model/lag selection

- Granger, C. W. J. (1969). "Investigating causal relations by econometric models and cross-spectral methods". Econometrica, 37(3), 424-438.
  - Granger causality and lag relationships

### Books

- Hyndman, R. J., & Athanasopoulos, G. (2021). "Forecasting: Principles and Practice" (3rd ed.)
  - Chapter 9: ARIMA models and lag selection

- Hamilton, J. D. (1994). "Time Series Analysis"
  - Comprehensive treatment of lag-based models

### Online Resources

- Statsmodels Documentation: https://www.statsmodels.org/stable/tsa.html
  - ACF/PACF computation and interpretation

- Rob Hyndman's Blog: https://robjhyndman.com/hyndsight/
  - Practical advice on lag selection

### Key Insights

1. **PACF more useful than ACF** for selecting specific lags (removes intermediate effects)
2. **Seasonal lags often critical** for periodic data
3. **Multicollinearity management essential** - penalize high VIF in fitness
4. **Lag selection depends on forecast horizon** - different lags optimal for different horizons
5. **Domain knowledge invaluable** - known cycles, reporting periods, etc.
---

**Next:** [Companion Slides](./02_lag_features_slides.md) | [Notebook](../notebooks/02_lag_selection.ipynb)

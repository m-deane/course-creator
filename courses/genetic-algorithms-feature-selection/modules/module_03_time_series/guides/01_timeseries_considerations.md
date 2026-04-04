# Time Series Considerations for GA Feature Selection

> **Reading time:** ~5 min | **Module:** 3 — Time Series | **Prerequisites:** Module 2 Fitness Functions

## The Time Series Challenge

Time series feature selection has unique challenges:

1. **Temporal dependence** - observations are not i.i.d.
2. **Look-ahead bias** - future data can't inform past predictions
3. **Non-stationarity** - feature relationships may change over time
4. **Limited data** - can't simply collect more samples


![Walk-Forward Timeline](./walk_forward_timeline.svg)

## Validation Strategies

### Walk-Forward Validation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">walk_forward_split.py</span>
</div>

```python
import numpy as np
from typing import List, Tuple

def walk_forward_split(
    n_samples: int,
    n_splits: int = 5,
    train_size: float = 0.7,
    gap: int = 0
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate walk-forward validation splits.

    Args:
        n_samples: Total number of samples
        n_splits: Number of evaluation periods
        train_size: Fraction of data for training in each split
        gap: Number of samples to skip between train and test (embargo)
    """
    splits = []
    test_size = int(n_samples * (1 - train_size) / n_splits)

    for i in range(n_splits):
        test_end = n_samples - (n_splits - i - 1) * test_size
        test_start = test_end - test_size
        train_end = test_start - gap

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits

# Usage
splits = walk_forward_split(1000, n_splits=5, gap=5)
for i, (train, test) in enumerate(splits):
    print(f"Split {i}: Train {train[0]}-{train[-1]}, Test {test[0]}-{test[-1]}")
```
</div>


### Purged K-Fold

```python
def purged_kfold(
    n_samples: int,
    n_splits: int = 5,
    purge_window: int = 10
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    K-fold with purging to prevent information leakage.
    Removes samples near train/test boundaries.
    """
    fold_size = n_samples // n_splits
    splits = []

    for i in range(n_splits):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_splits - 1 else n_samples

        # Training indices with purging
        train_idx = []

        # Before test fold (purge end)
        if test_start > 0:
            train_idx.extend(range(0, max(0, test_start - purge_window)))

        # After test fold (purge start)
        if test_end < n_samples:
            train_idx.extend(range(min(n_samples, test_end + purge_window), n_samples))

        train_idx = np.array(train_idx)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits
```

## GA Fitness with Time Series CV

```python
from sklearn.ensemble import RandomForestRegressor
from deap import base, creator, tools

def create_timeseries_fitness(X, y, n_splits=5, gap=5):
    """
    Create fitness function with proper time series validation.
    """
    n_samples = len(y)
    splits = walk_forward_split(n_samples, n_splits=n_splits, gap=gap)

    def fitness_function(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected) == 0:
            return (float('inf'),)

        X_selected = X[:, selected]
        errors = []

        for train_idx, test_idx in splits:
            X_train = X_selected[train_idx]
            y_train = y[train_idx]
            X_test = X_selected[test_idx]
            y_test = y[test_idx]

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            mse = np.mean((y_test - predictions) ** 2)
            errors.append(mse)

        # Penalize feature count
        feature_penalty = 0.01 * len(selected) / len(individual)

        return (np.mean(errors) + feature_penalty,)

    return fitness_function
```

## Handling Non-Stationarity

### Rolling Feature Selection

```python
def rolling_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
    step_size: int,
    ga_params: dict
) -> List[dict]:
    """
    Run GA feature selection on rolling windows.
    Track how selected features change over time.
    """
    from deap import algorithms

    n_samples = len(y)
    results = []

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size

        X_window = X[start:end]
        y_window = y[start:end]

        # Run GA on this window
        result = run_ga(
            X_window, y_window,
            pop_size=ga_params.get('pop_size', 30),
            n_generations=ga_params.get('n_generations', 20)
        )

        results.append({
            'window_start': start,
            'window_end': end,
            'selected_features': result['selected_features'],
            'fitness': result['fitness']
        })

    return results

def analyze_feature_stability(rolling_results: List[dict], n_features: int) -> dict:
    """
    Analyze stability of selected features across windows.
    """
    # Count feature selections
    feature_counts = np.zeros(n_features)

    for result in rolling_results:
        for feat in result['selected_features']:
            feature_counts[feat] += 1

    # Normalize by number of windows
    selection_frequency = feature_counts / len(rolling_results)

    # Identify stable features (selected in >50% of windows)
    stable_features = np.where(selection_frequency > 0.5)[0].tolist()

    return {
        'selection_frequency': selection_frequency,
        'stable_features': stable_features,
        'most_stable': np.argsort(selection_frequency)[::-1][:10].tolist()
    }
```

### Regime-Aware Selection

```python
def regime_aware_fitness(
    X: np.ndarray,
    y: np.ndarray,
    regime_labels: np.ndarray
) -> callable:
    """
    Fitness function that evaluates across different market regimes.
    """
    unique_regimes = np.unique(regime_labels)

    def fitness_function(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected) == 0:
            return (float('inf'),)

        X_selected = X[:, selected]
        regime_errors = []

        for regime in unique_regimes:
            regime_mask = regime_labels == regime
            X_regime = X_selected[regime_mask]
            y_regime = y[regime_mask]

            if len(y_regime) < 20:  # Skip if too few samples
                continue

            # Simple train/test split within regime
            split = int(len(y_regime) * 0.7)
            X_train, X_test = X_regime[:split], X_regime[split:]
            y_train, y_test = y_regime[:split], y_regime[split:]

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            regime_errors.append(np.mean((y_test - pred) ** 2))

        if not regime_errors:
            return (float('inf'),)

        # Average across regimes
        return (np.mean(regime_errors),)

    return fitness_function
```

## Multi-Horizon Selection

```python
def multi_horizon_fitness(
    X: np.ndarray,
    y_dict: dict,  # {horizon: y_values}
    horizon_weights: dict = None
) -> callable:
    """
    Select features that work across multiple forecast horizons.
    """
    horizons = list(y_dict.keys())
    horizon_weights = horizon_weights or {h: 1.0/len(horizons) for h in horizons}

    def fitness_function(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]

        if len(selected) == 0:
            return (float('inf'),)

        X_selected = X[:, selected]
        weighted_error = 0

        for horizon in horizons:
            y = y_dict[horizon]
            weight = horizon_weights[horizon]

            # Adjust for horizon-specific data availability
            valid_idx = ~np.isnan(y)
            X_valid = X_selected[valid_idx]
            y_valid = y[valid_idx]

            # Time series split
            split = int(len(y_valid) * 0.7)
            X_train, X_test = X_valid[:split], X_valid[split:]
            y_train, y_test = y_valid[:split], y_valid[split:]

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            mse = np.mean((y_test - pred) ** 2)
            weighted_error += weight * mse

        return (weighted_error,)

    return fitness_function
```

## Lag Feature Handling

```python
def create_lag_features(
    X: np.ndarray,
    feature_names: List[str],
    max_lag: int = 5
) -> Tuple[np.ndarray, List[str]]:
    """
    Create lagged versions of features.
    """
    n_samples, n_features = X.shape
    lag_features = []
    lag_names = []

    for lag in range(1, max_lag + 1):
        lagged = np.roll(X, lag, axis=0)
        lagged[:lag] = np.nan  # Mark invalid rows
        lag_features.append(lagged)

        for name in feature_names:
            lag_names.append(f"{name}_lag{lag}")

    # Combine original and lagged
    X_lagged = np.column_stack([X] + lag_features)
    all_names = list(feature_names) + lag_names

    # Remove rows with NaN
    valid_rows = ~np.any(np.isnan(X_lagged), axis=1)
    X_lagged = X_lagged[valid_rows]

    return X_lagged, all_names, valid_rows

def ga_with_lag_features(X, y, feature_names, max_lag=5):
    """
    Run GA on dataset including lag features.
    """
    # Create lag features
    X_lagged, all_names, valid_rows = create_lag_features(X, feature_names, max_lag)
    y_valid = y[valid_rows]

    # Run GA
    result = run_ga(X_lagged, y_valid, pop_size=50, n_generations=30)

    # Map selected indices to feature names
    selected_names = [all_names[i] for i in result['selected_features']]

    return {
        'selected_features': result['selected_features'],
        'selected_names': selected_names,
        'fitness': result['fitness']
    }
```

## Key Takeaways

<div class="callout-key">
🔑 **Key Points**

1. **Always use temporal validation** - never shuffle time series data

2. **Walk-forward is gold standard** - mimics real forecasting conditions

3. **Purging prevents leakage** - remove observations near boundaries

4. **Non-stationarity requires rolling analysis** - features may change importance

5. **Multi-horizon testing** ensures robust feature selection
</div>
---

**Next:** [Companion Slides](./01_timeseries_considerations_slides.md) | [Notebook](../notebooks/01_walk_forward_ga.ipynb)

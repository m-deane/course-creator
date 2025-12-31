# Module 3: Time Series Considerations

## Overview

Handle the unique challenges of feature selection for time series: temporal dependencies, stationarity, and walk-forward validation.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Implement walk-forward validation in GAs
2. Handle lag feature dependencies
3. Account for stationarity in selection
4. Design time-aware fitness functions

## Contents

### Guides
- `01_walk_forward.md` - Time-respecting validation
- `02_lag_features.md` - Handling temporal dependencies
- `03_stationarity.md` - Feature transformation

### Notebooks
- `01_walk_forward_ga.ipynb` - Time series GA implementation
- `02_lag_selection.ipynb` - Optimal lag selection

## Key Concepts

### Walk-Forward Validation

```
Time: ─────────────────────────────────────────────→

Fold 1: [Train════════][Test]
Fold 2:      [Train════════][Test]
Fold 3:           [Train════════][Test]
Fold 4:                [Train════════][Test]
```

### Lag Feature Handling

```python
def create_lag_features(df, columns, lags):
    for col in columns:
        for lag in lags:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    return df

# GA selects which lags are useful
# Chromosome encodes: [col1_lag1, col1_lag2, col2_lag1, ...]
```

### Time Series Fitness

```python
def time_series_fitness(chromosome, X, y, model, n_splits=5):
    """Walk-forward cross-validation fitness."""
    selected = X[:, chromosome == 1]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(selected):
        X_train, X_test = selected[train_idx], selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores.append(-mean_squared_error(y_test, pred))

    return np.mean(scores)
```

### Stationarity Checks

- Test selected features for stationarity
- Include differencing in feature engineering
- Penalize non-stationary combinations

## Prerequisites

- Module 0-2 completed
- Time series basics

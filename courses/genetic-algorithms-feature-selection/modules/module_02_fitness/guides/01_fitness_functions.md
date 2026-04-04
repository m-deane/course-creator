# Fitness Function Design

> **Reading time:** ~5 min | **Module:** 2 — Fitness Functions | **Prerequisites:** Module 1 GA Fundamentals

## The Critical Component

The fitness function determines GA success. For feature selection:

$$\text{fitness}(\mathbf{x}) = \text{model\_error}(\mathbf{x}) + \lambda \cdot \text{complexity}(\mathbf{x})$$


![Fitness Landscape](./fitness_landscape.svg)

## Basic Fitness Functions

### Cross-Validation Error

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">cv_fitness.py</span>
</div>

```python
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import Ridge
from typing import Callable

def cv_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    cv_folds: int = 5,
    scoring: str = 'neg_mean_squared_error'
) -> float:
    """
    Fitness based on cross-validation error.
    """
    # Get selected features
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')  # Penalty for empty selection

    X_selected = X[:, selected]

    # Cross-validation
    model = model_fn()
    scores = cross_val_score(
        model, X_selected, y,
        cv=cv_folds,
        scoring=scoring
    )

    # Return negative MSE (we minimize, so positive = bad)
    return -scores.mean()
```
</div>


### Multi-Objective Fitness

```python
def multi_objective_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    cv_folds: int = 5,
    feature_penalty: float = 0.01
) -> float:
    """
    Balance prediction error and feature count.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')

    X_selected = X[:, selected]

    # Prediction error
    model = model_fn()
    scores = cross_val_score(
        model, X_selected, y,
        cv=cv_folds,
        scoring='neg_mean_squared_error'
    )
    error = -scores.mean()

    # Feature count penalty
    complexity = len(selected) * feature_penalty

    return error + complexity

def pareto_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable
) -> tuple:
    """
    Return multiple objectives for Pareto optimization.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return (float('inf'), float('inf'))

    X_selected = X[:, selected]

    model = model_fn()
    scores = cross_val_score(model, X_selected, y, cv=5, scoring='neg_mean_squared_error')

    return (
        -scores.mean(),  # Prediction error (minimize)
        len(selected)    # Feature count (minimize)
    )
```

![Feature Selection Pipeline](./feature_selection_pipeline.svg)

<div class="callout-warning">

⚠️ **Warning:** The parsimony penalty weight `lambda` controls the accuracy-complexity tradeoff. Too small and the GA selects all features; too large and it selects too few. Start with `lambda = 0.01` and tune via validation.

</div>

## Time Series Specific Fitness

### Walk-Forward Validation

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    n_splits: int = 5,
    test_size: int = None
) -> float:
    """
    Time-series appropriate fitness using walk-forward validation.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')

    X_selected = X[:, selected]

    # Time series split
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)

    errors = []
    for train_idx, test_idx in tscv.split(X_selected):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = np.mean((y_test - predictions) ** 2)
        errors.append(mse)

    return np.mean(errors)
```

### Expanding Window

```python
def expanding_window_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    min_train_size: int = 100,
    step_size: int = 50
) -> float:
    """
    Expanding window validation for time series.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')

    X_selected = X[:, selected]
    n_samples = len(y)

    errors = []
    train_end = min_train_size

    while train_end < n_samples - step_size:
        X_train = X_selected[:train_end]
        y_train = y[:train_end]

        test_start = train_end
        test_end = min(train_end + step_size, n_samples)
        X_test = X_selected[test_start:test_end]
        y_test = y[test_start:test_end]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = np.mean((y_test - predictions) ** 2)
        errors.append(mse)

        train_end += step_size

    return np.mean(errors)
```

## Avoiding Overfitting

### Nested Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

def nested_cv_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    outer_cv: int = 5,
    inner_cv: int = 3
) -> float:
    """
    Nested CV to avoid selection bias.

    Outer loop: Evaluate feature subset
    Inner loop: Tune hyperparameters (if any)
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')

    X_selected = X[:, selected]

    outer_scores = []
    outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)

    for train_idx, test_idx in outer_kfold.split(X_selected):
        X_outer_train = X_selected[train_idx]
        y_outer_train = y[train_idx]
        X_outer_test = X_selected[test_idx]
        y_outer_test = y[test_idx]

        # Inner CV for model selection/tuning
        inner_kfold = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
        inner_scores = cross_val_score(
            model_fn(), X_outer_train, y_outer_train,
            cv=inner_kfold, scoring='neg_mean_squared_error'
        )

        # Train on full outer training set
        model = model_fn()
        model.fit(X_outer_train, y_outer_train)

        # Evaluate on outer test set
        predictions = model.predict(X_outer_test)
        mse = np.mean((y_outer_test - predictions) ** 2)
        outer_scores.append(mse)

    return np.mean(outer_scores)
```

### Regularized Fitness

```python
def regularized_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    alpha: float = 0.1,
    feature_penalty: float = 0.05
) -> float:
    """
    Fitness with explicit regularization.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')

    X_selected = X[:, selected]

    # Use regularized model
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha)

    scores = cross_val_score(
        model, X_selected, y,
        cv=5, scoring='neg_mean_squared_error'
    )

    base_error = -scores.mean()

    # Additional feature count penalty
    n_features = len(selected)
    total_features = len(chromosome)
    complexity_penalty = feature_penalty * (n_features / total_features)

    return base_error * (1 + complexity_penalty)
```

## Robust Fitness Estimation

### Bootstrap Aggregation

```python
def bootstrap_fitness(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    n_bootstrap: int = 10
) -> float:
    """
    Average fitness over bootstrap samples for robustness.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf')

    X_selected = X[:, selected]
    n_samples = len(y)

    errors = []
    for _ in range(n_bootstrap):
        # Bootstrap sample
        boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
        oob_idx = np.setdiff1d(np.arange(n_samples), boot_idx)

        if len(oob_idx) < 10:  # Need enough OOB samples
            continue

        X_boot, y_boot = X_selected[boot_idx], y[boot_idx]
        X_oob, y_oob = X_selected[oob_idx], y[oob_idx]

        model = model_fn()
        model.fit(X_boot, y_boot)
        predictions = model.predict(X_oob)

        mse = np.mean((y_oob - predictions) ** 2)
        errors.append(mse)

    return np.mean(errors) if errors else float('inf')
```

### Confidence Interval

```python
def fitness_with_uncertainty(
    chromosome: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    cv_folds: int = 10
) -> tuple:
    """
    Return fitness with confidence interval.
    Useful for risk-averse selection.
    """
    selected = np.where(chromosome == 1)[0]

    if len(selected) == 0:
        return float('inf'), float('inf')

    X_selected = X[:, selected]

    model = model_fn()
    scores = cross_val_score(
        model, X_selected, y,
        cv=cv_folds,
        scoring='neg_mean_squared_error'
    )

    errors = -scores
    mean_error = np.mean(errors)
    std_error = np.std(errors)

    # Upper confidence bound (pessimistic estimate)
    ucb = mean_error + 1.96 * std_error / np.sqrt(cv_folds)

    return mean_error, ucb
```

## Performance Considerations

### Caching Fitness Evaluations

```python
from functools import lru_cache

class CachedFitnessEvaluator:
    """
    Cache fitness evaluations to avoid redundant computation.
    """

    def __init__(self, X, y, model_fn):
        self.X = X
        self.y = y
        self.model_fn = model_fn
        self.cache = {}
        self.eval_count = 0

    def evaluate(self, chromosome: np.ndarray) -> float:
        """Evaluate with caching."""
        # Convert to hashable key
        key = tuple(chromosome.tolist())

        if key not in self.cache:
            self.eval_count += 1
            self.cache[key] = cv_fitness(
                chromosome, self.X, self.y, self.model_fn
            )

        return self.cache[key]

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of cached evaluations."""
        total_calls = len(self.cache) + self.eval_count - len(self.cache)
        return 1 - (len(self.cache) / total_calls) if total_calls > 0 else 0
```

## Key Takeaways

<div class="callout-key">
🔑 **Key Points**

1. **Cross-validation is essential** - never evaluate on training data alone

2. **Time series requires temporal splits** - use walk-forward or expanding window

3. **Include parsimony pressure** - penalize complex solutions

4. **Nested CV prevents bias** - separate evaluation from selection

5. **Cache evaluations** - same chromosome should return same fitness
</div>
---

**Next:** [Companion Slides](./01_fitness_functions_slides.md) | [Notebook](../notebooks/01_fitness_functions.ipynb)

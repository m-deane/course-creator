# The Feature Selection Problem

> **Reading time:** ~9 min | **Module:** 0 — Foundations | **Prerequisites:** Linear algebra, basic probability

## In Brief

Feature selection is the process of identifying the most relevant subset of features from a larger feature space to improve model performance, reduce overfitting, and enhance interpretability. In time series forecasting with many potential features and limited observations, this becomes critical for generalization.

<div class="callout-insight">

The feature selection problem grows exponentially with the number of features: with $p$ features, there are $2^p$ possible subsets to evaluate. For even moderate feature sets (p=30), exhaustive search requires evaluating over 1 billion combinations, making intelligent search strategies essential.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Feature selection is a combinatorial search problem where the search space grows exponentially with the number of features. The core challenge is not computational speed -- it is that exhaustive evaluation becomes physically impossible beyond ~20 features, forcing you to use intelligent search strategies that find good (not perfect) subsets.

</div>

![Feature Selection Pipeline](./feature_selection_pipeline.svg)

## Intuitive Explanation

Imagine building a weather forecasting model. You have access to 100 potential features: temperature readings from various locations, humidity levels, wind speeds, historical patterns, seasonal indicators, and more. Using all 100 features would be like asking 100 experts for opinions and trying to weigh them all equally -- you'd be overwhelmed and likely misled by irrelevant signals.

Feature selection is like assembling the right team of experts. You want:
- **Relevant experts** (features that actually help predict)
- **Diverse perspectives** (features that provide unique information)
- **A manageable team** (not so many that noise dominates)

The challenge: with 100 features, there are $2^{100} \approx 10^{30}$ possible teams to evaluate -- more than the number of atoms in the human body!

This is not just a "big number" problem. Even with a supercomputer evaluating a billion teams per second, you would need $10^{21}$ seconds -- roughly 30 trillion years. The universe is only 14 billion years old. No amount of hardware solves this; you need a fundamentally different approach to searching.

## Formal Definition

**Feature Selection Problem:**

Given:
- Feature matrix $X \in \mathbb{R}^{n \times p}$ with $n$ observations and $p$ features
- Target variable $y \in \mathbb{R}^n$
- Predictive model $M$
- Performance metric $L$ (e.g., MSE, MAE)

Find: Binary selection vector $s \in \{0,1\}^p$ that minimizes:

$$s^* = \argmin_{s \in \{0,1\}^p} L(M(X_s), y) + \lambda \cdot ||s||_0$$

Where:
- $X_s$ contains only columns where $s_i = 1$
- $||s||_0$ counts selected features (parsimony penalty)
- $\lambda$ controls complexity-accuracy tradeoff

## Mathematical Formulation

### Search Space Size

The number of possible feature subsets:

$$|\mathcal{S}| = \sum_{k=0}^{p} \binom{p}{k} = 2^p$$

Concrete examples:

| Features (p) | Possible Subsets ($2^p$) | Evaluation Time* |
|--------------|-------------------------|------------------|
| 10 | 1,024 | ~1 second |
| 20 | 1,048,576 | ~17 minutes |
| 30 | 1.07 × 10^9 | ~12 days |
| 50 | 1.13 × 10^15 | ~35,000 years |
| 100 | 1.27 × 10^30 | Heat death of universe |

*Assuming 1ms per evaluation

### The Curse of Dimensionality

For time series with $n$ observations and $p$ features:

**Overfitting Risk:**
$$\text{Risk} \propto \frac{p}{n}$$

<div class="callout-warning">

When $p \approx n$ or $p > n$, most models will overfit — training error decreases while test error explodes. A ratio of $p/n > 0.2$ is a serious red flag requiring aggressive feature selection.

</div>


When $p \approx n$ or $p > n$, most models will overfit:
- Training error decreases
- Test error increases
- Model learns noise instead of signal

**Example:** Stock price prediction
- $n = 250$ trading days (1 year)
- $p = 100$ technical indicators
- Ratio: $p/n = 0.4$ (HIGH RISK)

Without selection, model has too many degrees of freedom.

### Multi-Objective Formulation

Feature selection often balances competing goals:

$$\min_{s} \left\{ \begin{array}{l}
f_1(s) = \text{Prediction Error}(X_s, y) \\
f_2(s) = ||s||_0 = \text{Number of Features} \\
f_3(s) = \text{Computational Cost}(X_s)
\end{array} \right.$$

This creates a Pareto frontier: improving one objective may worsen others.

<div class="callout-key">

<strong>Key Takeaway:</strong> With $p > 20$ features, exhaustive search is physically impossible. This is not a performance problem -- it is a mathematical certainty. Any viable approach must use intelligent search heuristics.

</div>

## Code Implementation

### Naive Exhaustive Search


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">exhaustive_feature_selection.py</span>

```python
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def exhaustive_feature_selection(X, y, max_features=10):
    """
    Exhaustively search all feature subsets.

    WARNING: Only feasible for small feature sets (p <= 15)

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    max_features : int
        Maximum subset size to consider

    Returns:
    --------
    best_features : tuple
        Indices of best feature subset
    best_score : float
        Cross-validation score of best subset
    """
    n_features = X.shape[1]

    if n_features > 15:
        raise ValueError(f"Exhaustive search infeasible for {n_features} features")

    best_score = -np.inf
    best_features = None

    # Try all subset sizes
    for k in range(1, min(max_features + 1, n_features + 1)):
        # Try all combinations of size k
        for features in combinations(range(n_features), k):
            X_subset = X[:, features]

            # Evaluate with cross-validation
            model = LinearRegression()
            scores = cross_val_score(
                model, X_subset, y,
                cv=5,
                scoring='neg_mean_squared_error'
            )
            score = np.mean(scores)

            if score > best_score:
                best_score = score
                best_features = features

    return best_features, best_score


# Example usage
np.random.seed(42)
n, p = 100, 10

# Simulate data: first 3 features are relevant
X = np.random.randn(n, p)
y = 2*X[:, 0] - 1.5*X[:, 1] + X[:, 2] + np.random.randn(n)*0.5

# Search (warning: slow for p > 15)
best_features, best_score = exhaustive_feature_selection(X, y, max_features=5)
print(f"Best features: {best_features}")
print(f"Best score: {best_score:.4f}")
```

</div>
</div>


### Time Series Feature Selection Problem


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">time_series_features.py</span>

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

def create_time_series_features(df, lags=[1, 2, 3, 5, 10],
                                  windows=[5, 10, 20]):
    """
    Create candidate features for time series.

    Generates lag features and rolling statistics.
    """
    features = pd.DataFrame(index=df.index)

    # Lag features
    for lag in lags:
        features[f'lag_{lag}'] = df['value'].shift(lag)

    # Rolling statistics
    for window in windows:
        features[f'sma_{window}'] = df['value'].rolling(window).mean()
        features[f'std_{window}'] = df['value'].rolling(window).std()
        features[f'min_{window}'] = df['value'].rolling(window).min()
        features[f'max_{window}'] = df['value'].rolling(window).max()

    # Day of week, month
    if isinstance(df.index, pd.DatetimeIndex):
        features['day_of_week'] = df.index.dayofweek
        features['month'] = df.index.month

    return features.dropna()


# Demonstration of feature explosion
df = pd.DataFrame({
    'value': np.cumsum(np.random.randn(1000))
}, index=pd.date_range('2020-01-01', periods=1000))

features = create_time_series_features(df)
print(f"Original data: 1 column")
print(f"After feature engineering: {features.shape[1]} columns")
print(f"Possible subsets: 2^{features.shape[1]} = {2**features.shape[1]:,.0f}")
print(f"Need intelligent search strategy!")
```

</div>
</div>

## Common Pitfalls

<div class="callout-danger">

<strong>Danger:</strong> Data leakage from feature selection outside cross-validation is the most common and most damaging mistake in ML pipelines. It produces models that appear to perform well in development but fail catastrophically in production.

</div>

### Pitfall 1: Selection on Full Dataset

**Problem:** Selecting features using the entire dataset before cross-validation leads to data leakage and overoptimistic performance estimates.

**Why it happens:** It's tempting to select features once then validate, but this allows test information to influence selection.

**How to avoid:**


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">proper_cv_selection.py</span>

```python

# WRONG: Feature selection before CV
selected = select_features(X, y)  # Uses all data
cv_score = cross_val_score(model, X[:, selected], y, cv=5)

# RIGHT: Feature selection inside CV
tscv = TimeSeriesSplit(n_splits=5)
scores = []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Select features on training data only
    selected = select_features(X_train, y_train)

    # Evaluate on test data
    model.fit(X_train[:, selected], y_train)
    score = model.score(X_test[:, selected], y_test)
    scores.append(score)
```

</div>
</div>

### Pitfall 2: Ignoring Feature Redundancy

**Problem:** Selecting multiple highly correlated features wastes degrees of freedom and can reduce model stability.

**Why it happens:** Optimization focuses on prediction error, not feature independence.

**How to avoid:** Add correlation penalty or constraint:
```python

# Check feature correlations
corr_matrix = np.corrcoef(X.T)

# Penalize correlated features
def fitness_with_diversity(features, X, y):
    if sum(features) == 0:
        return -1e10

    X_selected = X[:, features == 1]

    # Prediction error
    error = cross_val_mse(X_selected, y)

    # Correlation penalty
    if X_selected.shape[1] > 1:
        corr = np.corrcoef(X_selected.T)
        avg_corr = (np.sum(np.abs(corr)) - X_selected.shape[1]) / (X_selected.shape[1]**2 - X_selected.shape[1])
    else:
        avg_corr = 0

    return -error - 0.1 * avg_corr
```

### Pitfall 3: Forgetting Computational Constraints

**Problem:** Selecting many features that work well statistically but are expensive to compute or collect in production.

**Why it happens:** Offline evaluation doesn't account for real-world constraints.

**How to avoid:** Include cost in objective:
```python
feature_costs = {
    'simple_lag': 0.001,      # Cheap
    'rolling_mean': 0.005,    # Medium
    'complex_indicator': 0.1   # Expensive
}

def fitness_with_cost(features, X, y, feature_names):
    error = cross_val_mse(X[:, features == 1], y)
    cost = sum(feature_costs[feature_names[i]]
               for i, selected in enumerate(features)
               if selected)

    return -error - lambda_cost * cost
```

## Connections

<div class="callout-info">

ℹ️ **How this connects to the rest of the course:**

</div>

### Builds On
- **Linear algebra**: Understanding feature spaces and dimensionality
- **Optimization theory**: Search strategies and objective functions
- **Probability theory**: Overfitting and generalization
- **Time series fundamentals**: Stationarity, autocorrelation, temporal dependencies

### Leads To
- **Genetic algorithms**: Population-based search for feature selection (Module 1)
- **Fitness function design**: How to evaluate feature subsets (Module 2)
- **Cross-validation strategies**: Proper evaluation in time series (Module 3)
- **Multi-objective optimization**: Balancing competing goals (Module 5)

### Related To
- **Regularization** (L1/L2): Continuous alternative to discrete selection
- **Dimensionality reduction** (PCA): Transform rather than select
- **Ensemble methods**: Feature importance from random forests
- **Neural architecture search**: Similar combinatorial optimization

## Practice Problems

### Problem 1: Conceptual Understanding

**Question:** You have 25 candidate features for a time series forecast. How many possible feature subsets exist? If you can evaluate 100 subsets per second, how long would exhaustive search take?

**Solution:**
- Number of subsets: $2^{25} = 33,554,432$
- Time: $33,554,432 / 100 = 335,544$ seconds $\approx 93$ hours $\approx 3.9$ days

### Problem 2: Feature Selection Simulation

**Task:** Implement a simple forward selection algorithm that starts with no features and adds the best feature at each step.

```python
def forward_selection(X, y, max_features=5):
    """
    Simple forward selection algorithm.

    Starts with empty set, adds best feature at each step.
    """
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))

    for _ in range(max_features):
        best_score = -np.inf
        best_feature = None

        for feature in remaining:
            # Try adding this feature
            trial_features = selected + [feature]
            X_trial = X[:, trial_features]

            # Evaluate
            model = LinearRegression()
            score = cross_val_score(
                model, X_trial, y, cv=5,
                scoring='neg_mean_squared_error'
            ).mean()

            if score > best_score:
                best_score = score
                best_feature = feature

        # Add best feature
        selected.append(best_feature)
        remaining.remove(best_feature)

        print(f"Step {len(selected)}: Added feature {best_feature}, score: {best_score:.4f}")

    return selected

# Test it
selected_features = forward_selection(X, y, max_features=5)
```

### Problem 3: Time Series Feature Engineering

**Task:** Create a function that generates all possible lag features for a multivariate time series and calculates how many features result.

```python
def count_lag_features(n_series, max_lag):
    """
    Count total lag features for multivariate time series.

    Parameters:
    -----------
    n_series : int
        Number of time series
    max_lag : int
        Maximum lag to include

    Returns:
    --------
    total_features : int
        Total number of lag features
    """
    # Each series generates max_lag features
    return n_series * max_lag

# Example: 5 time series, lags 1-20
n_features = count_lag_features(n_series=5, max_lag=20)
print(f"Total features: {n_features}")
print(f"Possible subsets: 2^{n_features} = {2**n_features:,.0f}")
```

### Problem 4: Conceptual — Why Not Use All Features?

**Question:** A colleague argues: "Modern models like gradient boosting can handle irrelevant features automatically through regularization. Why bother with feature selection at all?" Give two reasons why explicit feature selection is still valuable even when using regularized models.

### Problem 5: Conceptual — Search Space Structure

**Question:** Explain why the feature selection search space is harder to navigate than a continuous optimization problem (like finding the minimum of a smooth function). What property of the search space makes gradient-based methods inapplicable?

## Further Reading

### Foundational Papers
- **Guyon & Elisseeff (2003)**: "An Introduction to Variable and Feature Selection" - Comprehensive overview of feature selection methods and theory. Essential reading for understanding the landscape.

- **Kohavi & John (1997)**: "Wrappers for Feature Subset Selection" - Introduces wrapper methods and discusses the statistical validity of feature selection.

### Time Series Specific
- **Cerqueira et al. (2020)**: "Evaluating Time Series Forecasting Models: An Empirical Study on Performance Estimation Methods" - How to properly evaluate feature selection in time series context.

### Computational Complexity
- **Amaldi & Kann (1998)**: "On the Approximability of Minimizing Nonzero Variables or Unsatisfied Relations in Linear Systems" - Proves NP-hardness of feature selection, justifying heuristic approaches.

### Practical Guides
- **Kuhn & Johnson (2013)**: "Applied Predictive Modeling" - Chapter 19 on feature selection with practical R and Python examples.

- **Brownlee (2020)**: "Feature Selection for Machine Learning" - Practical guide with code examples and real datasets.

### Advanced Topics
- **Chandrashekar & Sahin (2014)**: "A Survey on Feature Selection Methods" - Taxonomy of methods including filters, wrappers, and embedded approaches.

- **Li et al. (2017)**: "Feature Selection: A Data Perspective" - Modern survey covering recent advances including deep learning approaches.
---

**Next:** [Companion Slides](./01_feature_selection_challenge_slides.md) | [Notebook](../notebooks/01_selection_comparison.ipynb)

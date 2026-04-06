# Feature Selection Approaches

> **Reading time:** ~13 min | **Module:** 0 — Foundations | **Prerequisites:** 01 Feature Selection Problem

## In Brief

Feature selection methods fall into three main categories: **filter methods** (statistical tests independent of models), **wrapper methods** (model-based evaluation), and **embedded methods** (selection built into model training). Each approach offers different tradeoffs between computational cost, model dependency, and optimization quality.

<div class="callout-insight">

The choice of feature selection approach determines both what you optimize (statistical relevance vs. prediction accuracy) and how you search (independent evaluation vs. model-guided search). Filters are fast but ignore model interactions; wrappers are accurate but computationally expensive; embedded methods balance both but are model-specific.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> The three feature selection families -- filter, wrapper, embedded -- represent a fundamental tradeoff between computational cost and selection quality. Filters are cheap but blind to feature interactions. Wrappers see interactions but are expensive. Embedded methods are a compromise tied to a specific model. Understanding this tradeoff is essential for choosing the right approach (or combining them).

</div>

![Feature Selection Pipeline](./feature_selection_pipeline.svg)

## Intuitive Explanation

Think of choosing team members for a project:

**Filter Methods** are like reviewing resumes before interviews. You screen candidates based on measurable criteria (GPA, years of experience, certifications) without testing how they'd actually perform on your specific project. This is fast and gets rid of obviously unqualified candidates, but you might miss someone who interviews poorly yet would excel at your task, or keep someone who looks great on paper but doesn't fit your team.

**Wrapper Methods** are like trial periods. You try different team combinations and see which actually performs best on your projects. This gives you the most accurate assessment but requires extensive testing time and the results are specific to your evaluation projects.

**Embedded Methods** are like hiring people with built-in evaluation criteria. You design a work process that naturally reveals who's contributing (like tracking code commits) and automatically adjusts team composition based on ongoing performance.

## Formal Definition

### Filter Methods

Evaluate features using statistical measures independent of the predictive model:

$$S_{\text{filter}} = \{f_i : \text{score}(f_i, y) > \theta\}$$

Where score functions include:
- **Correlation**: $|\text{corr}(f_i, y)|$
- **Mutual Information**: $I(f_i; y) = \sum \sum p(f_i, y) \log \frac{p(f_i, y)}{p(f_i)p(y)}$
- **Chi-squared**: $\chi^2 = \sum \frac{(O - E)^2}{E}$
- **ANOVA F-statistic**: $F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$

### Wrapper Methods

Use a predictive model to evaluate feature subsets:

$$S_{\text{wrapper}} = \argmin_{S \subseteq F} \text{CV\_Error}(M_S, S)$$

Search strategies:
- **Forward selection**: Start empty, add best feature iteratively
- **Backward elimination**: Start full, remove worst feature iteratively
- **Recursive Feature Elimination (RFE)**: Backward elimination with feature ranking
- **Genetic algorithms**: Population-based stochastic search

### Embedded Methods

Perform feature selection during model training:

$$\min_{\beta} \text{Loss}(y, X\beta) + \lambda \cdot \text{Penalty}(\beta)$$

Examples:
- **L1 regularization (Lasso)**: $\lambda \sum |\beta_i|$ (drives coefficients to zero)
- **Tree-based importance**: Feature usage in decision trees
- **Elastic Net**: $\lambda_1 \sum |\beta_i| + \lambda_2 \sum \beta_i^2$

<div class="compare">
<div class="compare-card">
<div class="header red">Filter Methods</div>
<ul>
<li>Fast: O(n · p) complexity</li>
<li>Model-independent</li>
<li>Miss feature interactions</li>
<li>Best for initial screening</li>
</ul>
</div>
<div class="compare-card">
<div class="header green">Wrapper Methods</div>
<ul>
<li>Slow: O(2^p) worst case</li>
<li>Model-specific evaluation</li>
<li>Capture interactions</li>
<li>Best for final selection</li>
</ul>
</div>
</div>

## Mathematical Formulation

### Filter Methods: Mutual Information

For feature $f_i$ and target $y$:

$$I(f_i; y) = H(y) - H(y | f_i)$$

Where:
- $H(y) = -\sum p(y) \log p(y)$ is the entropy of $y$
- $H(y|f_i) = -\sum \sum p(f_i, y) \log p(y|f_i)$ is conditional entropy

**Properties:**
- $I(f_i; y) \geq 0$ (zero means independent)
- Higher MI indicates stronger dependency
- Captures non-linear relationships (vs. correlation)

**Computational Cost:** $O(n \cdot p)$ for $n$ samples, $p$ features

### Wrapper Methods: Recursive Feature Elimination

```
Algorithm: RFE
Input: Feature set F, model M, target size k
Output: Selected features S

1. Train M on all features in F
2. Rank features by importance from M
3. Remove lowest-ranked feature
4. If |F| > k, goto 1
5. Return remaining features S
```

**Computational Cost:** $O(p \cdot T(M))$ where $T(M)$ is model training time

For cross-validated RFE: $O(p \cdot k \cdot T(M))$ where $k$ is CV folds

### Embedded Methods: Lasso Path

The Lasso optimization problem:

$$\min_{\beta} \frac{1}{2n} ||y - X\beta||_2^2 + \lambda ||\beta||_1$$

As $\lambda$ increases from 0 to $\infty$:
- $\lambda = 0$: All features (OLS solution)
- $\lambda \to \infty$: No features ($\beta = 0$)
- Intermediate $\lambda$: Sparse solutions

The **regularization path** shows which features are eliminated at each $\lambda$ value.

**Computational Cost:** $O(n \cdot p^2)$ via coordinate descent

## Code Implementation

### Filter Method: Mutual Information

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">filter_selection_mi.py</span>
</div>

```python
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectKBest
import matplotlib.pyplot as plt

def filter_selection_mi(X, y, k=10):
    """
    Select top k features using mutual information.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    k : int
        Number of features to select

    Returns:
    --------
    selected_indices : array
        Indices of selected features
    scores : array
        MI scores for all features
    """
    # Calculate MI scores
    mi_scores = mutual_info_regression(X, y, random_state=42)

    # Select top k
    selector = SelectKBest(score_func=mutual_info_regression, k=k)
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)

    return selected_indices, mi_scores


# Example usage
np.random.seed(42)
n, p = 200, 20

# Generate data: y depends on first 5 features non-linearly
X = np.random.randn(n, p)
y = (X[:, 0]**2 + 2*X[:, 1] + np.sin(X[:, 2]) +
     np.exp(X[:, 3]/2) + X[:, 4] + np.random.randn(n)*0.1)

# Filter selection
selected, mi_scores = filter_selection_mi(X, y, k=10)

# Visualize
plt.figure(figsize=(10, 5))
plt.bar(range(p), mi_scores)
plt.axhline(y=sorted(mi_scores)[-10], color='r', linestyle='--',
            label=f'Top 10 threshold')
plt.xlabel('Feature Index')
plt.ylabel('Mutual Information')
plt.title('Filter Method: Mutual Information Scores')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

print(f"Selected features: {selected}")
print(f"True relevant features: [0, 1, 2, 3, 4]")
```

</div>


Now that we have seen how filters evaluate features independently using statistical scores, let's examine the key limitation: filters miss feature interactions entirely. In the XOR problem, both features have zero individual correlation with the target, yet together they are perfectly predictive. Wrapper methods address this by evaluating feature *subsets* with the actual model, capturing interactions that filters cannot see. The cost is computational -- instead of scoring features one at a time, wrappers must train and validate the model for every candidate subset.

### Wrapper Method: Forward Selection

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">forward_selection_wrapper.py</span>
</div>

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

def forward_selection_wrapper(X, y, max_features=10, cv=5):
    """
    Forward selection using cross-validated model performance.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    max_features : int
        Maximum features to select
    cv : int
        Number of cross-validation folds

    Returns:
    --------
    selected : list
        Indices of selected features in order
    scores : list
        CV scores at each step
    """
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))
    scores = []

    for step in range(max_features):
        best_score = -np.inf
        best_feature = None

        # Try adding each remaining feature
        for feature in remaining:
            trial_features = selected + [feature]
            X_trial = X[:, trial_features]

            # Evaluate with cross-validation
            model = LinearRegression()
            cv_scores = cross_val_score(
                model, X_trial, y,
                cv=cv,
                scoring='neg_mean_squared_error'
            )
            score = cv_scores.mean()

            if score > best_score:
                best_score = score
                best_feature = feature

        # Add best feature
        if best_feature is not None:
            selected.append(best_feature)
            remaining.remove(best_feature)
            scores.append(best_score)

            print(f"Step {step+1}: Added feature {best_feature}, "
                  f"CV score: {best_score:.4f}")
        else:
            break

    return selected, scores


# Example usage
selected_wrapper, wrapper_scores = forward_selection_wrapper(X, y, max_features=10)

# Plot performance vs. number of features
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(wrapper_scores)+1), -np.array(wrapper_scores), 'o-')
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation MSE')
plt.title('Wrapper Method: Forward Selection Performance')
plt.grid(alpha=0.3)
plt.tight_layout()
```

</div>

Wrapper methods give us interaction-aware selection, but they require explicitly searching through feature subsets -- and that search can be expensive and incomplete (forward selection is greedy). Embedded methods take a different approach: they build feature selection directly into the model training process. Instead of searching externally, the model itself decides which features matter. The tradeoff: embedded methods are fast and elegant, but the selection is tied to the specific model structure. Features selected by Lasso may not be the best features for a random forest.

### Embedded Method: Lasso

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">lasso_selection.py</span>
</div>

```python
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler

def lasso_selection(X, y, n_alphas=100):
    """
    Feature selection using Lasso with cross-validated alpha.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix
    y : array-like, shape (n_samples,)
        Target variable
    n_alphas : int
        Number of alpha values to try

    Returns:
    --------
    selected : array
        Indices of non-zero features
    coefficients : array
        Lasso coefficients
    best_alpha : float
        Selected regularization parameter
    """
    # Standardize features (important for Lasso)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validated Lasso
    lasso_cv = LassoCV(cv=5, n_alphas=n_alphas, random_state=42)
    lasso_cv.fit(X_scaled, y)

    best_alpha = lasso_cv.alpha_
    coefficients = lasso_cv.coef_

    # Selected features (non-zero coefficients)
    selected = np.where(coefficients != 0)[0]

    return selected, coefficients, best_alpha


# Example usage
selected_lasso, coefs, alpha = lasso_selection(X, y)

# Visualize coefficients
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.bar(range(p), coefs)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.xlabel('Feature Index')
plt.ylabel('Lasso Coefficient')
plt.title(f'Embedded Method: Lasso (α={alpha:.4f})')
plt.grid(alpha=0.3)

# Show regularization path
alphas = np.logspace(-4, 1, 100)
coefs_path = []
for alpha_val in alphas:
    lasso = Lasso(alpha=alpha_val)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso.fit(X_scaled, y)
    coefs_path.append(lasso.coef_)

plt.subplot(1, 2, 2)
coefs_path = np.array(coefs_path).T
for i in range(p):
    plt.plot(alphas, coefs_path[i], alpha=0.7)
plt.xscale('log')
plt.xlabel('Alpha (regularization)')
plt.ylabel('Coefficients')
plt.title('Lasso Regularization Path')
plt.grid(alpha=0.3)
plt.tight_layout()

print(f"Selected features (Lasso): {selected_lasso}")
```

</div>

### Comparing All Three Approaches

```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def compare_selection_methods(X, y, n_features=10):
    """
    Compare filter, wrapper, and embedded selection methods.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    results = {}

    # 1. Filter (Mutual Information)
    selected_filter, _ = filter_selection_mi(X_train, y_train, k=n_features)
    model = LinearRegression()
    model.fit(X_train[:, selected_filter], y_train)
    y_pred = model.predict(X_test[:, selected_filter])
    results['Filter'] = {
        'features': selected_filter,
        'mse': mean_squared_error(y_test, y_pred),
        'n_features': len(selected_filter)
    }

    # 2. Wrapper (Forward Selection)
    selected_wrapper, _ = forward_selection_wrapper(
        X_train, y_train, max_features=n_features, cv=5
    )
    model = LinearRegression()
    model.fit(X_train[:, selected_wrapper], y_train)
    y_pred = model.predict(X_test[:, selected_wrapper])
    results['Wrapper'] = {
        'features': selected_wrapper,
        'mse': mean_squared_error(y_test, y_pred),
        'n_features': len(selected_wrapper)
    }

    # 3. Embedded (Lasso)
    selected_lasso, _, _ = lasso_selection(X_train, y_train)
    if len(selected_lasso) > 0:
        model = LinearRegression()
        model.fit(X_train[:, selected_lasso], y_train)
        y_pred = model.predict(X_test[:, selected_lasso])
        results['Embedded'] = {
            'features': selected_lasso,
            'mse': mean_squared_error(y_test, y_pred),
            'n_features': len(selected_lasso)
        }
    else:
        results['Embedded'] = {
            'features': [],
            'mse': np.inf,
            'n_features': 0
        }

    # 4. Baseline (all features)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results['All Features'] = {
        'features': list(range(X.shape[1])),
        'mse': mean_squared_error(y_test, y_pred),
        'n_features': X.shape[1]
    }

    return results


# Compare methods
results = compare_selection_methods(X, y, n_features=10)

# Display results
print("\nComparison of Feature Selection Methods")
print("=" * 60)
for method, metrics in results.items():
    print(f"\n{method}:")
    print(f"  Features selected: {metrics['n_features']}")
    print(f"  Test MSE: {metrics['mse']:.4f}")
    if method != 'All Features':
        print(f"  Selected indices: {metrics['features'][:10]}...")

# Visualize
methods = list(results.keys())
mses = [results[m]['mse'] for m in methods]
n_feats = [results[m]['n_features'] for m in methods]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.bar(methods, mses)
ax1.set_ylabel('Test MSE')
ax1.set_title('Prediction Error Comparison')
ax1.tick_params(axis='x', rotation=45)
ax1.grid(alpha=0.3)

ax2.bar(methods, n_feats)
ax2.set_ylabel('Number of Features')
ax2.set_title('Feature Count Comparison')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(alpha=0.3)

plt.tight_layout()
```

## Common Pitfalls

### Pitfall 1: Using Filters for Interaction Detection

**Problem:** Filter methods evaluate features independently and miss interactions.

**Example:**
```python
# XOR problem: y = x1 XOR x2
X = np.random.randint(0, 2, size=(1000, 2))
y = (X[:, 0] != X[:, 1]).astype(int)

# Individual correlations are zero
print(f"Corr(x1, y): {np.corrcoef(X[:, 0], y)[0, 1]:.4f}")
print(f"Corr(x2, y): {np.corrcoef(X[:, 1], y)[0, 1]:.4f}")
# Both ~0, but BOTH features are necessary!
```

**How to avoid:** Use wrapper methods for problems with known interactions, or add interaction terms to filter evaluation.

<div class="callout-danger">

<strong>Danger:</strong> Evaluating thousands of feature subsets on the same validation set is a form of data leakage. The "best" subset is partially optimized to the validation set, not the true data distribution. Always use nested cross-validation for wrapper methods.

</div>

### Pitfall 2: Overfitting with Wrapper Methods

**Problem:** Testing many feature subsets on the same validation set leads to overfitting the validation set.

**Why it happens:** Each subset evaluation uses the same folds, so information leaks across iterations.

**How to avoid:**
```python
# WRONG: Single train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
best_score = -inf
for subset in generate_subsets():
    score = evaluate(X_test, y_test, subset)  # Overfit to X_test!
    if score > best_score:
        best_score = score

# RIGHT: Nested cross-validation
from sklearn.model_selection import cross_val_score

def nested_selection(X, y, subset):
    # Inner loop: select features using CV
    inner_score = cross_val_score(model, X[:, subset], y, cv=5)
    return inner_score.mean()

# Outer loop: evaluate selected features
outer_scores = []
for train_idx, test_idx in KFold(5).split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Select features on training data
    best_subset = optimize(X_train, y_train, nested_selection)

    # Evaluate on held-out test data
    score = evaluate(X_test, y_test, best_subset)
    outer_scores.append(score)
```

<div class="callout-warning">

<strong>Warning:</strong> Wrapper methods can require thousands of model evaluations. For expensive models (deep learning, large ensembles), use filters for initial screening, then wrappers for final refinement.

</div>

### Pitfall 3: Embedded Methods with Non-Linear Models

**Problem:** Lasso and tree-based importance assume specific model structures and may miss features important for other models.

**Why it happens:** Embedded methods are model-specific by design.

**How to avoid:** Use wrapper methods or multiple embedded methods:
```python
# Combine multiple embedded methods
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV

# Lasso selection
lasso = LassoCV(cv=5)
lasso.fit(X, y)
lasso_features = np.where(lasso.coef_ != 0)[0]

# Random Forest importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)
rf_features = np.argsort(rf.feature_importances_)[-10:]

# Union of both methods
combined = np.union1d(lasso_features, rf_features)
print(f"Lasso selected: {len(lasso_features)}")
print(f"RF selected: {len(rf_features)}")
print(f"Combined: {len(combined)}")
```

## Connections

<div class="callout-info">

ℹ️ **How this connects to the rest of the course:**

</div>

### Builds On
- **01_feature_selection_problem.md**: Understanding why selection is necessary
- **Statistical testing**: Hypothesis tests for filter methods
- **Cross-validation**: Proper evaluation of wrapper methods
- **Regularization theory**: L1/L2 penalties in embedded methods

### Leads To
- **Module 1: GA Fundamentals**: GAs as advanced wrapper method
- **Module 2: Fitness Design**: Designing evaluation functions for wrappers
- **Module 4: DEAP Implementation**: Implementing GA-based selection

### Related To
- **Hyperparameter tuning**: Similar search problems
- **Model selection**: Choosing between model types
- **Ensemble methods**: Feature importance from multiple models
- **Dimensionality reduction**: PCA, autoencoders as alternatives

## Practice Problems

### Problem 1: Method Selection

**Question:** For each scenario, recommend filter, wrapper, or embedded:

a) 10,000 features, 500 samples, need fast initial screening
b) 20 features, 10,000 samples, want best predictive accuracy
c) 100 features, 1,000 samples, using linear model, prefer interpretability

**Solutions:**
- a) **Filter**: Too many features for wrapper, filter provides fast screening
- b) **Wrapper**: Enough data, few features, can afford exhaustive search for accuracy
- c) **Embedded (Lasso)**: Linear model fits embedded approach, interpretability from sparse coefficients

### Problem 2: Implement Backward Elimination

**Task:** Implement backward elimination (opposite of forward selection):

```python
def backward_elimination(X, y, min_features=5, cv=5):
    """
    Backward elimination: start with all features, remove worst.
    """
    n_features = X.shape[1]
    selected = list(range(n_features))

    while len(selected) > min_features:
        worst_score = np.inf
        worst_feature = None

        # Try removing each feature
        for feature in selected:
            trial_features = [f for f in selected if f != feature]
            X_trial = X[:, trial_features]

            # Evaluate
            model = LinearRegression()
            cv_scores = cross_val_score(
                model, X_trial, y, cv=cv,
                scoring='neg_mean_squared_error'
            )
            score = cv_scores.mean()

            # Lower score = better (neg MSE)
            # Remove feature with least impact (highest score when removed)
            if score > worst_score:
                worst_score = score
                worst_feature = feature

        if worst_feature is not None:
            selected.remove(worst_feature)
            print(f"Removed feature {worst_feature}, "
                  f"remaining: {len(selected)}, score: {worst_score:.4f}")

    return selected

# Test
selected_backward = backward_elimination(X, y, min_features=5)
```

### Problem 3: Stability Analysis

**Task:** Evaluate how stable each method is across different data samples:

```python
from sklearn.utils import resample

def evaluate_stability(X, y, method, n_iterations=20):
    """
    Measure selection stability using bootstrap.
    """
    selected_sets = []

    for i in range(n_iterations):
        # Bootstrap sample
        X_boot, y_boot = resample(X, y, random_state=i)

        # Apply method
        if method == 'filter':
            selected, _ = filter_selection_mi(X_boot, y_boot, k=10)
        elif method == 'wrapper':
            selected, _ = forward_selection_wrapper(X_boot, y_boot, max_features=10)
        elif method == 'embedded':
            selected, _, _ = lasso_selection(X_boot, y_boot)

        selected_sets.append(set(selected))

    # Calculate pairwise Jaccard similarity
    similarities = []
    for i in range(len(selected_sets)):
        for j in range(i+1, len(selected_sets)):
            intersection = len(selected_sets[i] & selected_sets[j])
            union = len(selected_sets[i] | selected_sets[j])
            if union > 0:
                similarities.append(intersection / union)

    return np.mean(similarities)

# Compare stability
for method in ['filter', 'wrapper', 'embedded']:
    stability = evaluate_stability(X, y, method, n_iterations=20)
    print(f"{method.capitalize()} stability: {stability:.3f}")
```

### Problem 4: Conceptual — Filter Limitations

**Question:** Explain why mutual information (a filter method) can detect non-linear relationships between a single feature and the target, but still cannot detect the XOR interaction between two features. What fundamental limitation of filter methods does this illustrate?

### Problem 5: Conceptual — Method Comparison

**Question:** A colleague suggests using Lasso for feature selection because it is fast and handles high-dimensional data well. You know the downstream model will be a gradient boosting machine (GBM). Give two reasons why Lasso's selected features might not be optimal for the GBM, and suggest an alternative approach.

## Further Reading

### Foundational Papers
- **John, Kohavi & Pfleger (1994)**: "Irrelevant Features and the Subset Selection Problem" - Theoretical analysis of why feature selection is hard and when it helps.

- **Guyon & Elisseeff (2003)**: "An Introduction to Variable and Feature Selection" - Comprehensive survey of all three approaches with practical guidance.

### Filter Methods
- **Peng, Long & Ding (2005)**: "Feature Selection Based on Mutual Information" - mRMR (minimum Redundancy Maximum Relevance) algorithm.

- **Yu & Liu (2003)**: "Feature Selection for High-Dimensional Data" - Fast Correlation-Based Filter (FCBF) method.

### Wrapper Methods
- **Kohavi & John (1997)**: "Wrappers for Feature Subset Selection" - Definitive paper on wrapper methods with extensive experiments.

- **Saeys, Inza & Larranaga (2007)**: "A Review of Feature Selection Techniques in Bioinformatics" - Wrapper methods for high-dimensional biological data.

### Embedded Methods
- **Tibshirani (1996)**: "Regression Shrinkage and Selection via the Lasso" - Original Lasso paper introducing L1 regularization.

- **Zou & Hastie (2005)**: "Regularization and Variable Selection via the Elastic Net" - Combines L1 and L2 for better feature selection.

### Comparative Studies
- **Bolón-Canedo, Sánchez-Maroño & Alonso-Betanzos (2013)**: "A Review of Feature Selection Methods on Synthetic Data" - Systematic comparison across different data characteristics.

- **Li et al. (2017)**: "Feature Selection: A Data Perspective" - Modern comprehensive review including recent developments.
---

**Next:** [Companion Slides](./04_selection_approaches_slides.md) | [Notebook](../notebooks/01_selection_comparison.ipynb)

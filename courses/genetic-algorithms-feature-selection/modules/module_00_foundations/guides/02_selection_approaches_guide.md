# Feature Selection Approaches: Filter, Wrapper, and Embedded Methods

## In Brief

Three paradigms exist for feature selection, and they differ fundamentally in how they evaluate candidate features:

- **Filter methods** score features using statistical measures computed independently of any predictive model.
- **Wrapper methods** evaluate subsets by training and cross-validating a model on each candidate subset.
- **Embedded methods** perform feature selection as a byproduct of model training, typically through regularization.

Understanding when to apply each paradigm is essential before reaching for a genetic algorithm, which is an advanced wrapper method.

## Key Insight

The central tradeoff is computational cost versus optimization quality. Filters are fast but evaluate features in isolation — they miss synergies and interactions. Wrappers find better subsets because they evaluate the actual predictive benefit of combinations, but the cost is proportional to the number of subsets evaluated times the model training time. GAs solve the wrapper's combinatorial search problem efficiently by exploring many subsets in parallel.

## Formal Definition

Let $F = \{f_1, f_2, \ldots, f_p\}$ be the full feature set, $\mathcal{M}$ a model class, and $L$ a loss function.

**Filter:**
$$S_{\text{filter}} = \{f_i : \text{score}(f_i, y) > \theta\}$$

Score functions include:
- Pearson correlation: $|\text{corr}(f_i, y)|$
- Mutual information: $I(f_i; y) = \sum \sum p(f_i, y) \log \frac{p(f_i, y)}{p(f_i) p(y)}$
- ANOVA F-statistic, chi-squared

Cost: $O(n \cdot p)$

**Wrapper:**
$$S_{\text{wrapper}} = \argmin_{S \subseteq F} \; \text{CV\_Error}(\mathcal{M}_S, S)$$

Search strategies: forward selection, backward elimination, RFE, genetic algorithm.
Cost: $O(p \cdot k \cdot T(\mathcal{M}))$ for RFE; GA typically evaluates $G \times P$ subsets.

**Embedded:**
$$\min_{\beta} \frac{1}{2n} \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \quad \text{(Lasso)}$$

As $\lambda$ increases, coefficients shrink toward zero. At intermediate $\lambda$, only important features have non-zero coefficients.

## Intuitive Explanation

Imagine you are hiring for a project team and evaluating job applicants.

**Filter = Résumé screening.** You score each applicant on GPA, years of experience, and relevant certifications — quickly, independently, without putting them in a room together. Some good team players get filtered out because their individual credentials look weak; some poor team players pass because credentials look strong on paper.

**Wrapper = Trial project.** You assemble candidate team compositions, give each team the actual project, and measure output quality. Expensive and time-consuming, but you are evaluating what actually matters — team performance on the real task.

**Embedded = Built-in performance tracking.** You hire everyone, run the project with regularization rules that automatically reduce the contribution of underperforming team members until their contribution becomes zero. The selection emerges from the work process itself.

## Code Implementation

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.feature_selection import (
    mutual_info_regression,
    SelectKBest,
    RFE,
)
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


# ─── FILTER METHOD ────────────────────────────────────────────────────────────

def filter_selection_mi(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10
):
    """
    Select top k features using mutual information.

    Mutual information captures non-linear relationships and is
    more powerful than Pearson correlation for many problems.
    """
    selector = SelectKBest(
        score_func=mutual_info_regression, k=k
    )
    selector.fit(X, y)
    selected_indices = selector.get_support(indices=True)
    mi_scores = selector.scores_
    return selected_indices, mi_scores


def filter_selection_correlation(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10
):
    """Select top k features by absolute Pearson correlation with target."""
    correlations = np.array([
        abs(np.corrcoef(X[:, i], y)[0, 1]) for i in range(X.shape[1])
    ])
    top_k = np.argsort(correlations)[::-1][:k]
    return top_k, correlations


# ─── WRAPPER METHOD ───────────────────────────────────────────────────────────

def forward_selection_wrapper(
    X: np.ndarray,
    y: np.ndarray,
    max_features: int = 10,
    cv: int = 5
):
    """
    Greedy forward selection: start empty, add best feature at each step.

    Complexity: O(p * max_features * T(model))
    """
    n_features = X.shape[1]
    selected = []
    remaining = list(range(n_features))
    scores = []

    for _ in range(min(max_features, n_features)):
        best_score, best_feat = -np.inf, None

        for feat in remaining:
            trial = selected + [feat]
            model = LinearRegression()
            cv_scores = cross_val_score(
                model, X[:, trial], y, cv=cv,
                scoring="neg_mean_squared_error"
            )
            score = cv_scores.mean()
            if score > best_score:
                best_score = score
                best_feat = feat

        if best_feat is not None:
            selected.append(best_feat)
            remaining.remove(best_feat)
            scores.append(best_score)

    return selected, scores


def rfe_wrapper(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 10
):
    """
    Recursive Feature Elimination with a linear model.

    Trains the model on all features, removes the least important,
    and repeats until the target count is reached.
    """
    model = LinearRegression()
    selector = RFE(
        estimator=model,
        n_features_to_select=n_features_to_select
    )
    selector.fit(X, y)
    return selector.get_support(indices=True), selector.ranking_


# ─── EMBEDDED METHOD ──────────────────────────────────────────────────────────

def lasso_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_alphas: int = 100
):
    """
    L1-regularized regression (Lasso) for embedded feature selection.

    LassoCV automatically tunes the regularization strength via CV.
    Features with non-zero coefficients are selected.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso_cv = LassoCV(cv=5, n_alphas=n_alphas, random_state=42)
    lasso_cv.fit(X_scaled, y)

    selected = np.where(lasso_cv.coef_ != 0)[0]
    return selected, lasso_cv.coef_, lasso_cv.alpha_


def random_forest_importance(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10
):
    """
    Embedded selection using tree-based feature importances.

    RandomForest measures contribution to split quality (Gini) or
    permutation importance for each feature.
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    top_k = np.argsort(rf.feature_importances_)[::-1][:k]
    return top_k, rf.feature_importances_


# ─── METHOD COMPARISON ────────────────────────────────────────────────────────

def compare_selection_methods(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int = 10
):
    """
    Evaluate filter, wrapper, and embedded methods on held-out test data.

    Uses a fixed train/test split so methods are directly comparable.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    results = {}

    # Filter
    selected_filter, _ = filter_selection_mi(
        X_train, y_train, k=n_features_to_select
    )
    model = LinearRegression()
    model.fit(X_train[:, selected_filter], y_train)
    y_pred = model.predict(X_test[:, selected_filter])
    results["filter_mi"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "n_features": len(selected_filter),
    }

    # Wrapper (forward selection)
    selected_wrapper, _ = forward_selection_wrapper(
        X_train, y_train, max_features=n_features_to_select
    )
    model = LinearRegression()
    model.fit(X_train[:, selected_wrapper], y_train)
    y_pred = model.predict(X_test[:, selected_wrapper])
    results["wrapper_forward"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "n_features": len(selected_wrapper),
    }

    # Embedded (Lasso)
    selected_lasso, _, alpha = lasso_selection(X_train, y_train)
    if len(selected_lasso) == 0:
        selected_lasso = np.arange(min(n_features_to_select, X.shape[1]))
    model = LinearRegression()
    model.fit(X_train[:, selected_lasso], y_train)
    y_pred = model.predict(X_test[:, selected_lasso])
    results["embedded_lasso"] = {
        "mse": mean_squared_error(y_test, y_pred),
        "n_features": len(selected_lasso),
        "alpha": alpha,
    }

    return results


# ─── STABILITY ANALYSIS ───────────────────────────────────────────────────────

def evaluate_stability(
    X: np.ndarray,
    y: np.ndarray,
    method: str = "filter",
    n_bootstrap: int = 20,
    k: int = 10
):
    """
    Measure selection stability across bootstrap resamples.

    Uses pairwise Jaccard similarity between selected feature sets.
    High similarity (>0.7) indicates stable, trustworthy selection.
    """
    from sklearn.utils import resample

    selected_sets = []
    for i in range(n_bootstrap):
        X_boot, y_boot = resample(X, y, random_state=i)

        if method == "filter":
            sel, _ = filter_selection_mi(X_boot, y_boot, k=k)
        elif method == "wrapper":
            sel, _ = forward_selection_wrapper(X_boot, y_boot, max_features=k)
        elif method == "embedded":
            sel, _, _ = lasso_selection(X_boot, y_boot)
            sel = sel[:k] if len(sel) >= k else sel
        else:
            raise ValueError(f"Unknown method: {method}")

        selected_sets.append(set(sel.tolist()))

    # Pairwise Jaccard similarities
    similarities = []
    n = len(selected_sets)
    for i in range(n):
        for j in range(i + 1, n):
            inter = len(selected_sets[i] & selected_sets[j])
            union = len(selected_sets[i] | selected_sets[j])
            if union > 0:
                similarities.append(inter / union)

    return float(np.mean(similarities))


if __name__ == "__main__":
    # Generate synthetic dataset: 20 features, only 5 are informative
    X, y = make_regression(
        n_samples=300, n_features=20,
        n_informative=5, noise=15, random_state=42
    )

    results = compare_selection_methods(X, y, n_features_to_select=5)
    print("\nMethod Comparison (test MSE, lower is better):")
    for method, info in results.items():
        print(f"  {method:20s}: MSE={info['mse']:.1f}, "
              f"n_features={info['n_features']}")

    print("\nStability Analysis (Jaccard similarity, higher is more stable):")
    for method in ["filter", "wrapper", "embedded"]:
        stab = evaluate_stability(X, y, method=method, n_bootstrap=15, k=5)
        print(f"  {method:10s}: {stab:.3f}")
```

## Common Pitfalls

**Pitfall 1: Filters miss feature interactions.** Mutual information and correlation measure the marginal relationship between each feature and the target, ignoring joint relationships. A feature that is individually uncorrelated with the target may be critically important when combined with another feature (the XOR problem from guide 01). If you suspect interaction effects, you need a wrapper or GA.

**Pitfall 2: Wrapper overfitting.** Evaluating candidate subsets on the same test data repeatedly leaks information. The subset that looks best on the test set may have overfitted to noise in that specific split. Use nested cross-validation: the outer loop provides an unbiased estimate of generalization performance, the inner loop performs the feature search.

**Pitfall 3: Embedded methods are model-specific.** Lasso selects features based on linear relationship with the target. Random forest selects features that provide the best binary splits. These two methods will often disagree. A feature that is truly important but has a non-linear relationship with the target may be overlooked by Lasso. Combining multiple embedded methods and taking the union provides a more model-agnostic set.

**Pitfall 4: Comparing methods with different feature counts.** A wrapper that selects 15 features should not be compared directly against a filter that selects 10. Fix the target feature count or normalize by computing the MSE per feature selected.

**Pitfall 5: Assuming the fastest method is good enough.** Filters are fast, but on interaction-heavy problems they select the wrong features. The downstream cost of a wrong feature selection (failed model, wasted deployment effort) usually far exceeds the extra computation cost of a wrapper.

## Connections

**Builds on:**
- Feature selection challenge (guide 01) — why we need principled selection methods
- Statistical testing fundamentals — underpins mutual information and F-statistics
- Cross-validation theory — how wrapper methods estimate out-of-sample performance

**Leads to:**
- Module 01: GAs as an advanced wrapper method
- Module 02: Designing fitness functions for wrapper evaluation
- Module 04: Implementing GA-based wrappers with DEAP

**Related:**
- Dimensionality reduction (PCA, autoencoders): transforms rather than selects features
- Hyperparameter tuning (Bayesian optimization): analogous search problem
- Model selection: choosing among model families rather than feature subsets

## Practice Problems

1. **XOR failure mode:** Construct a dataset with 12 features where the true relationship is $y = x_1 \oplus x_2$ and the remaining 10 features are independent noise. Run mutual information filter, forward selection wrapper, and Lasso. Which methods identify $x_1$ and $x_2$ as the selected features?

2. **Stability comparison:** On the `make_regression` dataset with 30 features, compare the Jaccard stability of mutual information filter versus forward selection wrapper over 20 bootstrap resamples. Which is more stable, and why?

3. **Decision flow application:** For each scenario, state which method you would choose and why:
   - $p = 500$ features, $n = 200$ samples, 5-minute compute budget, no interaction effects expected
   - $p = 20$ features, $n = 10{,}000$ samples, interactions likely, model is a gradient boosted tree
   - $p = 50$ features, $n = 300$ samples, regulatory requirement to use a linear model

4. **Regularization path:** Apply `LassoCV` to a dataset with 20 features (5 informative) and plot the coefficient paths as $\lambda$ increases from near-zero to large. At what $\lambda$ value do the noise feature coefficients first reach zero? Do the informative features survive longer?

5. **Combined approach:** Implement a two-stage pipeline that (1) uses mutual information to reduce from 100 features to 20, then (2) applies forward selection on the 20-feature subset. Compare this to applying forward selection directly on all 100 features in terms of final MSE and runtime.

## Further Reading

- Guyon, I. & Elisseeff, A. (2003). An introduction to variable and feature selection. *JMLR*, 3, 1157–1182.
- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *Journal of the Royal Statistical Society B*, 58(1), 267–288.
- Kohavi, R. & John, G. H. (1997). Wrappers for feature subset selection. *Artificial Intelligence*, 97(1–2), 273–324.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

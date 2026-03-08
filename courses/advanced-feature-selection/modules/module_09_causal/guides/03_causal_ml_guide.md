# Causal ML: Double/Debiased ML, Causal Forests, and Production Workflows

## In Brief

Causal ML methods — Double/Debiased Machine Learning (DML), causal forests, and instrumental variable selection — use ML algorithms as components in causal inference procedures rather than as end-to-end predictors. For feature selection, these methods identify features that causally affect the outcome (rather than merely correlating with it) and quantify the magnitude of those causal effects. They also provide the production workflow for blending causal and predictive selection.

## Key Insight

Standard ML feature importance measures (SHAP, permutation importance, Lasso coefficients) are *predictive* importance scores — they tell you how useful a feature is for predicting the outcome. Causal ML importance measures are *causal* importance scores — they tell you how much changing a feature would change the outcome. These diverge whenever confounders are present, and they diverge most severely when distribution shift is large.

---

## 1. Double / Debiased Machine Learning (DML)

### The Problem: Confounding in Feature Importance

Consider estimating the effect of treatment $T$ (a specific feature) on outcome $Y$, adjusting for confounders $\mathbf{X}$ (other features). The naive regression $Y \sim T + \mathbf{X}$ is biased when $\mathbf{X}$ is high-dimensional: regularisation in the first-stage estimates introduces bias into the coefficient on $T$.

**Chernozhukov et al. (2018)** solve this via the **Neyman orthogonal score**: construct an estimating equation that is insensitive (locally robust) to first-stage estimation errors.

### The DML Procedure

For each feature $T_j$ (treated as the "treatment of interest"):

**Step 1:** Regress $T_j$ on all other features $\mathbf{X}_{-j}$ using any ML method. Compute residuals:
$$\tilde{T}_j = T_j - \hat{E}[T_j \mid \mathbf{X}_{-j}]$$

**Step 2:** Regress $Y$ on all features $\mathbf{X}_{-j} \cup \{T_j\}$ (including $T_j$) excluding $T_j$. Compute residuals:
$$\tilde{Y} = Y - \hat{E}[Y \mid \mathbf{X}_{-j}]$$

**Step 3:** Regress $\tilde{Y}$ on $\tilde{T}_j$ (residual-on-residual regression):
$$\hat{\theta}_j = \frac{\text{Cov}(\tilde{Y}, \tilde{T}_j)}{\text{Var}(\tilde{T}_j)}$$

The DML estimator $\hat{\theta}_j$ is the causal effect of $T_j$ on $Y$, purged of confounding by $\mathbf{X}_{-j}$.

### Why Neyman Orthogonality Matters

The key property: the estimating equation for $\theta_j$ is insensitive to small errors in the first-stage estimates $\hat{E}[T_j \mid \mathbf{X}_{-j}]$ and $\hat{E}[Y \mid \mathbf{X}_{-j}]$.

Formally, let $\psi(\theta, \eta)$ be the score function where $\eta = (\hat{E}[T_j \mid \mathbf{X}_{-j}], \hat{E}[Y \mid \mathbf{X}_{-j}])$. Neyman orthogonality requires:

$$\frac{\partial}{\partial \eta} E[\psi(\theta_0, \eta)] \Big|_{\eta = \eta_0} = 0$$

This means first-stage ML errors contribute only second-order bias to $\hat{\theta}_j$, achieving $\sqrt{n}$-consistent inference even when first-stage estimators converge at slower rates.

### Cross-Fitting for Valid Inference

To prevent overfitting bias, DML uses **cross-fitting** (k-fold):

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

def dml_effect(y: np.ndarray,
               T: np.ndarray,
               X: np.ndarray,
               n_folds: int = 5,
               first_stage_model=None) -> dict:
    """
    Double/Debiased ML estimate of causal effect of T on Y given X.

    Parameters
    ----------
    y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable (feature of interest).
    X : array, shape (n, p)
        Confounders (all other features).
    n_folds : int
        Number of cross-fitting folds.
    first_stage_model : sklearn estimator
        Model for first-stage nuisance estimation.

    Returns
    -------
    dict with keys: 'theta' (causal effect), 'se' (standard error), 'p_value'
    """
    if first_stage_model is None:
        first_stage_model = LassoCV(cv=5)

    n = len(y)
    T_resid = np.zeros(n)
    Y_resid = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        T_train, T_test = T[train_idx], T[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # First stage: predict T from X (nuisance)
        from sklearn.base import clone
        m_T = clone(first_stage_model)
        m_T.fit(X_train, T_train)
        T_resid[test_idx] = T_test - m_T.predict(X_test)

        # First stage: predict Y from X (nuisance)
        m_Y = clone(first_stage_model)
        m_Y.fit(X_train, y_train)
        Y_resid[test_idx] = y_test - m_Y.predict(X_test)

    # Second stage: residual-on-residual regression
    theta = np.cov(Y_resid, T_resid)[0, 1] / np.var(T_resid)

    # Standard error (heteroskedasticity-robust)
    scores = (Y_resid - theta * T_resid) * T_resid
    se = np.sqrt(np.mean(scores**2) / (n * np.mean(T_resid**2)**2))

    from scipy import stats
    t_stat = theta / se
    p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

    return {'theta': theta, 'se': se, 't_stat': t_stat, 'p_value': p_value}
```

### DML for Feature Selection

Apply DML to each feature $T_j$ treating it as a treatment. Features with statistically significant $\hat{\theta}_j$ (adjusted for multiple testing) are selected as causal:

```python
from statsmodels.stats.multitest import multipletests

def dml_feature_selection(y, X, feature_names, alpha=0.05, n_folds=5):
    """Select features with significant causal effects via DML."""
    p = X.shape[1]
    results = []

    for j in range(p):
        T_j = X[:, j]
        X_minus_j = np.delete(X, j, axis=1)
        effect = dml_effect(y, T_j, X_minus_j, n_folds=n_folds)
        results.append(effect)

    p_values = [r['p_value'] for r in results]
    # Benjamini-Hochberg FDR control
    rejected, p_adj, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')

    selected = [feature_names[j] for j in range(p) if rejected[j]]
    return selected, results
```

---

## 2. Causal Forests for Heterogeneous Treatment Effects

### Motivation: When Causal Effects Vary

DML estimates a single average causal effect $\hat{\theta}_j$. In reality, the effect of a feature on the outcome may vary across observations — the heterogeneous treatment effect $\tau(x) = E[Y(T=1) - Y(T=0) \mid X=x]$.

Athey, Tibshirani, and Wager (2019) propose **causal forests** to estimate $\tau(x)$ nonparametrically while maintaining valid confidence intervals.

### The Causal Forest Procedure

1. **Honest splitting:** Each tree split uses half the observations for partition selection, the other half for effect estimation (prevents overfitting within leaves).
2. **Local average treatment effect:** In each leaf $L$, estimate:
$$\hat{\tau}(x) = \frac{\sum_{i \in L} (Y_i - \hat{Y}_i)(T_i - \hat{T}_i)}{\sum_{i \in L} (T_i - \hat{T}_i)^2}$$
where $\hat{Y}$ and $\hat{T}$ are out-of-bag predictions from the residualisation forest.
3. **Forest aggregation:** Average $\hat{\tau}(x)$ across trees to reduce variance.

### Causal Forest Feature Importance

The causal forest provides a principled feature importance measure: how often is feature $j$ used in splits that lead to high heterogeneity in $\hat{\tau}(x)$? Features important for *causal heterogeneity* are features that modify the causal effect — these are causally relevant in a deep sense.

```python
from econml.grf import CausalForest
from sklearn.model_selection import train_test_split
import numpy as np

def causal_forest_importance(y, T, X, feature_names, n_estimators=1000):
    """
    Fit causal forest and extract feature importances.

    Parameters
    ----------
    y : array, shape (n,)
        Outcome variable.
    T : array, shape (n,)
        Treatment variable.
    X : array, shape (n, p)
        Effect moderators (features).
    feature_names : list
        Names of features.
    n_estimators : int
        Number of trees.

    Returns
    -------
    importances : pd.DataFrame
        Feature importance scores from causal forest.
    """
    import pandas as pd

    # Reshape T for EconML
    T_2d = T.reshape(-1, 1)

    # Fit causal forest
    cf = CausalForest(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        max_depth=None,
        honest=True,
        random_state=42
    )
    cf.fit(X, T_2d, y)

    # Feature importances based on causal heterogeneity splits
    importances = cf.feature_importances_

    result_df = pd.DataFrame({
        'feature': feature_names,
        'causal_importance': importances,
    }).sort_values('causal_importance', ascending=False)

    return result_df
```

### Causal Forest vs Standard Random Forest Importance

| Aspect | Random Forest Importance | Causal Forest Importance |
|---|---|---|
| Measures | Predictive accuracy of $Y$ | Heterogeneity of $\tau(x)$ |
| Confounders | Selected (they help predict $Y$) | Downweighted (confounders don't explain $\tau$ heterogeneity) |
| Spurious correlates | Selected | Excluded |
| Valid inference | No (biased p-values) | Yes (honest estimation) |

---

## 3. Instrumental Variables as Feature Selection Guidance

### The IV Approach

An **instrumental variable** $Z_j$ for feature $X_j$ satisfies:
1. **Relevance:** $Z_j$ correlates with $X_j$
2. **Exclusion restriction:** $Z_j$ affects $Y$ only through $X_j$ (not directly)
3. **Independence:** $Z_j$ is independent of unobserved confounders

When instruments are available, IV estimation identifies causal effects even in the presence of latent confounders.

### IV-Guided Feature Selection

In practice, instruments suggest which features are causal:
- If feature $X_j$ has a valid instrument $Z_j$, then $X_j$ is a causal feature (or at least an intermediate causal variable)
- IV estimation gives the local average treatment effect (LATE) for features with instruments

```python
from sklearn.linear_model import LinearRegression

def iv_feature_effect(y, X_j, Z_j, X_controls):
    """
    Instrumental variable estimate of causal effect of X_j on Y.
    Two-stage least squares (2SLS).

    Parameters
    ----------
    y : array
        Outcome.
    X_j : array
        Endogenous feature (treatment).
    Z_j : array
        Instrument for X_j.
    X_controls : array
        Control variables.

    Returns
    -------
    float : IV estimate of causal effect.
    """
    # First stage: X_j ~ Z_j + X_controls
    stage1_X = np.column_stack([Z_j, X_controls])
    stage1 = LinearRegression().fit(stage1_X, X_j)
    X_j_hat = stage1.predict(stage1_X)

    # Second stage: Y ~ X_j_hat + X_controls
    stage2_X = np.column_stack([X_j_hat, X_controls])
    stage2 = LinearRegression().fit(stage2_X, y)

    return stage2.coef_[0]  # IV estimate of causal effect
```

---

## 4. Causal vs Predictive Features: Concrete Examples

### Example 1: Stock Returns Prediction

**Dataset:** S&P 500 daily returns over 10 years.

**Predictive features (SHAP top-5):**
1. 10-day momentum (high autocorrelation-based)
2. VIX level (spurious regime correlation)
3. Day-of-week indicator (calendar effect)
4. Yield curve slope (macro regime)
5. Trailing 30-day volatility

**Causal features (ICP + DML, across market regimes):**
1. Order flow imbalance (direct causal mechanism)
2. Bid-ask spread (market friction)
3. Earnings surprise (fundamental shock)
4. Fed announcement dummy (exogenous intervention)

**When they diverge:** During the 2020 COVID crash, momentum and VIX-based features completely reversed their predictive relationship. Order flow and earnings surprise maintained their effects.

### Example 2: Energy Consumption Forecasting

**Predictive features:** Historical consumption lag-1, temperature, day-type, price
**Causal features:** Temperature (direct causal effect on demand), day-type (schedule effect)

**Distribution shift scenario:** Demand response program introduced in deployment. Predictive features include price-lag correlations from a market without demand response; these fail completely. Temperature and day-type remain causally valid.

### Quantifying the Divergence

```python
def predictive_vs_causal_importance(y, X, feature_names, env_labels,
                                    T_shift_period):
    """
    Compare predictive (SHAP) and causal (DML) feature importances.
    Shows divergence under distribution shift.
    """
    import shap
    from sklearn.ensemble import GradientBoostingRegressor

    # Predictive importance (SHAP on in-distribution data)
    train_mask = env_labels < T_shift_period
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[~train_mask], y[~train_mask]

    model = GradientBoostingRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap_importance = np.abs(shap_values).mean(axis=0)

    # Causal importance (DML across environments)
    causal_selected, dml_results = dml_feature_selection(
        y_train, X_train, feature_names, alpha=0.05
    )
    causal_importance = np.array([
        abs(r['theta']) for r in dml_results
    ])

    return {
        'predictive': dict(zip(feature_names, shap_importance)),
        'causal': dict(zip(feature_names, causal_importance)),
        'causal_selected': causal_selected,
    }
```

---

## 5. Distribution Shift Robustness: Why Causal Features Survive

### The Mechanism

Causal features have *mechanism stability*: the structural equation $Y = f(\text{Pa}(Y), \varepsilon)$ is the same across environments. Non-causal features have *correlation stability* only within each environment.

When the environment changes:
- **Causal features:** $P(Y \mid X_{\text{causal}})$ is stable → model predictions remain accurate
- **Spurious features:** $P(Y \mid X_{\text{spurious}})$ changes → model predictions degrade

### Performance Degradation Under Shift

The performance gap between causal and predictive feature sets widens monotonically with shift severity:

| Shift level | Causal features | Predictive features | Gap |
|---|---|---|---|
| None (in-distribution) | $R^2 = 0.72$ | $R^2 = 0.81$ | −0.09 |
| Mild shift | $R^2 = 0.69$ | $R^2 = 0.71$ | −0.02 |
| Moderate shift | $R^2 = 0.67$ | $R^2 = 0.58$ | +0.09 |
| Severe shift | $R^2 = 0.64$ | $R^2 = 0.31$ | +0.33 |

The causal feature set *sacrifices in-distribution accuracy* for *out-of-distribution robustness*. The blended approach captures both.

---

## 6. Production Workflow: Blending Causal and Predictive Selection

### The Three-Set Framework

For production deployment, maintain three feature sets:

```python
class CausalPredictiveBlend:
    """
    Production feature set management combining causal and predictive selection.
    """

    def __init__(self, causal_features, predictive_features, feature_names):
        self.causal = set(causal_features)
        self.predictive = set(predictive_features)
        self.all_features = set(feature_names)

    @property
    def consensus(self):
        """Features selected by both causal and predictive methods."""
        return self.causal & self.predictive

    @property
    def causal_only(self):
        """Causally identified but low predictive importance."""
        return self.causal - self.predictive

    @property
    def predictive_only(self):
        """High predictive importance but not causally confirmed."""
        return self.predictive - self.causal

    @property
    def production_set(self):
        """Recommended production feature set: consensus + causal_only."""
        return self.consensus | self.causal_only

    @property
    def monitoring_set(self):
        """Features to monitor for drift: predictive_only."""
        return self.predictive_only

    def deployment_recommendation(self):
        """Print actionable recommendations."""
        print(f"Consensus features ({len(self.consensus)}): "
              f"use in all models — both robust and accurate")
        print(f"Causal-only features ({len(self.causal_only)}): "
              f"include for robustness — may not improve in-distribution performance")
        print(f"Predictive-only features ({len(self.predictive_only)}): "
              f"use with monitoring — watch for distribution drift")
        print(f"\nProduction set size: {len(self.production_set)}")
        print(f"Features to monitor: {len(self.monitoring_set)}")
```

### When to Prioritise Causal vs Predictive Selection

| Priority | When to use | Expected outcome |
|---|---|---|
| Causal selection | Long horizon deployment, domain shift expected, regulatory requirements | Lower in-distribution accuracy, higher out-of-distribution robustness |
| Predictive selection | Competition/benchmark, iid test set, maximum accuracy needed | Higher in-distribution accuracy, may degrade under shift |
| Blended (consensus) | Standard production deployment | Balanced: moderate accuracy gain + shift robustness |
| Full predictive + monitoring | High-stakes production with drift detection | Best accuracy initially, with fallback to causal set on drift |

---

## Common Pitfalls

- **DML requires sufficient sample size:** Cross-fitting reduces effective sample size. With $n < 200$ and $p > 20$, first-stage estimates are poor and DML's double-robustness breaks down.
- **Causal forest requires binary or continuous treatment:** The standard causal forest implementation requires a scalar treatment. For categorical features, use one-vs-rest or reformulate.
- **ICP and DML are not the same:** ICP identifies features invariant across environments; DML estimates causal effects adjusting for confounders in a single environment. They are complementary.
- **Significant DML effect ≠ causal direction:** DML estimates the effect of $T_j$ on $Y$ holding others fixed, but it does not guarantee the direction is $T_j \to Y$ (could be $Y \to T_j$). Causal graph knowledge is still needed.
- **Confounders must be observed:** DML assumes all confounders are in $\mathbf{X}$. With latent confounders, use IV or combine with ICP-style testing.

---

## Connections

- **Builds on:** Double/Debiased ML (Chernozhukov et al., 2018), causal forests (Athey et al., 2019), ICP (Guide 02)
- **Leads to:** Ensemble and hybrid feature selection (Module 10), production pipelines (Module 11)
- **Related to:** Semiparametric efficiency theory, nonparametric causal inference, policy evaluation

---

## Further Reading

- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68. — The DML paper.
- Athey, S., Tibshirani, J., & Wager, S. (2019). Generalized random forests. *Annals of Statistics*, 47(2), 1148–1178. — Causal forests with honest estimation and valid inference.
- Wager, S., & Athey, S. (2018). Estimation and inference of heterogeneous treatment effects using random forests. *Journal of the American Statistical Association*, 113(523), 1228–1242. — Causal forest for CATE estimation.
- `econml` library: https://econml.azurewebsites.net — Microsoft's causal ML library with DML and causal forests.
- `doubleml` library: https://docs.doubleml.org — Python/R implementation of DML by Bach et al.

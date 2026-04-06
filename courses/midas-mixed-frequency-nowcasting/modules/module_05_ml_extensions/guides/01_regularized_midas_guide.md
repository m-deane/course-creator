# Regularized MIDAS: Lasso, Ridge, Elastic Net, and Group Lasso

> **Reading time:** ~20 min | **Module:** 05 — Ml Extensions | **Prerequisites:** Module 4


## Learning Objectives


<div class="callout-key">

**Key Concept Summary:** Standard MIDAS regression with unrestricted Beta or Almon weights handles a single high-frequency predictor elegantly. The weight function imposes a smooth constraint, keeping the effective paramet...

</div>

By the end of this guide you will be able to:

1. Explain why regularization is necessary when MIDAS models have many predictors
2. Formulate Lasso-MIDAS, Ridge-MIDAS, and Elastic Net MIDAS as penalized regression problems
3. Interpret coefficient paths and select tuning parameters via cross-validation
4. Apply group Lasso to enforce structured sparsity across frequency groups
5. Implement regularized MIDAS in Python using `sklearn` and `midasml`-style utilities

---

## 1. Motivation: When Classical MIDAS Breaks Down

Standard MIDAS regression with unrestricted Beta or Almon weights handles a single high-frequency predictor elegantly. The weight function imposes a smooth constraint, keeping the effective parameter count low. Problems emerge when the forecaster has access to:

- Dozens of monthly predictors (employment components, survey indices, financial spreads)
- Hundreds of daily series (market prices, sentiment scores, search trends)
- Lags of each series stacked into the design matrix

<div class="callout-insight">

**Insight:** The mixed-frequency approach preserves within-period dynamics that aggregation destroys. This is especially valuable when the timing of high-frequency movements carries economic information.

</div>


In a MIDAS design matrix $X \in \mathbb{R}^{T \times p}$ where $p = K \times m$ (K series each observed at m high-frequency periods per low-frequency period), the number of regressors can easily exceed the number of observations. Classical OLS becomes ill-posed. The weight function partially solves this, but unrestricted Almon polynomials of high degree also overfit.

**Regularization** penalises coefficient magnitude during estimation, trading a small amount of bias for a large reduction in variance. This is the bias-variance tradeoff applied to the high-dimensional nowcasting setting.

---

## 2. Ridge-MIDAS

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


### 2.1 Formulation

Ridge regression adds an $\ell_2$ penalty on the coefficient vector:

$$\hat{\beta}^{\text{Ridge}} = \arg\min_{\beta} \left\{ \frac{1}{T} \sum_{t=1}^{T} \left( y_t - \alpha - \sum_{k=1}^{K} \sum_{j=0}^{m-1} \beta_{k,j} x_{k,t-j/m} \right)^2 + \lambda \sum_{k,j} \beta_{k,j}^2 \right\}$$

The penalty $\lambda \|\beta\|_2^2$ shrinks all coefficients toward zero but does not set any exactly to zero. Every predictor contributes to the forecast; the question is how much.

### 2.2 Properties

| Property | Ridge |
|----------|-------|
| Coefficient selection | No (never exactly zero) |
| Handles multicollinearity | Yes (strongly) |
| Computational cost | Low (closed form) |
| Bias introduced | Yes, uniformly |

The closed-form solution is $\hat{\beta}^{\text{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y$, which regularises the eigenvalues of $X^\top X$. In high-frequency settings with near-collinear lags, this is particularly valuable.

### 2.3 Effective Degrees of Freedom

Ridge does not perform variable selection but controls complexity through effective degrees of freedom:

$$\text{df}(\lambda) = \text{tr}\left[ X(X^\top X + \lambda I)^{-1} X^\top \right] = \sum_{j=1}^{p} \frac{d_j^2}{d_j^2 + \lambda}$$

where $d_j$ are singular values of $X$. As $\lambda \to 0$, df approaches rank($X$); as $\lambda \to \infty$, df approaches 0.

### 2.4 Python Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler

def build_midas_design_matrix(y_low, x_high, m, lags_low=1, lags_high=12):
    """
    Build MIDAS design matrix.

    Parameters
    ----------
    y_low : array, shape (T,) — low-frequency target
    x_high : array, shape (T*m,) — high-frequency predictor
    m : int — high-freq observations per low-freq period
    lags_low : int — lags of low-freq target to include
    lags_high : int — number of high-freq lags

    Returns
    -------
    X : array, shape (T - lags_low, lags_high + lags_low)
    y : array, shape (T - lags_low,)
    """
    T = len(y_low)
    X_rows = []
    y_rows = []

    for t in range(lags_low, T):
        # High-frequency lags (most recent first)
        hf_idx_end = t * m
        hf_lags = x_high[max(0, hf_idx_end - lags_high):hf_idx_end][::-1]
        if len(hf_lags) < lags_high:
            continue  # Skip incomplete windows

        # Low-frequency lags
        lf_lags = y_low[t - lags_low:t][::-1]

        row = np.concatenate([hf_lags, lf_lags])
        X_rows.append(row)
        y_rows.append(y_low[t])

    return np.array(X_rows), np.array(y_rows)


# Fit Ridge-MIDAS with cross-validation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_scaled, y)

print(f"Optimal lambda: {ridge_cv.alpha_:.4f}")
print(f"Number of nonzero coefficients: {np.sum(ridge_cv.coef_ != 0)}")  # Always all
```

</div>
</div>

---

## 3. Lasso-MIDAS

### 3.1 Formulation

Lasso (Least Absolute Shrinkage and Selection Operator) replaces the $\ell_2$ penalty with $\ell_1$:

$$\hat{\beta}^{\text{Lasso}} = \arg\min_{\beta} \left\{ \frac{1}{T} \sum_{t=1}^{T} \left( y_t - \alpha - \sum_{k,j} \beta_{k,j} x_{k,t-j/m} \right)^2 + \lambda \sum_{k,j} |\beta_{k,j}| \right\}$$

The $\ell_1$ penalty creates a corner solution at zero for many coefficients, performing simultaneous estimation and variable selection.

### 3.2 Geometric Intuition

In two dimensions, the Lasso constraint set $|\beta_1| + |\beta_2| \leq t$ is a diamond (rotated square). The OLS contours (ellipses) typically meet this diamond at a corner, setting one coefficient exactly to zero. Ridge constraint sets are circles, which meet ellipses on the edge rather than a corner.

This geometry explains why Lasso selects variables and Ridge does not.

### 3.3 MIDAS-Specific Considerations

When applying Lasso to MIDAS, each high-frequency lag $j$ of predictor $k$ receives an independent penalty. This means:

- Lasso can select specific lags (e.g., lag 1 but not lags 2-12) — often economically meaningful
- The solution may be "choppy" across lags (non-monotone selected lags), which can be hard to interpret
- Without structure in the penalty, the smooth lag profile of classical MIDAS is abandoned

Group Lasso (Section 5) addresses this by treating all lags of one series as a group.

### 3.4 Coordinate Descent Algorithm

Lasso is solved via coordinate descent. The update for coefficient $\beta_j$ is:

$$\hat{\beta}_j \leftarrow \mathcal{S}_{\lambda / \|x_j\|^2}\left(\frac{x_j^\top r_{-j}}{x_j^\top x_j}\right)$$

where $r_{-j} = y - X_{-j}\hat{\beta}_{-j}$ is the partial residual and $\mathcal{S}_\lambda(z) = \text{sign}(z)(|z| - \lambda)_+$ is the soft-thresholding operator.

### 3.5 Python Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from sklearn.linear_model import Lasso, LassoCV
import matplotlib.pyplot as plt

# LassoCV uses coordinate descent with cross-validation
lasso_cv = LassoCV(
    alphas=np.logspace(-4, 1, 100),
    cv=5,
    max_iter=10000,
    random_state=42
)
lasso_cv.fit(X_scaled, y)

print(f"Optimal lambda: {lasso_cv.alpha_:.6f}")
selected = np.sum(lasso_cv.coef_ != 0)
print(f"Selected predictors: {selected} / {X_scaled.shape[1]}")

# Plot regularization path
from sklearn.linear_model import lasso_path

alphas_path, coefs_path, _ = lasso_path(X_scaled, y, alphas=np.logspace(-4, 1, 100))

fig, ax = plt.subplots(figsize=(10, 5))
for i in range(coefs_path.shape[0]):
    ax.plot(np.log10(alphas_path), coefs_path[i], alpha=0.4, linewidth=0.8)
ax.axvline(np.log10(lasso_cv.alpha_), color='red', linestyle='--', label='CV optimal')
ax.set_xlabel('log10(lambda)')
ax.set_ylabel('Coefficient')
ax.set_title('Lasso-MIDAS Regularization Path')
ax.legend()
plt.tight_layout()
plt.show()
```

</div>
</div>

---

## 4. Elastic Net MIDAS

### 4.1 Formulation

Elastic Net combines Ridge and Lasso penalties, controlled by the mixing parameter $\alpha_\text{mix} \in [0, 1]$:

$$\hat{\beta}^{\text{EN}} = \arg\min_{\beta} \left\{ \frac{1}{T}\|y - X\beta\|_2^2 + \lambda \left[ \alpha_\text{mix} \|\beta\|_1 + \frac{(1-\alpha_\text{mix})}{2} \|\beta\|_2^2 \right] \right\}$$

- $\alpha_\text{mix} = 1$: pure Lasso
- $\alpha_\text{mix} = 0$: pure Ridge
- $\alpha_\text{mix} = 0.5$: balanced Elastic Net

### 4.2 Advantages Over Pure Lasso in MIDAS

1. **Grouped selection**: When predictors are correlated (adjacent lags are highly correlated), Lasso arbitrarily selects one. Elastic Net tends to select or drop correlated variables together — desirable for adjacent lags.
2. **More stable paths**: Coefficient paths are smoother than Lasso, easing interpretation.
3. **Oracle property**: With proper $\lambda$, Elastic Net achieves near-optimal prediction even with $p \gg T$.

### 4.3 Python Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from sklearn.linear_model import ElasticNetCV, ElasticNet

# Grid search over l1_ratio (alpha_mix) and lambda
l1_ratios = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]

en_cv = ElasticNetCV(
    l1_ratio=l1_ratios,
    alphas=np.logspace(-4, 1, 50),
    cv=5,
    max_iter=10000,
    random_state=42
)
en_cv.fit(X_scaled, y)

print(f"Optimal l1_ratio: {en_cv.l1_ratio_:.2f}")
print(f"Optimal lambda: {en_cv.alpha_:.6f}")
print(f"Selected: {np.sum(en_cv.coef_ != 0)} / {X_scaled.shape[1]}")

# Compare models
from sklearn.metrics import mean_squared_error

models = {
    'Ridge': ridge_cv,
    'Lasso': lasso_cv,
    'ElasticNet': en_cv
}

for name, model in models.items():
    y_pred = model.predict(X_scaled)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    nonzero = np.sum(model.coef_ != 0) if hasattr(model, 'coef_') else 'N/A'
    print(f"{name}: RMSE={rmse:.4f}, Selected={nonzero}")
```


---

## 5. Group Lasso for MIDAS

### 5.1 Motivation

In a MIDAS design matrix with $K$ indicators each having $m$ high-frequency lags, the natural group structure is:

$$\mathcal{G} = \{\{1, 2, \ldots, m\}, \{m+1, \ldots, 2m\}, \ldots, \{(K-1)m+1, \ldots, Km\}\}$$

Group Lasso respects this structure: it selects or drops entire groups (indicators) together, rather than individual lags.

### 5.2 Group Lasso Penalty

$$\hat{\beta}^{\text{GL}} = \arg\min_{\beta} \left\{ \frac{1}{T}\|y - X\beta\|_2^2 + \lambda \sum_{g \in \mathcal{G}} \sqrt{|g|} \|\beta_g\|_2 \right\}$$

The $\sqrt{|g|}$ factor adjusts for group size (larger groups are penalised proportionally more). The $\ell_2$ norm within each group means the group is either zeroed entirely or all its components are non-zero.

### 5.3 Comparison with Individual Lasso

| Feature | Lasso | Group Lasso |
|---------|-------|-------------|
| Penalty | $\sum_j |\beta_j|$ | $\sum_g \sqrt{|g|}\|\beta_g\|_2$ |
| Selection unit | Individual coefficients | Groups (indicators) |
| Within-group structure | Arbitrary | Dense (all or nothing) |
| Interpretability | Lag-level | Indicator-level |
| MIDAS natural grouping | No | Yes |

### 5.4 Implementation with group_lasso

```python

# Install: pip install group-lasso
from group_lasso import GroupLasso
import numpy as np

def fit_group_lasso_midas(X, y, K, m, reg=0.05):
    """
    Fit group Lasso MIDAS where each indicator's m lags form one group.

    Parameters
    ----------
    X : array, shape (T, K*m) — MIDAS design matrix
    y : array, shape (T,) — low-frequency target
    K : int — number of high-frequency indicators
    m : int — lags per indicator
    reg : float — regularisation strength

    Returns
    -------
    model : fitted GroupLasso model
    selected_indicators : list of indicator indices with nonzero groups
    """
    # Group labels: indicator k gets label k for all its m lags
    groups = np.repeat(np.arange(K), m)

    model = GroupLasso(
        groups=groups,
        group_reg=reg,
        l1_reg=0,          # pure group lasso (no individual penalty)
        scale_reg='group_size',
        supress_warning=True
    )
    model.fit(X, y.reshape(-1, 1))

    # Identify selected groups
    coef = model.coef_.ravel()
    selected = []
    for k in range(K):
        group_coef = coef[k*m:(k+1)*m]
        if np.any(group_coef != 0):
            selected.append(k)

    return model, selected


# Example: 10 monthly indicators, 12 lags each
K, m = 10, 12

# X_gl has shape (T, 120)
model_gl, selected = fit_group_lasso_midas(X_gl, y, K, m, reg=0.1)
print(f"Selected indicators: {selected}")
print(f"Indicators selected: {len(selected)} / {K}")
```

### 5.5 Sparse Group Lasso

A further extension combines group and individual penalties:

$$\hat{\beta}^{\text{SGL}} = \arg\min_\beta \left\{\frac{1}{T}\|y-X\beta\|_2^2 + \lambda \left[(1-\alpha)\sum_g \sqrt{|g|}\|\beta_g\|_2 + \alpha \|\beta\|_1\right]\right\}$$

This allows within-group sparsity (specific lags) while encouraging group-level selection. The `group_lasso` package supports this via the `l1_reg` parameter.

---

## 6. The midasml Package

The `midasml` Python package (Babii, Ghysels, Striaukas 2021) provides MIDAS-specific regularization tools. Its key contributions:

1. **Legendre polynomials as basis**: Instead of raw lag dummies, project onto orthogonal Legendre polynomials on $[-1, 1]$, then apply Lasso/Group Lasso to polynomial coefficients
2. **Structured group Lasso**: Groups defined by Legendre degree rather than lag index
3. **Semi-parametric MIDAS**: Nonparametric lag weights estimated with penalization

```python

# Conceptual midasml workflow (install: pip install midasml)

# from midasml import MIDASQuantileReg, MIDASReg

# The Legendre projection approach:
def legendre_projection(lags, degree=3):
    """Project lag indices onto Legendre polynomial basis."""
    from numpy.polynomial.legendre import legval
    # Map lag indices to [-1, 1]
    m = len(lags)
    x = np.linspace(-1, 1, m)
    # Evaluate Legendre polynomials
    basis = np.zeros((m, degree + 1))
    for d in range(degree + 1):
        coeffs = np.zeros(degree + 1)
        coeffs[d] = 1.0
        basis[:, d] = legval(x, coeffs)
    return basis

# After projection, apply group Lasso on polynomial coefficients

# This enforces smooth lag profiles while allowing sparsity
```

---

## 7. Tuning Parameter Selection

### 7.1 Time-Series Cross-Validation

Standard $k$-fold cross-validation violates temporal ordering. For time series, use:

**Expanding window CV**: Train on $[1, t_1]$, validate on $[t_1+1, t_1+h]$; expand and repeat.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5, gap=1)  # gap avoids look-ahead

scores = []
for train_idx, val_idx in tscv.split(X_scaled):
    X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    lasso = Lasso(alpha=0.01, max_iter=10000)
    lasso.fit(X_tr, y_tr)
    mse = mean_squared_error(y_val, lasso.predict(X_val))
    scores.append(mse)

print(f"Time-series CV RMSE: {np.sqrt(np.mean(scores)):.4f}")
```

### 7.2 Information Criteria

For large datasets where CV is expensive, use BIC-MIDAS (Babii et al. 2019):

$$\text{BIC}(\lambda) = T \cdot \log\left(\frac{\text{RSS}(\lambda)}{T}\right) + \hat{k}(\lambda) \cdot \log(T)$$

where $\hat{k}(\lambda)$ is the effective degrees of freedom (number of nonzero coefficients for Lasso).

---

## 8. Practical Guidelines

### When to Use Which Estimator

| Situation | Recommended |
|-----------|-------------|
| Few predictors, high collinearity | Ridge-MIDAS |
| Many predictors, want selection | Lasso-MIDAS |
| Correlated predictors, want selection | Elastic Net |
| Multiple indicators, want indicator-level selection | Group Lasso |
| Smooth lag profiles + sparsity | Sparse Group Lasso or midasml |
| $p \ll T$, interpretability important | Classical MIDAS (unrestricted or U-MIDAS) |

### Pre-Processing Checklist

1. Standardise all predictors (mean 0, variance 1) before applying regularization
2. Stationarise series (differencing, log-transformation)
3. Remove seasonal components if present
4. Handle ragged edges before constructing the design matrix

---

## 9. Key References

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.



- Tibshirani, R. (1996). Regression shrinkage and selection via the Lasso. *JRSS-B*, 58(1), 267–288.
- Yuan, M., & Lin, Y. (2006). Model selection and estimation in regression with grouped variables. *JRSS-B*, 68(1), 49–67.
- Babii, A., Ghysels, E., & Striaukas, J. (2021). Machine learning time series regressions with an application to nowcasting. *Journal of Business & Economic Statistics*, 40(3), 1094–1106.
- Zou, H., & Hastie, T. (2005). Regularization and variable selection via the Elastic Net. *JRSS-B*, 67(2), 301–320.

---

## Summary

Regularized MIDAS extends classical MIDAS to the high-dimensional setting by adding penalty terms that control coefficient magnitude and sparsity:

- **Ridge**: Shrinks all coefficients, handles multicollinearity, no selection
- **Lasso**: Shrinks and selects individual lags, may produce choppy profiles
- **Elastic Net**: Balanced shrinkage and selection, handles correlated lags better than Lasso
- **Group Lasso**: Selects entire indicators as groups, respects MIDAS natural structure
- **midasml / Sparse Group Lasso**: Combines group selection with smooth lag profiles

Next: [Machine Learning Nowcasting](02_ml_nowcasting_guide.md) — random forests and gradient boosting with mixed-frequency features.


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary advantage of the approach described in this guide over simpler alternatives?

**Practice Question 2:** What assumptions must hold for this method to produce reliable results?


---

## Cross-References

<a class="link-card" href="./02_ml_nowcasting_guide.md">
  <div class="link-card-title">02 Ml Nowcasting</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_ml_nowcasting_slides.md">
  <div class="link-card-title">02 Ml Nowcasting — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


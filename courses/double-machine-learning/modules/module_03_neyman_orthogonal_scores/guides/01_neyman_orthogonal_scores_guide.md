# Neyman Orthogonal Scores

## In Brief

You will learn why DML uses Neyman orthogonal score functions to achieve robustness against first-stage ML estimation errors. The orthogonal score $\psi(W; \theta_0, \eta_0)$ is insensitive to small perturbations in the nuisance parameter $\eta$, which means imperfect ML models still produce valid treatment effect estimates.

> 💡 **Key Insight:** A naive plug-in estimator has bias proportional to the first-stage ML error. The orthogonal score makes bias proportional to the PRODUCT of two errors — outcome model error times treatment model error — which vanishes much faster. This is the "double" in Double Machine Learning.

## Visual Explanation

```
SENSITIVITY TO FIRST-STAGE ERRORS

Naive Plug-in:                    DML (Orthogonal Score):
Bias ∝ rₙ                        Bias ∝ rₙ × sₙ
(first-order in ML error)         (second-order: product of errors)

If ML error = 0.1:               If ML errors = 0.1 each:
  Bias ≈ 0.1                       Bias ≈ 0.1 × 0.1 = 0.01
  (significant!)                    (negligible!)

This 10× reduction is why DML works with imperfect ML models.
```

## What Is an Influence Function?

For the partially linear model $Y = \theta_0 D + g_0(X) + \epsilon$, the influence function characterises how each observation affects the estimator. The Neyman orthogonal score is:

$$\psi(W; \theta, \eta) = (Y - g(X) - \theta(D - m(X))) \cdot (D - m(X))$$

where $W = (Y, D, X)$ is the data and $\eta = (g, m)$ are the nuisance functions. The estimator satisfies:

$$\frac{1}{n}\sum_{i=1}^n \psi(W_i; \hat{\theta}, \hat{\eta}) = 0$$

The "orthogonality" means:

$$\frac{\partial}{\partial \eta} E[\psi(W; \theta_0, \eta)]\bigg|_{\eta = \eta_0} = 0$$

This derivative being zero is the formal statement that small perturbations of $\eta$ around the truth do not affect the expected score.

## How to See Orthogonality in Action

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

np.random.seed(42)
n = 5000
p = 20
true_theta = 1.0

X = np.random.randn(n, p)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
Y = true_theta * D + X[:, 0] + 0.5 * X[:, 2] + np.random.randn(n)

def estimate_with_degraded_ml(Y, D, X, noise_level=0.0):
    """Run DML with intentionally degraded ML models."""
    n = len(Y)
    resid_Y = np.zeros(n)
    resid_D = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_d = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_y.fit(X[train_idx], Y[train_idx])
        rf_d.fit(X[train_idx], D[train_idx])

        # Add noise to predictions (simulating worse ML)
        pred_y = rf_y.predict(X[test_idx]) + noise_level * np.random.randn(len(test_idx))
        pred_d = rf_d.predict(X[test_idx]) + noise_level * np.random.randn(len(test_idx))

        resid_Y[test_idx] = Y[test_idx] - pred_y
        resid_D[test_idx] = D[test_idx] - pred_d

    theta = np.sum(resid_D * resid_Y) / np.sum(resid_D ** 2)
    return theta

# Test robustness across degradation levels
noise_levels = [0.0, 0.1, 0.2, 0.5, 1.0]
print(f"True effect: {true_theta:.2f}\n")
print(f"{'Noise Level':<15} {'DML Estimate':<15} {'Bias':<10}")
print("=" * 40)
for nl in noise_levels:
    theta_hat = estimate_with_degraded_ml(Y, D, X, noise_level=nl)
    print(f"{nl:<15.1f} {theta_hat:<15.3f} {theta_hat - true_theta:<+10.3f}")
```

DML remains approximately unbiased even with substantial noise in the ML predictions. This is the orthogonality property at work — errors in $\hat{g}$ and $\hat{m}$ contribute only as a second-order product.

## How the Naive Plug-in Fails

Compare the orthogonal score to a naive plug-in that only residualises $Y$ (not $D$):

```python
def naive_plugin(Y, D, X, noise_level=0.0):
    """Naive approach: only residualise Y, regress on raw D."""
    n = len(Y)
    resid_Y = np.zeros(n)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, test_idx in kf.split(X):
        rf_y = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_y.fit(X[train_idx], Y[train_idx])
        pred_y = rf_y.predict(X[test_idx]) + noise_level * np.random.randn(len(test_idx))
        resid_Y[test_idx] = Y[test_idx] - pred_y

    # Regress residualised Y on raw D (NOT residualised)
    theta = np.sum(D * resid_Y) / np.sum(D ** 2)
    return theta

print(f"\n{'Noise':<10} {'DML':<12} {'Naive':<12} {'DML Bias':<12} {'Naive Bias':<12}")
print("=" * 60)
for nl in noise_levels:
    dml_est = estimate_with_degraded_ml(Y, D, X, noise_level=nl)
    naive_est = naive_plugin(Y, D, X, noise_level=nl)
    print(f"{nl:<10.1f} {dml_est:<12.3f} {naive_est:<12.3f} "
          f"{dml_est - true_theta:<+12.3f} {naive_est - true_theta:<+12.3f}")
```

The naive plug-in diverges as noise increases, while DML stays close to the true effect. The "double" residualisation is essential.

> ⚠️ **Warning:** The orthogonal score requires residualising BOTH $Y$ and $D$. Residualising only one gives a non-orthogonal moment condition that is sensitive to first-stage errors. This is why the method is called "Double" ML — both nuisance functions must be estimated.

## How to Interpret the Score Function

The orthogonal score for the partially linear model can be written as:

$$\psi(W; \theta, g, m) = \underbrace{(D - m(X))}_{\text{treatment residual}} \cdot \underbrace{(Y - g(X) - \theta(D - m(X)))}_{\text{structural residual}}$$

Setting $E[\psi] = 0$ and solving for $\theta$:

$$\theta = \frac{E[(D - m_0(X))(Y - g_0(X))]}{E[(D - m_0(X))^2]}$$

This is the familiar residual-on-residual formula, but the score function reveals WHY it is robust: the Gateaux derivative with respect to $\eta = (g, m)$ vanishes at the true values, making the estimator locally insensitive to nuisance estimation errors.

## Connections

**Builds on:**
- Module 02: The orthogonalisation trick and residual-on-residual regression
- Influence functions in semiparametric statistics

**Leads to:**
- Module 04: Cross-fitting (second piece of the DML puzzle)
- Module 05: `doubleml` uses orthogonal scores internally
- Module 08: Orthogonal scores for CATE estimation

**Related concepts:**
- Semiparametric efficiency bounds (DML achieves the bound)
- Doubly robust estimation in biostatistics
- Targeted maximum likelihood estimation (TMLE)

## Practice Problems

### Conceptual

**1. Why "Double":**
Explain in your own words why residualising only $Y$ (not $D$) produces a biased estimator, while residualising both gives robustness. Use the bias formula $\text{Bias} \propto E[r_n \cdot V]$ to make your argument precise.

### Implementation

**2. Sensitivity Plot:**
Create a figure with two panels: (a) DML bias vs noise level, (b) naive plug-in bias vs noise level. Run 50 simulations at each noise level and plot mean bias with error bars.

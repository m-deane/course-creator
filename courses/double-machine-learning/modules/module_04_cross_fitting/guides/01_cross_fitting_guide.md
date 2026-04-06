# Cross-Fitting and Sample Splitting

> **Reading time:** ~5 min | **Module:** 4 — Cross Fitting | **Prerequisites:** Module 3 — Neyman Orthogonal Scores

## In Brief

You will learn why training ML models on the same data used for inference creates overfitting bias, and how K-fold cross-fitting eliminates it. Cross-fitting is the second pillar of DML (alongside orthogonal scores) — without it, even orthogonal estimators can be biased.

<div class="callout-insight">

<strong>Key Insight:</strong> In-sample ML predictions are too good — the model has partially memorised the training data. This makes residuals artificially small, which inflates the treatment effect. Cross-fitting ensures all predictions are genuinely out-of-sample, removing overfitting bias without sacrificing efficiency.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> You will learn why training ML models on the same data used for inference creates overfitting bias, and how K-fold cross-fitting eliminates it. Cross-fitting is the second pillar of DML (alongside orthogonal scores) — without it, even orthogonal estimators can be biased.

</div>

## Visual Explanation

```

CROSS-FITTING (K=5 folds)

Fold 1: [TRAIN TRAIN TRAIN TRAIN | TEST ]  → predict on fold 5
Fold 2: [TRAIN TRAIN TRAIN | TEST  TRAIN]  → predict on fold 4
Fold 3: [TRAIN TRAIN | TEST  TRAIN TRAIN]  → predict on fold 3
Fold 4: [TRAIN | TEST  TRAIN TRAIN TRAIN]  → predict on fold 2
Fold 5: [TEST  TRAIN TRAIN TRAIN TRAIN]   → predict on fold 1

Result: Every observation has an OUT-OF-SAMPLE prediction.
        No observation was used to train its own predictor.
```

<div class="compare">
<div class="compare-card">
<div class="header before">In-Sample Prediction (Biased)</div>
<div class="body">

- Model partially memorises training data
- Residuals artificially small
- Treatment effect estimate inflated
- Bias does not vanish with more data

</div>

</div>
<div class="compare-card">
<div class="header after">Cross-Fitted Prediction (Unbiased)</div>
<div class="body">

- Every observation predicted out-of-sample
- Residuals reflect genuine prediction error
- Treatment effect correctly estimated
- Converges at parametric rate

</div>

</div>

</div>

## Why In-Sample Predictions Create Bias

In a commodity context, suppose you are estimating the effect of inventory releases on oil futures spreads, with 50 market controls. If your random forest trains on all 2,000 observations and then predicts on the same data, it partially memorises each observation. This creates a subtle but damaging bias.

When an ML model trains on observation $i$ and then predicts observation $i$, the prediction is biased toward the observed value. For flexible models like random forests:

$$\hat{g}_{in}(X_i) \approx Y_i - \text{small residual}$$

This means $\tilde{Y}_i = Y_i - \hat{g}_{in}(X_i) \approx \text{small number}$. Similarly for $\tilde{D}_i$. The treatment effect formula becomes:

$$\hat{\theta} = \frac{\sum \tilde{D}_i \tilde{Y}_i}{\sum \tilde{D}_i^2}$$

With artificially small $\tilde{D}_i$ in the denominator, $\hat{\theta}$ is inflated.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

np.random.seed(42)
n = 2000
p = 50
true_theta = 1.0

X = np.random.randn(n, p)
D = 0.5 * X[:, 0] + 0.3 * X[:, 1] + np.random.randn(n) * 0.5
Y = true_theta * D + X[:, 0] + 0.5 * X[:, 2] + np.random.randn(n)

# In-sample predictions (WRONG)
rf_y = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, Y)
rf_d = RandomForestRegressor(n_estimators=200, random_state=42).fit(X, D)

resid_Y_insample = Y - rf_y.predict(X)
resid_D_insample = D - rf_d.predict(X)
theta_insample = np.sum(resid_D_insample * resid_Y_insample) / np.sum(resid_D_insample ** 2)

print(f"True effect:           {true_theta:.2f}")
print(f"In-sample DML:         {theta_insample:.2f}  (biased!)")
print(f"Mean |resid_D| in-sample: {np.mean(np.abs(resid_D_insample)):.4f}")
```

</div>

The in-sample residuals are too small because the random forest has memorised the training data.

## How Cross-Fitting Fixes the Problem

Cross-fitting uses K-fold sample splitting. For each fold $k$:
1. Train ML on all folds except $k$
2. Predict on fold $k$ (out-of-sample)
3. Compute residuals for fold $k$

Then concatenate all out-of-sample residuals and estimate $\theta$:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# Cross-fitting (CORRECT)
resid_Y_cf = np.zeros(n)
resid_D_cf = np.zeros(n)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X):
    rf_y = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_d = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_y.fit(X[train_idx], Y[train_idx])
    rf_d.fit(X[train_idx], D[train_idx])

    resid_Y_cf[test_idx] = Y[test_idx] - rf_y.predict(X[test_idx])
    resid_D_cf[test_idx] = D[test_idx] - rf_d.predict(X[test_idx])

theta_cf = np.sum(resid_D_cf * resid_Y_cf) / np.sum(resid_D_cf ** 2)

print(f"\nCross-fitted DML:      {theta_cf:.2f}  (unbiased!)")
print(f"Mean |resid_D| cross-fitted: {np.mean(np.abs(resid_D_cf)):.4f}")
```

</div>

<div class="callout-warning">

<strong>Warning:</strong> Simple sample splitting (50/50 train/test) also works but wastes half the data for inference. Cross-fitting uses all observations for both training and prediction, achieving full efficiency. Always use cross-fitting over simple splitting.

</div>

## How DML1 and DML2 Differ

Chernozhukov et al. (2018) define two aggregation methods:

**DML1:** Average fold-specific treatment effects.
$$\hat{\theta}_{DML1} = \frac{1}{K}\sum_{k=1}^K \hat{\theta}_k \quad \text{where} \quad \hat{\theta}_k = \frac{\sum_{i \in I_k} \tilde{D}_i \tilde{Y}_i}{\sum_{i \in I_k} \tilde{D}_i^2}$$

**DML2:** Pool all residuals, then estimate once.
$$\hat{\theta}_{DML2} = \frac{\sum_{i=1}^n \tilde{D}_i \tilde{Y}_i}{\sum_{i=1}^n \tilde{D}_i^2}$$

DML2 is generally preferred — it is more stable and recommended by the `doubleml` package.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
# DML1: average fold-specific estimates
thetas_per_fold = []
for train_idx, test_idx in kf.split(X):
    rY_k = resid_Y_cf[test_idx]
    rD_k = resid_D_cf[test_idx]
    theta_k = np.sum(rD_k * rY_k) / np.sum(rD_k ** 2)
    thetas_per_fold.append(theta_k)

theta_dml1 = np.mean(thetas_per_fold)
theta_dml2 = np.sum(resid_D_cf * resid_Y_cf) / np.sum(resid_D_cf ** 2)

print(f"DML1 (averaged):  {theta_dml1:.4f}")
print(f"DML2 (pooled):    {theta_dml2:.4f}")
print(f"Fold estimates:   {[f'{t:.3f}' for t in thetas_per_fold]}")
```

</div>

## How the Number of Folds Affects Results

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
for K in [2, 3, 5, 10, 20]:
    resid_Y_k = np.zeros(n)
    resid_D_k = np.zeros(n)
    kf_k = KFold(n_splits=K, shuffle=True, random_state=42)

    for train_idx, test_idx in kf_k.split(X):
        rf_y = RandomForestRegressor(100, random_state=42).fit(X[train_idx], Y[train_idx])
        rf_d = RandomForestRegressor(100, random_state=42).fit(X[train_idx], D[train_idx])
        resid_Y_k[test_idx] = Y[test_idx] - rf_y.predict(X[test_idx])
        resid_D_k[test_idx] = D[test_idx] - rf_d.predict(X[test_idx])

    theta_k = np.sum(resid_D_k * resid_Y_k) / np.sum(resid_D_k ** 2)
    print(f"K={K:2d}: theta = {theta_k:.4f}")
```

</div>

$K = 5$ is the default in most implementations and works well in practice. Larger $K$ uses more training data per fold but increases computation.

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- Module 02: The orthogonalisation trick
- Module 03: Neyman orthogonal scores
- Cross-validation in ML

**Leads to:**
- Module 05: `doubleml` uses cross-fitting by default
- Module 09: Production pipeline with optimal fold selection

**Related concepts:**
- Leave-one-out cross-validation (computationally prohibitive for DML)
- Repeated cross-fitting (average over multiple random fold assignments)

## Practice Problems

### Implementation

**1. In-Sample vs Cross-Fitted:**
Run 100 simulations comparing in-sample DML to cross-fitted DML. Plot the distribution of estimates for each. Which is centred on the true effect?

**2. DML1 vs DML2:**
Run 100 simulations comparing DML1 and DML2. Which has lower variance? Which is more robust to small fold sizes?


## Resources

<a class="link-card" href="../notebooks/01_cross_fitting_notebook.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

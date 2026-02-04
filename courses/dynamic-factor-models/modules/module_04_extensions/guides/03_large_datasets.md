# Large-Scale Dynamic Factor Models

## In Brief

High-dimensional DFMs (N > 100) require computational and statistical innovations: **two-step estimation** (PCA followed by Kalman filtering), **sparse factor loadings** (LASSO regularization), and **targeted predictors** (pre-selection of relevant series). These methods enable real-time nowcasting with hundreds of series while avoiding overfitting and computational bottlenecks.

## Key Insight

Full maximum likelihood with N=200 series requires inverting 200×200 matrices at every Kalman filter iteration—computationally prohibitive and prone to overfitting. Solution: Principal components provide consistent factor estimates for large N, then refine with Kalman smoothing. Result: 100× speedup with minimal accuracy loss. For variable selection: not all 200 series matter equally—use LASSO to identify the 20 that drive GDP forecasts.

---

## Visual Explanation

```
Computational Complexity:
═══════════════════════════════════════════════════════════

FULL MLE (Exact Kalman Filter):
  Operations per iteration: O(N²)
  Parameters to estimate: N×r + N + r²

  N=10:  100 ops, 100 params    → Fast ✓
  N=50:  2,500 ops, 500 params  → Manageable
  N=200: 40,000 ops, 2000 params → Slow ✗
  N=500: 250,000 ops, 5000 params → Impractical ✗✗


TWO-STEP ESTIMATION:
  Step 1: PCA (O(N·T·r), one-time)
  Step 2: Kalman smoother on factors (O(r²·T))

  Total: O(N·T·r) << O(N²·T)

  Speedup for N=200, r=5: ~40× faster!


Two-Step Procedure Flow:
═══════════════════════════════════════════════════════════

DATA: X (T x N)
  ↓
[STEP 1: PCA]
  ├→ Compute C = X'X / T
  ├→ Eigendecomp: C = Λ̂·Λ̂'
  └→ F̂ = X·Λ̂ / √T
  ↓
[STEP 2: Kalman Smoother]
  ├→ Estimate VAR on F̂: F_t = Φ̂·F_{t-1} + η_t
  ├→ Kalman smooth with Φ̂, Q̂, Λ̂ fixed
  └→ Refined F̃ (uses time-series info)
  ↓
OUTPUT: Factors F̃, Loadings Λ̂, Dynamics Φ̂


Sparse Loadings (LASSO):
═══════════════════════════════════════════════════════════

Standard DFM: All N variables load on all r factors
  ↓ N = 100, r = 5 → 500 loadings to estimate
  ↓ Many near-zero (noise)

Sparse DFM: Penalize small loadings
  min_{Λ,F} ||X - ΛF'||² + λ·||Λ||₁
                            ↑
                    L1 penalty (LASSO)

Result: Many loadings exactly zero
  ↓ Easier interpretation
  ↓ Less overfitting
  ↓ Identifies which variables matter

Example (5 variables, 2 factors):
  Standard:  Λ = [0.8  0.3]   All nonzero
                  [0.7  0.4]
                  [0.1  0.8]
                  [0.6  0.2]
                  [0.5  0.1]

  Sparse:    Λ = [0.8  0.0]   3 zeros!
                  [0.7  0.0]
                  [0.0  0.9]
                  [0.6  0.0]
                  [0.0  0.0]   ← Variable 5 dropped

Interpretation: Variables 1,2,4 load on Factor 1 (real activity)
                Variable 3 loads on Factor 2 (financial)
                Variable 5 is noise (excluded)
```

---

## Formal Definition

### Two-Step Estimator (Doz, Giannone, Reichlin 2011)

**Step 1: PCA estimation**

Solve eigenvalue problem for sample covariance:
$$\hat{\Sigma}_X = \frac{1}{T} X'X = \sum_{j=1}^N \mu_j v_j v_j'$$

Loadings: $\hat{\Lambda} = [\sqrt{\mu_1} v_1, ..., \sqrt{\mu_r} v_r]$ (first r eigenvectors, scaled)

Factors: $\hat{F}_t = \hat{\Lambda}' X_t$ (principal components)

**Step 2: Kalman smoothing**

Estimate factor VAR: $\hat{F}_t = \hat{\Phi} \hat{F}_{t-1} + \hat{\eta}_t$ via OLS

Run Kalman smoother with fixed parameters:
- Loadings Λ = Λ̂ (from Step 1)
- Dynamics Φ = Φ̂ (from VAR)
- Estimate Q, Σ_e from residuals

Output: Smoothed factors $\tilde{F}_t$ (incorporate time-series structure)

**Consistency:** As N, T → ∞, $\tilde{F}_t \xrightarrow{p} F_t$ (up to rotation)

### Sparse Factor Model

**LASSO-penalized objective:**
$$\min_{\Lambda, F} \frac{1}{2T} ||X - \Lambda F'||_F^2 + \lambda \sum_{i,j} |\lambda_{ij}|$$

**Estimator:** Coordinate descent or proximal gradient

**Tuning parameter:** Cross-validation for λ, or use BIC:
$$BIC(\lambda) = \log(\hat{\sigma}^2_\lambda) + \frac{\log(T)}{T} \cdot df(\lambda)$$

where df(λ) = number of nonzero loadings.

**Sparse PCA variant:**
$$\max_{\lambda_j: ||\lambda_j||_1 \leq c} \lambda_j' \hat{\Sigma}_X \lambda_j$$

Constrains L1-norm of loadings (induces sparsity).

---

## Code Implementation

### Two-Step Estimation

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def two_step_dfm(X, k_factors=5):
    """
    Fast two-step DFM estimation for large N.

    Parameters
    ----------
    X : array (T, N) or pd.DataFrame
        Data matrix
    k_factors : int
        Number of factors

    Returns
    -------
    factors_smoothed : array (T, r)
        Kalman-smoothed factors
    loadings : array (N, r)
        Factor loadings
    params : dict
        Estimated parameters
    """
    if isinstance(X, pd.DataFrame):
        X_array = X.values
        index = X.index
        columns = X.columns
    else:
        X_array = X
        index = None
        columns = None

    T, N = X_array.shape

    # STEP 1: PCA
    # Center data
    X_centered = X_array - X_array.mean(axis=0)

    # Compute covariance
    Sigma_X = X_centered.T @ X_centered / T

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(Sigma_X)

    # Sort descending
    idx = eigvals.argsort()[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Loadings (first r eigenvectors, scaled by sqrt(eigenvalue))
    loadings = eigvecs[:, :k_factors] * np.sqrt(eigvals[:k_factors])

    # Factors (principal components)
    factors_pca = X_centered @ eigvecs[:, :k_factors]

    print(f"Step 1: PCA complete")
    print(f"  Variance explained: {eigvals[:k_factors].sum() / eigvals.sum():.1%}")

    # STEP 2: VAR and Kalman Smoothing
    # Estimate VAR(1) on PC factors
    var_model = VAR(factors_pca)
    var_result = var_model.fit(maxlags=1, ic=None)

    Phi_hat = var_result.params[1:].T  # VAR(1) coefficient matrix
    Q_hat = var_result.sigma_u  # Residual covariance

    # Estimate measurement error covariance
    factor_fitted = factors_pca @ loadings.T
    resid = X_centered - factor_fitted
    Sigma_e_hat = np.diag(np.var(resid, axis=0))

    print(f"Step 2: VAR estimation complete")

    # Kalman smoother (simplified - use statsmodels for full version)
    # Here we just return PCA factors; full implementation needs smoother
    factors_smoothed = factors_pca  # Placeholder

    if index is not None:
        factors_smoothed = pd.DataFrame(
            factors_smoothed,
            index=index,
            columns=[f'Factor_{i+1}' for i in range(k_factors)]
        )

    return factors_smoothed, loadings, {
        'Phi': Phi_hat,
        'Q': Q_hat,
        'Sigma_e': Sigma_e_hat,
        'eigenvalues': eigvals
    }

# Example: Large dataset
np.random.seed(42)
T, N, r = 200, 100, 5

# True factors
F_true = np.random.randn(T, r)
for t in range(1, T):
    F_true[t] = 0.7 * F_true[t-1] + 0.3 * np.random.randn(r)

# Sparse loadings
Lambda_true = np.zeros((N, r))
Lambda_true[:20, 0] = np.random.randn(20) * 0.8  # First 20 on factor 1
Lambda_true[20:40, 1] = np.random.randn(20) * 0.8  # Next 20 on factor 2
# etc.

X = F_true @ Lambda_true.T + 0.5 * np.random.randn(T, N)

# Two-step estimation
factors, loadings, params = two_step_dfm(X, k_factors=5)

print(f"\nEstimation complete!")
print(f"  Factors shape: {factors.shape}")
print(f"  Loadings shape: {loadings.shape}")
```

### Sparse Factor Model (LASSO Loadings)

```python
from sklearn.linear_model import Lasso

def sparse_factor_model(X, k_factors=5, alpha=0.1):
    """
    Estimate sparse factor model using LASSO.

    Parameters
    ----------
    X : array (T, N)
    k_factors : int
    alpha : float
        LASSO regularization parameter

    Returns
    -------
    factors : array (T, r)
    sparse_loadings : array (N, r)
        Many entries will be exactly zero
    """
    # Initialize with PCA
    factors_init, loadings_init, _ = two_step_dfm(X, k_factors)

    T, N = X.shape
    sparse_loadings = np.zeros((N, k_factors))

    # For each variable, regress on factors with LASSO penalty
    for i in range(N):
        lasso = Lasso(alpha=alpha, fit_intercept=False)
        lasso.fit(factors_init, X[:, i])
        sparse_loadings[i] = lasso.coef_

    # Re-estimate factors given sparse loadings
    # (Placeholder - full algorithm iterates)
    factors = factors_init

    # Count sparsity
    sparsity = (sparse_loadings == 0).sum() / (N * k_factors) * 100

    print(f"Sparse factor model:")
    print(f"  {sparsity:.1f}% of loadings set to zero")

    return factors, sparse_loadings

# Example
factors_sparse, loadings_sparse = sparse_factor_model(X, k_factors=5, alpha=0.05)
```

### Targeted Predictors (Pre-Selection)

```python
def targeted_predictors(X, y, k_factors=3, n_select=20):
    """
    Pre-select most relevant predictors before DFM estimation.

    Parameters
    ----------
    X : pd.DataFrame (T x N)
        Candidate predictors
    y : pd.Series (T)
        Target variable
    n_select : int
        Number of predictors to keep

    Returns
    -------
    X_selected : pd.DataFrame (T x n_select)
        Top predictors
    """
    from sklearn.linear_model import LassoCV

    # LASSO to select relevant predictors
    lasso = LassoCV(cv=5, fit_intercept=True, max_iter=5000)
    lasso.fit(X, y)

    # Select variables with nonzero coefficients
    nonzero_idx = np.where(lasso.coef_ != 0)[0]

    if len(nonzero_idx) > n_select:
        # Rank by absolute coefficient size
        ranked_idx = np.argsort(np.abs(lasso.coef_[nonzero_idx]))[::-1]
        selected_idx = nonzero_idx[ranked_idx[:n_select]]
    else:
        selected_idx = nonzero_idx

    X_selected = X.iloc[:, selected_idx]

    print(f"Targeted predictor selection:")
    print(f"  Selected {len(selected_idx)} of {X.shape[1]} predictors")
    print(f"  Top 5: {list(X_selected.columns[:5])}")

    return X_selected

# Example: Select from 100 predictors
y_target = X[:, 0] + X[:, 1] + 0.5 * np.random.randn(T)  # Depends on first 2
X_df = pd.DataFrame(X, columns=[f'Var{i}' for i in range(N)])
y_series = pd.Series(y_target)

X_selected = targeted_predictors(X_df, y_series, n_select=20)
```

---

## Common Pitfalls

### 1. Using PCA Without Kalman Smoothing

**Problem:** PCA factors ignore time-series structure (autocorrelation)

**Impact:** Loses 10-15% efficiency compared to two-step

**Solution:** Always run Step 2 (Kalman smoother)

### 2. Over-Regularization

**Problem:** LASSO α too large → all loadings shrunk to zero

**Solution:** Use cross-validation or BIC to select α

### 3. Ignoring Standardization

**Problem:** Variables in different units (GDP in %, interest rate in levels)

**Solution:** Standardize before PCA: $\tilde{X}_i = (X_i - \mu_i) / \sigma_i$

### 4. Selecting Too Many Factors

**Problem:** With N=100, tempted to use r=20 factors → overfitting

**Rule of thumb:** r ≤ √N, typically 3-10 sufficient

### 5. Not Accounting for Sparsity in Inference

**Problem:** Standard errors ignore that some loadings were shrunk to zero

**Solution:** Post-selection inference (Lee et al. 2016) or bootstrap

---

## Connections

### Builds On
- **PCA Theory**: Consistency for large N
- **Two-Step Estimation**: Computational efficiency
- **Regularization**: LASSO, Ridge, Elastic Net

### Leads To
- **Big Data Econometrics**: Streaming data, online updates
- **Machine Learning Integration**: Neural network factor models
- **Distributed Computing**: Parallel factor estimation

---

## Practice Problems

### Implementation

1. **Benchmark two-step vs full MLE:**
   - Simulate N=50, T=200
   - Time both methods
   - Compare factor correlation

2. **Sparse loading recovery:**
   - Simulate sparse true loadings
   - Estimate with LASSO
   - Measure sparsity recovery accuracy

3. **FRED-MD analysis:**
   - Download 127 series from FRED-MD
   - Two-step estimation with r=5
   - Interpret extracted factors

---

## Further Reading

### Essential

- **Doz, C., Giannone, D., & Reichlin, L. (2011).** "A Two-Step Estimator for Large Approximate Dynamic Factor Models Based on Kalman Filtering." *Journal of Econometrics, 164*(1), 188-205.
  - *Foundational two-step paper*

### Advanced

- **Bai, J., & Ng, S. (2008).** "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics, 3*(2), 89-163.
  - *Comprehensive theoretical treatment*

---

**Next:** Module cheatsheet

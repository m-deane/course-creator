# Stock-Watson Two-Step Estimator

> **Reading time:** ~16 min | **Module:** Module 3: Estimation Pca | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** The Stock-Watson two-step estimator provides a computationally efficient method for estimating dynamic factor models: first extract factors via principal components analysis, then estimate factor dynamics via OLS regression. This approach avoids the computational burden of joint maximum likelihoo...

</div>

## In Brief

The Stock-Watson two-step estimator provides a computationally efficient method for estimating dynamic factor models: first extract factors via principal components analysis, then estimate factor dynamics via OLS regression. This approach avoids the computational burden of joint maximum likelihood while maintaining consistency under large N and T asymptotics.

<div class="callout-insight">

**Insight:** Instead of estimating factors and loadings jointly (which requires iterative optimization), we leverage the observation that PCA consistently estimates the factor space when N and T are large. Once we have factor estimates, the dynamics reduce to a standard VAR that can be estimated by OLS. This decomposition transforms a difficult high-dimensional problem into two simple steps.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Formal Definition

### The Stock-Watson Algorithm

**Model:**
$$X_{it} = \lambda_i' F_t + e_{it}$$
$$F_t = \Phi_1 F_{t-1} + ... + \Phi_p F_{t-p} + \eta_t$$

**Step 1: Extract Factors via PCA**

Solve the eigenproblem:
$$\hat{\Sigma}_X = \frac{1}{T} XX' = V D V'$$

where $V = [v_1, ..., v_N]$ are eigenvectors ordered by eigenvalues $d_1 \geq d_2 \geq ... \geq d_N$ in diagonal matrix $D$.

**Factor estimates:**
$$\hat{F} = \frac{1}{\sqrt{T}} V_r' X$$

where $V_r = [v_1, ..., v_r]$ contains the first $r$ eigenvectors corresponding to the $r$ largest eigenvalues.

**Loading estimates:**
$$\hat{\Lambda} = X' \hat{F} / T = \sqrt{T} \cdot V_r D_r$$

where $D_r$ is the $r \times r$ diagonal matrix of the first $r$ eigenvalues.

**Step 2: Estimate Factor Dynamics**

Run OLS regression:
$$\hat{F}_t = \hat{\Phi}_1 \hat{F}_{t-1} + ... + \hat{\Phi}_p \hat{F}_{t-p} + \hat{\eta}_t$$

for $t = p+1, ..., T$.

### Normalization Conventions

PCA factors satisfy the normalization:
$$\frac{1}{T} \hat{F}' \hat{F} = I_r$$

This identifies the factor scale but leaves rotation indeterminate. The loadings absorb the scale:
$$\hat{\Lambda}' \hat{\Lambda} = T \cdot D_r^2$$

is diagonal with factor variance on the diagonal.

---

## 2. Intuitive Explanation

### Why This Works

**The PCA Logic:** Imagine you observe test scores for 1,000 students across 50 subjects. If students have varying levels of "general intelligence" and "verbal vs. quantitative skill," those latent factors will create patterns in the correlation matrix. PCA finds the directions of maximum variance—exactly where the latent factors should be if they're driving most of the co-movement.

**The Two-Step Logic:** Once we've extracted the factors in Step 1, estimating their dynamics is just forecasting the factors from their own past values—a standard time series problem. We don't need to re-estimate the factors while learning the dynamics; the factors are already consistently estimated from Step 1.

### Geometric Interpretation

Think of the $N$-dimensional data cloud $X_1, ..., X_T$. The factor model says this cloud lies approximately in an $r$-dimensional subspace (the "factor space"). PCA finds this subspace by:
1. Computing the sample covariance ellipsoid
2. Finding its principal axes
3. Projecting the data onto the top $r$ axes

The projections are the factor estimates $\hat{F}_t$.

### Visual Analogy

```
Original Data (N = 3, r = 2)           PCA Factor Space
        X₃                                   F₂
         ↑                                    ↑
         |   •                                |  •
         |  • •                               | • •
         | •  •  •                            |•  • •
         |•   •   •                           ○────○→ F₁
         •─────────→ X₁                       •
        /   •  •  •                          /
       X₂   •   •                           (plane of data)
            •
```

The data cloud is approximately flat (2D) in 3D space. PCA finds this plane.

---

## 3. Mathematical Formulation

### Asymptotic Theory (Informal)

Under regularity conditions (detailed in next guide), as $N, T \to \infty$ with $\sqrt{T}/N \to 0$:

$$\|\hat{F}_t - H F_t\| = O_p\left(\min\left(N^{-1/2}, T^{-1/2}\right)\right)$$

where $H$ is an $r \times r$ rotation matrix. Key points:

<div class="flow">
<div class="flow-step mint">1. Consistency:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Rotation indetermina...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Rate of convergence:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Fast rate:</div>
</div>


1. **Consistency:** Estimation error vanishes asymptotically
2. **Rotation indeterminacy:** Factors estimated up to rotation $H$
3. **Rate of convergence:** $\min(\sqrt{N}, \sqrt{T})$ rate
4. **Fast rate:** Faster than $T^{-1/2}$ for single time series

### Why PCA Estimates Factors Consistently

**Intuition:** In population, the factor model implies:
$$\Sigma_X = \Lambda \Sigma_F \Lambda' + \Sigma_e$$

When $N$ is large and idiosyncratic errors are weakly correlated, the top eigenspaces of $\Sigma_X$ are dominated by $\Lambda \Sigma_F \Lambda'$. Thus, eigenvectors of $\Sigma_X$ span the factor space spanned by columns of $\Lambda$.

**Formal (simplified):** As $N \to \infty$, eigenvalues from factors diverge to infinity (proportional to $N$) while eigenvalues from idiosyncratic variance remain bounded. This separation ensures we can distinguish factors from noise.

### Variance Decomposition

After estimation, we can decompose variance:

**Total variance explained by $r$ factors:**
$$\frac{1}{NT} \sum_{i=1}^N \sum_{t=1}^T (\hat{\lambda}_i' \hat{F}_t)^2 = \frac{1}{NT} \|\hat{F}\hat{\Lambda}'\|_F^2 = \frac{1}{N} \sum_{j=1}^r d_j$$

where $d_j$ are the eigenvalues.

**Proportion of variance explained:**
$$R^2 = \frac{\sum_{j=1}^r d_j}{\sum_{j=1}^N d_j}$$

**Per-variable $R^2$:**
$$R_i^2 = \frac{\hat{\lambda}_i' \hat{\lambda}_i}{\text{Var}(X_i)} = \frac{\sum_{j=1}^r \hat{\lambda}_{ij}^2}{\frac{1}{T}\sum_t X_{it}^2}$$

(assuming standardized data)

---

## 4. Code Implementation

### Complete Implementation from Scratch

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">stockwatsonestimator.py</span>
</div>

```python
import numpy as np
from numpy.linalg import eigh, lstsq
from scipy import stats
import matplotlib.pyplot as plt

class StockWatsonEstimator:
    """
    Stock-Watson two-step estimator for dynamic factor models.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract
    factor_lags : int
        Number of lags in factor VAR
    standardize : bool, default=True
        Whether to standardize variables before estimation
    """

    def __init__(self, n_factors, factor_lags=1, standardize=True):
        self.r = n_factors
        self.p = factor_lags
        self.standardize = standardize

        # Will be populated by fit()
        self.F_hat = None
        self.Lambda_hat = None
        self.Phi_hat = None
        self.eigenvalues = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        """
        Estimate factors and factor dynamics.

        Parameters
        ----------
        X : array-like, shape (T, N)
            Panel data matrix (time x variables)

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.asarray(X)
        T, N = X.shape

        if T < self.r:
            raise ValueError(f"Need T >= n_factors, got T={T}, r={self.r}")
        if N < self.r:
            raise ValueError(f"Need N >= n_factors, got N={N}, r={self.r}")

        # Step 0: Standardize if requested
        if self.standardize:
            self.mean_ = np.mean(X, axis=0)
            self.std_ = np.std(X, axis=0, ddof=1)
            # Avoid division by zero
            self.std_[self.std_ < 1e-10] = 1.0
            X_std = (X - self.mean_) / self.std_
        else:
            X_std = X.copy()
            self.mean_ = np.zeros(N)
            self.std_ = np.ones(N)

        # Step 1: Extract factors via PCA
        self._extract_factors_pca(X_std)

        # Step 2: Estimate factor VAR by OLS
        self._estimate_factor_var()

        return self

    def _extract_factors_pca(self, X):
        """
        Extract factors via principal components.

        Solves eigenvalue problem for X'X / T and extracts top r eigenvectors.
        """
        T, N = X.shape

        # Compute sample covariance matrix
        Sigma_X = X.T @ X / T  # N x N

        # Eigenvalue decomposition (eigh returns ascending order)
        eigenvalues, eigenvectors = eigh(Sigma_X)

        # Reverse to descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store all eigenvalues (useful for diagnostics)
        self.eigenvalues = eigenvalues

        # Extract top r eigenvectors
        V_r = eigenvectors[:, :self.r]  # N x r

        # Compute factor estimates: F_hat = (1/sqrt(T)) * V_r' * X'
        # Equivalently: F_hat = X @ V_r / sqrt(T)
        self.F_hat = X @ V_r / np.sqrt(T)  # T x r

        # Compute loading estimates: Lambda_hat = X' @ F_hat / T
        # With normalization F'F/T = I, this gives Lambda = sqrt(T) * V_r * D_r
        self.Lambda_hat = X.T @ self.F_hat / T  # N x r

        # Alternative (equivalent): Lambda_hat = sqrt(T) * V_r @ diag(sqrt(eigenvalues[:r]))
        # self.Lambda_hat = np.sqrt(T) * V_r * np.sqrt(eigenvalues[:self.r])

    def _estimate_factor_var(self):
        """
        Estimate factor VAR(p) by OLS.

        Regression: F_t = Phi_1 F_{t-1} + ... + Phi_p F_{t-p} + eta_t
        """
        T, r = self.F_hat.shape
        p = self.p

        if T <= p:
            raise ValueError(f"Need T > factor_lags, got T={T}, p={p}")

        # Construct lagged factor matrix
        # X_reg = [F_{p}, F_{p+1}, ..., F_{T-1}]
        # Y_reg = [F_{p+1}, F_{p+2}, ..., F_{T}]

        Y = self.F_hat[p:, :]  # (T-p) x r

        # Stack lags: [F_{t-1}, F_{t-2}, ..., F_{t-p}]
        X_lags = np.column_stack([
            self.F_hat[p-lag-1:-lag-1, :] for lag in range(p)
        ])  # (T-p) x (r*p)

        # Add intercept
        X_reg = np.column_stack([np.ones(T - p), X_lags])  # (T-p) x (1 + r*p)

        # OLS: minimize ||Y - X_reg @ Phi||^2
        Phi_stacked, residuals, rank, s = lstsq(X_reg, Y, rcond=None)

        # Extract coefficients
        # Phi_stacked is (1 + r*p) x r
        # First row is intercept (should be near zero for demeaned factors)
        self.intercept_ = Phi_stacked[0, :]  # r x 1

        # Remaining rows are [Phi_1, Phi_2, ..., Phi_p] stacked
        self.Phi_hat = Phi_stacked[1:, :].T.reshape(r, r, p)  # r x r x p

        # Store residuals
        self.factor_residuals_ = Y - X_reg @ Phi_stacked
        self.residual_cov_ = np.cov(self.factor_residuals_.T)

    def transform(self, X):
        """
        Extract factors for new data using estimated loadings.

        Parameters
        ----------
        X : array-like, shape (T_new, N)
            New observations

        Returns
        -------
        F_new : ndarray, shape (T_new, r)
            Factor estimates
        """
        if self.Lambda_hat is None:
            raise ValueError("Model not fitted. Call fit() first.")

        X = np.asarray(X)

        # Standardize using training statistics
        if self.standardize:
            X_std = (X - self.mean_) / self.std_
        else:
            X_std = X

        # Project onto factor space: F = X @ (Lambda (Lambda' Lambda)^{-1})
        # Under normalization F'F/T = I, we have Lambda' Lambda = T * diag(eigenvalues)
        # Simpler: use pseudo-inverse
        F_new = X_std @ np.linalg.pinv(self.Lambda_hat.T)

        return F_new

    def forecast(self, h=1):
        """
        Forecast factors h periods ahead using VAR estimates.

        Parameters
        ----------
        h : int
            Forecast horizon

        Returns
        -------
        F_forecast : ndarray, shape (h, r)
            Factor forecasts
        """
        if self.Phi_hat is None:
            raise ValueError("Model not fitted. Call fit() first.")

        r, p = self.F_hat.shape[1], self.p
        F_forecast = np.zeros((h, r))

        # Initialize with last p observations
        F_history = self.F_hat[-p:, :].copy()  # p x r

        for t in range(h):
            # Forecast: F_{T+t+1} = sum_{j=1}^p Phi_j @ F_{T+t+1-j}
            F_t = self.intercept_.copy()
            for lag in range(p):
                if t - lag < 0:
                    # Use historical data
                    F_t += self.Phi_hat[:, :, lag] @ F_history[p + t - lag - 1, :]
                else:
                    # Use forecasted data
                    F_t += self.Phi_hat[:, :, lag] @ F_forecast[t - lag - 1, :]

            F_forecast[t, :] = F_t

        return F_forecast

    def explained_variance_ratio(self):
        """
        Compute proportion of variance explained by each factor.

        Returns
        -------
        ratios : ndarray, shape (r,)
            Variance ratio for each factor
        """
        if self.eigenvalues is None:
            raise ValueError("Model not fitted.")

        total_var = np.sum(self.eigenvalues)
        return self.eigenvalues[:self.r] / total_var

    def variable_r_squared(self):
        """
        Compute R-squared for each variable.

        Returns
        -------
        r_squared : ndarray, shape (N,)
            R-squared for each variable
        """
        if self.Lambda_hat is None:
            raise ValueError("Model not fitted.")

        # R^2_i = lambda_i' lambda_i / Var(X_i)
        # For standardized data, Var(X_i) = 1
        loadings_sq_sum = np.sum(self.Lambda_hat**2, axis=1)

        if self.standardize:
            return loadings_sq_sum  # Var = 1
        else:
            # Need to compute from original data variances
            return loadings_sq_sum / (self.std_**2)


# ============================================================================
# Demonstration
# ============================================================================

def simulate_dfm(T=200, N=50, r=3, p=1, noise_ratio=0.5, seed=42):
    """
    Simulate data from a dynamic factor model.

    Returns
    -------
    X : ndarray, shape (T, N)
        Observed data
    F_true : ndarray, shape (T, r)
        True factors
    Lambda_true : ndarray, shape (N, r)
        True loadings
    """
    np.random.seed(seed)

    # Generate factor loadings
    Lambda_true = np.random.randn(N, r) * 0.7

    # Generate factor VAR(p)
    Phi_true = np.zeros((r, r, p))
    for lag in range(p):
        Phi_true[:, :, lag] = np.random.randn(r, r) * 0.3 / (lag + 1)

    # Ensure stationarity (rough heuristic)
    for j in range(r):
        Phi_true[j, j, 0] = 0.6

    # Simulate factors
    F_true = np.zeros((T + 100, r))  # Burn-in
    for t in range(p, T + 100):
        F_t = np.random.randn(r) * 0.5  # Innovation
        for lag in range(p):
            F_t += Phi_true[:, :, lag] @ F_true[t - lag - 1, :]
        F_true[t, :] = F_t

    F_true = F_true[-T:, :]  # Drop burn-in

    # Generate idiosyncratic errors
    psi = np.ones(N) * noise_ratio
    e = np.random.randn(T, N) * psi

    # Generate observations
    X = F_true @ Lambda_true.T + e

    return X, F_true, Lambda_true


# Generate data
print("Simulating dynamic factor model...")
X, F_true, Lambda_true = simulate_dfm(T=300, N=60, r=3, p=2, noise_ratio=0.4)
T, N = X.shape
print(f"Data dimensions: T={T}, N={N}")

# Fit Stock-Watson estimator
print("\nFitting Stock-Watson two-step estimator...")
model = StockWatsonEstimator(n_factors=3, factor_lags=2, standardize=True)
model.fit(X)

# Results
print("\nVariance explained by factors:")
var_ratios = model.explained_variance_ratio()
for i, ratio in enumerate(var_ratios):
    print(f"  Factor {i+1}: {ratio:.1%}")
print(f"  Total: {np.sum(var_ratios):.1%}")

# Average R-squared across variables
avg_r2 = np.mean(model.variable_r_squared())
print(f"\nAverage R-squared: {avg_r2:.1%}")

# Align factors to true factors (resolve rotation)
# Use Procrustes alignment: minimize ||F_true - F_hat @ H||
from scipy.linalg import orthogonal_procrustes
H, _ = orthogonal_procrustes(model.F_hat, F_true)
F_aligned = model.F_hat @ H

# Compute alignment quality
correlation_per_factor = np.array([
    np.corrcoef(F_true[:, i], F_aligned[:, i])[0, 1]
    for i in range(3)
])
print(f"\nFactor correlations (after alignment):")
for i, corr in enumerate(correlation_per_factor):
    print(f"  Factor {i+1}: {corr:.3f}")

# Forecast factors
print("\nForecasting 10 periods ahead...")
F_forecast = model.forecast(h=10)
print(f"Forecast shape: {F_forecast.shape}")
print(f"First forecast:\n{F_forecast[0, :]}")
```

</div>

### Output (Representative)

```
Simulating dynamic factor model...
Data dimensions: T=300, N=60

Fitting Stock-Watson two-step estimator...

Variance explained by factors:
  Factor 1: 45.3%
  Factor 2: 18.7%
  Factor 3: 12.4%
  Total: 76.4%

Average R-squared: 65.2%

Factor correlations (after alignment):
  Factor 1: 0.987
  Factor 2: 0.971
  Factor 3: 0.963

Forecasting 10 periods ahead...
Forecast shape: (10, 3)
First forecast:
[ 0.234 -0.189  0.412]
```

### Using with scikit-learn API

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Alternative: Use sklearn PCA for Step 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=3)
F_hat_sklearn = pca.fit_transform(X_scaled)

# Step 2: Estimate VAR manually
# (Similar to our implementation above)
```

</div>

---

## 5. Common Pitfalls

### 1. Forgetting to Standardize

**Problem:** Variables with larger variance dominate the principal components.

**Example:** GDP (trillions) vs. unemployment rate (percentage points) will make PCA extract "GDP" as the first factor.

**Solution:** Always standardize to unit variance before PCA:
```python
X_std = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
```

### 2. Wrong Eigenvector Normalization

**Problem:** Some implementations use $F = X @ V_r$ instead of $F = X @ V_r / \sqrt{T}$.

**Consequence:** Factors don't satisfy $F'F/T = I_r$ normalization, breaking asymptotic theory.

**Check:** After estimation, verify:
```python
np.allclose(F_hat.T @ F_hat / T, np.eye(r), atol=1e-6)
```

### 3. Ignoring Sign and Rotation Indeterminacy

**Problem:** PCA factors are only identified up to sign flips and rotations. Comparing factors across different samples or software can give spurious "disagreement."

**Solution:**
- Use Procrustes alignment when comparing to ground truth
- Use economic/statistical identification for interpretation
- Report factor loadings alongside factors

### 4. Using PCA Covariance Instead of Correlation

**Problem:** `np.cov(X.T)` vs. `np.corrcoef(X.T)` give different results.

**Clarification:**
- Covariance PCA = variance-weighted
- Correlation PCA = equal weighting (equivalent to standardizing first)

For factor models, correlation PCA (standardization) is standard.

### 5. Overfitting with Too Many Factors

**Problem:** Including too many factors captures idiosyncratic noise rather than common variation.

**Symptom:** Later factors explain little additional variance, have weak loadings.

**Solution:** Use information criteria (Bai-Ng) or scree plot to select $r$ (next guide).

### 6. Insufficient Sample Size

**Problem:** Need $T$ and $N$ both large for consistency. With small $T$, overfitting occurs.

**Rule of Thumb:**
- Minimum: $T > 10 \times r \times p$
- Comfortable: $T > 100$, $N > 50$

---

## 6. Connections

### Builds On
- **PCA (Module 0):** Stock-Watson is PCA applied to factor model
- **Static Factor Models (Module 1):** Adds dynamics to static structure
- **VAR Models:** Step 2 is standard VAR estimation

### Leads To
- **Asymptotic Theory (Next Guide):** Formal convergence rates
- **Factor Number Selection (Guide 3):** How to choose $r$
- **Maximum Likelihood (Module 4):** Alternative to two-step PCA
- **Forecasting (Module 6):** Using estimated factors for prediction

### Related To
- **Kalman Filter (Module 2):** Alternative estimation via state space
- **Sparse Methods (Module 7):** PCA with sparsity constraints

---

## 7. Practice Problems

### Conceptual

1. **Why Two Steps?** Why not estimate factors and dynamics jointly? What computational advantage does the two-step approach provide?

2. **Rotation Indeterminacy:** If $F_t$ are factors and $H$ is an $r \times r$ invertible matrix, show that $\tilde{F}_t = H^{-1} F_t$ and $\tilde{\Lambda} = \Lambda H$ give the same $X_t$.

3. **Large N Intuition:** Explain intuitively why factor estimates become more accurate as $N$ increases, even with fixed $T$.

### Mathematical

4. **Derive Loading Formula:** Show that $\hat{\Lambda} = X' \hat{F} / T = \sqrt{T} V_r D_r$ where $D_r = \text{diag}(\sqrt{d_1}, ..., \sqrt{d_r})$.

5. **VAR Companion Form:** Write the factor VAR(2) in companion form as a VAR(1). How does this affect forecasting?

6. **Forecast Variance:** Derive the 1-step-ahead forecast variance $\text{Var}(F_{T+1} | F_T, ..., F_{T-p+1})$ for a VAR(p).

### Implementation

7. **Validation:** Modify the simulation to include a 4th weak factor (explaining <5% variance). Does Stock-Watson correctly identify it as negligible?

8. **Comparison:** Implement a naive forecast $\hat{F}_{T+1} = \bar{F}$ (sample mean). Compare MSE to VAR forecast. When does the VAR win?

9. **Missing Data:** Add 10% missing values to the simulated data. How does PCA performance degrade? (Teaser for Guide 3)

---

## 8. Further Reading

### Foundational Papers

- **Stock, J.H. & Watson, M.W. (2002).** "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association* 97(460), 1167-1179.
  - Original presentation of two-step estimator
  - Empirical application to forecasting inflation and IP
  - Establishes asymptotic properties

- **Stock, J.H. & Watson, M.W. (2002).** "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics* 20(2), 147-162.
  - Companion paper with more empirical detail
  - Comparison to other forecasting methods

### Asymptotic Theory

- **Bai, J. (2003).** "Inferential Theory for Factor Models of Large Dimensions." *Econometrica* 71(1), 135-171.
  - Formal proofs of consistency and convergence rates
  - Conditions for $\min(\sqrt{N}, \sqrt{T})$ rate
  - Distribution theory for loadings and factors

### Textbook Treatments

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics* Vol. 2A, 415-525.
  - Section 3: "Estimation by Principal Components"
  - Comprehensive overview with proofs sketched
  - Connects to broader literature

### Practical Implementation

- **McCracken, M.W. & Ng, S. (2016).** "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics* 34(4), 574-589.
  - Standard dataset for DFM applications
  - Preprocessing recommendations
  - Empirical benchmark results

---

## 9. Computational Considerations

### Speed Comparison

For $N = 127$, $T = 500$, $r = 5$:

| Method | Time | Notes |
|--------|------|-------|
| PCA (numpy) | 0.05s | Eigendecomposition of $127 \times 127$ |
| PCA (sklearn) | 0.08s | Includes standardization |
| EM-MLE | 30s | Iterative optimization |
| Kalman filter | 12s | State-space recursions |

**Conclusion:** PCA is 100-600x faster than likelihood methods.

### Memory Requirements

- PCA: Store $N \times N$ covariance matrix → $O(N^2)$
- For very large $N$ (e.g., $N > 10,000$): Use randomized PCA
- statsmodels: Stores full Kalman gain matrices → $O(TNr)$

### Parallelization

Step 1 (PCA) uses BLAS/LAPACK (automatically parallelized).
Step 2 (VAR OLS) can parallelize across equations (estimate each factor separately).

---

<div class="callout-insight">

**Insight:** Understanding stock-watson two-step estimator is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Summary

The Stock-Watson two-step estimator provides a simple, fast, and asymptotically valid method for estimating dynamic factor models:

1. **Step 1:** Extract factors via PCA (eigendecomposition)
2. **Step 2:** Estimate factor VAR via OLS regression

**Advantages:**
- Computationally efficient (no iterative optimization)
- No starting value sensitivity
- Asymptotically consistent under large $N, T$
- Easily implemented with standard tools

**Limitations:**
- Less efficient than ML in small samples
- No standard errors for loadings
- Requires both $N$ and $T$ large

**Next:** We formalize the asymptotic theory justifying this procedure.

---

## Conceptual Practice Questions

1. Explain the core idea of stock-watson two-step estimator in your own words to a colleague who has not studied it.

2. What is the most common mistake practitioners make when applying stock-watson two-step estimator, and how would you avoid it?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_stock_watson_estimator_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_stock_watson_estimation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_factor_number_selection.md">
  <div class="link-card-title">02 Factor Number Selection</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_missing_data_handling.md">
  <div class="link-card-title">03 Missing Data Handling</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


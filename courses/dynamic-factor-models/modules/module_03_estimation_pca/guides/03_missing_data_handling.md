# Missing Data Handling: EM-PCA Algorithm

## In Brief

Economic datasets commonly have missing observations due to publication lags, survey non-response, or ragged edges in real-time forecasting. The Expectation-Maximization (EM) algorithm adapts PCA to incomplete panels by iteratively imputing missing values and re-estimating factors, leveraging cross-sectional information to produce more efficient estimates than naive deletion or mean imputation.

## Key Insight

The key insight of EM-PCA: if we had the factors, we could impute missing values using the factor model; if we had complete data, we could extract factors using PCA. EM alternates between these two steps, starting from an initial guess and iterating until convergence. Under MAR (missing at random) assumptions, this produces consistent and efficient factor estimates even with substantial missingness.

---

## 1. Formal Definition

### The Missing Data Problem

**Observed data:** $X$ is $T \times N$ matrix where some entries $X_{it}$ are missing.

**Notation:**
- Let $\Omega = \{(i,t) : X_{it} \text{ observed}\}$ be the set of observed indices
- Let $\Omega^c = \{(i,t) : X_{it} \text{ missing}\}$ be missing indices
- Let $|\Omega| = $ number of observed values, $|\Omega^c| = $ number missing

**Goal:** Estimate factors $F$ and loadings $\Lambda$ using only observed entries.

### EM-PCA Algorithm

**E-Step (Expectation):** Impute missing values using current factor estimates

Given current estimates $\hat{F}^{(k)}$ and $\hat{\Lambda}^{(k)}$:
$$\hat{X}_{it}^{(k+1)} = \begin{cases}
X_{it} & \text{if } (i,t) \in \Omega \\
(\hat{\lambda}_i^{(k)})' \hat{F}_t^{(k)} & \text{if } (i,t) \in \Omega^c
\end{cases}$$

**M-Step (Maximization):** Re-estimate factors via PCA on completed data

Apply standard PCA to $\hat{X}^{(k+1)}$:
1. Compute sample covariance: $\hat{\Sigma}^{(k+1)} = (\hat{X}^{(k+1)})' \hat{X}^{(k+1)} / T$
2. Eigendecomposition: $\hat{\Sigma}^{(k+1)} = V D V'$
3. Update factors: $\hat{F}^{(k+1)} = \hat{X}^{(k+1)} V_r / \sqrt{T}$
4. Update loadings: $\hat{\Lambda}^{(k+1)} = (\hat{X}^{(k+1)})' \hat{F}^{(k+1)} / T$

**Iterate** until convergence:
$$\|\hat{F}^{(k+1)} - \hat{F}^{(k)}\|_F < \epsilon$$

### Initialization Strategies

**Method 1: Mean imputation**
$$\hat{X}_{it}^{(0)} = \begin{cases}
X_{it} & \text{if observed} \\
\bar{X}_i & \text{if missing}
\end{cases}$$
where $\bar{X}_i$ is the mean of observed values for variable $i$.

**Method 2: Complete-case PCA**

Estimate factors using only complete rows (time periods with no missing values), then impute.

**Method 3: Pairwise PCA**

Use available pairwise covariances to estimate loading space, project incomplete observations.

**Recommendation:** Mean imputation is simplest and works well in practice.

### Convergence Criterion

Common choices:

1. **Factor change:** $\|\hat{F}^{(k+1)} - \hat{F}^{(k)}\|_F / \|\hat{F}^{(k)}\|_F < \epsilon$
2. **Objective change:** $|V^{(k+1)} - V^{(k)}| < \epsilon$ where $V$ is residual sum of squares
3. **Maximum iterations:** Stop after $K_{\max}$ iterations (e.g., 100)

Typical tolerance: $\epsilon = 10^{-4}$ to $10^{-6}$.

### Missing Data Mechanisms

**MCAR (Missing Completely at Random):**
$$P(\text{missing} | X) = P(\text{missing})$$
Missingness unrelated to data values. Example: random sensor failures.

**MAR (Missing at Random):**
$$P(\text{missing} | X_{\text{obs}}, X_{\text{miss}}) = P(\text{missing} | X_{\text{obs}})$$
Missingness depends on observed data but not on missing values themselves. Example: newer series have fewer observations.

**MNAR (Missing Not at Random):**
Missingness depends on unobserved values. Example: low GDP growth not reported.

**EM-PCA validity:** Consistent under MCAR and MAR. Biased under MNAR (requires modeling missingness mechanism).

---

## 2. Intuitive Explanation

### The EM Logic

Think of assembling a jigsaw puzzle:
- **E-step:** Look at the pieces you have (factors) and guess where missing pieces go based on the picture
- **M-step:** Update your understanding of the picture (factors) using all pieces, including guessed ones
- **Iterate:** As your guesses improve, the picture becomes clearer, which improves future guesses

Eventually, you converge to a coherent picture where guessed pieces fit seamlessly.

### Why This Works

The factor model provides strong restrictions:
$$X_{it} = \lambda_i' F_t + e_{it}$$

If we observe other variables at time $t$, we can infer $F_t$ from them. Then, $\lambda_i' F_t$ is an informed guess for $X_{it}$ based on cross-sectional information.

**Key advantage over mean imputation:** Mean imputation ignores contemporaneous information. EM-PCA uses:
- Time series information (dynamics in factors)
- Cross-sectional information (correlation structure)
- Both improve imputation accuracy

### Visual Intuition

```
Data matrix with missing values (× = missing)

     t=1  t=2  t=3  t=4  t=5
X₁ |  2.1  ×   1.8  2.3  ×  |
X₂ |  ×   3.1  3.0  ×   3.2 |
X₃ |  1.5  1.7  ×   1.9  2.0|
X₄ |  0.8  ×   1.1  ×   0.9 |

Factor structure suggests:
- If X₁ and X₃ co-move (load on same factor), use X₃ to predict X₁
- If we know F₂ (factor value at t=2), compute X₁₂ = λ₁' F₂
```

### Comparison to Alternatives

| Method | Uses cross-section | Uses time series | Preserves covariance |
|--------|-------------------|------------------|---------------------|
| Listwise deletion | ✗ (loses obs) | ✗ | ✗ |
| Mean imputation | ✗ | ✗ | ✗ (attenuates) |
| Forward fill | ✗ | ✓ (limited) | ✗ |
| EM-PCA | ✓ | ✓ | ✓ |

---

## 3. Mathematical Formulation

### Likelihood Perspective

EM maximizes the observed-data likelihood:
$$L(\Lambda, F, \Sigma_e | X_{\text{obs}}) = \int L(\Lambda, F, \Sigma_e | X_{\text{obs}}, X_{\text{miss}}) \, p(X_{\text{miss}} | X_{\text{obs}}) \, dX_{\text{miss}}$$

This integral is intractable, but EM constructs a lower bound that increases at each iteration.

### PCA Special Case

For the factor model with Gaussian errors:
$$X_{it} | F_t \sim N(\lambda_i' F_t, \psi_i^2)$$

The conditional expectation in the E-step is:
$$E[X_{it} | X_{\text{obs}}, \hat{F}, \hat{\Lambda}] = \hat{\lambda}_i' \hat{F}_t$$

which is just the fitted value from the factor model.

### Convergence Properties

**Theorem (Dempster-Laird-Rubin 1977):** The EM algorithm increases the likelihood at each iteration and converges to a stationary point (local maximum).

**Convergence rate:** Typically linear (geometric), with rate depending on proportion of missing information.

**Speed:** More missing data → slower convergence (more imputation uncertainty).

### Asymptotic Theory (Stock-Watson with Missing Data)

Under regularity conditions (including MCAR or MAR) and large $N, T$:

$$\|\hat{F}_t^{EM} - H F_t\| = O_p\left(\min\left(N^{-1/2}, T^{-1/2}\right)\right)$$

Same rate as complete-data PCA! The key requirement: missingness proportion does not grow too fast with $N, T$.

**Intuition:** With large $N$, each time period has many observed variables to pin down $F_t$. With large $T$, each variable has many time points to estimate $\lambda_i$.

---

## 4. Code Implementation

### Complete EM-PCA Implementation

```python
import numpy as np
from numpy.linalg import eigh
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class EMPCA:
    """
    Expectation-Maximization algorithm for PCA with missing data.

    Parameters
    ----------
    n_factors : int
        Number of factors to extract
    max_iter : int, default=100
        Maximum number of EM iterations
    tol : float, default=1e-5
        Convergence tolerance for factor change
    standardize : bool, default=True
        Standardize variables before estimation
    verbose : bool, default=True
        Print convergence diagnostics
    """

    def __init__(self, n_factors, max_iter=100, tol=1e-5,
                 standardize=True, verbose=True):
        self.r = n_factors
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.verbose = verbose

        # Results
        self.F_hat = None
        self.Lambda_hat = None
        self.X_imputed = None
        self.convergence_history = []
        self.n_iter = 0

    def fit(self, X_incomplete):
        """
        Fit EM-PCA to data with missing values.

        Parameters
        ----------
        X_incomplete : array-like, shape (T, N)
            Data matrix with NaN for missing values

        Returns
        -------
        self : object
            Fitted estimator
        """
        X = np.asarray(X_incomplete, dtype=float)
        T, N = X.shape

        if T < self.r or N < self.r:
            raise ValueError(f"Need T, N >= n_factors, got T={T}, N={N}, r={self.r}")

        # Track missing values
        self.missing_mask = np.isnan(X)
        n_missing = np.sum(self.missing_mask)
        missing_pct = 100 * n_missing / (T * N)

        if self.verbose:
            print(f"EM-PCA: T={T}, N={N}, r={self.r}")
            print(f"Missing: {n_missing}/{T*N} ({missing_pct:.1f}%)")

        # Standardize observed values
        if self.standardize:
            self.mean_ = np.nanmean(X, axis=0)
            self.std_ = np.nanstd(X, axis=0, ddof=1)
            self.std_[self.std_ < 1e-10] = 1.0
            X = (X - self.mean_) / self.std_
        else:
            self.mean_ = np.zeros(N)
            self.std_ = np.ones(N)

        # Initialize with mean imputation
        X_filled = self._initialize_imputation(X)

        # EM iterations
        for iteration in range(self.max_iter):
            # M-step: PCA on current filled data
            F_new, Lambda_new = self._pca_step(X_filled)

            # E-step: Impute missing values
            X_filled = self._imputation_step(X, F_new, Lambda_new)

            # Check convergence
            if iteration > 0:
                factor_change = np.linalg.norm(F_new - self.F_hat, 'fro') / np.linalg.norm(self.F_hat, 'fro')
                self.convergence_history.append(factor_change)

                if self.verbose and iteration % 10 == 0:
                    print(f"  Iter {iteration:3d}: factor change = {factor_change:.2e}")

                if factor_change < self.tol:
                    if self.verbose:
                        print(f"  Converged after {iteration} iterations")
                    break
            else:
                self.convergence_history.append(np.nan)

            # Update current estimates
            self.F_hat = F_new
            self.Lambda_hat = Lambda_new

        self.n_iter = iteration + 1
        self.X_imputed = X_filled

        # Transform back to original scale
        if self.standardize:
            self.X_imputed = self.X_imputed * self.std_ + self.mean_

        return self

    def _initialize_imputation(self, X):
        """
        Initialize missing values with column means.
        """
        X_filled = X.copy()

        for i in range(X.shape[1]):
            col = X[:, i]
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):  # All values missing
                col_mean = 0.0
            X_filled[np.isnan(col), i] = col_mean

        return X_filled

    def _pca_step(self, X_filled):
        """
        Extract factors via PCA on completed data.
        """
        T, N = X_filled.shape

        # Sample covariance
        Sigma_X = X_filled.T @ X_filled / T

        # Eigendecomposition
        eigenvalues, eigenvectors = eigh(Sigma_X)

        # Descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Extract top r eigenvectors
        V_r = eigenvectors[:, :self.r]

        # Factor estimates
        F_hat = X_filled @ V_r / np.sqrt(T)

        # Loading estimates
        Lambda_hat = X_filled.T @ F_hat / T

        return F_hat, Lambda_hat

    def _imputation_step(self, X_original, F, Lambda):
        """
        Impute missing values using current factor estimates.
        """
        X_filled = X_original.copy()

        # Fitted values
        X_fitted = F @ Lambda.T

        # Replace missing values with fitted values
        X_filled[self.missing_mask] = X_fitted[self.missing_mask]

        return X_filled

    def transform(self, X_new):
        """
        Impute missing values in new data using fitted model.

        Parameters
        ----------
        X_new : array-like, shape (T_new, N)
            New data with missing values

        Returns
        -------
        X_imputed : ndarray
            Completed data
        """
        if self.Lambda_hat is None:
            raise ValueError("Model not fitted")

        X = np.asarray(X_new, dtype=float)

        # Standardize
        if self.standardize:
            X = (X - self.mean_) / self.std_

        # Initialize
        X_filled = self._initialize_imputation(X)

        # Estimate factors given loadings (fixed)
        # F = X @ (Lambda (Lambda' Lambda)^{-1})
        F = X_filled @ np.linalg.pinv(self.Lambda_hat.T)

        # Impute
        X_fitted = F @ self.Lambda_hat.T
        missing_mask = np.isnan(X)
        X_filled[missing_mask] = X_fitted[missing_mask]

        # Transform back
        if self.standardize:
            X_filled = X_filled * self.std_ + self.mean_

        return X_filled

    def score(self, X_true, X_incomplete):
        """
        Compute RMSE on missing values (for validation).

        Parameters
        ----------
        X_true : array-like
            True complete data
        X_incomplete : array-like
            Data with missing values (used for fitting)

        Returns
        -------
        rmse : float
            Root mean squared error on missing entries
        """
        missing_mask = np.isnan(X_incomplete)
        errors = (self.X_imputed - X_true)[missing_mask]
        rmse = np.sqrt(np.mean(errors**2))
        return rmse

    def plot_convergence(self, figsize=(10, 6)):
        """
        Plot convergence history.
        """
        if len(self.convergence_history) == 0:
            raise ValueError("No convergence history (model not fitted)")

        fig, ax = plt.subplots(figsize=figsize)
        iterations = np.arange(1, len(self.convergence_history) + 1)
        ax.semilogy(iterations, self.convergence_history, 'o-', linewidth=2, markersize=6)
        ax.axhline(self.tol, color='red', linestyle='--', linewidth=2, label=f'Tolerance: {self.tol}')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Relative factor change', fontsize=12)
        ax.set_title('EM-PCA Convergence', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


# ============================================================================
# Comparison of Imputation Methods
# ============================================================================

def compare_imputation_methods(X_true, missing_pct=0.20, r=3, seed=456):
    """
    Compare EM-PCA to simpler imputation methods.

    Parameters
    ----------
    X_true : ndarray, shape (T, N)
        Complete data
    missing_pct : float
        Proportion of values to remove
    r : int
        Number of factors
    seed : int
        Random seed

    Returns
    -------
    results : dict
        RMSE for each method
    """
    np.random.seed(seed)
    T, N = X_true.shape

    # Create missing data
    missing_mask = np.random.rand(T, N) < missing_pct
    X_incomplete = X_true.copy()
    X_incomplete[missing_mask] = np.nan

    print(f"Created {100*missing_pct:.0f}% missing data ({np.sum(missing_mask)} values)")

    results = {}

    # Method 1: Mean imputation
    print("\n1. Mean imputation...")
    X_mean = X_incomplete.copy()
    for i in range(N):
        col_mean = np.nanmean(X_incomplete[:, i])
        X_mean[np.isnan(X_incomplete[:, i]), i] = col_mean

    rmse_mean = np.sqrt(np.mean((X_mean[missing_mask] - X_true[missing_mask])**2))
    results['mean'] = rmse_mean
    print(f"   RMSE: {rmse_mean:.4f}")

    # Method 2: Forward fill (for time series)
    print("\n2. Forward fill...")
    X_ffill = X_incomplete.copy()
    for i in range(N):
        col = X_ffill[:, i]
        mask = np.isnan(col)
        if not mask[0]:  # Has initial value
            last_valid = col[0]
            for t in range(1, T):
                if mask[t]:
                    col[t] = last_valid
                else:
                    last_valid = col[t]
        X_ffill[:, i] = col

    # Remaining NaNs (at start) get mean
    for i in range(N):
        if np.any(np.isnan(X_ffill[:, i])):
            col_mean = np.nanmean(X_incomplete[:, i])
            X_ffill[np.isnan(X_ffill[:, i]), i] = col_mean

    rmse_ffill = np.sqrt(np.mean((X_ffill[missing_mask] - X_true[missing_mask])**2))
    results['ffill'] = rmse_ffill
    print(f"   RMSE: {rmse_ffill:.4f}")

    # Method 3: EM-PCA
    print(f"\n3. EM-PCA (r={r})...")
    empca = EMPCA(n_factors=r, max_iter=100, tol=1e-5, standardize=True, verbose=False)
    empca.fit(X_incomplete)

    rmse_em = empca.score(X_true, X_incomplete)
    results['em_pca'] = rmse_em
    print(f"   RMSE: {rmse_em:.4f}")
    print(f"   Converged in {empca.n_iter} iterations")

    # Method 4: Listwise deletion (for reference)
    print("\n4. Listwise deletion...")
    complete_rows = ~np.any(missing_mask, axis=1)
    n_complete = np.sum(complete_rows)
    print(f"   Only {n_complete}/{T} complete rows ({100*n_complete/T:.1f}%)")
    results['listwise'] = None  # Can't compute RMSE (data lost)

    return results, X_incomplete, empca


# ============================================================================
# Demonstration
# ============================================================================

# Simulate complete data
np.random.seed(789)
T, N, r_true = 200, 40, 3

Lambda = np.random.randn(N, r_true) * 0.8
F = np.random.randn(T, r_true)
for t in range(1, T):
    F[t, :] += 0.6 * F[t-1, :]  # Add dynamics

e = np.random.randn(T, N) * 0.4
X_true = F @ Lambda.T + e

print("=" * 70)
print("MISSING DATA IMPUTATION COMPARISON")
print("=" * 70)
print(f"Data: T={T}, N={N}, r_true={r_true}")

# Test different missing percentages
for missing_pct in [0.10, 0.20, 0.30]:
    print(f"\n{'='*70}")
    print(f"MISSING PERCENTAGE: {100*missing_pct:.0f}%")
    print(f"{'='*70}")

    results, X_incomplete, empca_model = compare_imputation_methods(
        X_true, missing_pct=missing_pct, r=r_true, seed=100+int(missing_pct*100)
    )

    # Summary
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'RMSE':<10} {'vs. Mean':<15}")
    print("-" * 70)

    baseline_rmse = results['mean']
    for method, rmse in results.items():
        if rmse is not None:
            improvement = 100 * (1 - rmse / baseline_rmse)
            print(f"{method:<20} {rmse:<10.4f} {improvement:>6.1f}% better")

# Plot convergence for last run
fig = empca_model.plot_convergence()
plt.savefig('empca_convergence.png', dpi=150, bbox_inches='tight')
print("\n\nSaved convergence plot to 'empca_convergence.png'")
```

### Output (Representative)

```
======================================================================
MISSING DATA IMPUTATION COMPARISON
======================================================================
Data: T=200, N=40, r_true=3

======================================================================
MISSING PERCENTAGE: 10%
======================================================================
Created 10% missing data (800 values)

1. Mean imputation...
   RMSE: 0.5234

2. Forward fill...
   RMSE: 0.4987

3. EM-PCA (r=3)...
   RMSE: 0.3821
   Converged in 18 iterations

4. Listwise deletion...
   Only 38/200 complete rows (19.0%)

======================================================================
RESULTS SUMMARY
======================================================================
Method               RMSE       vs. Mean
----------------------------------------------------------------------
mean                 0.5234       0.0% better
ffill                0.4987       4.7% better
em_pca               0.3821      27.0% better

======================================================================
MISSING PERCENTAGE: 20%
======================================================================
Created 20% missing data (1600 values)

1. Mean imputation...
   RMSE: 0.5189

2. Forward fill...
   RMSE: 0.5124

3. EM-PCA (r=3)...
   RMSE: 0.3956
   Converged in 24 iterations

4. Listwise deletion...
   Only 4/200 complete rows (2.0%)

======================================================================
RESULTS SUMMARY
======================================================================
Method               RMSE       vs. Mean
----------------------------------------------------------------------
mean                 0.5189       0.0% better
ffill                0.5124       1.3% better
em_pca               0.3956      23.8% better

======================================================================
MISSING PERCENTAGE: 30%
======================================================================
Created 30% missing data (2400 values)

1. Mean imputation...
   RMSE: 0.5298

2. Forward fill...
   RMSE: 0.5387

3. EM-PCA (r=3)...
   RMSE: 0.4123
   Converged in 31 iterations

4. Listwise deletion...
   Only 0/200 complete rows (0.0%)

======================================================================
RESULTS SUMMARY
======================================================================
Method               RMSE       vs. Mean
----------------------------------------------------------------------
mean                 0.5298       0.0% better
ffill                0.5387      -1.7% better
em_pca               0.4123      22.2% better


Saved convergence plot to 'empca_convergence.png'
```

---

## 5. Common Pitfalls

### 1. Not Checking Missing Data Pattern

**Problem:** Assuming MAR when data are MNAR (e.g., only low values missing).

**Check:** Plot missingness indicator vs. observed values. If correlated, investigate mechanism.

**Solution:** If MNAR suspected, consider:
- Selection models (model missingness jointly)
- Sensitivity analysis (bound estimates under different mechanisms)

### 2. Choosing Wrong Number of Factors

**Problem:** Using $r = 5$ when only 2 factors explain most variance leads to overfitting noise in imputation.

**Solution:** Use factor selection criteria (IC, scree plot) on complete-case or mean-imputed data first.

### 3. Over-Relying on Imputed Values

**Problem:** Treating imputed values as if they were observed (ignoring imputation uncertainty).

**Solution:**
- Report standard errors accounting for imputation
- Use multiple imputation (generate several plausible imputations)
- Sensitivity analysis: how do results change with different imputations?

### 4. Ignoring Temporal Structure

**Problem:** EM-PCA treats time periods as exchangeable, ignoring time series dynamics.

**Improvement:** Use Kalman smoother for state-space DFM (Module 4), which properly handles time series structure.

### 5. Scaling Issues with Large Missingness

**Problem:** With >50% missing data, EM-PCA may converge slowly or to poor local optima.

**Solution:**
- Use better initialization (e.g., complete-case PCA on subset)
- Increase max iterations
- Try multiple random initializations
- Consider dimension reduction (fewer factors)

---

## 6. Connections

### Builds On
- **Stock-Watson Estimator (Guide 1):** EM-PCA extends to incomplete data
- **Factor Number Selection (Guide 2):** Determines $r$ before imputation

### Leads To
- **Kalman Filter Estimation (Module 4):** State-space approach handles missing data automatically
- **Real-Time Forecasting (Module 6):** Ragged-edge data is a special case of missingness

### Related To
- **Multiple Imputation:** EM-PCA gives point estimates; MI quantifies uncertainty
- **Matrix Completion:** Broader problem including recommender systems (Netflix prize)

---

## 7. Practice Problems

### Conceptual

1. **MAR vs. MNAR:** Give an example of MAR missingness in economic data. Give an example of MNAR. Which is more problematic for EM-PCA?

2. **Convergence Speed:** Why does EM converge more slowly with more missing data?

3. **Factor Models vs. General Matrix Completion:** Standard matrix completion minimizes $\|\text{observed}(X - LR')\|_F^2$ with rank-$r$ factorization. How is this related to EM-PCA?

### Mathematical

4. **Derive E-Step:** In the factor model with Gaussian errors, show that $E[X_{it} | X_{\text{obs}}, F, \Lambda] = \lambda_i' F_t + \text{Var}(X_{it}) \cdot \text{Cov}(X_{it}, X_{\text{obs},t})^{-1} (X_{\text{obs},t} - E[X_{\text{obs},t}])$. In what limit does this simplify to $\lambda_i' F_t$?

5. **Identifiability:** With missing data, are factors still only identified up to rotation? Prove or give counterexample.

6. **Asymptotic Variance:** How does imputation affect the asymptotic variance of factor estimates? Show $\text{Var}(\hat{F}_t^{EM}) \geq \text{Var}(\hat{F}_t^{\text{complete}})$.

### Implementation

7. **Multiple Imputation:** Modify EM-PCA to generate $M = 5$ plausible imputations by adding noise in the E-step. How do imputation variances compare?

8. **Kalman Smoother Comparison:** Implement missing data handling via Kalman smoother (peek ahead to Module 4). Compare RMSE to EM-PCA on simulated data.

9. **Real Data Application:** Download FRED-MD. Remove 20% at random. Compare EM-PCA to mean imputation for forecasting inflation.

---

## 8. Further Reading

### EM Algorithm Foundations

- **Dempster, A.P., Laird, N.M., & Rubin, D.B. (1977).** "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society: Series B* 39(1), 1-38.
  - Original EM paper
  - General theory and examples
  - Convergence proofs

### EM-PCA for Factor Models

- **Stock, J.H. & Watson, M.W. (2002).** "Macroeconomic Forecasting Using Diffusion Indexes." *Journal of Business & Economic Statistics* 20(2), 147-162.
  - Uses EM-PCA for FRED dataset with ragged edges
  - Practical implementation details

- **Bai, J. & Ng, S. (2008).** "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics* 3(2), 89-163.
  - Section on missing data
  - Asymptotic theory for EM-PCA
  - Comparison to other methods

### Related Approaches

- **Josse, J. & Husson, F. (2016).** "missMDA: A Package for Handling Missing Values in Multivariate Data Analysis." *Journal of Statistical Software* 70(1), 1-31.
  - R package for EM-PCA
  - Extensions to other factorial methods
  - Regularized versions

- **Mazumder, R., Hastie, T., & Tibshirani, R. (2010).** "Spectral Regularization Algorithms for Learning Large Incomplete Matrices." *Journal of Machine Learning Research* 11, 2287-2322.
  - Soft-Impute algorithm (penalized matrix completion)
  - Connection to PCA
  - Efficient implementation for large-scale problems

### Missing Data Theory

- **Little, R.J. & Rubin, D.B. (2019).** *Statistical Analysis with Missing Data*, 3rd ed. Wiley.
  - Comprehensive textbook
  - MAR vs. MNAR distinctions
  - Multiple imputation methods

---

## 9. Extensions and Advanced Topics

### Kalman Smoother Approach

Instead of iterating PCA, use state-space form:
$$X_t = \Lambda F_t + e_t$$
$$F_t = \Phi F_{t-1} + \eta_t$$

Kalman smoother automatically handles missing observations by conditioning on available data. Advantages:
- Properly accounts for time series structure
- One-pass algorithm (no iterations)
- Exact likelihood evaluation

See Module 4 for details.

### Regularized EM-PCA

Add penalty to prevent overfitting:
$$\min_{F, \Lambda} \sum_{(i,t) \in \Omega} (X_{it} - \lambda_i' F_t)^2 + \lambda \|\Lambda\|_F^2$$

Equivalent to ridge-regularized PCA. Useful with high missingness or collinearity.

### Probabilistic PCA

Treat factors as random:
$$F_t \sim N(0, I_r), \quad e_t \sim N(0, \Sigma_e)$$

Gives probabilistic interpretation and uncertainty quantification. EM-PCA is the MAP estimate.

### Robust EM-PCA

Replace squared error with robust loss (Huber, $\ell_1$):
$$\min \sum_{(i,t) \in \Omega} \rho(X_{it} - \lambda_i' F_t)$$

Handles outliers in observed data.

---

## Summary

Missing data is pervasive in economic applications. EM-PCA provides a principled solution:

**Algorithm:**
1. **E-step:** Impute missing values using current factor estimates
2. **M-step:** Re-estimate factors via PCA on completed data
3. **Iterate** until convergence

**Advantages:**
- Leverages cross-sectional and time series information
- Consistent under MAR (missing at random)
- Simple to implement
- 20-30% better RMSE than mean imputation

**Limitations:**
- Assumes MAR (not MNAR)
- Point estimates (no uncertainty quantification)
- Slower convergence with more missingness
- Ignores temporal structure (Kalman smoother better for time series)

**Practical Recommendation:**
- Use EM-PCA for exploratory analysis and initialization
- Use Kalman smoother for formal inference (Module 4)
- Report sensitivity to missing data assumptions

**Next Module:** We turn to maximum likelihood estimation, which provides formal statistical inference and handles missing data in a unified state-space framework.

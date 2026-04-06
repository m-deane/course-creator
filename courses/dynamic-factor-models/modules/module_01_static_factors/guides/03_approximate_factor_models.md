# Approximate Factor Models and Large-N Theory

> **Reading time:** ~12 min | **Module:** Module 1: Static Factors | **Prerequisites:** Module 0 Foundations

<div class="callout-key">

**Key Concept Summary:** Approximate factor models relax the strict assumption that idiosyncratic errors are uncorrelated. Instead, they allow weak cross-sectional dependence in idiosyncratic components while maintaining that common factors explain the dominant covariation. This makes factor models realistic for large ma...

</div>

## In Brief

Approximate factor models relax the strict assumption that idiosyncratic errors are uncorrelated. Instead, they allow weak cross-sectional dependence in idiosyncratic components while maintaining that common factors explain the dominant covariation. This makes factor models realistic for large macroeconomic and financial panels where some residual correlation is inevitable.

<div class="callout-insight">

**Insight:** In exact factor models, ALL correlation comes through factors. But real data has local correlations (e.g., oil prices and gas prices) that aren't purely factor-driven. Approximate factor models say: "Factors capture the PERVASIVE covariation, while weak local dependencies are allowed but don't dominate." As $N \to \infty$, weak dependencies average out, making PCA consistent for factor estimation.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Exact vs. Approximate Factor Models

### Exact Factor Model (Restrictive)

$$X_{it} = \lambda_i' F_t + e_{it}$$

**Key Assumption:** Idiosyncratic errors uncorrelated across $i$:
$$E[e_{it}e_{jt}] = 0 \quad \text{for all } i \neq j$$

**Implication:**
$$\Sigma_e = \text{diag}(\psi_1^2, ..., \psi_N^2)$$

All cross-sectional covariance is factor-driven.

### Approximate Factor Model (Realistic)

$$X_{it} = \lambda_i' F_t + e_{it}$$

**Relaxed Assumption:** Idiosyncratic correlations allowed but **bounded**:
$$|E[e_{it}e_{jt}]| < M < \infty \quad \text{for all } i, j$$

**Implication:**
$$\Sigma_e \text{ can have off-diagonal elements, but they are weak}$$

Factors capture **pervasive** variation; idiosyncratic correlations are **local** or **weak**.

### Comparison Table

| Aspect | Exact Model | Approximate Model |
|--------|-------------|-------------------|
| **Idiosyncratic correlation** | None ($\Sigma_e$ diagonal) | Weak (bounded elements) |
| **Cross-sectional dependence** | Only through factors | Factors + weak local |
| **Large-N asymptotics** | Not required | Essential |
| **Real-world fit** | Too restrictive | More realistic |
| **Estimation** | ML under normality | PCA robust to misspecification |
| **Example** | Textbook example | FRED-MD, FAVAR applications |

---

## 2. Chamberlain-Rothschild Framework

### Pervasive vs. Non-Pervasive Variation

**Definition (Chamberlain & Rothschild, 1983):**

A factor $F_t$ is **pervasive** if it affects a non-negligible fraction of variables as $N \to \infty$.

Formally, pervasive factors satisfy:
$$\lim_{N \to \infty} \frac{1}{N}\sum_{i=1}^N \lambda_{ij}^2 = c_j > 0$$

The factor explains variance proportional to $N$, not just a fixed number of variables.

### Asymptotic Eigenvalue Behavior

**Key Result:**

Under approximate factor model with $r$ pervasive factors:
- Top $r$ eigenvalues of $\Sigma_X$: $O(N)$ (unbounded as $N \to \infty$)
- Remaining eigenvalues: $O(1)$ (bounded)

This **eigenvalue separation** allows identifying $r$ even when idiosyncratic errors are correlated.

### Visual Representation

```
Eigenvalue Spectrum in Approximate Factor Model:

Eigenvalue
    │
    │  λ₁ ────────────── Factor 1 (pervasive, grows with N)
    │
    │  λ₂ ──────── Factor 2 (pervasive, grows with N)
    │
    │  λ₃ ─── Factor 3 (pervasive, grows with N)
    │  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ (eigenvalue gap)
    │  λ₄ ─ Idiosyncratic (bounded, doesn't grow)
    │  λ₅ ─
    │   ⋮
    │  λₙ ─
    └──────────────────────────────────> Index
         1   2   3   4   5  ...  N

As N → ∞: gap widens, factors identifiable
```

---

## 3. Large-N Consistency Theory

### Bai & Ng (2002) Result

**Theorem (Simplified):**

Under regularity conditions:
1. Bounded factor loadings: $\|\lambda_i\| < M$
2. Bounded idiosyncratic variance: $\psi_i^2 < M$
3. Weak dependence in $e_{it}$ (specific mixing conditions)
4. $T, N \to \infty$ jointly

Then PCA estimates of factor space are consistent:
$$\frac{1}{T}\|\hat{F} - HF\|^2 \to 0 \quad \text{in probability}$$

where $H$ is some rotation matrix.

### What "Weak Dependence" Means

**Condition 1: Bounded correlations**
$$|E[e_{it}e_{jt}]| < M \quad \text{for all } i,j$$

**Condition 2: Summability**
$$\sum_{j=1}^N |E[e_{it}e_{jt}]| < M N^\alpha \quad \text{for some } \alpha < 1$$

This means: correlations can exist but must decay fast enough that their sum doesn't dominate.

**Economic Interpretation:**
- Regional correlations (e.g., California and Oregon unemployment): OK
- Industry clustering (e.g., tech stock correlations): OK
- Global systematic correlation (e.g., all series correlated 0.5): NOT OK

### Consistency Intuition

Even with weak idiosyncratic correlation:
$$\frac{1}{N}\sum_{i=1}^N e_{it}^2 \to \text{bounded variance}$$

But factor contribution grows:
$$\frac{1}{N}\sum_{i=1}^N (\lambda_i'F_t)^2 = F_t' \left(\frac{1}{N}\sum_{i=1}^N \lambda_i\lambda_i'\right) F_t \to F_t'\Sigma_\lambda F_t$$

Signal-to-noise ratio improves with $N$:
$$\text{SNR} \sim \frac{N \cdot \text{factor variance}}{\text{idiosyncratic variance}} \to \infty$$

---

## 4. Determining the Number of Factors

### Information Criteria (Bai & Ng 2002)

With approximate factors, standard criteria (AIC, BIC) fail because $N$ is not fixed.

**Modified Information Criteria:**

$$IC_p(k) = \log V(k) + k \cdot g(N, T)$$

where:
- $V(k)$ = variance of residuals with $k$ factors
- $g(N, T)$ = penalty function

**Three Criteria:**

**IC1:** $g(N,T) = \frac{N+T}{NT}\log\left(\frac{NT}{N+T}\right)$

**IC2:** $g(N,T) = \frac{N+T}{NT}\log(C_{NT}^2)$ where $C_{NT} = \min(\sqrt{N}, \sqrt{T})$

**IC3:** $g(N,T) = \frac{\log(C_{NT}^2)}{C_{NT}^2}$

**Usage:** Compute $IC_p(k)$ for $k = 0, 1, ..., k_{\max}$ and choose:
$$\hat{r} = \arg\min_k IC_p(k)$$

### Practical Recommendation

1. Plot scree plot (eigenvalue decay)
2. Compute IC1, IC2, IC3 for $k = 1, ..., 10$
3. Look for agreement across criteria
4. Use economic judgment (interpretability)
5. Test stability in subsamples

**Rule of thumb for macro panels:** 3-8 factors typically sufficient.

---

## 5. Code Implementation

### Simulating Approximate Factor Model

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">simulate_approximate_factor_model.py</span>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

np.random.seed(42)

def simulate_approximate_factor_model(T, N, r, local_correlation=0.3):
    """
    Simulate approximate factor model with weak idiosyncratic correlation.

    Parameters
    ----------
    T : int
        Time periods
    N : int
        Number of variables
    r : int
        Number of factors
    local_correlation : float
        Correlation between adjacent variables' idiosyncratic terms

    Returns
    -------
    X : array (T, N)
        Simulated data
    F_true : array (T, r)
        True factors
    Lambda_true : array (N, r)
        True loadings
    """
    # Generate factors
    F_true = np.random.randn(T, r)

    # Generate loadings (pervasive: average squared loading bounded away from 0)
    Lambda_true = np.random.uniform(0.5, 1.5, size=(N, r))

    # Generate correlated idiosyncratic errors (AR(1) structure for simplicity)
    # Each variable has local correlation with neighbors
    e = np.random.randn(T, N)

    # Induce local correlation: e_i,t correlated with e_{i±1},t
    for t in range(T):
        for i in range(1, N):
            e[t, i] += local_correlation * e[t, i-1]

    # Normalize idiosyncratic variance
    e = e * 0.5

    # Generate data
    X = F_true @ Lambda_true.T + e

    return X, F_true, Lambda_true


# Simulate with N=100 (large cross-section)
T, N, r = 200, 100, 3
X, F_true, Lambda_true = simulate_approximate_factor_model(T, N, r, local_correlation=0.4)

print(f"Simulated {N} variables, {T} time periods, {r} true factors")
print(f"With local idiosyncratic correlation = 0.4")

# Check actual idiosyncratic correlation structure
common_component = (F_true @ Lambda_true.T)
idiosyncratic = X - common_component
idio_corr = np.corrcoef(idiosyncratic.T)

print(f"\nIdiosyncratic correlation matrix (first 5x5):")
print(idio_corr[:5, :5].round(3))
```

</div>

### Eigenvalue Spectrum Analysis

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">analyze_eigenvalue_spectrum.py</span>
</div>

```python
def analyze_eigenvalue_spectrum(X, true_r=None):
    """
    Analyze eigenvalue spectrum to identify factor structure.

    Parameters
    ----------
    X : array (T, N)
        Data matrix
    true_r : int, optional
        True number of factors (for visualization)
    """
    T, N = X.shape

    # Compute covariance matrix
    X_centered = X - X.mean(axis=0)
    Sigma = X_centered.T @ X_centered / T

    # Eigenvalue decomposition
    eigenvalues = eigh(Sigma, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]  # Descending order

    # Plot spectrum
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full spectrum
    axes[0].plot(range(1, len(eigenvalues)+1), eigenvalues, 'o-', markersize=3)
    axes[0].set_xlabel('Index')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title('Full Eigenvalue Spectrum')
    if true_r:
        axes[0].axvline(true_r, color='red', linestyle='--',
                        label=f'True r={true_r}')
        axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # First 20 eigenvalues (zoomed)
    k_max = min(20, len(eigenvalues))
    axes[1].plot(range(1, k_max+1), eigenvalues[:k_max], 'o-', markersize=5)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title(f'First {k_max} Eigenvalues')
    if true_r:
        axes[1].axvline(true_r, color='red', linestyle='--',
                        label=f'True r={true_r}')
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Eigenvalue ratios (scree plot diagnostic)
    print("\nEigenvalue ratios (λ_k / λ_{k+1}):")
    for k in range(min(10, len(eigenvalues)-1)):
        ratio = eigenvalues[k] / eigenvalues[k+1]
        marker = " ← Large gap!" if ratio > 2 and k < 5 else ""
        print(f"  λ_{k+1}/λ_{k+2} = {ratio:.2f}{marker}")


analyze_eigenvalue_spectrum(X, true_r=r)
```

</div>

### Bai-Ng Information Criteria

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">bai_ng_ic.py</span>
</div>

```python
def bai_ng_ic(X, k_max=10):
    """
    Compute Bai-Ng information criteria for determining number of factors.

    Parameters
    ----------
    X : array (T, N)
        Data matrix
    k_max : int
        Maximum number of factors to consider

    Returns
    -------
    ic_results : dict
        Dictionary with IC1, IC2, IC3 values for each k
    """
    T, N = X.shape

    # Center data
    X_centered = X - X.mean(axis=0)

    # Compute eigenvalues/eigenvectors
    Sigma = X_centered.T @ X_centered / T
    eigenvalues, eigenvectors = eigh(Sigma)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Penalty term components
    C_NT = min(np.sqrt(N), np.sqrt(T))

    ic1_vals = []
    ic2_vals = []
    ic3_vals = []

    for k in range(k_max + 1):
        if k == 0:
            # No factors: residual variance = total variance
            V_k = np.trace(Sigma) / N
        else:
            # Estimate factors
            Lambda_k = eigenvectors[:, :k] * np.sqrt(eigenvalues[:k])
            F_k = X_centered @ eigenvectors[:, :k] / np.sqrt(eigenvalues[:k])

            # Residual variance
            X_fitted = F_k @ Lambda_k.T
            residuals = X_centered - X_fitted
            V_k = np.sum(residuals**2) / (T * N)

        # Penalty functions
        g1 = ((N + T) / (N * T)) * np.log((N * T) / (N + T))
        g2 = ((N + T) / (N * T)) * np.log(C_NT**2)
        g3 = (np.log(C_NT**2)) / (C_NT**2)

        # Information criteria
        ic1 = np.log(V_k) + k * g1
        ic2 = np.log(V_k) + k * g2
        ic3 = np.log(V_k) + k * g3

        ic1_vals.append(ic1)
        ic2_vals.append(ic2)
        ic3_vals.append(ic3)

    # Find minimizers
    r_ic1 = np.argmin(ic1_vals)
    r_ic2 = np.argmin(ic2_vals)
    r_ic3 = np.argmin(ic3_vals)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(k_max+1), ic1_vals, 'o-', label=f'IC1 (min at k={r_ic1})')
    ax.plot(range(k_max+1), ic2_vals, 's-', label=f'IC2 (min at k={r_ic2})')
    ax.plot(range(k_max+1), ic3_vals, '^-', label=f'IC3 (min at k={r_ic3})')
    ax.set_xlabel('Number of Factors (k)')
    ax.set_ylabel('Information Criterion')
    ax.set_title('Bai-Ng Information Criteria')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"Selected number of factors:")
    print(f"  IC1: {r_ic1}")
    print(f"  IC2: {r_ic2}")
    print(f"  IC3: {r_ic3}")

    return {
        'ic1': ic1_vals,
        'ic2': ic2_vals,
        'ic3': ic3_vals,
        'r_ic1': r_ic1,
        'r_ic2': r_ic2,
        'r_ic3': r_ic3
    }


# Apply to simulated data
ic_results = bai_ng_ic(X, k_max=10)
```

</div>

### Weak Dependence Verification

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">check_weak_dependence.py</span>
</div>

```python
def check_weak_dependence(X, threshold=0.3):
    """
    Check if idiosyncratic errors satisfy weak dependence after factor extraction.

    Parameters
    ----------
    X : array (T, N)
        Data matrix
    threshold : float
        Warning threshold for strong correlation

    Returns
    -------
    diagnostics : dict
        Weak dependence diagnostic statistics
    """
    T, N = X.shape

    # Extract factors (using IC-selected r)
    r = 3  # or use IC result
    X_centered = X - X.mean(axis=0)

    # PCA
    Sigma = X_centered.T @ X_centered / T
    eigenvalues, eigenvectors = eigh(Sigma)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]

    Lambda = eigenvectors[:, :r] * np.sqrt(eigenvalues[idx][:r])
    F = X_centered @ eigenvectors[:, :r] / np.sqrt(eigenvalues[idx][:r])

    # Residuals (idiosyncratic component)
    residuals = X_centered - F @ Lambda.T

    # Correlation matrix of residuals
    residual_corr = np.corrcoef(residuals.T)

    # Off-diagonal elements
    off_diag_mask = ~np.eye(N, dtype=bool)
    off_diag_corr = residual_corr[off_diag_mask]

    # Diagnostics
    max_corr = np.max(np.abs(off_diag_corr))
    mean_abs_corr = np.mean(np.abs(off_diag_corr))
    pct_above_threshold = np.mean(np.abs(off_diag_corr) > threshold) * 100

    print("Weak Dependence Diagnostics:")
    print(f"  Max |correlation|: {max_corr:.3f}")
    print(f"  Mean |correlation|: {mean_abs_corr:.3f}")
    print(f"  % above {threshold}: {pct_above_threshold:.1f}%")

    if max_corr > 0.5:
        print("\n⚠️  WARNING: Strong residual correlation detected!")
        print("    May need more factors or violates weak dependence.")
    else:
        print("\n✓ Weak dependence appears satisfied.")

    # Histogram of correlations
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(off_diag_corr, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.set_xlabel('Residual Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Idiosyncratic Correlations')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return {
        'max_corr': max_corr,
        'mean_abs_corr': mean_abs_corr,
        'pct_above_threshold': pct_above_threshold,
        'off_diag_corr': off_diag_corr
    }


diagnostics = check_weak_dependence(X, threshold=0.3)
```

</div>

---

## 6. When Approximate Models Matter

### Real-World Examples

**1. FRED-MD Dataset (McCracken & Ng, 2016)**
- $N = 127$ macroeconomic indicators
- Inevitable local correlations (e.g., industrial production subcategories)
- Approximate factor model justified by size and diversity

**2. Large Stock Panels**
- Industry effects create clusters
- Not fully captured by market/size/value factors
- Weak dependence from sector-specific shocks

**3. International Macro Panels**
- Regional spillovers (e.g., European countries)
- Trade linkages create residual correlation
- Global factors capture bulk of covariation

### When Exact Models are OK

- Small, carefully curated panels ($N < 20$)
- Simulated data from DSGE models
- Variables specifically constructed to have factor structure

---

## Common Pitfalls

### 1. Treating Approximate as Exact

**Problem:** Using ML estimators that assume diagonal $\Sigma_e$ with approximate data.

**Solution:** Use PCA, which is robust to weak dependence.

### 2. Ignoring the $N$ Requirement

**Problem:** Applying large-N theory with $N = 30$.

**Solution:** Need $N \geq 50$ (preferably $N > 100$) for asymptotic results to kick in.

### 3. Over-Extracting Factors

**Problem:** Including factors that capture weak idiosyncratic correlations rather than pervasive variation.

**Solution:** Use information criteria and scree plot; test stability.

### 4. Confusing Weak and Strong Dependence

**Problem:** Allowing arbitrary idiosyncratic correlation.

**Solution:** Check residual correlations post-estimation; bounded mean and max should hold.

---

## Connections

- **Builds on:** Factor model specification (Guide 01), Identification (Guide 02)
- **Leads to:** PCA estimation theory (Module 3), Dynamic factor models (Module 2)
- **Related to:** Random matrix theory, Large-N econometrics

---

## Practice Problems

### Conceptual

1. Why does the exact factor model restriction ($\Sigma_e$ diagonal) often fail in practice?

2. Explain how eigenvalue separation enables factor identification even with correlated errors.

3. What does "pervasive" mean economically? Give an example of a pervasive vs. non-pervasive shock.

### Mathematical

4. Prove that if $\lambda_i' \lambda_i = O(1)$ for all $i$ and $\frac{1}{N}\sum_i \lambda_i \lambda_i' \to \Sigma_\lambda$, then the first factor eigenvalue is $O(N)$.

5. Show that under weak dependence, $\frac{1}{\sqrt{N}}\sum_{i=1}^N e_{it} \to N(0, \sigma^2)$ for some finite $\sigma^2$.

6. Derive the penalty term structure in Bai-Ng IC1.

### Implementation

7. Simulate an approximate factor model with $N=200$ and controlled local correlation. Verify PCA recovers factors consistently.

8. Compare IC1, IC2, IC3 on FRED-MD data. Do they agree on the number of factors?

9. Implement a test for weak dependence: compute row sums of $|\Sigma_e|$ and check they don't grow with $N$.

---

<div class="callout-insight">

**Insight:** Understanding approximate factor models and large-n theory is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

### Foundational Theory

- **Chamberlain, G. & Rothschild, M. (1983).** "Arbitrage, Factor Structure, and Mean-Variance Analysis on Large Asset Markets." *Econometrica*, 51(5), 1281-1304.
  - Original definition of pervasive factors and approximate factor models.

- **Bai, J. (2003).** "Inferential Theory for Factor Models of Large Dimensions." *Econometrica*, 71(1), 135-171.
  - Asymptotic distribution theory under approximate factor structure.

### Number of Factors Determination

- **Bai, J. & Ng, S. (2002).** "Determining the Number of Factors in Approximate Factor Models." *Econometrica*, 70(1), 191-221.
  - Development of IC1, IC2, IC3 criteria with consistency proofs.

- **Onatski, A. (2010).** "Determining the Number of Factors from Empirical Distribution of Eigenvalues." *Review of Economics and Statistics*, 92(4), 1004-1016.
  - Alternative approach using eigenvalue gap statistics.

### Practical Applications

- **Stock, J.H. & Watson, M.W. (2002).** "Forecasting Using Principal Components from a Large Number of Predictors." *Journal of the American Statistical Association*, 97(460), 1167-1179.
  - Applied demonstration with N=215 macroeconomic predictors.

- **McCracken, M.W. & Ng, S. (2016).** "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business & Economic Statistics*, 34(4), 574-589.
  - Standard benchmark dataset for approximate factor models.

### Advanced Topics

- **Fan, J., Liao, Y., & Mincheva, M. (2013).** "Large Covariance Estimation by Thresholding Principal Orthogonal Complements." *Journal of the Royal Statistical Society B*, 75(4), 603-680.
  - Handling residual covariance structure after factor extraction.

- **Bai, J. & Ng, S. (2008).** "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics*, 3(2), 89-163.
  - Comprehensive survey covering approximate factor theory.

---

**Key Takeaway:** Approximate factor models make factor analysis practical for large, realistic datasets by allowing weak idiosyncratic correlation. Large-N asymptotics ensure PCA remains consistent, and information criteria help determine the number of pervasive factors. This framework underpins modern empirical macroeconomics and finance.

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_approximate_factor_models_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_static_factor_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_factor_model_specification.md">
  <div class="link-card-title">01 Factor Model Specification</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_identification_problem.md">
  <div class="link-card-title">02 Identification Problem</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


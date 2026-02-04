# Factor Extraction

```
┌─────────────────────────────────────────────────────────────────────┐
│ FACTOR EXTRACTION                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   High-dimensional data ──→ Low-dimensional factors                │
│                                                                     │
│   ┌─────────────────────┐      ┌──────────┐                        │
│   │  y₁  y₂  ... y_N    │      │  f₁  f₂  │                        │
│   │  ↓   ↓       ↓      │      │  ↓   ↓   │                        │
│   │ [100 x T data]      │  ──→ │ [3 x T]  │                        │
│   │                     │      │          │                        │
│   │ N=100 series        │      │ r=3      │                        │
│   │ T=200 timepoints    │      │ factors  │                        │
│   └─────────────────────┘      └──────────┘                        │
│                                                                     │
│   Factor model:   y_t = Λ·f_t + e_t                                │
│                                                                     │
│   Where:  Λ (N×r) = factor loadings (how series load on factors)   │
│           f_t (r×1) = factors at time t (common drivers)            │
│           e_t (N×1) = idiosyncratic noise (series-specific)         │
│                                                                     │
│   Three main methods:                                               │
│   1. PCA: Eigenvectors of sample covariance matrix                 │
│   2. ML:  Maximum likelihood (via EM or Kalman filter)             │
│   3. Bayesian: Posterior mean with priors on Λ, Q, H               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ TL;DR: Compress many correlated time series into few uncorrelated  │
│        factors that capture most of the systematic variation.      │
├─────────────────────────────────────────────────────────────────────┤
│ Code (< 15 lines):                                                  │
│                                                                     │
│   from sklearn.decomposition import PCA                             │
│   import numpy as np                                                │
│                                                                     │
│   # Method 1: PCA (fastest)                                         │
│   pca = PCA(n_components=3)                                         │
│   factors_pca = pca.fit_transform(data)  # data: T x N             │
│   loadings = pca.components_.T            # N x r                  │
│                                                                     │
│   # Method 2: Stock-Watson approach (standardize first)            │
│   data_std = (data - data.mean(0)) / data.std(0)                   │
│   factors_sw = PCA(n_components=3).fit_transform(data_std)         │
│                                                                     │
│   # Explained variance                                              │
│   print(f"Variance explained: {pca.explained_variance_ratio_}")    │
├─────────────────────────────────────────────────────────────────────┤
│ Common Pitfall: Using raw data without standardization causes      │
│                 high-variance series to dominate. Always normalize! │
└─────────────────────────────────────────────────────────────────────┘
```

## The Core Idea

**Problem:** You have 100 economic indicators. Running regressions with all 100 leads to:
- Overfitting
- Parameter uncertainty
- Multicollinearity
- Computational cost

**Solution:** Most variation is driven by a few common factors (e.g., "real activity", "inflation", "financial conditions"). Extract these and use them instead.

## Three Extraction Methods Compared

### 1. PCA (Principal Component Analysis)
**Pros:**
- Fast (closed-form solution)
- No distributional assumptions
- Stock-Watson (2002) showed consistency

**Cons:**
- Assumes factors are orthogonal
- No time series structure
- Equal treatment of all observations

**When to use:** Quick exploration, large N, or as initialization for ML

### 2. Maximum Likelihood
**Pros:**
- Statistically efficient
- Proper uncertainty quantification
- Handles missing data naturally
- Incorporates factor dynamics (VAR structure)

**Cons:**
- Slower (iterative EM algorithm)
- Requires distributional assumptions
- Can get stuck in local optima

**When to use:** Final estimation, when you need standard errors, missing data

### 3. Bayesian
**Pros:**
- Incorporates prior information
- Full posterior distribution (not just point estimates)
- Natural regularization
- Handles uncertainty in factor count

**Cons:**
- Slowest (MCMC sampling)
- Requires prior specification
- More complex interpretation

**When to use:** Small samples, strong prior knowledge, need full uncertainty

## Stock-Watson Two-Step Procedure

The workhorse for DFM practitioners:

**Step 1:** Extract factors via PCA
```python
F_hat = principal_components(X, r)  # r factors
```

**Step 2:** Estimate factor dynamics
```python
VAR(F_hat, p=1)  # Factor VAR(1)
```

This is **consistent** even though F_hat has estimation error (Stock & Watson 2002).

## How Many Factors?

Critical question with multiple approaches:

**1. Scree plot:** Elbow in eigenvalue plot (visual)
```python
plt.plot(pca.explained_variance_)
plt.xlabel('Factor number')
plt.ylabel('Eigenvalue')
```

**2. Bai-Ng (2002) criteria:** Information criteria designed for DFM
```python
IC = log(V(k)) + k * penalty_function(N, T)
# Choose k that minimizes IC
```

**3. Variance explained:** Cumulative percentage (rule of thumb: 70-90%)
```python
np.cumsum(pca.explained_variance_ratio_)
```

**Rule of thumb:** Start with 3-5 factors for macro data, 1-2 for financial data.

## Identification Issues

Factors are only identified up to rotation:
- If Λ·f_t fits the data, so does Λ·R · R⁻¹·f_t for any invertible R

**Standard normalization:**
1. Λ'Λ diagonal (factors are orthogonal)
2. Factors ordered by variance explained
3. First loading of each factor positive

PCA automatically does this. ML/Bayesian methods need explicit constraints.

## Real Example

**FRED-MD database:** 127 monthly US macro indicators

Typical factor interpretation (3 factors):
- **Factor 1:** Real activity (GDP, employment, production)
- **Factor 2:** Inflation (prices, costs)
- **Factor 3:** Financial conditions (interest rates, spreads)

These capture 60-70% of total variation!

## Quick Diagnostic

After extraction, check:
1. **Loadings plot:** Do factors have economic interpretation?
2. **Factor correlation:** Should be near-zero (orthogonality)
3. **Residual correlation:** Should be low (common variation extracted)
4. **Factor persistence:** Should show autocorrelation (true dynamics)

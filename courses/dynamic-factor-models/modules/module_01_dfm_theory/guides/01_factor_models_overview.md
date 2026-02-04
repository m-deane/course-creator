# Factor Models Overview

## In Brief

Factor models assume that many observed variables are driven by a few unobserved common factors. Static factor models (PCA) capture contemporaneous relationships only. Dynamic factor models extend this to allow factors to evolve over time and affect variables with lags, making them essential for time series applications.

## Key Insight

**The dimensionality reduction insight:** If 100 economic indicators are all driven by 3 underlying forces (business cycle, monetary policy, oil prices), you can:
1. Extract the 3 factors
2. Model their dynamics (much easier than modeling 100 series)
3. Forecast all 100 series using the factor forecasts

**The dynamic extension:** PCA gives you X(t) = Λ·F(t) + e(t). DFMs add dynamics to both F(t) and the relationship between F and X.

## Visual Explanation

```
STATIC FACTOR MODEL (PCA):

100 Variables ─────┐
(X₁, X₂, ..., X₁₀₀)│
                   ├──→ [PCA] ──→ 3 Factors (F₁, F₂, F₃)
Only uses          │                    │
contemporaneous    │                    │
relationships      │                    └──→ Captures 90% of variance
                   │
                   └──→ X(t) = Λ·F(t) + ε(t)
                         [No dynamics in F!]


DYNAMIC FACTOR MODEL:

100 Variables ─────┐
(X₁, X₂, ..., X₁₀₀)│
                   ├──→ [DFM] ──→ 3 Dynamic Factors
Uses dynamics:     │              │
• Autocorrelation  │              ├─→ F₁(t) = φ₁·F₁(t-1) + ...
• Lagged effects   │              ├─→ F₂(t) = φ₂·F₂(t-1) + ...
• Impulse          │              └─→ F₃(t) = φ₃·F₃(t-1) + ...
  responses        │
                   └──→ X(t) = Λ(L)·F(t) + ε(t)
                         [Factors are dynamic!]
                         [L = lag operator]


STRUCTURE:

           Common Shocks
                │
                ▼
      ┌─────────────────┐
      │  Hidden Factors │  ←─ AR dynamics
      │   F₁, F₂, F₃    │
      └─────────────────┘
           │  │  │
      ┌────┘  │  └────┐
      ▼       ▼       ▼
   ┌──────────────────────┐
   │ Observable Variables │  + idiosyncratic noise
   │  X₁, X₂, ..., X₁₀₀   │
   └──────────────────────┘
```

## Formal Definition

**Static Factor Model (PCA):**
```
X(t) = Λ·F(t) + ε(t)
```
Where:
- X(t) ∈ ℝ^N is the N×1 vector of observables
- F(t) ∈ ℝ^r is the r×1 vector of factors (r << N)
- Λ ∈ ℝ^(N×r) is the factor loading matrix
- ε(t) ∈ ℝ^N is idiosyncratic noise (uncorrelated across i)

**Dynamic Factor Model:**
```
X(t) = Λ(L)·F(t) + ε(t)
F(t) = Φ(L)·F(t-1) + η(t)
```
Where:
- Λ(L) = Λ₀ + Λ₁L + Λ₂L² + ... (lag polynomial)
- Φ(L) captures factor dynamics (usually VAR)
- η(t) ~ N(0, Q) are factor innovations

**State Space Form:**
```
X(t) = Z·α(t) + ε(t)        [Observation equation]
α(t) = T·α(t-1) + R·η(t)    [State equation]
```
Where α(t) contains factors and their lags.

## Intuitive Explanation

**The restaurant analogy:**
Imagine 100 restaurants. Their sales depend on:
- F₁: Overall economic conditions (affects all)
- F₂: Weather (affects outdoor seating)
- F₃: Local events (affects downtown spots)

**Static PCA** would tell you today's sales based on today's factors.

**Dynamic DFM** tells you:
- How economic conditions evolve (persistence, shocks)
- How weather today affects sales tomorrow (lags)
- How a sporting event impacts restaurants for several days

**The macroeconomic analogy:**
100 economic indicators (GDP, employment, spending, etc.) driven by:
- F₁: Business cycle (persistent, mean-reverting)
- F₂: Monetary policy stance (slow-moving)
- F₃: Oil price shocks (temporary but can persist)

DFMs capture how these factors evolve and propagate through the economy.

## Code Implementation

```python
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate data from factor model
np.random.seed(42)
T, N, r = 200, 50, 3

# Generate dynamic factors (AR(1) processes)
factors = np.zeros((T, r))
phi = np.array([0.9, 0.7, 0.5])  # Different persistence

for t in range(1, T):
    factors[t] = phi * factors[t-1] + np.random.randn(r) * 0.5

# Generate loadings
Lambda = np.random.randn(N, r)

# Generate observations
X = factors @ Lambda.T + np.random.randn(T, N) * 0.3

# Static PCA (ignores dynamics)
pca = PCA(n_components=r)
factors_pca = pca.fit_transform(X)

# Plot comparison
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

for i in range(r):
    axes[i].plot(factors[:, i], 'k-', linewidth=2,
                 label=f'True Factor {i+1}', alpha=0.7)
    axes[i].plot(factors_pca[:, i], 'r--', linewidth=2,
                 label=f'PCA Estimate', alpha=0.7)
    axes[i].legend()
    axes[i].set_title(f'Factor {i+1}: φ={phi[i]:.1f} (persistence)')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# PCA captures factors but loses sign and scale
corr = np.corrcoef(factors.T, factors_pca.T)[:r, r:]
print("Correlation between true factors and PCA:")
print(np.abs(np.diag(corr)))

# Variance explained
print(f"\nVariance explained by {r} factors: "
      f"{pca.explained_variance_ratio_.sum():.1%}")
```

## Common Pitfalls

### 1. Using PCA for Time Series (Most Common!)
**Problem:** PCA ignores autocorrelation and cross-correlation dynamics.

**Why it happens:** PCA is easy to implement, tempting to apply to everything.

**How to avoid:** Use PCA only as initialization for DFM estimation, never as final model.

### 2. Confusing Factors with Principal Components
**Problem:** PCs are rotations of factors, not the factors themselves.

**Why it happens:** "PCA extracts factors" is common but misleading terminology.

**How to avoid:**
- PCA finds orthogonal directions of maximum variance
- Factors are economic/structural concepts
- Need identification restrictions to recover interpretable factors

### 3. Wrong Number of Factors
**Problem:** Too many factors → overfitting. Too few → missing dynamics.

**Why it happens:** No clear rule for factor selection.

**How to avoid:**
```python
# Use information criteria
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

ic_values = []
for r in range(1, 10):
    model = DynamicFactor(X, k_factors=r, factor_order=2)
    results = model.fit()
    ic_values.append(results.aic)

r_optimal = np.argmin(ic_values) + 1
print(f"Optimal factors: {r_optimal}")
```

### 4. Ignoring Identification
**Problem:** Estimating unidentified model gives arbitrary results.

**Why it happens:** Forgetting that factors are latent (unobserved).

**How to avoid:** Always impose identification restrictions:
- Normalize factor variance to 1
- Impose structure on Λ (triangular, zero restrictions)
- See Module 01 Guide 3 on identification

### 5. Assuming Independent Idiosyncratic Errors
**Problem:** Real data often has cross-sectional correlation in ε(t).

**Why it happens:** Standard DFM assumption for tractability.

**How to avoid:** Use approximate factor models (allow weak correlation in ε).

## Connections

### Builds on:
- **PCA** - Static factor extraction
- **VAR Models** - Factor dynamics
- **State Space Models (Module 00)** - Representation framework
- **Kalman Filter** - Estimation algorithm

### Leads to:
- **DFM Specification (Next Guide)** - How to write DFM in state space form
- **Identification (Guide 3)** - How to pin down unique factors
- **Maximum Likelihood (Module 02)** - How to estimate parameters

### Equivalent Formulations:
- **Approximate Factor Models** - Bai & Ng framework (allows some ε correlation)
- **Generalized Dynamic Factor Models** - Block structure in factors
- **Factor-Augmented VAR (FAVAR)** - Add factors as regressors in VAR

### Applications:
- **Nowcasting** - Real-time GDP estimation from monthly indicators
- **Forecasting** - Predict many variables using few factors
- **Structural Analysis** - Identify macroeconomic shocks
- **Risk Management** - Common risk factors in portfolios

## Practice Problems

### Conceptual Questions

1. **Why does PCA fail for time series?**
   - Hint: What property of time series does PCA ignore?

2. **Explain the "curse of dimensionality" in forecasting.**
   - How do factor models solve it?

3. **What's the difference between a factor and a principal component?**
   - When are they the same?

### Implementation Challenges

4. **Generate data from a 2-factor DFM:**
   ```python
   # F₁ follows AR(1) with φ=0.8
   # F₂ follows AR(1) with φ=0.5
   # Generate N=30 variables
   # Add idiosyncratic noise
   ```

5. **Compare static vs dynamic factor extraction:**
   - Use PCA on generated data
   - Correlate with true factors
   - Why does PCA struggle?

6. **Implement scree plot:**
   ```python
   # Plot eigenvalues
   # Identify "elbow" point
   # How many factors?
   ```

### Advanced

7. **Prove PCA maximizes variance:**
   - Show first PC is eigenvector of covariance matrix
   - Connect to factor model interpretation

8. **Derive orthogonality of PCs:**
   - Why are principal components uncorrelated?
   - Does this hold for dynamic factors?

## Further Reading

- Stock & Watson (2002): "Forecasting Using Principal Components from a Large Number of Predictors"
- Bai & Ng (2002): "Determining the Number of Factors in Approximate Factor Models"
- Forni et al. (2000): "The Generalized Dynamic Factor Model"
- See [Additional Readings](../resources/additional_readings.md) for complete list

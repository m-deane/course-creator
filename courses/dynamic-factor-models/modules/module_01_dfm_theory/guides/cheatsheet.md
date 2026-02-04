# Module 01 Cheatsheet: Dynamic Factor Models

## Quick Reference

### DFM Structure

```python
# Observation equation
X(t) = Λ(L)·F(t) + ε(t)
     = Λ₀·F(t) + Λ₁·F(t-1) + ... + Λ_p·F(t-p) + ε(t)

# Factor dynamics (VAR)
F(t) = Φ₁·F(t-1) + ... + Φ_q·F(t-q) + η(t)

# State space form
X(t) = Z·α(t) + ε(t)        ε ~ N(0, H)
α(t) = T·α(t-1) + R·η(t)    η ~ N(0, Q)
```

### Dimensions
- N: number of observed variables
- r: number of factors (r << N)
- p: lags in observation equation
- q: factor VAR order
- State dimension: m = r × max(p+1, q)

### State Space Conversion

```python
# State vector: α(t) = [F(t)', F(t-1)', ..., F(t-s+1)']'
# where s = max(p+1, q)

# Z matrix (N × m)
Z = [Λ₀, Λ₁, ..., Λ_p, 0, ...]

# T matrix (m × m) - Companion form
T = [[Φ₁, Φ₂, ..., Φ_q],
     [I,  0,  ..., 0  ],
     [0,  I,  ..., 0  ],
     ...]

# R matrix (m × r)
R = [I; 0; 0; ...]
```

## Identification Restrictions

**Need r² restrictions total:**
- r for scale
- r(r-1)/2 for rotation

**Common choices:**

1. **Triangular (Stock-Watson):**
```python
Λ[:r, :r] = lower triangular with unit diagonal
```

2. **PC normalization:**
```python
Λ'Λ = I
Var(F) = diagonal, decreasing
```

## Code Patterns

### Create DFM
```python
import numpy as np

# Parameters
N, r, p, q = 20, 3, 1, 1

# Loadings
Lambda_0 = np.random.randn(N, r)
Lambda_1 = np.random.randn(N, r) * 0.3

# Factor dynamics
Phi_1 = np.diag([0.9, 0.7, 0.5])

# Noise
H = np.eye(N) * 0.5
Q = np.eye(r)
```

### Estimate with statsmodels
```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

model = DynamicFactor(
    endog=X,
    k_factors=r,
    factor_order=q,
    error_order=0
)
results = model.fit(maxiter=1000)

# Extract factors
factors = results.factors.filtered
```

### Information Criteria
```python
# Determine number of factors
from statsmodels.tools.eval_measures import aic, bic

ic_results = []
for r in range(1, 10):
    model = DynamicFactor(X, k_factors=r, factor_order=1)
    res = model.fit(disp=False)
    ic_results.append({'r': r, 'aic': res.aic, 'bic': res.bic})

# Choose r with min(IC)
```

## Diagnostics

```python
# Factor loadings
Lambda = results.coefficient_matrices_var[0]

# R-squared by variable
X_fitted = results.fittedvalues
r_squared = 1 - np.var(X - X_fitted, axis=0) / np.var(X, axis=0)

# Variance explained
total_var = np.trace(np.cov(X.T))
factor_var = np.trace(Lambda @ results.factor_covariance @ Lambda.T)
pct_explained = factor_var / total_var
```

## Common Model Specifications

### 1. Static Factors (p=0)
```python
X(t) = Λ·F(t) + ε(t)
F(t) = Φ·F(t-1) + η(t)
```

### 2. Dynamic Loadings (p>0)
```python
X(t) = Λ₀·F(t) + Λ₁·F(t-1) + ε(t)
F(t) = Φ·F(t-1) + η(t)
```

### 3. Block Structure
```python
# Separate factors for different variable groups
F = [F_real, F_nominal, F_financial]
Λ = block diagonal
```

## Formulas

### Variance Decomposition
```
Var(X_i) = Σⱼ λ²ᵢⱼ·Var(Fⱼ) + Var(εᵢ)

R²ᵢ = [Σⱼ λ²ᵢⱼ·Var(Fⱼ)] / Var(X_i)
```

### Factor Covariance
```
Cov(F(t)) = Φ·Cov(F(t-1))·Φ' + Q
```

### Long-run Loading
```
Λ(1) = Λ₀ + Λ₁ + ... + Λ_p  (cumulative response)
```

## Quick Checks

- [ ] N > 5r (rule of thumb for identification)
- [ ] max(eigenvalues(Φ)) < 1 (stationarity)
- [ ] Variance explained > 60% (goodness of fit)
- [ ] Factor autocorrelations decay (not random walk)
- [ ] Idiosyncratic errors not serially correlated

## Useful Functions

```python
def scree_plot(X, max_factors=15):
    """Plot eigenvalues to determine number of factors."""
    cov_matrix = np.cov(X.T)
    eigenvalues = np.linalg.eigvalsh(cov_matrix)[::-1]

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_factors+1), eigenvalues[:max_factors],
             'o-', linewidth=2)
    plt.xlabel('Factor Number')
    plt.ylabel('Eigenvalue')
    plt.title('Scree Plot')
    plt.grid(True)
    plt.show()

def variance_explained(X, factors, Lambda):
    """Compute variance explained by factors."""
    X_common = factors @ Lambda.T
    var_common = np.var(X_common, axis=0).sum()
    var_total = np.var(X, axis=0).sum()
    return var_common / var_total
```

## Resources
- statsmodels DynamicFactor: `statsmodels.tsa.statespace.dynamic_factor`
- R package: `dfms` or `nowcasting`
- MATLAB: DFM toolbox by Stock & Watson

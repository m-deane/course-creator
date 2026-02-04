# Module 04 Cheatsheet: Advanced Extensions

## Time-Varying Parameters

### Random Walk Parameters
```
λ_t = λ_{t-1} + ω_t,  ω_t ~ N(0, σ²_ω)
```

### Rolling Window Estimation
```python
for t in range(window_size, T, step):
    window_data = data[t-window_size:t]
    dfm = DynamicFactor(window_data, k_factors=r)
    dfm_res = dfm.fit()
    loadings_t = dfm_res.params['loading']
```

### Forgetting Factor
```
Weight at time t: w_t = λ^(T-t),  λ ∈ (0,1)
Common values: λ = 0.95 (slow), 0.99 (very slow)
```

### When to Use TVP
- Suspected structural breaks (COVID, crises)
- Parameters unstable over time (Chow test)
- Rolling forecast errors show pattern

---

## Mixed-Frequency Models

### Temporal Aggregation
```
Flow (GDP): F̄_τ = (1/3)(F_{3τ-2} + F_{3τ-1} + F_{3τ})
Stock (Rate): F_τ = F_{3τ} (end of quarter)
```

### State-Space Mixed-Frequency
```
Monthly: X^(m)_t = Λ^(m) F_t + e^(m)_t
Quarterly: y^(q)_τ = Λ^(q) F̄_τ + e^(q)_τ
Dynamics: F_t = Φ F_{t-1} + η_t
```

### MIDAS Regression
```
y_τ = α + Σ_{k=0}^K β_k(θ) X_{3τ-k} + ε

Exponential Almon: β_k = exp(θ₁k + θ₂k²)
Beta weights: β_k = B(k/K; θ₁, θ₂)
```

### Skip-Sampling Pattern
```
Monthly GDP observations:
t:    1   2   3 | 4   5   6 | 7   8   9
GDP:  ✗   ✗   ✓ | ✗   ✗   ✓ | ✗   ✗   ✓
```

---

## Large Datasets

### Two-Step Estimation
```
STEP 1: PCA
  Σ_X = (1/T) X'X
  Eigendecomp: Σ_X = V Λ V'
  Loadings: Λ̂ = [√μ₁v₁, ..., √μᵣvᵣ]
  Factors: F̂ = X Λ̂

STEP 2: Kalman Smoother
  VAR on F̂: F_t = Φ̂ F_{t-1} + η_t
  Smooth with fixed Φ̂, Λ̂
```

### Computational Complexity
```
Full MLE: O(N² · T)
Two-step: O(N · T · r)

Speedup for N=200, r=5: ~40×
```

### Sparse Loadings (LASSO)
```
min_{Λ,F} (1/2T)||X - ΛF'||² + λ Σ|λ_ij|

Result: Many λ_ij = 0 (sparse)
Tuning: BIC or cross-validation for λ
```

### Factor Number Selection
```
Information Criteria:
IC_p(r) = log(V_r) + r·p(N,T)

PC1: p = (N+T)/(NT) log(NT/(N+T))
PC2: p = (N+T)/(NT) log(min(N,T))
BIC: p = log(min(N,T))/(min(N,T))

Choose r that minimizes IC_p(r)
```

---

## Decision Trees

### When to Use Time-Varying Parameters?
```
Constant forecast errors over time?
├─ Yes → Constant parameters OK
└─ No (errors higher post-2015?)
   ├─ Known break dates? → Structural break DFM
   └─ Unknown/gradual? → Rolling window or random walk
```

### MIDAS vs State-Space Mixed-Frequency?
```
How many high-frequency predictors?
├─ Single variable → MIDAS (simple, fast)
└─ Multiple variables
   ├─ Ragged edges? → State-space DFM
   └─ No missing data? → Either works
```

### Two-Step vs Full MLE?
```
How large is N?
├─ N < 50 → Full MLE (more efficient)
└─ N ≥ 50
   ├─ Need speed? → Two-step (much faster)
   ├─ Need optimal inference? → Full MLE
   └─ Large N (>100)? → Two-step (only option)
```

---

## Code Snippets

### Rolling Window DFM
```python
def rolling_window_dfm(data, window=60, step=1, k_factors=3):
    loadings_history = []
    for t in range(window, len(data), step):
        window_data = data.iloc[t-window:t]
        dfm = DynamicFactor(window_data, k_factors=k_factors)
        dfm_res = dfm.fit(disp=False)
        loadings_history.append(dfm_res.params['loading'])
    return loadings_history
```

### Mixed-Frequency Bridge
```python
# Monthly factors → quarterly GDP
factors_monthly = dfm_res.factors.filtered
factors_quarterly = factors_monthly.resample('Q').mean()

bridge = LinearRegression()
bridge.fit(factors_quarterly, gdp_quarterly)
```

### Two-Step Estimation
```python
# Step 1: PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=k_factors)
factors_pca = pca.fit_transform(X)

# Step 2: VAR + Kalman smooth
from statsmodels.tsa.api import VAR
var = VAR(factors_pca)
var_res = var.fit(maxlags=1)
# (then Kalman smooth with fixed parameters)
```

### Sparse Loadings
```python
from sklearn.linear_model import Lasso
sparse_loadings = np.zeros((N, r))

for i in range(N):
    lasso = Lasso(alpha=0.1)
    lasso.fit(factors, X[:, i])
    sparse_loadings[i] = lasso.coef_
```

---

## Key Formulas

### TVP State-Space
```
X_t = Λ_t F_t + e_t
F_t = Φ F_{t-1} + η_t
Λ_t = Λ_{t-1} + ω_t  (random walk loadings)
```

### MIDAS Aggregation
```
Quarterly from monthly:
y_τ = β₀ + Σ_{j=0}^{K} w_j(θ) X_{m,3τ-j} + ε_τ

Weights sum to 1: Σ w_j = 1
```

### Variance Explained (PCA)
```
Proportion: μ₁ + ... + μᵣ
            ─────────────
            μ₁ + ... + μₙ

Typical: 3-5 factors explain 50-70% of variance
```

---

## Common Parameter Values

### Time-Varying
- **Window size:** 60 months (5 years)
- **Forgetting factor:** 0.95-0.99
- **Parameter variance:** σ²_ω = 0.001-0.01

### Mixed-Frequency
- **MIDAS lags:** 6-12 months for quarterly target
- **Aggregation:** Flow (average), Stock (last)

### Large Datasets
- **Factors:** r = 3-10 (even for N=200)
- **LASSO penalty:** α = 0.01-0.1
- **Sparsity target:** 20-30% zero loadings

---

## Diagnostics

### Parameter Stability Test
```python
# Chow test for structural break
from statsmodels.stats.diagnostic import break_var

# Split at candidate date
pre = data.loc[:break_date]
post = data.loc[break_date:]

# Estimate separate models, test equality
```

### Mixed-Frequency Alignment Check
```python
# Ensure quarterly aligns with last month
assert gdp_quarterly.index == factors_monthly.resample('Q').last().index
```

### Factor Number Selection
```python
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X)

# Scree plot
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Factor')
plt.ylabel('Variance Explained')

# Elbow point = optimal r
```

---

## Resources

**Software:**
- `statsmodels.tsa.statespace.DynamicFactor` - DFM estimation
- `sklearn.decomposition.PCA` - Fast PCA
- `sklearn.linear_model.Lasso` - Sparse estimation

**Datasets:**
- FRED-MD: 127 monthly US series
- FRED-QD: Quarterly version
- ECB Statistical Data Warehouse: Euro area

**Benchmarks:**
- Rolling window: 5-year (60 months) standard
- Two-step: Use for N > 50
- MIDAS: Exponential Almon most common

---

*For detailed explanations, see full guides in this module.*

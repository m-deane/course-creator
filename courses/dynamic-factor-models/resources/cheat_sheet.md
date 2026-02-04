# Dynamic Factor Models: 1-Page Cheat Sheet

**Print this page.** Reference while coding.

---

## The Model in One Picture

```
Observed Data (X)  =  Factor Loadings (Λ) × Latent Factors (F)  +  Idiosyncratic Noise (e)
    [T × N]              [N × r]              [T × r]                 [T × N]

Where factors evolve: F_t = Φ × F_{t-1} + η_t
```

**Key idea:** N series share r << N common factors. Extract factors, use for forecasting.

---

## Essential Equations

### Static Factor Model
```
X_it = λ_i1 * F_1t + λ_i2 * F_2t + ... + λ_ir * F_rt + e_it

Matrix form: X_t = Λ × F_t + e_t
```

### Dynamic Factor Model (State-Space Form)
```
Measurement:  X_t = Λ × F_t + e_t        [Observation equation]
Transition:   F_t = Φ × F_{t-1} + η_t    [Factor dynamics]
```

### Principal Components Estimator
```
F̂ = X × v  where v are eigenvectors of X'X
Λ̂ = X' × F̂ / T
```

### Kalman Filter Recursions
```
Predict:  F_t|t-1 = Φ × F_t-1|t-1
         P_t|t-1 = Φ × P_t-1|t-1 × Φ' + Q

Update:   K_t = P_t|t-1 × Λ' × (Λ × P_t|t-1 × Λ' + R)^(-1)
         F_t|t = F_t|t-1 + K_t × (X_t - Λ × F_t|t-1)
         P_t|t = (I - K_t × Λ) × P_t|t-1
```

---

## Code Snippets: Copy-Paste Ready

### Extract Factors (PCA Approach)
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Extract r factors
pca = PCA(n_components=3)
factors = pca.fit_transform(X_scaled)
loadings = pca.components_.T

# Variance explained
print(f"Variance: {pca.explained_variance_ratio_.sum():.1%}")
```

### Fit Dynamic Factor Model (Statsmodels)
```python
import statsmodels.api as sm

# State-space DFM with AR(1) factors
mod = sm.tsa.DynamicFactor(
    data,
    k_factors=3,          # Number of factors
    factor_order=1        # AR(1) dynamics
)
res = mod.fit(disp=False, maxiter=1000)

# Extract factors
filtered = res.filtered_state[0]    # Real-time F_t|t
smoothed = res.smoothed_state[0]    # Retrospective F_t|T
```

### Choose Number of Factors (Scree Plot)
```python
pca = PCA().fit(X_scaled)

plt.plot(range(1, 11), pca.explained_variance_ratio_[:10], 'o-')
plt.xlabel('Factor Number')
plt.ylabel('Variance Explained')
plt.title('Scree Plot')
# Pick elbow point
```

### Nowcast with Bridge Equation
```python
# Monthly factors → Quarterly GDP
factors_monthly = pca.fit_transform(monthly_data)

# Aggregate to quarterly
factors_quarterly = pd.DataFrame(
    factors_monthly,
    index=monthly_data.index
).resample('Q').mean()

# Bridge regression
model = sm.OLS(gdp_quarterly, sm.add_constant(factors_quarterly)).fit()

# Nowcast current quarter
current_q_factors = factors_monthly[-3:].mean(axis=0)  # Last 3 months
nowcast = model.predict([1] + list(current_q_factors))[0]
```

### Handle Ragged Edge (Kalman)
```python
# Data with NaN at end (ragged edge)
data_ragged = data.copy()
data_ragged.iloc[-5:, [0, 2, 5]] = np.nan  # Some series lag

# Kalman filter handles automatically
mod = sm.tsa.DynamicFactor(data_ragged, k_factors=3, factor_order=1)
res = mod.fit()  # Works with missing data!
```

### Load FRED-MD Data
```python
# Direct download
url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
fred_md = pd.read_csv(url, skiprows=1, index_col=0, parse_dates=True)

# Apply transformations (see FRED-MD documentation)
# 1=level, 2=diff, 4=log, 5=log-diff
```

---

## Decision Flowchart

```
START: Do I need a factor model?
    |
    ├─→ [N < 10 series] → Use VAR or individual models
    |
    ├─→ [N ≥ 10 series, no common movement] → Model separately
    |
    └─→ [N ≥ 10 series, common drivers] → FACTOR MODEL
            |
            ├─→ [Static relationships, no dynamics] → STATIC FACTOR (PCA)
            |       Code: PCA(n_components=r).fit_transform(X)
            |
            ├─→ [Factors have dynamics] → DYNAMIC FACTOR
            |       |
            |       ├─→ [No missing data, same frequency] → PCA + VAR
            |       |       Code: factors = PCA().fit_transform(X)
            |       |             VAR(factors).fit()
            |       |
            |       └─→ [Missing data OR mixed frequency] → STATE-SPACE DFM
            |               Code: DynamicFactor(X, k_factors=r).fit()
            |
            └─→ [Need real-time nowcast] → KALMAN FILTER DFM
                    Code: res.filtered_state[0]  # Real-time estimates
```

---

## Common Pitfalls & Fixes

| Problem | Symptom | Fix |
|---------|---------|-----|
| **Forgot to standardize** | One series dominates factors | `StandardScaler().fit_transform(X)` |
| **Non-stationary data** | Spurious factors, trends | Difference or log-difference first |
| **Too many factors** | Overfitting, noise capture | Use scree plot or IC criteria |
| **Too few factors** | Poor fit, missed info | Check cumulative variance explained |
| **Kalman not converging** | `LinAlgError: singular matrix` | Reduce factors, increase iterations, check for perfect collinearity |
| **Bridge equation mismatch** | Frequencies don't align | Resample properly: `monthly.resample('Q').mean()` |
| **Using revised data** | Overstated accuracy | Use real-time vintages for backtest |
| **Perfect multicollinearity** | `LinAlgError` | Drop redundant series |

---

## Key Parameters Reference

### PCA
```python
PCA(
    n_components=3,          # Number of factors (or 0.95 for % variance)
    whiten=False,            # Set True for unit variance factors
    svd_solver='auto'        # 'full', 'arpack', 'randomized'
)
```

### DynamicFactor
```python
sm.tsa.DynamicFactor(
    endog,                   # Data [T × N]
    k_factors=3,             # Number of factors
    factor_order=1,          # AR order for factors (usually 1)
    error_cov_type='diagonal',  # 'diagonal' or 'unstructured'
    error_order=0,           # MA order for errors (usually 0)
    error_var=False          # Allow heteroskedastic errors
)
```

---

## Model Selection: Information Criteria

### Bai-Ng IC Criteria
```python
def compute_IC(X, r, ic_type='IC2'):
    """
    IC1: penalty = σ̂² × (N+T)/(NT) × log(NT/(N+T))
    IC2: penalty = σ̂² × (N+T)/(NT) × log(min(N,T))
    IC3: penalty = σ̂² × log(min(N,T)) / min(N,T)
    """
    N, T = X.shape
    pca = PCA(n_components=r).fit(X)
    sigma2 = np.var(X - pca.inverse_transform(pca.transform(X)))

    if ic_type == 'IC1':
        penalty = (N + T) / (N * T) * np.log((N * T) / (N + T))
    elif ic_type == 'IC2':
        penalty = (N + T) / (N * T) * np.log(min(N, T))
    elif ic_type == 'IC3':
        penalty = np.log(min(N, T)) / min(N, T)

    return sigma2 * penalty

# Usage
ic_values = [compute_IC(X_scaled, r) for r in range(1, 11)]
optimal_r = np.argmin(ic_values) + 1
```

---

## Quick Diagnostics

### Factor Quality Checks
```python
# 1. Variance explained (aim for 60%+)
print(f"Total variance: {pca.explained_variance_ratio_.sum():.1%}")

# 2. Loading magnitudes (should be spread, not concentrated)
print(f"Max loading: {np.abs(loadings).max():.2f}")
print(f"Mean loading: {np.abs(loadings).mean():.2f}")

# 3. Factor correlations (should be ~0 if orthogonal)
print(pd.DataFrame(factors).corr())

# 4. Persistence (factors should be somewhat persistent)
from statsmodels.tsa.stattools import acf
for i in range(factors.shape[1]):
    print(f"Factor {i+1} AR(1): {acf(factors[:, i], nlags=1)[1]:.2f}")
```

### Nowcast Accuracy
```python
# Out-of-sample RMSE
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = np.mean(np.abs(y_test - y_pred))

# Relative to naive forecast
naive_rmse = np.sqrt(np.mean((y_test - y_test.shift(1))**2))
skill = 1 - rmse / naive_rmse  # >0 = better than naive

print(f"RMSE: {rmse:.3f}")
print(f"Skill score: {skill:.1%}")
```

---

## Data Transformations (FRED-MD Codes)

| Code | Transformation | Use Case |
|------|----------------|----------|
| 1 | No transform | Already stationary (rates) |
| 2 | Δx_t | Make stationary (levels) |
| 3 | Δ²x_t | Remove trend + seasonality |
| 4 | log(x_t) | Stabilize variance |
| 5 | Δlog(x_t) | Growth rates |
| 6 | Δ²log(x_t) | Acceleration |

```python
def transform_series(x, code):
    if code == 1: return x
    if code == 2: return x.diff()
    if code == 4: return np.log(x)
    if code == 5: return np.log(x).diff()
    # ... etc
```

---

## Typical Workflow

1. **Load & Transform Data**
   ```python
   X = load_data()
   X_transformed = apply_transformations(X)
   X_scaled = StandardScaler().fit_transform(X_transformed)
   ```

2. **Choose Number of Factors**
   ```python
   pca = PCA().fit(X_scaled)
   plt.plot(pca.explained_variance_ratio_[:10])  # Scree plot
   r = 3  # Pick from elbow
   ```

3. **Extract Factors**
   ```python
   factors = PCA(n_components=r).fit_transform(X_scaled)
   ```

4. **Forecast with Factors**
   ```python
   model = sm.OLS(y, sm.add_constant(factors)).fit()
   forecast = model.predict(future_factors)
   ```

5. **Evaluate**
   ```python
   rmse = np.sqrt(mean_squared_error(y_test, forecast))
   ```

---

## When to Use Alternatives

| Situation | Use Instead | Why |
|-----------|-------------|-----|
| N < 10 | VAR | Not enough series for factors |
| No common factors | Individual models | Factor extraction won't help |
| Need causal inference | Structural VAR | Factors obscure causal links |
| High-frequency (< daily) | Direct modeling | Factor extraction too slow |
| Perfect data, same freq | Simple PCA + regression | State-space overkill |

---

## Resources Quick Links

- [Quick-Starts](../quick-starts/) - Working examples
- [Templates](../templates/) - Production code
- [Glossary](glossary.md) - Term definitions
- [Setup](setup.md) - Installation help

---

**Pro Tips:**
- Always standardize before PCA
- Use real-time data vintages for backtesting
- Start with 3 factors, adjust from there
- Plot factors over time to check if they make economic sense
- Kalman filter handles missing data - don't impute manually

---

**Version:** 1.0 | **Last Updated:** 2025

**Print this page and keep by your desk while coding!**

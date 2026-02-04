# Glossary: Key Terms with Working Examples

Quick reference for Dynamic Factor Models terminology. Each term includes a practical example you can run.

---

## Core Concepts

### Approximate Factor Model
**What it is:** Factor model allowing weak correlation in idiosyncratic errors. Real-world data violates strict independence.

**Example:** `factors = PCA(n_components=3).fit_transform(data)` works even when errors are slightly correlated.

---

### Autoregressive (AR) Process
**What it is:** Current value depends on past values: y_t = 0.8*y_{t-1} + noise

**Example:** `model = ARIMA(data, order=(1,0,0)).fit()` - First factor often follows AR(1) process.

---

### Bai-Ng Criteria
**What it is:** Choose number of factors by balancing fit vs overfitting.

**Example:**
```python
from sklearn.decomposition import PCA
# Try 1-10 factors, pick where IC is minimized
ic_values = [compute_IC(PCA(n).fit(X)) for n in range(1, 11)]
optimal_factors = np.argmin(ic_values) + 1
```

---

### Bridge Equation
**What it is:** Link monthly data to quarterly GDP: GDP_Q = β₁*Factor₁ + β₂*Factor₂ + ε

**Example:**
```python
# Monthly factors → Quarterly GDP
quarterly_factors = monthly_factors.resample('Q').mean()
model = sm.OLS(gdp_quarterly, quarterly_factors).fit()
```

---

### Common Component
**What it is:** Part of each series explained by shared factors: X_it = λᵢ'F_t (systematic movement)

**Example:** When Fed raises rates, ALL bond yields move together - that's the common component.

---

### Communality
**What it is:** % variance explained by factors. High communality = mostly common movement.

**Example:**
```python
pca = PCA(n_components=3).fit(X)
communality = pca.explained_variance_ratio_.sum()  # e.g., 0.65 = 65%
```

---

## Estimation Methods

### Diffusion Index
**What it is:** First few principal components summarizing 100+ series. Stock-Watson's forecasting tool.

**Example:**
```python
# Extract 3 diffusion indices from FRED-MD
pca = PCA(n_components=3)
diffusion_indices = pca.fit_transform(fred_md_data)
# Use in forecast: y_t+1 = α + β'*indices_t
```

---

### EM Algorithm
**What it is:** Iterate: (1) Estimate factors given parameters, (2) Estimate parameters given factors.

**Example:** Statsmodels DynamicFactor uses EM automatically:
```python
mod = sm.tsa.DynamicFactor(data, k_factors=3, factor_order=1)
res = mod.fit()  # EM runs behind the scenes
```

---

### Exact Factor Model
**What it is:** Requires completely independent idiosyncratic errors. Rarely true, hence "approximate" models.

**Example:** Not used in practice. Approximate models work better with real data.

---

## Advanced Structures

### Factor Augmented VAR (FAVAR)
**What it is:** VAR using factors + key variables: [Factors, PolicyRate]_t = Φ*[Factors, PolicyRate]_{t-1} + ε

**Example:**
```python
# Add factors to VAR with interest rate
factors_and_policy = pd.concat([factors_df, policy_rate], axis=1)
var_model = VAR(factors_and_policy).fit(maxlags=2)
```

---

### Factor Loadings
**What it is:** Weights λᵢⱼ showing how variable i responds to factor j.

**Example:**
```python
pca = PCA(n_components=3).fit(X)
loadings = pca.components_.T  # [N x 3] matrix
# Large loading = strong response to that factor
```

---

### Factor Scores
**What it is:** Estimated values of latent factors F_t at each time t.

**Example:**
```python
factors = pca.fit_transform(X)  # T x 3 array of factor values
plt.plot(factors[:, 0])  # First factor over time
```

---

### FRED-MD
**What it is:** 127 monthly US macro series, pre-transformed, ready for factor analysis. Updated monthly.

**Example:**
```python
url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv"
fred_md = pd.read_csv(url, skiprows=1, index_col=0, parse_dates=True)
```

---

## Kalman Filter Concepts

### Kalman Filter
**What it is:** Real-time factor estimation using data up to time t: F_t | {data up to t}

**Example:**
```python
from statsmodels.tsa.statespace import DynamicFactor
mod = DynamicFactor(data, k_factors=2, factor_order=1)
res = mod.fit()
filtered_factors = res.filtered_state[0]  # Real-time estimates
```

---

### Kalman Smoother
**What it is:** Best factor estimates using ALL data: F_t | {all data}. Smoother than filtered.

**Example:**
```python
smoothed_factors = res.smoothed_state[0]  # Uses full dataset
# smoothed_factors are less noisy than filtered_factors
```

---

### Identification
**What it is:** Factors are unique only up to rotation. Need constraints: F'F/T = I or Λ'Λ diagonal.

**Example:** PCA automatically identifies by orthogonal rotation. Different rotations give same fit.

---

### Idiosyncratic Component
**What it is:** Series-specific variation NOT explained by factors: e_it = X_it - λᵢ'F_t

**Example:** Fed policy affects all rates (common) but mortgage rates have housing-specific noise (idiosyncratic).

---

## Sparse Methods

### LASSO (L1 Penalty)
**What it is:** Penalized regression setting some coefficients exactly to zero. Selects variables.

**Example:**
```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(factors, target)
selected = factors.columns[lasso.coef_ != 0]  # Non-zero factors
```

---

### Targeted Predictors
**What it is:** Pre-select variables correlated with your target before extracting factors.

**Example:**
```python
# Want to forecast inflation? Use only variables correlated with it
correlations = X.corrwith(inflation).abs()
top_predictors = X[correlations.nlargest(50).index]
factors = PCA(n_components=3).fit_transform(top_predictors)
```

---

## Mixed Frequency

### MIDAS (Mixed Data Sampling)
**What it is:** Combine daily, monthly, quarterly data using distributed lags with constraints.

**Example:**
```python
# Quarterly GDP from monthly indicators
# GDP_Q = β₀ + Σ β(i)*indicator_{t-i} where β(i) = polynomial weights
```

---

### Nowcasting
**What it is:** Predict current quarter GDP before it's released using high-frequency data.

**Example:**
```python
# It's July 15. Official Q2 GDP releases July 30.
# Use June data to predict Q2 GDP NOW:
current_factors = kalman_filter.predict(june_data)
gdp_nowcast = beta @ current_factors
```

---

### Ragged Edge
**What it is:** Missing data at dataset end due to different publication lags.

**Example:**
```
Today: July 15
Employment: Available through June ✓
Retail sales: Available through June ✓
GDP: Last known value is Q1 (ends March) ⚠️
```

Kalman filter handles this automatically.

---

## Model Selection

### Number of Factors (r)
**What it is:** How many latent factors to extract. Too few = miss info, too many = overfit.

**Example:**
```python
pca = PCA().fit(X)
plt.plot(pca.explained_variance_ratio_)
plt.xlabel('Factor number')
# Look for "elbow" in scree plot
```

---

### Scree Plot
**What it is:** Plot of eigenvalues showing how much variance each factor explains. Elbow suggests cutoff.

**Example:**
```python
pca = PCA().fit(X)
plt.plot(range(1, 11), pca.explained_variance_ratio_[:10], 'o-')
# Steep drop after factor 3 → use 3 factors
```

---

## State-Space Framework

### State-Space Model
**What it is:** Two equations: (1) Observation: X_t = Λ*F_t + e_t, (2) Transition: F_t = Φ*F_{t-1} + η_t

**Example:**
```python
# Statsmodels state-space representation
mod = sm.tsa.statespace.DynamicFactor(
    endog=data,
    k_factors=3,      # Number of factors
    factor_order=1    # AR(1) factors
)
```

---

### Stock-Watson Estimator
**What it is:** Two steps: (1) PCA to get factors, (2) Regression for forecasting. Fast and consistent.

**Example:**
```python
# Step 1: Extract factors
factors = PCA(n_components=3).fit_transform(X_train)

# Step 2: Forecast with factors
model = sm.OLS(y_train, sm.add_constant(factors)).fit()
```

---

## Data Handling

### Unbalanced Panel
**What it is:** Not all series cover same time period. Some start/end early.

**Example:**
```python
# Series 1: 1990-2020
# Series 2: 2000-2020
# Series 3: 1995-2018
# DFM handles this! Just use pd.DataFrame with NaN
```

---

### Vintage Data
**What it is:** Data as it was known at time t, before revisions. For real-time evaluation.

**Example:**
```python
# July 2020 vintage: GDP initially reported as -5.0%
# July 2021 vintage: GDP revised to -5.4%
# For backtesting, use 2020 vintage only!
```

---

## Transformations

### Stationarity Transformation
**What it is:** Make series stationary (constant mean, variance). Usually: differencing or log-differencing.

**Example:**
```python
# FRED-MD transformation codes:
# 1 = no transformation
# 2 = first difference
# 5 = log difference (growth rate)
transformed = np.log(series).diff()  # Log-difference
```

---

## Statistical Concepts

### Maximum Likelihood Estimation (MLE)
**What it is:** Find parameters maximizing probability of observed data.

**Example:**
```python
mod = sm.tsa.DynamicFactor(data, k_factors=3)
res = mod.fit(maxiter=1000)  # MLE via EM algorithm
print(res.llf)  # Log-likelihood value
```

---

### Orthogonal Factors
**What it is:** Uncorrelated factors: Cor(F₁, F₂) = 0. Standard assumption.

**Example:** PCA always produces orthogonal factors by construction.

---

### Rotation (Varimax, etc.)
**What it is:** Rotate factors to make loadings interpretable (0 or 1, not 0.5).

**Example:**
```python
from sklearn.decomposition import FactorAnalysis
fa = FactorAnalysis(n_components=3, rotation='varimax')
fa.fit(X)
# Loadings are more interpretable after rotation
```

---

## Forecasting Metrics

### R² Ratio
**What it is:** Proportion of variance explained by common factors.

**Example:**
```python
pca = PCA(n_components=3).fit(X)
X_reconstructed = pca.inverse_transform(pca.transform(X))
r2 = 1 - np.var(X - X_reconstructed) / np.var(X)
```

---

### Prediction Error
**What it is:** Difference between forecast and actual: ε_t = y_t - ŷ_t

**Example:**
```python
forecast = model.predict(factors_test)
rmse = np.sqrt(np.mean((y_test - forecast)**2))
```

---

## Quick Reference Table

| Term | One-Liner | Code Hint |
|------|-----------|-----------|
| Extract factors | PCA on standardized data | `PCA(n_components=3).fit_transform(X)` |
| Kalman filter | Real-time factor estimates | `DynamicFactor().fit().filtered_state` |
| Loadings | Variable sensitivities to factors | `pca.components_.T` |
| Nowcast | Predict before official release | Bridge equation with factors |
| Ragged edge | Missing recent data | Kalman filter handles automatically |
| Scree plot | Choose number of factors | `plot(explained_variance_ratio_)` |
| FAVAR | VAR with factors + key variables | `VAR([factors, key_vars])` |
| EM algorithm | Estimate DFM with dynamics | `DynamicFactor().fit()` |

---

## Math Notation Reference

| Symbol | Meaning | Code Equivalent |
|--------|---------|-----------------|
| X_t | Data at time t | `data.iloc[t]` |
| F_t | Factors at time t | `factors[t]` |
| Λ | Loading matrix | `pca.components_.T` |
| e_t | Idiosyncratic errors | `X - X_reconstructed` |
| Φ | AR coefficient matrix | `res.transition[:, :, 0]` |
| r | Number of factors | `n_components=r` |
| N | Number of series | `X.shape[1]` |
| T | Number of time periods | `X.shape[0]` |

---

## When to Use: Quick Decision Tree

```
Many series (N > 50)?
  YES → Common drivers?
    YES → Different frequencies?
      YES → DFM with Kalman filter (this course!)
      NO → Static factor model (PCA)
    NO → Model individually
  NO → Use VAR or direct methods
```

---

## Common Pitfalls

| Pitfall | Why It Happens | Fix |
|---------|----------------|-----|
| Too many factors | Overfitting noise | Use IC criteria or scree plot |
| Non-stationary data | Trends break PCA | Transform first (difference, log) |
| Forgot to standardize | Different units dominate | `StandardScaler().fit_transform(X)` |
| Used revised data for backtest | Overestimates accuracy | Use real-time vintages |
| Mixed freq without care | Aggregation issues | MIDAS or state-space models |

---

## Essential Papers (Optional Reading)

1. **Stock & Watson (2002)** - Principal components approach
2. **Bai & Ng (2002)** - Determining number of factors
3. **Giannone et al. (2008)** - Nowcasting framework

Full citations in main course bibliography.

---

**Pro tip:** Bookmark this page - you'll reference it constantly while coding!

# Dynamic Factor Models - Troubleshooting Guide

Common errors, their causes, and solutions when working with DFMs.

---

## Convergence Issues

### Error: "Maximum iterations reached"

**Symptom:**
```python
WARNING: Maximum number of iterations has been exceeded.
```

**Cause:**
- Poor initial parameter values
- Model is overparameterized
- Data has numerical issues (scaling, multicollinearity)

**Solutions:**

1. **Reduce model complexity:**
```python
# Instead of:
model = DynamicFactor(data, k_factors=3, factor_order=4)

# Try:
model = DynamicFactor(data, k_factors=1, factor_order=2)
```

2. **Increase max iterations and use better optimizer:**
```python
results = model.fit(
    maxiter=2000,          # Increase from default 500
    method='lbfgs',        # Try different optimizer
    disp=True              # Monitor progress
)
```

3. **Standardize your data:**
```python
# ALWAYS standardize before DFM
data_std = (data - data.mean()) / data.std()
model = DynamicFactor(data_std, k_factors=2, factor_order=2)
```

4. **Use approximate diffuse initialization:**
```python
model = DynamicFactor(data, k_factors=2, factor_order=2)
model.initialize_approximate_diffuse()
results = model.fit()
```

---

## Singular Matrix Errors

### Error: "LinAlgError: Singular matrix"

**Symptom:**
```python
numpy.linalg.LinAlgError: Singular matrix
```

**Cause:**
- Perfect multicollinearity in data
- Too many factors for available series
- Numerical precision issues

**Solutions:**

1. **Check for perfect correlation:**
```python
# Find highly correlated series
correlation_matrix = data.corr()
high_corr = (correlation_matrix.abs() > 0.95) & (correlation_matrix != 1.0)

if high_corr.any().any():
    print("Highly correlated series found:")
    print(high_corr[high_corr].stack())
    # Remove one of the correlated series
```

2. **Reduce number of factors:**
```python
# Rule of thumb: k_factors < n_series / 3
n_series = data.shape[1]
max_factors = n_series // 3

model = DynamicFactor(data, k_factors=max_factors, factor_order=2)
```

3. **Add small noise for numerical stability:**
```python
# Add tiny jitter to break perfect collinearity
data_jittered = data + np.random.randn(*data.shape) * 1e-6
model = DynamicFactor(data_jittered, k_factors=2, factor_order=2)
```

4. **Use PCA preprocessing to remove collinearity:**
```python
from sklearn.decomposition import PCA

# Reduce dimensionality first
pca = PCA(n_components=5)  # Keep 5 principal components
data_pca = pca.fit_transform(data)
data_pca = pd.DataFrame(data_pca, index=data.index)

model = DynamicFactor(data_pca, k_factors=2, factor_order=2)
```

---

## Missing Data Handling

### Error: "ValueError: NaNs in data"

**Symptom:**
```python
ValueError: The model cannot be estimated with missing data
```

**Cause:**
- DynamicFactor can handle missing data, but initialization fails if too many missing values

**Solutions:**

1. **Interpolate missing values before estimation:**
```python
# Linear interpolation
data_filled = data.interpolate(method='linear', limit_direction='both')

# Check remaining missing
print(data_filled.isnull().sum())
```

2. **Use forward/backward fill for gaps:**
```python
# Fill short gaps only
data_filled = data.fillna(method='ffill', limit=3)
data_filled = data_filled.fillna(method='bfill', limit=3)
```

3. **Drop series with too many missing values:**
```python
# Keep only series with < 10% missing
threshold = len(data) * 0.1
data_clean = data.loc[:, data.isnull().sum() < threshold]

print(f"Kept {data_clean.shape[1]}/{data.shape[1]} series")
```

4. **Use robust estimation window:**
```python
# Start estimation after missing data period
first_complete = data.dropna().index[0]
data_subset = data.loc[first_complete:]

model = DynamicFactor(data_subset, k_factors=2, factor_order=2)
```

---

## Identification Problems

### Error: "Model is not identified"

**Symptom:**
- Estimates change dramatically with small data changes
- Very large standard errors
- Factors have no clear interpretation

**Cause:**
- Factor loadings not uniquely determined
- Need restrictions for identification

**Solutions:**

1. **Use factor rotation for identification:**
```python
# Estimate with default identification
results = model.fit()

# Extract factors and loadings
factors = results.factors.smoothed
loadings = extract_loadings(results)  # See common_patterns.py

# Apply varimax rotation
from sklearn.decomposition import FactorAnalysis
rotated_loadings = varimax_rotation(loadings)  # Custom function
```

2. **Fix first loading to 1.0:**
```python
# This is done automatically in statsmodels DynamicFactor
# First loading for each factor is normalized
# Check: results.params shows 'loading.f1.Series_1' is fixed
```

3. **Use fewer factors:**
```python
# If k_factors too large, identification issues arise
# Use information criteria to select
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

aics = []
for k in range(1, 5):
    model = DynamicFactor(data, k_factors=k, factor_order=2)
    results = model.fit(disp=False)
    aics.append(results.aic)

optimal_k = aics.index(min(aics)) + 1
print(f"Optimal factors: {optimal_k}")
```

4. **Add economic restrictions:**
```python
# Restrict signs of loadings based on economic theory
# Example: All loadings for "activity" factor should be positive
# (Requires custom implementation - see advanced topics)
```

---

## Numerical Instability

### Error: "RuntimeWarning: invalid value encountered"

**Symptom:**
```python
RuntimeWarning: invalid value encountered in divide
RuntimeWarning: overflow encountered in exp
```

**Cause:**
- Numerical overflow/underflow in Kalman filter
- Poorly scaled data
- Explosive factor dynamics

**Solutions:**

1. **Always standardize data:**
```python
# Z-score standardization
data_std = (data - data.mean()) / data.std()

# Verify scaling
assert data_std.mean().abs().max() < 1e-10
assert (data_std.std() - 1.0).abs().max() < 1e-10
```

2. **Enforce stationarity in factor dynamics:**
```python
model = DynamicFactor(
    data,
    k_factors=2,
    factor_order=2,
    enforce_stationarity=True  # Constrains AR parameters
)
```

3. **Check for outliers:**
```python
# Detect outliers (> 4 std devs)
outliers = (data.abs() > 4 * data.std())
print(f"Outliers found: {outliers.sum().sum()}")

# Winsorize outliers
from scipy.stats import mstats
data_winsorized = data.apply(
    lambda x: mstats.winsorize(x, limits=[0.01, 0.01])
)
```

4. **Use square-root Kalman filter:**
```python
# More numerically stable (built into statsmodels)
model = DynamicFactor(data, k_factors=2, factor_order=2)
model.ssm.filter_method = 'chandrasekhar'  # Alternative filter
results = model.fit()
```

---

## Forecast Errors

### Error: Forecasts are constant or nonsensical

**Symptom:**
- All forecasts are the same value
- Forecasts diverge to infinity
- Forecasts ignore recent data

**Cause:**
- Model not properly updated with recent data
- Non-stationary series not transformed
- Forecast horizon too long

**Solutions:**

1. **Ensure data is stationary:**
```python
# Check stationarity with ADF test
from statsmodels.tsa.stattools import adfuller

for col in data.columns:
    adf_stat, p_value, *_ = adfuller(data[col].dropna())
    if p_value > 0.05:
        print(f"{col} is non-stationary (p={p_value:.4f})")
        # Transform to differences
        data[col] = data[col].diff()

data = data.dropna()
```

2. **Use recent data for updating:**
```python
# Re-estimate on recent window for forecasting
recent_data = data.iloc[-120:]  # Last 10 years
model = DynamicFactor(recent_data, k_factors=2, factor_order=2)
results = model.fit()

# Now forecast
forecast = results.forecast(steps=12)
```

3. **Check forecast horizon is reasonable:**
```python
# Rule of thumb: forecast_horizon < estimation_window / 4
estimation_window = 120
max_horizon = estimation_window // 4

forecast = results.forecast(steps=max_horizon)
```

4. **Get prediction intervals:**
```python
# Use get_forecast for intervals
forecast_obj = results.get_forecast(steps=12)
forecast_summary = forecast_obj.summary_frame(alpha=0.05)

print(forecast_summary[['mean', 'mean_ci_lower', 'mean_ci_upper']])
```

---

## Memory and Performance Issues

### Error: "MemoryError" or very slow estimation

**Symptom:**
- Estimation takes hours
- System runs out of memory
- Kernel crashes

**Cause:**
- Too many series or too long time series
- Inefficient state space representation
- Large forecast horizons

**Solutions:**

1. **Reduce data size strategically:**
```python
# Use recent data only
data_recent = data.iloc[-240:]  # Last 20 years

# Subsample less frequent data if appropriate
data_quarterly = data.resample('Q').last()
```

2. **Reduce state space dimension:**
```python
# Fewer factors and lower AR order
model = DynamicFactor(
    data,
    k_factors=1,        # Instead of 3
    factor_order=1,     # Instead of 4
    error_order=0       # No AR errors
)
```

3. **Use sparse methods for many series:**
```python
# For N > 50 series, consider:
# 1. PCA pre-processing to reduce dimensionality
from sklearn.decomposition import PCA

pca = PCA(n_components=10)  # Reduce to 10 PCs
data_reduced = pca.fit_transform(data)

# 2. Estimate DFM on reduced data
model = DynamicFactor(data_reduced, k_factors=2, factor_order=2)
```

4. **Parallelize if doing multiple estimations:**
```python
from joblib import Parallel, delayed

def estimate_model(k_factors):
    model = DynamicFactor(data, k_factors=k_factors, factor_order=2)
    return model.fit(disp=False)

# Parallel estimation for model selection
results = Parallel(n_jobs=4)(
    delayed(estimate_model)(k) for k in range(1, 5)
)
```

---

## Parameter Interpretation Issues

### Issue: Parameters don't make economic sense

**Symptom:**
- Negative loadings where positive expected
- Factor is uncorrelated with key series
- AR coefficients suggest explosive process

**Cause:**
- Factor sign/scale indeterminacy
- Wrong model specification
- Data quality issues

**Solutions:**

1. **Check and flip factor signs:**
```python
# Extract factors
factors = results.factors.smoothed

# Check correlation with key series
correlation = data.corrwith(pd.Series(factors[:, 0], index=data.index))
print(correlation)

# Flip if necessary
if correlation.mean() < 0:
    factors[:, 0] = -factors[:, 0]
    # Flip corresponding loadings too
```

2. **Verify AR parameters are stable:**
```python
# Extract AR parameters
ar_params = results.params[results.model.param_names.index('L1.f1.f1'):
                          results.model.param_names.index('L1.f1.f1') + 2]

# Check eigenvalues
companion = build_companion_matrix(ar_params)
eigenvalues = np.linalg.eigvals(companion)

if np.any(np.abs(eigenvalues) >= 1):
    print("WARNING: Non-stationary factor dynamics!")
    print(f"Eigenvalues: {eigenvalues}")
```

3. **Compare with PCA factors:**
```python
from sklearn.decomposition import PCA

# PCA for comparison
pca = PCA(n_components=2)
pca_factors = pca.fit_transform(data)

# Correlation between DFM and PCA factors
corr = np.corrcoef(factors[:, 0], pca_factors[:, 0])[0, 1]
print(f"DFM-PCA correlation: {corr:.3f}")
# High correlation (>0.8) suggests factors are sensible
```

4. **Examine factor contributions:**
```python
# Variance decomposition
from recipes.common_patterns import variance_decomposition

decomp = variance_decomposition(results)
print(decomp)

# Each series should have reasonable factor loading
# (not all near zero or all very large)
```

---

## Quick Diagnostic Checklist

Before filing a bug report or asking for help, verify:

- [ ] Data is properly loaded with datetime index
- [ ] No missing values or handled appropriately
- [ ] Data is standardized (mean ≈ 0, std ≈ 1)
- [ ] Series are stationary (or transformed to be)
- [ ] Number of factors < number of series / 3
- [ ] Factor AR order is reasonable (1-4)
- [ ] Model converged (check `results.mle_retvals['converged']`)
- [ ] Parameters are stable (eigenvalues < 1)
- [ ] Loadings have expected signs

**Minimal working example for debugging:**

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# Generate simple synthetic data
np.random.seed(42)
T = 100
N = 3

# True factor
factor = np.random.randn(T)
for t in range(1, T):
    factor[t] = 0.7 * factor[t-1] + np.random.randn()

# Observed series
loadings = [1.0, 0.8, 0.6]
data = np.column_stack([
    loadings[i] * factor + np.random.randn(T) * 0.5
    for i in range(N)
])

df = pd.DataFrame(data, columns=['Y1', 'Y2', 'Y3'])
df = (df - df.mean()) / df.std()

# Estimate
model = DynamicFactor(df, k_factors=1, factor_order=1)
results = model.fit(disp=True)

print("\nEstimation successful!")
print(f"Log-likelihood: {results.llf:.2f}")
print(f"AIC: {results.aic:.2f}")
```

If this simple example fails, check your statsmodels installation:
```bash
pip install --upgrade statsmodels
```

---

## Getting Help

If problems persist:

1. **Check statsmodels documentation:**
   - https://www.statsmodels.org/stable/statespace.html

2. **Search existing issues:**
   - https://github.com/statsmodels/statsmodels/issues

3. **Provide minimal reproducible example:**
   - Include data shape, head(), and describe()
   - Show exact error message
   - List package versions: `pip list | grep statsmodels`

4. **Common version issues:**
   ```bash
   # Update to latest statsmodels
   pip install --upgrade statsmodels numpy pandas scipy
   ```

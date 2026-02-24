# Seasonality Analysis in Commodity Markets

## In Brief

Seasonality refers to predictable, recurring patterns in commodity prices tied to calendar periods (months, quarters, seasons). Identifying and modeling these patterns is crucial for accurate forecasting and distinguishing fundamental cycles from random noise.

> 💡 **Key Insight:** Many commodities exhibit strong seasonal patterns driven by predictable demand cycles (heating oil in winter, gasoline in summer) or supply cycles (agricultural harvest seasons). Failing to account for seasonality leads to systematic forecast errors and missed trading opportunities.

## Formal Definition

**Seasonal Component:**

A time series $y_t$ can be decomposed into:
$$y_t = T_t + S_t + C_t + I_t$$

Where:
- $T_t$: Trend (long-term movement)
- $S_t$: Seasonal component (periodic pattern)
- $C_t$: Cyclical component (non-fixed period fluctuations)
- $I_t$: Irregular/noise component

**Periodicity:**

A series is seasonal with period $m$ if:
$$\mathbb{E}[S_t] = \mathbb{E}[S_{t+m}] \quad \forall t$$

For monthly data: $m = 12$, for daily data with weekly patterns: $m = 7$

## Intuitive Explanation

### The Four Seasons of Natural Gas

Natural gas prices exhibit clear seasonality:

```
Price Pattern (Stylized):

High │     ╱╲           ╱╲
     │    ╱  ╲         ╱  ╲
     │   ╱    ╲       ╱    ╲
     │  ╱      ╲     ╱      ╲
Low  │─╱────────╲───╱────────╲─
     └──W─Sp─Su──F──W─Sp─Su──F──→ Time

W = Winter (heating demand)
Su = Summer (cooling demand - less than winter)
Sp/F = Shoulder seasons (low demand)
```

**Why?**
- **Winter (Dec-Feb):** Heating demand spikes → High prices
- **Summer (Jun-Aug):** Moderate cooling demand → Moderate prices
- **Spring/Fall:** Low demand → Low prices

This pattern repeats year after year, making it predictable.

### Agricultural Example: Corn Prices

Corn prices follow a **harvest cycle**:

- **Spring (Mar-May):** Planting season, old crop depleting → Prices rise
- **Summer (Jun-Aug):** Growing season, weather uncertainty → Volatility
- **Fall (Sep-Nov):** Harvest, new supply → Prices fall
- **Winter (Dec-Feb):** Post-harvest, inventory draw → Prices stabilize

**Implication:** A trader buying corn in July (pre-harvest) should expect lower prices in October (harvest) - this is NOT a forecasting error, it's seasonality!

## Mathematical Formulation

### Classical Seasonal Decomposition

**Additive Model:**
$$y_t = T_t + S_t + e_t$$

Use when seasonal variation is roughly constant over time.

**Multiplicative Model:**
$$y_t = T_t \times S_t \times e_t$$

Use when seasonal variation scales with the level of the series (common in commodities).

**Log transformation converts multiplicative → additive:**
$$\log(y_t) = \log(T_t) + \log(S_t) + \log(e_t)$$

### Seasonal Dummy Variables

For monthly data, define 11 dummies (omit one for reference):
$$y_t = \alpha + \sum_{j=1}^{11} \beta_j D_{jt} + \epsilon_t$$

Where $D_{jt} = 1$ if $t$ is in month $j$, else 0.

**Interpretation:** $\beta_j$ is the average deviation of month $j$ from the reference month.

### Fourier Representation

Represent seasonality as sum of sine/cosine waves:
$$S_t = \sum_{k=1}^K \left[ a_k \sin\left(\frac{2\pi k t}{m}\right) + b_k \cos\left(\frac{2\pi k t}{m}\right) \right]$$

**Advantages:**
- Smooth seasonal pattern
- Fewer parameters than dummies (choose $K \ll m$)
- Natural for continuous-time modeling

### STL Decomposition

**STL:** Seasonal and Trend decomposition using Loess (locally weighted regression)

**Algorithm:**
1. Smooth to estimate trend $T_t$
2. Detrend: $y_t - T_t$
3. Smooth seasonal: Average each calendar period
4. Iterate to refine

**Advantages:**
- Robust to outliers
- Handles changing seasonality
- No assumption of fixed pattern

## Code Implementation

### 1. Visual Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load data (example: natural gas prices)
# Assume df with 'date' and 'price' columns
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# Seasonal boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='month', y='price')
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('Natural Gas Price Seasonality (Boxplot by Month)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True, alpha=0.3)
plt.show()
```

### 2. Classical Decomposition

```python
# Ensure time series index
ts = df.set_index('date')['price']

# Decompose (additive or multiplicative)
decomposition = seasonal_decompose(ts, model='multiplicative', period=12)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Extract seasonal component
seasonal_pattern = decomposition.seasonal
```

### 3. Seasonal Dummies Regression

```python
import statsmodels.api as sm

# Create monthly dummies
dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)
df = pd.concat([df, dummies], axis=1)

# Regression
X = df[[c for c in df.columns if c.startswith('month_')]]
X = sm.add_constant(X)
y = df['price']

model = sm.OLS(y, X).fit()
print(model.summary())

# Seasonal coefficients
seasonal_effects = model.params.drop('const')
print("\nSeasonal Effects (relative to January):")
print(seasonal_effects)
```

### 4. Fourier Seasonal Model

```python
def fourier_terms(t, period, K):
    """
    Generate Fourier terms for seasonal modeling
    t: time index (e.g., 0, 1, 2, ...)
    period: seasonal period (e.g., 12 for monthly)
    K: number of Fourier pairs
    """
    terms = {}
    for k in range(1, K+1):
        terms[f'sin_{k}'] = np.sin(2 * np.pi * k * t / period)
        terms[f'cos_{k}'] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(terms)

# Generate Fourier features
df['time_index'] = range(len(df))
fourier_features = fourier_terms(df['time_index'], period=12, K=3)
df_fourier = pd.concat([df, fourier_features], axis=1)

# Fit model
X = df_fourier[[c for c in df_fourier.columns if c.startswith(('sin_', 'cos_'))]]
X = sm.add_constant(X)
y = df_fourier['price']

fourier_model = sm.OLS(y, X).fit()
print(fourier_model.summary())

# Predict seasonal component
seasonal_fourier = fourier_model.predict(X)
```

### 5. Bayesian Seasonal Model (PyMC)

```python
import pymc as pm
import arviz as az

# Prepare data
y_obs = df['price'].values
months = df['month'].values - 1  # 0-indexed for PyMC

with pm.Model() as seasonal_model:
    # Overall mean
    mu = pm.Normal('mu', mu=50, sigma=20)

    # Seasonal effects (sum-to-zero constraint)
    seasonal_raw = pm.Normal('seasonal_raw', mu=0, sigma=10, shape=12)
    seasonal = pm.Deterministic('seasonal', seasonal_raw - seasonal_raw.mean())

    # Observation noise
    sigma = pm.HalfNormal('sigma', sigma=10)

    # Likelihood
    y_pred = mu + seasonal[months]
    y_likelihood = pm.Normal('y_obs', mu=y_pred, sigma=sigma, observed=y_obs)

    # Sample
    trace = pm.sample(2000, return_inferencedata=True)

# Analyze seasonal pattern
seasonal_posterior = trace.posterior['seasonal'].mean(dim=['chain', 'draw'])
print("\nBayesian Seasonal Effects by Month:")
for i, effect in enumerate(seasonal_posterior):
    print(f"Month {i+1}: {effect:.2f}")

# Plot
az.plot_posterior(trace, var_names=['seasonal'])
plt.tight_layout()
plt.show()
```

## Visual Representation

```
Seasonal Decomposition Workflow:

Original Series (y_t)
        │
        ├─→ Estimate Trend (T_t)
        │    ↓
        ├─→ Detrend: y_t - T_t
        │    ↓
        ├─→ Extract Seasonal (S_t)
        │    ↓ (average each period)
        └─→ Residual: y_t - T_t - S_t


Example: Natural Gas Seasonality

Month │ Effect │ Interpretation
──────┼────────┼───────────────────────
Jan   │ +$2.50 │ Peak heating demand
Feb   │ +$1.80 │ Still cold
Mar   │ +$0.50 │ Spring shoulder
Apr   │ -$1.20 │ Low demand
May   │ -$1.50 │ Low demand
Jun   │ -$0.80 │ Summer AC starts
Jul   │ +$0.30 │ Moderate cooling
Aug   │ +$0.20 │ Moderate cooling
Sep   │ -$0.90 │ Fall shoulder
Oct   │ -$1.40 │ Low demand
Nov   │ +$0.60 │ Winter approaching
Dec   │ +$2.00 │ Heating demand rises
```

## Common Pitfalls

### 1. Ignoring Changing Seasonality

**Problem:** Seasonality can evolve over time (e.g., climate change affects heating demand)

**Example:**
```python
# BAD: Assume fixed seasonal pattern
seasonal_avg = df.groupby('month')['price'].mean()

# BETTER: Use rolling seasonal estimates or time-varying model
```

**Solution:** Use STL with time-varying smoothness or Bayesian model with evolving seasonal effects

### 2. Confusing Seasonality with Trend

**Problem:** Both can look like patterns, but seasonality is periodic

**Diagnostic:**
- Trend: Sustained directional movement
- Seasonality: Returns to same level after period $m$

**Test:**
```python
# Check for unit root (trend) vs stationary seasonality
from statsmodels.tsa.stattools import adfuller
adf_result = adfuller(df['price'])
print(f"ADF p-value: {adf_result[1]:.4f}")
# p < 0.05 → stationary (no trend), seasonality possible
```

### 3. Overfitting with Too Many Fourier Terms

**Problem:** Using $K$ too large captures noise, not true seasonality

**Solution:**
- Use AIC/BIC to select $K$
- Typically $K = 1$ or $2$ sufficient for annual seasonality
- Cross-validate forecast accuracy

### 4. Not Accounting for Trading Days

**Problem:** Different months have different numbers of days/trading days

**Fix:** Adjust for calendar effects
```python
df['trading_days'] = df.groupby('month')['date'].transform('count')
df['price_per_day'] = df['price'] / df['trading_days']
```

### 5. Using Wrong Decomposition Type

**Problem:** Multiplicative seasonality with additive model (or vice versa)

**Diagnostic:**
- If seasonal swings grow with level → Multiplicative
- If seasonal swings constant → Additive

**Test:**
```python
# Plot seasonal variance over time
df['seasonal_var'] = df.groupby('year')['price'].transform('std')
plt.plot(df.groupby('year')['price'].mean(), df.groupby('year')['seasonal_var'].mean(), 'o')
# If positive slope → multiplicative
```

## Connections

### Builds on:
- **Module 0 (Foundations):** Time series properties, stationarity
- **Module 1 (Bayesian Fundamentals):** Regression with categorical variables (seasonal dummies)

### Leads to:
- **Module 3 (State Space):** Seasonal state space models (BSM - Basic Structural Model)
- **Module 4 (Hierarchical):** Shared seasonal patterns across related commodities
- **Module 8 (Fundamentals):** Combining seasonal patterns with fundamental drivers

### Related to:
- **Calendar effects:** Holidays, day-of-week patterns
- **Regime switching:** Seasonal patterns may differ across regimes
- **Forecasting:** Seasonal adjustment improves accuracy

## Practice Problems

### 1. Identify Seasonality

You have 5 years of daily copper prices. How would you test for:
a) Annual seasonality
b) Day-of-week effects
c) Holiday effects

**Hints:**
- Use autocorrelation function (ACF) at seasonal lags
- Boxplots by month/day
- Formal tests: Friedman test, Kruskal-Wallis test

---

### 2. Decompose Agricultural Prices

Given corn prices (monthly, 2000-2023):
- Apply STL decomposition
- Interpret the seasonal pattern
- Compare to USDA planting/harvest calendar
- Does the pattern make economic sense?

---

### 3. Forecast with Seasonality

Build a forecasting model for gasoline prices that incorporates:
- Driving season (summer peak)
- Holiday travel (Thanksgiving, Christmas)
- Refinery maintenance (spring)

**Approach:**
- Use seasonal dummies or Fourier terms
- Add calendar variables (days to holiday, etc.)
- Validate on holdout period

---

### 4. Bayesian Seasonal Model

Implement a Bayesian hierarchical model where:
- Each year has its own seasonal pattern
- Seasonal patterns are drawn from a common distribution
- Test if seasonality has changed over time

**PyMC structure:**
```python
seasonal_year[year, month] ~ Normal(seasonal_mean[month], tau)
seasonal_mean[month] ~ Normal(0, sigma_seasonal)
```

## Further Reading

### Foundational Texts
- **Hyndman & Athanasopoulos (2021):** *Forecasting: Principles and Practice* - Chapter 3 on decomposition
- **Hamilton (1994):** *Time Series Analysis* - Chapter 8 on seasonal models
- **Harvey (1989):** *Forecasting, Structural Time Series Models* - Seasonal state space models

### Commodity-Specific Research
- **Pindyck & Rotemberg (1990):** "The Excess Co-Movement of Commodity Prices" - Seasonal vs non-seasonal components
- **Borovkova & Geman (2006):** "Seasonal and stochastic effects in commodity forward curves"

### Statistical Methods
- **Cleveland et al. (1990):** "STL: A Seasonal-Trend Decomposition Procedure Based on Loess"
- **Findley et al. (1998):** "New Capabilities and Methods of the X-12-ARIMA Seasonal-Adjustment Program"

### Software Documentation
- **statsmodels:** `seasonal_decompose`, X13-ARIMA-SEATS
- **PyMC:** Seasonal models examples
- **R packages:** `forecast::stl`, `seasonal::seas`

---

**Next:** Apply these methods to real commodity data in `03_seasonality_decomposition.ipynb`

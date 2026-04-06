# Fundamental Variables in Commodity Forecasting

> **Reading time:** ~11 min | **Module:** 8 — Fundamentals Integration | **Prerequisites:** Modules 1-7


## In Brief

Fundamental variables are economic, physical, and market factors that drive commodity prices through supply and demand mechanisms. These include inventory levels, production capacity, consumption patterns, weather conditions, and macroeconomic indicators. Integrating fundamentals into Bayesian models transforms price forecasting from pure time-series extrapolation to economically-grounded inference.

<div class="callout-insight">

<strong>Insight:</strong> Pure statistical models (ARIMA, GP) learn patterns from price history but ignore why prices move. Fundamental models incorporate the causal drivers: oil prices rise because inventories are low and demand is high, not just because they were rising yesterday. Bayesian frameworks excel at combining uncertain fundamental data with price history, yielding forecasts that are both statistically sound and economically interpretable.

</div>

## Formal Definition

### Fundamental-Augmented Price Model

**Standard time-series model:**
$$p_t = f(p_{t-1}, p_{t-2}, ...) + \epsilon_t$$

**Fundamental-augmented model:**
$$p_t = f(p_{t-1}, ..., \mathbf{X}_t, \mathbf{Z}_t) + \epsilon_t$$

Where:
- $\mathbf{X}_t$: Observable fundamentals (inventory, production, etc.)
- $\mathbf{Z}_t$: Latent fundamentals (supply/demand balance)

### Categories of Fundamental Variables

**1. Supply-Side Fundamentals**
- **Production:** $Q^S_t$ — current output (barrels/day, bushels/year)
- **Capacity utilization:** $U_t = Q^S_t / Q^{\max}_t$
- **Inventory:** $I_t$ — stocks available
- **Imports:** $M_t$ — incoming supply

**2. Demand-Side Fundamentals**
- **Consumption:** $Q^D_t$ — actual usage
- **Economic activity:** GDP growth, industrial production
- **Substitution:** Prices of alternatives
- **Exports:** $X_t$ — outgoing demand

**3. Market Structure**
- **Inventory ratio:** $\text{Cover} = I_t / Q^D_t$ (days of supply)
- **Convenience yield:** $y_t = r + s - (F_t - S_t)/S_t$
- **Basis:** $B_t = F_t - S_t$ (futures - spot)

**4. Exogenous Factors**
- **Weather:** Temperature deviations, precipitation
- **Geopolitical risk:** Conflict indicators, policy uncertainty
- **Currency:** Exchange rates affecting import/export

### Structural Relationship

**Theory of storage:**
$$F_t = S_t e^{(r + s - y)(T - t)}$$

Where:
- $F_t$: Futures price
- $S_t$: Spot price
- $r$: Risk-free rate
- $s$: Storage cost
- $y$: Convenience yield (function of inventory)

**Supply-demand equilibrium:**
$$p_t = p^* + \beta_1 (Q^D_t - Q^S_t) + \beta_2 I_t^{-1}$$

Low inventory → high price sensitivity

## Intuitive Explanation

Think of commodity prices like housing prices:
<div class="callout-insight">

<strong>Insight:</strong> Think of commodity prices like housing prices:

</div>


**Without fundamentals (pure time series):**
- "Prices rose 5% last quarter, so they'll rise 5% next quarter"
- Ignores: Are there jobs? Is housing supply tight?

**With fundamentals:**
- Population growth (demand) ↑
- New construction (supply) ↓
- Mortgage rates (cost) ↑
- → Prices likely to rise

For crude oil:

**Fundamental indicators:**
1. **OECD inventory** (days of forward cover)
   - Above 5-year average → bearish (surplus)
   - Below 5-year average → bullish (deficit)

2. **U.S. production** (million barrels/day)
   - Increasing → bearish (more supply)
   - Decreasing → bullish (less supply)

3. **China GDP growth**
   - Above 6% → bullish (strong demand)
   - Below 4% → bearish (weak demand)

4. **Crack spreads** (refining margins)
   - High spreads → bullish (strong demand for crude)
   - Low spreads → bearish (weak demand)

Bayesian model learns:
- Historical relationship between fundamentals and prices
- Uncertainty in each fundamental variable
- How to weight conflicting signals
- Non-linear relationships (e.g., inventory effect stronger when stocks very low)

## Code Implementation

### Fundamental Variable Selection


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class FundamentalVariables:
    """
    Container for fundamental variables with metadata.
    """

    def __init__(self):
        self.variables = {}

    def add_variable(self, name, data, category, unit, description):
        """
        Add fundamental variable.

        Args:
            name: Variable identifier
            data: Time series (pandas Series with datetime index)
            category: 'supply' | 'demand' | 'market' | 'exogenous'
            unit: Measurement unit
            description: Human-readable description
        """
        self.variables[name] = {
            'data': data,
            'category': category,
            'unit': unit,
            'description': description,
            'mean': data.mean(),
            'std': data.std()
        }

    def get_matrix(self, normalize=True):
        """
        Get design matrix of all fundamental variables.

        Returns:
            X: [T, K] array of fundamental variables
            names: List of variable names
        """
        names = list(self.variables.keys())
        data_list = [self.variables[name]['data'].values for name in names]

        X = np.column_stack(data_list)

        if normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        return X, names

    def summary(self):
        """
        Print summary of fundamental variables.
        """
        print("=" * 80)
        print("FUNDAMENTAL VARIABLES SUMMARY")
        print("=" * 80)

        for name, info in self.variables.items():
            print(f"\n{name.upper()}")
            print(f"  Category: {info['category']}")
            print(f"  Unit: {info['unit']}")
            print(f"  Description: {info['description']}")
            print(f"  Mean: {info['mean']:.2f}, Std: {info['std']:.2f}")
            print(f"  Data points: {len(info['data'])}")


# Example: Crude oil fundamental variables
fundamentals = FundamentalVariables()

# Generate synthetic fundamental data
np.random.seed(42)
dates = pd.date_range('2018-01-01', periods=250, freq='W')

# Supply side
production = pd.Series(
    13.0 + np.cumsum(np.random.normal(0, 0.05, 250)),  # Million bpd
    index=dates
)
fundamentals.add_variable(
    'production',
    production,
    'supply',
    'million bpd',
    'U.S. crude oil production'
)

inventory = pd.Series(
    450 + np.cumsum(np.random.normal(0, 5, 250)),  # Million barrels
    index=dates
)
fundamentals.add_variable(
    'inventory',
    inventory,
    'supply',
    'million barrels',
    'U.S. commercial crude inventory'
)

# Demand side
refinery_utilization = pd.Series(
    85 + 5 * np.sin(np.arange(250) * 2 * np.pi / 52) + np.random.normal(0, 2, 250),
    index=dates
)
fundamentals.add_variable(
    'refinery_util',
    refinery_utilization,
    'demand',
    'percent',
    'U.S. refinery utilization rate'
)

# Market structure
days_of_cover = inventory / (production * 7)  # Days of supply
fundamentals.add_variable(
    'days_cover',
    days_of_cover,
    'market',
    'days',
    'Days of forward cover (inventory/production)'
)

# Exogenous
gdp_growth = pd.Series(
    2.5 + np.random.normal(0, 0.5, 250),  # Percent
    index=dates
)
fundamentals.add_variable(
    'gdp_growth',
    gdp_growth,
    'exogenous',
    'percent',
    'U.S. GDP growth rate'
)

# Print summary
fundamentals.summary()
```

</div>
</div>

### Bayesian Linear Model with Fundamentals


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def fit_fundamental_model(prices, fundamentals):
    """
    Bayesian linear regression with fundamental variables.

    Model: price_t = alpha + beta * fundamentals_t + epsilon_t

    Args:
        prices: Price series (log returns or levels)
        fundamentals: FundamentalVariables object

    Returns:
        model: PyMC model
        trace: MCMC samples
    """
    # Get design matrix
    X, names = fundamentals.get_matrix(normalize=True)

    # Align prices and fundamentals
    aligned_prices = prices.loc[fundamentals.variables[names[0]]['data'].index]

    n, k = X.shape

    with pm.Model() as model:
        # Priors on regression coefficients
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=2, shape=k)

        # Prior on noise
        sigma = pm.HalfNormal('sigma', sigma=5)

        # Linear predictor
        mu = alpha + pm.math.dot(X, beta)

        # Likelihood
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=aligned_prices.values)

        # Sample
        trace = pm.sample(2000, tune=1000, return_inferencedata=True, random_seed=42)

    return model, trace, names


# Generate synthetic price data correlated with fundamentals
X, names = fundamentals.get_matrix(normalize=True)
true_beta = np.array([2.0, -1.5, 1.0, -2.5, 0.5])  # True relationships
prices = pd.Series(
    50 + X @ true_beta + np.random.normal(0, 2, len(X)),
    index=fundamentals.variables['production'].data.index
)

# Fit model
model, trace, var_names = fit_fundamental_model(prices, fundamentals)

# Analyze results
import arviz as az

print("\nPosterior Summary:")
print(az.summary(trace, var_names=['alpha', 'beta', 'sigma']))

# Plot coefficient estimates
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Coefficient estimates with uncertainty
az.plot_forest(trace, var_names=['beta'], combined=True, ax=axes[0])
axes[0].set_yticklabels(var_names)
axes[0].set_xlabel('Coefficient Value')
axes[0].set_title('Fundamental Variable Coefficients (95% HDI)')
axes[0].axvline(0, color='red', linestyle='--', linewidth=1)

# Plot 2: True vs estimated
beta_post_mean = trace.posterior['beta'].mean(dim=['chain', 'draw']).values
axes[1].scatter(true_beta, beta_post_mean, s=100, alpha=0.6)
axes[1].plot([-3, 3], [-3, 3], 'r--', linewidth=2, label='Perfect fit')

for i, name in enumerate(var_names):
    axes[1].annotate(name, (true_beta[i], beta_post_mean[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)

axes[1].set_xlabel('True Coefficient')
axes[1].set_ylabel('Estimated Coefficient')
axes[1].set_title('Coefficient Recovery')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('fundamental_coefficients.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>
</div>

### Non-Linear Fundamental Effects


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def fit_nonlinear_fundamental_model(prices, fundamentals):
    """
    Non-linear model: inventory effect stronger when stocks low.

    Model: price_t = f(inventory_t) + g(other fundamentals) + epsilon

    where f(inventory) = beta_inv * exp(-gamma * inventory)
    """
    X, names = fundamentals.get_matrix(normalize=False)

    # Extract inventory (assume it's first variable)
    inventory_raw = fundamentals.variables['inventory']['data'].values
    other_X = X[:, 1:]  # Other fundamentals
    other_names = names[1:]

    # Normalize other variables
    scaler = StandardScaler()
    other_X_norm = scaler.fit_transform(other_X)

    aligned_prices = prices.loc[fundamentals.variables['production'].data.index]

    with pm.Model() as model:
        # Non-linear inventory effect
        beta_inv = pm.Normal('beta_inv', mu=0, sigma=5)
        gamma = pm.HalfNormal('gamma', sigma=0.01)  # Decay rate

        inventory_effect = beta_inv * pm.math.exp(-gamma * inventory_raw)

        # Linear effects of other fundamentals
        beta_other = pm.Normal('beta_other', mu=0, sigma=2,
                               shape=other_X_norm.shape[1])
        other_effect = pm.math.dot(other_X_norm, beta_other)

        # Total effect
        alpha = pm.Normal('alpha', mu=50, sigma=10)
        mu = alpha + inventory_effect + other_effect

        # Likelihood
        sigma = pm.HalfNormal('sigma', sigma=5)
        y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma,
                          observed=aligned_prices.values)

        # Sample
        trace = pm.sample(2000, tune=1000, target_accept=0.95,
                         return_inferencedata=True, random_seed=42)

    return model, trace


# Fit non-linear model
model_nl, trace_nl = fit_nonlinear_fundamental_model(prices, fundamentals)

# Visualize non-linear inventory effect
inventory_range = np.linspace(inventory.min(), inventory.max(), 100)

beta_inv_post = trace_nl.posterior['beta_inv'].values.flatten()
gamma_post = trace_nl.posterior['gamma'].values.flatten()

# Sample from posterior
n_samples = 500
effect_samples = []
for i in np.random.choice(len(beta_inv_post), n_samples):
    effect = beta_inv_post[i] * np.exp(-gamma_post[i] * inventory_range)
    effect_samples.append(effect)

effect_samples = np.array(effect_samples)
effect_mean = effect_samples.mean(axis=0)
effect_lower = np.percentile(effect_samples, 2.5, axis=0)
effect_upper = np.percentile(effect_samples, 97.5, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(inventory_range, effect_mean, 'b-', linewidth=2, label='Mean effect')
plt.fill_between(inventory_range, effect_lower, effect_upper,
                 alpha=0.3, label='95% CI')
plt.xlabel('Inventory (million barrels)')
plt.ylabel('Price Effect ($)')
plt.title('Non-Linear Inventory Effect on Price')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig('nonlinear_inventory_effect.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>
</div>

### Variable Importance via Posterior Predictive


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def compute_variable_importance(trace, X, var_names):
    """
    Compute variable importance by predictive impact.

    Importance = Variance explained by each variable.
    """
    # Get posterior samples
    beta_samples = trace.posterior['beta'].values.reshape(-1, X.shape[1])

    # Compute contribution of each variable
    contributions = []
    for i in range(X.shape[1]):
        # Contribution = beta_i * X_i
        contrib = beta_samples[:, i:i+1] * X[:, i:i+1].T
        # Variance of contribution
        var_contrib = np.var(contrib)
        contributions.append(var_contrib)

    contributions = np.array(contributions)
    importance = contributions / contributions.sum()

    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]

    return importance, sorted_idx


X, names = fundamentals.get_matrix(normalize=True)
importance, sorted_idx = compute_variable_importance(trace, X, names)

# Plot variable importance
plt.figure(figsize=(10, 6))
plt.barh(range(len(names)), importance[sorted_idx])
plt.yticks(range(len(names)), [names[i] for i in sorted_idx])
plt.xlabel('Importance (Fraction of Variance Explained)')
plt.title('Fundamental Variable Importance')
plt.grid(alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('variable_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nVariable Importance Ranking:")
for i, idx in enumerate(sorted_idx):
    print(f"{i+1}. {names[idx]}: {importance[idx]:.3f}")
```


## Common Pitfalls

**1. Data Quality Issues**
- **Problem:** Fundamental data revised, missing, or low-frequency
- **Symptom:** Spurious relationships, unstable coefficients
- **Solution:** Use revised data, impute missing values, match frequencies

**2. Multicollinearity**
- **Problem:** Production and inventory highly correlated
- **Symptom:** Large coefficient uncertainty, sign flips
- **Solution:** PCA, drop redundant variables, regularization (horseshoe prior)

**3. Ignoring Lags**
- **Problem:** Using contemporaneous fundamentals (inventory today → price today)
- **Symptom:** Look-ahead bias, poor out-of-sample forecasts
- **Solution:** Use lagged fundamentals (inventory last week → price this week)

**4. Linear Assumptions**
- **Problem:** Assuming linear effects when relationships non-linear
- **Symptom:** Poor fit during extreme events (very low inventory)
- **Solution:** Transform variables (log, inverse), splines, GPs

**5. Missing Fundamental Variables**
- **Problem:** Omitting key drivers (e.g., weather for agriculture)
- **Symptom:** Large unexplained variance
- **Solution:** Domain expertise, literature review, iterative addition

## Connections

**Builds on:**
- Module 8.1: Storage theory (inventory → convenience yield → price)
- Module 5: GP models (non-linear fundamental relationships)
- Economic theory (supply/demand equilibrium)

**Leads to:**
- Module 8.3: Bayesian model averaging (combining fundamental models)
- Module 8.4: Forecast evaluation (fundamental vs technical forecasts)
- Trading strategies (fundamental-based signals)

**Related fields:**
- Econometrics (cointegration, error correction models)
- Structural VAR (impulse responses to fundamental shocks)
- Machine learning (feature engineering for fundamentals)

## Practice Problems

1. **Inventory Effect**
   Oil inventory: 450 million barrels (5-year avg: 420)
   Coefficient: β_inventory = -0.05

   What's the estimated price impact of excess inventory?

2. **Variable Selection**
   You have 20 potential fundamental variables
   - Use horseshoe prior: τ ~ HalfCauchy(0, 1)
   - After fitting, 5 variables have β_i with 95% HDI excluding zero

   How many variables are "important"? How to decide?

3. **Non-Linear Inventory**
   Model: price = 60 - 100 * exp(-0.002 * inventory)

   - When inventory = 200, what's the price effect?
   - When inventory = 600, what's the price effect?
   - At what inventory level does effect drop below $5?

4. **Forecast Construction**
   Fundamental model: price_t = 50 + 2*production_t - 1*inventory_t
   Current: production = 13, inventory = 450
   Forecast: production rises to 14, inventory falls to 420

   What's the predicted price change?

5. **Causal Inference**
   You observe: β_gdp = 2.0 (higher GDP → higher oil price)

   Does this prove GDP growth causes oil prices to rise?
   What confounders might exist?
   How would you test causality?


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving fundamental variables in commodity forecasting, what would be your first three steps to apply the techniques from this guide?




## Further Reading

**Economic Theory:**
1. **Kaldor (1939)** - "Speculation and Economic Stability" - Theory of storage
2. **Pindyck (2001)** - "The Dynamics of Commodity Spot and Futures Markets" - Storage model
3. **Hamilton (2009)** - "Causes and Consequences of the Oil Shock of 2007-08" - Oil fundamentals

**Empirical Studies:**
4. **Kilian (2009)** - "Not All Oil Price Shocks Are Alike" - Supply vs demand shocks
5. **Baumeister & Kilian (2016)** - "Forty Years of Oil Price Fluctuations" - Structural drivers

**Bayesian Methods:**
6. **Giannone et al. (2015)** - "Prior Selection for Vector Autoregressions" - Fundamental VARs
7. **Carriero et al. (2019)** - "Large Bayesian VARs for Commodity Markets" - High-dimensional fundamentals

**Data Sources:**
8. **EIA (Energy Information Administration)** - U.S. energy fundamental data
9. **USDA (U.S. Dept. of Agriculture)** - Agricultural supply/demand
10. **World Bank Commodity Markets** - Global fundamentals


<div class="callout-key">

<strong>Key Concept Summary:</strong> Fundamental variables are economic, physical, and market factors that drive commodity prices through supply and demand mechanisms.


---

*"Fundamentals explain why prices move. Integrating them transforms forecasting from pattern recognition to economic reasoning."*

---



## Cross-References

<a class="link-card" href="./02_fundamental_variables_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_storage_theory.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

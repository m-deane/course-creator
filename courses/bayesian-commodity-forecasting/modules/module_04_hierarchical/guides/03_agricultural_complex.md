# Hierarchical Models for Agricultural Commodities

> **Reading time:** ~11 min | **Module:** 4 — Hierarchical Models | **Prerequisites:** Module 3 State-Space Models


## In Brief

Agricultural markets form interconnected networks through crop rotation, feed demand, and land allocation decisions. Hierarchical models capture these structural relationships—pooling information across crops, regions, and time scales to improve forecasts while respecting biological and economic constraints.

<div class="callout-insight">

<strong>Insight:</strong> **Corn doesn't exist in isolation.** When corn prices spike, farmers plant more corn (less soybeans). When corn is cheap, livestock producers expand herds (increasing feed demand). A hierarchical model for the agricultural complex jointly models these substitution and complementarity relationships, preventing forecast inconsistencies.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> Agricultural markets form interconnected networks through crop rotation, feed demand, and land allocation decisions.

</div>

---

## Formal Definition

### Three-Level Agricultural Hierarchy

**Level 1: Global Agricultural Market Factor**
$$\mu_{\text{ag}} \sim \mathcal{N}(m_0, s_0^2)$$
$$\sigma_{\text{ag}} \sim \text{HalfNormal}(\tau_0)$$

**Level 2: Crop Category Factors**
$$\mu_{\text{category}} \sim \mathcal{N}(\mu_{\text{ag}}, \sigma_{\text{ag}}^2), \quad \text{category} \in \{\text{grains}, \text{oilseeds}, \text{softs}\}$$

**Level 3: Individual Crop Prices**
$$y_{c,t} \sim \mathcal{N}(\alpha_c + \beta_c \cdot \mu_{\text{category}} + \gamma_c \cdot X_{c,t}, \sigma_c^2)$$

Where:
- $\alpha_c$: Crop-specific intercept (base price level)
- $\beta_c$: Loading on category factor (correlation strength)
- $\gamma_c$: Fundamentals coefficient (weather, stocks, demand)
- $X_{c,t}$: Observable fundamentals (stocks-to-use ratio, yield forecasts)

---

## Intuitive Explanation

### The Agricultural Pyramid

```
Global Ag Factor (food demand, energy prices, USD)
         │
         ├── Grains Factor
         │        ├── Corn (feed, ethanol)
         │        ├── Wheat (food)
         │        └── Rice (food)
         │
         ├── Oilseeds Factor
         │        ├── Soybeans (feed, oil)
         │        ├── Canola/Rapeseed
         │        └── Palm Oil
         │
         └── Softs Factor
                  ├── Sugar (food, ethanol)
                  ├── Cotton (textile)
                  └── Coffee (beverage)
```

**Information propagation:**
1. Drought in Argentina → Soybeans spike → Oilseeds factor shifts → Canola forecast adjusts
2. Weak Chinese demand → Global Ag factor drops → All categories weaken
3. Good US corn harvest → Corn drops → Grains factor weakens → Wheat forecast adjusts

---

## Why Hierarchical Models for Agriculture?

### 1. Crop Rotation & Land Competition
<div class="callout-warning">

<strong>Warning:</strong> Farmers choose what to plant based on relative profitability.

</div>


Farmers choose what to plant based on relative profitability.

**Planting competition:**
- Corn-Soybean ratio drives US planting decisions
- High corn/soy ratio → More corn acres → Lower corn prices next year

**Hierarchical structure captures this:**
- If corn forecast is too high relative to soybeans, partial pooling pulls it down
- Forecasts respect economic equilibrium (farmers won't plant 100% corn)

---

### 2. Regional Production Patterns

**US:** Corn belt (IA, IL, IN), wheat plains (KS, ND, MT)
**South America:** Brazil (soybeans), Argentina (corn, wheat)
**Black Sea:** Ukraine/Russia (wheat, corn)

**Challenges:**
- Emerging markets have short price history
- Weather shocks affect multiple crops in same region
- Transportation costs create regional basis differentials

**Solution:** Regional hierarchies borrow strength from established markets.

---

### 3. Stocks-to-Use Relationships

**Carry-out ratio:** Ending stocks / Annual consumption

| Ratio | Price Implication |
|-------|------------------|
| < 15% | Tight supply → High volatility |
| 15-25% | Comfortable → Moderate volatility |
| > 25% | Oversupply → Low prices, low volatility |

**Hierarchical approach:** Pool information about stock-price relationships across crops.

---

### 4. Seasonal Patterns

**Northern Hemisphere:**
- Planting: April-May (uncertainty increases)
- Growing: June-August (weather matters most)
- Harvest: September-November (supply revealed, prices drop)

**Southern Hemisphere:** Opposite seasons (Brazil corn harvest Feb-March)

**Result:** Global supply is staggered. Hierarchical models capture how supply in one region affects global markets.

---

## Code Implementation

### Basic Grain Complex Hierarchy


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# Simulated data: Corn, Wheat, Soybeans over 120 months
np.random.seed(42)
n_months = 120
n_crops = 3
crop_names = ['Corn', 'Wheat', 'Soybeans']

# True global ag factor (food demand, energy prices)
global_ag = np.cumsum(np.random.normal(0, 0.5, n_months)) + 4.5  # $/bushel baseline

# Category: All are grains/oilseeds (simplified)

# Crop-specific parameters
crop_intercepts = np.array([0.0, 1.5, 5.0])  # Wheat premium, Soybean premium
crop_loadings = np.array([1.0, 0.85, 0.90])  # Corn tracks perfectly, others less
crop_noise = np.array([0.3, 0.4, 0.5])

# Generate prices
prices = np.zeros((n_months, n_crops))
for c in range(n_crops):
    prices[:, c] = (crop_intercepts[c] +
                    crop_loadings[c] * global_ag +
                    np.random.normal(0, crop_noise[c], n_months))

# Add harvest seasonality (prices drop in Sep-Nov for NH crops)
months_idx = np.arange(n_months) % 12
harvest_effect = -0.5 * (months_idx == 9) - 0.8 * (months_idx == 10) - 0.5 * (months_idx == 11)
prices[:, 0] += harvest_effect  # Corn harvest effect
prices[:, 1] += harvest_effect * 0.6  # Wheat (smaller effect)

# Build hierarchical model
with pm.Model() as grain_hierarchy:
    # Hyperpriors (global ag market)
    mu_global = pm.Normal('mu_global', mu=5, sigma=2)
    sigma_global = pm.HalfNormal('sigma_global', sigma=1)

    # Crop-level priors (partial pooling)
    crop_intercept = pm.Normal('crop_intercept',
                               mu=0,
                               sigma=3,
                               shape=n_crops)

    crop_loading = pm.Normal('crop_loading',
                            mu=1,
                            sigma=0.3,
                            shape=n_crops)

    crop_sigma = pm.HalfNormal('crop_sigma',
                              sigma=sigma_global,
                              shape=n_crops)

    # Global ag factor (random walk with drift)
    factor_drift = pm.Normal('factor_drift', mu=0, sigma=0.1)
    factor_innov = pm.Normal('factor_innov', mu=0, sigma=1, shape=n_months-1)

    factor = pm.Deterministic(
        'factor',
        pm.math.concatenate([
            [mu_global],
            mu_global + pm.math.cumsum(factor_drift + sigma_global * factor_innov)
        ])
    )

    # Seasonal effects (harvest lows)
    seasonal_effect = pm.Normal('seasonal_effect',
                               mu=0,
                               sigma=0.5,
                               shape=(n_crops, 12))

    # Observations
    for c in range(n_crops):
        seasonal = seasonal_effect[c, months_idx]

        pm.Normal(f'price_{crop_names[c]}',
                 mu=crop_intercept[c] + crop_loading[c] * factor + seasonal,
                 sigma=crop_sigma[c],
                 observed=prices[:, c])

    # Sample
    trace = pm.sample(1000, tune=2000,
                     target_accept=0.9,
                     return_inferencedata=True)

# Results
print(az.summary(trace, var_names=['crop_intercept', 'crop_loading', 'crop_sigma']))

# Check harvest season effect
harvest_seasonal = trace.posterior['seasonal_effect'].mean(dim=['chain', 'draw'])
print("\nHarvest months (Sep=9, Oct=10, Nov=11) effects:")
for c, crop in enumerate(crop_names):
    print(f"{crop}: Sep={harvest_seasonal[c, 9]:.2f}, Oct={harvest_seasonal[c, 10]:.2f}, Nov={harvest_seasonal[c, 11]:.2f}")
```

</div>
</div>

---

## Corn-Soybean Land Competition Model

### Economic Constraint: Farmer Planting Decisions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Data: corn and soybean prices, planted acres

# Farmers maximize: π = P_corn * Yield_corn * Acres_corn + P_soy * Yield_soy * Acres_soy

# Subject to: Acres_corn + Acres_soy = Total_acres

with pm.Model() as corn_soy_competition:
    n_years = 20

    # Historical data (simulated)
    corn_soy_ratio = prices[:n_years, 0] / prices[:n_years, 2]  # Corn price / Soy price
    corn_acres_pct = 0.5 + 0.2 * (corn_soy_ratio - corn_soy_ratio.mean())  # More corn when ratio high

    # Level 1: Profitability incentive
    profit_signal = pm.Normal('profit_signal', mu=0, sigma=1, shape=n_years)

    # Level 2: Corn vs. Soy prices respond to planting decisions
    # If too much corn planted → Supply increases → Price drops
    planted_corn_pct = pm.Deterministic(
        'planted_corn_pct',
        pm.math.invlogit(profit_signal)  # Bounded [0, 1]
    )

    # Price dynamics: Prices adjust to expected supply
    base_corn_price = pm.Normal('base_corn_price', mu=4.0, sigma=0.5)
    base_soy_price = pm.Normal('base_soy_price', mu=10.0, sigma=1.0)

    # Supply pressure coefficient (more acres → lower price)
    supply_pressure_corn = pm.HalfNormal('supply_pressure_corn', sigma=2)
    supply_pressure_soy = pm.HalfNormal('supply_pressure_soy', sigma=2)

    # Prices adjust for expected supply
    corn_price = pm.Deterministic(
        'corn_price',
        base_corn_price - supply_pressure_corn * (planted_corn_pct - 0.5)
    )

    soy_price = pm.Deterministic(
        'soy_price',
        base_soy_price - supply_pressure_soy * ((1 - planted_corn_pct) - 0.5)
    )

    # Observations (actual prices)
    pm.Normal('obs_corn', mu=corn_price, sigma=0.3, observed=prices[:n_years, 0])
    pm.Normal('obs_soy', mu=soy_price, sigma=0.5, observed=prices[:n_years, 2])

    trace_competition = pm.sample(1000, tune=1500,
                                  target_accept=0.9,
                                  return_inferencedata=True)

# Forecast: What happens if soy prices spike?

# Model predicts farmers shift to soybeans → corn prices rise (less supply)
```

</div>
</div>

---

## Regional Hierarchy: US vs. South America

### Brazil and Argentina Production


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with pm.Model() as regional_soy:
    n_obs = 100
    regions = ['US', 'Brazil', 'Argentina']
    n_regions = len(regions)

    # Level 1: Global soybean market
    global_soy_mu = pm.Normal('global_soy_mu', mu=10, sigma=2)
    global_soy_sigma = pm.HalfNormal('global_soy_sigma', sigma=2)

    global_factor = pm.GaussianRandomWalk('global_factor',
                                         mu=global_soy_mu,
                                         sigma=global_soy_sigma,
                                         shape=n_obs)

    # Level 2: Regional basis (transport costs, local supply/demand)
    # Brazil exports → Prices track global + freight to China
    # US exports → Prices track global + Gulf freight
    # Argentina → Export tax makes local prices lower

    region_basis = pm.Normal('region_basis',
                            mu=0,
                            sigma=1,
                            shape=n_regions)

    region_sigma = pm.HalfNormal('region_sigma',
                                 sigma=global_soy_sigma,
                                 shape=n_regions)

    # Seasonal harvest timing (opposite hemispheres)
    # US harvest: Sep-Nov (months 9-11)
    # Brazil harvest: Feb-Apr (months 2-4)
    # Argentina harvest: Apr-Jun (months 4-6)

    harvest_timing = {
        'US': [9, 10, 11],
        'Brazil': [2, 3, 4],
        'Argentina': [4, 5, 6]
    }

    seasonal_harvest = pm.Normal('seasonal_harvest',
                                mu=-0.5,  # Prices drop during harvest
                                sigma=0.2,
                                shape=n_regions)

    months_idx = np.arange(n_obs) % 12

    # Regional prices
    for i, region in enumerate(regions):
        # Seasonal effect (harvest months)
        seasonal = pm.math.sum([
            seasonal_harvest[i] * (months_idx == month)
            for month in harvest_timing[region]
        ], axis=0)

        regional_price = global_factor + region_basis[i] + seasonal

        pm.Normal(f'price_{region}',
                 mu=regional_price,
                 sigma=region_sigma[i],
                 observed=soy_prices[:, i])

    trace_regional = pm.sample(1000, tune=2000,
                              target_accept=0.9,
                              return_inferencedata=True)
```

</div>
</div>

**Result:** Model captures:
1. Global soybean demand (China imports)
2. Regional basis (Brazil discount to US due to export taxes)
3. Opposite-hemisphere harvest timing (Brazil harvest fills gap when US stocks low)

---

## Incorporating Fundamentals: Stocks-to-Use Ratio

### Nonlinear Price-Stock Relationship


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with pm.Model() as stocks_to_use:
    n_obs = 60
    stocks_to_use_ratio = np.random.uniform(0.1, 0.35, n_obs)  # 10-35%

    # Price is nonlinear function of stocks
    # Low stocks → High prices (convex relationship)

    # Parameters for convenience yield function
    # P(S) = a + b / S  (Pindyck & Rotemberg "working's curve")

    alpha = pm.Normal('alpha', mu=3, sigma=0.5)  # Base price
    beta = pm.HalfNormal('beta', sigma=5)  # Scarcity premium

    # Convenience yield model
    expected_price = pm.Deterministic(
        'expected_price',
        alpha + beta / stocks_to_use_ratio
    )

    # Volatility also depends on stocks (low stocks → high volatility)
    sigma_base = pm.HalfNormal('sigma_base', sigma=0.5)
    vol_stocks_coef = pm.HalfNormal('vol_stocks_coef', sigma=2)

    sigma_t = pm.Deterministic(
        'sigma_t',
        sigma_base + vol_stocks_coef / stocks_to_use_ratio
    )

    # Observations
    pm.Normal('price_obs',
             mu=expected_price,
             sigma=sigma_t,
             observed=corn_prices[:n_obs])

    trace_stocks = pm.sample(1000, tune=1500,
                            target_accept=0.9,
                            return_inferencedata=True)

# Interpretation: When stocks-to-use drops below 15%, prices become explosive
```

</div>
</div>

---

## Multi-Crop Forecast Consistency

### Constraint: Corn-Soybean Ratio Bounds

Economic theory: Corn/Soy price ratio should stay in reasonable range.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with pm.Model() as ratio_constrained:
    # Individual crop models
    corn_factor = pm.GaussianRandomWalk('corn_factor', sigma=0.3, shape=n_forecast)
    soy_factor = pm.GaussianRandomWalk('soy_factor', sigma=0.5, shape=n_forecast)

    # Base prices
    corn_base = pm.Normal('corn_base', mu=4.5, sigma=0.5)
    soy_base = pm.Normal('soy_base', mu=11.0, sigma=1.0)

    # Prices
    corn_price = pm.Deterministic('corn_price', corn_base + corn_factor)
    soy_price = pm.Deterministic('soy_price', soy_base + soy_factor)

    # Constraint: Ratio should be in [0.3, 0.5] (typical range)
    ratio = pm.Deterministic('ratio', corn_price / soy_price)

    # Soft constraint (penalize extreme ratios)
    pm.Potential('ratio_bound',
                pm.math.switch(ratio < 0.25, -10 * (0.25 - ratio)**2, 0) +
                pm.math.switch(ratio > 0.55, -10 * (ratio - 0.55)**2, 0))

    # Observations
    pm.Normal('corn_obs', mu=corn_price, sigma=0.2, observed=corn_data)
    pm.Normal('soy_obs', mu=soy_price, sigma=0.3, observed=soy_data)

    trace_constrained = pm.sample(1000, tune=1500,
                                  target_accept=0.9,
                                  return_inferencedata=True)
```

</div>

**Result:** Forecasts respect economic equilibrium (no arbitrage opportunities).

---

## Common Pitfalls

### 1. Ignoring Cross-Crop Constraints

Forecasting corn and soybeans independently can produce inconsistent results (farmers would plant 100% corn).

**Fix:** Joint hierarchical model with planting competition.

---

### 2. Assuming Fixed Seasonality

Harvest timing varies by year (weather delays). Fixed seasonal effects miss this.

**Fix:** Time-varying seasonality or use crop progress reports as covariates.

---

### 3. Pooling Unrelated Crops

Corn and cotton share little beyond global food demand. Over-pooling dilutes signal.

**Check:** Examine posterior shrinkage. If category-level variance is huge, crops are too different.

---

### 4. Forgetting Hemisphere Differences

US corn harvest (Oct) and Brazil corn harvest (Mar) shouldn't use same seasonal pattern.

**Fix:** Region-specific seasonal components.

---

## Model Comparison


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Compare hierarchical vs. independent models
with pm.Model() as independent_crops:
    for c, crop in enumerate(crop_names):
        mu_c = pm.Normal(f'mu_{crop}', 0, 5)
        sigma_c = pm.HalfNormal(f'sigma_{crop}', 2)
        pm.Normal(f'price_{crop}', mu=mu_c, sigma=sigma_c, observed=prices[:, c])

    trace_independent = pm.sample(1000, tune=1000, return_inferencedata=True)

# Compare
comparison = az.compare({
    'Hierarchical': trace,
    'Independent': trace_independent
})
print(comparison)

# Hierarchical should win (borrows strength across crops)
```


---

## Connections

**Builds on:**
- Module 4: Partial pooling fundamentals
- Module 2: Agricultural data sources (USDA WASDE)
- Economics: Crop substitution and complementarity

**Leads to:**
- Module 7: Regime switching (policy changes, structural breaks)
- Module 8: Integrating agronomic models (yield functions)

**Related concepts:**
- Spatial models (geographic basis differentials)
- Panel data econometrics (crop × time)

---

## Practice Problems

### Problem 1
Design a hierarchical model for wheat prices with three classes: Hard Red Winter (HRW), Soft Red Winter (SRW), and Spring Wheat. What determines the hierarchy structure?
<div class="callout-warning">

<strong>Warning:</strong> Design a hierarchical model for wheat prices with three classes: Hard Red Winter (HRW), Soft Red Winter (SRW), and Spring Wheat. What determines the hierarchy structure?



### Problem 2
The corn-soybean price ratio is currently 0.48. Historical range is [0.35, 0.50]. Your forecast predicts 0.60 next quarter. What constraint have you violated? How do you incorporate this into a Bayesian model?

### Problem 3
Brazil soybean harvest (Feb-Apr) typically pressures US soybean prices. Implement a hierarchical model that captures this cross-hemisphere relationship.

### Problem 4
You're forecasting corn and ethanol prices jointly. Corn is an input to ethanol production (crush margin). Design a hierarchy that respects this production relationship.

### Problem 5
USDA forecasts ending stocks at 1.2 billion bushels (stocks-to-use = 8%, very tight). How would you incorporate this into a hierarchical corn price model? What prior should you use for the scarcity premium?

---


---



## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving hierarchical models for agricultural commodities, what would be your first three steps to apply the techniques from this guide?




## Further Reading

1. **Wright, B.D. (2011)**. "The Economics of Grain Price Volatility." *Applied Economic Perspectives and Policy*, 33(1), 32-58.
   - Theory of storage and price-stock relationships

2. **Roberts, M.J. & Schlenker, W. (2013)**. "Identifying Supply and Demand Elasticities of Agricultural Commodities." *Review of Economic Studies*, 80(4), 1569-1599.
   - Structural models of ag supply/demand

3. **Irwin, S.H. & Sanders, D.R. (2012)**. "Testing the Masters Hypothesis in Commodity Futures Markets." *Energy Economics*, 34(1), 256-269.
   - Cross-commodity relationships and financialization

4. **Baylis, K., Paulson, N.D., & Piras, G. (2017)**. "Spatial Approaches to Panel Data in Agricultural Economics." *Journal of Agricultural and Resource Economics*, 42(3), 373-393.
   - Spatial hierarchies for regional crop prices

---

*"In agricultural markets, every crop is a neighbor. Hierarchical models respect the community."*

---

## Cross-References

<a class="link-card" href="./03_agricultural_complex_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_pooling_comparison.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

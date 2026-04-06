# Hierarchical Models for Energy Commodities

> **Reading time:** ~9 min | **Module:** 4 — Hierarchical Models | **Prerequisites:** Module 3 State-Space Models


## In Brief

Energy markets are interconnected through refining, transportation, and substitution. Hierarchical models pool information across crude grades, refined products, and geographic regions while respecting their structural relationships, improving forecasts for thinly-traded markets.

<div class="callout-insight">

<strong>Insight:</strong> **Borrow strength across related markets.** WTI and Brent prices are highly correlated but not identical. A hierarchical model learns the common "oil market" dynamics while preserving spread relationships, preventing overfitting on individual time series.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> Energy markets are interconnected through refining, transportation, and substitution.

</div>

---

## Formal Definition

### Three-Level Hierarchical Energy Model

**Level 1: Global Energy Factor (Hyperprior)**
$$\mu_{\text{global}} \sim \mathcal{N}(m_0, s_0^2)$$
$$\sigma_{\text{global}} \sim \text{HalfNormal}(\tau_0)$$

**Level 2: Product-Specific Parameters (Prior)**
$$\mu_{\text{product}} \sim \mathcal{N}(\mu_{\text{global}}, \sigma_{\text{global}}^2), \quad \text{product} \in \{\text{crude}, \text{products}, \text{gas}\}$$
$$\sigma_{\text{product}} \sim \text{HalfNormal}(\sigma_{\text{global}})$$

**Level 3: Individual Market Observations (Likelihood)**
$$y_{i,t} \sim \mathcal{N}(\alpha_i + \beta_i \cdot \mu_{\text{product}} + f(X_{i,t}), \sigma_{\text{product}}), \quad i \in \text{markets}$$

Where:
- $\alpha_i$: Market-specific intercept (location premium)
- $\beta_i$: Loading on product factor (correlation strength)
- $f(X_{i,t})$: Observable fundamentals (inventories, capacity)

---

## Intuitive Explanation

### The Hierarchy of Energy Markets

```
Global Energy Factor
         ├── Crude Oil Factor
         │        ├── WTI (NYMEX)
         │        ├── Brent (ICE)
         │        ├── Dubai/Oman
         │        └── Canadian Heavy
         │
         ├── Refined Products Factor
         │        ├── Gasoline (RBOB)
         │        ├── Diesel (ULSD)
         │        ├── Jet Fuel
         │        └── Heating Oil
         │
         └── Natural Gas Factor
                  ├── Henry Hub
                  ├── TTF (Europe)
                  ├── JKM (Asia LNG)
                  └── Regional Hubs
```

**Information Flow:**
1. Henry Hub gas price shock → updates natural gas factor
2. Natural gas factor update → informs regional hub forecasts
3. Global energy factor → weakly influences all markets

---

## Why Hierarchical Models for Energy?

### 1. Data Sparsity

Many regional markets have:
- Short history (new trading venues)
- Illiquid trading (wide bid-ask spreads)
- Missing data (plant shutdowns)

**Solution:** Borrow strength from liquid markets (WTI, Brent) to improve thin market forecasts.

### 2. Structural Relationships

**Crack Spreads:** Gasoline price ≈ Crude price + Refining margin
**Cross-Regional Arbitrage:** |WTI - Brent| < Transport cost
**Substitution:** High gas prices → increased coal/oil demand

Hierarchical models encode these as partial pooling weights.

### 3. Risk Management

Portfolio VaR requires correlation structure. Hierarchical models provide:
- Time-varying correlations (through dynamic factors)
- Uncertainty in correlation estimates
- Conditional dependence (given fundamentals)

---

## Code Implementation

### Basic Crude Oil Hierarchy


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# Simulated data: 3 crude grades over 200 weeks
np.random.seed(42)
n_weeks = 200
n_grades = 3
grade_names = ['WTI', 'Brent', 'Dubai']

# True global factor
global_factor = np.cumsum(np.random.normal(0, 2, n_weeks)) + 80

# Grade-specific parameters (true values)
grade_intercepts = np.array([0, 2, -3])  # Brent premium, Dubai discount
grade_loadings = np.array([1.0, 0.98, 0.95])  # WTI tracks perfectly, others slightly less
grade_noise = np.array([1.5, 1.2, 2.0])  # WTI most liquid

# Generate prices
prices = np.zeros((n_weeks, n_grades))
for g in range(n_grades):
    prices[:, g] = (grade_intercepts[g] +
                    grade_loadings[g] * global_factor +
                    np.random.normal(0, grade_noise[g], n_weeks))

# Build hierarchical model
with pm.Model() as crude_hierarchy:
    # Hyperpriors (global energy market)
    mu_global = pm.Normal('mu_global', mu=80, sigma=20)
    sigma_global = pm.HalfNormal('sigma_global', sigma=10)

    # Grade-level priors (partial pooling)
    grade_intercept = pm.Normal('grade_intercept',
                                mu=0,
                                sigma=5,
                                shape=n_grades)

    grade_loading = pm.Normal('grade_loading',
                             mu=1,
                             sigma=0.2,
                             shape=n_grades)

    grade_sigma = pm.HalfNormal('grade_sigma',
                               sigma=sigma_global,
                               shape=n_grades)

    # Global factor (random walk)
    factor_innov = pm.Normal('factor_innov', mu=0, sigma=1, shape=n_weeks-1)
    factor = pm.Deterministic(
        'factor',
        pm.math.concatenate([
            [mu_global],
            mu_global + pm.math.cumsum(sigma_global * factor_innov)
        ])
    )

    # Observations (vectorized across grades)
    for g in range(n_grades):
        pm.Normal(f'price_{grade_names[g]}',
                 mu=grade_intercept[g] + grade_loading[g] * factor,
                 sigma=grade_sigma[g],
                 observed=prices[:, g])

    # Sample
    trace = pm.sample(1000, tune=2000,
                     target_accept=0.9,
                     return_inferencedata=True)

# Results
print(az.summary(trace, var_names=['grade_intercept', 'grade_loading', 'grade_sigma']))

# Compare to true values
print("\nTrue vs. Estimated:")
print("Intercepts:", grade_intercepts)
print("Estimated:", trace.posterior['grade_intercept'].mean(dim=['chain', 'draw']).values)
```

</div>
</div>

---

## The Crude-Products Hierarchy

### Crack Spread Model
<div class="callout-insight">

<strong>Insight:</strong> Refining adds value to crude oil. Model the spread hierarchically.

</div>


Refining adds value to crude oil. Model the spread hierarchically.


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
with pm.Model() as crack_spread_model:
    n_obs = 200

    # Crude oil process (Level 1)
    crude_mu = pm.Normal('crude_mu', mu=70, sigma=10)
    crude_sigma = pm.HalfNormal('crude_sigma', sigma=5)
    crude_innov = pm.Normal('crude_innov', 0, 1, shape=n_obs)
    crude_price = pm.Deterministic(
        'crude_price',
        crude_mu + pm.math.cumsum(crude_sigma * crude_innov)
    )

    # Crack spread (Level 2) - the refining margin
    # Mean reverts to long-run refining profitability
    crack_mu = pm.Normal('crack_mu', mu=15, sigma=5)  # ~$15/bbl typical
    crack_phi = pm.Beta('crack_phi', alpha=10, beta=1) * 2 - 1  # Mean reversion
    crack_sigma = pm.HalfNormal('crack_sigma', sigma=3)

    crack_innov = pm.Normal('crack_innov', 0, 1, shape=n_obs)

    # Crack spread mean-reverts
    crack_spread = pm.Deterministic(
        'crack_spread',
        crack_mu + pm.math.cumsum(
            crack_phi * crack_innov * crack_sigma
        )
    )

    # Gasoline price = Crude + Crack Spread
    gasoline_price = pm.Deterministic('gasoline_price',
                                     crude_price + crack_spread)

    # Observations
    pm.Normal('crude_obs', mu=crude_price, sigma=1, observed=crude_data)
    pm.Normal('gasoline_obs', mu=gasoline_price, sigma=1.5, observed=gasoline_data)
```

</div>
</div>

This ensures gasoline and crude forecasts are consistent (no arbitrage opportunities).

---

## Geographic Hierarchy: Regional Natural Gas

### US Regional Hub Hierarchy
<div class="callout-warning">

<strong>Warning:</strong> hubs = ['Henry_Hub', 'Chicago', 'NY', 'California']

</div>



<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Regional hubs: Henry Hub (benchmark), Chicago, NY, California
hubs = ['Henry_Hub', 'Chicago', 'NY', 'California']
n_hubs = len(hubs)
n_obs = 250

with pm.Model() as gas_regional:
    # Level 1: National natural gas market
    national_mu = pm.Normal('national_mu', mu=3, sigma=1)  # $/mmBtu
    national_sigma = pm.HalfNormal('national_sigma', sigma=1)

    national_innov = pm.Normal('national_innov', 0, 1, shape=n_obs)
    national_price = pm.Deterministic(
        'national_price',
        national_mu + pm.math.cumsum(national_sigma * national_innov)
    )

    # Level 2: Regional basis differentials (transport costs, local supply/demand)
    # Basis = Regional price - Henry Hub price
    basis_mu = pm.Normal('basis_mu', mu=0, sigma=1, shape=n_hubs)
    basis_sigma = pm.HalfNormal('basis_sigma', sigma=0.5, shape=n_hubs)

    # California has highest basis (transport from Texas)
    # Chicago/NY moderate (some regional production)
    # Henry Hub basis = 0 by definition

    # Seasonal basis variation (winter peaks for heating demand)
    months = np.arange(n_obs) % 12
    seasonal_basis = pm.Normal('seasonal_basis', mu=0, sigma=0.3, shape=(n_hubs, 12))

    # Regional prices
    for i, hub in enumerate(hubs):
        if hub == 'Henry_Hub':
            # Benchmark (no basis)
            regional_price = national_price
        else:
            # Non-benchmark hubs
            basis_innov = pm.Normal(f'basis_innov_{hub}', 0, 1, shape=n_obs)
            basis = basis_mu[i] + seasonal_basis[i, months] + basis_sigma[i] * basis_innov
            regional_price = national_price + basis

        pm.Normal(f'price_{hub}',
                 mu=regional_price,
                 sigma=0.2,  # Observation noise
                 observed=gas_data[hub])
```

</div>
</div>

**Result:** Regional forecasts account for:
1. National supply/demand (shared factor)
2. Transport costs (basis differentials)
3. Seasonal heating/cooling demand (seasonal basis)

---

## Advanced: Dynamic Correlation Structure

### Time-Varying Crude Correlations
<div class="callout-key">

<strong>Key Point:</strong> During crises, crude grades converge (arbitrage opportunities shrink).

</div>


During crises, crude grades converge (arbitrage opportunities shrink).


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
with pm.Model() as dynamic_correlation:
    n_grades = 3
    n_obs = 200

    # Factor volatility is time-varying (high during crises)
    vol_mu = pm.Normal('vol_mu', mu=0, sigma=1)
    vol_phi = pm.Beta('vol_phi', alpha=20, beta=1) * 2 - 1
    vol_sigma = pm.HalfNormal('vol_sigma', sigma=0.5)

    # Log-volatility process (stochastic volatility)
    h = pm.GaussianRandomWalk('h', mu=vol_mu, sigma=vol_sigma, shape=n_obs)
    factor_vol = pm.Deterministic('factor_vol', pm.math.exp(h))

    # Global factor with time-varying volatility
    factor_innov = pm.Normal('factor_innov', 0, 1, shape=n_obs)
    factor = pm.Deterministic(
        'factor',
        pm.math.cumsum(factor_vol * factor_innov)
    )

    # Grade loadings (higher during high-vol periods)
    base_loading = pm.Beta('base_loading', alpha=8, beta=2, shape=n_grades)

    # Loading increases with global volatility (more synchronization in crises)
    loading = pm.Deterministic(
        'loading',
        base_loading[None, :] + 0.1 * factor_vol[:, None]
    )

    # Observations
    for g in range(n_grades):
        pm.Normal(f'price_{g}',
                 mu=loading[:, g] * factor,
                 sigma=1,
                 observed=prices[:, g])
```

</div>
</div>

---

## Model Comparison & Diagnostics

### Compare Hierarchical vs. Independent Models


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Independent model (no pooling)
with pm.Model() as independent:
    for g in range(n_grades):
        mu_g = pm.Normal(f'mu_{g}', 0, 10)
        sigma_g = pm.HalfNormal(f'sigma_{g}', 5)
        pm.Normal(f'price_{g}', mu=mu_g, sigma=sigma_g, observed=prices[:, g])

    trace_independent = pm.sample(1000, tune=1000, return_inferencedata=True)

# Complete pooling (all grades identical)
with pm.Model() as pooled:
    mu = pm.Normal('mu', 0, 10)
    sigma = pm.HalfNormal('sigma', 5)
    pm.Normal('price', mu=mu, sigma=sigma, observed=prices.flatten())

    trace_pooled = pm.sample(1000, tune=1000, return_inferencedata=True)

# Compare
comparison = az.compare({
    'Hierarchical': trace,
    'Independent': trace_independent,
    'Pooled': trace_pooled
})
print(comparison)
```

</div>
</div>

**Typically:** Hierarchical model wins (best of both worlds).

---

## Common Pitfalls

### 1. Over-Pooling

If markets are truly independent (e.g., WTI vs. lumber), hierarchy hurts.

**Check:** Examine posterior shrinkage. If all parameters collapse to hyperprior mean, you've over-pooled.

### 2. Ignoring Transportation Constraints

WTI-Brent spread cannot exceed tanker cost (~$5/bbl). Add constraints:


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
spread = pm.Deterministic('spread', brent_price - wti_price)
pm.Potential('arbitrage_bound', pm.math.switch(spread > 7, -np.inf, 0))
```

</div>
</div>

### 3. Static Loadings During Regime Shifts

Shale revolution changed WTI-Brent relationship. Use time-varying loadings or regime-switching.

### 4. Circular Dependencies

Don't make crude depend on gasoline *and* gasoline depend on crude. Choose a causal direction.

---

## Connections

**Builds on:**
- Module 4: Partial pooling fundamentals
- Module 3: State space models (dynamic factors)

**Leads to:**
- Module 5: GPs for continuous spatial interpolation (pipelines)
- Module 7: Regime-switching hierarchies
- Module 8: Incorporating storage/capacity constraints

**Related concepts:**
- Factor models (finance)
- Panel data models (econometrics)
- Multi-level models (statistics)

---

## Practice Problems

### Problem 1
You have 5 years of daily WTI data but only 1 year of Mars (US Gulf crude) data. Design a hierarchical model to improve Mars price forecasts.

### Problem 2
The WTI-Brent spread is currently $8/bbl. Your model predicts $12/bbl next month. What constraint have you violated? How do you fix it?

### Problem 3
During COVID-19, crude oil volatility spiked and correlations across grades increased. Implement a hierarchical model with time-varying correlations that captures this.

### Problem 4
Natural gas prices in California are typically $2/mmBtu above Henry Hub. Last week they spiked to $10 above (pipeline outage). How does a hierarchical model handle this outlier compared to an independent model?

### Problem 5
You're forecasting diesel and jet fuel prices. Both are distillates refined from crude. Design a three-level hierarchy: crude → distillates → individual products. What does each level represent?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving hierarchical models for energy commodities, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

1. **Gelman, A. & Hill, J. (2006)**. *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
   - Foundational text on hierarchical modeling

2. **Fattouh, B. (2010)**. "The Dynamics of Crude Oil Price Differentials." *Energy Economics*, 32(2), 334-342.
   - Empirical analysis of crude grade relationships

3. **Pindyck, R.S. & Rotemberg, J.J. (1990)**. "The Excess Co-Movement of Commodity Prices." *The Economic Journal*, 100(403), 1173-1189.
   - Evidence for common factors across commodities

4. **Büyükşahin, B. & Rober, M.A. (2014)**. "Speculators, Commodities and Cross-Market Linkages." *Journal of International Money and Finance*, 42, 38-70.
   - Time-varying correlations in energy markets

---

*"In energy markets, no crude grade is an island. Hierarchical models respect the interconnected reality."*

---

## Cross-References

<a class="link-card" href="./02_energy_complex_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_pooling_comparison.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

# Module 4 Quiz: Hierarchical Models and Partial Pooling

**Course:** Bayesian Commodity Forecasting
**Module:** 04 - Hierarchical Models for Multi-Commodity Analysis
**Time Limit:** 30 minutes
**Total Points:** 100 points
**Instructions:** Answer all questions. Show mathematical derivations where requested.

---

## Section A: Hierarchical Model Foundations (35 points)

### Question 1 (10 points)
Explain the three pooling strategies for estimating parameters across multiple groups (commodities). For each strategy, provide:
- **The model specification**
- **One advantage**
- **One disadvantage**
- **An example scenario where it's appropriate**

Consider estimating seasonal effects for corn, soybeans, and wheat.

**Answer:**

### **1. Complete Pooling (No Hierarchy)**

**Model specification:**
$$\beta_{\text{season}} = \beta \quad \text{(same for all commodities)}$$

All commodities share identical seasonal effects.

```python
with pm.Model():
    beta_season = pm.Normal('beta_season', mu=0, sigma=10, shape=12)  # One set for all
    # Use same beta for corn, soy, wheat
```

**Advantage:**
- **Maximum statistical power:** All data pooled → precise estimates
- Useful when few observations per commodity

**Disadvantage:**
- **Ignores heterogeneity:** Assumes all commodities identical
- Poor fit if seasonal patterns actually differ

**Appropriate scenario:**
- **Closely related commodities with same growing regions**
- Example: Different wheat varieties (hard red winter, soft white) in same area
- Strong prior belief that seasonality is identical

---

### **2. No Pooling (Independent Estimates)**

**Model specification:**
$$\beta_j \sim \mathcal{N}(\mu_0, \sigma_0^2) \quad \text{independently for each commodity } j$$

Each commodity has completely separate parameters.

```python
with pm.Model():
    beta_corn = pm.Normal('beta_corn', mu=0, sigma=10, shape=12)
    beta_soy = pm.Normal('beta_soy', mu=0, sigma=10, shape=12)
    beta_wheat = pm.Normal('beta_wheat', mu=0, sigma=10, shape=12)
    # No sharing of information
```

**Advantage:**
- **Maximum flexibility:** Each commodity can have unique pattern
- No bias from imposing similarity

**Disadvantage:**
- **Wastes information:** Doesn't learn from related commodities
- High variance when data is limited per commodity
- Prone to overfitting

**Appropriate scenario:**
- **Very different commodities**
- Example: Oil, gold, and wheat (different drivers)
- Abundant data for each commodity

---

### **3. Partial Pooling (Hierarchical Model)**

**Model specification:**
$$\begin{aligned}
\beta_j &\sim \mathcal{N}(\mu_{\text{pop}}, \tau^2) \quad \text{commodity-specific effects} \\
\mu_{\text{pop}} &\sim \mathcal{N}(0, \sigma^2) \quad \text{population mean} \\
\tau &\sim \text{HalfNormal}(\sigma_\tau) \quad \text{between-commodity variation}
\end{aligned}$$

Commodity-specific parameters drawn from common distribution.

```python
with pm.Model():
    mu_pop = pm.Normal('mu_pop', mu=0, sigma=10, shape=12)  # Population seasonal
    tau = pm.HalfNormal('tau', sigma=5, shape=12)           # Between-commodity SD

    beta_corn = pm.Normal('beta_corn', mu=mu_pop, sigma=tau, shape=12)
    beta_soy = pm.Normal('beta_soy', mu=mu_pop, sigma=tau, shape=12)
    beta_wheat = pm.Normal('beta_wheat', mu=mu_pop, sigma=tau, shape=12)
```

**Advantage:**
- **Best of both worlds:** Shares information but allows differences
- **Shrinkage:** Extreme estimates pulled toward group mean (regularization)
- Better predictions for commodities with little data

**Disadvantage:**
- **More complex:** Additional hyperparameters to estimate
- Computational cost higher (MCMC slower)

**Appropriate scenario:**
- **Related but not identical commodities**
- **Example: Corn, soybeans, wheat (all agricultural, similar regions)**
- Especially valuable when data imbalance (e.g., more corn data than wheat)

---

**Summary Table:**

| Strategy | Commodities treated as | Best when |
|----------|----------------------|-----------|
| Complete pooling | Identical | Strong similarity, little data |
| No pooling | Completely different | Unrelated, abundant data per group |
| Partial pooling | Similar but distinct | Related commodities, typical scenario |

**Scoring:**
- Complete pooling: 3 points
- No pooling: 3 points
- Partial pooling: 4 points (most important for this module)

---

### Question 2 (8 points)
Consider a hierarchical model for volatility across crude oil, natural gas, and gasoline:

$$\begin{aligned}
r_{j,t} &\sim \mathcal{N}(0, \sigma_j^2) \quad \text{returns for commodity } j \\
\sigma_j &\sim \text{LogNormal}(\mu_\sigma, \tau_\sigma) \quad \text{commodity-specific volatility} \\
\mu_\sigma &\sim \mathcal{N}(\log(0.02), 1) \quad \text{population mean (log scale)} \\
\tau_\sigma &\sim \text{HalfNormal}(0.5) \quad \text{between-commodity dispersion}
\end{aligned}$$

**(a)** Why is LogNormal used for $\sigma_j$ instead of Normal? (3 points)

**(b)** If the posterior mean of $\tau_\sigma$ is very small (near 0), what does this imply about the three commodities? (3 points)

**(c)** How would you extend this model to allow for time-varying volatility? (2 points)

**Answer:**

**(a)** Why LogNormal for volatility:

**Reason 1: Positive constraint**
- Volatility $\sigma_j$ must be positive
- Normal distribution allows negative values → inappropriate
- LogNormal: $\sigma_j = \exp(\theta_j)$ where $\theta_j \sim \mathcal{N}(\mu_\sigma, \tau_\sigma)$
- Automatically ensures $\sigma_j > 0$

**Reason 2: Multiplicative structure**
- Volatility ratios are natural: "Gas is 1.5x more volatile than oil"
- LogNormal: Differences in log-space → ratios in original space
- Example: $\log(\sigma_{\text{gas}}) - \log(\sigma_{\text{oil}}) = \log(\sigma_{\text{gas}} / \sigma_{\text{oil}})$

**Reason 3: Skewness**
- Volatility distributions are right-skewed (occasional very high volatility)
- LogNormal captures this naturally
- Normal is symmetric (inappropriate)

**Alternative:** HalfNormal also constrains to positive, but LogNormal better for hierarchical structure.

**(b)** Small $\tau_\sigma$ interpretation:

**Implication: Commodities have SIMILAR volatilities**

**Mechanism:**
- $\tau_\sigma$ controls **between-group variation**
- Small $\tau_\sigma$ → $\sigma_j$ tightly clustered around $\mu_\sigma$
- All commodities have approximately $\sigma \approx \exp(\mu_\sigma)$

**Example:**
- If $\mu_\sigma = \log(0.02) \approx -3.91$ and $\tau_\sigma = 0.1$:
- $\sigma_j \in [0.018, 0.022]$ for all three commodities (95% interval)
- Oil, gas, gasoline have nearly identical daily volatility

**Contrast:**
- Large $\tau_\sigma$ (e.g., 0.5) → wide dispersion:
  - Oil: $\sigma = 0.015$ (less volatile)
  - Gas: $\sigma = 0.035$ (much more volatile)
  - Gasoline: $\sigma = 0.022$ (intermediate)

**Statistical consequence:**
- Small $\tau_\sigma$ → strong shrinkage toward population mean
- Effectively closer to complete pooling

**(c)** Time-varying volatility extension:

**Approach 1: Stochastic Volatility**
$$\begin{aligned}
r_{j,t} &\sim \mathcal{N}(0, \exp(h_{j,t})) \\
h_{j,t+1} &= \mu_j + \phi_j (h_{j,t} - \mu_j) + \eta_{j,t} \\
\mu_j &\sim \mathcal{N}(\mu_{\text{pop}}, \tau^2) \quad \text{hierarchical log-volatility}
\end{aligned}$$

Each commodity has time-varying log-volatility $h_{j,t}$ (AR(1) process), with hierarchical mean levels.

**Approach 2: GARCH with hierarchical parameters**
$$\begin{aligned}
\sigma_{j,t}^2 &= \omega_j + \alpha_j r_{j,t-1}^2 + \beta_j \sigma_{j,t-1}^2 \\
[\alpha_j, \beta_j] &\sim \mathcal{N}(\mu_{\text{GARCH}}, \Sigma_{\text{GARCH}}) \quad \text{hierarchical GARCH params}
\end{aligned}$$

**Scoring:**
- Part (a): 3 points (positive constraint + multiplicative structure)
- Part (b): 3 points (similarity interpretation)
- Part (c): 2 points (viable extension)

---

### Question 3 (9 points)
**Shrinkage in action:** You fit a hierarchical model to estimate trend slopes for five agricultural commodities. The data shows:

| Commodity | Sample Size | Raw OLS Slope | Hierarchical Posterior Mean Slope |
|-----------|-------------|---------------|--------------------------------|
| Corn | 500 obs | +0.05 | +0.04 |
| Soybeans | 480 obs | +0.03 | +0.03 |
| Wheat | 520 obs | -0.02 | -0.01 |
| Rice | 50 obs | +0.15 | +0.06 |
| Cotton | 45 obs | -0.12 | -0.03 |

**(a)** Why are the Rice and Cotton hierarchical estimates "shrunk" more dramatically than Corn/Soybeans/Wheat? (4 points)

**(b)** Toward what value are the estimates being shrunk? (2 points)

**(c)** Under what condition would the hierarchical estimates equal the raw OLS estimates? (3 points)

**Answer:**

**(a)** Why Rice and Cotton are shrunk more:

**Reason: Small sample size → high uncertainty**

**Mechanism of shrinkage:**
$$\hat{\beta}_j^{\text{hierarchical}} \approx \frac{n_j}{n_j + \tau^{-2}} \hat{\beta}_j^{\text{OLS}} + \frac{\tau^{-2}}{n_j + \tau^{-2}} \mu_{\text{pop}}$$

Where:
- $n_j$ = effective sample size for commodity $j$
- $\tau^2$ = between-commodity variance
- $\mu_{\text{pop}}$ = population mean slope

**Intuition:**
- **Large $n_j$ (Corn, Soybeans, Wheat):** Trust the data
  - Weight toward OLS estimate
  - Shrinkage is minimal
- **Small $n_j$ (Rice, Cotton):** Data unreliable
  - Weight toward population mean
  - Heavy shrinkage

**Numerical example:**
- **Rice:** OLS = +0.15, $n=50$ (small) → High variance in OLS
  - Hierarchical model: "This extreme estimate is likely noise, pull toward group average"
  - Result: +0.15 → +0.06 (shrunk by 0.09)

- **Corn:** OLS = +0.05, $n=500$ (large) → Low variance in OLS
  - Hierarchical model: "This estimate is reliable, trust it"
  - Result: +0.05 → +0.04 (shrunk by only 0.01)

**Statistical principle:**
- **James-Stein estimator:** Extreme values in small samples are likely overfitting
- Shrinkage toward group mean reduces mean squared error

**(b)** Target of shrinkage:

Estimates are shrunk toward the **population mean slope** $\mu_{\text{pop}}$.

**Calculation from data:**
Approximate $\mu_{\text{pop}}$ as weighted average of hierarchical estimates:
$$\mu_{\text{pop}} \approx \frac{0.04 + 0.03 - 0.01 + 0.06 - 0.03}{5} \approx +0.018$$

Or weight by sample size for a rough estimate:
$$\mu_{\text{pop}} \approx \frac{500(0.04) + 480(0.03) + 520(-0.01) + 50(0.06) + 45(-0.03)}{1595} \approx +0.02$$

**All estimates pulled toward ~+0.02 (slightly positive trend)**

**Visual:**
```
OLS Estimates:
Rice:   +0.15 ●━━━━━━━━━→ +0.06  Hierarchical
Cotton: -0.12 ←━━━━━━━━━● -0.03  (shrunk toward μ_pop ≈ +0.02)
Corn:   +0.05 ●→ +0.04
Wheat:  -0.02 ←● -0.01
                 ↑
            μ_pop ≈ +0.02
```

**(c)** When hierarchical = OLS:

**Condition 1: Infinite sample size**
- As $n_j \to \infty$, shrinkage weight → 0
- Trust data completely, ignore population

**Condition 2: Infinite between-group variance**
- If $\tau^2 \to \infty$ (commodities extremely different), no reason to pool
- Each commodity fully independent
- Equivalent to no pooling

**Condition 3: No pooling prior**
- If we set $\tau^2 \to \infty$ explicitly (weak hierarchical prior)
- Model collapses to independent estimates

**Mathematical:**
$$\text{If } \tau^2 \to \infty \text{ or } n_j \to \infty, \text{ then } \hat{\beta}_j^{\text{hierarchical}} \to \hat{\beta}_j^{\text{OLS}}$$

**Practical check:**
- Examine posterior of $\tau$
- If $\tau$ posterior does not exclude large values, model may be close to no pooling

**Scoring:**
- Part (a): 4 points (sample size mechanism)
- Part (b): 2 points (population mean identification)
- Part (c): 3 points (conditions explained)

---

### Question 4 (8 points)
Write a PyMC model specification for a hierarchical regression of commodity prices on storage levels:

$$y_{j,t} = \alpha_j + \beta_j \cdot \text{Storage}_{j,t} + \epsilon_{j,t}$$

Where:
- $j$ indexes commodities (crude oil, natural gas, gasoline)
- $\alpha_j, \beta_j$ are commodity-specific parameters drawn from population distributions
- Include appropriate priors for a Bayesian hierarchical model

**Answer:**

```python
import pymc as pm
import numpy as np

# Data setup
# Assume data is stacked: N total observations across J commodities
# commodity_idx: array of commodity indices [0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2]
# storage: array of storage values for each observation
# prices: array of price values (outcome)

J = 3  # Number of commodities (oil, gas, gasoline)
commodity_idx = np.array([...])  # Shape: (N,)
storage = np.array([...])        # Shape: (N,)
prices = np.array([...])         # Shape: (N,)

with pm.Model() as hierarchical_regression:

    # ================================================
    # HYPERPARAMETERS (Population-level)
    # ================================================

    # Population mean for intercepts
    mu_alpha = pm.Normal('mu_alpha', mu=0, sigma=50)
    # Justification: Prices centered around 0 after standardization, wide prior

    # Population mean for slopes (storage effect)
    mu_beta = pm.Normal('mu_beta', mu=0, sigma=10)
    # Justification: Storage coefficient typically order of magnitude 0.1-1

    # Between-commodity standard deviation for intercepts
    tau_alpha = pm.HalfNormal('tau_alpha', sigma=20)
    # Justification: Commodities can have different baseline prices, but related

    # Between-commodity standard deviation for slopes
    tau_beta = pm.HalfNormal('tau_beta', sigma=5)
    # Justification: Storage effect magnitude may vary across commodities

    # ================================================
    # GROUP-LEVEL PARAMETERS (Commodity-specific)
    # ================================================

    # Non-centered parameterization (better sampling)
    # α_j = μ_α + τ_α * α_j^raw
    alpha_raw = pm.Normal('alpha_raw', mu=0, sigma=1, shape=J)
    alpha = pm.Deterministic('alpha', mu_alpha + tau_alpha * alpha_raw)

    beta_raw = pm.Normal('beta_raw', mu=0, sigma=1, shape=J)
    beta = pm.Deterministic('beta', mu_beta + tau_beta * beta_raw)

    # ================================================
    # OBSERVATION MODEL
    # ================================================

    # Linear predictor
    mu = alpha[commodity_idx] + beta[commodity_idx] * storage

    # Observation noise (could also be hierarchical if desired)
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=10)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=prices)

    # ================================================
    # INFERENCE
    # ================================================

    trace = pm.sample(2000,
                      tune=1000,
                      return_inferencedata=True,
                      target_accept=0.90)

# ================================================
# POSTERIOR ANALYSIS
# ================================================

import arviz as az

# Examine population-level parameters
print(az.summary(trace, var_names=['mu_alpha', 'mu_beta', 'tau_alpha', 'tau_beta']))

# Examine commodity-specific parameters
print(az.summary(trace, var_names=['alpha', 'beta']))

# Forest plot to visualize shrinkage
az.plot_forest(trace,
               var_names=['alpha', 'beta'],
               combined=True,
               figsize=(10, 6))

# Compare to no-pooling estimates (for reference)
# (Would fit separate regressions for each commodity)
```

**Key features of this specification:**

1. **Non-centered parameterization:**
   - `alpha_raw ~ Normal(0, 1)` then `alpha = mu_alpha + tau_alpha * alpha_raw`
   - Improves MCMC sampling efficiency (decorrelates hyperparameters from group parameters)
   - Critical for hierarchical models to avoid divergences

2. **Appropriate priors:**
   - Population means: Weakly informative Normal
   - Between-group SD: HalfNormal (positive constraint, mild regularization)
   - Observation noise: HalfNormal

3. **Indexing:**
   - `alpha[commodity_idx]` selects correct commodity-specific intercept for each observation
   - Allows vectorized likelihood computation

4. **Extensions:**
   - Could make `sigma_obs` hierarchical (commodity-specific noise)
   - Could add time-varying effects or interactions

**Scoring:**
- Correct hierarchical structure: 3 points
- Appropriate priors: 2 points
- Non-centered parameterization: 2 points
- Likelihood specification: 1 point

---

## Section B: Practical Applications (35 points)

### Question 5 (12 points)
You are analyzing storage effects on prices for crude oil, natural gas, and gasoline. You fit three models:

1. **Pooled:** Single $\beta$ for all commodities
2. **Unpooled:** Separate $\beta_j$ for each (independent priors)
3. **Hierarchical:** $\beta_j \sim \mathcal{N}(\mu_\beta, \tau_\beta^2)$

**Results:**

| Model | $\hat{\beta}_{\text{oil}}$ | $\hat{\beta}_{\text{gas}}$ | $\hat{\beta}_{\text{gasoline}}$ | WAIC |
|-------|--------|--------|----------|------|
| Pooled | -0.50 | -0.50 | -0.50 | 2450 |
| Unpooled | -0.45 | -0.62 | -0.48 | 2380 |
| Hierarchical | -0.47 | -0.58 | -0.49 | 2360 |

Hierarchical model hyperparameters: $\hat{\mu}_\beta = -0.51$, $\hat{\tau}_\beta = 0.08$

**(a)** Interpret the storage effect coefficients. Why are they negative? (4 points)

**(b)** Compare the three model results. What does the pattern of coefficients tell you about the commodities? (4 points)

**(c)** Natural gas has the most extreme coefficient in the unpooled model. Why is it shrunk more in the hierarchical model? (4 points)

**Answer:**

**(a)** Interpretation of negative storage coefficients:

**Coefficient meaning:**
$$\beta_j = \frac{\partial \text{Price}_j}{\partial \text{Storage}_j}$$

**$\beta < 0$ implies:** Higher storage → Lower prices

**Economic rationale (Theory of Storage):**

1. **Supply and demand:** More inventory → more supply available → prices fall
2. **Scarcity premium:** Low stocks → scarcity → high prices
3. **Convenience yield:** When storage is tight, spot prices spike above futures (backwardation)

**Magnitude interpretation:**
- Oil: $\beta = -0.47$ means 1 million barrel increase in storage → $0.47 decrease in price (units depend on price scale)
- Gas: $\beta = -0.58$ (more sensitive to storage)

**Typical empirical findings:**
- Energy commodities: $\beta \in [-0.3, -0.7]$
- Stronger effect when storage near capacity constraints

**Why not positive?**
- Positive would imply more supply → higher prices (contradicts economics)
- If observed, suggests omitted variable bias or reverse causality

**(b)** Comparison of three models:

**Pattern:**

| Commodity | Pooled | Unpooled | Hierarchical | Interpretation |
|-----------|--------|----------|--------------|----------------|
| Oil | -0.50 | -0.45 | -0.47 | Moderate storage sensitivity |
| Gas | -0.50 | -0.62 | -0.58 | **Most** storage-sensitive |
| Gasoline | -0.50 | -0.48 | -0.49 | Similar to average |

**Key observations:**

1. **Natural gas is different:**
   - Unpooled: $\beta_{\text{gas}} = -0.62$ (most extreme)
   - Hierarchical: Shrunk toward population mean but still most negative
   - **Implication:** Gas prices genuinely more responsive to storage

2. **Oil and gasoline are similar:**
   - Both near population mean $\mu_\beta = -0.51$
   - Little difference between unpooled and hierarchical
   - **Implication:** Storage effects are similar, correctly grouped

3. **Pooled model biased:**
   - Forces all coefficients to $-0.50$
   - Misses important heterogeneity (gas is different)
   - **WAIC worse:** 2450 vs 2360 (hierarchical)
   - Underfit: Doesn't capture true variation

4. **Hierarchical model optimal:**
   - Best WAIC: 2360
   - Balances flexibility (allows differences) with regularization (shrinkage)
   - Still recognizes gas is distinct while not overfitting

**Economic interpretation:**
- **Natural gas:** Storage capacity constraints tighter → price more elastic to inventory changes
- **Oil & gasoline:** Larger relative storage capacity → less price sensitivity

**(c)** Why natural gas shrunk more:

**Observation:**
- Gas: Unpooled $-0.62$ → Hierarchical $-0.58$ (shrinkage of 0.04)
- Oil: Unpooled $-0.45$ → Hierarchical $-0.47$ (shrinkage of 0.02 in opposite direction!)
- Gasoline: Unpooled $-0.48$ → Hierarchical $-0.49$ (shrinkage of 0.01)

**Reason 1: Distance from population mean**
- Population mean: $\mu_\beta = -0.51$
- Gas estimate ($-0.62$) is **farthest** from $\mu_\beta$
- Hierarchical model assumes commodities cluster around $\mu_\beta$
- Extreme values treated skeptically → pulled back toward mean

**Reason 2: Small between-group variance**
- $\tau_\beta = 0.08$ (small)
- Small $\tau$ → commodities expected to be similar
- Gas being at $-0.62$ is $(−0.62 − (−0.51))/0.08 = 1.375$ standard deviations away
- Model says "unlikely gas is that different, probably sampling error"

**Reason 3: Regularization principle**
- Hierarchical model trades off:
  - Fit to data (unpooled likelihood)
  - Fit to population distribution (hierarchical prior)
- For gas: Unpooled fits data better, but hierarchical prior pulls toward $\mu_\beta$
- Result: Compromise at $-0.58$

**Visual:**
```
          Pooled            Hierarchical    Unpooled
Gas:        -0.50 ←←←←←←←← -0.58 ←←←- -0.62
              ↑ (ignores heterogeneity)  ↑ (no regularization)
              └─ μ_pop = -0.51
```

**If we wanted less shrinkage:**
- Use weaker hierarchical prior (larger $\sigma_{\tau}$)
- Allow $\tau_\beta$ to be larger
- Then gas estimate would stay closer to $-0.62$

**Scoring:**
- Part (a): 4 points (economic interpretation)
- Part (b): 4 points (model comparison insights)
- Part (c): 4 points (shrinkage mechanism)

---

### Question 6 (10 points)
**Forecasting with hierarchical models:** After fitting a hierarchical model to historical data, you need to forecast prices for a **new commodity** (ethanol) not in the training data. You have ethanol storage data but no price history.

**(a)** How can the hierarchical model provide forecasts for ethanol? (4 points)

**(b)** Compare the uncertainty of ethanol forecasts to forecasts for the three original commodities (oil, gas, gasoline). (3 points)

**(c)** Write the PyMC code to sample from the posterior predictive distribution for ethanol. (3 points)

**Answer:**

**(a)** Forecasting for new commodity:

**Approach: Sample from population distribution**

Since we don't have ethanol-specific parameters, we use the hierarchical prior:

$$\beta_{\text{ethanol}} \sim \mathcal{N}(\mu_\beta, \tau_\beta^2)$$

**Procedure:**
1. **Use posterior of hyperparameters** $\mu_\beta, \tau_\beta$ from training data
2. **Sample new commodity parameters** from population distribution
3. **Generate predictions** using ethanol's storage data

**Intuition:**
- Hierarchical model learned that "commodities in this class have storage effects around $\mu_\beta$ with variation $\tau_\beta$"
- Ethanol is assumed to be from the same population
- Best guess: Parameters are typical of the group

**Mathematical:**
$$p(y_{\text{ethanol}} | X_{\text{ethanol}}, \text{Data}_{\text{train}}) = \int p(y | X, \beta) p(\beta | \mu_\beta, \tau_\beta) p(\mu_\beta, \tau_\beta | \text{Data}_{\text{train}}) d\beta d\mu_\beta d\tau_\beta$$

**This is "shrinkage all the way to the population mean"** (no ethanol-specific data).

**(b)** Uncertainty comparison:

**Ethanol forecasts will have HIGHER uncertainty:**

**Source 1: Parameter uncertainty**
- **Original commodities:** Parameters $\beta_j$ have posteriors narrowed by data
  - Example: $\beta_{\text{oil}} \sim \mathcal{N}(-0.47, 0.05^2)$ (learned from data)
- **Ethanol:** Parameter sampled from population distribution
  - $\beta_{\text{ethanol}} \sim \mathcal{N}(-0.51, 0.08^2)$ (only population info)
  - Wider distribution: $0.08 > 0.05$

**Source 2: No shrinkage**
- Original commodities: Data + population prior
- Ethanol: Only population prior (no data to refine estimate)

**Quantitative:**
$$\begin{aligned}
\text{Var}(y_{\text{oil}}) &= \text{Var}(\beta_{\text{oil}}) \cdot X^2 + \sigma_{\text{obs}}^2 \\
\text{Var}(y_{\text{ethanol}}) &= \text{Var}(\beta_{\text{ethanol}}) \cdot X^2 + \sigma_{\text{obs}}^2 \\
\text{Var}(\beta_{\text{ethanol}}) &> \text{Var}(\beta_{\text{oil}}) \\
\implies \text{Var}(y_{\text{ethanol}}) &> \text{Var}(y_{\text{oil}})
\end{aligned}$$

**Visual:**
```
Prediction Interval Width:
Oil (known):     ●━━━━━━━━━━● (narrow)
Ethanol (new):   ●━━━━━━━━━━━━━━━━━━● (wide)
```

**Practical implication:**
- Ethanol forecasts less precise
- Reflects honest uncertainty: We don't know ethanol's specific parameters
- As ethanol data accumulates, can update model → uncertainty decreases

**(c)** PyMC posterior predictive code:

```python
# Assume hierarchical model already fit (see Question 4)
# Now generate predictions for new commodity (ethanol)

# Ethanol data (storage levels, but no price history)
storage_ethanol = np.array([100, 105, 98, ...])  # Storage values
n_new = len(storage_ethanol)

with hierarchical_regression:
    # Sample NEW commodity-specific parameters from population
    # These are not in the training data, so sample from hierarchical prior

    # Non-centered parameterization
    alpha_ethanol_raw = pm.Normal('alpha_ethanol_raw', mu=0, sigma=1)
    alpha_ethanol = pm.Deterministic('alpha_ethanol',
                                      mu_alpha + tau_alpha * alpha_ethanol_raw)

    beta_ethanol_raw = pm.Normal('beta_ethanol_raw', mu=0, sigma=1)
    beta_ethanol = pm.Deterministic('beta_ethanol',
                                     mu_beta + tau_beta * beta_ethanol_raw)

    # Predictions for ethanol
    mu_ethanol = alpha_ethanol + beta_ethanol * storage_ethanol

    # Observation model
    y_ethanol_pred = pm.Normal('y_ethanol_pred',
                                mu=mu_ethanol,
                                sigma=sigma_obs,
                                shape=n_new)

    # Sample from posterior predictive
    # This draws from p(β_ethanol | hyperparameters) and then p(y | β_ethanol, X)
    ppc_ethanol = pm.sample_posterior_predictive(trace,
                                                  var_names=['y_ethanol_pred'],
                                                  return_inferencedata=True)

# Extract predictions
y_ethanol_samples = ppc_ethanol.posterior_predictive['y_ethanol_pred']

# Summarize
y_ethanol_mean = y_ethanol_samples.mean(dim=['chain', 'draw'])
y_ethanol_hdi = az.hdi(ppc_ethanol, var_names=['y_ethanol_pred'])

print(f"Ethanol predictions: {y_ethanol_mean.values}")
print(f"95% HDI: {y_ethanol_hdi['y_ethanol_pred'].values}")
```

**Key points:**
- `alpha_ethanol` and `beta_ethanol` are NEW draws from population distribution
- Uses posterior of `mu_alpha`, `tau_alpha`, etc. from training data
- Integrates uncertainty in both population parameters AND new commodity parameters
- Result: Predictive distribution that's appropriately uncertain

**Scoring:**
- Part (a): 4 points (population sampling approach)
- Part (b): 3 points (uncertainty comparison)
- Part (c): 3 points (correct PyMC code)

---

### Question 7 (13 points)
**Variance decomposition:** In a hierarchical time series model for commodity returns, you find:

- Total variance in returns: $\sigma_{\text{total}}^2 = 0.04$
- Within-commodity variance: $\sigma_{\text{within}}^2 = 0.03$
- Between-commodity variance: $\sigma_{\text{between}}^2 = 0.01$

**(a)** Calculate the intraclass correlation coefficient (ICC). Interpret its value. (5 points)

**(b)** How does ICC relate to the strength of pooling in the hierarchical model? (4 points)

**(c)** If ICC were 0.90 instead, what would that imply for forecasting strategy? (4 points)

**Answer:**

**(a)** Intraclass Correlation Coefficient (ICC):

**Formula:**
$$\rho_{\text{ICC}} = \frac{\sigma_{\text{between}}^2}{\sigma_{\text{between}}^2 + \sigma_{\text{within}}^2} = \frac{\sigma_{\text{between}}^2}{\sigma_{\text{total}}^2}$$

**Calculation:**
$$\rho_{\text{ICC}} = \frac{0.01}{0.01 + 0.03} = \frac{0.01}{0.04} = 0.25$$

**Interpretation:**

**Statistical meaning:**
- **ICC = 0.25** means 25% of total variance is **between commodities**, 75% is **within commodities** (time variation)

**Correlation interpretation:**
- ICC is the expected correlation between two observations from the same commodity
- Two returns from same commodity: Corr = 0.25
- Two returns from different commodities: Corr ≈ 0

**Practical meaning:**
- **Moderate group structure:** Commodities have some shared characteristics but also substantial individual variation
- Returns for oil on different days are somewhat correlated (0.25)
- But most variation (75%) is due to time-specific factors within each commodity

**Example:**
- Oil return on Monday: +2%
- Oil return on Tuesday: Given Monday's return, expect some persistence (ICC = 0.25)
- But 75% of Tuesday's return is "new" variation

**(b)** ICC and strength of pooling:

**Relationship:** ICC determines how much information is borrowed across groups.

**High ICC (close to 1):**
- Most variance is **between** groups
- Groups are very different from each other
- Little benefit from pooling
- **Weak pooling:** Estimates stay close to unpooled (group-specific)

**Low ICC (close to 0):**
- Most variance is **within** groups
- Groups are similar to each other
- Large benefit from pooling
- **Strong pooling:** Estimates shrunk heavily toward population mean

**Our case (ICC = 0.25):**
- **Moderate pooling** is optimal
- Commodities share enough similarity to benefit from pooling (75% within variation means groups are not wildly different)
- But enough between-group variation (25%) to justify commodity-specific parameters

**Mathematical connection:**
The shrinkage factor in hierarchical models is approximately:
$$\lambda_j \approx \frac{n_j}{n_j + \sigma_{\text{within}}^2 / \sigma_{\text{between}}^2}$$

With ICC = 0.25:
$$\frac{\sigma_{\text{within}}^2}{\sigma_{\text{between}}^2} = \frac{0.03}{0.01} = 3$$

So shrinkage depends on sample size:
- If $n_j = 30$: $\lambda_j = 30/(30+3) = 0.91$ → trust group estimate 91%
- If $n_j = 3$: $\lambda_j = 3/(3+3) = 0.50$ → shrink 50% toward population

**Design implications:**
- ICC = 0.25 suggests hierarchical model adds value
- If ICC were 0.01 (nearly identical commodities) → complete pooling sufficient
- If ICC were 0.95 (very different commodities) → no pooling better

**(c)** If ICC = 0.90:

**Implication:** 90% of variance is **between commodities**, only 10% within.

**Interpretation:**
- Commodities are **vastly different**
- Knowing which commodity explains 90% of return variation
- Time-series dynamics within commodities are weak

**Forecasting strategy:**

**1. Minimal pooling:**
- Little to gain from hierarchical model
- Commodities too heterogeneous to share information effectively
- **Use unpooled (independent) models** for each commodity

**2. Focus on cross-sectional differences:**
- Most predictive power comes from commodity identity, not time-series patterns
- Build models that exploit differences between commodities
- Example: Factor models with commodity-specific loadings

**3. Within-commodity forecasting weak:**
- Only 10% of variance from time dynamics
- Hard to forecast individual commodity returns (weak autocorrelation)
- Forecasting which commodity is higher/lower (cross-sectional) easier

**4. Portfolio implications:**
- Strong diversification benefits (commodities move independently over time)
- Cross-commodity spread trades attractive

**Example scenario fitting ICC = 0.90:**
- Comparing oil, gold, and wheat (very different asset classes)
- Oil always around $80, gold always around $2000, wheat always around $6
- Over time, each fluctuates only ±10% (within variance small)
- Between-asset differences dominate (oil ≠ gold ≠ wheat)

**Contrast with ICC = 0.25 (our case):**
- Oil, gasoline, natural gas (related energy commodities)
- Shared energy market dynamics (within variance substantial)
- But still have commodity-specific features (between variance moderate)
- Hierarchical pooling valuable

**Scoring:**
- Part (a): 5 points (calculation + interpretation)
- Part (b): 4 points (pooling strength relationship)
- Part (c): 4 points (forecasting implications)

---

## Section C: Advanced Concepts (30 points)

### Question 8 (10 points)
**Nested hierarchies:** You want to model crude oil prices for different grades (WTI, Brent, Dubai) in different regions (North America, Europe, Middle East).

Structure:
- **Level 1 (observations):** Daily prices for each grade-region combination (9 groups total)
- **Level 2 (grades):** Grade-specific parameters
- **Level 3 (regions):** Region-specific parameters
- **Level 4 (global):** Global population parameters

**(a)** Sketch the hierarchical structure and write the model equations. (6 points)

**(b)** Why is this better than a single-level hierarchy pooling all 9 groups? (4 points)

**Answer:**

**(a)** Nested hierarchical structure:

**Visual structure:**
```
                Global Population
                    μ_global, τ_global
                    /              \
              Grade Level      Region Level
            /      |     \       /    |    \
         WTI    Brent  Dubai   NA   EU   ME
          /\      /\     /\
    WTI-NA  WTI-EU ...  Dubai-ME
     (9 grade-region combinations)
          ↓
    Daily price observations
```

**Model equations:**

**Level 4 (Global):**
$$\begin{aligned}
\mu_{\text{global}} &\sim \mathcal{N}(80, 50) \quad \text{global mean price} \\
\tau_{\text{grade}} &\sim \text{HalfNormal}(10) \quad \text{between-grade variation} \\
\tau_{\text{region}} &\sim \text{HalfNormal}(5) \quad \text{between-region variation} \\
\tau_{\text{combo}} &\sim \text{HalfNormal}(2) \quad \text{within combo (grade-region) variation}
\end{aligned}$$

**Level 3 (Grade effects):**
$$\alpha_{\text{grade}_g} \sim \mathcal{N}(\mu_{\text{global}}, \tau_{\text{grade}}^2), \quad g \in \{\text{WTI, Brent, Dubai}\}$$

**Level 3 (Region effects):**
$$\gamma_{\text{region}_r} \sim \mathcal{N}(0, \tau_{\text{region}}^2), \quad r \in \{\text{NA, EU, ME}\}$$

**Level 2 (Grade-Region combination):**
$$\delta_{g,r} \sim \mathcal{N}(0, \tau_{\text{combo}}^2)$$

**Level 1 (Observations):**
$$y_{g,r,t} \sim \mathcal{N}(\mu_{g,r,t}, \sigma_{\text{obs}}^2)$$

where:
$$\mu_{g,r,t} = \alpha_{\text{grade}_g} + \gamma_{\text{region}_r} + \delta_{g,r} + \beta \cdot X_t$$

$X_t$ could include time-varying covariates (storage, production, etc.).

**Interpretation:**
- $\alpha_{\text{grade}}$: Base price for each grade (WTI vs Brent vs Dubai quality differences)
- $\gamma_{\text{region}}$: Region premium/discount (transport costs, local demand)
- $\delta_{g,r}$: Specific grade-region interaction (beyond additive effects)

**PyMC sketch:**
```python
with pm.Model():
    # Global
    mu_global = pm.Normal('mu_global', mu=80, sigma=50)
    tau_grade = pm.HalfNormal('tau_grade', sigma=10)
    tau_region = pm.HalfNormal('tau_region', sigma=5)
    tau_combo = pm.HalfNormal('tau_combo', sigma=2)

    # Grade level (3 grades)
    alpha_grade = pm.Normal('alpha_grade', mu=mu_global, sigma=tau_grade, shape=3)

    # Region level (3 regions)
    gamma_region = pm.Normal('gamma_region', mu=0, sigma=tau_region, shape=3)

    # Combination level (9 combinations)
    delta_combo = pm.Normal('delta_combo', mu=0, sigma=tau_combo, shape=9)

    # Observation level
    mu = alpha_grade[grade_idx] + gamma_region[region_idx] + delta_combo[combo_idx]
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma_obs, observed=prices)
```

**(b)** Advantages over single-level hierarchy:

**Single-level approach (9 independent groups):**
$$\theta_{g,r} \sim \mathcal{N}(\mu_{\text{global}}, \tau^2)$$

Treats all 9 combinations as equally related.

**Nested hierarchy advantages:**

**1. Structured sharing of information:**
- **Grade similarity:** WTI-NA and WTI-EU share grade effects (both WTI)
- **Region similarity:** WTI-EU and Brent-EU share region effects (both Europe)
- Single-level: No structure, treats WTI-NA and Brent-EU as equally related to WTI-EU

**2. Interpretability:**
- Separate grade effects from region effects
- Can answer: "How much does grade matter vs region?"
- Single-level: Conflates these, just has 9 different means

**3. Forecasting new combinations:**
- New grade in existing region: Use region effect + typical grade effect
- Existing grade in new region: Use grade effect + typical region effect
- Single-level: No principled way to forecast new combinations

**4. Efficiency:**
- Nested model estimates fewer high-level parameters (3 grades + 3 regions + 9 interactions = 15 parameters)
- But with structure that shares information optimally
- Single-level: 9 independent parameters, no structure

**Example:**
- Limited data for Dubai-EU (rare trades)
- **Nested:** Borrows from Dubai-ME and Dubai-NA (same grade) AND from Brent-EU and WTI-EU (same region)
- **Single-level:** Only borrows from global mean, ignores structure

**Variance decomposition:**
- Can compute: What % of variation is due to grades? Regions? Interactions?
- Informs trading strategy (e.g., if grade dominates, focus on grade spreads)

**Scoring:**
- Part (a): 6 points (correct hierarchical equations)
- Part (b): 4 points (advantages explained)

---

### Question 9 (10 points)
**Time-varying hierarchical parameters:** Modify a standard hierarchical model to allow the population mean $\mu_{\text{pop}}$ to evolve over time as a random walk. Write the model specification and discuss implications for inference and forecasting.

**Answer:**

**Standard hierarchical model (time-invariant):**
$$\begin{aligned}
y_{j,t} &\sim \mathcal{N}(\theta_j, \sigma^2) \\
\theta_j &\sim \mathcal{N}(\mu_{\text{pop}}, \tau^2) \\
\mu_{\text{pop}} &\sim \mathcal{N}(0, \sigma_0^2) \quad \text{(fixed)}
\end{aligned}$$

**Time-varying hierarchical model:**

**Level 1 (Observations):**
$$y_{j,t} \sim \mathcal{N}(\theta_{j,t}, \sigma^2)$$

**Level 2 (Group parameters, now time-varying):**
$$\theta_{j,t} \sim \mathcal{N}(\mu_{t}, \tau^2)$$

**Level 3 (Population mean evolves):**
$$\begin{aligned}
\mu_{t+1} &= \mu_t + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_\mu^2) \\
\mu_1 &\sim \mathcal{N}(0, \sigma_0^2)
\end{aligned}$$

**Complete model:**
$$\begin{aligned}
y_{j,t} &\sim \mathcal{N}(\theta_{j,t}, \sigma^2) \\
\theta_{j,t} &\sim \mathcal{N}(\mu_t, \tau^2) \\
\mu_{t+1} &= \mu_t + \eta_t \\
\tau &\sim \text{HalfNormal}(\sigma_\tau) \\
\sigma_\mu &\sim \text{HalfNormal}(\sigma_0) \\
\sigma &\sim \text{HalfNormal}(\sigma_1)
\end{aligned}$$

**PyMC implementation:**

```python
import pymc as pm

with pm.Model() as time_varying_hierarchical:
    # Variance parameters
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=10)
    tau = pm.HalfNormal('tau', sigma=5)  # Between-group SD
    sigma_mu = pm.HalfNormal('sigma_mu', sigma=1)  # Population drift SD

    # Time-varying population mean (random walk)
    mu_init = pm.Normal('mu_init', mu=0, sigma=20)
    mu_innovations = pm.Normal('mu_innovations', mu=0, sigma=1, shape=T-1)
    mu_t = pm.Deterministic('mu_t',
                            pm.math.concatenate([[mu_init],
                                                  mu_init + pm.math.cumsum(sigma_mu * mu_innovations)]))

    # Group-specific parameters at each time (centered on mu_t)
    # Option 1: θ_j,t = μ_t + ε_j (fixed offset from time-varying mean)
    theta_offset = pm.Normal('theta_offset', mu=0, sigma=tau, shape=J)
    theta_jt = mu_t[time_idx] + theta_offset[group_idx]  # Broadcasting

    # Option 2: θ_j,t also random walk (more complex)
    # theta_jt = pm.GaussianRandomWalk('theta_jt', mu=mu_t, sigma=tau, shape=(J, T))

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=theta_jt, sigma=sigma_obs, observed=y)

    trace = pm.sample(2000, tune=1000, return_inferencedata=True)
```

**Implications:**

**1. Inference:**

**Advantage:**
- Captures structural shifts in population mean over time
- Example: Commodity supercycle (2000-2014 rising trend in all commodity prices)
- Static $\mu_{\text{pop}}$ would misspecify model

**Challenge:**
- More complex posterior (higher-dimensional state space)
- Potential for non-identification if $\sigma_\mu$ very small (collapses to static model)
- Slower MCMC sampling (need to sample $T$ values of $\mu_t$)

**2. Forecasting:**

**Advantage:**
- Can extrapolate population trend forward
- New commodities: Use forecasted $\mu_{T+h}$ as their prior mean

**Prediction at $T+h$:**
$$\begin{aligned}
\mu_{T+h} &\sim \mathcal{N}(\mu_T, h \cdot \sigma_\mu^2) \quad \text{(random walk forecasted)} \\
\theta_{j, T+h} &\sim \mathcal{N}(\mu_{T+h}, \tau^2) \quad \text{(group parameters around forecasted mean)} \\
y_{j, T+h} &\sim \mathcal{N}(\theta_{j,T+h}, \sigma^2)
\end{aligned}$$

**Uncertainty compounds:**
- Uncertainty in $\mu_T$
- Accumulation of $\mu$ innovations over $h$ steps
- Group-specific deviations $\tau$
- Observation noise $\sigma$

**3. Model comparison:**

**When to use time-varying $\mu_t$?**

**Use if:**
- Long time series with structural shifts
- Evidence of common trends across commodities (cointegration)
- Forecasting requires extrapolating trend

**Don't use if:**
- Short time series (overparameterization)
- Population mean is truly stable
- Increases computational cost without benefit

**Diagnostic:**
- Examine posterior of $\sigma_\mu$
- If $\sigma_\mu \approx 0$: No time variation needed, use static model
- If $\sigma_\mu > 0.05 \times \text{scale}(y)$: Time variation substantial

**Scoring:**
- Model specification: 4 points
- PyMC implementation: 3 points
- Inference implications: 2 points
- Forecasting implications: 1 point

---

### Question 10 (10 points)
**Model criticism:** After fitting a hierarchical model to commodity returns, you perform posterior predictive checks. The observed data shows occasional extreme returns (>5 standard deviations), but posterior predictive samples rarely generate such extremes. What does this diagnostic suggest, and how would you address it?

**Answer:**

**Diagnostic interpretation:**

**Problem: Model underestimates tail risk**

The hierarchical Gaussian model:
$$y_{j,t} \sim \mathcal{N}(\theta_j, \sigma^2)$$

assumes returns are **normally distributed**. But observed data has **fat tails** (extreme events more common than Gaussian predicts).

**Evidence:**
- Observed: Returns occasionally >5σ
- Posterior predictive: Such extremes rare
- **Implication:** Model does not capture true data-generating process

**Why this happens:**
- Commodity returns exhibit **volatility clustering** and **jumps**
- Normal distribution: Pr(|Z| > 5) ≈ 0.0000006 (extremely rare)
- Reality: Crises, supply shocks, policy changes cause fat tails

**Solutions:**

### **1. Use Student-t Distribution (Robust Likelihood)**

Replace Gaussian with Student-t:
$$y_{j,t} \sim t_\nu(\theta_j, \sigma^2)$$

where $\nu$ = degrees of freedom parameter.

**Properties:**
- $\nu \to \infty$: Approaches Normal
- $\nu < 10$: Fat tails (allows extreme observations)
- Estimate $\nu$ from data

**PyMC:**
```python
with pm.Model():
    # Hierarchical structure (same as before)
    theta_j = pm.Normal('theta_j', mu=mu_pop, sigma=tau, shape=J)

    # Robust observation model
    nu = pm.Exponential('nu', lam=1/10)  # Prior on degrees of freedom
    y_obs = pm.StudentT('y_obs', nu=nu, mu=theta_j[group_idx], sigma=sigma, observed=y)
```

**Advantage:**
- Automatically downweights extreme observations (treats as outliers)
- Posterior predictive will match observed tail behavior

---

### **2. Mixture Model (Outlier Component)**

Model returns as mixture of normal and outlier process:
$$y_{j,t} \sim \pi \cdot \mathcal{N}(\theta_j, \sigma^2) + (1-\pi) \cdot \mathcal{N}(\theta_j, \sigma_{\text{outlier}}^2)$$

where $\sigma_{\text{outlier}} \gg \sigma$ (wide outlier distribution).

**Interpretation:**
- Probability $\pi$: Normal market conditions
- Probability $1-\pi$: Crisis/shock (fat tails)

**PyMC:**
```python
with pm.Model():
    # Hierarchical parameters
    theta_j = pm.Normal('theta_j', mu=mu_pop, sigma=tau, shape=J)

    # Mixture weights
    pi = pm.Beta('pi', alpha=9, beta=1)  # Prior: 90% normal, 10% outlier

    # Normal component
    normal_component = pm.Normal.dist(mu=theta_j[group_idx], sigma=sigma)

    # Outlier component (wide variance)
    outlier_component = pm.Normal.dist(mu=theta_j[group_idx], sigma=sigma_outlier)

    # Mixture
    y_obs = pm.Mixture('y_obs', w=[pi, 1-pi],
                       comp_dists=[normal_component, outlier_component],
                       observed=y)
```

---

### **3. Time-Varying Volatility (Stochastic Volatility)**

Allow $\sigma$ to change over time:
$$\begin{aligned}
y_{j,t} &\sim \mathcal{N}(\theta_j, \exp(h_{j,t})) \\
h_{j,t+1} &= \mu_j + \phi_j(h_{j,t} - \mu_j) + \eta_{j,t}
\end{aligned}$$

**Captures:**
- Volatility clustering (high volatility persists)
- Extreme returns coincide with high volatility periods

---

### **4. Regularized Horseshoe Prior (Sparse Outliers)**

Use horseshoe prior to identify specific time points as outliers:
$$\begin{aligned}
y_{j,t} &\sim \mathcal{N}(\theta_j + \delta_t, \sigma^2) \\
\delta_t &\sim \text{Horseshoe}(\tau) \quad \text{sparse outlier adjustments}
\end{aligned}$$

Most $\delta_t \approx 0$ (normal), a few $\delta_t$ large (outliers).

---

**Model comparison:**

Fit multiple models and compare:
- **WAIC/LOO:** Quantify predictive accuracy
- **Posterior predictive checks:** Do new models generate realistic extremes?
- **QQ plots:** Compare empirical vs predicted quantiles (especially tails)

**Check:**
```python
import arviz as az

# Posterior predictive samples
ppc = pm.sample_posterior_predictive(trace)

# Graphical check
az.plot_ppc(ppc, num_pp_samples=100)

# Quantile check
observed_quantiles = np.percentile(y, [1, 5, 95, 99])
predicted_quantiles = np.percentile(ppc['y_obs'], [1, 5, 95, 99], axis=(0,1))

print(f"Observed 1%/99% quantiles: {observed_quantiles[[0,3]]}")
print(f"Predicted 1%/99% quantiles: {predicted_quantiles[[0,3]]}")
# Should match for good model
```

**Conclusion:**
- Student-t is simplest and often sufficient
- Mixture model if distinct "crisis" vs "normal" regimes
- Stochastic volatility if high-frequency data with clustering

**Scoring:**
- Problem diagnosis: 3 points
- Student-t solution: 3 points
- Alternative approaches: 2 points
- Model comparison strategy: 2 points

---

## Answer Key Summary

| Question | Points | Topic |
|----------|--------|-------|
| 1 | 10 | Pooling strategies |
| 2 | 8 | Hierarchical volatility model |
| 3 | 9 | Shrinkage mechanics |
| 4 | 8 | PyMC hierarchical regression |
| 5 | 12 | Storage effects, model comparison |
| 6 | 10 | Forecasting new commodities |
| 7 | 13 | Intraclass correlation |
| 8 | 10 | Nested hierarchies |
| 9 | 10 | Time-varying hyperparameters |
| 10 | 10 | Posterior predictive checks, robust models |
| **Total** | **100** | |

---

## Grading Rubric

**A (90-100):** Deep understanding of hierarchical modeling philosophy, technical implementation in PyMC, and practical commodity applications. Can design custom hierarchical structures.

**B (80-89):** Solid grasp of partial pooling concepts and standard hierarchical models. May struggle with nested structures or time-varying parameters.

**C (70-79):** Basic competency in hierarchical regression but lacks depth in shrinkage mechanics or model criticism.

**D (60-69):** Significant gaps in understanding. Confuses pooling strategies or cannot implement hierarchical models.

**F (<60):** Does not demonstrate minimum competency in hierarchical modeling.

---

**Study Resources:**
- Gelman & Hill (2006): *Data Analysis Using Regression and Multilevel/Hierarchical Models*
- McElreath (2020): *Statistical Rethinking* - Chapter 13 (Multilevel Models)
- PyMC Documentation: Hierarchical Models examples
- Module notebooks: `01_partial_pooling.ipynb`, `02_hierarchical_commodities.ipynb`

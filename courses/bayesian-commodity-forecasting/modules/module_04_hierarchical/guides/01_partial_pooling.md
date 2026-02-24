# Partial Pooling: The Logic of Hierarchical Models

## In Brief

Partial pooling is a compromise between treating each commodity independently (no pooling) and treating all commodities identically (complete pooling). It enables information sharing while respecting individual differences.

> 💡 **Key Insight:** **Shrinkage is intelligent borrowing.** When a commodity has limited data, its estimates are "shrunk" toward the group mean. When it has abundant data, it stands on its own. The model automatically determines how much to shrink based on data quality and group coherence.

---

## Formal Definition

### Hierarchical Model Structure

**Level 1 (Observation):**
$$y_{ij} \sim \mathcal{N}(\mu_j, \sigma^2)$$

**Level 2 (Group):**
$$\mu_j \sim \mathcal{N}(\mu_0, \tau^2)$$

**Level 3 (Hyperprior):**
$$\mu_0 \sim \mathcal{N}(m, s^2)$$
$$\tau \sim \text{HalfNormal}(\sigma_\tau)$$

Where:
- $y_{ij}$: Observation $i$ for commodity $j$
- $\mu_j$: Mean for commodity $j$
- $\mu_0$: Group mean (shared across commodities)
- $\tau$: Group standard deviation (controls pooling)

---

## The Three Pooling Regimes

### 1. No Pooling (Complete Separation)

Each commodity estimated independently:

$$\mu_j = \bar{y}_j$$

**Pros:** Captures individual commodity characteristics
**Cons:** High variance, especially with limited data
**When to use:** Commodities are truly unrelated

### 2. Complete Pooling (Ignore Groups)

All commodities share one parameter:

$$\mu_j = \mu \quad \forall j$$

**Pros:** Low variance, stable estimates
**Cons:** Ignores important differences between commodities
**When to use:** Commodities are nearly identical

### 3. Partial Pooling (Hierarchical)

Compromise between extremes:

$$\mu_j = \lambda_j \bar{y}_j + (1 - \lambda_j) \mu_0$$

Where $\lambda_j$ is the **shrinkage factor** for commodity $j$.

**The shrinkage factor depends on:**
- Sample size for commodity $j$
- Variance within commodity $j$
- Group variance $\tau^2$

---

## Mathematical Derivation of Shrinkage

For the Normal-Normal hierarchical model, the posterior mean for commodity $j$ is:

$$\hat{\mu}_j = \frac{\frac{n_j}{\sigma^2}}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}} \bar{y}_j + \frac{\frac{1}{\tau^2}}{\frac{n_j}{\sigma^2} + \frac{1}{\tau^2}} \mu_0$$

This simplifies to:

$$\hat{\mu}_j = \lambda_j \bar{y}_j + (1 - \lambda_j) \mu_0$$

Where:

$$\lambda_j = \frac{n_j / \sigma^2}{n_j / \sigma^2 + 1 / \tau^2} = \frac{n_j \tau^2}{n_j \tau^2 + \sigma^2}$$

### Shrinkage Properties

| Condition | $\lambda_j$ | Behavior |
|-----------|-------------|----------|
| Large $n_j$ | → 1 | Trust data (less shrinkage) |
| Small $n_j$ | → 0 | Trust group (more shrinkage) |
| Large $\tau^2$ | → 1 | Groups differ (less pooling) |
| Small $\tau^2$ | → 0 | Groups similar (more pooling) |

---

## Visual Intuition

### Shrinkage Diagram

```
Individual         Shrinkage          Group
Estimates          Direction          Mean
    •                  →                │
        •              →                │
    •                  →                │
            •          →                │
        •              →                │
                                        │
                                       μ₀

Commodities with extreme estimates or small samples
are pulled more strongly toward the group mean.
```

### Bias-Variance Tradeoff

```
                    Prediction Error
                         │
                         │   Total Error
                         │    ╱╲
                         │   ╱  ╲
                         │  ╱    ╲
                         │ ╱      ╲────── Variance
                         │╱        ╲
                         │──────────╲──── Bias²
                         │           ╲
                         └────────────────────
                         No        Partial   Complete
                         Pooling   Pooling   Pooling
```

Partial pooling minimizes total prediction error.

---

## Code Example

```python
import pymc as pm
import numpy as np
import arviz as az

# Simulated data: 5 commodities with different sample sizes
np.random.seed(42)
n_commodities = 5
true_group_mean = 80
true_group_std = 10
true_obs_std = 5

# Generate true commodity means
true_means = np.random.normal(true_group_mean, true_group_std, n_commodities)

# Generate observations (different sample sizes)
sample_sizes = [10, 20, 50, 100, 200]
observations = []
commodity_idx = []

for j, (mu, n) in enumerate(zip(true_means, sample_sizes)):
    obs = np.random.normal(mu, true_obs_std, n)
    observations.extend(obs)
    commodity_idx.extend([j] * n)

y = np.array(observations)
idx = np.array(commodity_idx)

# Hierarchical model
with pm.Model() as hierarchical_model:
    # Hyperpriors
    mu_group = pm.Normal('mu_group', mu=80, sigma=20)
    sigma_group = pm.HalfNormal('sigma_group', sigma=15)

    # Group-level priors (one per commodity)
    mu_commodity = pm.Normal('mu_commodity', mu=mu_group, sigma=sigma_group,
                              shape=n_commodities)

    # Observation noise
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=10)

    # Likelihood
    y_obs = pm.Normal('y_obs', mu=mu_commodity[idx], sigma=sigma_obs, observed=y)

    # Sample
    trace = pm.sample(2000, tune=1000, random_seed=42)

# Compare estimates
print("Comparison of Estimates:")
print("-" * 50)
print(f"{'Commodity':<12} {'True':<10} {'Sample Mean':<12} {'Hierarchical':<12}")
print("-" * 50)

for j in range(n_commodities):
    sample_mean = y[idx == j].mean()
    hier_mean = trace.posterior['mu_commodity'].values[:, :, j].mean()
    print(f"{j:<12} {true_means[j]:<10.2f} {sample_mean:<12.2f} {hier_mean:<12.2f}")
```

---

## When Hierarchical Models Excel

### 1. Small Sample Problems
New commodity contracts with limited history benefit from established contracts.

### 2. Heterogeneous Data Quality
Some commodities have reliable data, others don't. Hierarchical models adapt.

### 3. Related Group Structure
Energy products, grain complex, precious metals—natural groupings exist.

### 4. Regularization Needs
Hierarchical priors provide automatic regularization, reducing overfitting.

---

## Common Pitfalls

### 1. Ignoring Group Structure
Not using hierarchical models when data has natural groupings leaves information on the table.

### 2. Wrong Grouping
Grouping unrelated commodities (e.g., gold with corn) adds noise rather than information.

### 3. Overparameterization
Too many hierarchy levels with limited data leads to non-identifiability.

### 4. Forgetting to Check Shrinkage
Always examine how much shrinkage occurred—extreme shrinkage may indicate model misspecification.

---

## Connections

**Builds on:**
- Module 1: Prior-posterior updating
- Module 3: State space models (hierarchical state evolution)

**Leads to:**
- Module 8: Hierarchical fundamental models
- Capstone: Multi-commodity forecasting systems

---

## Practice Problems

1. A new biofuels contract has only 20 observations. Existing ethanol contracts have 500+ observations each. How would hierarchical modeling help forecast the new contract?

2. You're modeling the grain complex (corn, wheat, soybeans). What aspects might share parameters (partial pooling) vs. remain commodity-specific (no pooling)?

3. Derive the shrinkage factor $\lambda_j$ for a commodity with $n_j = 50$ observations, assuming $\sigma^2 = 25$ and $\tau^2 = 100$.

---

## Further Reading

1. **Gelman & Hill** *Data Analysis Using Regression and Multilevel/Hierarchical Models* — Definitive reference
2. **McElreath** *Statistical Rethinking* Chapter 13 — Excellent intuitive treatment
3. **Kruschke** *Doing Bayesian Data Analysis* Chapter 9 — Worked examples

---

*"Hierarchical models are not about assuming commodities are the same—they're about learning how different they actually are."*

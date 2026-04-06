# Bayesian ITS: Why Bayes for Causal Inference

> **Reading time:** ~8 min | **Module:** 2 — Bayesian Its | **Prerequisites:** Module 1 — ITS Fundamentals, basic Bayesian concepts

## In Brief

The Bayesian approach to ITS models uncertainty at every level: over parameters, over the counterfactual trajectory, and over the causal effect. Instead of a point estimate with a standard error, you get a full probability distribution over the quantity of interest. This transforms causal inference from "reject the null" to "quantify the evidence."

<div class="callout-key">

<strong>Key Concept:</strong> The Bayesian approach to ITS models uncertainty at every level: over parameters, over the counterfactual trajectory, and over the causal effect. Instead of a point estimate with a standard error, you get a full probability distribution over the quantity of interest.

</div>

## Key Insight

The fundamental advantage of Bayesian ITS is not computational — it is epistemic. Bayesian inference forces you to express your prior knowledge about the treatment effect before seeing the post-intervention data, then updates that belief with data. The posterior is a coherent probability statement: "Given the data and my prior knowledge, what is my credence that the effect is positive?"

---

## Why Bayesian for ITS?

### 1. Uncertainty About the Counterfactual

In frequentist ITS, the standard error of the causal effect is computed from the regression standard errors using the delta method. This gives a single number. But the counterfactual trajectory has two sources of uncertainty:

1. **Parameter uncertainty:** We do not know the true pre-intervention intercept and slope. The fitted values are estimates with associated uncertainty.
2. **Future variance:** Even if we knew the true parameters perfectly, the outcome would still vary around the regression line due to irreducible noise.

The Bayesian posterior propagates **both** sources of uncertainty into the causal effect estimate. The result is correctly calibrated posterior predictive intervals.

### 2. Natural Probability Statements

Frequentist hypothesis testing answers: "What is the probability of observing data this extreme if the null hypothesis were true?" (p-value).

Bayesian inference answers: "Given the data, what is the probability that the treatment effect is positive?" ($P(\tau > 0 | \text{data})$).

The Bayesian statement is what practitioners actually want. A p-value of 0.03 does not mean there is a 97% chance the effect is real — that interpretation is technically wrong but ubiquitous. The Bayesian $P(\tau > 0 | \text{data}) = 97\%$ does mean what it says.

### 3. Prior Information as Domain Knowledge

Domain experts have knowledge about the magnitude of plausible effects before seeing the data. A smoking ban study might have prior evidence from other cities or other interventions. Bayesian priors allow this information to be formally incorporated.

**Weakly informative priors:** When prior information is vague, weakly informative priors regularize the estimates slightly, preventing extreme values caused by numerical instability. This is better than flat priors, which can cause improper posteriors.

**Informative priors:** When previous studies provide strong guidance (e.g., meta-analyses), informative priors can substantially improve precision, especially in small samples.

### 4. Handling Small Samples

ITS often has limited post-intervention data. With only 6-12 post-intervention observations, frequentist estimates have wide standard errors and p-values are uninformative. Bayesian priors can provide regularization that makes the estimates more stable without biasing them toward zero (as ridge regression does).

### 5. Model Comparison

Bayesian model comparison using LOO-CV (Leave-One-Out Cross-Validation) via ArviZ gives a principled way to compare model specifications (e.g., full vs. level-only vs. slope-only models) without requiring the models to be nested.

---

## The Bayesian ITS Model Formally

### Likelihood

$$Y_t \sim \mathcal{N}(\mu_t, \sigma^2)$$

$$\mu_t = \alpha + \beta_1 t + \beta_2 D_t + \beta_3 (t - t^*) D_t$$

### Priors

With weakly informative priors:

$$\alpha \sim \mathcal{N}(\bar{Y}, 10 \cdot \text{sd}(Y))$$

$$\beta_1 \sim \mathcal{N}(0, 1)$$

$$\beta_2 \sim \mathcal{N}(0, 10 \cdot \text{sd}(Y))$$

$$\beta_3 \sim \mathcal{N}(0, 1)$$

$$\sigma \sim \text{HalfNormal}(10 \cdot \text{sd}(Y))$$

### Posterior

By Bayes' theorem:

$$P(\alpha, \beta_1, \beta_2, \beta_3, \sigma | Y, t) \propto P(Y | \alpha, \beta_1, \beta_2, \beta_3, \sigma, t) \cdot P(\alpha, \beta_1, \beta_2, \beta_3, \sigma)$$

The posterior is not analytically tractable (it is a high-dimensional integral), so MCMC sampling (specifically NUTS — the No-U-Turn Sampler) generates draws from it.

### Posterior of the Causal Effect

From the posterior samples, the causal effect at each post-intervention time $t$ is:

$$P(\tau_t | \text{data}) = P(\beta_2 + \beta_3 (t - t^*) | \text{data})$$

This is computed by sampling from the joint posterior of $(\beta_2, \beta_3)$ and evaluating $\beta_2 + \beta_3 (t - t^*)$ for each sample. The result is a full distribution — not just a point estimate.

---

## Uncertainty Quantification: HDI vs Confidence Interval

### Confidence Interval (Frequentist)

A 95% confidence interval says: "If we repeated this experiment infinitely many times with different data, 95% of the constructed intervals would contain the true parameter."

This is a statement about the procedure, not about the parameter. The parameter is fixed (frequentist) — it is either in the interval or not.

### Highest Density Interval (Bayesian)

A 94% HDI says: "Given the data and our priors, we assign 94% probability to the parameter being within this interval."

This is a direct probability statement about the parameter. The HDI contains the most probable values — unlike a confidence interval, it is the narrowest interval containing the specified probability.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import arviz as az
import numpy as np

# Posterior samples
beta_samples = np.random.normal(loc=-5, scale=2, size=4000)

# Compute HDI
hdi = az.hdi(beta_samples, hdi_prob=0.94)
print(f"94% HDI: [{hdi[0]:.2f}, {hdi[1]:.2f}]")

# Direct probability statement
p_negative = (beta_samples < 0).mean()
print(f"P(effect < 0) = {p_negative:.1%}")
```

</div>
</div>

---

## Counterfactual Distribution

The Bayesian counterfactual is a **distribution** over what would have happened. For each post-intervention time $t$:

$$P(Y_t(0) | \text{pre-period data}) = P(\hat{\alpha} + \hat{\beta}_1 t | \text{data})$$

This distribution:
- Is centered on the extrapolated pre-trend line
- Has width determined by the posterior uncertainty on $(\alpha, \beta_1)$
- Widens the further we extrapolate from the last pre-period observation

The causal effect distribution is:

$$P(\tau_t | \text{data}) = P(Y_t(1) - Y_t(0) | \text{data}) = P(Y_t^{obs} - Y_t(0) | \text{data})$$

where $Y_t^{obs}$ is the observed outcome and $Y_t(0)$ is the counterfactual.

---

## Prior Predictive Checks

Before fitting the model to data, check that the prior is reasonable by sampling from it and examining the implied distribution of outcomes.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

def prior_predictive_check_its(y_scale, n_obs):
    """
    Draw from the prior predictive distribution for an ITS model.
    Use this to verify the prior is reasonable before fitting.

    Parameters
    ----------
    y_scale : float
        Rough scale of the outcome variable (e.g., mean or SD)
    n_obs : int
        Number of observations
    """
    t = np.arange(n_obs)
    t_star = n_obs // 2
    treated = (t >= t_star).astype(float)
    t_post = np.maximum(t - t_star, 0).astype(float)

    with pm.Model() as prior_model:
        # Weakly informative priors
        alpha = pm.Normal("alpha", mu=y_scale, sigma=y_scale * 2)
        beta_trend = pm.Normal("beta_trend", mu=0, sigma=y_scale * 0.1)
        beta_level = pm.Normal("beta_level", mu=0, sigma=y_scale)
        beta_slope = pm.Normal("beta_slope", mu=0, sigma=y_scale * 0.05)
        sigma = pm.HalfNormal("sigma", sigma=y_scale * 0.5)

        # Prior predictive
        mu = alpha + beta_trend * t + beta_level * treated + beta_slope * t_post
        y_prior = pm.Normal("y_prior", mu=mu, sigma=sigma)

        prior_pred = pm.sample_prior_predictive(samples=200, random_seed=42)

    fig, ax = plt.subplots(figsize=(12, 5))
    prior_samples = prior_pred.prior_predictive["y_prior"].values.squeeze()

    for i in range(min(50, prior_samples.shape[0])):
        ax.plot(t, prior_samples[i], alpha=0.1, color="#3498db")

    ax.axvline(t_star, color="red", linestyle="--", label="Intervention")
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome (prior predictive)")
    ax.set_title("Prior Predictive Check: Are These Plausible Outcomes?")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print("If the prior predictive trajectories span plausible outcome ranges, the prior is acceptable.")
    print("If they span unreasonably extreme values (e.g., negative counts, impossibly large effects),")
    print("tighten the priors.")
```

</div>
</div>

---

## When to Use Informative vs Weakly Informative Priors

### Use Weakly Informative Priors When:
- Little prior evidence exists about the effect size
- The outcome scale is standardized or normalized
- You want the data to dominate the inference
- You are presenting results to a skeptical audience

### Use Informative Priors When:
- Meta-analysis or previous studies provide reliable effect size estimates
- The sample size is small (prior provides regularization)
- The direction of the effect is known with high confidence (e.g., a policy that can only reduce, not increase, an outcome)
- You want to explicitly test sensitivity to the prior

### Prior Sensitivity Analysis

Always report how sensitive conclusions are to the prior specification. If the posterior changes dramatically when you change the prior, either:
1. The data are uninformative (small sample, weak signal) — the prior matters a lot
2. The prior is too strong — it is overriding the data

A robust finding is one that is stable across a range of reasonable priors.

---

## The NUTS Sampler

CausalPy uses PyMC, which uses the **No-U-Turn Sampler (NUTS)** by default. NUTS is a variant of Hamiltonian Monte Carlo (HMC) that:

1. Uses gradient information to make large jumps in parameter space
2. Automatically tunes the step size and trajectory length
3. Dramatically reduces autocorrelation between samples compared to random-walk MCMC
4. Scales well to models with many parameters

**Key sampling parameters:**
- `draws`: Number of posterior samples per chain (after warmup)
- `tune`: Warmup iterations for adaptation (should be at least 500)
- `chains`: Number of parallel chains (at least 4 for R-hat assessment)
- `target_accept`: NUTS target acceptance rate (0.8 default, 0.9-0.95 for difficult models)

**When NUTS has trouble:**
- **Many divergences:** Model geometry is difficult — try reparameterization or stronger priors
- **Low ESS:** Chains are correlated — increase `draws` or `tune`
- **High R-hat:** Chains did not explore the same distribution — increase `tune`

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

- **Builds on:** Potential outcomes (Module 00), ITS fundamentals (Module 01)
- **Leads to:** PyMC internals (Guide 2), Prior specification (Guide 3)
- **Related to:** Bayesian statistics, MCMC, ArviZ


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Bayesian ITS: Why Bayes for Causal Inference and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Gelman, A. et al. (2013). *Bayesian Data Analysis* (3rd ed.) — the definitive Bayesian reference
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.) — accessible Bayesian modeling
- Vehtari, A. et al. (2021). "Rank-normalization, folding, and localization: An improved R-hat for assessing convergence." *Bayesian Analysis*
- Betancourt, M. (2017). "A Conceptual Introduction to Hamiltonian Monte Carlo." ArXiv


## Resources

<a class="link-card" href="../notebooks/01_its_from_scratch_pymc.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

# Prior Specification for Bayesian ITS

> **Reading time:** ~7 min | **Module:** 2 — Bayesian Its | **Prerequisites:** Module 1 — ITS Fundamentals, basic Bayesian concepts

## In Brief

Choosing priors is the most subjective step in Bayesian modeling, and therefore the most important to get right. A prior encodes your beliefs about parameter values before seeing the post-intervention data. The goal is to be honest: neither too restrictive (overriding the data) nor too vague (allowing physically impossible values).

<div class="callout-key">

<strong>Key Concept:</strong> Choosing priors is the most subjective step in Bayesian modeling, and therefore the most important to get right. A prior encodes your beliefs about parameter values before seeing the post-intervention data.

</div>

## Key Insight

The best prior for an ITS analysis is one that: (1) centers on a plausible value based on domain knowledge, (2) has a width that spans the full range of plausible effects, and (3) prevents the sampler from exploring physically impossible parameter values (negative counts, effects larger than the outcome range).

---

## Why Prior Choice Matters

### In Large Samples

With 50+ pre-intervention observations and a clear effect, the posterior is dominated by the likelihood. The prior has minimal impact. Any reasonable weakly informative prior gives virtually the same posterior.

### In Small Samples

With fewer than 20 observations, the prior can substantially influence the posterior. A prior centered far from the true effect will bias estimates toward zero (or toward the prior mean). Informative priors from previous studies help regularize in this regime.

### For the Intercept

The intercept represents the outcome level at $t = 0$. If $t = 0$ is far from the observed data range, the intercept posterior may be highly uncertain. **Fix:** Center the time variable: use $(t - t^*)$ instead of $t$. Then the intercept represents the outcome at the intervention point — much more meaningful and stable.

### For the Effect Size Parameters

$\beta_2$ (level change) and $\beta_3$ (slope change) are the causal parameters of interest. The prior should be centered on zero (no assumption about direction) with a scale proportional to the typical variation in the outcome:

- If the outcome has standard deviation $\sigma_Y \approx 10$, a prior $\beta_2 \sim \mathcal{N}(0, 10)$ says "the level change is probably within ±20 units"
- A prior $\beta_2 \sim \mathcal{N}(0, 1000)$ says "the level change could be anything up to ±2000 units" — too vague, may cause sampling problems

---

## A Principled Approach to Prior Setting

### Step 1: Identify the Outcome Scale

Compute the standard deviation and range of the pre-intervention outcome:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
pre_std = df.loc[df["treated"] == 0, "outcome"].std()
pre_mean = df.loc[df["treated"] == 0, "outcome"].mean()
pre_range = df.loc[df["treated"] == 0, "outcome"].max() - df.loc[df["treated"] == 0, "outcome"].min()

print(f"Pre-intervention mean: {pre_mean:.1f}")
print(f"Pre-intervention std: {pre_std:.1f}")
print(f"Pre-intervention range: {pre_range:.1f}")
```

</div>
</div>

### Step 2: Define Plausible Effect Bounds

Ask domain experts: "What is the largest plausible effect this intervention could have?"

For a smoking ban on AMI rates:
- The AMI rate is around 85 cases / 100k / month
- The largest published effect from other cities is a 20% reduction ≈ 17 cases
- A prior of $\beta_2 \sim \mathcal{N}(0, 10)$ puts 95% of the prior mass within ±20 cases — reasonable

### Step 3: Run a Prior Predictive Check

Before fitting, sample from the prior and generate hypothetical outcomes. Ask: "Do these look like plausible data from my system?"


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

def prior_predictive_check(
    n_pre: int,
    n_post: int,
    prior_alpha_mu: float,
    prior_alpha_sigma: float,
    prior_beta1_sigma: float,
    prior_beta2_sigma: float,
    prior_beta3_sigma: float,
    prior_sigma_sigma: float,
    n_samples: int = 100,
):
    """
    Run a prior predictive check for an ITS model.

    Parameters correspond to Normal(0, sigma) priors on each coefficient.
    prior_alpha_mu and prior_alpha_sigma define the intercept prior.
    """
    n_total = n_pre + n_post
    t = np.arange(n_total)
    treated = (t >= n_pre).astype(float)
    t_post = np.maximum(t - n_pre, 0).astype(float)

    with pm.Model() as ppc_model:
        alpha = pm.Normal("alpha", mu=prior_alpha_mu, sigma=prior_alpha_sigma)
        beta1 = pm.Normal("beta1", mu=0, sigma=prior_beta1_sigma)
        beta2 = pm.Normal("beta2", mu=0, sigma=prior_beta2_sigma)
        beta3 = pm.Normal("beta3", mu=0, sigma=prior_beta3_sigma)
        sigma = pm.HalfNormal("sigma", sigma=prior_sigma_sigma)

        mu = alpha + beta1 * t + beta2 * treated + beta3 * t_post
        y_prior = pm.Normal("y_prior", mu=mu, sigma=sigma)

        prior_samples = pm.sample_prior_predictive(samples=n_samples, random_seed=42)

    y_prior_samples = prior_samples.prior_predictive["y_prior"].values.squeeze()

    fig, ax = plt.subplots(figsize=(12, 5))
    for i in range(min(50, y_prior_samples.shape[0])):
        ax.plot(t, y_prior_samples[i], alpha=0.15, color="#3498db", linewidth=0.8)
    ax.axvline(n_pre, color="red", linestyle="--", linewidth=2, label="Intervention")
    ax.set_xlabel("Time")
    ax.set_ylabel("Outcome (prior predictive)")
    ax.set_title("Prior Predictive Check")
    ax.legend()
    plt.tight_layout()
    plt.show()

    print(f"Prior predictive range: [{y_prior_samples.min():.1f}, {y_prior_samples.max():.1f}]")
    print(f"Prior predictive mean: {y_prior_samples.mean():.1f}")
    print("Ask: Do these trajectories span plausible outcome values?")
```

</div>
</div>

### Step 4: Refine and Iterate

If the prior predictive shows:
- Negative outcomes where the outcome must be positive → tighten the intercept prior or use a log-normal model
- Effects larger than the entire outcome range → tighten the beta_2 prior
- Effects so small they are undetectable → the prior may be too tight (regularizing toward zero)

---

## Prior Specifications by Parameter

### Intercept ($\alpha$)

**Recommendation:** Center at the pre-intervention mean, with sigma equal to the pre-intervention standard deviation times 2-3.

```python
prior_alpha = pm.Normal("alpha", mu=y_pre_mean, sigma=2 * y_pre_std)
```

**Why:** The intercept is the outcome at $t = 0$. Centering it near the observed data prevents the sampler from exploring implausible regions. The sigma of 2× the data SD allows for substantial uncertainty while preventing extreme values.

### Pre-Intervention Slope ($\beta_1$)

**Recommendation:** Centered at 0 with sigma equal to the typical monthly change (which is roughly $y_{std} / \sqrt{n_{pre}}$).

```python
typical_monthly_change = y_pre_std / np.sqrt(n_pre)
prior_beta1 = pm.Normal("beta1", mu=0, sigma=typical_monthly_change * 2)
```

### Level Change ($\beta_2$)

**Recommendation:** Centered at 0 (no assumed direction), sigma equal to 1× the pre-intervention standard deviation.

```python
prior_beta2 = pm.Normal("beta2", mu=0, sigma=y_pre_std)
```

If domain knowledge strongly suggests the direction (e.g., a safety intervention can only reduce accidents):

```python

# Informative prior: effect is probably negative, within 0 to -2*sigma
prior_beta2 = pm.Normal("beta2", mu=-y_pre_std / 2, sigma=y_pre_std)
```

### Slope Change ($\beta_3$)

**Recommendation:** Smaller sigma than beta_2, since slope changes are typically smaller effects than level changes.

```python
prior_beta3 = pm.Normal("beta3", mu=0, sigma=y_pre_std * 0.1)
```

### Noise ($\sigma$)

**Recommendation:** HalfNormal with sigma proportional to the data standard deviation.

```python
prior_sigma = pm.HalfNormal("sigma", sigma=y_pre_std)
```

---

## Prior Sensitivity Analysis

A robust causal conclusion should be stable under perturbations to the prior. Test three prior specifications:

1. **Tight prior:** $\beta_2 \sim \mathcal{N}(0, \sigma_Y / 2)$ — strongly regularizes toward zero
2. **Default prior:** $\beta_2 \sim \mathcal{N}(0, \sigma_Y)$ — weakly informative
3. **Diffuse prior:** $\beta_2 \sim \mathcal{N}(0, 3\sigma_Y)$ — nearly flat

If all three give similar posteriors, the result is prior-robust. If they differ substantially, either:
- The data are not informative enough to overcome the prior
- The prior is genuinely strong (informative)


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import causalpy as cp
import pymc as pm
import arviz as az
import numpy as np

def its_with_custom_beta2_prior(df, t_star, beta2_sigma, y_pre_std):
    """Fit ITS with a specific prior on the level change parameter."""

    class CustomPrior(cp.pymc_models.LinearRegression):
        def build_model(self, X, y, coords):
            with pm.Model(coords=coords) as self.model:
                X_ = pm.Data("X", X, dims=["obs", "coeffs"])
                y_ = pm.Data("y_obs", y, dims=["obs"])

                # Variable-width prior on beta_2
                n = X.shape[1]
                col_names = list(X.columns)
                treated_idx = col_names.index("treated")

                betas_other = pm.Normal("beta_other", mu=0, sigma=10, shape=n - 1)
                beta_treated = pm.Normal("beta_treated", mu=0, sigma=beta2_sigma)

                # Reconstruct beta vector in correct order
                beta_list = []
                j = 0
                for i in range(n):
                    if i == treated_idx:
                        beta_list.append(beta_treated)
                    else:
                        beta_list.append(betas_other[j])
                        j += 1

                beta = pm.math.stack(beta_list)
                sigma = pm.HalfNormal("sigma", sigma=y_pre_std)
                mu = pm.Deterministic("mu", pm.math.dot(X_, beta))
                y_hat = pm.Normal("y_hat", mu=mu, sigma=sigma, observed=y_, dims=["obs"])
            return self.model

    return cp.InterruptedTimeSeries(
        data=df,
        treatment_time=t_star,
        formula="outcome ~ 1 + t + treated + t_post",
        model=CustomPrior(
            sample_kwargs={"draws": 500, "tune": 500, "chains": 2, "random_seed": 42}
        ),
    )
```


---

## Reporting Prior Sensitivity Results

A sensitivity analysis report should include:

| Prior Specification | $\beta_2$ Posterior Mean | 94% HDI | P(β₂ < 0) |
|--------------------|--------------------------|---------|------------|
| Tight: N(0, 5) | −7.2 | [−12.4, −2.0] | 99.1% |
| Default: N(0, 10) | −7.8 | [−13.1, −2.5] | 99.4% |
| Diffuse: N(0, 30) | −8.1 | [−14.2, −2.1] | 99.3% |

In this example, all three priors give essentially the same conclusion. The data is informative enough to dominate.

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>


- **Builds on:** Bayesian ITS (Guide 1), CausalPy internals (Guide 2)
- **Leads to:** Notebook 2 (prior sensitivity analysis)
- **Related to:** Gelman's prior recommendations, Stan prior choice wiki


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Prior Specification for Bayesian ITS and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Gelman, A. et al. (2017). "Prior distributions for variance parameters in hierarchical models." *Bayesian Analysis* — the definitive reference on variance priors
- Simpson, D. et al. (2017). "Penalising Model Component Complexity: A Principled, Practical Approach to Constructing Priors." *Statistical Science*
- Stan Development Team. "Prior Choice Recommendations." https://github.com/stan-dev/stan/wiki/Prior-Choice-Recommendations


## Resources

<a class="link-card" href="../notebooks/01_its_from_scratch_pymc.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

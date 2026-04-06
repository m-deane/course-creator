# Hamiltonian Monte Carlo for Bayesian Inference

> **Reading time:** ~10 min | **Module:** 6 — Inference Methods | **Prerequisites:** Module 1 Bayesian Fundamentals


## In Brief

Hamiltonian Monte Carlo (HMC) samples from high-dimensional posterior distributions by simulating Hamiltonian dynamics—treating parameters as positions and introducing auxiliary momentum variables. This enables efficient exploration of complex commodity forecasting models that defeat traditional MCMC.

<div class="callout-insight">

<strong>Insight:</strong> **Random walk MCMC is drunk; HMC is a guided missile.** Traditional Metropolis walks randomly and wastes time revisiting the same regions. HMC uses gradient information to cruise through parameter space along high-probability contours, achieving better mixing with fewer samples.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> Hamiltonian Monte Carlo (HMC) samples from high-dimensional posterior distributions by simulating Hamiltonian dynamics—treating parameters as positions and introducing auxiliary momentum variables.

</div>

---

## Formal Definition

### Hamiltonian Dynamics

**Goal:** Sample from target distribution $\pi(\theta) \propto \exp(-U(\theta))$ where $U(\theta) = -\log p(\theta, y)$.
<div class="callout-key">

<strong>Key Point:</strong> **Goal:** Sample from target distribution $\pi(\theta) \propto \exp(-U(\theta))$ where $U(\theta) = -\log p(\theta, y)$.

</div>


**Augmented Distribution:**
Introduce momentum $p \in \mathbb{R}^d$ and define Hamiltonian:

$$H(\theta, p) = U(\theta) + K(p)$$

where:
- $U(\theta)$: Potential energy (negative log posterior)
- $K(p) = \frac{1}{2} p^\top M^{-1} p$: Kinetic energy ($M$ is mass matrix)

**Hamilton's Equations:**
$$\frac{d\theta}{dt} = \frac{\partial H}{\partial p} = M^{-1} p$$
$$\frac{dp}{dt} = -\frac{\partial H}{\partial \theta} = -\nabla U(\theta)$$

**Key property:** Hamiltonian flow preserves total energy $H(\theta, p)$ and phase space volume (Liouville's theorem).

### Leapfrog Integrator

Discretize Hamilton's equations with step size $\epsilon$ and $L$ steps:

```
for l = 1 to L:
    p ← p - (ε/2) ∇U(θ)        # Half-step momentum
    θ ← θ + ε M⁻¹ p             # Full-step position
    p ← p - (ε/2) ∇U(θ)        # Half-step momentum
```

**Output:** Proposal $(\theta^*, p^*)$ after $L$ leapfrog steps.

### Metropolis Accept/Reject

Accept proposal with probability:
$$\alpha = \min\left(1, \exp\left(H(\theta, p) - H(\theta^*, p^*)\right)\right)$$

If discretization is perfect, $H(\theta, p) = H(\theta^*, p^*)$ and acceptance is always 1.

---

## Intuitive Explanation

### The Physics Analogy

**Imagine a marble rolling on a curved surface:**

1. **Potential Energy (U):** Surface height = negative log posterior
   - Valleys = high probability regions
   - Hills = low probability regions

2. **Kinetic Energy (K):** Marble velocity = momentum
   - Random initial push (sample from $\mathcal{N}(0, M)$)
   - Converts to potential energy as marble climbs hills

3. **Dynamics:** Marble rolls along contours, naturally exploring probable regions
   - Gradient $\nabla U$ is the slope—guides momentum
   - Momentum carries marble across low-probability regions quickly

**Result:** Efficient exploration without random wandering.

---

## Why HMC for Commodity Models?

### 1. High-Dimensional State Spaces

Dynamic linear models have 200+ latent states (weekly oil prices over 4 years). Random walk MCMC gets stuck.

**HMC:** Gradients guide sampling through 200D space efficiently.

### 2. Strong Posterior Correlations

Stochastic volatility models have correlated $(h_t, \phi, \sigma_\eta)$:
- High persistence $\phi$ → smooth $h_t$ paths
- Low $\phi$ → erratic $h_t$ paths

**HMC:** Follows curved posterior contours directly.

### 3. Hierarchical Models

Energy complex hierarchy: global factor → product factors → individual markets.
300+ parameters with complex dependencies.

**HMC:** Scales to high dimensions better than Gibbs sampling.

---

## Code Implementation

### Basic HMC for Crude Oil State Space Model


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import pymc as pm
import numpy as np
import arviz as az

# Simulated crude oil price data
np.random.seed(42)
n_obs = 150
true_level = np.cumsum(np.random.normal(0.1, 1, n_obs)) + 70
oil_prices = true_level + np.random.normal(0, 2, n_obs)

# State space model (local level)
with pm.Model() as oil_state_space:
    # Priors
    sigma_level = pm.HalfNormal('sigma_level', sigma=2)
    sigma_obs = pm.HalfNormal('sigma_obs', sigma=3)
    level_init = pm.Normal('level_init', mu=70, sigma=10)

    # State evolution (random walk)
    level_innov = pm.Normal('level_innov', mu=0, sigma=1, shape=n_obs-1)
    level = pm.Deterministic(
        'level',
        pm.math.concatenate([
            [level_init],
            level_init + pm.math.cumsum(sigma_level * level_innov)
        ])
    )

    # Observations
    y_obs = pm.Normal('y_obs', mu=level, sigma=sigma_obs, observed=oil_prices)

    # Sample with HMC (via NUTS)
    trace = pm.sample(
        1000,
        tune=1500,
        cores=4,
        return_inferencedata=True,
        nuts_sampler='nutpie'  # Fast Rust implementation
    )

# Diagnostics
print(az.summary(trace, var_names=['sigma_level', 'sigma_obs']))
print(f"Effective sample size: {az.ess(trace)['sigma_level'].values}")
print(f"R-hat: {az.rhat(trace)['sigma_level'].values}")

# Visualize sampling
az.plot_trace(trace, var_names=['sigma_level', 'sigma_obs'])
```

</div>
</div>

---

## NUTS: No-U-Turn Sampler

**Problem:** HMC requires tuning $\epsilon$ (step size) and $L$ (trajectory length).

**Solution:** NUTS adapts both automatically during warmup.

### Key Idea

Run leapfrog steps until trajectory starts doubling back (makes a "U-turn"). This means we've explored the contour sufficiently.

**U-Turn Criterion:**
$$(\theta_+ - \theta_-) \cdot p_- < 0 \quad \text{or} \quad (\theta_+ - \theta_-) \cdot p_+ < 0$$

Where $\theta_-, \theta_+$ are endpoints of trajectory, $p_-, p_+$ are momenta.

**PyMC automatically uses NUTS** when you call `pm.sample()`. No manual tuning needed!

---

## Tuning Parameters

### Step Size (ε)

**Too small:** Tiny steps, slow exploration (wastes computation)
**Too large:** Proposals rejected (low acceptance rate)
<div class="callout-warning">

<strong>Warning:</strong> **Too small:** Tiny steps, slow exploration (wastes computation)

</div>


**Optimal:** Acceptance rate ≈ 0.65 for high-dimensional problems


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
trace = pm.sample(
    1000,
    tune=1500,
    target_accept=0.85,  # Increase for complex posteriors
    return_inferencedata=True
)
```

</div>
</div>

**When to increase `target_accept`:**
- Divergences during sampling
- Complex geometry (stochastic volatility, hierarchical models)
- High posterior correlations

---

### Mass Matrix (M)

**Diagonal mass matrix (default):** $M = \text{diag}(m_1, ..., m_d)$
- Assumes parameters have different scales but are uncorrelated

**Dense mass matrix:** $M$ is full positive-definite matrix
- Accounts for posterior correlations
- More computation per step but better exploration


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
trace = pm.sample(
    1000,
    tune=1500,
    nuts_sampler='pymc',
    nuts_sampler_kwargs={'target_accept': 0.9},
    return_inferencedata=True,
    initvals='adapt_diag'  # Adapt diagonal mass matrix (default)
)

# For highly correlated posteriors
trace_dense = pm.sample(
    1000,
    tune=1500,
    nuts_sampler='pymc',
    return_inferencedata=True,
    initvals='adapt_full'  # Adapt full mass matrix
)
```

</div>
</div>

---

## Advanced: Reparameterization

### Non-Centered Parameterization
<div class="callout-key">

<strong>Key Point:</strong> **Problem:** Centered parameterization has poor geometry.

</div>


**Problem:** Centered parameterization has poor geometry.

**Example (Stochastic Volatility):**

**Centered (bad):**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
h[t] = mu + phi * (h[t-1] - mu) + sigma_eta * eta[t]
```

</div>
</div>

When $\sigma_\eta \to 0$, posterior becomes funnel-shaped (hard for HMC).

**Non-centered (good):**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
h_raw[t] = phi * h_raw[t-1] + eta[t]  # eta ~ N(0, 1)
h[t] = mu + sigma_eta * h_raw[t]
```

</div>
</div>

**Implementation:**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
with pm.Model() as sv_noncentered:
    mu = pm.Normal('mu', 0, 5)
    phi = pm.Beta('phi', alpha=20, beta=1.5) * 2 - 1
    sigma_eta = pm.HalfNormal('sigma_eta', sigma=1)

    # Non-centered parameterization
    h_raw_innov = pm.Normal('h_raw_innov', 0, 1, shape=n_obs-1)

    # Build h_raw (AR(1) with unit variance innovations)
    h_raw = pm.Deterministic(
        'h_raw',
        pm.math.concatenate([
            [0],  # Initialize at 0
            pm.math.cumsum(phi * h_raw_innov)
        ])
    )

    # Transform to actual log-volatility
    h = pm.Deterministic('h', mu + sigma_eta * h_raw)

    # Observations
    y_obs = pm.Normal('y_obs', mu=0, sigma=pm.math.exp(h/2), observed=returns)

    trace = pm.sample(1000, tune=2000, target_accept=0.95,
                     return_inferencedata=True)
```

</div>
</div>

**Result:** Fewer divergences, better mixing.

---

## Diagnosing HMC Problems

### 1. Divergences

**Symptom:** Warning message: "X divergences after tuning"
<div class="callout-insight">

<strong>Insight:</strong> **Symptom:** Warning message: "X divergences after tuning"

</div>


**Cause:** Numerical instability in leapfrog integration (steep gradients, sudden curvature)

**Fixes:**
1. Increase `target_accept` (smaller step size):

<span class="filename">example.py</span>
</div>
<div class="code-body">

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   trace = pm.sample(1000, tune=2000, target_accept=0.95)
   ```

</div>
</div>

2. Reparameterize model (non-centered)

3. Add stronger priors (reduce curvature)

---

### 2. Low Effective Sample Size (ESS)

**Symptom:** `ESS << n_samples`

**Cause:** High autocorrelation (samples not independent)

**Check:**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
ess = az.ess(trace)
print(ess['sigma_level'])  # Should be > 400 for 1000 samples
```

</div>
</div>

**Fixes:**
1. Longer chains
2. Better mass matrix adaptation
3. Reparameterization

---

### 3. High R-hat

**Symptom:** $\hat{R} > 1.01$

**Cause:** Chains haven't converged to same distribution

**Fixes:**
1. Longer warmup: `tune=3000`
2. Better initialization: `initvals={'param': reasonable_value}`
3. Check for multimodality (fundamental model issue)

---

### 4. Energy Diagnostic (E-BFMI)

**Bayesian Fraction of Missing Information:**
$$\text{E-BFMI} = \frac{\mathbb{E}[\text{Var}(\Delta E)]}{\text{Var}(E)}$$

**Threshold:** E-BFMI < 0.3 indicates problems

**Check:**

<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
bfmi = az.bfmi(trace)
print(f"E-BFMI: {bfmi}")
```

</div>
</div>

**Low E-BFMI → Poor exploration** (likely funnel geometry)

**Fix:** Non-centered parameterization

---

## HMC for Hierarchical Commodity Model


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Energy complex hierarchy with HMC
n_grades = 5
n_obs = 200

with pm.Model() as energy_hierarchy:
    # Hyperpriors
    mu_global = pm.Normal('mu_global', mu=80, sigma=20)
    sigma_global = pm.HalfNormal('sigma_global', sigma=10)

    # Grade-level parameters (partial pooling)
    grade_intercept = pm.Normal('grade_intercept',
                                mu=0,
                                sigma=5,
                                shape=n_grades)

    grade_sigma = pm.HalfNormal('grade_sigma',
                               sigma=sigma_global,
                               shape=n_grades)

    # Global factor (random walk)
    factor_innov = pm.Normal('factor_innov', 0, 1, shape=n_obs)
    factor = pm.Deterministic(
        'factor',
        mu_global + pm.math.cumsum(sigma_global * factor_innov)
    )

    # Observations
    for g in range(n_grades):
        pm.Normal(f'price_{g}',
                 mu=grade_intercept[g] + factor,
                 sigma=grade_sigma[g],
                 observed=prices[:, g])

    # Sample with NUTS (automatically adapts)
    trace = pm.sample(
        1000,
        tune=2000,
        target_accept=0.9,
        cores=4,
        return_inferencedata=True
    )

# Check diagnostics
print(az.summary(trace))
divergences = trace.sample_stats['diverging'].sum()
print(f"Divergences: {divergences}")

if divergences > 0:
    print("WARNING: Divergences detected. Consider:")
    print("  1. Increase target_accept to 0.95")
    print("  2. Reparameterize model")
    print("  3. Inspect posterior geometry with az.plot_pair()")
```

</div>
</div>

---

## When to Use HMC vs. Other Samplers

| Method | Best For | Limitations |
|--------|----------|-------------|
| **HMC/NUTS** | Continuous, smooth posteriors; high dimensions | Requires gradients (no discrete params) |
| **Metropolis** | Low dimensions (<5); discrete parameters | Slow mixing in high dimensions |
| **Gibbs** | Conjugate priors; block-updatable parameters | Requires closed-form conditionals |
| **SMC** | Multimodal posteriors; difficult initialization | More samples needed |
| **Variational Inference** | Very large data; need fast approximation | Underestimates uncertainty |

**Commodity forecasting → HMC/NUTS** (continuous parameters, complex posteriors, 100-1000 dimensions)

---

## Common Pitfalls

### 1. Forgetting to Tune

**Problem:** Using defaults for complex models.

**Fix:** Always check diagnostics. For complex models, increase `target_accept` to 0.9+.

---

### 2. Ignoring Divergences

**Problem:** "Only 5 divergences, probably fine."

**Reality:** Divergences indicate biased sampling. Posterior mean may be wrong.

**Fix:** Address root cause (reparameterize or tighter priors).

---

### 3. Too Few Warmup Samples

**Problem:** `tune=500` for 300-parameter hierarchical model.

**Fix:** Use `tune >= 1500` for complex models. Mass matrix needs time to adapt.

---

### 4. Discrete Parameters

**Problem:** Trying to use HMC with discrete parameters (e.g., regime indicators).

**Fix:** Marginalize out discrete parameters or use MCMC-within-Gibbs.

---

## Connections

**Builds on:**
- Module 1: MCMC fundamentals (Metropolis-Hastings)
- Calculus: Gradient computation for $\nabla U(\theta)$

**Leads to:**
- Module 7: Sampling from regime-switching models (discrete + continuous)
- Advanced: Riemannian HMC (adaptive geometry)

**Related concepts:**
- Langevin dynamics (gradient-based sampling)
- Variational inference (optimization alternative)

---

## Practice Problems

### Problem 1
A crude oil state space model has 200 latent states. Estimate the number of gradient evaluations HMC performs per iteration if $L=50$ leapfrog steps. Compare to Metropolis (no gradients).
<div class="callout-key">

<strong>Key Point:</strong> A crude oil state space model has 200 latent states. Estimate the number of gradient evaluations HMC performs per iteration if $L=50$ leapfrog steps. Compare to Metropolis (no gradients).

</div>


### Problem 2
Your HMC sampler reports 120 divergences out of 1000 samples. The model is a stochastic volatility model for natural gas.
1. What does this indicate?
2. Propose two fixes.

### Problem 3
Implement a hierarchical model for 3 crude grades with HMC. Check:
1. Effective sample size for grade-level parameters
2. R-hat values
3. E-BFMI

Interpret the diagnostics.

### Problem 4
Why is non-centered parameterization more efficient for hierarchical models? Provide a geometric intuition using the funnel pathology.

### Problem 5
You're forecasting copper prices with a GP model (100 inducing points, 5 kernel parameters). Estimate whether HMC is practical for this problem.

**Hint:** Consider the $O(n^3)$ cost of GP covariance matrix operations and the number of gradient evaluations per sample.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving hamiltonian monte carlo for bayesian inference, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

1. **Neal, R.M. (2011)**. "MCMC Using Hamiltonian Dynamics." *Handbook of Markov Chain Monte Carlo*. CRC Press.
   - Definitive HMC reference

2. **Hoffman, M.D. & Gelman, A. (2014)**. "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo." *Journal of Machine Learning Research*, 15, 1593-1623.
   - Original NUTS paper

3. **Betancourt, M. (2017)**. "A Conceptual Introduction to Hamiltonian Monte Carlo." arXiv:1701.02434.
   - Excellent intuitive explanation with geometric insights

4. **Betancourt, M. & Girolami, M. (2015)**. "Hamiltonian Monte Carlo for Hierarchical Models." *Current Trends in Bayesian Methodology with Applications*. CRC Press.
   - HMC for partial pooling models

---

*"HMC doesn't wander through parameter space—it surfs the posterior contours with purpose."*

---

## Cross-References

<a class="link-card" href="./02_hamiltonian_monte_carlo_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_mcmc_foundations.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

# Convergence Diagnostics for MCMC

> **Reading time:** ~11 min | **Module:** 6 — Inference Methods | **Prerequisites:** Module 1 Bayesian Fundamentals


## In Brief

Convergence diagnostics assess whether MCMC chains have reached their stationary distribution (the true posterior). Key metrics include $\hat{R}$ (comparing between-chain to within-chain variance), effective sample size (accounting for autocorrelation), and trace plots showing mixing behavior.

<div class="callout-insight">

<strong>Insight:</strong> MCMC provides samples from the posterior—but only after the chain has converged. Running MCMC without checking convergence is like leaving bread in the oven without setting a timer: you might get perfect results or a charred mess. Diagnostics tell you when the chain has "burned in" and is providing valid posterior samples.

</div>

## Formal Definition

### The Convergence Question

MCMC generates sequence $\{\theta^{(1)}, \theta^{(2)}, ..., \theta^{(T)}\}$
<div class="callout-key">

<strong>Key Point:</strong> MCMC generates sequence $\{\theta^{(1)}, \theta^{(2)}, ..., \theta^{(T)}\}$

</div>


**Goal:** $\theta^{(t)} \sim p(\theta | \mathbf{y})$ for large $t$

**Problem:** How large is "large enough"?

### Key Diagnostics

**1. Gelman-Rubin Statistic ($\hat{R}$)**

Run M chains with different initial values. For each parameter:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

Where:
- $W$ = within-chain variance (average variance across chains)
- $\hat{V}$ = marginal posterior variance estimate (combining within and between-chain variance)

**Interpretation:**
- $\hat{R} \approx 1$: Chains have converged (rule: $\hat{R} < 1.01$)
- $\hat{R} > 1.1$: Chains have not mixed; continue sampling

**2. Effective Sample Size (ESS)**

Accounts for autocorrelation in MCMC samples:

$$\text{ESS} = \frac{T}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

Where:
- $T$ = number of samples
- $\rho_k$ = autocorrelation at lag $k$

**Interpretation:**
- ESS = T: No autocorrelation (independent samples)
- ESS << T: High autocorrelation (fewer effective samples)
- Rule: ESS > 400 per parameter for reliable inference

**3. Monte Carlo Standard Error (MCSE)**

Uncertainty in posterior mean estimate due to finite sampling:

$$\text{MCSE} = \frac{\sigma}{\sqrt{\text{ESS}}}$$

Where $\sigma$ = posterior standard deviation

**Interpretation:**
- MCSE << posterior sd: Sufficient samples
- MCSE ≈ 0.01 * posterior sd: Good rule of thumb

### Trace Plots

Visual inspection of $\theta^{(t)}$ vs $t$:

**Good mixing:**
```
θ
  │  ∼∼∼∼∼∼∼∼∼∼∼∼  (Hairy caterpillar)
  │ ∼∼∼∼∼∼∼∼∼∼∼∼∼
  │∼∼∼∼∼∼∼∼∼∼∼∼∼∼
  └──────────────── t
```

**Poor mixing:**
```
θ
  │       ___________  (Stuck in region)
  │      /
  │_____/
  └──────────────── t
```

## Intuitive Explanation

Think of MCMC like stirring dye into water:

**Initial state (burn-in):**
- Drop of dye in one spot
- Water mostly clear
- Not representative of final mixture

**Convergence:**
- Dye spreads throughout
- Uniform color everywhere
- Samples now representative

**Diagnostics answer:**
1. **Has the dye spread fully?** ($\hat{R}$) — Compare multiple drops from different spots
2. **How many independent "scoops" do I have?** (ESS) — Accounting for correlation between nearby scoops
3. **Is the color still changing?** (Trace plots) — Visual check for drift

For commodity price models:
- Model has 100 parameters
- Run 4 chains, 5,000 samples each
- Check: $\hat{R} < 1.01$ for all parameters? ESS > 400? Traces look stationary?

## Code Implementation

### Comprehensive Diagnostic Suite


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az
from scipy import stats

def run_mcmc_with_diagnostics(model, tune=1000, draws=2000, chains=4):
    """
    Run MCMC with comprehensive convergence diagnostics.

    Args:
        model: PyMC model
        tune: Number of tuning samples (burn-in)
        draws: Number of post-tuning samples
        chains: Number of independent chains

    Returns:
        trace: MCMC samples
        diagnostics: Dictionary of diagnostic results
    """
    # Sample with multiple chains
    with model:
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            return_inferencedata=True,
            random_seed=[42, 43, 44, 45]  # Different seeds for each chain
        )

    # Compute diagnostics
    diagnostics = {}

    # 1. Gelman-Rubin (R-hat)
    rhat = az.rhat(trace)
    diagnostics['rhat'] = rhat
    diagnostics['rhat_max'] = float(rhat.max().values)
    diagnostics['rhat_converged'] = diagnostics['rhat_max'] < 1.01

    # 2. Effective Sample Size
    ess_bulk = az.ess(trace, method='bulk')
    ess_tail = az.ess(trace, method='tail')
    diagnostics['ess_bulk'] = ess_bulk
    diagnostics['ess_tail'] = ess_tail
    diagnostics['ess_bulk_min'] = float(ess_bulk.min().values)
    diagnostics['ess_tail_min'] = float(ess_tail.min().values)
    diagnostics['ess_sufficient'] = (diagnostics['ess_bulk_min'] > 400 and
                                     diagnostics['ess_tail_min'] > 400)

    # 3. Monte Carlo Standard Error
    mcse = az.mcse(trace)
    diagnostics['mcse'] = mcse

    # 4. Divergences
    diagnostics['n_divergences'] = trace.sample_stats.diverging.sum().values
    diagnostics['divergence_rate'] = float(diagnostics['n_divergences'] / (draws * chains))
    diagnostics['no_divergences'] = diagnostics['n_divergences'] == 0

    # 5. Overall assessment
    diagnostics['converged'] = (diagnostics['rhat_converged'] and
                                diagnostics['ess_sufficient'] and
                                diagnostics['no_divergences'])

    return trace, diagnostics


def print_diagnostic_report(diagnostics):
    """
    Print human-readable diagnostic report.
    """
    print("=" * 60)
    print("MCMC CONVERGENCE DIAGNOSTIC REPORT")
    print("=" * 60)

    print("\n1. GELMAN-RUBIN STATISTIC (R-hat)")
    print(f"   Max R-hat: {diagnostics['rhat_max']:.4f}")
    print(f"   Status: {'✓ CONVERGED' if diagnostics['rhat_converged'] else '✗ NOT CONVERGED'}")
    print(f"   (Threshold: R-hat < 1.01)")

    print("\n2. EFFECTIVE SAMPLE SIZE")
    print(f"   Min ESS (bulk): {diagnostics['ess_bulk_min']:.0f}")
    print(f"   Min ESS (tail): {diagnostics['ess_tail_min']:.0f}")
    print(f"   Status: {'✓ SUFFICIENT' if diagnostics['ess_sufficient'] else '✗ INSUFFICIENT'}")
    print(f"   (Threshold: ESS > 400)")

    print("\n3. DIVERGENCES")
    print(f"   Number of divergences: {diagnostics['n_divergences']}")
    print(f"   Divergence rate: {diagnostics['divergence_rate']:.4f}")
    print(f"   Status: {'✓ NO DIVERGENCES' if diagnostics['no_divergences'] else '✗ DIVERGENCES DETECTED'}")

    print("\n" + "=" * 60)
    print("OVERALL ASSESSMENT")
    print("=" * 60)
    if diagnostics['converged']:
        print("✓ CHAINS HAVE CONVERGED - Inference is reliable")
    else:
        print("✗ CONVERGENCE ISSUES DETECTED")
        print("\nRECOMMENDATIONS:")
        if not diagnostics['rhat_converged']:
            print("  - Increase sampling iterations (tune + draws)")
            print("  - Check for multimodality in posterior")
        if not diagnostics['ess_sufficient']:
            print("  - Increase number of draws")
            print("  - Reparameterize model to reduce autocorrelation")
        if not diagnostics['no_divergences']:
            print("  - Increase target_accept (e.g., 0.95)")
            print("  - Reparameterize model to improve geometry")
    print("=" * 60)


def plot_convergence_diagnostics(trace, param_names=None):
    """
    Comprehensive convergence diagnostic plots.

    Args:
        trace: ArviZ InferenceData
        param_names: List of parameters to plot (default: all)
    """
    if param_names is None:
        param_names = list(trace.posterior.data_vars)

    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, 4, figsize=(16, 4*n_params))
    if n_params == 1:
        axes = axes.reshape(1, -1)

    for i, param in enumerate(param_names):
        # Get samples for this parameter
        samples = trace.posterior[param].values

        # Plot 1: Trace plot (all chains)
        for chain in range(samples.shape[0]):
            axes[i, 0].plot(samples[chain, :], alpha=0.5, linewidth=0.5)
        axes[i, 0].set_ylabel(param)
        axes[i, 0].set_xlabel('Iteration')
        axes[i, 0].set_title('Trace Plot')
        axes[i, 0].grid(alpha=0.3)

        # Plot 2: Posterior density (all chains overlaid)
        for chain in range(samples.shape[0]):
            axes[i, 1].hist(samples[chain, :], bins=50, alpha=0.3,
                           density=True, label=f'Chain {chain}')
        axes[i, 1].set_xlabel(param)
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].set_title('Posterior Distribution')
        axes[i, 1].legend(fontsize=8)

        # Plot 3: Autocorrelation
        az.plot_autocorr(trace, var_names=[param], ax=axes[i, 2],
                        combined=False, textsize=8)
        axes[i, 2].set_title('Autocorrelation')

        # Plot 4: Rank plot (check for uniformity)
        az.plot_rank(trace, var_names=[param], ax=axes[i, 3])
        axes[i, 3].set_title('Rank Plot')

    plt.tight_layout()
    return fig


# Example: Bayesian Linear Regression with Diagnostics
np.random.seed(42)
n = 100
X = np.random.normal(0, 1, n)
y = 2.5 + 1.3 * X + np.random.normal(0, 0.5, n)

with pm.Model() as model:
    # Priors
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=2)

    # Likelihood
    mu = alpha + beta * X
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # Sample with diagnostics
    trace, diagnostics = run_mcmc_with_diagnostics(
        model,
        tune=1000,
        draws=2000,
        chains=4
    )

# Print diagnostic report
print_diagnostic_report(diagnostics)

# Plot diagnostics
fig = plot_convergence_diagnostics(trace, param_names=['alpha', 'beta', 'sigma'])
plt.savefig('convergence_diagnostics.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary statistics with diagnostics
print("\nPosterior Summary:")
print(az.summary(trace, hdi_prob=0.95))
```

</div>
</div>

### Detecting Specific Issues


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def diagnose_convergence_issues(trace):
    """
    Identify specific convergence problems and suggest fixes.

    Returns:
        Dictionary of issues and recommendations
    """
    issues = {
        'high_rhat': [],
        'low_ess': [],
        'high_autocorr': [],
        'divergences': [],
        'recommendations': []
    }

    # Check R-hat
    rhat = az.rhat(trace)
    for var in rhat.data_vars:
        rhat_val = float(rhat[var].max())
        if rhat_val > 1.01:
            issues['high_rhat'].append((var, rhat_val))

    # Check ESS
    ess = az.ess(trace, method='bulk')
    for var in ess.data_vars:
        ess_val = float(ess[var].min())
        if ess_val < 400:
            issues['low_ess'].append((var, ess_val))

    # Check autocorrelation
    for var in trace.posterior.data_vars:
        samples = trace.posterior[var].values.flatten()
        if len(samples) > 100:
            # Compute lag-1 autocorrelation
            autocorr_1 = np.corrcoef(samples[:-1], samples[1:])[0, 1]
            if autocorr_1 > 0.5:
                issues['high_autocorr'].append((var, autocorr_1))

    # Check divergences
    n_divergences = int(trace.sample_stats.diverging.sum())
    if n_divergences > 0:
        issues['divergences'] = n_divergences

    # Generate recommendations
    if issues['high_rhat']:
        issues['recommendations'].append(
            "High R-hat detected: Run longer chains (increase draws) or check for multimodality"
        )

    if issues['low_ess']:
        issues['recommendations'].append(
            "Low ESS detected: Increase samples or reparameterize to reduce autocorrelation"
        )

    if issues['high_autocorr']:
        issues['recommendations'].append(
            "High autocorrelation: Consider centered parameterization or informative priors"
        )

    if issues['divergences']:
        issues['recommendations'].append(
            f"{n_divergences} divergences detected: Increase target_accept to 0.95 or reparameterize"
        )

    return issues


# Run diagnosis
issues = diagnose_convergence_issues(trace)

print("\nCONVERGENCE ISSUE DIAGNOSIS")
print("=" * 60)
if issues['high_rhat']:
    print("\n⚠ HIGH R-HAT:")
    for var, val in issues['high_rhat']:
        print(f"  {var}: {val:.4f}")

if issues['low_ess']:
    print("\n⚠ LOW ESS:")
    for var, val in issues['low_ess']:
        print(f"  {var}: {val:.0f}")

if issues['high_autocorr']:
    print("\n⚠ HIGH AUTOCORRELATION:")
    for var, val in issues['high_autocorr']:
        print(f"  {var}: {val:.3f}")

if issues['divergences']:
    print(f"\n⚠ DIVERGENCES: {issues['divergences']}")

print("\nRECOMMENDATIONS:")
for rec in issues['recommendations']:
    print(f"  • {rec}")
```

</div>
</div>

### Automated Resampling


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def sample_until_converged(model, max_attempts=3, initial_draws=1000):
    """
    Automatically resample until convergence criteria met.

    Args:
        model: PyMC model
        max_attempts: Maximum resampling attempts
        initial_draws: Starting number of draws

    Returns:
        trace or None if convergence not achieved
    """
    draws = initial_draws

    for attempt in range(1, max_attempts + 1):
        print(f"\n{'='*60}")
        print(f"SAMPLING ATTEMPT {attempt}/{max_attempts}")
        print(f"Draws: {draws}, Tune: {draws//2}")
        print(f"{'='*60}")

        trace, diagnostics = run_mcmc_with_diagnostics(
            model,
            tune=draws // 2,
            draws=draws,
            chains=4
        )

        print_diagnostic_report(diagnostics)

        if diagnostics['converged']:
            print(f"\n✓ Convergence achieved after {attempt} attempt(s)")
            return trace

        # Increase sample size for next attempt
        draws = int(draws * 1.5)

        if attempt < max_attempts:
            print(f"\n→ Increasing draws to {draws} and resampling...")

    print("\n✗ Failed to achieve convergence after maximum attempts")
    print("Consider: Model reparameterization, stronger priors, or different sampler")
    return None


# Example usage
with pm.Model() as difficult_model:
    # Intentionally difficult: hierarchical model with weak priors
    mu_global = pm.Normal('mu_global', 0, 10)
    sigma_global = pm.HalfNormal('sigma_global', 5)

    mu_group = pm.Normal('mu_group', mu_global, sigma_global, shape=10)

    y_obs = pm.Normal('y_obs', mu_group[0], 1, observed=np.random.randn(50))

# This will automatically increase samples if needed
trace = sample_until_converged(difficult_model, max_attempts=3, initial_draws=500)
```

</div>
</div>

## Common Pitfalls

**1. Ignoring R-hat for Individual Parameters**
- **Problem**: Checking overall convergence but missing issues in specific parameters
- **Symptom**: One parameter has $\hat{R} = 1.15$ while others are fine
- **Solution**: Check $\hat{R}$ for each parameter individually
<div class="callout-key">

<strong>Key Point:</strong> **1. Ignoring R-hat for Individual Parameters**

</div>


**2. Accepting Low ESS**
- **Problem**: ESS = 50 seems "good enough"
- **Symptom**: Unstable posterior quantiles, poor coverage
- **Solution**: Target ESS > 400 per parameter (some recommend 1000+)

**3. Visual Inspection Only**
- **Problem**: Trace plots "look fine" without quantitative checks
- **Symptom**: Subtle non-convergence missed
- **Solution**: Always compute numerical diagnostics alongside visual

**4. Discarding Divergences**
- **Problem**: "Only 1% divergences, probably fine"
- **Symptom**: Biased posterior estimates, especially in tails
- **Solution**: Zero divergences should be the goal; investigate and fix causes

**5. Not Running Multiple Chains**
- **Problem**: Single chain cannot detect non-convergence
- **Symptom**: Cannot compute $\hat{R}$, miss multimodality
- **Solution**: Always run ≥4 chains from dispersed initial values

## Connections

**Builds on:**
- Module 6.1: MCMC foundations (sampling mechanics)
- Module 6.2: Hamiltonian Monte Carlo (understanding divergences)
- Time series analysis (autocorrelation concepts)

**Leads to:**
- Model selection (cannot compare models without converged samples)
- Posterior predictive checks (require valid posterior samples)
- Production deployment (convergence = reliability)

**Related concepts:**
- Bootstrap convergence diagnostics
- Cross-validation (alternative to MCMC for some models)
- Variational inference diagnostics (ELBO convergence)

## Practice Problems

1. **Interpreting R-hat**
   You run 4 chains for parameter $\theta$:
   - Chain 1 mean: 2.3
   - Chain 2 mean: 2.4
   - Chain 3 mean: 2.3
   - Chain 4 mean: 8.1

   Without computing: Will $\hat{R}$ be close to 1? Why or why not?

2. **ESS Calculation**
   Chain with 1000 samples has autocorrelations:
   - Lag 1: ρ₁ = 0.8
   - Lag 2: ρ₂ = 0.6
   - Lag 3: ρ₃ = 0.4
   - Lag 4+: ρₖ ≈ 0

   Estimate ESS using: ESS ≈ T / (1 + 2(ρ₁ + ρ₂ + ρ₃))

3. **Divergence Diagnosis**
   Model: Hierarchical with group-level variance σ
   Observation: 50 divergences, all when σ < 0.1

   What does this suggest? How to fix?

4. **Minimum Sample Size**
   You want to estimate posterior mean with MCSE < 0.01
   Posterior sd = 0.5
   ESS = 200

   Current MCSE = 0.5 / sqrt(200) ≈ 0.035

   How many additional effective samples needed?

5. **Commodity Model Scenario**
   GP with 100 inducing points, 5 hyperparameters
   After 2000 samples:
   - R-hat: All < 1.01
   - ESS: Inducing points > 1000, hyperparameters 100-200
   - Divergences: 0

   Should you: (a) Stop, (b) Continue sampling, (c) Reparameterize?


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving convergence diagnostics for mcmc, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

**Foundational:**
1. **Gelman & Rubin (1992)** - "Inference from Iterative Simulation Using Multiple Sequences" - Original $\hat{R}$
2. **Vehtari et al. (2021)** - "Rank-normalization, folding, and localization: An improved $\hat{R}$" - Modern improvements
3. **Geyer (1992)** - "Practical Markov Chain Monte Carlo" - ESS theory

**Practical Guides:**
4. **ArviZ Documentation** - Diagnostic implementation and interpretation
5. **PyMC Convergence Guide** - Diagnosing and fixing issues
6. **Stan User's Guide: MCMC Sampling** - Divergence diagnostics

**Advanced:**
7. **Betancourt (2017)** - "A Conceptual Introduction to Hamiltonian Monte Carlo" - Understanding divergences
8. **Vehtari et al. (2020)** - "Pareto Smoothed Importance Sampling" - Diagnostic for importance sampling

**Commodity Applications:**
9. **"Bayesian Workflow" (Gelman et al.)** - End-to-end diagnostic protocol
10. **"Validating Bayesian Inference Algorithms with Simulation-Based Calibration"** - Rigorous validation


<div class="callout-key">

<strong>Key Concept Summary:</strong> Convergence diagnostics assess whether MCMC chains have reached their stationary distribution (the true posterior).

</div>

---

*"Convergence diagnostics ensure your MCMC samples are reliable. No diagnostics = no trust in posterior inference."*

---

## Cross-References

<a class="link-card" href="./04_convergence_diagnostics_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_mcmc_foundations.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

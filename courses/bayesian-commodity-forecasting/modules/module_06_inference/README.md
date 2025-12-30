# Module 6: MCMC and Variational Inference

## Overview

When conjugate priors aren't available (most real models), we need computational methods to approximate posteriors. This module covers the two main approaches: Markov Chain Monte Carlo (MCMC) for accurate sampling, and Variational Inference (VI) for scalable approximation.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. **Understand** MCMC foundations and the Metropolis-Hastings algorithm
2. **Apply** Hamiltonian Monte Carlo (HMC) and the NUTS sampler
3. **Diagnose** convergence using R-hat, ESS, and trace plots
4. **Compare** MCMC vs. Variational Inference tradeoffs
5. **Choose** appropriate inference methods for different model types

## Module Contents

### Guides
- `01_mcmc_foundations.md` - From Metropolis to modern samplers
- `02_hamiltonian_monte_carlo.md` - Physics-inspired efficient sampling
- `03_variational_inference.md` - Optimization-based approximation
- `04_convergence_diagnostics.md` - Ensuring reliable inference

### Notebooks
- `01_metropolis_hastings.ipynb` - Build MH sampler from scratch
- `02_hmc_intuition.ipynb` - Visualizing Hamiltonian dynamics
- `03_nuts_in_practice.ipynb` - PyMC's default sampler
- `04_variational_inference_pymc.ipynb` - VI for large models
- `05_diagnosing_problems.ipynb` - Fixing common sampling issues

### Assessments
- `quiz.md` - Inference concepts (15 questions)
- `coding_exercises.py` - Implement samplers and diagnostics

## Key Concepts

### The Inference Problem

Given:
- Prior: $p(\theta)$
- Likelihood: $p(y | \theta)$
- Observed data: $y$

Compute the posterior: $p(\theta | y) \propto p(y | \theta) p(\theta)$

**Challenge:** The normalizing constant $p(y) = \int p(y|\theta) p(\theta) d\theta$ is usually intractable.

### Solution Approaches

```
                    Posterior Inference
                           │
            ┌──────────────┴──────────────┐
            │                             │
        Sampling                    Optimization
        (MCMC)                         (VI)
            │                             │
    ┌───────┴───────┐             ┌───────┴───────┐
    │               │             │               │
 Metropolis      HMC/NUTS      Mean-Field      Full-Rank
 Hastings                        ADVI           ADVI
    │               │             │               │
  Slow,          Fast,        Very fast,      Fast,
  general        efficient     approximate   approximate
```

### MCMC vs VI Comparison

| Aspect | MCMC | Variational Inference |
|--------|------|----------------------|
| **Accuracy** | Asymptotically exact | Approximate |
| **Speed** | Slower | Faster |
| **Scalability** | O(n) per sample | O(n) total |
| **Multimodal** | Can explore | May miss modes |
| **Uncertainty** | Full posterior | Often underestimates |
| **Use case** | Final inference | Exploration, large data |

## Convergence Diagnostics

### R-hat (Gelman-Rubin)

Compares between-chain and within-chain variance:
$$\hat{R} = \sqrt{\frac{\text{Var}^+(\theta)}{\text{W}}}$$

**Interpretation:**
- $\hat{R} < 1.01$: Converged
- $\hat{R} > 1.1$: Definitely not converged
- Check for ALL parameters

### Effective Sample Size (ESS)

Accounts for autocorrelation in chains:
$$n_{\text{eff}} = \frac{mn}{1 + 2\sum_{t=1}^T \rho_t}$$

**Guidelines:**
- ESS > 400 for reliable posterior summaries
- ESS > 1000 for tail estimates
- ESS_bulk and ESS_tail both matter

### Visual Diagnostics

1. **Trace plots:** Chains should look like "fuzzy caterpillars"
2. **Rank plots:** Ranks should be uniform across chains
3. **Autocorrelation:** Should decay quickly

## Completion Criteria

- [ ] MH sampler notebook produces valid samples
- [ ] Can diagnose and fix convergence issues
- [ ] Understand when to use MCMC vs VI
- [ ] Quiz score ≥ 80%

## Prerequisites

- Module 1-3 completed
- Probability distributions
- Basic optimization concepts

---

*"The sampler doesn't lie. If diagnostics look bad, the posterior is unreliable—no matter how reasonable the point estimates appear."*

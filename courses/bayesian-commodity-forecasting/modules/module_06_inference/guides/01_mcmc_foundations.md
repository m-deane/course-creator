# MCMC Foundations: From Metropolis to Modern Samplers

## In Brief

Markov Chain Monte Carlo (MCMC) constructs a Markov chain whose stationary distribution is the target posterior. By simulating this chain long enough, we obtain samples that represent the posterior distribution.

## Key Insight

**MCMC is random walk + acceptance rule.** We propose moves through parameter space and accept or reject them based on how much they improve the posterior density. The magic: this simple procedure produces samples from arbitrarily complex posteriors.

---

## The Core Problem

We want to sample from:
$$p(\theta | y) = \frac{p(y | \theta) p(\theta)}{p(y)}$$

But $p(y) = \int p(y | \theta) p(\theta) d\theta$ is intractable.

**Key observation:** We can evaluate $p(\theta | y) \propto p(y | \theta) p(\theta)$ up to a constant. MCMC only needs this unnormalized density.

---

## The Metropolis-Hastings Algorithm

### Algorithm

1. Initialize $\theta^{(0)}$
2. For $t = 1, 2, ..., T$:
   a. Propose $\theta^* \sim q(\theta^* | \theta^{(t-1)})$
   b. Calculate acceptance probability:
      $$\alpha = \min\left(1, \frac{p(\theta^* | y) \cdot q(\theta^{(t-1)} | \theta^*)}{p(\theta^{(t-1)} | y) \cdot q(\theta^* | \theta^{(t-1)})}\right)$$
   c. Accept with probability $\alpha$:
      - If accept: $\theta^{(t)} = \theta^*$
      - If reject: $\theta^{(t)} = \theta^{(t-1)}$
3. Return $\{\theta^{(1)}, ..., \theta^{(T)}\}$

### For Symmetric Proposals

If $q(\theta^* | \theta) = q(\theta | \theta^*)$ (e.g., Normal centered at current):

$$\alpha = \min\left(1, \frac{p(\theta^* | y)}{p(\theta^{(t-1)} | y)}\right)$$

This simplifies to: accept if new position has higher posterior density, accept with some probability if lower.

---

## Visual Intuition

### Random Walk Through Posterior

```
                 Posterior Density
                      ╱╲
                     ╱  ╲
                    ╱    ╲
                   ╱      ╲
            ────╱──────────╲────

Chain path:  ●─●─●───●─●─●──●
                  └─rejected─┘
                    proposal

The chain spends more time in high-density regions,
proportional to the posterior probability.
```

### Acceptance Rate Intuition

```
Proposal too wide:          Proposal too narrow:
Most proposals rejected     Chain moves slowly
      ●───────────x                ●─●─●─●─●
      ●──────────x                 (many tiny steps)
      ●───────x
      (stuck in place)

Optimal: ~25-50% acceptance rate for high dimensions
```

---

## Implementation from Scratch

```python
import numpy as np
from scipy import stats

def metropolis_hastings(log_posterior, initial, n_samples, proposal_std):
    """
    Basic Metropolis-Hastings sampler.

    Parameters
    ----------
    log_posterior : callable
        Function computing log p(theta | y) up to constant
    initial : array
        Starting parameter values
    n_samples : int
        Number of samples to draw
    proposal_std : float
        Standard deviation of Normal proposal

    Returns
    -------
    samples : array
        MCMC samples (n_samples x n_params)
    accept_rate : float
        Fraction of proposals accepted
    """
    n_params = len(initial)
    samples = np.zeros((n_samples, n_params))
    current = initial.copy()
    current_log_prob = log_posterior(current)

    accepted = 0

    for i in range(n_samples):
        # Propose new position
        proposal = current + np.random.normal(0, proposal_std, n_params)
        proposal_log_prob = log_posterior(proposal)

        # Compute acceptance probability (log scale for numerical stability)
        log_alpha = proposal_log_prob - current_log_prob

        # Accept or reject
        if np.log(np.random.random()) < log_alpha:
            current = proposal
            current_log_prob = proposal_log_prob
            accepted += 1

        samples[i] = current

    return samples, accepted / n_samples

# Example: Sample from a 2D Gaussian
def log_posterior(theta):
    """Log density of N([0,0], [[1, 0.5], [0.5, 1]])"""
    mu = np.array([0, 0])
    cov = np.array([[1, 0.5], [0.5, 1]])
    return stats.multivariate_normal.logpdf(theta, mu, cov)

# Run sampler
samples, rate = metropolis_hastings(
    log_posterior,
    initial=np.array([5.0, 5.0]),  # Start far from mode
    n_samples=10000,
    proposal_std=1.0
)

print(f"Acceptance rate: {rate:.2%}")
print(f"Sample mean: {samples[1000:].mean(axis=0)}")  # Discard burn-in
```

---

## Challenges with Basic MCMC

### 1. Random Walk Inefficiency

In high dimensions, random walk proposals become very inefficient:
- Most directions don't improve posterior
- Acceptance rate drops dramatically
- Need $O(d^2)$ samples for $d$ dimensions

### 2. Tuning the Proposal

- Too wide: Low acceptance, stuck in place
- Too narrow: High acceptance, slow exploration
- Optimal depends on posterior shape (unknown!)

### 3. Correlated Parameters

When parameters are correlated, axis-aligned proposals are inefficient:

```
       θ₂
        │    ╱
        │   ╱   Posterior is elongated
        │  ╱
        │ ╱
        │╱
        └──────────── θ₁

Axis-aligned proposals (→↑) are inefficient.
Need proposals along the correlation direction.
```

---

## Solution: Hamiltonian Monte Carlo

HMC solves these problems using gradient information:

1. **Uses gradients:** Proposes moves in directions of increasing density
2. **Suppresses random walk:** Momentum carries sampler efficiently
3. **Handles correlations:** Follows the posterior geometry

Details in the next guide.

---

## Practical Guidelines

### Burn-in (Warm-up)

Discard initial samples before chain reaches stationary distribution:
- Conservative: Discard first 50%
- Check: Trace plot should look stable after burn-in

### Thinning

Keep every $k$-th sample to reduce autocorrelation:
- Usually unnecessary with HMC/NUTS
- If needed, thin by autocorrelation time

### Multiple Chains

Run several chains from different starting points:
- Enables R-hat convergence diagnostic
- Detects multimodality
- Standard: 4 chains

---

## When MCMC Fails

### Symptoms

1. R-hat > 1.01 (chains haven't mixed)
2. Low ESS (high autocorrelation)
3. Divergences (HMC-specific)
4. Different chains find different modes

### Solutions

1. **Reparameterize:** Center parameters, use non-centered parameterization
2. **Increase tuning:** More warm-up samples
3. **Simplify model:** Reduce complexity, use stronger priors
4. **Use better sampler:** Switch to NUTS, adjust step size

---

## Connections

**Builds on:**
- Module 1: Bayes' theorem, posterior computation
- Probability: Markov chains, stationary distributions

**Leads to:**
- HMC/NUTS for efficient sampling
- Variational inference for scalability
- Diagnosing model problems

---

## Practice Problems

1. Implement Metropolis-Hastings for a 1D Normal target with unknown mean and known variance. Verify samples match the analytical posterior.

2. What happens to the acceptance rate when you increase proposal_std by 10x? When you decrease by 10x?

3. For a 100-dimensional standard Normal target, approximately how many MH samples would you need for reliable inference? (Hint: Random walk in $d$ dimensions mixes in $O(d^2)$ steps.)

---

## Further Reading

1. **Gelman et al.** *BDA* Chapter 11 — Comprehensive MCMC treatment
2. **Robert & Casella** *Monte Carlo Statistical Methods* — Mathematical foundations
3. **Brooks et al.** *Handbook of MCMC* — Advanced topics

---

*"MCMC is the engine that makes Bayesian inference practical. Understanding how it works—and when it fails—is essential for any Bayesian modeler."*

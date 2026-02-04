# Bayesian Estimation for DFMs

## In Brief

Bayesian methods treat parameters as random variables with prior distributions. Using MCMC (Gibbs sampling), we draw from the joint posterior of parameters and factors, giving full uncertainty quantification. Ideal for complex models where ML is unstable.

## Key Insight

**The Bayesian advantage:** Instead of point estimates, get entire distributions:
- Parameter uncertainty (credible intervals)
- Factor uncertainty (prediction intervals)
- Model uncertainty (Bayes factors)

**The cost:** Computationally intensive (need thousands of draws).

## Formal Definition

**Posterior:**
```
p(θ, F₁:T | y₁:T) ∝ p(y₁:T | F₁:T, θ) × p(F₁:T | θ) × p(θ)
```

**Gibbs Sampler:**
1. Draw F₁:T | y₁:T, θ (Kalman smoother simulation)
2. Draw Λ | F₁:T, y₁:T (regression with normal prior)
3. Draw Φ | F₁:T (VAR with normal prior)
4. Draw Q, H | residuals (inverse-Wishart priors)

## Code Implementation

```python
def gibbs_sampler(y, n_draws=5000, burnin=1000):
    """Bayesian estimation via Gibbs sampling."""
    # Priors
    Λ_prior_mean = np.zeros((N, r))
    Λ_prior_prec = np.eye(r) * 0.01

    # Initialize
    Λ = np.random.randn(N, r)
    Φ = np.eye(r) * 0.5
    Q = np.eye(r)
    H = np.eye(N)

    draws = {'Lambda': [], 'Phi': [], 'Q': [], 'H': []}

    for i in range(n_draws):
        # 1. Draw factors F | y, θ (forward-filter backward-sample)
        F = sample_factors(y, Λ, Φ, Q, H)

        # 2. Draw Λ | F, y (conjugate normal)
        Λ = sample_loadings(y, F, Λ_prior_mean, Λ_prior_prec, H)

        # 3. Draw Φ | F (conjugate normal)
        Φ = sample_transition(F, Q)

        # 4. Draw Q | F (inverse-Wishart)
        Q = sample_Q(F, Φ)

        # 5. Draw H | y, F, Λ (inverse-Wishart)
        H = sample_H(y, F, Λ)

        # Store draws after burnin
        if i >= burnin:
            draws['Lambda'].append(Λ.copy())
            draws['Phi'].append(Φ.copy())

    return draws

# Usage
posterior_draws = gibbs_sampler(y, n_draws=5000)
Λ_mean = np.mean(posterior_draws['Lambda'], axis=0)
```

## Common Pitfalls

### 1. Poor Mixing
**Problem:** MCMC chain doesn't explore posterior well.
**Solution:** Tune priors, use block sampling, run longer chains.

### 2. Identification in Bayesian Setting
**Problem:** Posterior is multimodal due to rotation.
**Solution:** Impose identification via constrained priors.

## Practice Problems

1. Implement forward-filter backward-sample
2. Diagnose MCMC convergence with traceplots
3. Compare Bayesian intervals to ML standard errors

## Further Reading

- Kim & Nelson (1999): *State-Space Models with Regime Switching*
- Koop & Korobilis (2010): "Bayesian Multivariate Time Series Methods"

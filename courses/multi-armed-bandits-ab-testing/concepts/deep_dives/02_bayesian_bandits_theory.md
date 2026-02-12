# Bayesian Bandits: Theory and Optimality

## TL;DR

The Bayesian approach to bandits maintains a full probability distribution (belief) over each arm's reward. Thompson Sampling is the practical algorithm; the Gittins Index is the theoretically optimal solution. Thompson Sampling approximates Gittins well in practice while being far simpler to implement.

## Visual Explanation

```
Bayesian Bandit Decision Framework

                  ┌──────────────────┐
                  │   PRIOR BELIEFS  │
                  │  P(θ_a) for each │
                  │      arm a       │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │  SELECT ACTION   │
                  │                  │
                  │  Thompson: sample│
                  │  from posteriors │
                  │  pick highest    │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │ OBSERVE REWARD   │
                  │    r_t ~ P(r|θ)  │
                  └────────┬─────────┘
                           │
                  ┌────────▼─────────┐
                  │ UPDATE BELIEFS   │
                  │                  │
                  │ P(θ|data) ∝      │
                  │ P(data|θ)·P(θ)   │
                  └────────┬─────────┘
                           │
                           ▼
                      REPEAT ───►
```

## Intuitive Analogy

Imagine you're a commodity trader evaluating 5 different trading signals. For each signal, you have a "confidence meter" — wide and uncertain at first, then narrowing as you see more data. Thompson Sampling asks each signal to "audition" by drawing from its confidence meter. Uncertain signals sometimes give impressive auditions (and get tested). Confident losers rarely impress. Over time, the best signal wins most auditions.

## The Gittins Index

The **Gittins Index** (1979) provides the theoretically optimal solution to the discounted multi-armed bandit problem.

**Setup**: Each arm has an unknown reward distribution. You want to maximize discounted total reward:

$$\max \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right]$$

where $\gamma \in (0,1)$ is the discount factor.

**Gittins' Theorem**: The optimal policy computes an "index" for each arm based solely on that arm's state (posterior), and always pulls the arm with the highest index.

$$\nu_a(s) = \sup_{\tau > 0} \frac{\mathbb{E}\left[\sum_{t=0}^{\tau-1} \gamma^t r_t \mid s\right]}{\mathbb{E}\left[\sum_{t=0}^{\tau-1} \gamma^t \mid s\right]}$$

**Why it's impractical**: Computing the Gittins index requires solving a stopping problem for each arm state — computationally expensive and doesn't extend easily to contextual or restless settings.

## Why Thompson Sampling Works

Thompson Sampling approximates the Bayesian optimal solution through a simple mechanism:

1. **Probability matching**: Thompson Sampling selects each arm with probability equal to the posterior probability that the arm is optimal
2. **Automatic exploration**: Arms with high uncertainty get wide posterior samples, giving them a chance to "prove themselves"
3. **Convergence**: As data accumulates, posteriors concentrate around the true parameters, and sampling naturally converges to exploitation

**Key result** (Russo & Van Roy, 2014): Thompson Sampling satisfies:

$$\text{BayesRegret}(T) \leq O(\sqrt{KT \ln T})$$

And for structured problems, it can achieve much better bounds by exploiting the problem structure.

## Conjugate Prior Families

| Reward Type | Likelihood | Prior | Posterior |
|------------|-----------|-------|-----------|
| Binary (click/no-click) | Bernoulli | Beta(α,β) | Beta(α+s, β+f) |
| Continuous (returns) | Normal(μ,σ²) | Normal(μ₀,σ₀²) | Normal(μ_n, σ_n²) |
| Count (arrivals) | Poisson(λ) | Gamma(α,β) | Gamma(α+Σx, β+n) |
| Duration (holding time) | Exponential(λ) | Gamma(α,β) | Gamma(α+n, β+Σx) |

## Connection to Bayesian Commodity Forecasting

This course's Bayesian bandit module connects directly to the Bayesian Commodity Forecasting course:

- **Shared foundation**: Both use posterior updating as the core mechanism
- **Priors encode domain knowledge**: In commodity forecasting, priors encode seasonal patterns and fundamental relationships. In bandits, priors encode initial beliefs about strategy effectiveness.
- **Uncertainty quantification**: Both produce full distributions, not point estimates — critical for risk management
- **Sequential decision making**: Bayesian commodity forecasts inform trading decisions; bandits optimize which forecasts/strategies to act on

The natural workflow: Bayesian forecasting models generate signals → bandit algorithms decide which signals to allocate capital toward.

## When to Prefer Bayesian vs. Frequentist Bandits

| Criterion | Bayesian (Thompson) | Frequentist (UCB) |
|-----------|-------------------|-------------------|
| Prior knowledge available | Strong advantage | No mechanism |
| Delayed feedback | Handles naturally | Requires modification |
| Batched updates | Straightforward | Harder to adapt |
| Theoretical guarantees | Asymptotic | Finite-time |
| Implementation | Simple | Simple |
| Non-stationarity | Discounted priors | Sliding windows |

## Key References

- Gittins, J.C. (1979). "Bandit Processes and Dynamic Allocation Indices." *Journal of the Royal Statistical Society*.
- Russo, D. and Van Roy, B. (2014). "Learning to Optimize via Posterior Sampling." *Mathematics of Operations Research*.
- Chapelle, O. and Li, L. (2011). "An Empirical Evaluation of Thompson Sampling." *NeurIPS*.
- Russo, D., Van Roy, B., Kazerouni, A., Osband, I., and Wen, Z. (2018). "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning*.

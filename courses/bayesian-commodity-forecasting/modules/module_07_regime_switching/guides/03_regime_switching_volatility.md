# Regime-Switching Volatility Models

> **Reading time:** ~10 min | **Module:** 7 — Regime Switching | **Prerequisites:** Module 3 State-Space Models


## In Brief

Regime-switching volatility models capture the observation that market volatility changes dramatically across different states—calm periods with low volatility versus crisis periods with high volatility. These models allow variance to switch between regimes governed by a latent Markov process.

<div class="callout-insight">
<strong>Insight:</strong> Asset returns don't have constant volatility. Oil prices might have σ = 2% daily in stable markets, but σ = 8% during supply disruptions. A single volatility parameter cannot capture both regimes. Regime-switching models learn: (1) distinct volatility levels, (2) regime persistence, (3) transition probabilities, enabling better risk management and option pricing.
</div>

## Formal Definition

### Markov-Switching GARCH (MS-GARCH)

**State space:**
- Latent regime: $s_t \in \{1, 2, ..., K\}$
- Transition matrix: $P_{ij} = P(s_{t+1} = j | s_t = i)$

**Regime-dependent returns:**

$$y_t | s_t = k \sim \mathcal{N}(\mu_k, \sigma^2_{k,t})$$

**Conditional variance (GARCH within regime):**

$$\sigma^2_{k,t} = \omega_k + \alpha_k \epsilon^2_{t-1} + \beta_k \sigma^2_{k,t-1}$$

Where:
- $\mu_k$: Regime-specific mean return
- $\omega_k, \alpha_k, \beta_k$: Regime-specific GARCH parameters
- $\epsilon_t = y_t - \mu_{s_t}$: Innovation

### Markov-Switching Stochastic Volatility (MS-SV)

**Log-volatility evolves as:**

$$\log \sigma^2_{k,t} = \alpha_k + \phi_k \log \sigma^2_{k,t-1} + \eta_{k,t}, \quad \eta_{k,t} \sim \mathcal{N}(0, \tau^2_k)$$

**Returns:**

$$y_t | s_t = k, \sigma^2_{k,t} \sim \mathcal{N}(\mu_k, \sigma^2_{k,t})$$

### Two-Regime Example

**Low volatility regime (k=1):**
- $\mu_1 = 0.05\%$ (slight positive drift)
- $\sigma_1 = 1.5\%$ (low volatility)
- $P(s_{t+1} = 1 | s_t = 1) = 0.98$ (persistent)

**High volatility regime (k=2):**
- $\mu_2 = -0.1\%$ (slight negative drift)
- $\sigma_2 = 5.0\%$ (high volatility)
- $P(s_{t+1} = 2 | s_t = 2) = 0.92$ (persistent but shorter-lived)

## Intuitive Explanation

Think of driving conditions:

**Regime 1: Clear weather**
- Speed variance: ±5 mph
- Average speed: 60 mph
- Stays clear for weeks

**Regime 2: Storm**
- Speed variance: ±20 mph
- Average speed: 40 mph
- Storm lasts days, not weeks

Your driving model needs both regimes. Using average variance (±12.5 mph) wouldn't capture either state well.

For commodities:
- **Regime 1 (Normal)**: Oil prices vary ±2% daily, production stable
- **Regime 2 (Crisis)**: Oil prices vary ±8% daily, supply disruptions

Regime-switching model learns:
1. The two volatility levels (2% vs 8%)
2. How long each regime lasts (98% vs 92% persistence)
3. When to switch between them (transition probabilities)

This enables:
- Better Value-at-Risk estimates (use crisis regime σ)
- Conditional forecasts ("if we're in high-vol regime, expect...")
- Early warning signals (regime probability shift)

## Code Implementation

### Two-Regime Switching Volatility (PyMC)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def generate_regime_switching_data(n=500, seed=42):
    """
    Generate synthetic data from regime-switching model.

    Returns:
        returns: Observed returns
        true_regimes: Actual regime sequence (for validation)
    """
    np.random.seed(seed)

    # True parameters
    mu = np.array([0.05, -0.10])  # Regime means (%)
    sigma = np.array([1.5, 5.0])  # Regime volatilities (%)

    # Transition matrix
    P = np.array([
        [0.98, 0.02],  # From low-vol to {low-vol, high-vol}
        [0.10, 0.90]   # From high-vol to {low-vol, high-vol}
    ])

    # Simulate regime sequence
    regimes = np.zeros(n, dtype=int)
    regimes[0] = 0  # Start in low-vol regime

    for t in range(1, n):
        regimes[t] = np.random.choice(2, p=P[regimes[t-1]])

    # Generate returns conditional on regimes
    returns = np.zeros(n)
    for t in range(n):
        k = regimes[t]
        returns[t] = np.random.normal(mu[k], sigma[k])

    return returns, regimes


def fit_markov_switching_volatility(returns, n_regimes=2):
    """
    Fit Markov-switching volatility model.

    Args:
        returns: Observed return series
        n_regimes: Number of volatility regimes

    Returns:
        model: PyMC model
        trace: MCMC samples
    """
    n = len(returns)

    with pm.Model() as model:
        # Regime-specific parameters
        # Use ordered constraint to ensure σ₁ < σ₂
        sigma_raw = pm.HalfNormal('sigma_raw', sigma=5, shape=n_regimes)
        sigma = pm.Deterministic('sigma', pm.math.sort(sigma_raw))

        mu = pm.Normal('mu', mu=0, sigma=2, shape=n_regimes)

        # Transition probabilities (each row of transition matrix)
        # Dirichlet ensures row sums to 1
        p_transition = pm.Dirichlet('p_transition',
                                     a=np.ones((n_regimes, n_regimes)),
                                     shape=(n_regimes, n_regimes))

        # Initial regime distribution
        p_initial = pm.Dirichlet('p_initial', a=np.ones(n_regimes))

        # Regime likelihood using marginalization
        # This is the tricky part - we marginalize over regime sequences
        # Use forward algorithm

        def forward_algorithm(returns, mu, sigma, p_trans, p_init):
            """
            Forward algorithm for HMM likelihood.
            Returns log P(y_1, ..., y_T).
            """
            n = len(returns)
            K = len(mu)

            # Forward probabilities: alpha[t, k] = P(y_1:t, s_t = k)
            log_alpha = np.zeros((n, K))

            # t = 0
            log_alpha[0, :] = (
                np.log(p_init) +
                stats.norm.logpdf(returns[0], mu, sigma)
            )

            # t = 1, ..., T-1
            for t in range(1, n):
                for k in range(K):
                    # P(s_t = k | s_{t-1}) * P(y_t | s_t = k) * P(s_{t-1})
                    log_trans = np.log(p_trans[:, k])  # P(k | previous states)
                    log_emission = stats.norm.logpdf(returns[t], mu[k], sigma[k])

                    # Log-sum-exp for numerical stability
                    log_alpha[t, k] = (
                        log_emission +
                        np.logaddexp.reduce(log_alpha[t-1, :] + log_trans)
                    )

            # Total likelihood: sum over final states
            log_likelihood = np.logaddexp.reduce(log_alpha[-1, :])
            return log_likelihood

        # Custom likelihood
        log_lik = pm.Potential('log_lik',
                               forward_algorithm(returns, mu, sigma,
                                                 p_transition, p_initial))

        # Sample
        trace = pm.sample(
            2000,
            tune=2000,
            target_accept=0.95,
            return_inferencedata=True,
            random_seed=42
        )

    return model, trace


def decode_regimes(returns, trace, n_regimes=2):
    """
    Decode most likely regime sequence using Viterbi algorithm.

    Args:
        returns: Observed returns
        trace: MCMC trace with posterior samples

    Returns:
        regime_probs: Smoothed regime probabilities [T, K]
    """
    # Use posterior mean parameters
    mu_post = trace.posterior['mu'].mean(dim=['chain', 'draw']).values
    sigma_post = trace.posterior['sigma'].mean(dim=['chain', 'draw']).values
    p_trans_post = trace.posterior['p_transition'].mean(dim=['chain', 'draw']).values
    p_init_post = trace.posterior['p_initial'].mean(dim=['chain', 'draw']).values

    n = len(returns)
    K = n_regimes

    # Forward-backward algorithm for regime probabilities
    # Forward pass
    log_alpha = np.zeros((n, K))
    log_alpha[0, :] = (
        np.log(p_init_post) +
        stats.norm.logpdf(returns[0], mu_post, sigma_post)
    )

    for t in range(1, n):
        for k in range(K):
            log_trans = np.log(p_trans_post[:, k])
            log_emission = stats.norm.logpdf(returns[t], mu_post[k], sigma_post[k])
            log_alpha[t, k] = (
                log_emission +
                np.logaddexp.reduce(log_alpha[t-1, :] + log_trans)
            )

    # Backward pass
    log_beta = np.zeros((n, K))
    log_beta[-1, :] = 0  # P(no future | s_T) = 1

    for t in range(n-2, -1, -1):
        for k in range(K):
            log_trans = np.log(p_trans_post[k, :])
            log_emission = stats.norm.logpdf(returns[t+1], mu_post, sigma_post)
            log_beta[t, k] = np.logaddexp.reduce(
                log_trans + log_emission + log_beta[t+1, :]
            )

    # Smoothed probabilities
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    regime_probs = np.exp(log_gamma)

    return regime_probs


# Generate data
returns, true_regimes = generate_regime_switching_data(n=500)

# Fit model
print("Fitting Markov-switching volatility model...")
model, trace = fit_markov_switching_volatility(returns, n_regimes=2)

# Decode regimes
regime_probs = decode_regimes(returns, trace, n_regimes=2)
inferred_regimes = np.argmax(regime_probs, axis=1)

# Plot results
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Plot 1: Returns with true regimes
axes[0].plot(returns, linewidth=0.8, color='black', alpha=0.7)
axes[0].fill_between(range(len(returns)),
                     returns.min(), returns.max(),
                     where=(true_regimes == 0),
                     alpha=0.2, color='blue', label='Low-vol regime (true)')
axes[0].fill_between(range(len(returns)),
                     returns.min(), returns.max(),
                     where=(true_regimes == 1),
                     alpha=0.2, color='red', label='High-vol regime (true)')
axes[0].set_ylabel('Returns (%)')
axes[0].set_title('Returns with True Regime Labels')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Returns with inferred regimes
axes[1].plot(returns, linewidth=0.8, color='black', alpha=0.7)
axes[1].fill_between(range(len(returns)),
                     returns.min(), returns.max(),
                     where=(inferred_regimes == 0),
                     alpha=0.2, color='blue', label='Low-vol regime (inferred)')
axes[1].fill_between(range(len(returns)),
                     returns.min(), returns.max(),
                     where=(inferred_regimes == 1),
                     alpha=0.2, color='red', label='High-vol regime (inferred)')
axes[1].set_ylabel('Returns (%)')
axes[1].set_title('Returns with Inferred Regime Labels')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Plot 3: Regime probabilities
axes[2].fill_between(range(len(returns)),
                     0, regime_probs[:, 0],
                     alpha=0.5, color='blue', label='P(Low-vol)')
axes[2].fill_between(range(len(returns)),
                     regime_probs[:, 0], 1,
                     alpha=0.5, color='red', label='P(High-vol)')
axes[2].set_ylabel('Regime Probability')
axes[2].set_xlabel('Time')
axes[2].set_title('Regime Probability (Smoothed)')
axes[2].legend()
axes[2].grid(alpha=0.3)

# Plot 4: Posterior distributions of volatilities
import arviz as az
az.plot_posterior(trace, var_names=['sigma', 'mu'], ax=axes[3])
axes[3].set_title('Posterior Distributions of Regime Parameters')

plt.tight_layout()
plt.savefig('regime_switching_volatility.png', dpi=150, bbox_inches='tight')
plt.show()

# Print parameter estimates
print("\nPosterior Parameter Estimates:")
print(az.summary(trace, var_names=['sigma', 'mu', 'p_transition']))
```

</div>
</div>

### Simplified Implementation (hmmlearn)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
from hmmlearn import hmm

def fit_gaussian_hmm(returns, n_components=2):
    """
    Fit Gaussian HMM using hmmlearn (simpler but less flexible).

    Args:
        returns: Return series
        n_components: Number of regimes

    Returns:
        model: Fitted HMM
        hidden_states: Most likely state sequence
    """
    # Reshape for hmmlearn
    X = returns.reshape(-1, 1)

    # Initialize model
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type='diag',
        n_iter=100,
        random_state=42
    )

    # Fit
    model.fit(X)

    # Decode states
    hidden_states = model.predict(X)

    # Get parameters
    means = model.means_.flatten()
    stds = np.sqrt(model.covars_.flatten())

    print(f"\nRegime Parameters (hmmlearn):")
    for i in range(n_components):
        print(f"  Regime {i}: μ = {means[i]:.3f}, σ = {stds[i]:.3f}")

    print(f"\nTransition Matrix:")
    print(model.transmat_)

    return model, hidden_states


# Fit with hmmlearn
hmm_model, hmm_states = fit_gaussian_hmm(returns, n_components=2)
```

</div>
</div>

### Volatility Forecasting

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
def forecast_volatility(trace, returns, horizon=20):
    """
    Forecast volatility considering regime uncertainty.

    Args:
        trace: MCMC samples
        returns: Historical returns
        horizon: Forecast horizon

    Returns:
        volatility_forecast: [horizon] array of expected volatility
    """
    # Decode current regime probabilities
    regime_probs = decode_regimes(returns, trace)
    current_regime_prob = regime_probs[-1, :]  # P(s_T = k)

    # Get posterior regime parameters
    sigma_samples = trace.posterior['sigma'].values.reshape(-1, 2)
    p_trans_samples = trace.posterior['p_transition'].values.reshape(-1, 2, 2)

    vol_forecast = []

    for h in range(1, horizon + 1):
        # For each posterior sample
        vol_h = []
        for i in range(len(sigma_samples)):
            sigma = sigma_samples[i]
            P = p_trans_samples[i]

            # Forecast regime distribution h steps ahead
            regime_dist_h = current_regime_prob @ np.linalg.matrix_power(P, h)

            # Expected volatility = weighted average
            expected_vol = np.sum(regime_dist_h * sigma)
            vol_h.append(expected_vol)

        vol_forecast.append(np.mean(vol_h))

    return np.array(vol_forecast)


# Forecast volatility
vol_forecast = forecast_volatility(trace, returns, horizon=20)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), vol_forecast, 'o-', linewidth=2, markersize=6)
plt.xlabel('Forecast Horizon (days)')
plt.ylabel('Expected Volatility (%)')
plt.title('Volatility Forecast with Regime Uncertainty')
plt.grid(alpha=0.3)
plt.savefig('volatility_forecast.png', dpi=150, bbox_inches='tight')
plt.show()
```

</div>
</div>

## Common Pitfalls

**1. Label Switching Problem**
- **Problem**: MCMC can swap regime labels across iterations (regime 1 ↔ regime 2)
- **Symptom**: Posterior distributions bimodal or averaged across regimes
- **Solution**: Impose ordering constraint (σ₁ < σ₂) or post-process samples

**2. Too Many Regimes**
- **Problem**: Fitting K=5 regimes when K=2 is sufficient
- **Symptom**: Overfitting, unstable estimates, poor forecasts
- **Solution**: Use model selection (BIC, WAIC), start with K=2

**3. Ignoring Regime Uncertainty**
- **Problem**: Using only most-likely regime for forecasts
- **Symptom**: Overconfident predictions
- **Solution**: Marginalize over regimes weighted by probabilities

**4. Short Regime Episodes**
- **Problem**: High-volatility regimes last only 2-3 days
- **Symptom**: Model cannot reliably identify short regimes
- **Solution**: May need higher-frequency data or alternative model

**5. Regime Identification in Real-Time**
- **Problem**: Smoothed probabilities use future data (not available in trading)
- **Symptom**: Look-ahead bias
- **Solution**: Use filtered probabilities (forward algorithm only) for real-time

## Connections

**Builds on:**
- Module 7.1: HMM foundations (regime transitions)
- Module 7.2: Change point detection (discrete vs continuous regime switching)
- GARCH models (time-varying volatility without regimes)

**Leads to:**
- Option pricing (regime-dependent volatility)
- Risk management (tail risk in high-vol regime)
- Portfolio optimization (regime-conditional strategies)

**Extensions:**
- Markov-switching DSGE models (macro regimes)
- Regime-switching copulas (multivariate dependence)
- Neural HMMs (learning regime dynamics with deep learning)

## Practice Problems

1. **Regime Persistence**
   Transition matrix:
   ```
   P = [[0.95, 0.05],
        [0.20, 0.80]]
   ```
   - If currently in regime 1, what's probability of being in regime 1 after 5 days?
   - What's the long-run fraction of time in each regime? (Hint: stationary distribution)

2. **Volatility Regimes**
   Data shows:
   - 80% of days: returns ~ N(0.1%, 2%)
   - 20% of days: returns ~ N(-0.2%, 6%)

   - Design a two-regime model
   - What transition matrix achieves 80/20 split?

3. **Conditional Forecasting**
   Currently in high-vol regime (σ₂ = 5%)
   P(stay in high-vol) = 0.9
   P(switch to low-vol) = 0.1 (σ₁ = 2%)

   - Expected volatility tomorrow?
   - Expected volatility in 5 days?

4. **Model Comparison**
   - Single-regime: BIC = 2500
   - Two-regime: BIC = 2350
   - Three-regime: BIC = 2380

   Which model is preferred? Why?

5. **Real-Time Application**
   Trading system needs volatility estimate for VaR
   - Smoothed probability: P(high-vol | all data) = 0.7
   - Filtered probability: P(high-vol | data up to now) = 0.5

   Which should you use? Why does this matter?


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving regime-switching volatility models, what would be your first three steps to apply the techniques from this guide?
</div>

## Further Reading

**Foundational:**
1. **Hamilton (1989)** - "A New Approach to Economic Analysis of Nonstationary Time Series" - Original MS model
2. **Gray (1996)** - "Modeling the Conditional Distribution of Interest Rates" - MS-GARCH
3. **Kim & Nelson (1999)** - *State-Space Models with Regime Switching* - Comprehensive book

**Bayesian Approaches:**
4. **Frühwirth-Schnatter (2006)** - *Finite Mixture and Markov Switching Models* - Bayesian estimation
5. **Billio & Casarin (2011)** - "Identifying Business Cycle Turning Points with Sequential Monte Carlo" - Particle filters

**Commodity Applications:**
6. **Fong & See (2002)** - "A Markov Switching Model of Gold Price Dynamics" - Precious metals
7. **Nomikos & Andriosopoulos (2012)** - "Modelling Energy Spot Prices" - Oil/gas regimes

**Implementation:**
8. **hmmlearn Documentation** - Practical HMM fitting
9. **PyMC Examples: Regime Switching** - Bayesian implementation patterns

**Extensions:**
10. **Haas et al. (2004)** - "A New Approach to Markov-Switching GARCH Models" - Advanced GARCH regimes


<div class="callout-key">
<strong>Key Concept Summary:</strong> Regime-switching volatility models capture the observation that market volatility changes dramatically across different states—calm periods with low volatility versus crisis periods with high volatility.
</div>

---

*"Volatility isn't constant—it switches between regimes. Model both calm and crisis periods to capture true risk."*

---

## Cross-References

<a class="link-card" href="./03_regime_switching_volatility_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_hmm_from_scratch.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

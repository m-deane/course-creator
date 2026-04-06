# Module 6 Cheatsheet: Advanced Topics in Bandits

> **Reading time:** ~20 min | **Module:** 06 — Advanced Topics | **Prerequisites:** Module 5


## Algorithm Selection Decision Tree


<div class="callout-key">

**Key Concept Summary:** This guide covers the core concepts of module 6 cheatsheet: advanced topics in bandits, with worked examples and practical implementation guidance.

</div>

```
START: Is your environment non-standard?
│
├─ NO → Use standard algorithms (Module 2)
│       Thompson Sampling, UCB, ε-greedy
│
└─ YES → What's the problem?
    │
    ├─ Rewards change over time
    │   │
    │   ├─ Gradual drift? → Discounted Thompson Sampling
    │   │                    γ = 0.9-0.99
    │   │
    │   ├─ Abrupt regime changes? → Sliding-Window UCB
    │   │                            W = 200-500
    │   │
    │   └─ Detectable breakpoints? → Change-Point Detection + Restart
    │                                 CUSUM or Bayesian CPD
    │
    ├─ Unselected arms evolve
    │   │
    │   ├─ Can observe all arms? → Multi-Armed Restless Bandit
    │   │                           (track all, select best)
    │   │
    │   └─ Can only observe one? → Greedy Restless Bandit
    │                               with recency penalty λ
    │
    └─ Adversarial rewards
        │
        ├─ Known adversary? → EXP3
        │                     γ = sqrt(K ln K / T)
        │
        └─ Market impact? → Assess impact size:
                            │
                            ├─ Large → EXP3 (randomize to avoid exploitation)
                            └─ Small → Stochastic bandits (better regret)
```

## Non-Stationary Algorithm Comparison

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


| Algorithm | Best For | Hyperparameter | Adaptation Speed | Regret vs Static Best |
|-----------|----------|----------------|------------------|----------------------|
| **Discounted Thompson Sampling** | Gradual drift, smooth transitions | γ ∈ (0.9, 0.99) | Fast (exponential) | O(√T) |
| **Sliding-Window UCB** | Abrupt changes, regime shifts | W = 200-500 | Medium (linear) | O(√T) |
| **Change-Point Detection** | Known breakpoints, detectable shifts | Threshold δ | Very fast (immediate restart) | O(K log T) per regime |
| **Standard TS/UCB** | Stationary rewards | — | N/A (doesn't adapt) | O(log T) |

**Rule of thumb:**
- Use **discounting** when you don't know when changes happen
- Use **sliding windows** when changes are abrupt and you want hard cutoffs
- Use **change-point detection** when you can afford computational cost for active monitoring

## Change-Point Detection Methods

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


| Method | Detection Lag | False Alarm Rate | Computational Cost |
|--------|---------------|------------------|-------------------|
| **CUSUM** | Low (5-20 steps) | Medium | Low (online, O(1)) |
| **Bayesian CPD** | Low (1-10 steps) | Low | High (MCMC) |
| **Mann-Kendall** | High (30-50 steps) | Low | Medium (O(n²)) |
| **Simple Threshold** | Medium (10-30 steps) | High | Very low |

**Commodity applications:**
- **CUSUM:** Real-time regime detection (fast, good for HFT)
- **Bayesian CPD:** Post-analysis, high accuracy (good for backtesting)
- **Threshold:** Quick-and-dirty (moving average crossover)

## Key Formulas

### Discounted Thompson Sampling
```
α_i(t) = 1 + Σ_{s=1}^{t-1} γ^(t-s) · r_s · 𝟙(arm_s = i)
β_i(t) = 1 + Σ_{s=1}^{t-1} γ^(t-s) · (1-r_s) · 𝟙(arm_s = i)

Sample: θ_i ~ Beta(α_i, β_i)
Select: argmax_i θ_i

Discount factor: γ ∈ (0.9, 0.99)
Half-life: ln(0.5) / ln(γ)
```

### Sliding-Window UCB
```
UCB_i = μ_i,W + sqrt((2 ln t) / T_i,W)

Where:
  μ_i,W = mean reward of arm i in last W pulls
  T_i,W = times arm i was pulled in last W pulls

Window size: W = 200-500 (tune to regime duration)
```

### CUSUM Change Detection
```
S_t = max(0, S_{t-1} + (x_t - μ₀) - k)

If S_t > h: signal change detected

Where:
  x_t = observed reward at time t
  μ₀ = baseline mean (pre-change)
  k = slack parameter (noise tolerance)
  h = threshold (sensitivity)

Typical: k = 0.5·σ, h = 4-5·σ
```

### EXP3 (Adversarial)
```
p_i(t) = (1-γ) · w_i(t) / Σ_j w_j(t) + γ/K

w_i(t+1) = w_i(t) · exp(γ · r̂_i(t) / K)

r̂_i(t) = r_i(t) / p_i(t)  (inverse probability weighting)

Optimal: γ = sqrt(K ln K / T)
```

### Restless Bandit (Greedy Approximation)
```
Score_i(t) = μ_i - λ · (t - last_observed_i)

Where:
  μ_i = estimated mean of arm i
  λ = recency penalty (0.001-0.05)
  last_observed_i = last time arm i was selected

Select: argmax_i Score_i(t)
```

## Hyperparameter Tuning Guide

### Discount Factor (γ)

| Regime Duration | Recommended γ | Half-Life (periods) |
|----------------|---------------|---------------------|
| 10-20 periods | 0.90 | 7 |
| 50-100 periods | 0.95 | 14 |
| 100-200 periods | 0.97 | 23 |
| 200-500 periods | 0.99 | 69 |

**Formula:** `γ = exp(-1 / regime_duration)`

### Window Size (W)

```
W = α · K · regime_duration

Where:
  K = number of arms
  regime_duration = expected regime length
  α = safety factor (2-5)

Example: 5 arms, 50-period regimes → W = 3 × 5 × 50 = 750
```

### Recency Penalty (λ)

```
λ = reward_difference / acceptable_lag

Example:
  Expected reward difference between arms: 0.02
  Acceptable lag before re-checking: 20 periods
  λ = 0.02 / 20 = 0.001
```

### EXP3 Exploration (γ)

```
Theory: γ = sqrt(K ln K / T)

Practical:
  Short horizon (T < 1000): γ = 0.1-0.3
  Medium horizon (1000 < T < 10000): γ = 0.05-0.1
  Long horizon (T > 10000): Use theoretical formula
```

## Common Pitfalls & Fixes

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| **Discount too high** | Slow adaptation to regime changes | Lower γ (try 0.95 → 0.90) |
| **Discount too low** | Thrashing, high variance | Raise γ (try 0.90 → 0.95) |
| **Window too small** | Noisy estimates, unstable | Increase W (try 2-5× current) |
| **Window too large** | Slow adaptation | Decrease W to ~2× regime duration |
| **Recency penalty too high** | Perpetual exploration | Lower λ (try 0.1× current) |
| **Recency penalty too low** | Ignoring unobserved arms | Raise λ (try 2-3× current) |
| **Using EXP3 on stochastic** | Poor regret vs UCB/TS | Switch to stochastic algorithms |
| **Using UCB on adversarial** | Getting exploited | Switch to EXP3 |

## Diagnostic Tests

### Is My Environment Non-Stationary?

**Test 1: Split-Sample Comparison**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Split data in half
first_half_mean = rewards[:T//2].mean(axis=0)
second_half_mean = rewards[T//2:].mean(axis=0)

# If means differ significantly → non-stationary
from scipy.stats import ttest_ind
p_value = ttest_ind(rewards[:T//2], rewards[T//2:])[1]

if p_value < 0.05:
    print("Non-stationary detected")
```

</div>

**Test 2: Rolling Mean Variance**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
rolling_means = pd.Series(rewards).rolling(50).mean()
variance_of_rolling_means = rolling_means.var()

# High variance of rolling means → non-stationary
threshold = overall_variance / 10
if variance_of_rolling_means > threshold:
    print("Non-stationary detected")
```

</div>

### Is My Environment Adversarial?

**Test: Backtest UCB vs EXP3**
```python
ucb_regret = run_algorithm(UCB, historical_data)
exp3_regret = run_algorithm(EXP3, historical_data)

if exp3_regret < 0.9 * ucb_regret:
    print("Adversarial dynamics likely present")
else:
    print("Stochastic assumptions seem fine")
```

### Do I Have Restless Arms?

**Test: Correlation Between Selection and Reward**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
for arm_i in arms:
    selected = (history['arm'] == arm_i).astype(int)
    reward = history['reward']

    # If selecting arm_i affects its future rewards → restless
    autocorr = pd.Series(reward).autocorr(lag=1)

    if autocorr < -0.1:  # Negative autocorr suggests depletion
        print(f"Arm {arm_i} appears restless (decays when selected)")
```

</div>

## Commodity-Specific Guidelines

### Energy Commodities (Oil, Gas)
- **High volatility, frequent shocks** → γ = 0.90-0.95, W = 100-200
- **Geopolitical events** → Use change-point detection
- **Seasonality** → Don't use non-stationary (seasonal ≠ non-stationary)

### Metals (Gold, Copper, Silver)
- **Slower regime changes** → γ = 0.97-0.99, W = 300-500
- **Macro-driven** → Combine with contextual features (Fed policy, inflation)

### Agricultural (Corn, Wheat, Soybeans)
- **Strong seasonality** → Use seasonal adjustments, not non-stationary algorithms
- **Weather shocks** → Change-point detection for drought/flood events
- **Harvest cycles** → Standard bandits within-season, restart between seasons

### High-Frequency Trading
- **Market impact** → EXP3 for large positions, UCB for small
- **Fast adaptation** → γ = 0.90-0.92, W = 50-100
- **Adversarial** → Always consider EXP3 when other HFTs present

## When NOT to Use Advanced Algorithms

| Scenario | Recommended Approach | Reason |
|----------|---------------------|--------|
| A/B testing static variants | Standard Thompson Sampling | Variants don't change, no non-stationarity |
| Small sample sizes (T < 100) | ε-greedy or UCB | Advanced algorithms need data to adapt |
| No regime changes observed | Standard algorithms | Over-engineering, worse regret |
| Academic/research setting | Standard algorithms first | Establish baseline before complexity |

## Quick Reference: Algorithm Code Snippets

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Discounted Thompson Sampling (3 lines)
```python
self.alpha *= gamma; self.beta_params *= gamma
self.alpha[arm] += reward; self.beta_params[arm] += (1-reward)
samples = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta_params)]
```

### Sliding-Window UCB (3 lines)
```python
counts = np.bincount([a for a, r in self.history[-W:]], minlength=K)
sums = np.bincount([a for a, r in self.history[-W:]], weights=[r for a, r in self.history[-W:]], minlength=K)
ucb = sums/np.maximum(counts, 1) + np.sqrt(2*np.log(t)/np.maximum(counts, 1))
```

### CUSUM Change Detection (3 lines)
```python
self.S = max(0, self.S + (x - self.mu0) - self.k)
if self.S > self.h:
    self.reset_bandit()  # Restart exploration
```

### EXP3 (3 lines)
```python
probs = (1-gamma)*(weights/weights.sum()) + gamma/K
arm = np.random.choice(K, p=probs)
weights[arm] *= np.exp(gamma * reward/probs[arm] / K)
```

### Restless Greedy (2 lines)
```python
staleness = t - last_observed
scores = means - lambda_penalty * staleness
```

## Further Reading Pointers

- **Non-Stationary:** Garivier & Moulines (2011), "On Upper-Confidence Bound Policies for Switching Bandit Problems"
- **Restless:** Whittle (1988), "Restless Bandits: Activity Allocation in a Changing World"
- **Adversarial:** Auer et al. (2002), "The Nonstochastic Multiarmed Bandit Problem"
- **Change-Point:** Adams & MacKay (2007), "Bayesian Online Changepoint Detection"

See `resources/additional_readings.md` for full references.


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary tradeoff this approach makes compared to simpler alternatives?

**Practice Question 2:** Under what conditions would this approach fail or underperform?


---

## Cross-References

<a class="link-card" href="./01_non_stationary_bandits.md">
  <div class="link-card-title">01 Non Stationary Bandits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_non_stationary_bandits.md">
  <div class="link-card-title">01 Non Stationary Bandits — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_restless_bandits.md">
  <div class="link-card-title">02 Restless Bandits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_restless_bandits.md">
  <div class="link-card-title">02 Restless Bandits — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_adversarial_bandits.md">
  <div class="link-card-title">03 Adversarial Bandits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_adversarial_bandits.md">
  <div class="link-card-title">03 Adversarial Bandits — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


# Non-Stationary Bandits

> **Reading time:** ~20 min | **Module:** 06 — Advanced Topics | **Prerequisites:** Module 5


## In Brief


<div class="callout-key">

**Key Concept Summary:** Non-stationary bandits handle environments where reward distributions change over time — the best option today might not be best tomorrow. In commodity trading, regime shifts (seasonal patterns, su...

</div>

Non-stationary bandits handle environments where reward distributions change over time — the best option today might not be best tomorrow. In commodity trading, regime shifts (seasonal patterns, supply shocks, macro changes) make non-stationarity the norm, not the exception.

> 💡 **Key Insight:** **Standard bandits assume stationarity:** The expected reward of an arm doesn't change over time. But when WTI crude's risk/return profile shifted dramatically in 2020, or when natural gas volatility spiked in 2022, algorithms that weighted 2019 data equally with 2022 data made poor decisions.

**The fix:** Weight recent observations more heavily. Either forget old data explicitly (sliding windows) or discount it exponentially (decay factors).

## Visual Explanation

```
Standard Thompson Sampling (FAILS):
Time:     0 ─────────── 500 ─────────── 1000 ─────────── 1500 ────→
Arm A:    ████████████  (best)           ░░░░░░░░░░░░    (now worse)
Arm B:    ░░░░░░░░░░░░  (worse)          ████████████    (now best!)
                                              ↑
                                    Regime shift happens,
                                    but algorithm is stuck on Arm A
                                    (too much historical evidence)

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


Discounted Thompson Sampling (ADAPTS):
Time:     0 ─────────── 500 ─────────── 1000 ─────────── 1500 ────→
Arm A:    ████████████                   → detects decline → switches
Arm B:    ░░░░░░░░░░░░                   → detects improvement → selects
                                              ↑
                                    Recent data weighted heavily,
                                    algorithm adapts within ~50-100 pulls
```

**Sliding-Window UCB:**
```
Only use last W=200 observations:

[────────── ancient history (ignored) ──────────][─ recent W pulls ─]
                                                        ↑
                                                Only this counts for
                                                confidence bounds
```

## Formal Definition

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Discounted Thompson Sampling

For each arm `i`, maintain discounted success/failure counts:

```
α_i(t) = 1 + Σ_{s=1}^{t-1} γ^(t-s) · r_s · 𝟙(arm_s = i)
β_i(t) = 1 + Σ_{s=1}^{t-1} γ^(t-s) · (1-r_s) · 𝟙(arm_s = i)
```

Where:
- `γ ∈ (0, 1)` is the discount factor (typical: 0.9-0.99)
- `γ^(t-s)` exponentially down-weights observations from time `s < t`
- Recent rewards get weight ≈1, rewards from 100 steps ago get weight ≈γ^100

Sample: `θ_i ~ Beta(α_i, β_i)` and select `argmax_i θ_i`

### Sliding-Window UCB

For each arm `i`, compute UCB using only the last `W` pulls:

```
Let T_i,W = number of times arm i pulled in last W steps
Let μ_i,W = average reward of arm i in last W steps

UCB_i = μ_i,W + sqrt((2 ln t) / T_i,W)
```

Select `argmax_i UCB_i`

**Trade-off:**
- Small `W` → fast adaptation, but high variance (noisy estimates)
- Large `W` → stable estimates, but slow adaptation
- Typical: `W = 200-500` for commodity applications

## Intuitive Explanation

**It's like navigating with a GPS that updates.**

Imagine driving with GPS directions. If the GPS uses traffic data from last week equally with real-time data, it'll route you into jams that cleared days ago. You'd trust recent updates more.

**Discounting = exponential forgetting:**
- Yesterday's traffic: weight 0.99
- Last week's traffic: weight 0.99^7 ≈ 0.93
- Last month's traffic: weight 0.99^30 ≈ 0.74
- Last year's traffic: weight 0.99^365 ≈ 0.03 (almost ignored)

**Sliding window = hard cutoff:**
- Last 200 observations: count them fully
- Observation #201 and older: completely ignored

## Code Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from scipy.stats import beta

class DiscountedThompsonSampling:
    def __init__(self, n_arms, gamma=0.95):
        """
        Discounted Thompson Sampling for non-stationary rewards.

        Args:
            n_arms: Number of arms
            gamma: Discount factor (0 < gamma < 1). Higher = slower forgetting.
        """
        self.n_arms = n_arms
        self.gamma = gamma
        self.alpha = np.ones(n_arms)  # Pseudo-counts for successes
        self.beta_params = np.ones(n_arms)  # Pseudo-counts for failures

    def select_arm(self):
        # Sample from Beta(alpha, beta) for each arm
        samples = [beta.rvs(a, b) for a, b in zip(self.alpha, self.beta_params)]
        return np.argmax(samples)

    def update(self, arm, reward):
        # Discount all arms' parameters
        self.alpha *= self.gamma
        self.beta_params *= self.gamma

        # Update selected arm with new observation
        self.alpha[arm] += reward
        self.beta_params[arm] += (1 - reward)

# Example usage
bandit = DiscountedThompsonSampling(n_arms=3, gamma=0.95)

for t in range(1000):
    arm = bandit.select_arm()
    reward = get_reward(arm, t)  # Your reward function
    bandit.update(arm, reward)
```

</div>

**Sliding-Window UCB:**

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from collections import deque

class SlidingWindowUCB:
    def __init__(self, n_arms, window_size=200):
        self.n_arms = n_arms
        self.window_size = window_size
        self.history = deque(maxlen=window_size)  # (arm, reward) pairs
        self.t = 0

    def select_arm(self):
        self.t += 1

        # Compute counts and means from window
        counts = np.zeros(self.n_arms)
        sums = np.zeros(self.n_arms)
        for arm, reward in self.history:
            counts[arm] += 1
            sums[arm] += reward

        # UCB calculation
        ucb_values = []
        for i in range(self.n_arms):
            if counts[i] == 0:
                ucb_values.append(float('inf'))  # Force exploration
            else:
                mean = sums[i] / counts[i]
                exploration = np.sqrt(2 * np.log(self.t) / counts[i])
                ucb_values.append(mean + exploration)

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        self.history.append((arm, reward))
```

</div>

## Common Pitfalls

### Pitfall 1: Discount Factor Too High (γ ≈ 1.0)
**What happens:** Algorithm becomes nearly stationary — adapts too slowly.

**Example:** With γ=0.99, data from 100 steps ago still has weight 0.99^100 ≈ 0.366. If regimes change every 50 steps, you're averaging across multiple regimes.

**Fix:** Tune γ based on expected regime duration. For commodity regimes lasting ~100-200 periods:
- γ = 0.95 → half-life of 14 periods
- γ = 0.99 → half-life of 69 periods

**Rule of thumb:** Set γ such that `1/(1-γ)` ≈ expected regime duration.

### Pitfall 2: Window Too Small
**What happens:** High variance in estimates → thrashing between arms on noise.

**Example:** With window_size=20, a few unlucky draws can make the best arm look worst.

**Fix:** Window should contain enough samples for stable estimates. Minimum: 50-100 observations per arm. For 5 arms, use window ≥ 250-500.

### Pitfall 3: Not Handling Regime Detection
**What happens:** Slow adaptation to abrupt changes.

**Example:** Discounting with γ=0.95 still takes ~50 steps to adapt. If a supply shock changes everything instantly, you lose 50 steps of regret.

**Fix:** Combine non-stationary bandits with change-point detection (see Module 6.2). When a regime shift is detected, reset beliefs and re-explore.

### Pitfall 4: Over-Adapting to Noise
**What happens:** Treating random fluctuations as regime changes.

**Example:** Commodity returns are volatile. A single bad day doesn't mean the regime changed.

**Fix:**
- Use longer windows or higher discount factors for noisy environments
- Apply change-point tests (CUSUM, Bayesian online changepoint detection)
- Require statistical significance before declaring regime shift

## Connections

### Builds On
- **Thompson Sampling (Module 2):** Discounted TS extends standard TS with exponential forgetting
- **UCB (Module 2):** Sliding-window UCB adapts the confidence bound framework
- **Bayesian Inference:** Discounting modifies the Bayesian update to prioritize recent data

### Leads To
- **Change-Point Detection (Module 6.2):** Explicit detection of regime boundaries
- **Restless Bandits (Module 6.3):** When arms evolve even without being selected
- **Contextual Non-Stationary Bandits:** Combine context (Module 3) with non-stationarity

### Related Concepts
- **Kalman Filters:** Track time-varying parameters (more complex but more flexible)
- **Online Learning:** Regret bounds for non-stationary environments
- **Regime-Switching Models:** Econometric approach to modeling structural breaks

## Practice Problems

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Problem 1: Commodity Seasonality
Agricultural commodities (corn, wheat) have seasonal patterns. Would you use discounting or sliding windows? What window size or γ would you choose for monthly data?

**Hint:** Seasons repeat annually. Think about whether you want to forget last year's harvest season.

### Problem 2: Tuning Discount Factors
You observe that a commodity regime lasts approximately 150 trading days on average. What discount factor γ should you use to give observations from 150 days ago weight ≈ 0.1?

**Hint:** Solve `γ^150 = 0.1` for γ.

### Problem 3: Hybrid Approach
Design an algorithm that uses sliding-window UCB normally, but switches to aggressive exploration (ε-greedy with high ε) when a change-point is detected. Sketch the logic.

### Problem 4: Performance Comparison
You run standard Thompson Sampling and Discounted Thompson Sampling on a bandit with 3 arms. Arm 1 is best for steps 1-500, Arm 2 is best for 501-1000, Arm 3 is best for 1001-1500.

Which algorithm will have:
- Lower regret in the first 500 steps?
- Lower regret overall?
- Faster adaptation after each regime change?

**Hint:** Think about the explore-exploit trade-off. Discounting explores more continuously.

## Commodity Application Example

**Scenario:** You're allocating between 3 energy commodities (WTI crude, natural gas, heating oil). Historical correlations and volatilities shift with:
- Seasonal demand (winter heating, summer driving)
- Geopolitical events (Russia-Ukraine impact on natural gas)
- Economic cycles (recessions reduce oil demand)

**Standard Thompson Sampling problem:**
During COVID-19 crash (March 2020), oil went negative. Standard TS with data from 2019 still believed oil was the best risk-adjusted asset for months afterward.

**Discounted TS solution:**
With γ=0.95 (half-life ~14 days), post-crash data quickly dominated. Algorithm shifted to natural gas and heating oil within 2-3 weeks.

**Implementation:**
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Pseudo-code
returns_dict = load_commodity_returns(['WTI', 'NATGAS', 'HEAT'])

bandit = DiscountedThompsonSampling(n_arms=3, gamma=0.95)

for date in trading_dates:
    arm = bandit.select_arm()  # Select commodity
    commodity = ['WTI', 'NATGAS', 'HEAT'][arm]

    # Normalize return to [0, 1] for Bernoulli reward
    reward = normalize_return(returns_dict[commodity][date])

    bandit.update(arm, reward)

    # Allocate capital to selected commodity
    allocate(commodity, capital)
```

</div>

**Key insight:** Discount factor becomes a tunable parameter for regime sensitivity. Backtest with historical regime changes to calibrate.


---

## Cross-References

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

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


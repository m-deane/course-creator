# Module 0 Quick Reference: Foundations

> **Reading time:** ~20 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## A/B Testing vs Bandits


<div class="callout-key">

**Key Concept Summary:** Reject null hypothesis if |z| > z_α/2 (e.g., 1.96 for α=0.05).

</div>

| Aspect | A/B Testing | Multi-Armed Bandits |
|--------|-------------|---------------------|
| **Allocation** | Fixed 50/50 split | Adaptive, shifts to winner |
| **Goal** | Statistical certainty | Minimize regret while learning |
| **Regret growth** | O(T) — linear | O(log T) — logarithmic |
| **Best for** | Static environments, one-time decisions | Dynamic environments, ongoing decisions |
| **Sample efficiency** | Low (wastes traffic on loser) | High (exploits winner ASAP) |
| **Non-stationarity** | Fails (assumes fixed means) | Can adapt (with sliding windows) |

**When to use A/B testing:**
- You need p-values for stakeholders/regulators
- Decisions are rare and high-stakes (launch new product)
- Environment is stable over test period

**When to use bandits:**
- Decisions are frequent (every trade, every ad impression)
- Opportunity cost of testing is high (live capital)
- Environment may shift (market regimes)

---

## A/B Test Formulas

### Sample Size (Per Variant)
```
n = 2(z_α/2 + z_β)² · p̄(1-p̄) / (p_B - p_A)²
```
where:
- z_α/2: critical value for significance level α (1.96 for α=0.05)
- z_β: critical value for power 1-β (0.84 for 80% power)
- p̄ = (p_A + p_B) / 2

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


**Example:** Detect 0.05 → 0.08 conversion rate (60% relative lift):
```
n = 2(1.96 + 0.84)² · 0.065(0.935) / (0.03)²
  ≈ 1,954 per variant → 3,908 total
```

### Test Statistic (Two-Proportion Z-Test)
```
z = (p̂_B - p̂_A) / √(p̂(1-p̂) · (1/n_A + 1/n_B))
```
where p̂ = (x_A + x_B)/(n_A + n_B) is the pooled proportion.

Reject null hypothesis if |z| > z_α/2 (e.g., 1.96 for α=0.05).

---

## Explore-Exploit Tradeoff

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Key Definitions

**Instantaneous regret** at time t:
```
δ(t) = μ* - μ_a(t)
```

**Cumulative regret** over T rounds:
```
R(T) = Σ_{t=1}^T δ(t) = T·μ* - Σ_{t=1}^T r_t
```

**Average regret per round:**
```
R̄(T) = R(T) / T
```

Good algorithms achieve R̄(T) → 0 as T → ∞.

### Regret Bounds by Algorithm

| Algorithm | Regret Bound | When to Use |
|-----------|--------------|-------------|
| Random exploration | O(T) | Never (baseline only) |
| Pure exploitation | O(T) | Never (unless oracle) |
| Epsilon-greedy (fixed ε) | O(T) | Simple, interpretable |
| Epsilon-greedy (decay ε) | O(T^(2/3)) | Better than fixed |
| UCB | O(√(KT log T)) | Theoretically optimal |
| Thompson Sampling | O(√(KT)) | Often best in practice |

**Key insight:** Logarithmic or sublinear regret means bounded average loss per round.

---

## Decision Theory Essentials

### Expected Value
```
E[X] = Σ x_i · P(X = x_i)
```

### Expected Utility
```
EU = Σ U(x_i) · P(x_i)
```

**Common utility functions:**
- Risk-neutral: U(x) = x
- Risk-averse: U(x) = √x or log(x)
- Risk-seeking: U(x) = x²

### Sharpe Ratio (Risk-Adjusted Return)
```
Sharpe = (μ - r_f) / σ
```
Higher Sharpe = better risk-adjusted performance.

**Example:**
- Strategy A: 10% return, 20% vol → Sharpe = 0.5
- Strategy B: 8% return, 10% vol → Sharpe = 0.8
- Risk-averse trader prefers B

---

## Multi-Armed Bandit Notation

| Symbol | Meaning |
|--------|---------|
| K | Number of arms |
| T | Time horizon (total rounds) |
| a(t) | Arm chosen at time t |
| r(t) | Reward received at time t |
| μ_k | True expected reward of arm k |
| μ* | Best arm's expected reward: max_k μ_k |
| μ̂_k | Estimated reward of arm k |
| Q_k | Value estimate for arm k (alternative notation) |
| N_k(t) | Number of times arm k pulled by time t |
| R(T) | Cumulative regret over T rounds |
| δ_k | Suboptimality gap: μ* - μ_k |

---

## When to Use What: Decision Tree

```
START: Do you need to choose between options?
│
├─ One-time decision (product launch, policy change)
│  └─ Use A/B test (fixed allocation, get p-value)
│
├─ Repeated decision (trading, ad serving, content)
│  │
│  ├─ Environment is stable (means don't change)
│  │  └─ Use bandit with decreasing exploration (UCB, epsilon-decay)
│  │
│  └─ Environment shifts (non-stationary)
│     └─ Use bandit with adaptation (sliding window, discounted UCB)
│
├─ Need interpretability for stakeholders
│  └─ Use epsilon-greedy (easy to explain: "try best 90%, explore 10%")
│
├─ Need theoretical guarantees
│  └─ Use UCB (provable logarithmic regret bounds)
│
└─ Need empirical performance
   └─ Use Thompson Sampling (often wins in practice)
```

---

## Commodity Trading Applications

### Mapping Trading Decisions to Bandits

| Trading Decision | Bandit Formulation |
|------------------|-------------------|
| **Sector allocation** | Arms = commodity sectors (energy, metals, ag) |
| **Strategy selection** | Arms = trading strategies (trend, mean-rev, arb) |
| **Contract choice** | Arms = futures contracts (WTI, Brent, Dubai) |
| **Signal testing** | Arms = alpha signals (test new vs current) |
| **Execution venue** | Arms = exchanges/brokers (minimize slippage) |

**Reward definitions:**
- Absolute return: r_t = (P_close - P_open) / P_open
- Risk-adjusted: r_t = Sharpe ratio over window
- Binary: r_t = 1 if profitable, 0 otherwise

**Time horizons:**
- Day trading: T = 100-1000 (trades per month)
- Swing trading: T = 50-200 (positions per quarter)
- Portfolio allocation: T = 12-52 (weeks per year)

---

## Common Pitfalls Checklist

### A/B Testing
- [ ] **Peeking:** Checking results before predetermined sample size (inflates Type I error)
- [ ] **Early stopping:** Stopping when p < 0.05 without correction (p-hacking)
- [ ] **Simpson's paradox:** Ignoring subgroup heterogeneity
- [ ] **Non-stationarity:** Assuming means stay constant over long tests

### Bandits
- [ ] **Premature exploitation:** Trusting small samples (use confidence bounds)
- [ ] **Excessive exploration:** Continuing to explore obviously inferior arms
- [ ] **Ignoring variance:** Treating high-variance arms as high-mean
- [ ] **Stale estimates:** Not adapting to regime changes (use sliding windows)

### Decision Theory
- [ ] **Risk-neutral fallacy:** Ignoring risk aversion (use utility functions)
- [ ] **Sunk cost fallacy:** Continuing failing strategy due to past investment
- [ ] **Horizon mismatch:** Over-exploring on short horizons, under-exploring on long

---

## Quick Code Snippets

### Calculate A/B Test Sample Size
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from scipy.stats import norm
import numpy as np

def ab_sample_size(p_A, p_B, alpha=0.05, power=0.8):
    """Sample size needed per variant."""
    z_alpha = norm.ppf(1 - alpha/2)  # 1.96 for 0.05
    z_beta = norm.ppf(power)          # 0.84 for 0.80
    p_bar = (p_A + p_B) / 2
    n = 2 * (z_alpha + z_beta)**2 * p_bar * (1 - p_bar) / (p_B - p_A)**2
    return int(np.ceil(n))

# Example: 5% → 8% conversion
n = ab_sample_size(0.05, 0.08)
print(f"Need {n} per variant, {2*n} total")
```

</div>

### Calculate Cumulative Regret
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def cumulative_regret(arm_means, choices):
    """Regret for a sequence of arm choices."""
    best_mean = max(arm_means)
    chosen_means = [arm_means[a] for a in choices]
    regret = [best_mean - m for m in chosen_means]
    return np.cumsum(regret)

# Example
arm_means = [0.15, 0.25, 0.18]  # Energy, Metals, Ag
choices = [0, 1, 0, 1, 1, 2, 1, 1]  # Your trades
regret = cumulative_regret(arm_means, choices)
print(f"Total regret: {regret[-1]:.3f}")
```

</div>

### Simple Epsilon-Greedy
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def epsilon_greedy(Q, epsilon=0.1):
    """Choose arm using epsilon-greedy policy."""
    if np.random.rand() < epsilon:
        return np.random.randint(len(Q))  # Explore
    else:
        return np.argmax(Q)  # Exploit

# Example
Q = [0.15, 0.25, 0.18]  # Value estimates
arm = epsilon_greedy(Q, epsilon=0.1)
print(f"Chose arm {arm}")
```

</div>

---

## Resources for Going Deeper

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


**Textbooks:**
- *Bandit Algorithms* by Lattimore & Szepesvári (definitive reference)
- *Introduction to Multi-Armed Bandits* by Slivkins (free, excellent)

**Classic papers:**
- Lai & Robbins (1985): Asymptotically optimal allocation
- Auer et al. (2002): UCB algorithm and finite-time analysis
- Agrawal & Goyal (2012): Thompson Sampling analysis

**Commodity-specific:**
- *Evidence-Based Technical Analysis* by Aronson (hypothesis testing in trading)
- *Systematic Trading* by Carver (portfolio construction under uncertainty)

---

## Next Steps

1. **Complete the three notebooks** in this module to see these concepts in action
2. **Try the exercises** to practice calculations
3. **Move to Module 1** to implement your first bandit algorithm (epsilon-greedy)
4. **Bookmark this cheatsheet** — you'll reference it throughout the course

**Ready to build?** → Start with `notebooks/01_ab_test_simulation.ipynb`


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary tradeoff this approach makes compared to simpler alternatives?

**Practice Question 2:** Under what conditions would this approach fail or underperform?



---

## Cross-References

<a class="link-card" href="./01_ab_testing_limits.md">
  <div class="link-card-title">01 Ab Testing Limits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_ab_testing_limits.md">
  <div class="link-card-title">01 Ab Testing Limits — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_explore_exploit_tradeoff.md">
  <div class="link-card-title">02 Explore Exploit Tradeoff</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_explore_exploit_tradeoff.md">
  <div class="link-card-title">02 Explore Exploit Tradeoff — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_decision_theory_basics.md">
  <div class="link-card-title">03 Decision Theory Basics</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_decision_theory_basics.md">
  <div class="link-card-title">03 Decision Theory Basics — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


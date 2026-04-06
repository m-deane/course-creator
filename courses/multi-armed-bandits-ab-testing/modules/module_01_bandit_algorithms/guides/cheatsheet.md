# Core Bandit Algorithms Cheatsheet

> **Reading time:** ~20 min | **Module:** 01 — Bandit Algorithms | **Prerequisites:** Module 0 Foundations


## Algorithm Comparison Table


<div class="callout-key">

**Key Concept Summary:** This guide covers the core concepts of core bandit algorithms cheatsheet, with worked examples and practical implementation guidance.

</div>

| Algorithm | Selection Rule | Hyperparameters | Regret Bound | Best For |
|-----------|---------------|-----------------|--------------|----------|
| **Epsilon-Greedy** | Random w.p. ε, else argmax Q̂ | ε ∈ [0,1] | O(εT + K/ε) = O(T^(2/3)) | Non-stationary, simplicity |
| **UCB1** | argmax[Q̂ + c√(ln(t)/N)] | c (default √2) | O(ln T) gap-dep., O(√T) worst | Stationary, bounded rewards |
| **Softmax** | Sample from exp(Q̂/τ) | τ > 0 | O(T^(2/3)) | Portfolio allocation, smooth |

## Key Formulas

### Epsilon-Greedy

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Action selection
if random() < epsilon:
    action = random_arm()
else:
    action = argmax(Q)

# Update
N[a] += 1
Q[a] += (reward - Q[a]) / N[a]

# Decaying epsilon
epsilon_t = min(1.0, C / sqrt(t + 1))  # C ∈ [1, 10]
```

</div>
</div>

**Optimal ε:** ε* ≈ (K/T)^(1/3) for T steps, K arms

**Expected regret:** E[R_T] = ε·T + (K·Δ²)/ε

### UCB1

<span class="filename">example.py</span>
</div>
<div class="callout-insight">
**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Action selection (after pulling each arm once)
ucb_values = Q + c * sqrt(log(t) / (N + 1e-10))
action = argmax(ucb_values)

# Update (same as epsilon-greedy)
N[a] += 1
Q[a] += (reward - Q[a]) / N[a]
```

</div>
</div>

**Standard constant:** c = √2

**UCB bonus interpretation:** Confidence radius for high-probability bound

**Expected regret:** E[R_T] ≤ Σ_a (8 ln(T) / Δ_a) + O(1)

### Softmax (Boltzmann)
```python
# Action selection (numerically stable)
q_max = max(Q)
exp_q = exp((Q - q_max) / tau)
probs = exp_q / sum(exp_q)
action = random_choice(K, p=probs)

# Update (same as epsilon-greedy)
N[a] += 1
Q[a] += (reward - Q[a]) / N[a]

# Decaying temperature
tau_t = tau_0 / log(t + 2)
```

**Temperature interpretation:**
- τ → 0: Deterministic (argmax)
- τ → ∞: Uniform random
- τ = 1: Balanced (common default)

**Selection probability:** π(a) = exp(Q̂(a)/τ) / Σ_a' exp(Q̂(a')/τ)

## Decision Guide: Which Algorithm Should I Use?

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Choose **Epsilon-Greedy** if:
- You want simplicity (easiest to implement and debug)
- Your environment is non-stationary (reward distributions change over time)
- You have domain knowledge to set ε
- You need robustness to extreme outliers
- Short time horizon (T < 1000)

**Recommended settings:**
- Fixed: ε ∈ [0.05, 0.2]
- Decaying: ε(t) = 10/√(t+1)

### Choose **UCB1** if:
- Rewards are bounded (e.g., in [0, 1])
- Environment is stationary (reward distributions don't change)
- You want parameter-free algorithm (c=√2 works universally)
- Long time horizon (T > 1000) where logarithmic regret matters
- You want theoretical guarantees

**Recommended settings:**
- c = √2 (standard)
- c = 1 for conservative exploration
- c = 2 for aggressive exploration

### Choose **Softmax** if:
- You want smooth, probabilistic selection
- Portfolio allocation (interpret π(a) as weights)
- Many arms with similar values (softmax focuses on top few)
- Reward scale is consistent (e.g., always in [0, 1])
- You want interpretable selection probabilities

**Recommended settings:**
- τ ∈ [0.1, 1] for rewards in [0, 1]
- Scale τ proportionally if rewards in different range
- Decaying: τ(t) = 1/log(t+2)

## Hyperparameter Tuning Tips

### Epsilon (ε-greedy)
**Too low (ε < 0.01):**
- Symptom: One arm dominates early, regret flattens quickly but at high value
- Fix: Increase ε or use decaying schedule

**Too high (ε > 0.5):**
- Symptom: Regret grows linearly, selection frequencies nearly uniform
- Fix: Decrease ε

**Rule of thumb:** ε·T ≥ 10K (explore each arm ~10 times in expectation)

**Decay schedule:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# Aggressive (fast decay)
epsilon = 1 / sqrt(t + 1)

# Balanced
epsilon = 10 / sqrt(t + 1)

# Conservative (slow decay)
epsilon = min(0.2, 100 / sqrt(t + 1))
```

</div>
</div>

### Exploration Constant (UCB1)
**Too low (c < 0.5):**
- Symptom: Premature convergence, some arms never explored
- Fix: Increase c to 1 or √2

**Too high (c > 5):**
- Symptom: Exploration never stops, selection frequencies nearly uniform
- Fix: Decrease c to √2 or 1

**Rule of thumb:** c ∈ [√2, 2] works for 95% of problems

### Temperature (Softmax)
**Too low (τ < 0.01):**
- Symptom: One arm gets >99% of pulls (deterministic)
- Fix: Increase τ to 0.1-1.0

**Too high (τ > 5):**
- Symptom: Selection probabilities nearly uniform
- Fix: Decrease τ

**Diagnostic check:**
```python
q_range = Q.max() - Q.min()
effective_scale = q_range / tau
# Target: 1 < effective_scale < 5
```

**Rule of thumb:** Start with τ=1, adjust based on Q̂ scale

## Quick Reference: Regret Analysis

### Regret Definition
```
Regret_T = T·μ* - Σ_{t=1}^T r_t

where μ* = max_a μ(a) is the best arm's expected reward
```

**Interpretation:** Total reward lost by not always pulling the best arm.

### Regret Decomposition
```
Regret_T = Σ_a Δ_a · E[N_a(T)]

where:
  Δ_a = μ* - μ(a)  (gap for arm a)
  N_a(T) = number of times arm a pulled in T steps
```

**Key insight:** Minimize pulls to bad arms (large Δ_a).

### Algorithm Regret Bounds

| Algorithm | Gap-Dependent | Gap-Independent (Worst-Case) |
|-----------|---------------|------------------------------|
| **Epsilon-Greedy** | O(K/ε + εT) | O(T^(2/3)) with optimal ε |
| **UCB1** | O(Σ_a ln(T)/Δ_a) | O(√(KT ln T)) |
| **Softmax** | O(Σ_a 1/Δ_a²) | O(T^(2/3)) with optimal τ |
| **Thompson Sampling** | O(Σ_a ln(T)/Δ_a) | O(√(KT)) |

**Gap-dependent:** Assumes gaps Δ_a > 0 are known
**Gap-independent:** No assumptions about gaps (worst-case over all problems)

### Minimax Lower Bound
No algorithm can achieve better than:
```
E[R_T] = Ω(√(KT))
```

**Interpretation:** UCB1 and Thompson Sampling are near-optimal (match lower bound up to log factors).

## Common Pitfalls & Fixes

### All Algorithms

**Pitfall:** Not breaking ties in argmax
```python
# WRONG: Always picks first arm when tied
action = argmax(values)

# CORRECT: Random tie-breaking
max_val = max(values)
best_actions = [a for a in range(K) if values[a] == max_val]
action = random.choice(best_actions)
```

**Pitfall:** Using exponential moving average instead of sample mean
```python
# WRONG (for stationary problems)
Q[a] = 0.9 * Q[a] + 0.1 * reward  # Forgets old data

# CORRECT (sample mean)
N[a] += 1
Q[a] += (reward - Q[a]) / N[a]
```

**Note:** EMA is correct for non-stationary problems.

**Pitfall:** Not initializing properly
```python
# WRONG: Negative rewards break UCB/Softmax
Q = [0, 0, 0, 0, 0]  # If rewards can be negative

# CORRECT: Optimistic initialization or normalization
Q = [10, 10, 10, 10, 10]  # Encourages early exploration
# OR normalize rewards to [0, 1]
```

### Epsilon-Greedy Specific

**Pitfall:** Fixed ε in stationary environment
- Fix: Use decaying ε(t) = C/√t

**Pitfall:** ε too low relative to K and T
- Fix: Ensure ε·T ≥ 10K

### UCB1 Specific

**Pitfall:** Division by zero when N[a]=0
```python
# WRONG
ucb = Q + c * sqrt(log(t) / N)  # N=0 → inf

# CORRECT
ucb = Q + c * sqrt(log(t) / (N + 1e-10))
# OR pull each arm once initially
```

**Pitfall:** Using ln(N[a]) instead of ln(t)
```python
# WRONG
ucb = Q + c * sqrt(log(N) / N)  # Bonus shrinks too fast

# CORRECT
ucb = Q + c * sqrt(log(t) / N)  # t = total time
```

**Pitfall:** Unbounded rewards
- Fix: Normalize to [0, 1] or use UCB-V (variance-aware)

### Softmax Specific

**Pitfall:** Numerical overflow
```python
# WRONG
probs = exp(Q / tau) / sum(exp(Q / tau))  # Overflow if Q large

# CORRECT (log-sum-exp trick)
q_max = max(Q)
exp_q = exp((Q - q_max) / tau)
probs = exp_q / sum(exp_q)
```

**Pitfall:** Temperature on wrong scale
- Fix: Normalize Q to consistent range, or scale τ appropriately

## Code Templates

### Epsilon-Greedy (10 lines)
```python
class EpsilonGreedy:
    def __init__(self, K, epsilon=0.1):
        self.Q, self.N = np.zeros(K), np.zeros(K)
        self.epsilon = epsilon

    def select(self):
        return np.random.randint(len(self.Q)) if np.random.rand() < self.epsilon else np.argmax(self.Q)

    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]
```

### UCB1 (12 lines)
```python
class UCB1:
    def __init__(self, K, c=np.sqrt(2)):
        self.Q, self.N, self.c, self.t = np.zeros(K), np.zeros(K), c, 0

    def select(self):
        self.t += 1
        if self.t <= len(self.Q): return self.t - 1
        ucb = self.Q + self.c * np.sqrt(np.log(self.t) / (self.N + 1e-10))
        return np.argmax(ucb)

    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]
```

### Softmax (12 lines)
```python
class Softmax:
    def __init__(self, K, tau=1.0):
        self.Q, self.N, self.tau = np.zeros(K), np.zeros(K), tau

    def select(self):
        q_max = np.max(self.Q)
        exp_q = np.exp((self.Q - q_max) / self.tau)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(len(self.Q), p=probs)

    def update(self, a, r):
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]
```

## Commodity Trading Examples

### Energy Sector Selection (5 commodities)
```python
# Commodities: WTI, Brent, Nat Gas, Heating Oil, Gasoline
# Reward = daily return (%)

bandit = UCB1(K=5)
for day in range(252):  # 1 trading year
    commodity = bandit.select()
    return_pct = get_daily_return(commodity)
    bandit.update(commodity, normalize_to_01(return_pct))
```

### Cross-Asset Hedging (4 instruments)
```python
# Instruments: Futures, Options, Swaps, Basis Swaps
# Reward = -portfolio_variance (minimize risk)

bandit = EpsilonGreedy(K=4, epsilon=0.1)
for week in range(52):
    instrument = bandit.select()
    variance = -compute_portfolio_variance(instrument)
    bandit.update(instrument, variance)
```

### Adaptive Basket Allocation (10 sectors)
```python
# Sectors: Energy, Metals, Grains, Softs, Livestock, ...
# Reward = Sharpe ratio

bandit = Softmax(K=10, tau=0.5)
allocations = []

for month in range(12):
    probs = bandit.get_probabilities()
    allocations.append(probs * 1e6)  # $1M portfolio

    # Execute and observe
    sharpe = compute_sharpe_ratio(probs)
    bandit.update_all(sharpe)  # Update all arms
```

## Further Reading

- **Lattimore & Szepesvári (2020):** "Bandit Algorithms" - Comprehensive theory
- **Sutton & Barto (2018):** "Reinforcement Learning" - Chapter 2 on bandits
- **Auer et al. (2002):** "Finite-time Analysis of the Multiarmed Bandit Problem" - Original UCB paper
- **Russo et al. (2018):** "A Tutorial on Thompson Sampling" - Bayesian alternative

## Quick Diagnostic Checklist

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


Regret not decreasing:
- [ ] Check ε/τ/c values (too high = too much exploration)
- [ ] Verify update rule (using sample mean, not EMA for stationary)
- [ ] Check for bugs in argmax (tie-breaking)

One arm dominates:
- [ ] Check ε/τ/c values (too low = premature convergence)
- [ ] Verify initialization (optimistic init can help)
- [ ] Check if that arm is actually best (maybe working correctly!)

Selection frequencies uniform:
- [ ] ε/τ too high → decrease
- [ ] UCB bonus too large → decrease c
- [ ] All arms actually have similar values (working correctly)

Numerical errors (NaN, inf):
- [ ] Softmax: use log-sum-exp trick
- [ ] UCB: add small constant to denominator (N + 1e-10)
- [ ] Check reward normalization (should be bounded)

## One-Page Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   MULTI-ARMED BANDIT                        │
│                                                             │
│  Problem: K arms, unknown rewards μ₁,...,μₖ                │
│  Goal: Maximize Σ r_t (minimize regret)                    │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ ε-Greedy    │  │   UCB1      │  │  Softmax    │        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤        │
│  │ if rand<ε:  │  │ a = argmax  │  │ π ∝ exp(Q/τ)│        │
│  │   random    │  │ Q+c√(ln t/N)│  │ sample ~ π  │        │
│  │ else:       │  │             │  │             │        │
│  │   argmax Q  │  │             │  │             │        │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤        │
│  │ O(T^(2/3))  │  │ O(ln T)     │  │ O(T^(2/3))  │        │
│  │ Simple      │  │ Optimal     │  │ Smooth      │        │
│  │ Robust      │  │ Bounded     │  │ Portfolio   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                             │
│  Exploration ←────────────────────────────→ Exploitation   │
│  (learn about arms)              (use best arm)            │
│                                                             │
│  Regret = T·μ* - Σ r_t                                     │
└─────────────────────────────────────────────────────────────┘
```


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary tradeoff this approach makes compared to simpler alternatives?

**Practice Question 2:** Under what conditions would this approach fail or underperform?


---

## Cross-References

<a class="link-card" href="./01_epsilon_greedy.md">
  <div class="link-card-title">01 Epsilon Greedy</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_epsilon_greedy.md">
  <div class="link-card-title">01 Epsilon Greedy — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_upper_confidence_bound.md">
  <div class="link-card-title">02 Upper Confidence Bound</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_upper_confidence_bound.md">
  <div class="link-card-title">02 Upper Confidence Bound — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_softmax_boltzmann.md">
  <div class="link-card-title">03 Softmax Boltzmann</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_softmax_boltzmann.md">
  <div class="link-card-title">03 Softmax Boltzmann — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


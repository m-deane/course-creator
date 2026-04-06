# Upper Confidence Bound (UCB1)

> **Reading time:** ~20 min | **Module:** 01 — Bandit Algorithms | **Prerequisites:** Module 0 Foundations


## In Brief
UCB1 selects the arm with the highest upper confidence bound: the estimated mean reward plus a bonus that decreases as you pull that arm more. It eliminates the need for tuning ε by using "optimism in the face of uncertainty"—always bet on arms that could plausibly be the best.

<div class="flow">
<div class="flow-step mint">1. Compute UCB Score</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Select Max UCB</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Pull Arm</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Observe Reward</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step rose">5. Update Statistics</div>
</div>


<div class="callout-key">

**Key Concept Summary:** This guide covers the core concepts of upper confidence bound (ucb1), with worked examples and practical implementation guidance.

</div>

## Key Insight
UCB1 solves the exploration-exploitation tradeoff automatically by adding a shrinking confidence bonus to each arm's value estimate. Arms you've pulled rarely get a large bonus (optimistic assumption), encouraging exploration. Arms you've pulled often have tight confidence bounds, so only truly good arms stay competitive.

<div class="callout-insight">

**Insight:** UCB algorithms are deterministic given the same history, which makes them easier to debug and reproduce than Thompson Sampling. This is a practical advantage in production systems where reproducibility matters.

</div>


## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
Reward Estimate with Confidence Bounds at t=100

Arm 1: μ̂=0.5, N=80 pulls
    |━━━━━●━━━━━|                       UCB = 0.5 + 0.05 = 0.55
    0.45      0.55

Arm 2: μ̂=0.6, N=15 pulls
    |━━━━━━━━━●━━━━━━━━━|               UCB = 0.6 + 0.22 = 0.82 ← SELECTED
    0.4               0.82

Arm 3: μ̂=0.7, N=5 pulls
    |━━━━━━━━━━━━━●━━━━━━━━━━━━|         UCB = 0.7 + 0.38 = 1.08
    0.3                      1.08

Arm 4: μ̂=0.3, N=50 pulls
    |━━━●━━━|                           UCB = 0.3 + 0.07 = 0.37
    0.23     0.37

Legend:
● = estimated mean μ̂
|━━━| = confidence interval
```

**Key observation:** Arm 3 has the highest mean (0.7) but hasn't been pulled enough. UCB gives it a huge bonus (0.38) making its upper bound 1.08—it could plausibly be the best arm, so we should explore it more.

**Confidence Bound Evolution:**

```
t=10:    Arm A: |━━━━━━━━━━━━━━━━━━━|  (wide, uncertain)
         Arm B: |━━━━━━━━━━━━━━━━━━━|

t=100:   Arm A: |━━━━━●━━━━━|          (narrower, more certain)
         Arm B: |━━━━━━━━━━━●━━━━━━━━|  (still wide, under-explored)

t=1000:  Arm A: |━━●━━|                 (tight, well-estimated)
         Arm B: |━━━━━━●━━━━━━|         (tighter, but still wider)
```

## Formal Definition

**Algorithm:** UCB1 (Upper Confidence Bound, version 1)

**Input:**
- K arms
- T time steps

**Initialization:**
- Pull each arm once (required to avoid division by zero)
- Q̂(a) = reward from initial pull
- N(a) = 1 for all arms

**At each time step t > K:**

1. **Select action:**
   ```
   a_t = argmax_a [Q̂(a) + c·√(ln(t) / N(a))]
   ```

   Where:
   - Q̂(a) = estimated mean reward of arm a
   - N(a) = number of times arm a has been pulled
   - t = total number of pulls so far
   - c = exploration constant (typically c=√2 for UCB1)

2. **Observe reward:** r_t ~ R(a_t)

3. **Update estimates:**
   ```
   N(a_t) ← N(a_t) + 1
   Q̂(a_t) ← Q̂(a_t) + (r_t - Q̂(a_t))/N(a_t)
   ```

**The UCB Bonus Term:**

```
U(a, t) = c·√(ln(t) / N(a))
```

This term has three key properties:

1. **Decreases with N(a):** More pulls → smaller bonus → less exploration
2. **Increases with t:** More total time → more budget to explore
3. **Logarithmic in t:** Exploration slows down over time (ln(t) grows slowly)

**Regret Bound:**

UCB1 achieves logarithmic regret:

```
E[R_T] ≤ Σ_{a: Δ_a > 0} (8 ln(T) / Δ_a) + (1 + π²/3)·Σ_a Δ_a
```

Where Δ_a = μ* - μ_a (gap between best arm and arm a).

**Simplified:** E[R_T] = O(√(KT ln T)) for the worst case, but often much better in practice.

**Key advantage over ε-greedy:** Regret is O(ln T) in the gap-dependent bound, versus O(T^(2/3)) for ε-greedy.

## Derivation Sketch: Why This Bonus?

UCB1 uses Hoeffding's inequality to bound how far the sample mean can deviate from the true mean:

```
P(|Q̂(a) - μ(a)| ≥ u) ≤ 2 exp(-2 N(a) u²)
```

We want this probability to be small, say p = 1/t² (decreasing over time). Solving for u:

```
2 exp(-2 N(a) u²) = 1/t²
-2 N(a) u² = ln(1/t²) - ln(2)
u² = (2 ln(t) + ln(2)) / (2 N(a))
u ≈ √(ln(t) / N(a))  [ignoring constants]
```

So with high probability, the true mean μ(a) is within √(ln(t)/N(a)) of Q̂(a). UCB uses the *upper* bound:

```
UCB(a) = Q̂(a) + √(ln(t) / N(a))
```

**Interpretation:** "Assume each arm is as good as it could plausibly be (upper bound), then pick the best."

## Intuitive Explanation

Think of UCB as a commodity trader with an optimistic bias:

**The Rule:**
For each commodity sector, estimate its potential as:
```
Potential = (historical average return) + (uncertainty bonus)
```

Where uncertainty bonus = "how wrong could I be, given how little I've traded this sector?"

**Example:**
- **Energy:** Traded 100 times, average return 0.5%/week, uncertainty ±0.1%
  - Potential = 0.5% + 0.1% = 0.6%

- **Agriculture:** Traded 10 times, average return 0.4%/week, uncertainty ±0.5%
  - Potential = 0.4% + 0.5% = 0.9% ← PICK THIS

**Why this works:**
- Agriculture *might* actually be better than energy—we're just not sure yet
- The only way to reduce uncertainty is to trade it more
- Eventually, after many pulls, the uncertainty shrinks and true performance dominates

**Why this beats ε-greedy:**
- No wasted random exploration—every pull is an informed decision
- Automatically balances: high uncertainty → explore, low uncertainty → exploit
- No hyperparameter to tune (c=√2 is standard)

**When it fails:**
- Assumes rewards are bounded (e.g., in [0, 1])—heavy-tailed distributions break the theory
- Cold start is slow (must pull each arm once initially)
- Non-stationary rewards (best arm changes over time) → UCB keeps pulling the old winner

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

class UCB1:
    def __init__(self, k_arms, c=np.sqrt(2)):
        self.k = k_arms
        self.c = c
        self.q_estimates = np.zeros(k_arms)
        self.action_counts = np.zeros(k_arms)
        self.t = 0

    def select_action(self):
        self.t += 1

        # First K pulls: try each arm once
        if self.t <= self.k:
            return self.t - 1

        # Compute UCB for each arm
        ucb_values = self.q_estimates + self.c * np.sqrt(
            np.log(self.t) / (self.action_counts + 1e-10)
        )
        return np.argmax(ucb_values)

    def update(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_estimates[action] += (reward - self.q_estimates[action]) / n
```

</div>
</div>

**Usage:**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
ucb = UCB1(k_arms=5)
for t in range(1000):
    action = ucb.select_action()
    reward = get_reward(action)
    ucb.update(action, reward)

print(f"Final estimates: {ucb.q_estimates}")
print(f"Pull counts: {ucb.action_counts}")
```

</div>
</div>

**Variant: Different exploration constant:**
```python
# Conservative (less exploration)
ucb = UCB1(k_arms=5, c=1.0)

# Aggressive (more exploration)
ucb = UCB1(k_arms=5, c=2.0)

# Standard (theory-optimal)
ucb = UCB1(k_arms=5, c=np.sqrt(2))
```

## Common Pitfalls

### 1. Division by Zero
**Problem:** If N(a) = 0, the UCB formula has √(ln(t)/0) = ∞.

**Fix:** Initialize by pulling each arm once, or add a small constant:
```python
ucb_values = q + c * np.sqrt(np.log(t) / (counts + 1e-10))
```

### 2. Unbounded Rewards
**Problem:** UCB1 theory assumes rewards are bounded (e.g., in [0, 1]). If rewards can be arbitrarily large, Hoeffding's inequality doesn't apply.

**Symptom:** A single large reward gives an arm a huge Q̂ that never shrinks, even if subsequent rewards are poor.

**Fix:**
- Normalize rewards to [0, 1]: `r_norm = (r - r_min) / (r_max - r_min)`
- Use UCB-V (variance-aware UCB) for unbounded rewards
- Clip extreme outliers

### 3. Wrong Time Index
**Problem:** Using N(a) instead of t in the logarithm:

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
# WRONG
ucb = q + c * np.sqrt(np.log(counts) / counts)

# CORRECT
ucb = q + c * np.sqrt(np.log(t) / counts)
```

</div>
</div>

**Why it matters:** The ln(t) term ensures the bonus grows (slowly) with total time, encouraging continuous exploration. Using ln(N(a)) would make the bonus shrink too fast.

### 4. Not Breaking Ties
**Problem:** If two arms have identical UCB values, `argmax` picks the first (index bias).

**Fix:**
```python
max_ucb = np.max(ucb_values)
best_actions = np.where(ucb_values == max_ucb)[0]
return np.random.choice(best_actions)
```

### 5. Applying to Non-Stationary Problems
**Problem:** If the reward distribution changes over time (e.g., commodity returns shift after a market regime change), UCB keeps trusting old estimates.

**Symptom:** Algorithm locks onto an arm that was best historically but is now suboptimal.

**Fix:** Use discounted UCB or a sliding window:
```python
# Exponential recency weighting (α=0.1)
q_new = (1-α)*q_old + α*reward  # Instead of sample mean

# Sliding window (last 100 pulls)
q = np.mean(recent_rewards[-100:])
```

But note: This breaks UCB1's theoretical guarantees.

## Comparison to Epsilon-Greedy

| Aspect | Epsilon-Greedy | UCB1 |
|--------|----------------|------|
| **Hyperparameters** | Requires tuning ε | Parameter-free (c=√2 standard) |
| **Exploration** | Random (wastes pulls) | Directed (pulls informative arms) |
| **Regret** | O(T^(2/3)) optimal | O(ln T) gap-dependent, O(√T) worst-case |
| **Assumptions** | None (works anywhere) | Bounded rewards, stationarity |
| **Implementation** | Simpler (5 lines) | Slightly more complex (10 lines) |
| **Non-stationary** | Handles well (with fixed ε) | Fails (trusts old data) |
| **Cold start** | Explores immediately | Must pull each arm once first |

**When UCB wins:**
- Stationary environments (reward distributions don't change)
- Bounded rewards (e.g., success/failure, ratings in [0, 5])
- Long time horizons (logarithmic regret dominates)
- You don't want to tune hyperparameters

**When ε-greedy wins:**
- Non-stationary environments (regime changes)
- Unbounded or heavy-tailed rewards
- Short time horizons (simplicity matters)
- You have domain knowledge to set ε

**Empirical observation:** UCB1 often beats ε-greedy by 2-5× in total regret on stationary problems.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Builds On
- **Hoeffding's Inequality:** Concentration bound for sample means
- **Confidence Intervals:** Statistical uncertainty quantification
- **Optimism Under Uncertainty:** Exploration principle from reinforcement learning

### Leads To
- **UCB-V:** Variance-aware UCB for unbounded rewards
- **Bayes-UCB:** Bayesian confidence bounds
- **Thompson Sampling:** Posterior sampling alternative
- **Contextual UCB:** LinUCB for linear reward models

### Related Concepts
- **Regret Decomposition:** Breaking regret into bias and variance terms
- **Upper Confidence Trees (UCT):** UCB for game tree search (AlphaGo)
- **Gittins Index:** Optimal solution for discounted infinite-horizon bandits

## Practice Problems

### 1. Conceptual Understanding
**Q:** Why does the UCB bonus include ln(t) instead of just t?

**A:** Logarithmic growth balances exploration and exploitation. If the bonus were √(t/N), it would grow without bound, forcing infinite exploration. If it were constant, exploration would stop too early. ln(t) grows slowly enough that exploitation dominates asymptotically, but fast enough to ensure sufficient exploration.

**Q:** You run UCB1 for 1000 steps. Arm 1 has been pulled 800 times, arm 2 has been pulled 200 times. What does this tell you?

**A:** Arm 1 has consistently had higher UCB values, meaning it's either:
- Clearly the best arm (high Q̂₁, narrow confidence bound still beats others)
- Just lucky early (but 800 pulls is usually enough to overcome luck)

Arm 2's UCB = Q̂₂ + √(ln(1000)/200) ≈ Q̂₂ + 0.18. For UCB1 to keep choosing arm 1, we likely have Q̂₁ > Q̂₂ + 0.18.

### 2. Implementation Challenge
Modify UCB1 to use different confidence levels for different arms. For example, a "high-stakes" arm (where mistakes are costly) should require higher confidence before exploitation.

```python
class RiskAwareUCB(UCB1):
    def __init__(self, k_arms, risk_levels):
        super().__init__(k_arms)
        self.risk_levels = risk_levels  # Array of c values per arm

    def select_action(self):
        self.t += 1
        if self.t <= self.k:
            return self.t - 1

        ucb_values = self.q_estimates + self.risk_levels * np.sqrt(
            np.log(self.t) / (self.action_counts + 1e-10)
        )
        return np.argmax(ucb_values)
```

### 3. Real-World Scenario
You're hedging a natural gas position using 4 instruments: futures, options, swaps, basis swaps. Each has different reward distributions:
- Futures: narrow range, well-understood
- Options: wide range, non-linear payoff
- Swaps: medium range, counterparty risk
- Basis swaps: narrow range, illiquid

How would you modify UCB1 to account for these differences?

**Suggested approach:**
1. Normalize all rewards to [0, 1] using historical min/max
2. Use different c values based on known variance:
   - Futures: c=1.0 (low uncertainty)
   - Options: c=2.0 (high uncertainty)
   - Swaps: c=1.5 (medium uncertainty)
   - Basis: c=1.2 (narrow but illiquid → slightly higher)

### 4. Debugging Exercise
Your UCB1 implementation keeps pulling the same arm forever after t=100. What could be wrong?

**Diagnostic steps:**
1. Print `ucb_values` at each step—is one arm always highest?
2. Check if ln(t) is being computed correctly (not ln(N(a)))
3. Verify rewards are being updated (not stuck at initial values)
4. Check for numerical overflow if t is very large

**Common bug:**
```python
# WRONG: bonus doesn't grow with time
ucb = q + np.sqrt(2 / counts)

# CORRECT
ucb = q + np.sqrt(2 * np.log(t) / counts)
```

### 5. Extension
Implement UCB-Tuned, which uses variance estimates to shrink confidence bounds:

```
UCB-Tuned: a_t = argmax_a [Q̂(a) + √(ln(t)/N(a) · min(1/4, V_a(t)))]

where V_a(t) = (1/N_a)·Σ(r² - Q̂²) + √(2ln(t)/N_a)
```

This exploits the fact that arms with low variance need smaller confidence bounds.

**Hint:** Track sum of squared rewards for each arm to compute variance efficiently.


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

<a class="link-card" href="./03_softmax_boltzmann.md">
  <div class="link-card-title">03 Softmax Boltzmann</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_softmax_boltzmann.md">
  <div class="link-card-title">03 Softmax Boltzmann — Companion Slides</div>
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


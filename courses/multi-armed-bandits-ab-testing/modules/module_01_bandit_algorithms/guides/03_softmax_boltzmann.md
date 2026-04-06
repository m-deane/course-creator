# Softmax (Boltzmann) Exploration

> **Reading time:** ~20 min | **Module:** 01 — Bandit Algorithms | **Prerequisites:** Module 0 Foundations


## In Brief
Softmax exploration selects arms probabilistically, with probabilities proportional to their estimated values. Instead of the hard explore-exploit switch of ε-greedy, softmax smoothly allocates more pulls to better arms while still giving worse arms a chance. The temperature parameter τ controls how "sharp" the distribution is.


<div class="callout-key">

**Key Concept Summary:** Notice: Softmax gives more probability to arm 4 (Q̂=0.4) than arm 2 (Q̂=0.3),

</div>

## Key Insight
Softmax solves a key limitation of ε-greedy: with ε-greedy, a terrible arm and a mediocre arm are equally likely to be explored. Softmax is smarter—it explores in proportion to how promising each arm looks. The temperature τ plays the same role as ε, but with a smoother effect.

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
Q̂ values: [0.2, 0.5, 0.7, 0.3, 0.1]

Temperature τ = 0.1 (cold, exploitation):
P(a) = [0.00, 0.07, 0.93, 0.00, 0.00]
        │     │     ▓▓▓▓▓▓▓▓▓│    │
        └─────┴─────────────┴────┘
        Almost always pick arm 3

Temperature τ = 0.5 (balanced):
P(a) = [0.05, 0.23, 0.51, 0.12, 0.03]
        ▓░    ▓▓▓░  ▓▓▓▓▓░  ▓▓░   ▓
        Good arms favored, bad arms rare

Temperature τ = 2.0 (hot, exploration):
P(a) = [0.16, 0.21, 0.25, 0.19, 0.14]
        ▓▓▓   ▓▓▓▓  ▓▓▓▓▓  ▓▓▓▓  ▓▓▓
        Nearly uniform (random)

Temperature τ → 0 (coldest):
P(a) = [0, 0, 1, 0, 0]  (pure exploitation, argmax)

Temperature τ → ∞ (hottest):
P(a) = [0.2, 0.2, 0.2, 0.2, 0.2]  (uniform random)
```

**Softmax vs Epsilon-Greedy Selection:**

```
5 arms with Q̂ = [0.1, 0.3, 0.8, 0.4, 0.2]

Epsilon-greedy (ε=0.2):
  With prob 0.2: pick uniformly → P = [0.04, 0.04, 0.04, 0.04, 0.04]
  With prob 0.8: pick best     → P = [0,    0,    0.8,  0,    0   ]
  Total:                         P = [0.04, 0.04, 0.84, 0.04, 0.04]

Softmax (τ=0.3):
  P(a) ∝ exp(Q̂(a)/τ)
  P = [0.01, 0.08, 0.80, 0.10, 0.02]

Notice: Softmax gives more probability to arm 4 (Q̂=0.4) than arm 2 (Q̂=0.3),
while ε-greedy treats them equally during exploration.
```

## Formal Definition

**Algorithm:** Softmax (Boltzmann) Exploration

**Input:**
- K arms
- τ > 0 (temperature parameter)

**Initialization:**
- Q̂(a) ← 0 for all arms
- N(a) ← 0 for all arms

**At each time step t:**

1. **Compute selection probabilities:**
   ```
   π(a) = exp(Q̂(a) / τ) / Σ_{a'=1}^K exp(Q̂(a') / τ)
   ```

   Equivalently (more stable numerically):
   ```
   Let Q_max = max_a Q̂(a)
   π(a) = exp((Q̂(a) - Q_max) / τ) / Σ_{a'} exp((Q̂(a') - Q_max) / τ)
   ```

2. **Select action:** Sample a ~ π

3. **Observe reward:** r ~ R(a)

4. **Update estimates:**
   ```
   N(a) ← N(a) + 1
   Q̂(a) ← Q̂(a) + (r - Q̂(a)) / N(a)
   ```

**Temperature Parameter τ:**

- **τ → 0:** Probability concentrates on argmax (exploitation)
  - π(a*) → 1 where a* = argmax Q̂(a)
  - π(a) → 0 for a ≠ a*

- **τ → ∞:** Probability becomes uniform (exploration)
  - π(a) → 1/K for all a

- **τ = 1:** Balanced (common default)

**Decaying Temperature:**

Like decaying ε for ε-greedy, you can decay τ:

```
τ(t) = τ_0 / log(t + 2)
```

This ensures:
- Early: τ high → explore broadly
- Late: τ low → exploit best arm

**Regret:**

Softmax with optimal temperature schedule achieves:
```
E[R_T] = O(T^(2/3))
```

Same asymptotic regret as ε-greedy (not as good as UCB1's O(ln T)), but often better constant factors in practice.

## Intuitive Explanation

Think of softmax as a commodity trader who allocates capital proportionally to expected performance:

**The Rule:**
Instead of "exploit best or explore randomly," allocate capital in proportion to how good each sector looks, with some "noise" added.

**Example (5 sectors, $100K to allocate):**

Estimated weekly returns: [0.1%, 0.3%, 0.8%, 0.4%, 0.2%]

With τ = 0.5:
```
Energy:       exp(0.1/0.5) / Z = $1K   (1%)
Metals:       exp(0.3/0.5) / Z = $8K   (8%)
Agriculture:  exp(0.8/0.5) / Z = $80K  (80%)
Livestock:    exp(0.4/0.5) / Z = $10K  (10%)
Softs:        exp(0.2/0.5) / Z = $2K   (2%)
```

**Why this is better than ε-greedy:**
- You're not wasting 10% on purely random bets
- Good-but-not-best sectors (Livestock, 0.4%) get more capital than terrible sectors (Energy, 0.1%)
- Still exploring everything (all probabilities > 0)

**Temperature as risk tolerance:**
- **High τ (τ=2):** "I'm uncertain, so I'll spread capital broadly" → diversification
- **Low τ (τ=0.1):** "I'm confident in my estimates" → concentration
- **τ→0:** "I know the best sector for certain" → go all-in

**When it fails:**
- If Q̂ estimates are very wrong early, softmax can get stuck exploiting a bad arm
- Temperature tuning is as hard as tuning ε for ε-greedy
- Numerical instability if Q̂ values are very large (fix: subtract max before exp)

## Code Implementation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np

class SoftmaxBandit:
    def __init__(self, k_arms, tau=1.0):
        self.k = k_arms
        self.tau = tau
        self.q_estimates = np.zeros(k_arms)
        self.action_counts = np.zeros(k_arms)

    def select_action(self):
        # Numerical stability: subtract max before exp
        q_max = np.max(self.q_estimates)
        exp_values = np.exp((self.q_estimates - q_max) / self.tau)
        probs = exp_values / np.sum(exp_values)
        return np.random.choice(self.k, p=probs)

    def update(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_estimates[action] += (reward - self.q_estimates[action]) / n

    def get_probabilities(self):
        """Return current selection probabilities (for visualization)"""
        q_max = np.max(self.q_estimates)
        exp_values = np.exp((self.q_estimates - q_max) / self.tau)
        return exp_values / np.sum(exp_values)
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
bandit = SoftmaxBandit(k_arms=5, tau=0.5)

for t in range(1000):
    action = bandit.select_action()
    reward = get_reward(action)
    bandit.update(action, reward)

# Visualize final selection probabilities
probs = bandit.get_probabilities()
print(f"Q̂: {bandit.q_estimates}")
print(f"P: {probs}")
```

</div>
</div>

**Decaying Temperature:**
```python
class DecayingSoftmax(SoftmaxBandit):
    def __init__(self, k_arms, tau_fn=lambda t: 1.0 / np.log(t + 2)):
        super().__init__(k_arms, tau=1.0)
        self.tau_fn = tau_fn
        self.t = 0

    def select_action(self):
        self.tau = self.tau_fn(self.t)
        self.t += 1
        return super().select_action()
```

**Preference-based variant (add learning to preferences):**

<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
class GradientBandit(SoftmaxBandit):
    """Softmax with learned preferences instead of Q̂ values"""
    def __init__(self, k_arms, alpha=0.1):
        super().__init__(k_arms, tau=1.0)
        self.preferences = np.zeros(k_arms)  # H(a) in Sutton & Barto
        self.alpha = alpha  # Learning rate
        self.avg_reward = 0

    def select_action(self):
        # Use preferences instead of Q̂
        h_max = np.max(self.preferences)
        exp_h = np.exp(self.preferences - h_max)
        self.probs = exp_h / np.sum(exp_h)
        return np.random.choice(self.k, p=self.probs)

    def update(self, action, reward):
        self.action_counts[action] += 1

        # Update average reward
        self.avg_reward += (reward - self.avg_reward) / self.t

        # Update preferences (gradient ascent)
        for a in range(self.k):
            if a == action:
                self.preferences[a] += self.alpha * (reward - self.avg_reward) * (1 - self.probs[a])
            else:
                self.preferences[a] -= self.alpha * (reward - self.avg_reward) * self.probs[a]
```

</div>
</div>

## Common Pitfalls

### 1. Numerical Overflow
**Problem:** exp(Q̂/τ) can overflow if Q̂ values are large or τ is very small.

**Example:** If Q̂ = 100 and τ = 0.1, then exp(100/0.1) = exp(1000) = overflow.

**Fix:** Subtract the maximum Q̂ before exponentiating (log-sum-exp trick):
```python
# WRONG (can overflow)
probs = np.exp(q / tau) / np.sum(np.exp(q / tau))

# CORRECT (numerically stable)
q_max = np.max(q)
exp_q = np.exp((q - q_max) / tau)
probs = exp_q / np.sum(exp_q)
```

**Why it works:** exp(q/τ) / Σexp(q'/τ) = exp((q-max)/τ) / Σexp((q'-max)/τ) (constant cancels).

### 2. Wrong Temperature Scale
**Problem:** Using τ=1 when Q̂ values are in [0, 1000] instead of [0, 1].

**Symptom:** Selection probabilities are almost uniform (temperature is too high relative to Q̂ scale).

**Fix:** Normalize Q̂ values to a consistent range, or scale τ appropriately:
```python
# Normalize Q̂ to [0, 1]
q_norm = (q - q.min()) / (q.max() - q.min() + 1e-10)
probs = softmax(q_norm / tau)
```

### 3. Temperature Too Low → Premature Convergence
**Problem:** τ=0.01 makes softmax almost deterministic (like argmax).

**Symptom:** One arm gets >99% of pulls, others are never explored.

**Fix:**
- Start with τ ≥ 1 for most problems
- Use decaying τ: τ(t) = max(0.1, 1 / log(t + 2))

### 4. Temperature Too High → Random Search
**Problem:** τ=10 when Q̂ values are in [0, 1] makes all probabilities nearly equal.

**Symptom:** Arm selection frequencies are uniform, regardless of Q̂ values.

**Fix:**
- Reduce τ to 0.1 - 1.0 range
- Check that exp((Q̂_max - Q̂_min)/τ) is not too close to 1

**Diagnostic:**
```python
q_range = q.max() - q.min()
effective_scale = q_range / tau
print(f"Effective scale: {effective_scale:.2f}")
# Should be in range [1, 5] for good exploration-exploitation balance
# < 1: too much exploration (random)
# > 5: too much exploitation (deterministic)
```

### 5. Forgetting to Update Q̂
**Problem:** Computing probabilities from preferences H(a) but updating Q̂(a) instead (mixing gradient bandit and softmax).

**Symptom:** Selection probabilities don't change over time, even though rewards vary.

**Fix:** Be clear about whether you're using:
- **Value-based softmax:** π(a) ∝ exp(Q̂(a)/τ), update Q̂ with sample mean
- **Preference-based (gradient bandit):** π(a) ∝ exp(H(a)), update H with gradient ascent

Don't mix the two.

## When to Prefer Softmax Over ε-Greedy or UCB

### Softmax Wins When:

1. **You want smooth exploration**
   - ε-greedy has a hard switch (explore vs exploit)
   - Softmax gradually shifts probability as confidence grows

2. **You have many arms with similar values**
   - ε-greedy wastes exploration on bad arms equally
   - Softmax focuses on promising-but-uncertain arms

3. **You want interpretable selection probabilities**
   - Can visualize π(a) as a "portfolio allocation"
   - Useful for risk management (never go 100% into one asset)

4. **Your reward scale is consistent**
   - Softmax works well when Q̂ ∈ [0, 1] or similar bounded range
   - Temperature τ has clear interpretation

### Softmax Loses When:

1. **You want theoretical guarantees**
   - UCB1 has better regret bounds (O(ln T) vs O(T^(2/3)))
   - ε-greedy is simpler and also achieves O(T^(2/3))

2. **You have unbounded or heavy-tailed rewards**
   - One outlier can dominate exp(Q̂/τ)
   - UCB and ε-greedy are more robust

3. **You don't want to tune hyperparameters**
   - τ tuning is as hard as ε tuning
   - UCB1 is parameter-free

4. **You need fast cold-start**
   - Softmax can get stuck exploiting the first arm that looks good
   - UCB's confidence bounds force exploration of all arms

## Comparison Table

| Feature | ε-Greedy | UCB1 | Softmax |
|---------|----------|------|---------|
| **Selection** | Discrete (exploit or random) | Deterministic (max UCB) | Smooth (probabilistic) |
| **Exploration** | Uniform random | Confidence-based | Value-proportional |
| **Hyperparameters** | ε (easy to interpret) | c (usually fixed at √2) | τ (sensitive to scale) |
| **Regret** | O(T^(2/3)) | O(ln T) gap-dependent | O(T^(2/3)) |
| **Assumptions** | None | Bounded rewards | Consistent reward scale |
| **Interpretability** | Simple | Confidence intervals | Selection probabilities |
| **Best for** | Non-stationary, simple | Stationary, bounded | Portfolio allocation, smooth |

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Builds On
- **Boltzmann Distribution:** From statistical mechanics, probability ∝ exp(-E/kT)
- **Logistic Regression:** Softmax is the multi-class generalization of sigmoid
- **Gibbs Sampling:** MCMC method using Boltzmann distribution

### Leads To
- **Policy Gradient Methods:** Softmax policies in reinforcement learning (REINFORCE, PPO)
- **Gradient Bandit Algorithm:** Learn preferences H(a) instead of values Q̂(a)
- **Contextual Softmax:** Extend to contextual bandits with feature vectors

### Related Concepts
- **Temperature Annealing:** Gradually reduce τ over time (simulated annealing)
- **Cross-Entropy Loss:** Related to softmax probability updates
- **Multinomial Logit Model:** Discrete choice theory in economics

## Practice Problems

### 1. Conceptual Understanding
**Q:** You run softmax with τ=1 on 3 arms with Q̂ = [0.1, 0.5, 0.9]. What are the selection probabilities?

**A:**
```
π(a) = exp(Q̂(a)/τ) / Σ exp(Q̂(a')/τ)
     = exp(Q̂(a)) / Σ exp(Q̂(a'))

exp(0.1) = 1.11, exp(0.5) = 1.65, exp(0.9) = 2.46
Sum = 5.22

π = [1.11/5.22, 1.65/5.22, 2.46/5.22]
  = [0.21, 0.32, 0.47]
```

Best arm gets 47% of pulls, second-best 32%, worst 21%.

**Q:** What happens to these probabilities if you set τ=0.1?

**A:**
```
exp(0.1/0.1) = exp(1) = 2.72
exp(0.5/0.1) = exp(5) = 148.4
exp(0.9/0.1) = exp(9) = 8103
Sum ≈ 8254

π ≈ [0.0003, 0.018, 0.982]
```

Best arm gets 98% of pulls—nearly deterministic exploitation.

### 2. Implementation Challenge
Implement a "contextual softmax" where temperature depends on the variance of Q̂ values:
- High variance → increase τ (more exploration needed)
- Low variance → decrease τ (confident in estimates)

```python
class AdaptiveSoftmax(SoftmaxBandit):
    def select_action(self):
        q_var = np.var(self.q_estimates)
        self.tau = 0.1 + q_var  # τ ∈ [0.1, ∞)
        return super().select_action()
```

Test this on a multi-armed bandit problem. Does it beat fixed-τ softmax?

### 3. Real-World Scenario
You're allocating $1M across 10 commodity futures. You want to:
- Never allocate 0% to any commodity (maintain liquidity)
- Favor commodities with higher estimated returns
- Smoothly adjust allocations as estimates update

How would you use softmax for this?

**Suggested approach:**
```python
# Convert Q̂ to portfolio weights
def allocate_capital(q_estimates, tau=0.5, total_capital=1e6):
    # Softmax probabilities
    q_max = np.max(q_estimates)
    exp_q = np.exp((q_estimates - q_max) / tau)
    weights = exp_q / np.sum(exp_q)

    # Convert to dollar allocations
    allocations = weights * total_capital

    # Ensure minimum $10K per commodity
    min_allocation = 10000
    allocations = np.maximum(allocations, min_allocation)
    allocations *= total_capital / np.sum(allocations)  # Renormalize

    return allocations
```

### 4. Debugging Exercise
Your softmax implementation always selects the first arm, even though Q̂ values vary widely. What could be wrong?

**Diagnostic checklist:**
```python
print(f"Q̂: {q_estimates}")
print(f"τ: {tau}")
print(f"exp(Q̂/τ): {np.exp(q_estimates / tau)}")
print(f"Probabilities: {probs}")
```

**Common bugs:**
- τ is too large (e.g., τ=1000) → all probs ≈ 1/K
- Forgot to subtract max before exp → overflow → NaN
- Using argmax instead of sampling: `return np.argmax(probs)` instead of `np.random.choice(k, p=probs)`

### 5. Extension
Implement "optimistic softmax" that adds a bonus to Q̂ before computing probabilities:

```
π(a) ∝ exp((Q̂(a) + bonus(a)) / τ)

where bonus(a) = c · √(ln(t) / N(a))  (UCB-style bonus)
```

This combines UCB's exploration bonus with softmax's smooth selection. Compare to pure UCB and pure softmax.

**Hypothesis:** Optimistic softmax should interpolate between UCB (when τ→0) and softmax (when c→0).


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

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


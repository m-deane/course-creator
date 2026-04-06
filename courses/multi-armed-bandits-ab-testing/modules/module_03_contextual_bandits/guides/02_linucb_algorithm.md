# LinUCB Algorithm

> **Reading time:** ~20 min | **Module:** 03 — Contextual Bandits | **Prerequisites:** Module 2


## In Brief

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

**Key Concept Summary:** LinUCB (Linear Upper Confidence Bound) is a contextual bandit algorithm that assumes rewards are linear in features and uses ridge regression with confidence bounds for exploration. It's the UCB1 a...

</div>

LinUCB (Linear Upper Confidence Bound) is a contextual bandit algorithm that assumes rewards are linear in features and uses ridge regression with confidence bounds for exploration. It's the UCB1 algorithm extended to contextual settings with principled uncertainty quantification.

## Key Insight

LinUCB combines two powerful ideas:
1. **Linear model:** Assume E[r | x, a] = x^T θ_a (reward is linear in context features)
2. **Confidence bounds:** Use ridge regression uncertainty to compute UCB scores

<div class="callout-insight">

**Insight:** UCB algorithms are deterministic given the same history, which makes them easier to debug and reproduce than Thompson Sampling. This is a practical advantage in production systems where reproducibility matters.

</div>


Instead of tracking simple reward averages like UCB1, LinUCB maintains a regression model per arm. The "confidence bound" comes from the regression's prediction uncertainty — contexts far from previous observations get wider confidence intervals, encouraging exploration.

## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
LINUCB DECISION PROCESS:

Round t:
  1. Observe context x_t = [volatility, term_spread, seasonality]

  2. For each arm a, compute:
     ┌────────────────────────────────────────┐
     │ UCB_a(x_t) = x_t^T θ̂_a + α·σ_a(x_t)  │
     │              ⎯⎯⎯⎯⎯⎯⎯⎯   ⎯⎯⎯⎯⎯⎯⎯⎯⎯  │
     │              predicted  uncertainty   │
     │              reward     bonus         │
     └────────────────────────────────────────┘

  3. Choose arm with highest UCB:
     a_t = argmax_a UCB_a(x_t)

  4. Observe reward r_t

  5. Update ridge regression for arm a_t:
     θ̂_a ← (X_a^T X_a + λI)^{-1} X_a^T r_a


CONFIDENCE ELLIPSOID (shrinks with data):

    Reward
      │     ╱╲        Initial: Wide uncertainty
      │    ╱  ╲
      │   ╱    ╲
      │  ╱      ╲
      │ ╱        ╲
      │╱__________╲___Context

    Reward
      │    ╱╲         After 100 rounds: Narrow
      │   ╱  ╲
      │  ╱____╲
      │╱        ╲_____Context
```

## Formal Definition

### Problem Setup
- Context space: x ∈ ℝ^d
- Arms: a ∈ {1, ..., K}
- Assumption: E[r | x, a] = x^T θ_a^* + ε, where ε is zero-mean noise

### Algorithm Parameters
- α > 0: Exploration parameter (controls confidence bound width)
- λ > 0: Ridge regression regularization parameter

### LinUCB Algorithm

**Initialize:** For each arm a:
- A_a = λI (d × d identity matrix scaled by λ)
- b_a = 0 (d-dimensional vector)

**At each round t:**

1. **Observe context** x_t ∈ ℝ^d

2. **Compute UCB for each arm:**
   ```
   θ̂_a = A_a^{-1} b_a
   σ_a(x_t) = √(x_t^T A_a^{-1} x_t)
   UCB_a = x_t^T θ̂_a + α · σ_a(x_t)
   ```

3. **Choose arm:** a_t = argmax_a UCB_a

4. **Observe reward:** r_t

5. **Update chosen arm's statistics:**
   ```
   A_{a_t} ← A_{a_t} + x_t x_t^T
   b_{a_t} ← b_{a_t} + r_t x_t
   ```

### Why This Works

**Ridge regression solution:**
```
θ̂_a = argmin_θ [Σ(r_i - x_i^T θ)² + λ||θ||²]
    = (X_a^T X_a + λI)^{-1} X_a^T r_a
    = A_a^{-1} b_a
```

**Uncertainty quantification:**
The standard error of prediction at x is:
```
σ_a(x) = √(x^T (X_a^T X_a + λI)^{-1} x)
       = √(x^T A_a^{-1} x)
```

Larger σ_a(x) → less data in that region → higher exploration bonus.

### Regret Bound

With proper choice of α, LinUCB achieves regret:
```
Regret(T) = O(d√(T log T))
```

where d is context dimension. This is optimal up to logarithmic factors.

## Intuitive Explanation

**It's like a GPS that learns road speeds:**

Imagine a GPS learning traffic patterns:
- **Context:** Time of day, day of week, weather
- **Arms:** Different route options
- **Reward:** Negative travel time

LinUCB learns: "Route A is fast during rush hour (context), Route B is fast at night." The confidence bounds ensure it occasionally tries Route C in untested conditions (e.g., Sunday morning) to discover if it might be faster.

**In commodity trading:**
- **Context:** [VIX, term_spread, inventory_surprise]
- **Arms:** [Overweight Energy, Overweight Metals, Overweight Ag]
- **Reward:** Next-period return

LinUCB learns the linear relationship: "Energy performs well when VIX is low and term spread is positive." The confidence bounds ensure it explores Metals during unusual VIX/spread combinations to verify its model.

## Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np

class LinUCB:
    """Linear Upper Confidence Bound for contextual bandits."""

    def __init__(self, n_arms, context_dim, alpha=1.0, lambda_=1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        # Initialize for each arm
        self.A = [lambda_ * np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def choose_arm(self, context):
        """Choose arm with highest UCB score."""
        ucb_scores = []
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            pred = context @ theta
            std = np.sqrt(context @ A_inv @ context)
            ucb = pred + self.alpha * std
            ucb_scores.append(ucb)
        return np.argmax(ucb_scores)

    def update(self, arm, context, reward):
        """Update ridge regression for chosen arm."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

</div>
</div>

## Common Pitfalls

### 1. **Wrong alpha parameter**
- **Too small (α → 0):** Pure exploitation, insufficient exploration → suboptimal regret
- **Too large (α → ∞):** Pure exploration, ignores learned information → linear regret
- **Typical values:** α ∈ [0.1, 2.0]; start with α = 1.0
- **Tuning:** Cross-validation or bandit-over-bandits (meta-learning)

### 2. **Numerical instability with matrix inversion**
- **Problem:** A_a becomes ill-conditioned after many updates
- **Symptom:** `np.linalg.LinAlgError` or wildly incorrect predictions
- **Fix:** Use `np.linalg.solve(A, b)` instead of `inv(A) @ b`, or Sherman-Morrison formula for online updates
- **Better:**
  ```python
  theta = np.linalg.solve(self.A[a], self.b[a])
  ```

### 3. **Forgetting to regularize (λ = 0)**
- **Problem:** Without regularization, A_a is singular when data < context_dim
- **Symptom:** Matrix inversion fails in early rounds
- **Fix:** Always use λ > 0 (typical: λ = 1.0)

### 4. **Feature scale issues**
- **Problem:** Features with large magnitudes dominate the regression
- **Example:** Price (range 50-100) vs. normalized VIX (range 0-1)
- **Fix:** Standardize features before LinUCB: `x = (x - mean) / std`

### 5. **Using LinUCB when rewards aren't linear**
- **Problem:** If E[r | x, a] is highly nonlinear, LinUCB underperforms
- **Example:** Regime-switching models where relationship flips at threshold
- **Fix:** Use kernel LinUCB, neural bandits, or tree-based contextual bandits

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Builds On
- **Module 1 (UCB1):** LinUCB extends UCB1 to contextual setting with linear models
- **Ridge regression:** The core prediction engine
- **Bayesian interpretation:** Can derive LinUCB from Bayesian linear regression with Gaussian priors

### Leads To
- **Module 5 (Commodity Trading):** LinUCB for regime-aware portfolio allocation
- **Neural contextual bandits:** Replace linear model with neural network
- **Kernel LinUCB:** Use kernel trick for nonlinear reward functions
- **Thompson Sampling for linear bandits:** Bayesian alternative to LinUCB

### Variants and Extensions

**1. Disjoint LinUCB:**
Each arm has independent θ_a (what we implemented above).

**2. Hybrid LinUCB:**
```
E[r | x, a] = x^T θ_a + z^T β
```
where z are arm-specific features (e.g., commodity sector attributes) and x are context features.

**3. Kernelized LinUCB:**
Replace x^T θ with kernel function k(x, x') for nonlinear relationships.

**4. Neural LinUCB:**
Use neural network for feature representation, LinUCB on final layer embeddings.

## Practice Problems

### 1. Parameter Selection
**Question:** You have 1000 observations and 5 context features. What's a reasonable starting point for α and λ?

**Answer:**
- λ = 1.0 (standard regularization)
- α = 1.0 (balanced exploration)
- Then tune based on offline evaluation or online performance

### 2. Computational Complexity
**Question:** What's the time complexity of choosing an arm in LinUCB with K arms and d-dimensional context?

**Answer:** O(K · d³) due to matrix inversion for each arm. With Sherman-Morrison updates, this can be reduced to O(K · d²).

### 3. Implementation Exercise
**Task:** Add a method to compute prediction uncertainty for a given context and arm.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def get_uncertainty(self, arm, context):
    """Return prediction uncertainty for given arm and context."""
    A_inv = np.linalg.inv(self.A[arm])
    return np.sqrt(context @ A_inv @ context)
```


### 4. Diagnostic Question
**Scenario:** Your LinUCB always chooses arm 0, even after 1000 rounds with varying contexts. What might be wrong?

**Possible causes:**
- α too small → pure exploitation, stuck on first good arm
- Features not standardized → one feature dominates
- Reward signal too noisy → can't distinguish arms
- Bug in context generation → all contexts are identical

**Debug:** Print UCB scores for all arms, check feature variance, verify reward distributions.


---

## Cross-References

<a class="link-card" href="./01_contextual_bandit_framework.md">
  <div class="link-card-title">01 Contextual Bandit Framework</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_contextual_bandit_framework.md">
  <div class="link-card-title">01 Contextual Bandit Framework — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_feature_engineering_bandits.md">
  <div class="link-card-title">03 Feature Engineering Bandits</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_feature_engineering_bandits.md">
  <div class="link-card-title">03 Feature Engineering Bandits — Companion Slides</div>
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


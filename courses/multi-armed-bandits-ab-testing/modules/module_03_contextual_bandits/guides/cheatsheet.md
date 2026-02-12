# Contextual Bandits Cheatsheet

## Quick Reference

### Core Concepts

| Concept | Standard Bandit | Contextual Bandit |
|---------|----------------|-------------------|
| **Input** | None (just choose arm) | Context vector x_t |
| **Decision** | a_t ∈ {1, ..., K} | a_t = f(x_t) |
| **Reward** | r_t ~ P(r \| a_t) | r_t ~ P(r \| x_t, a_t) |
| **Goal** | Learn best arm | Learn best arm per context |
| **Regret** | O(√T) | O(d√T) where d = context dim |

### Notation

- **x_t** = d-dimensional context vector at round t
- **a_t** = chosen arm at round t
- **r_t** = observed reward at round t
- **θ_a** = parameter vector for arm a
- **K** = number of arms
- **d** = context dimension
- **T** = number of rounds

### Key Algorithms

**LinUCB (Linear UCB):**
```
Initialize: A_a = λI, b_a = 0 for all arms a

At each round t:
  1. Observe context x_t
  2. For each arm a:
     θ̂_a = A_a^{-1} b_a
     UCB_a = x_t^T θ̂_a + α√(x_t^T A_a^{-1} x_t)
  3. Choose a_t = argmax_a UCB_a
  4. Observe reward r_t
  5. Update:
     A_{a_t} ← A_{a_t} + x_t x_t^T
     b_{a_t} ← b_{a_t} + r_t x_t
```

**Parameters:**
- α ∈ [0.1, 2.0]: Exploration strength (start with 1.0)
- λ > 0: Regularization (start with 1.0)

## LinUCB Implementation Template

```python
import numpy as np

class LinUCB:
    def __init__(self, n_arms, context_dim, alpha=1.0, lambda_=1.0):
        self.n_arms = n_arms
        self.d = context_dim
        self.alpha = alpha
        self.A = [lambda_ * np.eye(context_dim) for _ in range(n_arms)]
        self.b = [np.zeros(context_dim) for _ in range(n_arms)]

    def choose_arm(self, context):
        ucb_scores = []
        for a in range(self.n_arms):
            theta = np.linalg.solve(self.A[a], self.b[a])
            pred = context @ theta
            std = np.sqrt(context @ np.linalg.solve(self.A[a], context))
            ucb_scores.append(pred + self.alpha * std)
        return np.argmax(ucb_scores)

    def update(self, arm, context, reward):
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

## Feature Engineering Checklist

### Must-Have Properties
- [ ] **Observable** — Known before decision time (no future leakage)
- [ ] **Predictive** — Different arms optimal in different contexts
- [ ] **Scaled** — Normalized to similar ranges
- [ ] **Stationary** — Distribution doesn't drift wildly
- [ ] **Non-redundant** — Low correlation with other features

### Commodity Context Features (Tier 1)

**Volatility Regime:**
```python
vol = returns.rolling(20).std()
vol_zscore = (vol - vol.mean()) / vol.std()
```

**Term Structure:**
```python
term_spread = (back_price - front_price) / front_price
term_zscore = (term_spread - term_spread.mean()) / term_spread.std()
```

**Seasonality:**
```python
month_sin = np.sin(2 * np.pi * month / 12)
month_cos = np.cos(2 * np.pi * month / 12)
```

**Trend:**
```python
ma_short = prices.rolling(20).mean()
ma_long = prices.rolling(50).mean()
trend = (ma_short - ma_long) / ma_long
```

**Macro Regime:**
```python
# Dollar + VIX composite
dollar_z = (dollar - dollar.mean()) / dollar.std()
vix_z = (vix - vix.mean()) / vix.std()
macro = 0.5 * dollar_z + 0.5 * vix_z
```

## Parameter Selection Guide

### Alpha (Exploration)

| α Value | Behavior | Use When |
|---------|----------|----------|
| 0.1 | Mostly exploitation | High confidence in linear model |
| 0.5 | Balanced | General purpose |
| 1.0 | **Default** | Starting point |
| 2.0 | Aggressive exploration | Noisy rewards, uncertain model |
| 5.0+ | Over-exploration | Rarely useful |

**Tuning:** Run offline simulation with different α values, pick one with lowest regret.

### Lambda (Regularization)

| λ Value | Effect | Use When |
|---------|--------|----------|
| 0.01 | Minimal reg | Large dataset (n >> d) |
| 1.0 | **Default** | General purpose |
| 10.0 | Strong reg | Small dataset, many features |

**Rule of thumb:** λ = 1.0 works well in most cases.

### Context Dimension

| d | Complexity | Recommendation |
|---|------------|----------------|
| 2-3 | Low | Start here |
| 4-7 | Medium | Add features if they help |
| 8-15 | High | Careful feature selection needed |
| 16+ | Very high | Consider dimensionality reduction |

**Regret scales with d:** More features = slower learning.

## Common Patterns

### Pattern 1: Regime-Dependent Allocation
```python
# Context: market regime
context = np.array([
    vol_zscore,
    term_spread_zscore,
    seasonality_sin
])

# Arms: sector allocations
arms = ['Energy', 'Metals', 'Agriculture']

# LinUCB learns which sector wins in each regime
bandit = LinUCB(n_arms=3, context_dim=3)
arm = bandit.choose_arm(context)
```

### Pattern 2: Personalized Recommendations
```python
# Context: user features
context = np.array([
    user_engagement_score,
    content_freshness,
    topic_match
])

# Arms: content pieces to recommend
bandit = LinUCB(n_arms=10, context_dim=3)
content_id = bandit.choose_arm(context)
```

### Pattern 3: Dynamic Pricing
```python
# Context: demand signals
context = np.array([
    time_of_day_sin,
    day_of_week_sin,
    recent_sales_velocity
])

# Arms: price points
arms = [0.99, 1.99, 2.99, 3.99]
bandit = LinUCB(n_arms=4, context_dim=3)
price_idx = bandit.choose_arm(context)
```

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Always chooses same arm | α too small | Increase α to 1.0+ |
| Random choices (no learning) | α too large or features unpredictive | Decrease α, check feature correlations |
| Poor performance vs standard bandit | Bad features | Check feature-reward correlation |
| Numerical errors | λ too small | Set λ ≥ 1.0 |
| Slow convergence | Too many features | Feature selection, drop redundant ones |

## Quick Diagnostics

```python
# Check feature quality
features.describe()  # Look for variance, range
features.corr()      # Identify redundant features
features.corrwith(rewards)  # Predictive power

# Check LinUCB state
for a in range(bandit.n_arms):
    theta = np.linalg.solve(bandit.A[a], bandit.b[a])
    print(f"Arm {a} weights: {theta}")

# Compare to baseline
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
model.fit(X, y)
print(f"Offline R²: {model.score(X, y)}")
```

## Performance Metrics

**Regret:**
```python
regret_t = max_a E[r|x_t, a] - E[r|x_t, a_t]
cumulative_regret = sum(regret_t)
```

**Suboptimality:**
```python
# How often did we choose best arm?
best_arm = oracle(context)
accuracy = (chosen_arm == best_arm).mean()
```

**Reward comparison:**
```python
# Compare to baselines
bandit_reward = sum(rewards_contextual)
random_reward = sum(rewards_random)
best_reward = sum(rewards_oracle)

# Relative performance
relative = (bandit_reward - random_reward) / (best_reward - random_reward)
```

## When to Use Contextual vs Standard Bandits

**Use Contextual Bandits when:**
- Optimal arm changes with observable conditions
- You have reliable, predictive context features
- d (context dim) is small relative to T (time horizon)
- Interpretability matters (linear model is transparent)

**Use Standard Bandits when:**
- Optimal arm is relatively stable
- No good context features available
- Very short time horizon (< 100 rounds)
- Simplicity is critical

**Use Full RL when:**
- Actions affect future states (sequential dependencies)
- Long-term planning required
- State space is complex

## Key Formulas

**Ridge regression solution:**
```
θ̂_a = (X_a^T X_a + λI)^{-1} X_a^T r_a = A_a^{-1} b_a
```

**Prediction uncertainty:**
```
σ_a(x) = √(x^T A_a^{-1} x)
```

**UCB score:**
```
UCB_a(x) = x^T θ̂_a + α · σ_a(x)
```

**Regret bound:**
```
Regret(T) = O(d√(T log T))
```

## Code Snippets

**Feature normalization:**
```python
def normalize_features(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return (X - mean) / std
```

**Sherman-Morrison update (efficient):**
```python
# Instead of recomputing A^{-1} each time
# Update A^{-1} directly when adding x x^T to A
def sherman_morrison_update(A_inv, x):
    numerator = A_inv @ np.outer(x, x) @ A_inv
    denominator = 1 + x @ A_inv @ x
    return A_inv - numerator / denominator
```

**Contextual epsilon-greedy:**
```python
def contextual_epsilon_greedy(context, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(n_arms)
    else:
        # Greedy: choose arm with best predicted reward
        predictions = [predict(context, a) for a in range(n_arms)]
        return np.argmax(predictions)
```

---

**Remember:** Contextual bandits are worth the complexity when context genuinely predicts which arm is best. If not, stick with simpler approaches.

# Contextual Bandit Framework

## In Brief

A contextual bandit observes features about the current situation (context) before choosing an action, enabling adaptive decisions that depend on circumstances. Unlike standard bandits that learn "arm 2 is best on average," contextual bandits learn "arm 2 is best when context features indicate high volatility and contango."

## Key Insight

**Standard bandit:** Fixed arm preferences learned from experience
**Contextual bandit:** Context-dependent preferences learned as mappings from features to rewards

The transformation is profound. A commodity allocator using contextual bandits doesn't just learn "energy beats metals on average." It learns "energy wins in low-volatility contango, metals win in high-volatility backwardation." The context vector makes decisions situational.

## Visual Explanation

```
STANDARD BANDIT:
┌─────────────────────────────────────────┐
│  Choose Arm → Observe Reward → Update  │
│                                         │
│  Same decision process every round     │
└─────────────────────────────────────────┘

CONTEXTUAL BANDIT:
┌─────────────────────────────────────────────────────┐
│  Observe Context → Choose Arm → Reward → Update    │
│        x_t             a_t         r_t              │
│                                                     │
│  x_t = [volatility, term_structure, seasonality]   │
│                                                     │
│  Decision depends on current market regime         │
└─────────────────────────────────────────────────────┘

CONTEXTUAL DECISION MAPPING:
     Context Features          Best Arm
┌─────────────────────┐    ┌───────────┐
│ Vol=Low, Contango   │───→│  Energy   │
│ Vol=High, Backw.    │───→│  Metals   │
│ Vol=Med, Harvest    │───→│  Ag       │
└─────────────────────┘    └───────────┘
```

## Formal Definition

**Standard Multi-Armed Bandit:**
- At each round t = 1, 2, ..., T
- Choose arm a_t ∈ {1, 2, ..., K}
- Observe reward r_t ~ P(r | a_t)
- Goal: maximize Σ r_t

**Contextual Multi-Armed Bandit:**
- At each round t:
  1. **Observe context:** x_t ∈ ℝ^d (d-dimensional feature vector)
  2. **Choose arm:** a_t ∈ {1, 2, ..., K} based on x_t
  3. **Observe reward:** r_t ~ P(r | x_t, a_t)
  4. **Update model:** Learn f(x, a) → E[r | x, a]

**Key mathematical statement:**
```
E[r_t | x_t, a_t] = f(x_t, a_t)
```

The expected reward is a function of BOTH the context and the chosen arm. The goal is to learn this function f through experience while minimizing regret.

**Regret definition:**
```
Regret(T) = Σ_{t=1}^T [r_t^* - r_t]

where r_t^* = max_a E[r | x_t, a] is the best possible reward given context x_t
```

## Intuitive Explanation

**It's like a doctor choosing treatment:**

- **Standard bandit:** "Drug A works best on average across all patients" → prescribe Drug A to everyone
- **Contextual bandit:** "Drug A works best for patients with symptoms X and Y, but Drug B is better for patients with symptom Z" → prescribe based on patient features

In commodity trading:
- **Standard bandit:** "Energy sector earns 8% annually on average" → always overweight energy
- **Contextual bandit:** "Energy earns 12% when VIX < 15 and in contango, but loses money in backwardation with VIX > 25" → switch allocation based on current regime

The context vector encodes the current market state. The bandit learns which allocation works in which state.

## Code Implementation

```python
import numpy as np

class ContextualBandit:
    """Simple contextual bandit with separate models per arm."""

    def __init__(self, n_arms, context_dim):
        self.n_arms = n_arms
        self.context_dim = context_dim
        # Store context-reward pairs for each arm
        self.contexts = [[] for _ in range(n_arms)]
        self.rewards = [[] for _ in range(n_arms)]

    def predict(self, context, arm):
        """Predict reward for arm given context."""
        if len(self.rewards[arm]) == 0:
            return 0.0  # No data yet
        # Simple: average reward for similar contexts
        X = np.array(self.contexts[arm])
        y = np.array(self.rewards[arm])
        return np.mean(y)  # Simplified

    def choose_arm(self, context):
        """Choose best arm for given context."""
        predictions = [
            self.predict(context, a)
            for a in range(self.n_arms)
        ]
        return np.argmax(predictions)

    def update(self, context, arm, reward):
        """Update model with new observation."""
        self.contexts[arm].append(context)
        self.rewards[arm].append(reward)

# Example: 3 commodities, 2 features
bandit = ContextualBandit(n_arms=3, context_dim=2)
context = np.array([0.15, 0.02])  # [VIX, term_spread]
arm = bandit.choose_arm(context)
```

## Common Pitfalls

### 1. **Using irrelevant features**
- **Problem:** Adding features that don't predict rewards just adds noise
- **Example:** Including day-of-week for long-term commodity allocation (irrelevant)
- **Fix:** Feature selection — only use features with predictive power

### 2. **Feature scale mismatch**
- **Problem:** Volatility ranges 0-1, while price ranges 50-100; model weights favor high-magnitude features
- **Example:** Term spread (-5 to +5) dominates normalized VIX (0 to 1)
- **Fix:** Standardize features: `(x - mean) / std` or normalize to [0, 1]

### 3. **Overfitting to context**
- **Problem:** Learning noise in context features rather than true signal
- **Example:** With 20 features and 100 observations, model memorizes rather than generalizes
- **Fix:** Regularization (ridge, lasso), feature selection, more data

### 4. **Context leakage**
- **Problem:** Using future information in context that wouldn't be available at decision time
- **Example:** Including tomorrow's VIX in today's context
- **Fix:** Strict temporal discipline — only use lagged features

### 5. **Ignoring exploration**
- **Problem:** Contextual bandits still need exploration; greedy selection leads to suboptimal convergence
- **Example:** Always picking the current best arm → never learn about alternatives in new contexts
- **Fix:** Use UCB or Thompson Sampling with contextual models (LinUCB, Thompson with regression)

## Connections

### Builds On
- **Module 1 (Bandit Algorithms):** UCB and Thompson Sampling extend to contextual setting
- **Module 2 (Bayesian Bandits):** Bayesian updating applies to contextual models
- **Regression fundamentals:** Contextual bandits are essentially online regression with exploration

### Leads To
- **Module 4 (Content Optimization):** Personalization = contextual bandits with user features
- **Module 5 (Commodity Trading):** Regime-aware allocation = contextual bandits with market features
- **Module 6 (Non-stationary Bandits):** Time-varying contexts handle regime shifts
- **Full Reinforcement Learning:** Contextual bandits are stateless RL (no sequential state transitions)

### Key Distinction from Full RL
- **Contextual Bandit:** Context x_t is exogenous (not affected by your actions)
- **Reinforcement Learning:** State s_t depends on previous actions (sequential decision process)

## Practice Problems

### 1. Conceptual Understanding
**Question:** A commodity trader observes [current_inventory, term_spread, VIX] before choosing between three sector allocations. Is this a standard bandit or contextual bandit problem? Why?

**Answer:** Contextual bandit. The trader observes features (inventory, term spread, VIX) that describe the current market state before making a decision. The optimal allocation depends on these features.

### 2. Feature Design
**Question:** You're building a contextual bandit for intraday commodity trading. Which of these features are appropriate?
- a) Yesterday's closing price
- b) Current bid-ask spread
- c) Tomorrow's inventory report (not yet released)
- d) Current rolling 20-day volatility

**Answer:** b and d are appropriate (observable before decision). a is okay if properly lagged. c is invalid (future information).

### 3. Implementation Challenge
**Task:** Modify the code above to use ridge regression for prediction instead of simple averaging.

**Hint:**
```python
from sklearn.linear_model import Ridge

def predict(self, context, arm):
    if len(self.rewards[arm]) < 2:
        return 0.0
    X = np.array(self.contexts[arm])
    y = np.array(self.rewards[arm])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model.predict([context])[0]
```

### 4. Regret Analysis
**Question:** In a contextual bandit with 3 arms and 2-dimensional context, you always choose the greedy arm (no exploration). Why might this be worse than a standard bandit with proper exploration?

**Answer:** Without exploration, you never learn about alternative arms in new context regions. You might permanently commit to suboptimal arms in certain contexts because you never tried the alternatives. Standard bandits with UCB/Thompson at least explore all arms eventually.

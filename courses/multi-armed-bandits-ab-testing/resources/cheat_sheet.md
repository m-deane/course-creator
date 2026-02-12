# Multi-Armed Bandits Cheat Sheet

## Algorithm Comparison

| Algorithm | Exploration | Tuning | Regret | Best For |
|-----------|-------------|--------|---------|----------|
| **Epsilon-Greedy** | Random (ε%) | Need ε | O(T) if ε fixed | Simple baseline |
| **UCB1** | Optimistic | None | O(log T) | Finite arms, stationary |
| **Thompson Sampling** | Probabilistic | Prior only | O(log T) | Bayesian, works everywhere |
| **LinUCB** | Contextual | λ (regularization) | O(√T·d·log T) | Contextual, linear rewards |
| **Discounted TS** | Adaptive | γ (decay) | - | Non-stationary |

**Rule of thumb:**
- **Start with:** Thompson Sampling (robust, minimal tuning)
- **Add context:** Use LinUCB if features available
- **Non-stationary:** Add discounting or sliding window
- **Simplicity:** UCB1 if you want zero tuning

## Key Formulas

### UCB1 (Upper Confidence Bound)

```
Pick arm i = argmax [μ̂ᵢ + √(2·ln(t) / nᵢ)]

Where:
  μ̂ᵢ = estimated mean reward of arm i
  t = total rounds so far
  nᵢ = times arm i was pulled
```

### Thompson Sampling (Beta-Bernoulli)

```
For each arm i:
  Sample θᵢ ~ Beta(αᵢ, βᵢ)
  Pick i = argmax θᵢ

Update after observing reward r ∈ {0,1}:
  If r=1: αᵢ ← αᵢ + 1
  If r=0: βᵢ ← βᵢ + 1
```

### LinUCB (Linear Contextual)

```
For arm i with context x:
  θᵢ = Aᵢ⁻¹ bᵢ              (linear model)
  UCBᵢ = xᵀθᵢ + α·√(xᵀAᵢ⁻¹x)  (upper confidence bound)
  Pick i = argmax UCBᵢ

Update after reward r:
  Aᵢ ← Aᵢ + xxᵀ
  bᵢ ← bᵢ + r·x
```

### Regret

```
Cumulative Regret:
  R(T) = T·μ* - Σₜ rₜ

Where:
  μ* = mean of best arm
  rₜ = reward at round t

Logarithmic regret (optimal):
  R(T) = O(log T)

Linear regret (bad):
  R(T) = O(T)
```

### Sharpe Ratio (Risk-Adjusted Reward)

```
Sharpe = (μ - rᶠ) / σ

Where:
  μ = mean return
  rᶠ = risk-free rate
  σ = volatility (standard deviation)

Annualized (weekly data):
  Sharpe_annual = Sharpe_weekly × √52
```

## Decision Tree: Which Algorithm?

```
Do you have contextual features?
├─ YES → LinUCB or Contextual Thompson Sampling
│         ├─ Linear rewards? → LinUCB
│         └─ Non-linear? → Neural bandits (advanced)
│
└─ NO → Classic bandits
          ├─ Non-stationary (changing rewards)?
          │   └─ YES → Discounted Thompson Sampling or UCB with sliding window
          │
          └─ Stationary rewards?
                ├─ Want Bayesian interpretation?
                │   └─ YES → Thompson Sampling
                │
                └─ Want simplicity + guarantees?
                      └─ UCB1
```

## Reward Design Checklist

For commodity trading bandits:

- [ ] **Risk-adjusted:** Use Sharpe, not raw returns (avoid chasing volatility)
- [ ] **Stationary:** Normalize by recent volatility (make comparable across regimes)
- [ ] **Relevant:** Matches real goal (accumulate? minimize regret? maximize Sharpe?)
- [ ] **Observable:** Can measure it reliably each period
- [ ] **Timely:** Not delayed more than 1-2 periods
- [ ] **Bounded:** Outliers won't break learning (clip extreme values)

**Anti-patterns:**
- ❌ Raw returns (optimize for vol, not skill)
- ❌ Views/clicks (vanity metrics, not value)
- ❌ Sparse rewards (success only once per month → slow learning)
- ❌ Delayed rewards (feedback 6 months later → can't adapt)

## Guardrail Checklist

For commodity allocators:

- [ ] **Position limits:** Max 40% any single commodity
- [ ] **Minimum allocation:** Min 5% all commodities (maintain diversification)
- [ ] **Tilt speed:** Max 15-20% allocation change per period
- [ ] **Volatility scaling:** Reduce exposure when VIX > 30
- [ ] **Drawdown breaker:** Halt if portfolio down > 15% from peak
- [ ] **Correlation monitor:** Alert if portfolio correlation > 0.8
- [ ] **Liquidity check:** Never exceed 10% of average daily volume

**Implementation:**
```python
def apply_guardrails(weights, portfolio_state):
    # Position limits
    weights = np.clip(weights, min_alloc, max_alloc)
    weights = weights / weights.sum()

    # Tilt speed
    if np.max(np.abs(weights - prev_weights)) > max_tilt:
        weights = 0.8 * weights + 0.2 * prev_weights

    # Drawdown breaker
    if drawdown < -max_drawdown:
        weights = equal_weight()  # Revert to safe allocation

    return weights
```

## Commodity Allocation Quick-Start

**6 steps from the Accumulator Bandit Playbook:**

1. **Choose core allocation** (80%)
   - Simple: Equal-weight across all commodities
   - Strategic: Based on long-term fundamental view

2. **Set bandit sleeve** (20%)
   - Conservative: 10-15%
   - Moderate: 20-25%
   - Aggressive: 30-40%

3. **Define arms** (5-10 arms)
   - Broad: Energy, Metals, Grains, Softs, Livestock
   - Specific: WTI, Gold, Copper, NatGas, Corn, etc.

4. **Weekly execution**
   - Monday: Get Thompson Sampling allocation
   - Execute trades to match target
   - Friday: Calculate realized returns, update beliefs

5. **Track risk-adjusted score**
   - Reward = Sharpe ratio (return / vol)
   - OR: Return - λ × max_drawdown
   - NOT: Raw returns

6. **Apply guardrails**
   - Position limits, tilt speed, drawdown breaker
   - See guardrail checklist above

## Code Snippets

### Thompson Sampling (5 lines)

```python
from scipy.stats import beta

# Select
samples = [beta.rvs(alpha[i], beta_[i]) for i in range(K)]
arm = np.argmax(samples)

# Update
if reward == 1:
    alpha[arm] += 1
else:
    beta_[arm] += 1
```

### UCB1 (3 lines)

```python
t = sum(pulls)
ucb = means + np.sqrt(2 * np.log(t) / (pulls + 1e-10))
arm = np.argmax(ucb)
```

### Two-Wallet Allocation (4 lines)

```python
core_weights = np.ones(K) / K  # Equal-weight
bandit_weights = thompson_sampling.get_weights()
total_weights = 0.8 * core_weights + 0.2 * bandit_weights
```

### Risk-Adjusted Reward (2 lines)

```python
sharpe = returns / np.maximum(volatilities, 0.01)
reward = sharpe  # Use this instead of raw returns
```

## Common Pitfalls

| Pitfall | Why It Fails | Fix |
|---------|-------------|-----|
| Using raw returns as reward | Chases volatility | Use Sharpe ratio |
| No minimum allocation | Abandons arms after one bad draw | Set min 5% |
| Bandit sleeve too large | Excessive turnover, concentration | Start with 20% |
| No guardrails | Overfits to recent noise | Add position limits, tilt speed |
| Fixed epsilon | Too much or too little exploration | Use UCB/Thompson (auto-tunes) |
| Ignoring non-stationarity | Stale beliefs in changing markets | Discount old data or use sliding window |
| Arms = specific events | Can't repeat, can't learn | Arms = repeatable strategies |

## Parameter Recommendations

### Epsilon-Greedy
```
Initial ε: 0.3-0.5 (explore a lot early)
Decay: ε = 1/√t or ε = c/t
Minimum ε: 0.05-0.1 (always explore a bit)
```

### UCB1
```
Confidence: c = 2 (standard, don't tune)
Or c = 1 for conservative, c = 3 for aggressive
```

### Thompson Sampling
```
Prior: Beta(1, 1) for uniform (uninformative)
Or Beta(α, β) matching historical data
```

### LinUCB
```
Regularization: λ = 1.0 (default)
Exploration: α = 1.0-2.0
```

### Discounting
```
Decay: γ = 0.95-0.99 (per period)
Effective window: ~1/(1-γ) periods
  γ=0.95 → ~20 periods
  γ=0.99 → ~100 periods
```

## When to Use Bandits vs A/B Testing

**Use Bandits When:**
- ✅ You can't stop operating (must keep making decisions)
- ✅ Opportunity cost matters (exploration is expensive)
- ✅ Many options to test (> 3 arms)
- ✅ Environment non-stationary (rewards change over time)
- ✅ Willing to deploy adaptive system

**Use A/B Testing When:**
- ✅ Need clean statistical inference (p-values, confidence intervals)
- ✅ One-time decision (pick winner, done)
- ✅ Few options (2-3 variants)
- ✅ Can afford to waste samples on losers
- ✅ Regulatory/compliance requires fixed design

**Best of Both:**
Start with bandit for efficiency, run A/B test on top-2 finalists for clean inference.

---

**Print this and keep it handy while working through the course!**

# Accumulator Bandit Playbook

## In Brief

A practical 6-step system for applying multi-armed bandits to commodity portfolio allocation. Separates stable core holdings from an adaptive "bandit sleeve" that tilts toward better-performing assets while maintaining safety guardrails.

> 💡 **Key Insight:** Don't use bandits to predict prices. Use them to **allocate contributions under uncertainty**.

The two-wallet framework:
- **Core wallet (80%)**: Boring, stable, long-term allocation
- **Bandit sleeve (20%)**: Adaptive tilts based on recent performance

This prevents the bandit from becoming a dangerous dopamine machine while still capturing the benefits of adaptive allocation.

## Visual Explanation

```
┌─────────────────────────────────────────────────────┐
│         Total Portfolio ($100K)                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  CORE WALLET (80% = $80K)                           │
│  ┌────────────────────────────────────────┐         │
│  │ Equal-weight across all commodities    │         │
│  │ Rebalanced monthly                     │         │
│  │ Stable, predictable, boring            │         │
│  │                                        │         │
│  │ Energy: $16K | Metals: $16K            │         │
│  │ Grains: $16K | Softs: $16K             │         │
│  │ Livestock: $16K                        │         │
│  └────────────────────────────────────────┘         │
│                                                      │
│  BANDIT SLEEVE (20% = $20K)                         │
│  ┌────────────────────────────────────────┐         │
│  │ Thompson Sampling adaptive allocation   │         │
│  │ Rebalanced weekly                      │         │
│  │ Tilts toward outperformers             │         │
│  │                                        │         │
│  │ THIS WEEK:                             │         │
│  │ WTI: $8K (40%) ← Tilted up             │         │
│  │ Gold: $6K (30%)                        │         │
│  │ Copper: $4K (20%)                      │         │
│  │ NatGas: $2K (10%) ← Tilted down        │         │
│  └────────────────────────────────────────┘         │
│                                                      │
└─────────────────────────────────────────────────────┘

Weekly process:
1. Core wallet: Buy $800 across all commodities (equal-weight)
2. Bandit sleeve: Thompson Sampling chooses allocation for $200
3. Observe realized returns
4. Update beliefs
5. Repeat next week
```

## Formal Definition

**Two-Wallet Bandit Portfolio:**

Let:
- `C` = core allocation proportion (e.g., 0.8)
- `B` = bandit sleeve proportion (e.g., 0.2)
- `K` = number of commodity arms
- `w_core` = core weight vector (typically uniform: `1/K` for all arms)
- `w_bandit(t)` = bandit weight vector at time `t` (adaptive)

Total portfolio weight at time `t`:
```
w_total(t) = C * w_core + B * w_bandit(t)
```

**Accumulator Objective:**

Maximize cumulative risk-adjusted returns over horizon `T`:
```
max Σ_t [r(t) - λ * σ(t)]
```

Where:
- `r(t)` = portfolio return at time `t`
- `σ(t)` = portfolio volatility at time `t`
- `λ` = risk aversion parameter

Subject to guardrails:
- Position limits: `w_i(t) ≤ w_max` for all arms `i`
- Tilt speed: `|w_i(t) - w_i(t-1)| ≤ Δ_max`
- Minimum allocation: `w_i(t) ≥ w_min > 0`

## Intuitive Explanation

Think of it like investing in a diversified mutual fund (core) while also having a small "play money" account (bandit sleeve) that you use to experiment with tilts.

The core provides stability and ensures you're always participating in the market. The bandit sleeve learns which tilts work, but it's constrained so it can't blow up your portfolio.

**Analogy:** It's like a restaurant with:
- **Core menu** (80%): Classic dishes that always work
- **Chef's specials** (20%): Experiments based on what's fresh and working well

The core keeps customers happy. The specials let you adapt and innovate.

## Code Implementation

```python
import numpy as np
import pandas as pd

class TwoWalletBandit:
    """
    Two-wallet commodity allocator with Thompson Sampling.
    """
    def __init__(
        self,
        arms,
        core_pct=0.8,
        bandit_pct=0.2,
        prior_mean=0.001,
        prior_std=0.02,
        min_allocation=0.05,
        max_allocation=0.50
    ):
        self.arms = arms
        self.K = len(arms)
        self.core_pct = core_pct
        self.bandit_pct = bandit_pct

        # Thompson Sampling parameters
        self.means = np.full(self.K, prior_mean)
        self.stds = np.full(self.K, prior_std)
        self.n = np.zeros(self.K)

        # Guardrails
        self.min_alloc = min_allocation
        self.max_alloc = max_allocation

    def get_core_weights(self):
        """Core: equal-weight across all arms."""
        return np.ones(self.K) / self.K

    def get_bandit_weights(self):
        """Bandit sleeve: Thompson Sampling with guardrails."""
        # Sample from posterior for each arm
        samples = np.random.normal(self.means, self.stds)

        # Softmax to convert samples to weights
        exp_samples = np.exp(samples - samples.max())
        weights = exp_samples / exp_samples.sum()

        # Apply guardrails
        weights = np.clip(weights, self.min_alloc, self.max_alloc)
        weights = weights / weights.sum()  # Re-normalize

        return weights

    def get_total_weights(self):
        """Combine core and bandit allocations."""
        w_core = self.get_core_weights()
        w_bandit = self.get_bandit_weights()
        return self.core_pct * w_core + self.bandit_pct * w_bandit

    def update(self, returns):
        """
        Update beliefs based on observed returns.

        Args:
            returns: Array of returns for each arm
        """
        for i in range(self.K):
            # Bayesian update (assuming known variance for simplicity)
            self.n[i] += 1
            lr = 1 / (self.n[i] + 1)
            self.means[i] = (1 - lr) * self.means[i] + lr * returns[i]
            self.stds[i] = self.stds[i] / np.sqrt(1 + self.n[i])


# Example usage
if __name__ == "__main__":
    # Define commodity arms
    commodities = ['WTI', 'Gold', 'Copper', 'NatGas', 'Corn']

    # Initialize bandit
    bandit = TwoWalletBandit(
        arms=commodities,
        core_pct=0.8,
        bandit_pct=0.2
    )

    # Simulate one week
    print("Week 1 Allocation:")
    weights = bandit.get_total_weights()
    for commodity, weight in zip(commodities, weights):
        print(f"  {commodity}: {weight:.2%}")

    # Simulate returns (in practice, these come from real market data)
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    bandit.update(returns)

    print("\nWeek 2 Allocation (after update):")
    weights = bandit.get_total_weights()
    for commodity, weight in zip(commodities, weights):
        print(f"  {commodity}: {weight:.2%}")
```

## The 6-Step Accumulator Bandit Playbook

### Step 1: Choose Your Core Allocation

**Boring is beautiful.**

Your core should be:
- Simple (equal-weight or strategic weights based on fundamentals)
- Stable (rebalanced monthly or quarterly, not weekly)
- Diversified (across commodity sectors)

Example core portfolios:
- **Equal-weight**: 20% Energy, 20% Metals, 20% Grains, 20% Softs, 20% Livestock
- **Strategic**: Based on long-term fundamental view (e.g., 30% Energy, 25% Metals, 15% Grains, 15% Softs, 15% Livestock)

### Step 2: Set Your Bandit Sleeve

**Small enough to experiment, large enough to matter.**

Typical ranges:
- Conservative: 10-15% bandit sleeve
- Moderate: 20-25% bandit sleeve
- Aggressive: 30-40% bandit sleeve

Start with 20%. You can always adjust.

### Step 3: Define Your Arms

**5-10 arms is the sweet spot.**

Too few (< 5): Not enough diversification
Too many (> 10): Slow learning, high noise

Example arm definitions:
- **Broad sectors**: Energy, Metals, Grains, Softs, Livestock
- **Specific commodities**: WTI, Gold, Copper, NatGas, Corn, Soybeans, Coffee, Cattle
- **Strategy factors**: Momentum, Mean-reversion, Carry, Seasonality

### Step 4: Weekly Execution

**Consistency beats perfection.**

Every Monday (or your chosen day):
1. Core wallet: Execute planned contributions (equal-weight or strategic)
2. Bandit sleeve: Get current weights from Thompson Sampling
3. Execute trades to match target allocation
4. Record executed prices

End of week (Friday close):
1. Calculate realized returns for each arm
2. Update bandit beliefs
3. Log results

### Step 5: Track the Right Score

**Your reward function IS your strategy.**

Match your score to your real goal:

| Goal | Reward Function |
|------|----------------|
| Accumulate steadily | Sharpe ratio (return / volatility) |
| Minimize regret | Return relative to best arm in hindsight |
| Hold comfortably | Return - λ * max_drawdown |
| Align with thesis | Return if matches thesis, penalty if not |

See [Reward Design Guide](02_reward_design_commodities.md) for full details.

### Step 6: Hard Guardrails

**These prevent self-sabotage.**

Implement ALL of these:

1. **Position limits**: No arm > 50% of bandit sleeve
2. **Minimum allocation**: No arm < 5% of bandit sleeve (prevents abandoning arms after one bad week)
3. **Tilt speed limits**: Allocation can't change more than 20% week-over-week
4. **Core protection**: Core rebalances monthly, bandit weekly (different timescales)
5. **Volatility dampening**: When VIX > 30, reduce bandit sleeve to 10%

See [Guardrails Guide](03_guardrails_and_safety.md) for implementations.

## Common Pitfalls

### Pitfall 1: Using Raw Returns as Reward
**What happens:** Your bandit becomes a trend-chasing machine that buys high and sells low.

**Why:** Last week's winner is often next week's loser (mean reversion).

**Fix:** Use risk-adjusted or regret-relative rewards.

### Pitfall 2: No Minimum Allocation
**What happens:** One bad week causes you to zero-out an arm. Then it rebounds and you miss it.

**Why:** Sample size of one is insufficient. Volatility != bad arm.

**Fix:** Set `min_allocation >= 5%` to maintain exposure to all arms.

### Pitfall 3: Bandit Sleeve Too Large
**What happens:** Your portfolio becomes a high-frequency trading machine, churning excessively.

**Why:** Large sleeve magnifies every belief update.

**Fix:** Start with 20% or less. You can always increase later.

### Pitfall 4: No Core Wallet
**What happens:** Pure bandit allocation leads to extreme concentration and wild swings.

**Why:** Exploration requires taking risks. Without a stable core, those risks can be portfolio-destroying.

**Fix:** Always maintain 60-80% in core allocation.

### Pitfall 5: Ignoring Transaction Costs
**What happens:** Theoretical gains are eaten by bid-ask spreads and commissions.

**Why:** Bandits naturally want to rebalance frequently.

**Fix:** Add transaction cost penalty to reward function, or use tilt speed limits.

## Connections

### Builds On
- **Module 1**: Thompson Sampling (the core algorithm)
- **Module 2**: Regret analysis (why adaptive allocation helps)
- **Module 3**: Contextual bandits (regime-aware extension)

### Leads To
- **Module 6**: When to use bandits vs A/B testing
- **Project 2**: Build your own commodity allocation system
- **Bayesian Commodity Forecasting course**: Enhanced regime detection

### Related Concepts
- **Bayesian updating**: How beliefs evolve with each observation
- **Risk parity**: Alternative allocation framework (static, not adaptive)
- **Kelly criterion**: Optimal bet sizing (related but different objective)

## Practice Problems

### Problem 1: Core vs Bandit Split
Your portfolio is $100K. You're considering three setups:
- A: 90% core, 10% bandit
- B: 80% core, 20% bandit
- C: 60% core, 40% bandit

For each, calculate:
- Maximum possible loss from bandit in one week (assume worst arm loses 20%)
- Minimum number of weeks to learn arm ranking (assume 5 arms)

Which would you choose and why?

### Problem 2: Arm Definition
You trade commodities across three sectors:
- Energy: WTI, Brent, NatGas, Gasoline
- Metals: Gold, Silver, Copper, Platinum
- Grains: Corn, Soybeans, Wheat

Design two arm structures:
- **Broad**: Sector-level arms (3 arms)
- **Granular**: Individual commodity arms (11 arms)

What are the tradeoffs? Which would you use?

### Problem 3: Reward Design
Your goal is to accumulate a commodity position over 12 months for a project starting next year. You don't need to maximize returns; you need to:
- Get to your target position
- Not blow up your budget
- Sleep well at night

Design a reward function that matches this goal. Write it in Python.

---

**Next Steps:**
- Read [Reward Design for Commodities](02_reward_design_commodities.md) to master the most critical decision
- Or jump to [Two-Wallet Framework Notebook](../notebooks/01_two_wallet_framework.ipynb) to build it yourself

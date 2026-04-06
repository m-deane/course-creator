# Reward Design for Commodities

> **Reading time:** ~20 min | **Module:** 05 — Commodity Trading Bandits | **Prerequisites:** Module 4


## In Brief


<div class="callout-key">

**Key Concept Summary:** Your reward function is the most important decision in bandit-based trading. It defines what "success" means and shapes every allocation decision. Choose poorly and you train a system that sabotage...

</div>

Your reward function is the most important decision in bandit-based trading. It defines what "success" means and shapes every allocation decision. Choose poorly and you train a system that sabotages your real goals.

> 💡 **Key Insight:** Your reward function IS your trading strategy.

A bandit maximizes whatever you tell it to maximize. If you say "maximize weekly returns," it will trend-chase and blow up. If you say "maximize risk-adjusted returns with drawdown protection," it will build a stable allocator.

The reward function isn't just a metric. It's the objective your system optimizes.

## Visual Explanation

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


```
┌──────────────────────────────────────────────────────────────┐
│  How Different Rewards Shape Allocator Behavior              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  REWARD 1: Raw Weekly Returns                                │
│  ┌────────────────────────────────────────┐                  │
│  │  Allocation ████████░░░░░░░░░░░░░░░░   │ ← Chases        │
│  │  over time  ░░░░░░░░████████░░░░░░░░   │   last week's   │
│  │             ░░░░░░░░░░░░░░░░████████   │   winner        │
│  │  Result: High turnover, buys tops      │                  │
│  └────────────────────────────────────────┘                  │
│                                                               │
│  REWARD 2: Sharpe Ratio Only                                 │
│  ┌────────────────────────────────────────┐                  │
│  │  Allocation ████████████████████████   │ ← Avoids        │
│  │  over time  ████████████████████████   │   anything      │
│  │             ████████████████████████   │   volatile      │
│  │  Result: Misses opportunities          │                  │
│  └────────────────────────────────────────┘                  │
│                                                               │
│  REWARD 3: Regret-Relative                                   │
│  ┌────────────────────────────────────────┐                  │
│  │  Allocation ██████████░░░░░░░░░░░░░░   │ ← Balances      │
│  │  over time  ░░░░██████████░░░░░░░░░░   │   exploration   │
│  │             ░░░░░░░░░░██████████░░░░   │   and           │
│  │  Result: Smooth tilts, learns well     │   exploitation  │
│  └────────────────────────────────────────┘                  │
│                                                               │
│  REWARD 4: Stability-Weighted                                │
│  ┌────────────────────────────────────────┐                  │
│  │  Allocation ████████████████░░░░░░░░   │ ← Slow, smooth  │
│  │  over time  ██████████████████░░░░░░   │   tilts         │
│  │             ░░░░████████████████████   │   No big        │
│  │  Result: Comfortable holds, low regret │   swings        │
│  └────────────────────────────────────────┘                  │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

## The Problem: Bad Rewards and What They Train

<div class="callout-warning">

**Warning:** A bandit system that optimizes the wrong reward function will converge efficiently to the wrong answer. Spend more time designing the reward function than choosing the algorithm.

</div>


### Bad Reward 1: Raw Returns


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def naive_reward(returns):
    """Worst possible reward: maximize last period's return."""
    return returns  # This is a disaster
```

</div>
</div>

**What it trains:**
- Buy whatever went up last week
- Sell whatever went down
- Classic "buy high, sell low" pattern

**Example behavior:**
- Week 1: WTI +5%, Gold -2% → Allocate 80% to WTI
- Week 2: WTI -8% (mean reversion) → Lose big
- Week 3: Pivot to Gold (which is now recovering)
- Result: Always late, always wrong

**When it might work:** Never. Seriously, never use raw returns.

### Bad Reward 2: Sharpe Ratio Only


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def sharpe_only_reward(returns, volatility):
    """Sounds smart, but creates risk-averse allocator."""
    return returns / (volatility + 1e-6)
```

</div>
</div>

**What it trains:**
- Avoid anything with high volatility
- Concentrate in low-vol assets
- Miss high-return opportunities

**Example behavior:**
- Gold: 12% annualized, 15% vol → Sharpe = 0.8
- NatGas: 18% annualized, 40% vol → Sharpe = 0.45
- Allocates heavily to Gold, misses NatGas gains

**When it might work:** If your only goal is minimize volatility, regardless of returns.

### Bad Reward 3: Win Rate Only


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def win_rate_reward(returns):
    """Counts how often you make money."""
    return 1.0 if returns > 0 else 0.0
```


**What it trains:**
- Take tiny gains quickly
- Let losses run (because a -10% and -1% both count as one loss)
- Death by a thousand cuts

**Example behavior:**
- Takes 100 small wins (+0.5% each) = +50%
- Takes 10 large losses (-5% each) = -50%
- Net: 0%, but win rate looks great at 90%

**When it might work:** If you're reporting to someone who only looks at win rate (fix your reporting instead).

### Bad Reward 4: Ignoring Transaction Costs

```python
def no_transaction_cost_reward(returns):
    """Forgets that trading isn't free."""
    return returns  # Missing the 5-10 bps per trade
```

**What it trains:**
- Rebalance constantly
- Churn the portfolio
- Give all profits to your broker

**Example behavior:**
- Theoretical gain: +2% per month
- Transaction costs: -1.5% per month (from excessive rebalancing)
- Net: +0.5% (would've been better with buy-and-hold)

## Good Reward Designs for Commodities

### Good Reward 1: Risk-Adjusted with Drawdown Penalty

```python
def risk_adjusted_reward(returns, volatility, drawdown, lambda_dd=2.0):
    """
    Penalize both volatility and drawdowns.

    Args:
        returns: Array of returns
        volatility: Standard deviation of returns
        drawdown: Maximum drawdown from peak
        lambda_dd: Drawdown penalty weight
    """
    sharpe = returns.mean() / (volatility + 1e-6)
    dd_penalty = lambda_dd * abs(drawdown)
    return sharpe - dd_penalty
```

**What it trains:**
- Seek positive returns
- Control volatility
- Avoid large drawdowns (sleeping well matters)

**Commodity example:**
- WTI: +2% avg, 5% vol, -3% max drawdown → Reward ≈ 0.4 - 2(0.03) = 0.34
- NatGas: +3% avg, 12% vol, -15% max drawdown → Reward ≈ 0.25 - 2(0.15) = -0.05

Result: Prefers WTI despite lower raw returns, because smoother path.

### Good Reward 2: Regret-Relative

```python
def regret_relative_reward(arm_returns, all_arm_returns):
    """
    Reward relative to best arm in hindsight.

    Args:
        arm_returns: Returns for chosen arm
        all_arm_returns: Returns for all arms
    """
    best_possible = all_arm_returns.max()
    regret = best_possible - arm_returns
    return -regret  # Minimize regret
```

**What it trains:**
- Learn which arms are consistently good
- Balance exploration (trying all arms) with exploitation
- Focus on relative performance, not absolute

**Commodity example:**
- Your allocation: Gold +1%, Copper +3%, WTI -2%
- Best arm this week: Copper +3%
- Your regret: 3% - (weighted avg) = small if you had Copper exposure

Result: Doesn't chase absolute returns, learns relative rankings.

### Good Reward 3: Stability-Weighted Returns

```python
def stability_weighted_reward(
    returns,
    allocation_change,
    lambda_churn=1.0
):
    """
    Reward returns but penalize excessive rebalancing.

    Args:
        returns: Portfolio returns
        allocation_change: L1 norm of allocation change
        lambda_churn: Penalty for turnover
    """
    return returns - lambda_churn * allocation_change
```

**What it trains:**
- Earn returns
- Minimize portfolio turnover
- Hold positions longer (transaction costs, taxes)

**Commodity example:**
- Week 1: Return +2%, allocation change 5% → Reward = 2% - 1(5%) = -3%
- Week 2: Return +1.5%, allocation change 1% → Reward = 1.5% - 1(1%) = +0.5%

Result: Prefers steady allocation over frequent changes.

### Good Reward 4: Thesis-Aligned

```python
def thesis_aligned_reward(
    returns,
    allocation,
    strategic_weights,
    lambda_alignment=0.5
):
    """
    Reward returns but stay close to strategic thesis.

    Args:
        returns: Portfolio returns
        allocation: Current allocation weights
        strategic_weights: Your fundamental view
        lambda_alignment: How much to penalize deviation
    """
    alignment_penalty = np.linalg.norm(
        allocation - strategic_weights
    )
    return returns - lambda_alignment * alignment_penalty
```

**What it trains:**
- Tilt toward your fundamental view
- Opportunistically deviate when justified
- Return to thesis when data is unclear

**Commodity example:**
- Strategic view: 40% Energy, 30% Metals, 30% Grains
- Current: 50% Energy, 25% Metals, 25% Grains
- Alignment penalty: L2 distance = 0.14
- If Energy returns justify the tilt, reward is positive

Result: Bandit as tactical overlay on strategic allocation.

### Good Reward 5: Multi-Objective

```python
def multi_objective_reward(
    returns,
    volatility,
    drawdown,
    turnover,
    weights=(0.4, 0.3, 0.2, 0.1)
):
    """
    Combine multiple objectives with explicit weights.

    Args:
        returns: Portfolio returns
        volatility: Return volatility
        drawdown: Maximum drawdown
        turnover: Portfolio turnover
        weights: (w_return, w_vol, w_dd, w_turnover)
    """
    w_ret, w_vol, w_dd, w_turn = weights

    # Normalize to [0, 1] scale
    return_score = returns / 0.10  # Normalize by 10% target
    vol_score = max(0, 1 - volatility / 0.20)  # Penalize vol > 20%
    dd_score = max(0, 1 - abs(drawdown) / 0.15)  # Penalize dd > 15%
    turnover_score = max(0, 1 - turnover / 0.30)  # Penalize >30% turnover

    return (
        w_ret * return_score +
        w_vol * vol_score +
        w_dd * dd_score +
        w_turn * turnover_score
    )
```

**What it trains:**
- Explicit tradeoff between competing objectives
- Customizable to your priorities
- Transparent scoring

**Commodity example:**
- Returns: +8% (score: 0.8)
- Volatility: 18% (score: 0.1)
- Drawdown: -10% (score: 0.33)
- Turnover: 15% (score: 0.5)
- Final: 0.4(0.8) + 0.3(0.1) + 0.2(0.33) + 0.1(0.5) = 0.48

Result: Balanced performance across all objectives.

## Commodity-Specific Considerations

### Contango/Backwardation in Reward

Commodities have term structure. Rolling futures contracts can add/subtract returns independent of spot price movement.

```python
def commodity_adjusted_reward(
    spot_returns,
    roll_yield,
    volatility
):
    """
    Account for futures roll yield in commodity returns.

    Args:
        spot_returns: Spot price change
        roll_yield: Gain/loss from rolling futures
        volatility: Return volatility
    """
    total_returns = spot_returns + roll_yield
    return total_returns / (volatility + 1e-6)
```

**Why it matters:**
- Contango (upward-sloping curve): Roll yield is negative
- Backwardation (downward-sloping curve): Roll yield is positive
- Ignoring this overweights contango commodities (bad)

### Seasonality in Reward

Agricultural commodities have strong seasonal patterns.

```python
def seasonal_adjusted_reward(
    returns,
    month,
    commodity,
    seasonal_patterns
):
    """
    Adjust reward based on known seasonality.

    Args:
        returns: Realized returns
        month: Current month (1-12)
        commodity: Commodity name
        seasonal_patterns: Dict of expected returns by month
    """
    expected = seasonal_patterns[commodity][month]
    surprise = returns - expected
    return surprise  # Reward beating seasonal expectation
```

**Why it matters:**
- Corn typically strong in summer (growing season risk)
- NatGas typically strong in winter (heating demand)
- Rewarding absolute returns misses context

### Inventory Dynamics

Commodity supply/demand imbalances show up in inventory levels.

```python
def inventory_aware_reward(
    returns,
    inventory_level,
    historical_range
):
    """
    Adjust reward based on inventory context.

    Args:
        returns: Realized returns
        inventory_level: Current inventory (e.g., EIA stocks)
        historical_range: (min, max) of historical inventory
    """
    # Normalize inventory to [0, 1]
    inv_min, inv_max = historical_range
    inv_pct = (inventory_level - inv_min) / (inv_max - inv_min)

    # Low inventory (< 20th percentile): Upside potential
    # High inventory (> 80th percentile): Downside risk
    if inv_pct < 0.2:
        return returns * 1.2  # Boost reward for low-inventory upside
    elif inv_pct > 0.8:
        return returns * 0.8  # Penalize high-inventory risk
    else:
        return returns
```

## Reward Function Design Checklist

When designing a reward for your commodity bandit:

- [ ] **Aligned with goal**: Does this measure what I actually want?
- [ ] **Bounded**: Can extreme values cause numerical issues?
- [ ] **Stationary**: Is the reward scale consistent over time?
- [ ] **Observable**: Can I compute this from available data?
- [ ] **Actionable**: Can the bandit actually influence this metric?
- [ ] **Robust**: Does it handle outliers and missing data?
- [ ] **Interpretable**: Can I explain this to stakeholders?

## Common Pitfalls

### Pitfall 1: Reward-Goal Mismatch
**Example:** Goal is steady accumulation, but reward is maximum returns.

**Fix:** Write down your real goal first, then design reward to match.

### Pitfall 2: Overfitting to Recent Data
**Example:** Market was trending for 6 months, so you design reward for trend-following. Then regime shifts.

**Fix:** Use regime-aware contextual bandits (see [Module 04](04_regime_aware_allocation.md)).

### Pitfall 3: Ignoring Non-Stationarity
**Example:** Volatility in 2019 (VIX ~15) vs 2020 (VIX ~30). Same reward scale doesn't work.

**Fix:** Normalize rewards by recent volatility or use rolling windows.

### Pitfall 4: Too Many Objectives
**Example:** Trying to maximize returns, minimize vol, minimize drawdown, minimize turnover, align with thesis, account for seasonality, and inventory.

**Fix:** Pick 2-3 primary objectives. Make hard constraints for the rest.

## Connections

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.



### Builds On
- **Module 2**: Regret analysis (regret-relative rewards)
- **Bayesian statistics**: Prior beliefs in reward design

### Leads To
- **Module 4**: Contextual bandits (regime-dependent rewards)
- **Guardrails**: Constraints that complement rewards

### Related Concepts
- **Utility theory**: Foundations of preference modeling
- **Portfolio optimization**: Modern Portfolio Theory objectives
- **Reinforcement learning**: Reward shaping in RL



## Practice Problems

### Problem 1: Design Your Reward
You're building a commodity accumulation strategy with these goals:
1. Accumulate 10,000 barrels of WTI equivalent over 12 months
2. Stay within $1M budget
3. Minimize regret vs equal-weight baseline
4. Sleep well (no >10% monthly drawdowns)

Design a reward function that captures all four objectives. Write it in Python.

### Problem 2: Diagnose the Failure
Your bandit allocator has been running for 3 months. Performance:
- Returns: +5% (vs market +8%)
- Volatility: 5% (vs market 12%)
- Max drawdown: -2% (vs market -6%)
- Portfolio turnover: 5% per week

Your reward was: `reward = returns / volatility`

Why did you underperform the market? What should you change?

### Problem 3: Seasonal Adjustment
You're trading agricultural commodities (Corn, Soybeans, Wheat). Historical data shows:
- Corn: Strong Apr-Aug (planting/growing), weak Sep-Mar
- Soybeans: Strong May-Sep, weak Oct-Apr
- Wheat: Strong Nov-Feb (winter wheat), weak Mar-Oct

Design a seasonal-adjusted reward function. How would you handle years when patterns break (e.g., drought)?

---

**Next Steps:**
- Read [Guardrails and Safety](03_guardrails_and_safety.md) to learn constraints that complement rewards
- Try [Reward Function Lab Notebook](../notebooks/02_reward_function_lab.ipynb) to see rewards in action


---

## Cross-References

<a class="link-card" href="./01_accumulator_bandit_playbook.md">
  <div class="link-card-title">01 Accumulator Bandit Playbook</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_accumulator_bandit_playbook.md">
  <div class="link-card-title">01 Accumulator Bandit Playbook — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_guardrails_and_safety.md">
  <div class="link-card-title">03 Guardrails And Safety</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_guardrails_and_safety.md">
  <div class="link-card-title">03 Guardrails And Safety — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./04_regime_aware_allocation.md">
  <div class="link-card-title">04 Regime Aware Allocation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./04_regime_aware_allocation.md">
  <div class="link-card-title">04 Regime Aware Allocation — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


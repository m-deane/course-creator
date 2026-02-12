# Project 2: Commodity Allocation Engine (Intermediate)

## What You'll Build

A production-ready two-wallet commodity allocator that separates stable core holdings (80%) from an adaptive "bandit sleeve" (20%) using Thompson Sampling. You'll design reward functions that capture risk-adjusted performance, implement safety guardrails to prevent self-sabotage, and backtest on real commodity data.

By the end, you'll have a deployable system that adaptively tilts toward better-performing commodities while maintaining diversification and safety constraints.

## Learning Goals

1. Implement the two-wallet framework for portfolio allocation
2. Design reward functions beyond simple returns (Sharpe ratio, regret-relative, risk-adjusted)
3. Build safety guardrails (position limits, tilt speed, minimum allocation)
4. Work with real commodity price data (yfinance)
5. Backtest a bandit strategy on historical data
6. Generate allocation reports for decision-making
7. Handle continuous rewards (not just Bernoulli)

## Requirements

```bash
pip install numpy pandas matplotlib scipy yfinance
```

- Python 3.11+
- numpy (array operations)
- pandas (data handling)
- matplotlib (visualization)
- scipy (Beta distribution)
- yfinance (commodity data)

## The Problem

You're managing a $100K commodity portfolio with exposure to 5 sectors:
- Energy: WTI Crude Oil
- Precious Metals: Gold
- Industrial Metals: Copper
- Energy: Natural Gas
- Agriculture: Corn

**Unknown:** Which commodities will perform best over the next year?

**Constraint:** You can't just "go all-in" on the best one — you need diversification, risk management, and the ability to adapt as markets change.

**Solution:** Two-wallet framework:
- **Core wallet (80% = $80K):** Equal-weight across all 5, rebalanced monthly
- **Bandit sleeve (20% = $20K):** Thompson Sampling allocates weekly based on recent performance

## Two-Wallet Framework

```
Total Portfolio: $100K
├── Core (80% = $80K)
│   ├── WTI: $16K (20%)
│   ├── Gold: $16K (20%)
│   ├── Copper: $16K (20%)
│   ├── NatGas: $16K (20%)
│   └── Corn: $16K (20%)
│   └── Rebalanced: Monthly
│
└── Bandit Sleeve (20% = $20K)
    ├── Allocation determined by Thompson Sampling
    ├── Based on recent risk-adjusted returns
    ├── Subject to guardrails:
    │   ├── Max 50% to any single commodity
    │   ├── Min 5% to all commodities
    │   └── Max 20% weekly change
    └── Rebalanced: Weekly
```

**Why this works:**
- Core provides stability and guarantees diversification
- Bandit sleeve learns and adapts without destroying the portfolio
- Guardrails prevent overfitting to recent noise

## Project Steps

### Step 1: Data Loading (15 minutes)

Load historical commodity data using yfinance. Tickers:
- `CL=F` (WTI Crude Oil)
- `GC=F` (Gold)
- `HG=F` (Copper)
- `NG=F` (Natural Gas)
- `ZC=F` (Corn)

Calculate weekly returns for backtesting.

**Provided:** Complete data loading code with synthetic fallback if yfinance fails.

### Step 2: Implement TwoWalletAllocator (30 minutes)

Complete the core allocation engine:

**Your tasks:**
1. `get_core_weights()` — return equal-weight allocation
2. `get_bandit_weights()` — Thompson Sampling with guardrails
3. `get_total_weights()` — combine core + bandit
4. `update()` — update posteriors based on observed returns

**Key insight:** Unlike Project 1 (Bernoulli), commodity returns are continuous Gaussian. You'll use Normal-Normal conjugacy instead of Beta-Bernoulli.

### Step 3: Design Reward Functions (20 minutes)

Implement 3 reward functions and choose the best for your goals:

1. **Raw returns** (baseline, not recommended)
   ```python
   reward = returns[arm]
   ```

2. **Sharpe ratio** (risk-adjusted, recommended)
   ```python
   reward = returns[arm] / volatility[arm]
   ```

3. **Regret-relative** (compared to best arm in hindsight)
   ```python
   reward = returns[arm] - max(returns)
   ```

**Your task:** Implement all three, then choose which to use based on your investment goals.

### Step 4: Add Safety Guardrails (25 minutes)

Implement 3 critical guardrails to prevent the bandit from self-destructing:

1. **Position limits:** No arm > 50% of bandit sleeve
2. **Minimum allocation:** No arm < 5% (prevents abandoning diversification)
3. **Tilt speed limits:** Allocation can't change more than 20% week-over-week

**Your task:** Modify `get_bandit_weights()` to enforce these constraints.

### Step 5: Backtest on Real Data (20 minutes)

Run a 52-week backtest:
- Start with $100K
- Each week:
  - Get allocation from TwoWalletAllocator
  - Observe realized returns
  - Update beliefs
  - Rebalance
- Track:
  - Weekly portfolio value
  - Allocation history
  - Cumulative returns
  - Sharpe ratio

**Provided:** Complete backtesting loop, you supply the allocator.

### Step 6: Generate Allocation Report (15 minutes)

Create a decision-making report showing:
1. Final portfolio value vs benchmarks
2. Allocation evolution over time
3. Risk metrics (volatility, max drawdown, Sharpe)
4. Regime adaptation (how bandit responded to market shifts)

**Provided:** Reporting and visualization functions.

## Expected Output

```
=== COMMODITY ALLOCATION ENGINE BACKTEST ===

Period: 2023-01-01 to 2024-01-01 (52 weeks)
Starting Capital: $100,000

PERFORMANCE SUMMARY
┌────────────────────────┬────────────┬──────────┬────────┐
│ Strategy               │ Final Val  │ Return   │ Sharpe │
├────────────────────────┼────────────┼──────────┼────────┤
│ Two-Wallet Bandit      │ $112,450   │ +12.45%  │  1.38  │
│ Equal Weight (core)    │ $108,200   │  +8.20%  │  1.12  │
│ Best Single (oracle)   │ $116,800   │ +16.80%  │  1.25  │
└────────────────────────┴────────────┴──────────┴────────┘

FINAL ALLOCATION
Core Wallet (80%):
  WTI:    $17,920 (20.0%)
  Gold:   $17,920 (20.0%)
  Copper: $17,920 (20.0%)
  NatGas: $17,920 (20.0%)
  Corn:   $17,920 (20.0%)

Bandit Sleeve (20%):
  WTI:    $ 4,860 (21.6%)  ← Tilted up
  Gold:   $ 6,735 (30.0%)  ← Highest
  Copper: $ 5,390 (24.0%)
  NatGas: $ 2,245 (10.0%)  ← Tilted down
  Corn:   $ 3,245 (14.4%)

RISK METRICS
  Volatility: 12.3% annualized
  Max Drawdown: -8.4%
  Win Rate: 57.7% (30/52 weeks positive)

✅ Bandit sleeve improved returns by 4.25% vs pure equal-weight
✅ Maintained diversification (no position > 30% total portfolio)
✅ Adapted to regime changes (Gold tilt increased during volatility spike)
```

## Success Criteria

Your implementation succeeds if:

1. **Outperforms equal-weight:** Bandit beats pure core allocation by 2%+
2. **Maintains safety:** No weekly drawdown exceeds 15%
3. **Reasonable Sharpe:** Risk-adjusted return (Sharpe) > 1.0
4. **Adapts to regimes:** Allocation changes when market conditions shift
5. **Guardrails work:** No constraint violations (position limits, tilt speed, etc.)

## File Structure

```
project_2_intermediate/
├── README.md              # This file
├── starter_code.py        # Your working skeleton (complete TODOs)
└── solution.py            # Reference implementation
```

## Common Pitfalls to Avoid

1. **Using raw returns as reward:** Leads to chasing volatility. Use Sharpe or risk-adjusted returns.

2. **No minimum allocation:** One bad week → zero out an arm → miss its rebound. Always maintain 5%+ in all arms.

3. **Bandit sleeve too large:** Start with 20% or less. Large sleeve = high turnover and risk.

4. **Ignoring transaction costs:** Real trading has costs. Add a small penalty (0.1% per trade) to reward function.

5. **No regime awareness:** Markets change. Consider adding volatility-based dampening (reduce bandit when VIX spikes).

## Extensions (Optional)

1. **Contextual features:** Add VIX, term structure, seasonal indicators as context
2. **Volatility dampening:** Reduce bandit sleeve during high-volatility regimes
3. **Transaction costs:** Penalize excessive rebalancing
4. **Multi-period optimization:** Optimize over rolling 4-week windows, not just 1-week
5. **Out-of-sample testing:** Train on 2022, test on 2023

## Next Steps

After completing this project:
- **Project 3:** Add regime detection, contextual features, and full production system
- **Module 6:** Learn non-stationary bandits for changing markets
- **Module 7:** Deploy with monitoring and alerting

## Key Lessons for Commodity Trading

1. **Risk-adjusted rewards matter:** Raw returns optimize for volatility, not skill
2. **Diversification is non-negotiable:** Core wallet prevents catastrophic losses
3. **Guardrails prevent self-sabotage:** Constraints save you from overfitting
4. **Adaptation beats prediction:** You don't need to predict oil prices — just adapt faster than the market

---

**Remember:** The goal isn't to pick the single best commodity. It's to tilt toward better performers while protecting against being wrong.

# Module 5: Bandits for Commodity Trading — Adaptive Allocation Under Uncertainty

## Overview

This is where theory meets practice. Apply multi-armed bandits to real commodity trading problems.

**The Key Shift:** Stop trying to predict which commodity will outperform. Instead, build a system that learns optimal allocation through repeated small experiments.

You'll learn:
- The **two-wallet framework** (stable core + adaptive bandit sleeve)
- **Reward function design** for trading (not just returns)
- **Regime-aware allocation** with contextual bandits
- **Guardrails** that prevent self-sabotage

By the end, you'll have a complete, deployable commodity allocation system.

## Learning Objectives

By completing this module, you will:

1. **Implement the two-wallet framework** separating core holdings from adaptive tilts
2. **Design reward functions** that align with real trading goals (not just maximize returns)
3. **Build regime-aware allocators** using contextual bandits with commodity market features
4. **Add production guardrails** that prevent concentration risk and overtrading
5. **Deploy a complete system** that learns optimal commodity allocation from real market data

## Why This Matters

Traditional commodity allocation uses:
- Static weights (equal-weight, market-cap)
- Optimization based on past correlations (breaks when regimes shift)
- Discretionary tilts (hard to systematize)

Bandit-based allocation:
- Adapts to changing market conditions
- Learns from realized outcomes, not predictions
- Quantifies exploration vs exploitation tradeoff
- Works when correlations are non-stationary

## Module Contents

### Concept Guides

1. **[Accumulator Bandit Playbook](guides/01_accumulator_bandit_playbook.md)**
   - The 6-step system from research to production
   - Two-wallet framework architecture
   - Connection to Bayesian commodity forecasting

2. **[Reward Design for Commodities](guides/02_reward_design_commodities.md)**
   - Why reward function is your most important decision
   - Bad rewards and what they train
   - Risk-adjusted, regret-aware, stability-weighted designs

3. **[Guardrails and Safety](guides/03_guardrails_and_safety.md)**
   - Position limits, tilt speed caps, volatility dampening
   - Why these aren't weakness — they're system protection
   - Commodity-specific guardrails

4. **[Regime-Aware Allocation](guides/04_regime_aware_allocation.md)**
   - Market regimes as contextual features
   - Feature engineering for commodities
   - Connection to Hidden Markov Models course

5. **[Cheatsheet](guides/cheatsheet.md)**
   - Quick reference for all key concepts
   - Decision flowcharts and lookup tables

### Interactive Notebooks

1. **[Two-Wallet Framework](notebooks/01_two_wallet_framework.ipynb)** (15 min)
   - Build core + bandit allocation system
   - Backtest on real commodity futures (WTI, Gold, Copper, NatGas, Corn)
   - See adaptive tilts in action

2. **[Reward Function Lab](notebooks/02_reward_function_lab.ipynb)** (15 min)
   - Test 4 different reward designs side-by-side
   - Observe how rewards shape allocator behavior
   - Design your own custom reward

3. **[Regime-Aware Commodity Bandit](notebooks/03_regime_commodity_bandit.ipynb)** (15 min)
   - Add market regime detection
   - Build contextual bandit that adapts to regimes
   - Deploy full system with guardrails

### Practice

- **[Self-Check Exercises](exercises/exercises.py)**
  - Custom reward functions
  - Correlation guardrails
  - Seasonal-aware bandits

### Resources

- **[Additional Readings](resources/additional_readings.md)**
  - Academic papers on bandit portfolio selection
  - Commodity trading systems literature
  - Connections to other courses in this repo

## Completion Criteria

You've mastered this module when you can:

- [ ] Explain why the two-wallet framework prevents overtrading
- [ ] Design a reward function aligned with your trading goal
- [ ] Implement at least 3 guardrails from scratch
- [ ] Build a regime-aware contextual bandit for commodity allocation
- [ ] Run a backtest showing adaptive allocation vs static baseline
- [ ] Identify when your reward function is misaligned with your goal

## Prerequisite Knowledge

Required:
- Module 1: Thompson Sampling basics
- Module 2: Regret analysis
- Module 3: Contextual bandits

Recommended:
- Basic understanding of commodities (futures, term structure)
- Python data analysis (pandas, numpy)

## Time Estimate

- Concept guides: 60 minutes
- Notebooks: 45 minutes (3 × 15 min)
- Exercises: 30 minutes
- **Total: ~2.5 hours**

## What's Next

After completing this module:
- **Module 6**: Comparison to A/B testing (when to use which)
- **Project 2**: Build your own commodity allocation system
- **Bonus**: Connect to Bayesian Commodity Forecasting course for enhanced regime detection

## Real-World Application

This module teaches the exact system used by:
- Systematic commodity trading desks
- Multi-strategy hedge funds (commodity sleeve)
- Institutional allocators experimenting with alternative weightings

The two-wallet framework with guardrails is production-ready. You could deploy this.

---

**Start with:** [Accumulator Bandit Playbook](guides/01_accumulator_bandit_playbook.md) or jump straight to [Two-Wallet Framework Notebook](notebooks/01_two_wallet_framework.ipynb) if you learn by doing.

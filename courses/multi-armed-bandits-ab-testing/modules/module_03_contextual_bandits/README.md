# Module 3: Contextual Bandits — Decisions That Use Features

## Overview

Regular bandits treat every decision the same way, regardless of the situation. You pick an arm, observe a reward, update your beliefs. But in real markets, context matters enormously. Is the market in contango or backwardation? Is volatility spiking or stable? Are we in harvest season or planting season?

**Contextual bandits** incorporate observable features (context) before making decisions. Instead of learning "Energy is the best arm," you learn "Energy is best when volatility is low and we're in contango, but Agriculture wins during harvest season with high inventory surprise."

This module bridges the gap between simple bandits and full reinforcement learning. You'll learn to build regime-aware allocation systems that adapt dynamically to market conditions.

## Why Contextual Bandits Matter for Commodity Trading

1. **Regime-dependent performance** — Strategies that work in contango fail in backwardation; contextual bandits detect and exploit this
2. **Feature-rich environments** — Commodity markets provide abundant context: term structure, inventory levels, seasonality, macro indicators
3. **Non-stationary adaptation** — When "the best arm" changes with market conditions, context helps you track what's driving performance
4. **Personalization at scale** — Just as news recommendation personalizes to users, commodity allocation personalizes to market regimes

## Learning Objectives

By completing this module, you will be able to:

1. **Explain** the fundamental difference between standard and contextual bandits
2. **Implement** the LinUCB algorithm from scratch with ridge regression and confidence bounds
3. **Engineer** effective context features for commodity markets (volatility regimes, term structure, seasonality)
4. **Build** a regime-aware commodity allocator that outperforms static allocation
5. **Evaluate** when contextual bandits are worth the added complexity versus simpler approaches

## Module Contents

### Guides
- `01_contextual_bandit_framework.md` — The core framework: context → decision → reward
- `02_linucb_algorithm.md` — LinUCB deep dive: linear models meet exploration
- `03_feature_engineering_bandits.md` — Building context features for commodity markets
- `cheatsheet.md` — Quick reference for contextual bandit concepts and code

### Notebooks
- `01_contextual_bandit_intro.ipynb` — Why context matters: see standard bandits fail in regime-dependent environments (15 min)
- `02_linucb_implementation.ipynb` — Build LinUCB from scratch and watch it learn (15 min)
- `03_commodity_regime_bandit.ipynb` — Real commodity allocation with regime features (15 min)

### Exercises
- `exercises.py` — Self-check exercises: contextual ε-greedy, feature engineering, signal routing

### Resources
- `additional_readings.md` — Papers, references, and deeper dives
- `figures/` — Diagrams and visual assets

## Completion Criteria

You've mastered this module when you can:

- [ ] Explain why a standard bandit will underperform in a regime-dependent environment
- [ ] Implement LinUCB with proper ridge regression and UCB exploration
- [ ] Extract and normalize context features from real commodity market data
- [ ] Build a backtest comparing contextual vs non-contextual bandits
- [ ] Articulate the bias-variance tradeoff in contextual bandit feature engineering

## Key Insight

**Standard bandits ask:** "Which arm is best on average?"

**Contextual bandits ask:** "Which arm is best *in this situation*?"

The difference is transformative. A regime-aware commodity allocator can switch between energy (in stable, low-vol markets) and metals (in defensive, high-vol regimes) automatically. The bandit learns the mapping from context to optimal decision through experience.

## Prerequisites

Before starting this module, you should:
- Understand basic bandit algorithms (Module 1: epsilon-greedy, UCB)
- Be comfortable with linear regression and ridge regression
- Have basic feature engineering intuition (scaling, normalization)
- Understand the explore-exploit tradeoff

## Estimated Time

- Reading guides: 45 minutes
- Working through notebooks: 45 minutes
- Exercises: 30 minutes
- **Total: ~2 hours**

## What's Next

After completing this module:
- **Module 4:** Apply contextual bandits to content and growth optimization
- **Module 5:** Build production commodity trading systems with contextual allocation
- **Module 6:** Explore non-stationary bandits and change detection
- **Project 2 (Intermediate):** Regime-aware portfolio tilting system

---

*"The market doesn't care what worked last month. It cares about the situation today. Contextual bandits learn that mapping."*

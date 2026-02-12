# Module 6: Advanced Topics — When the World Moves Under Your Feet

## Overview

In the real world, the bandit problem gets harder. Commodity markets don't sit still — regime shifts happen (COVID crash, energy crisis, inflation spikes), seasonal patterns emerge and evolve, and the competitive landscape changes as other traders adapt to strategies that work. This module equips you with algorithms designed for these non-stationary, adversarial, and restless environments.

**The Core Challenge:** Standard bandits assume rewards are stationary (oil's expected return today equals its expected return tomorrow). But in commodities, regime changes are the rule, not the exception. You need algorithms that forget outdated information, detect structural breaks, and adapt quickly.

**What You'll Learn:**
- Non-stationary bandits that weight recent observations more heavily
- Change-point detection to restart exploration when markets shift
- Restless bandits where unselected options still evolve
- Adversarial bandits for game-theoretic trading environments

**Commodity Context:**
- Handling seasonal regime shifts in agricultural commodities
- Adapting to energy market disruptions (supply shocks, geopolitical events)
- Dealing with market impact when your strategy moves prices

## Learning Objectives

By the end of this module, you will be able to:

1. **Implement non-stationary bandit algorithms** (Discounted Thompson Sampling, Sliding-Window UCB) and explain when to use each
2. **Build change-point detection systems** that trigger re-exploration when commodity regimes shift
3. **Apply restless bandit concepts** to scenarios where asset characteristics evolve independently of your actions
4. **Recognize adversarial environments** and apply EXP3 when market participants react to your strategy
5. **Compare algorithm performance** across regime-change scenarios using real commodity data

## Module Contents

### Guides
1. **Non-Stationary Bandits** — Algorithms that adapt when rewards change over time
2. **Restless Bandits** — When unselected options evolve on their own
3. **Adversarial Bandits** — Game-theoretic approaches for competitive environments
4. **Cheatsheet** — Quick reference for algorithm selection

### Notebooks (15 minutes each)
1. **Non-Stationary Bandits** — Compare standard vs adaptive algorithms on shifting rewards
2. **Change Detection** — Build a CUSUM detector that restarts exploration when regimes change
3. **Commodity Regime Shifts** — Apply adaptive bandits to real commodity data with historical shocks

### Exercises
- Implement Discounted Thompson Sampling with tunable decay
- Build CUSUM-based bandit restarter
- Compare algorithms on seasonal commodity patterns

### Resources
- Research papers on non-stationary bandits
- Regime detection methods for financial markets
- Case studies of commodity market structural breaks

## Prerequisites

Before starting this module, you should:
- Understand standard MAB algorithms (Thompson Sampling, UCB) from Module 2
- Be comfortable with Python data structures and NumPy
- Have completed Module 3 (Contextual Bandits) or equivalent experience

## Completion Criteria

You've mastered this module when you can:

- [ ] Explain why standard bandits fail in non-stationary environments
- [ ] Implement at least two non-stationary bandit variants
- [ ] Build a change-point detector and integrate it with a bandit
- [ ] Identify real-world scenarios requiring adversarial vs stochastic approaches
- [ ] Apply adaptive bandits to commodity data with regime shifts
- [ ] Choose the right algorithm for a given non-stationarity pattern

## Time Commitment

- **Guides:** 45-60 minutes total
- **Notebooks:** 45 minutes (3 × 15 min)
- **Exercises:** 30-45 minutes
- **Total:** ~2.5-3 hours

## Key Takeaway

**In commodity trading, non-stationarity isn't the exception — it's the rule.** Markets shift regimes, seasonal patterns evolve, and competitive dynamics change. The algorithms in this module help you stay adaptive when the world moves under your feet.

## What's Next?

After completing this module:
- **Module 7: Production Systems** — Deploy these algorithms in live trading infrastructure
- **Projects** — Build a regime-adaptive commodity portfolio selector
- **Advanced Topics** — Combine contextual bandits with non-stationarity detection

## Getting Help

- Check the cheatsheet for quick algorithm selection guidance
- Review visualizations in the guides to build intuition
- Run notebooks multiple times with different parameters
- Consult `resources/additional_readings.md` for theoretical depth

# Module 2: Bayesian Bandits — Thompson Sampling & Posterior-Guided Decisions

## Overview

Thompson Sampling is the Bayesian approach to multi-armed bandits: maintain a probability distribution representing your belief about each arm's reward, sample from these beliefs, then act on those samples. It's mathematically elegant, empirically effective, and connects directly to Bayesian inference.

Unlike UCB's deterministic optimism, Thompson Sampling explores through randomness guided by uncertainty. It naturally balances exploration (sampling widely when uncertain) and exploitation (sampling near the mode when confident), all through a simple algorithm: sample a plausible reward for each arm, pick the best sample, update your beliefs.

This module connects to the **Bayesian Commodity Forecasting** course — the same posterior updating logic that powers probabilistic price forecasts also drives adaptive allocation. You'll build Thompson Sampling from scratch, watch beliefs converge to truth, and apply it to commodity portfolio decisions.

By the end of this module, you'll understand why Thompson Sampling is the go-to algorithm for practitioners who need both strong performance and natural handling of delayed feedback, batched updates, and non-stationary environments.

## Learning Objectives

After completing this module, you will be able to:

- **Implement Thompson Sampling** for Bernoulli and Gaussian rewards from scratch
- **Explain posterior updating** with conjugate priors (Beta-Bernoulli, Normal-Normal)
- **Visualize belief evolution** as data accumulates and posteriors concentrate
- **Compare Thompson Sampling to UCB** in terms of exploration patterns and practical trade-offs
- **Apply Bayesian bandits** to commodity allocation with real market data
- **Choose appropriate priors** and understand their impact on convergence speed

## Module Contents

### Concept Guides
- **01_thompson_sampling.md** — Beta-Bernoulli Thompson Sampling algorithm and intuition
- **02_posterior_updating.md** — Conjugate priors and Bayesian belief updates
- **03_thompson_vs_ucb.md** — Theoretical and practical comparison of the two leading algorithms
- **cheatsheet.md** — Quick reference for Thompson Sampling implementations

### Interactive Notebooks
- **01_thompson_sampling_from_scratch.ipynb** — Build Thompson Sampling in 10 lines (15 min)
- **02_belief_evolution.ipynb** — Watch your beliefs converge to truth (15 min)
- **03_gaussian_thompson_commodities.ipynb** — Apply to real commodity allocation (15 min)

### Self-Check Exercises
- **exercises.py** — Implement Poisson Thompson Sampling, compare prior strengths, batched updates (ungraded)

### Supporting Resources
- **additional_readings.md** — Key papers and connections to Bayesian Commodity Forecasting course
- **figures/** — Diagrams for posterior evolution and Beta distributions

## Completion Criteria

You're ready to move to Module 3 (Contextual Bandits) when you can:

1. ✅ Implement Thompson Sampling for Bernoulli and Gaussian rewards without looking at notes
2. ✅ Explain why sampling from the posterior naturally balances exploration and exploitation
3. ✅ Update Beta and Normal posteriors given new observations
4. ✅ Articulate when to use Thompson Sampling vs UCB in practical applications
5. ✅ Run all three notebooks and explain why beliefs concentrate over time

## Time Estimate

- **Quick path** (just notebooks): 45 minutes
- **Full path** (guides + notebooks + exercises): 2-3 hours
- **Deep dive** (everything + additional readings): 4-5 hours

## What's Next?

Module 3 introduces **Contextual Bandits** — decisions that depend on context features like volatility regime, term structure, and seasonality. You'll learn LinUCB and extend Thompson Sampling to incorporate market conditions, enabling adaptive commodity strategies that respond to changing environments.

## Commodity Trading Connection

Thompson Sampling is ideal for commodity allocation because:
- **Handles noisy signals** — Posteriors naturally integrate evidence strength
- **Adapts to regime changes** — Discount priors to allow faster adaptation
- **Works with delayed feedback** — Batch updates fit weekly/monthly rebalancing
- **Provides uncertainty estimates** — Posterior credible intervals guide risk management

Think of each commodity as an arm, weekly returns as rewards, and Thompson Sampling as your adaptive portfolio manager that learns which assets work in current conditions.

# Module 0: Foundations — Why A/B Testing Isn't Enough

## Overview

Traditional A/B testing is the gold standard for controlled experiments, but it has a critical flaw: it wastes traffic on inferior options while collecting evidence. In commodity trading, where market conditions shift constantly and every allocation decision matters, this inefficiency can be costly.

This module introduces the fundamental problem that multi-armed bandits solve: **how to learn AND earn at the same time**. You'll discover why static A/B tests break down in non-stationary environments (like commodity markets), understand the explore-exploit tradeoff that drives all sequential decision making, and build intuition for decision theory under uncertainty.

By the end of this module, you'll see why adaptive allocation strategies outperform fixed experiments when you're trading live capital, not just running academic studies.

## Learning Objectives

After completing this module, you will be able to:

- **Identify when A/B testing wastes resources** and quantify the opportunity cost of static allocation
- **Explain the explore-exploit tradeoff** using formal regret definitions and real-world examples
- **Calculate cumulative regret** for different allocation strategies in sequential decision problems
- **Apply decision theory basics** to commodity trading scenarios with uncertain payoffs
- **Recognize non-stationary environments** where traditional testing methodologies fail

## Module Contents

### Concept Guides
- **01_ab_testing_limits.md** — Why traditional A/B testing leaves money on the table
- **02_explore_exploit_tradeoff.md** — The fundamental tension in sequential decision making
- **03_decision_theory_basics.md** — Expected value, utility, and regret minimization
- **cheatsheet.md** — One-page quick reference for all key concepts

### Interactive Notebooks
- **01_ab_test_simulation.ipynb** — See cumulative regret grow as A/B tests waste traffic (15 min)
- **02_explore_exploit_interactive.ipynb** — Compare exploration strategies with live controls (15 min)
- **03_commodity_decision_lab.ipynb** — Apply decision theory to real commodity price data (15 min)

### Self-Check Exercises
- **exercises.py** — Practice calculations and implement core concepts (ungraded)

### Supporting Resources
- **additional_readings.md** — Curated papers, books, and blog posts
- **figures/** — Visual assets and diagrams

## Completion Criteria

You're ready to move to Module 1 when you can:

1. ✅ Explain why A/B testing is inefficient for sequential decisions (not just "it's slow")
2. ✅ Calculate cumulative regret for a given sequence of actions
3. ✅ Implement a basic epsilon-greedy strategy from scratch
4. ✅ Articulate the explore-exploit tradeoff in your own words using a real example
5. ✅ Run all three notebooks and modify parameters to see how behavior changes

## Time Estimate

- **Quick path** (just notebooks): 45 minutes
- **Full path** (guides + notebooks + exercises): 2-3 hours
- **Deep dive** (everything + additional readings): 4-5 hours

## What's Next?

Module 1 introduces the **epsilon-greedy algorithm** — the simplest bandit strategy that balances exploration and exploitation with a single parameter. You'll implement it from scratch and apply it to commodity sector rotation.

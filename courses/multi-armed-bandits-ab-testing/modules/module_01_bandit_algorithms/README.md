# Module 1: Core Bandit Algorithms — Your Decision-Making Toolkit

## Overview

Learn to implement and deploy the fundamental multi-armed bandit algorithms that power real-world decision systems. You'll build epsilon-greedy, UCB1, and softmax exploration from scratch, understand their theoretical guarantees, and benchmark them on commodity trading scenarios.

**Time Commitment:** 2-3 hours
**Prerequisites:** Basic Python, NumPy, understanding of mean/variance
**Difficulty:** Beginner to Intermediate

## What You'll Build

By the end of this module, you'll have working implementations of:
- Epsilon-greedy bandit with decaying exploration schedules
- Upper Confidence Bound (UCB1) algorithm
- Softmax/Boltzmann exploration
- A comparative benchmarking framework for algorithm selection

## Learning Objectives

After completing this module, you will be able to:

1. **Implement** epsilon-greedy, UCB1, and softmax algorithms from scratch in <20 lines each
2. **Explain** the exploration-exploitation tradeoff and how each algorithm addresses it
3. **Analyze** regret bounds and understand O(ε*T + K/ε) for ε-greedy, O(√T log T) for UCB1
4. **Compare** algorithms empirically on commodity allocation problems
5. **Select** the appropriate algorithm for different problem characteristics
6. **Deploy** bandit algorithms for real-world commodity trading decisions

## Module Contents

### Concept Guides
Complete guides with theory, visuals, and code:
- **[Epsilon-Greedy Algorithm](guides/01_epsilon_greedy.md)** - The simplest exploration strategy that works
- **[Upper Confidence Bound (UCB1)](guides/02_upper_confidence_bound.md)** - Optimism in the face of uncertainty
- **[Softmax/Boltzmann Exploration](guides/03_softmax_boltzmann.md)** - Probabilistic action selection
- **[Cheatsheet](guides/cheatsheet.md)** - Quick reference for all algorithms

### Interactive Notebooks
15-minute hands-on implementations:
- **[01: Epsilon-Greedy from Scratch](notebooks/01_epsilon_greedy_from_scratch.ipynb)** - Build and test in 5 lines
- **[02: UCB Exploration](notebooks/02_ucb_exploration.ipynb)** - Confidence bounds in action
- **[03: Algorithm Shootout](notebooks/03_algorithm_shootout.ipynb)** - Compare all three on real commodity data

### Exercises
Self-check exercises with instant feedback:
- **[Exercises](exercises/exercises.py)** - Implement variants, run tournaments, test edge cases

### Additional Resources
- **[Reading List](resources/additional_readings.md)** - Papers, books, and blog posts
- **[Figures](resources/figures/)** - All diagrams and visualizations

## Completion Criteria

You've mastered this module when you can:

- [ ] Implement all three core algorithms (ε-greedy, UCB1, softmax) from memory in <20 lines each
- [ ] Explain when each algorithm excels and when it fails
- [ ] Run a comparative benchmark and interpret the regret curves
- [ ] Tune hyperparameters (ε, c, τ) for a specific problem
- [ ] Choose the right algorithm for a new commodity allocation problem
- [ ] Explain the theoretical regret bounds intuitively (no proof required)

## Practical Skills Checklist

After this module, you should be able to:

- [ ] Replace a random allocation strategy with a bandit algorithm
- [ ] Implement decaying exploration schedules (ε = 1/√t)
- [ ] Visualize confidence bounds and exploration patterns
- [ ] Debug poor performance (too much/too little exploration)
- [ ] Adapt algorithms to non-stationary reward distributions

## How to Use This Module

### If you're a **Quick Learner** (1 hour):
1. Read the [Cheatsheet](guides/cheatsheet.md) (5 min)
2. Run [Algorithm Shootout notebook](notebooks/03_algorithm_shootout.ipynb) (15 min)
3. Complete exercises 1-3 (30 min)
4. Skim the concept guides for details as needed

### If you're **Building Intuition** (2-3 hours):
1. Read [Epsilon-Greedy Guide](guides/01_epsilon_greedy.md) (20 min)
2. Work through [Epsilon-Greedy Notebook](notebooks/01_epsilon_greedy_from_scratch.ipynb) (15 min)
3. Read [UCB Guide](guides/02_upper_confidence_bound.md) (20 min)
4. Work through [UCB Notebook](notebooks/02_ucb_exploration.ipynb) (15 min)
5. Read [Softmax Guide](guides/03_softmax_boltzmann.md) (15 min)
6. Compare all three in [Algorithm Shootout](notebooks/03_algorithm_shootout.ipynb) (20 min)
7. Complete all exercises (30 min)
8. Review [Cheatsheet](guides/cheatsheet.md) (5 min)

### If you're **Going Deep** (4+ hours):
- Follow the "Building Intuition" path above
- Read all papers in [Additional Readings](resources/additional_readings.md)
- Implement your own variants (Bayes-UCB, Thompson Sampling preview)
- Apply to your own dataset

## Key Insights from This Module

> **The Central Tradeoff:** You must explore unknown options to discover better rewards, but exploration costs you immediate gains. Every bandit algorithm is a different answer to "how much should I explore?"

> **No Free Lunch:** No algorithm dominates everywhere. Epsilon-greedy is simple but requires tuning ε. UCB1 is parameter-free but assumes bounded rewards. Softmax is smooth but sensitive to temperature.

> **Theory Meets Practice:** Regret bounds tell you asymptotic guarantees (what happens as T→∞), but finite-sample performance depends on your specific reward distributions.

## Real-World Application: Commodity Portfolio Allocation

Throughout this module, you'll see examples like:
- **Energy basket selection:** Which of {WTI, Brent, Natural Gas, Heating Oil} to overweight this week?
- **Cross-commodity hedging:** Which instrument {futures, options, swaps} minimizes portfolio variance?
- **Regime-adaptive allocation:** How to handle structural breaks (e.g., Russia-Ukraine war shifting gas markets)

These aren't toy problems—they're the actual decisions commodity traders face daily.

## What's Next?

After mastering core algorithms, you'll move to:
- **Module 2: Contextual Bandits** - Incorporate market features (volatility, correlation, sentiment)
- **Module 3: Thompson Sampling** - Bayesian approach to exploration
- **Module 4: A/B Testing** - Apply bandits to controlled experiments

## Get Started

Open [notebooks/01_epsilon_greedy_from_scratch.ipynb](notebooks/01_epsilon_greedy_from_scratch.ipynb) and build your first bandit in 2 minutes.

# Module 00: State Space Models & Kalman Filter

## Overview

Master the foundational framework for dynamic factor models by building working state space models from scratch. In 2 hours, you'll implement a Kalman filter, visualize hidden states, and understand why this approach dominates modern time series econometrics.

**Time Commitment:** 2 hours
**Difficulty:** Intermediate
**Prerequisites:** Linear algebra, Python basics, time series fundamentals

## Why This Matters

State space models are the engine behind:
- Central bank nowcasting systems (Fed, ECB, BoE)
- High-frequency trading algorithms
- Macroeconomic forecasting platforms
- Sensor fusion in robotics (yes, same math!)

The Kalman filter is arguably the most important algorithm in applied time series analysis. You'll implement it from first principles.

## Learning Objectives

By the end of this module, you will:

1. **Build** a state space model for any time series problem
2. **Implement** the Kalman filter in < 50 lines of NumPy
3. **Visualize** how hidden states emerge from noisy observations
4. **Diagnose** model fit using innovation statistics
5. **Extend** to multivariate systems

## Module Contents

### Guides (Read First)
1. **[State Space Models](guides/01_state_space_models.md)** - The unifying framework (20 min)
2. **[Kalman Filter](guides/02_kalman_filter.md)** - Optimal filtering in linear systems (25 min)
3. **[Cheatsheet](guides/cheatsheet.md)** - Quick reference for formulas and code patterns

### Notebooks (Hands-On)
1. **[State Space Introduction](notebooks/01_state_space_intro.ipynb)** - Build your first state space model (15 min)
2. **[Kalman Filter Visualization](notebooks/02_kalman_filter_visual.ipynb)** - Interactive filtering demo (15 min)

### Practice
- **[Self-Check Exercises](exercises/exercises.py)** - Test your understanding (ungraded)

### Resources
- **[Additional Readings](resources/additional_readings.md)** - Papers, books, and tutorials
- **[Figures](resources/figures/)** - Visual assets and diagrams

## Recommended Path

### Fast Track (1 hour)
1. Skim [State Space Models guide](guides/01_state_space_models.md)
2. Run [State Space Introduction notebook](notebooks/01_state_space_intro.ipynb)
3. Run [Kalman Filter Visualization notebook](notebooks/02_kalman_filter_visual.ipynb)
4. Check [Cheatsheet](guides/cheatsheet.md)

### Deep Dive (2-3 hours)
1. Read both guides thoroughly
2. Work through both notebooks, modifying parameters
3. Complete self-check exercises
4. Read 2-3 additional resources
5. Implement Kalman filter from scratch without looking at code

### Portfolio Extension
Build a nowcasting system for GDP using state space models (see Module 03).

## Key Concepts

- **State Space Representation** - Latent dynamics + observation mapping
- **Kalman Filter** - Recursive Bayesian estimation for linear Gaussian systems
- **Innovations** - One-step-ahead forecast errors (white noise if model is correct)
- **Filtering vs Smoothing** - Real-time vs retrospective estimation

## Common Questions

**Q: Why not just use ARIMA?**
A: State space models handle missing data naturally, accommodate mixed frequencies, and extend to multivariate systems more elegantly. They're also the foundation for dynamic factor models.

**Q: Is the Kalman filter hard to implement?**
A: No! The core algorithm is ~40 lines of NumPy. The math looks intimidating but the code is straightforward.

**Q: Do I need to understand Bayesian statistics?**
A: Not yet. The Kalman filter can be derived from classical or Bayesian perspectives. We focus on the algorithmic implementation first.

## Next Steps

After completing this module:
- **Module 01:** Learn how dynamic factor models extend state space models to high-dimensional data
- **Module 02:** Estimate DFM parameters using maximum likelihood and EM algorithm
- **Project 1:** Build a real-time nowcasting dashboard

## Getting Help

- Check [Common Pitfalls](guides/01_state_space_models.md#common-pitfalls) sections in guides
- Review [Cheatsheet](guides/cheatsheet.md) for quick debugging
- Compare your results to notebook solutions
- Consult [Additional Readings](resources/additional_readings.md) for deeper explanations

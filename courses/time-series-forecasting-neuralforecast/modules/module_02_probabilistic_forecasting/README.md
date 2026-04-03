# Module 02: Probabilistic Forecasting — Why Quantiles Aren't Enough

## The Central Insight

Prediction intervals describe uncertainty at a **single point in time**. They cannot describe how uncertainty unfolds across multiple periods together. The 80th percentile of Monday's demand plus the 80th percentile of Tuesday's demand is **not** the 80th percentile of the two-day total.

This module builds the case for why marginal quantile forecasts are necessary but not sufficient — and sets up Module 3's sample paths as the correct solution.

## Learning Objectives

By the end of this module you will be able to:

1. Distinguish marginal distributions $F_t(y_t)$ from joint distributions $F_{1:H}(y_1, \ldots, y_H)$
2. Explain why the sum of per-step quantiles overestimates the quantile of the sum
3. Train NHITS with MQLoss and interpret all output columns
4. Compute and visualize prediction intervals (fan charts) for a 7-day bakery forecast
5. Verify that the sum-of-quantiles error is substantial, universal, and grows with the horizon
6. Explain why MQLoss produces marginals (the loss sums independently across time steps)

## Contents

```
guides/
    01_marginal_vs_joint.md           Core concept: marginal vs joint distributions
    01_marginal_vs_joint_slides.md    Companion slide deck (15 slides)
    02_training_quantile_models.md    MQLoss mechanics: pinball loss and interpretation
    02_training_quantile_models_slides.md  Companion slide deck (14 slides)

notebooks/
    01_quantiles_not_enough.ipynb     Demonstration with French Bakery data (~12 min)
    02_mqloss_deep_dive.ipynb         Pinball loss, calibration, MQLoss vs DistributionLoss (~13 min)

exercises/
    01_probabilistic_exercises.py     Self-check: 4 exercises with assertions
```

## Sequence

1. Read **Guide 01** — the conceptual argument for why marginals fail
2. Skim **Guide 01 Slides** — or use them if presenting to others
3. Read **Guide 02** — how MQLoss trains the model
4. Run **Notebook 01** — see the numbers. The failure is concrete.
5. Run **Notebook 02** — calibration checks and loss function comparisons
6. Run **Exercises** — verify your understanding with assertions

## The Key Formula

$$q^{0.8}(Y_1 + Y_2 + \cdots + Y_7) \neq q^{0.8}(Y_1) + q^{0.8}(Y_2) + \cdots + q^{0.8}(Y_7)$$

The left side is the 80th percentile of the weekly total. The right side is the sum of daily 80th percentiles. They are equal only under perfect positive correlation. In reality, the right side overestimates the left side by an amount that grows as $\sqrt{H}$ for a horizon of $H$ independent days.

For a 7-day bakery horizon: approximately 10–20% overestimate depending on the actual day-to-day correlation.

## Three Business Failure Scenarios

| Scenario | Error from using marginals | Cost |
|----------|---------------------------|------|
| Annual budget | Sum 12 monthly 80th pcts → near-certainty budget | Capital over-reserved |
| Safety stock | Sum 14-day 95th pcts → safety stock too high | Working capital tied up |
| Weekly order sizing | Sum 7-day 90th pcts → order too much flour | Waste and reduced margin |

## Dataset

All exercises use the **French Bakery dataset** — daily baguette transaction records from a real bakery. The dataset is public and loads directly from GitHub in all notebooks.

## Prerequisites

- Module 01: Point forecasting with NHITS
- Basic probability: marginal and conditional distributions

## What Comes Next

**Module 03: Sample Paths** — generating coherent multi-day trajectories using `ConformalIntervals` in NeuralForecast. Sample paths represent the joint distribution empirically, enabling correct answers to any multi-period question with a single `np.percentile(paths.sum(axis=1), level)`.

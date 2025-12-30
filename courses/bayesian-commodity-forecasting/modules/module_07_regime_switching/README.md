# Module 7: Regime-Switching and Structural Breaks

## Overview

Commodity markets exhibit distinct regimes: bull and bear markets, contango and backwardation, high and low volatility periods. This module covers Bayesian methods for detecting, modeling, and forecasting with regime-switching dynamics.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. **Understand** Hidden Markov Models (HMMs) and regime dynamics
2. **Detect** structural breaks and change points in commodity prices
3. **Build** Markov-switching models for price and volatility
4. **Interpret** regime probabilities for trading signals
5. **Evaluate** regime forecasts and regime-dependent predictions

## Module Contents

### Guides
- `01_hmm_fundamentals.md` - Hidden Markov Models explained
- `02_change_point_detection.md` - Bayesian change point models
- `03_regime_switching_volatility.md` - MS-GARCH and stochastic volatility

### Notebooks
- `01_hmm_from_scratch.ipynb` - Building an HMM step by step
- `02_commodity_regime_detection.ipynb` - Identifying oil market regimes
- `03_markov_switching_pymc.ipynb` - MS models in PyMC
- `04_change_point_analysis.ipynb` - Detecting structural breaks
- `05_regime_based_forecasting.ipynb` - Regime-conditional predictions

### Assessments
- `quiz.md` - Regime model concepts (15 questions)
- `mini_project_rubric.md` - Regime detection for chosen commodity

## Key Concepts

### Why Regimes Matter for Commodities

```
              WTI Crude Oil Regimes
    │
$150│                     ●  Commodity
    │                    ╱ ╲  Supercycle Peak
    │                   ╱   ╲
$100│            ╱╲    ╱     ╲
    │           ╱  ╲  ╱       ╲
 $50│      ╱╲  ╱    ╲╱         ╲────────
    │─────╱  ╲╱                   │
    │     │   │      │      │    │
    └─────┴───┴──────┴──────┴────┴────▶
        1998  2003    2008   2014  2020

Different regimes have different:
- Mean price levels
- Volatility
- Response to fundamentals
- Persistence
```

### Commodity Market Regimes

| Regime Type | Characteristics | Examples |
|-------------|-----------------|----------|
| **Bull/Bear** | Rising vs falling trend | 2003-2008 bull, 2014-2016 bear |
| **Contango/Backwardation** | Curve shape | Storage economics |
| **High/Low Volatility** | Risk levels | Crisis vs calm periods |
| **Tight/Loose Supply** | Inventory levels | Spare capacity |

### Hidden Markov Model Structure

```
Hidden States:    S₁ ──→ S₂ ──→ S₃ ──→ S₄
(Regime)           │      │      │      │
                   ↓      ↓      ↓      ↓
Observations:     Y₁     Y₂     Y₃     Y₄
(Prices)

Transition Matrix:
         To Bull  To Bear
From Bull  0.95    0.05
From Bear  0.08    0.92

High diagonal = persistent regimes
```

### HMM Components

1. **Initial Distribution:** $\pi_k = P(S_1 = k)$
2. **Transition Matrix:** $A_{jk} = P(S_t = k | S_{t-1} = j)$
3. **Emission Distribution:** $P(Y_t | S_t = k)$

### Bayesian Change Point Detection

For detecting structural breaks:

$$y_t \sim \mathcal{N}(\mu_{r(t)}, \sigma^2)$$

Where $r(t)$ is the regime active at time $t$, with change points at unknown times $\tau_1, \tau_2, ...$.

**Prior on number of change points:**
- Poisson prior: $K \sim \text{Poisson}(\lambda)$
- Encourages parsimony (few change points)

## Regime-Switching Applications

### 1. Bull/Bear Market Classification

Two-state model for price trends:
- **Bull:** $\mu_{\text{bull}} > 0$, lower volatility
- **Bear:** $\mu_{\text{bear}} < 0$, higher volatility

### 2. Volatility Regimes

High/low volatility states:
- **Calm:** $\sigma_{\text{low}} \approx 15\%$ (oil)
- **Crisis:** $\sigma_{\text{high}} \approx 50\%$ (oil)

### 3. Fundamental Response Regimes

Different coefficient on inventory in different regimes:
- **Tight supply:** Strong price response to inventory draws
- **Ample supply:** Weak price response

## Completion Criteria

- [ ] HMM implementation produces valid regime estimates
- [ ] Regime detection notebook on commodity data
- [ ] Change point analysis identifies known structural breaks
- [ ] Quiz score ≥ 80%

## Prerequisites

- Module 1-3 completed
- MCMC concepts (Module 6)
- Basic time series familiarity

---

*"Markets move in regimes. Models that ignore regimes will be systematically wrong—and dangerously so at regime transitions."*

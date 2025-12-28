# Bayesian Time Series Forecasting for Commodity Trading

## Course Overview

This advanced course teaches Bayesian approaches to time series modeling with a specific focus on commodity market fundamentals. You'll learn to build probabilistic forecasting models that incorporate supply/demand dynamics, storage economics, and market microstructure into principled uncertainty quantification.

**Level:** Graduate / Professional
**Prerequisites:** Linear algebra, probability theory, basic Python, introductory statistics
**Duration:** 8 modules + capstone (10-12 weeks)
**Estimated Effort:** 8-10 hours per week

## Why Bayesian Methods for Commodities?

Commodity markets present unique challenges that Bayesian methods address naturally:

1. **Uncertainty Quantification** - Trading decisions require knowing not just point forecasts but confidence intervals
2. **Incorporating Prior Knowledge** - Storage costs, seasonality, and fundamental relationships can be encoded as priors
3. **Sparse Data** - Many fundamental series (inventory reports, crop conditions) are infrequent; Bayesian methods handle small samples gracefully
4. **Regime Changes** - Bayesian regime-switching models capture structural breaks in commodity cycles
5. **Hierarchical Structure** - Related commodities (energy complex, agricultural grains) share information through hierarchical models

## Learning Outcomes

By completing this course, you will be able to:

1. **Build** Bayesian time series models using PyMC, NumPyro, or Stan
2. **Incorporate** commodity fundamentals (inventories, production, consumption) into probabilistic forecasts
3. **Quantify** forecast uncertainty and translate it into trading risk metrics
4. **Detect** regime changes and structural breaks in commodity price dynamics
5. **Evaluate** model performance using proper scoring rules and backtesting frameworks
6. **Deploy** Bayesian forecasting pipelines for real-time commodity analysis

## Course Structure

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| 0 | Foundations | Probability review, Python setup, commodity market basics |
| 1 | Bayesian Fundamentals | Bayes' theorem, conjugate priors, posterior inference |
| 2 | Commodity Data & Features | Fundamental data sources, seasonality, term structure |
| 3 | State Space Models | Kalman filter, local level/trend, stochastic volatility |
| 4 | Hierarchical Models | Partial pooling, related commodities, cross-market info |
| 5 | Gaussian Processes | Non-parametric regression, kernel design, uncertainty |
| 6 | Inference Algorithms | MCMC, HMC, variational inference, diagnostics |
| 7 | Regime Switching | Hidden Markov models, structural breaks, cycle detection |
| 8 | Fundamentals Integration | Supply/demand models, storage theory, forecast combination |
| Capstone | End-to-End Project | Complete forecasting system for chosen commodity |

## Technical Requirements

```bash
# Create environment
conda create -n bayes-commodity python=3.11
conda activate bayes-commodity

# Core packages
pip install pymc arviz xarray pandas numpy matplotlib seaborn
pip install numpyro jax jaxlib  # Alternative backend
pip install cmdstanpy  # Stan interface

# Commodity data
pip install yfinance fredapi quandl

# Notebooks
pip install jupyterlab ipywidgets
```

## Assessment Structure

| Component | Weight | Description |
|-----------|--------|-------------|
| Weekly Quizzes | 15% | Conceptual understanding checks |
| Coding Exercises | 25% | Module notebooks with auto-graded tests |
| Mini-Projects | 30% | Bi-weekly applied modeling tasks |
| Capstone | 30% | End-to-end forecasting system |

## Data Sources Used

- **Prices:** CME futures (via Yahoo Finance), ICE, LME
- **Fundamentals:** EIA (energy), USDA (agriculture), LME (metals)
- **Macro:** FRED (Federal Reserve Economic Data)
- **Alternative:** Satellite imagery proxies, shipping data

## Getting Started

1. Complete the environment setup in `resources/environment_setup.md`
2. Run the diagnostic assessment in `module_00_foundations/`
3. Begin with Module 1 if prerequisites are met, or review foundations first

## Instructors & Support

- Office Hours: Weekly live Q&A sessions
- Forum: Peer discussion and instructor support
- AI Assistant: 24/7 conceptual help via course chatbot

---

*"The goal is not to predict the future perfectly, but to make better decisions under uncertainty."*

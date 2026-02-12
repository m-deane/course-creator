# Multi-Armed Bandits & Adaptive Experimentation for Commodity Trading

## Course Overview

Learn to make better decisions under uncertainty — while you're still operating. This course teaches multi-armed bandit algorithms as the practical upgrade to A/B testing, with deep applications to commodity trading, portfolio allocation, and data-driven growth systems.

Classical A/B testing asks you to freeze the world while you learn. Bandits accept reality: you learn while moving, improve while shipping, and waste less while exploring. This course takes you from the core explore-exploit tradeoff through production-grade bandit systems, with specific applications to energy, agriculture, and metals markets.

**Level:** Intermediate to Advanced
**Prerequisites:** Python, basic probability/statistics, familiarity with commodity markets (helpful but not required)
**Duration:** 8 modules (8-10 weeks)
**Estimated Effort:** 6-8 hours per week

## Why Multi-Armed Bandits?

Traditional A/B testing was built for a cleaner world: split traffic 50/50, wait, declare a winner. But real-world problems move while you test:

1. **Commodity markets shift** — Seasonal patterns, supply shocks, and regime changes mean yesterday's best strategy may fail tomorrow
2. **Exploration is expensive** — Every sub-optimal allocation costs real money; bandits minimize regret while learning
3. **Feedback is noisy and delayed** — Commodity signals (inventory reports, crop conditions) arrive irregularly; bandits handle partial feedback naturally
4. **You can't stop operating** — Traders, portfolio managers, and growth teams must keep making decisions while they learn
5. **Non-stationarity is the norm** — Bandits with decay and change detection adapt to shifting reward distributions

## Learning Outcomes

By completing this course, you will be able to:

1. **Explain** why A/B testing fails in non-stationary, high-stakes environments and when bandits are the right tool
2. **Implement** core bandit algorithms (epsilon-greedy, UCB, Thompson Sampling) from scratch
3. **Apply** Bayesian bandits with Thompson Sampling to real allocation problems
4. **Build** contextual bandits that adapt decisions based on market features (term structure, volatility regime, seasonality)
5. **Design** bandit systems for commodity portfolio allocation — the "two-wallet" framework with guardrails
6. **Deploy** production bandit systems with monitoring, guardrails, and non-stationarity detection
7. **Evaluate** bandit performance using regret analysis, policy comparison, and proper backtesting

## Course Structure

| Module | Topic | Key Concepts |
|--------|-------|--------------|
| 0 | Foundations | A/B testing mechanics, explore-exploit tradeoff, decision theory basics |
| 1 | Core Bandit Algorithms | Epsilon-greedy, UCB1, softmax, regret bounds |
| 2 | Bayesian Bandits | Thompson Sampling, Beta-Bernoulli, posterior updating |
| 3 | Contextual Bandits | LinUCB, feature-based decisions, personalization |
| 4 | Bandits for Content & Growth | Creator optimization, conversion testing, adaptive allocation |
| 5 | Bandits for Commodity Trading | Portfolio tilting, regime adaptation, accumulator playbook |
| 6 | Advanced Topics | Non-stationary bandits, restless bandits, adversarial settings |
| 7 | Production Systems | Deployment, monitoring, guardrails, A/B-to-bandit migration |

## Start Here

1. **New to bandits?** Start with `quick-starts/00_your_first_bandit.ipynb` — a working bandit in 2 minutes
2. **Want production code?** Grab `templates/bandit_engine_template.py` — plug-and-play bandit system
3. **Need a specific pattern?** Check `recipes/` — copy-paste solutions for common bandit problems
4. **Full learning path?** Begin with `modules/module_00_foundations/`

## Technical Requirements

```bash
# Create environment
conda create -n bandits python=3.11
conda activate bandits

# Core packages
pip install numpy pandas matplotlib seaborn scipy
pip install scikit-learn statsmodels

# Bayesian
pip install pymc arviz

# Commodity data
pip install yfinance fredapi

# Notebooks
pip install jupyterlab ipywidgets plotly
```

## Data Sources Used

- **Prices:** CME futures via Yahoo Finance (WTI, Natural Gas, Corn, Copper, Gold)
- **Fundamentals:** EIA (energy inventories), USDA (crop reports), LME (metals)
- **Macro:** FRED (rates, dollar index, economic indicators)
- **Alternative:** VIX, term structure spreads, seasonal indices

## Connections to Other Courses

This course connects directly to:
- **Bayesian Commodity Forecasting** — Thompson Sampling extends the Bayesian inference framework
- **GenAI for Commodities** — Bandit-based routing for LLM prompt strategies and model selection
- **Hidden Markov Models** — Regime detection feeds contextual bandit features

---

*"The goal isn't to find the single best option. It's to learn what works while wasting as little as possible on what doesn't."*

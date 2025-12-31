# Module 4: Financial Applications

## Overview

Apply HMMs to financial time series: market regime detection, volatility state identification, and regime-dependent forecasting.

**Time Estimate:** 10-12 hours

## Learning Objectives

By completing this module, you will:
1. Detect bull/bear market regimes
2. Identify volatility states
3. Build regime-switching strategies
4. Combine HMMs with other models

## Contents

### Guides
- `01_regime_detection.md` - Market state identification
- `02_volatility_states.md` - Low/high volatility regimes
- `03_regime_strategies.md` - Trading on regime signals

### Notebooks
- `01_market_regimes.ipynb` - S&P 500 regime analysis
- `02_volatility_regimes.ipynb` - VIX state modeling
- `03_strategy_backtest.ipynb` - Regime-based strategies

## Key Concepts

### Market Regime Detection

```python
# Fit HMM to returns
model = GaussianHMM(n_components=2)
model.fit(returns)

# Identify bull/bear by mean
means = model.means_.flatten()
bull_state = np.argmax(means)
bear_state = np.argmin(means)

# Get regime probabilities
probs = model.predict_proba(returns)
bull_prob = probs[:, bull_state]
```

### Regime Characteristics

| Regime | Mean Return | Volatility | Duration |
|--------|-------------|------------|----------|
| Bull | Positive | Lower | Longer |
| Bear | Negative | Higher | Shorter |

### Volatility Regimes

```python
# Realized volatility as observation
realized_vol = returns.rolling(21).std() * np.sqrt(252)

# Fit HMM to log volatility
log_vol = np.log(realized_vol).dropna().values.reshape(-1, 1)
vol_model = GaussianHMM(n_components=2)
vol_model.fit(log_vol)

# Low vs high vol states
vol_states = vol_model.predict(log_vol)
```

### Regime-Based Strategy

```python
def regime_strategy(returns, regimes, bull_weight=1.0, bear_weight=0.0):
    """Position based on detected regime."""
    weights = np.where(regimes == bull_state, bull_weight, bear_weight)
    strategy_returns = weights[:-1] * returns[1:]
    return strategy_returns
```

### Model Selection

- Number of states: BIC/AIC
- Covariance type: full, diag, tied
- Stability: multiple initializations

## Prerequisites

- Module 0-3 completed
- Financial markets knowledge

# Module 5: Extensions and Advanced Topics

## Overview

Explore advanced HMM variants: hierarchical HMMs, switching autoregressive models, and non-parametric approaches.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Understand hierarchical HMM structures
2. Implement regime-switching AR models
3. Apply sticky HMMs for persistence
4. Compare with other regime models

## Contents

### Guides
- `01_hierarchical_hmm.md` - Multi-level hidden states
- `02_switching_ar.md` - AR models with regime switching
- `03_sticky_hmm.md` - Persistent regimes

### Notebooks
- `01_hhmm_implementation.ipynb` - Hierarchical HMMs
- `02_markov_switching_ar.ipynb` - MS-AR models

## Key Concepts

### Hierarchical HMM

```
Super-states (macro regimes)
     ↓
Sub-states (micro regimes within each macro)
     ↓
Observations
```

Applications:
- Market regime → Sector behavior
- Economic cycle → Asset class dynamics

### Markov-Switching AR

$$y_t = c_{s_t} + \phi_{s_t} y_{t-1} + \sigma_{s_t} \epsilon_t$$

Each regime $s_t$ has its own:
- Intercept $c_{s_t}$
- AR coefficient $\phi_{s_t}$
- Volatility $\sigma_{s_t}$

```python
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

model = MarkovRegression(
    returns,
    k_regimes=2,
    order=1,
    switching_variance=True
)
results = model.fit()
```

### Sticky HMM

Increase self-transition probability:

$$A_{ii} = \kappa + (1-\kappa) \cdot \alpha_i$$

Higher $\kappa$ = more persistent regimes

### Model Comparison

| Model | Complexity | Interpretability | Use Case |
|-------|------------|------------------|----------|
| Basic HMM | Low | High | Simple regimes |
| Gaussian HMM | Medium | High | Continuous data |
| MS-AR | Medium | Medium | Dynamic patterns |
| HHMM | High | Medium | Multi-scale |

### Practical Considerations

1. **State persistence**: Real regimes are sticky
2. **State number**: Use information criteria
3. **Initialization**: Multiple random starts
4. **Stationarity**: Ensure ergodic chains

## Prerequisites

- Module 0-4 completed
- Time series modeling

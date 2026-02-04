# State-Space Representation

```
┌─────────────────────────────────────────────────────────────────────┐
│ STATE-SPACE REPRESENTATION                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   HIDDEN STATE (factors):    OBSERVATIONS (data):                  │
│                                                                     │
│   α₀ ──→ α₁ ──→ α₂ ──→ α₃     y₀    y₁    y₂    y₃                │
│    ↓      ↓      ↓      ↓      ↑     ↑     ↑     ↑                 │
│    └──────┴──────┴──────┴──────┴─────┴─────┴─────┘                 │
│                                                                     │
│   Transition equation:    α_t = T·α_{t-1} + R·η_t                  │
│   Observation equation:   y_t = Z·α_t + ε_t                        │
│                                                                     │
│   Two-layer model:                                                 │
│   1. States evolve over time (hidden dynamics)                     │
│   2. Observations depend on current state (measurement)            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ TL;DR: Split your model into hidden dynamics (state) and noisy     │
│        observations, enabling optimal filtering with Kalman.       │
├─────────────────────────────────────────────────────────────────────┤
│ Code (< 15 lines):                                                  │
│                                                                     │
│   import numpy as np                                                │
│   from statsmodels.tsa.statespace.mlemodel import MLEModel         │
│                                                                     │
│   # Define state-space model                                       │
│   class DFM(MLEModel):                                              │
│       def __init__(self, data, k_states):                           │
│           super().__init__(data, k_states=k_states)                 │
│           self['transition'] = np.eye(k_states) * 0.8              │
│           self['selection'] = np.eye(k_states)                      │
│           self['design'] = np.random.randn(data.shape[1], k_states)│
│           self['obs_cov'] = np.eye(data.shape[1]) * 0.1            │
│                                                                     │
│   model = DFM(data, k_states=3)                                     │
│   result = model.smooth(model.start_params)                         │
├─────────────────────────────────────────────────────────────────────┤
│ Common Pitfall: Setting T matrix with eigenvalues > 1 causes       │
│                 explosive states. Always check stationarity!        │
└─────────────────────────────────────────────────────────────────────┘
```

## Why State-Space?

State-space representation is the **Swiss Army knife** of time series modeling. It unifies:
- ARIMA models
- Factor models
- Structural time series
- Mixed-frequency models
- Missing data handling

All in a single framework that can be estimated with the Kalman filter.

## Real-World Example

**Problem:** You want to measure "economic conditions" but only observe noisy indicators (GDP, unemployment, retail sales).

**Solution:**
- **State (α_t):** Unobserved "true" economic condition
- **Observations (y_t):** Your noisy indicators
- **State equation:** How economic conditions evolve
- **Observation equation:** How indicators relate to true conditions

The Kalman filter optimally separates signal from noise.

## When to Use

Use state-space when you need to:
- Extract latent factors from multiple time series
- Handle missing data or ragged edges
- Combine mixed-frequency data (daily + monthly)
- Do nowcasting (real-time estimation)
- Model time-varying parameters

## Connection to Dynamic Factor Models

In DFM context:
- **States (α_t):** The r latent factors (unobserved)
- **Observations (y_t):** Your N observed time series
- **Goal:** Estimate factors and their dynamics from observations

The beauty: You can estimate 3 factors from 100+ series, dramatically reducing dimensionality while preserving information.

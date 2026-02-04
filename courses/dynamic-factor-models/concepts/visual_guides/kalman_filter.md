# Kalman Filter

```
┌─────────────────────────────────────────────────────────────────────┐
│ KALMAN FILTER                                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Two-step recursive algorithm:                                    │
│                                                                     │
│   ┌─────────────┐         ┌─────────────┐                          │
│   │  PREDICT    │         │   UPDATE    │                          │
│   │             │         │             │                          │
│   │  Use model  │────────→│  Use new    │                          │
│   │  dynamics   │  α_t|t-1│  data point │                          │
│   │             │         │             │                          │
│   │ α̂_t|t-1 = T·α̂_t-1     │ α̂_t|t = α̂_t|t-1 + K_t·v_t             │
│   │ P_t|t-1 = T·P_t-1·T'+Q│ P_t|t = (I-K_t·Z)·P_t|t-1              │
│   └─────────────┘         └─────────────┘                          │
│         │                        │                                 │
│         └────────────┬───────────┘                                 │
│                      ↓                                              │
│               Next time step                                        │
│                                                                     │
│   Innovation: v_t = y_t - Z·α̂_t|t-1  (prediction error)            │
│   Kalman Gain: K_t = P_t|t-1·Z'·F_t⁻¹  (how much to trust data)    │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ TL;DR: Optimal recursive filter that combines model predictions    │
│        with new data, giving more weight to the more certain.      │
├─────────────────────────────────────────────────────────────────────┤
│ Code (< 15 lines):                                                  │
│                                                                     │
│   from statsmodels.tsa.statespace.kalman_filter import KalmanFilter│
│                                                                     │
│   kf = KalmanFilter(k_endog=N, k_states=r)                         │
│   kf['design'] = Z          # Observation matrix                   │
│   kf['transition'] = T      # State transition                     │
│   kf['selection'] = R       # State disturbance selector           │
│   kf['obs_cov'] = H         # Observation noise                    │
│   kf['state_cov'] = Q       # State noise                          │
│                                                                     │
│   # Run filter                                                      │
│   kf.initialize_approximate_diffuse()                               │
│   filtered_states = kf.filter(data)                                 │
│   smoothed_states = kf.smooth(data)  # Two-pass for full sample    │
├─────────────────────────────────────────────────────────────────────┤
│ Common Pitfall: Forgetting to initialize the filter! Use           │
│                 initialize_approximate_diffuse() for unknown start. │
└─────────────────────────────────────────────────────────────────────┘
```

## Intuition: GPS for Your Data

Think of the Kalman filter like your phone's GPS:

1. **Predict:** "Based on your last position and speed, I think you're here"
2. **Update:** "New satellite data says you're actually slightly left"
3. **Combine:** Weighted average based on trust (uncertainty)

If GPS signal is weak (high uncertainty), trust the prediction more.
If GPS signal is strong (low uncertainty), trust the new data more.

The Kalman gain **K_t** automatically adjusts this balance.

## Why It's Optimal

The Kalman filter is provably the **best linear unbiased estimator** (BLUE) when:
- State dynamics are linear
- Noise is Gaussian
- You want minimum mean-squared error

Even when these assumptions fail, it's still a good approximation.

## One-Pass vs Two-Pass

**Filtering (one-pass):**
- Uses data up to time t: α̂_t|t
- Real-time application (nowcasting)
- Causal (can't see the future)

**Smoothing (two-pass):**
- Uses ALL data: α̂_t|T (T is final time)
- Better estimates but not real-time
- For historical analysis and model estimation

## Connection to Dynamic Factor Models

In DFM, Kalman filter solves two problems:

1. **Factor estimation:** Given parameters (Z, T, Q, H), extract factors
2. **Parameter estimation:** Via EM algorithm or MLE (maximizing likelihood from Kalman filter)

The filter gives you:
- Filtered factors: α̂_t|t (for nowcasting)
- Smoothed factors: α̂_t|T (for historical analysis)
- Prediction errors: v_t (for likelihood calculation)
- Forecast uncertainty: P_t|t (for confidence bands)

## Quick Check

**Q:** What if observation noise H → 0 (perfect measurements)?
**A:** Kalman gain → 1, filter fully trusts data: α̂_t|t ≈ Z⁻¹·y_t

**Q:** What if state noise Q → 0 (perfect model)?
**A:** Kalman gain → 0, filter fully trusts prediction: α̂_t|t ≈ T·α̂_t-1

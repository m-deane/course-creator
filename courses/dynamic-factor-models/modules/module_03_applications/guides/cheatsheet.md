# Module 03 Cheatsheet: Applications - Nowcasting & Forecasting

## Quick Reference

### Nowcasting Framework

**Bridge Equation:**
```
GDP_τ = β₀ + β₁·F̄_τ + ε_τ
where F̄_τ = (1/3)(F₃τ₋₂ + F₃τ₋₁ + F₃τ)  (quarterly average of monthly factors)
```

**Nowcast:**
```
ŷ_τ|t = β₀ + β₁·E[F̄_τ | X₁:t]
```

**Uncertainty:**
```
Var(ŷ_τ|t) = β₁²·Var(F̄_τ | X₁:t) + σ_ε²
```

---

### Kalman Filter with Missing Data

**Prediction (always):**
```
F_t|t-1 = Φ·F_t-1|t-1
P_t|t-1 = Φ·P_t-1|t-1·Φ' + Q
```

**Update (only observed):**
```
Let O_t = indices of observed variables

Λ_obs = Λ[O_t, :]
Σ_obs = Σ_e[O_t, O_t]
X_obs = X_t[O_t]

K_t = P_t|t-1·Λ_obs'·(Λ_obs·P_t|t-1·Λ_obs' + Σ_obs)⁻¹
F_t|t = F_t|t-1 + K_t·(X_obs - Λ_obs·F_t|t-1)
P_t|t = (I - K_t·Λ_obs)·P_t|t-1
```

**All missing:** Skip update, `F_t|t = F_t|t-1`

---

### Forecast Evaluation Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| **MSE** | `(1/T)Σ(y - ŷ)²` | Penalize large errors |
| **RMSE** | `√MSE` | Same units as target |
| **MAE** | `(1/T)Σ\|y - ŷ\|` | Robust to outliers |
| **MAPE** | `(100/T)Σ\|y - ŷ\|/\|y\|` | Percentage errors |
| **CRPS** | `∫[F(z) - 1{y≤z}]² dz` | Full distribution |
| **Dir. Acc.** | `(1/T)Σ 1{sign(ŷ)=sign(y)}` | Trading signals |

---

### Diebold-Mariano Test

**Null:** Equal predictive accuracy between models A and B

**Test statistic:**
```python
d_t = loss(e_A,t) - loss(e_B,t)
DM = d̄ / √(Var(d)/T)  →  N(0,1)

Reject H₀ if |DM| > 1.96 (5% level)
```

**Harvey small-sample correction:**
```
DM* = DM · √[(T+1-2h+h(h-1)/T) / T]
```

---

### Optimal Forecast Combination

**Weights that minimize MSE:**
```
w = Σ_e⁻¹·1 / (1'·Σ_e⁻¹·1)

where Σ_e = Cov(forecast errors)
```

**Simple average (1/n each):** Often nearly optimal!

---

### Publication Lag Patterns (US Data)

| Series | Typical Lag | Revision Frequency |
|--------|-------------|-------------------|
| PMI surveys | t+1 day | Rarely revised |
| Initial claims | t+4 days | Rarely revised |
| Employment | t+5 days | Annual benchmarks |
| Retail sales | t+12 days | Monthly for 2 months |
| Industrial production | t+15 days | Monthly for 3 months |
| GDP (advance) | t+30 days | 2nd, 3rd, annual revisions |
| GDP (final) | t+90 days | Revised for years |

---

### Missing Data Mechanisms

| Type | Definition | Nowcasting Implication |
|------|------------|----------------------|
| **MCAR** | `P(M\|X,F) = P(M)` | No bias, just less efficient |
| **MAR** | `P(M\|X,F) = P(M\|X_obs)` | Kalman filter handles well |
| **MNAR** | `P(M\|X,F) depends on X_missing` | Potential bias, model missingness |

**Ragged edge:** Usually MAR (fixed publication schedules)

---

## Code Snippets

### Build Nowcasting Model
```python
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

# Step 1: Estimate DFM
dfm = DynamicFactor(monthly_data, k_factors=3, factor_order=2)
dfm_res = dfm.fit()

# Step 2: Extract factors
factors_monthly = dfm_res.factors.filtered
factors_quarterly = factors_monthly.resample('Q').mean()

# Step 3: Bridge equation
from sklearn.linear_model import LinearRegression
bridge_model = LinearRegression()
bridge_model.fit(factors_quarterly, gdp_quarterly)

# Step 4: Nowcast
nowcast = bridge_model.predict(current_quarter_factors)
```

### Kalman Filter with Missing Data
```python
def kalman_update_missing(F_pred, P_pred, X_t, Lambda, Sigma_e):
    """Update step handling NaNs."""
    obs_mask = ~np.isnan(X_t)
    obs_idx = np.where(obs_mask)[0]

    if len(obs_idx) == 0:
        return F_pred, P_pred  # No update

    X_obs = X_t[obs_idx]
    Λ_obs = Lambda[obs_idx, :]
    Σ_obs = Sigma_e[np.ix_(obs_idx, obs_idx)]

    v = X_obs - Λ_obs @ F_pred
    S = Λ_obs @ P_pred @ Λ_obs.T + Σ_obs
    K = P_pred @ Λ_obs.T @ np.linalg.inv(S)

    F_filt = F_pred + K @ v
    P_filt = (np.eye(len(F_pred)) - K @ Λ_obs) @ P_pred

    return F_filt, P_filt
```

### Diebold-Mariano Test
```python
def diebold_mariano(errors_A, errors_B, h=1):
    """Test equal forecast accuracy."""
    d = errors_A**2 - errors_B**2  # MSE loss
    T = len(d)

    d_bar = np.mean(d)
    var_d = np.var(d, ddof=1) / T  # Simplified (ignores autocorr)

    dm_stat = d_bar / np.sqrt(var_d)
    # Harvey correction
    dm_stat *= np.sqrt((T + 1 - 2*h + h*(h-1)/T) / T)

    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))
    return dm_stat, p_value
```

### Backtest Framework
```python
def expanding_window_backtest(data, model_fn, start, end):
    """Out-of-sample evaluation."""
    results = []
    for date in pd.date_range(start, end, freq='Q'):
        # Train on data up to (but not including) forecast date
        train = data.loc[:date].iloc[:-1]
        model = model_fn(train)

        # Forecast current period
        y_pred = model.predict(date)
        y_true = data.loc[date, 'target']

        results.append({'date': date, 'pred': y_pred, 'true': y_true})

    return pd.DataFrame(results)
```

---

## Decision Trees

### When to Use Each Scoring Rule

```
Is forecast distributional (not just point)?
├─ Yes → Use CRPS or Log Score
└─ No (point forecast only)
   ├─ Outliers present? → Use MAE
   ├─ Direction matters more than magnitude? → Use Directional Accuracy
   ├─ Different scales across periods? → Use MAPE
   └─ Standard case → Use RMSE
```

### Handling Missing Data Strategy

```
Is data missing at random (MAR)?
├─ Yes
│  ├─ Do you have factor model? → Kalman filter with missing obs
│  └─ No model → Forward-fill (conservative) or interpolation
└─ No (MNAR)
   └─ Model missingness mechanism explicitly
      (e.g., Heckman selection, pattern-mixture models)
```

### Forecast Combination Strategy

```
Do you have multiple forecasts?
├─ Yes
│  ├─ Know error covariances? → Optimal MSE weights
│  ├─ Uncertain about covariances? → Simple average (robust)
│  └─ Some forecasts clearly bad? → Trimmed mean
└─ No → Use best single model (select via cross-validation)
```

---

## Common Mistakes to Avoid

### In Nowcasting
- ❌ Using final revised GDP instead of first-release for evaluation
- ❌ Ignoring publication lags in backtest (look-ahead bias)
- ❌ Assuming all data available simultaneously
- ✅ Use ALFRED real-time vintages
- ✅ Simulate actual data flow (ragged edges)

### In Missing Data
- ❌ Interpolating with future data
- ❌ Treating imputed values as observed (no uncertainty)
- ❌ Inverting singular matrices when all missing
- ✅ Only use past data for imputation
- ✅ Propagate uncertainty from Kalman filter
- ✅ Add safety checks for edge cases

### In Evaluation
- ❌ In-sample testing (overfitting)
- ❌ Cherry-picking evaluation period
- ❌ Multiple testing without correction
- ✅ Out-of-sample backtest
- ✅ Report full sample + recessions separately
- ✅ Pre-specify primary comparison

---

## Key Formulas at a Glance

**News vs Uncertainty:**
```
Δ Nowcast = β₁·[E[F|X₁:t] - E[F|X₁:t₋₁]]  ← News
Uncertainty reduction = Var(F|X₁:t₋₁) - Var(F|X₁:t)
```

**CRPS (Gaussian forecast):**
```
CRPS(N(μ,σ²), y) = σ·[z·(2Φ(z)-1) + 2φ(z) - 1/√π]
where z = (y-μ)/σ
```

**Forecast encompassing regression:**
```
y_t = α + β₁·ŷ_A + β₂·ŷ_B + ε
H₀: β₁=1, β₂=0  (A encompasses B)
```

---

## Resources

**Datasets:**
- FRED-MD: https://research.stlouisfed.org/econ/mccracken/fred-databases/
- ALFRED (vintages): https://alfred.stlouisfed.org/
- Survey of Professional Forecasters: https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/spf

**Software:**
- `statsmodels.tsa.statespace.DynamicFactor` - DFM estimation
- `properscoring` package - CRPS and other proper scores
- `pandas_datareader` - FRED data access

**Benchmarks to Compare Against:**
- NY Fed Nowcast: https://www.newyorkfed.org/research/policy/nowcast
- Atlanta Fed GDPNow: https://www.atlantafed.org/cqer/research/gdpnow
- SPF consensus: Survey of Professional Forecasters

---

## Quick Diagnostic Checklist

**Before deploying nowcasting system:**
- [ ] Verified publication lag patterns match reality
- [ ] Backtested on real-time vintages (not revised data)
- [ ] Compared to simple AR benchmark
- [ ] Evaluated during recessions specifically
- [ ] Uncertainty bands calibrated (68% interval contains 68% of outcomes)
- [ ] Automated data refresh pipeline tested
- [ ] Missing data handled via Kalman filter (not naive fill)
- [ ] News decomposition implemented for interpretability
- [ ] Model documented for stakeholders (what drives nowcast changes?)

---

*For detailed explanations, see full guides in this module.*

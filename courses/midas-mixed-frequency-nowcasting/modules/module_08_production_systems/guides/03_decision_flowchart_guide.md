# Decision Flowcharts for Production Nowcasting

## Learning Objectives

After reading this guide you will be able to:

1. Follow a complete decision tree for building and deploying a MIDAS nowcasting model
2. Identify the correct ragged-edge fill strategy for any indicator type
3. Select the appropriate model specification and regularisation approach
4. Decide when to re-estimate, retrain, or retire a model
5. Diagnose and resolve the most common production failure modes

---

## 1. Overview

This guide presents the key decision points encountered when building a production MIDAS nowcasting system as explicit flowcharts. Each flowchart covers one phase of the system lifecycle.

The five flowcharts are:

1. **Model specification**: choosing indicators, lag depth, and weight function
2. **Ragged-edge handling**: per-indicator fill strategy selection
3. **Estimation and regularisation**: model family and cross-validation design
4. **Evaluation**: deciding whether the model is production-ready
5. **Production maintenance**: re-estimation, monitoring, and retirement

---

## 2. Flowchart 1 — Model Specification

```
START: Define forecasting objective
         │
         ▼
Q1: What is the target frequency?
     │
     ├─ Monthly ──────────────────────────────────────┐
     │                                                │
     └─ Quarterly ────────────────────────────────────┤
                                                      │
                                                      ▼
Q2: How many candidate indicators are available?
     │
     ├─ 1–3 ────► Basic MIDAS (NLS with Beta weights)
     │             K = 6–12 lags per indicator
     │             No regularisation needed
     │
     ├─ 4–20 ───► Regularised MIDAS (Elastic Net or Ridge)
     │             K = 12 lags per indicator
     │             CV lambda selection on expanding window
     │
     └─ >20 ────► Group Lasso or midasml
                  K = 12 lags, group = indicator
                  Simultaneous indicator + lag selection

         │
         ▼
Q3: What is the highest-frequency predictor?
     │
     ├─ Monthly ──► Standard MIDAS, K = 1 lag
     │              (effectively a distributed lag model)
     │
     ├─ Weekly ───► MIDAS-W, K = 4 lags (4 weeks per month)
     │              or K = 5 for months with 5 full weeks
     │
     └─ Daily ────► K = 22 lags (≈22 trading days per month)
                    or K = 63 for quarterly target
                    Beta weights essential (22/63 free params → 2)

         │
         ▼
Q4: Should Beta or Almon weights be used?
     │
     ├─ High-frequency predictor (weekly/daily) ──► Beta weights
     │   Reason: Beta is monotone-decreasing or hump-shaped;
     │            both shapes are a priori plausible for
     │            daily data with 22+ lags
     │
     └─ Low-frequency predictor (monthly, few lags) ──► Unrestricted
         Reason: K=3 or K=4 lags → unrestricted OLS is fine;
                  Beta restriction buys nothing with few lags
```

---

## 3. Flowchart 2 — Ragged-Edge Fill Strategy

The choice of fill method should be made per indicator, not globally. This flowchart guides the decision for each series in your indicator set.

```
For each indicator i in {1, ..., N}:

         ▼
Q1: Is the series a level or a growth rate?
     │
     ├─ Growth rate (e.g. IP growth, payroll growth)
     │         │
     │         ▼
     │   Q2: Is zero a plausible prior for the missing period?
     │         │
     │         ├─ Yes ──► ZERO FILL
     │         │           Rationale: expected growth ≈ 0 is
     │         │           a reasonable Bayesian prior for a
     │         │           series with no strong momentum
     │         │
     │         └─ No ───► CARRY FORWARD
     │                     (series has strong trend, e.g. wages)
     │
     └─ Level (e.g. unemployment rate, IP index)
               │
               ▼
         Q3: Is the series highly persistent (AR1 coeff > 0.9)?
               │
               ├─ Yes ──► CARRY FORWARD
               │           Rationale: last observation is near-optimal
               │           1-step predictor for a highly persistent series
               │
               └─ No ───► AR1 PROJECTION
                           Rationale: series has measurable mean reversion;
                           AR1 projection improves on the last obs

         After filling:

         Q4: Will missing periods affect the first lag?
               │
               ├─ Yes ──► Shift lag window: use lags 2..K+1
               │           instead of 1..K to exclude the filled obs
               │           from the most informative position
               │
               └─ No ───► Use lags 1..K as normal
```

**Practical rule**: Use carry-forward as the default for all series. Switch to AR1 projection only for series where you have verified that the AR1 fit substantially improves OOS RMSE relative to carry-forward. The extra complexity is rarely justified.

---

## 4. Flowchart 3 — Estimation and Regularisation

```
         ▼
Q1: How many observations are available in the training window?
     │
     ├─ n < 40 ──► BASIC OLS or Ridge only
     │              Reason: insufficient data for CV lambda selection
     │              with more complex models; use L2 only to avoid
     │              overfitting; do NOT use Elastic Net or Lasso here
     │
     └─ n ≥ 40 ──► Proceed to Q2

         ▼
Q2: Do you want to select which indicators matter?
     │
     ├─ Yes (sparsity prior) ──► LASSO or ELASTIC NET
     │                            alpha=1.0 (Lasso) for strong sparsity
     │                            alpha=0.5 (EN) for correlated predictors
     │
     └─ No (use all indicators) ──► RIDGE
                                     Shrinks all coefficients toward zero;
                                     no variable selection

         ▼
Q3: Are indicators grouped (same series, different lags)?
     │
     ├─ Yes, and you want group-level selection ──► GROUP LASSO
     │                                              or Elastic Net with
     │                                              group standardisation
     │
     └─ No, or lag-level selection is acceptable ──► Standard Elastic Net

         ▼
Q4: How should cross-validation be designed?
     │
     ├─ Use TimeSeriesSplit (ALWAYS for time series)
     │
     ├─ n_splits = 5 for standard CV
     │
     ├─ Minimum training fold size:
     │    n_min = max(40, 3 × n_lags × n_indicators)
     │
     └─ Gap between train and test = 1 period
        (avoids look-ahead in expanding-window CV)

         ▼
Q5: How to select lambda?
     │
     ├─ LassoCV / ElasticNetCV ──► Built-in CV; use cv=TimeSeriesSplit(5)
     │
     └─ Manual grid ──────────────► np.logspace(-4, 0, 50)
                                    Log-spaced: denser near zero where
                                    solutions differ most
```

---

## 5. Flowchart 4 — Evaluation and Production Readiness

```
         ▼
Q1: Has an expanding-window OOS evaluation been run?
     │
     ├─ No ──► Run it before proceeding. No exceptions.
     │          Minimum: 8 OOS periods (2 years for quarterly)
     │
     └─ Yes ──► Proceed to Q2

         ▼
Q2: Does the model beat the AR(1) benchmark?
     │
     ├─ No (OOS RMSE ≥ AR RMSE) ──► DO NOT DEPLOY
     │   Investigate:
     │    a) Insufficient training data
     │    b) Feature multicollinearity
     │    c) Overfitting (reduce K or n_indicators)
     │    d) Wrong fill method introducing noise
     │
     └─ Yes ──► Proceed to Q3

         ▼
Q3: Is the improvement statistically significant?
     │     (Diebold-Mariano test, 10% level)
     │
     ├─ No (DM p > 0.10) ──► Proceed with caution
     │   Document the result. The model may still be useful if:
     │    a) The directional accuracy is substantially higher
     │    b) The improvement is consistent across subperiods
     │    c) The cost of errors is asymmetric
     │
     └─ Yes (DM p ≤ 0.10) ──► Proceed to Q4

         ▼
Q4: Has the model been evaluated on a pseudo-real-time basis?
     │  (using ragged-edge data, not final revised data)
     │
     ├─ No ──► Run pseudo-real-time evaluation. Naive OOS
     │          with final data overstates model performance
     │          because it uses data that did not exist at
     │          the forecast date.
     │
     └─ Yes ──► Proceed to Q5

         ▼
Q5: Does pseudo-real-time RMSE exceed naive OOS RMSE by >30%?
     │
     ├─ Yes ──► Investigate revision patterns
     │           The model is overly sensitive to preliminary data.
     │           Solutions:
     │            a) Increase pub lag assumption (be more conservative)
     │            b) Use AR1 fill instead of carry-forward for volatile series
     │            c) Down-weight indicators with large revision histories
     │
     └─ No ──► APPROVE FOR DEPLOYMENT
                Document: backtest RMSE, DM p-value, pseudo-real-time RMSE,
                           training window, indicators, model config
```

---

## 6. Flowchart 5 — Production Maintenance

```
DAILY: Pipeline runs on each data release
         │
         ▼
Pipeline completes successfully?
     │
     ├─ No ──► [CRITICAL ALERT]
     │           Check: data source availability,
     │           database connectivity, Python environment
     │           Retry up to 3 times with 5-min backoff
     │           If still failing: use last-known-good forecast
     │           and escalate to engineer
     │
     └─ Yes ──► Store forecast record, generate health report

QUARTERLY: After GDP advance release
         │
         ▼
Append (terminal nowcast, actual GDP) to error database
         │
         ▼
Compute rolling 8-quarter RMSE and bias
         │
         ▼
Run bias t-test and CUSUM test
         │
         ▼
Consult ReEstimationTrigger:

         ├─ Calendar trigger fires (4 quarters elapsed)?
         │         │
         │         └─ YES ──► Re-estimate on updated training window
         │                    Log: "calendar re-estimation"
         │
         ├─ Performance trigger fires (RMSE > 120% baseline, 2 periods)?
         │         │
         │         └─ YES ──► Re-estimate on updated training window
         │                    Log: "performance re-estimation"
         │                    Update baseline RMSE to new backtest value
         │
         └─ Structural break trigger fires (CUSUM crosses boundary)?
                   │
                   └─ YES ──► Determine break date (Chow test)
                              Decision: retrain from break date only,
                              or retrain on full sample with break dummy?

                              Break is temporary (GFC, COVID)?
                               ├─ Yes ──► Retrain on full sample + dummy variable
                               └─ No  ──► Retrain from break date forward only

ANNUALLY: Model review
         │
         ▼
Are there new indicators that were unavailable at initial deployment?
         │
         ├─ Yes ──► Run indicator selection on full sample
         │           Add indicators that pass DM test vs current model
         │
         └─ No ──► Confirm current indicator set is still optimal

         ▼
Has the publication lag for any series changed?
         │
         ├─ Yes ──► Update publication calendar
         │           Re-run pseudo-real-time evaluation
         │
         └─ No ──► No action

RETIREMENT CRITERIA: Retire the model if:
     1. OOS RMSE > 2× AR(1) RMSE for 4 consecutive quarters
     2. No combination of re-estimation strategies recovers performance
     3. A structurally different model (DFM, ML ensemble) significantly
        outperforms on DM test for 2 consecutive evaluation rounds
```

---

## 7. Decision Matrix: Fill Method × Series Type

Use this matrix as a quick reference when onboarding a new indicator.

| Series type | Publication lag | Persistence | Recommended fill |
|-------------|----------------|-------------|-----------------|
| ISM PMI (level) | 1 day | Low | AR1 projection |
| Payrolls (level, '000s) | 4 days | High | Carry forward |
| Initial claims (weekly level) | 5 days | Moderate | AR1 projection |
| IP index (level) | 16 days | High | Carry forward |
| Retail sales (level) | 14 days | Moderate | Carry forward |
| CPI (level) | 16 days | Very high | Carry forward |
| IP growth rate (MoM %) | 16 days | Low | Zero fill |
| Payroll growth (MoM %) | 4 days | Low | Zero fill |
| Treasury yield (daily level) | 0 days | High | Carry forward |
| Oil return (daily %) | 0 days | Very low | Zero fill |

**Note**: Daily financial series (yields, oil prices, equity indices) typically have zero publication lag — they are available in real time. The ragged-edge problem does not apply to them within the current month; it only arises at the monthly aggregation boundary.

---

## 8. Decision Matrix: Model Specification × Target Horizon

| Target | Horizon before release | Recommended specification |
|--------|----------------------|--------------------------|
| Quarterly GDP | ≥60 days | MIDAS with monthly indicators, K=3 |
| Quarterly GDP | 30–60 days | Add weekly claims and PMI |
| Quarterly GDP | <30 days | Full indicator set, regularised MIDAS |
| Monthly CPI | >14 days | AR(1) only (little useful HF data) |
| Monthly CPI | 7–14 days | MIDAS with daily oil/commodity prices |
| Monthly payrolls | >4 days | AR(1) + ADP if available |
| Monthly payrolls | 1–4 days | MIDAS-W with weekly claims |
| Daily RV | Any | MIDAS-RV with intraday RV regressors |

The key insight from this matrix: the **horizon** (how far before the release you forecast) is as important as the model specification. Earlier in the quarter there is less data and simpler models are competitive. Later in the quarter, richer models that incorporate all available releases are superior.

---

## 9. Diagnostic Decision Tree: Common Failures

### Failure: Model beats AR in-sample but not OOS

```
    ├─ Check for data leakage
    │   Did you use final revised data in the evaluation window?
    │   Did you fit the scaler on the full sample?
    │   → Rerun with strict expanding-window protocol
    │
    ├─ Check for overfitting
    │   Is K too large relative to n? (Rule of thumb: K < n/10)
    │   Is n_indicators × K > n/3?
    │   → Reduce K or apply stronger regularisation
    │
    └─ Check for publication lag errors
        Did you assume tighter lags than actual?
        → Rerun pseudo-real-time evaluation with conservative lags
```

### Failure: NLS Beta weight estimation fails to converge

```
    ├─ Multiple starting points
    │   Try (θ₁, θ₂) ∈ {(1,5), (2,2), (5,1), (1,1), (3,1)}
    │
    ├─ Constrain parameter space
    │   Both θ₁, θ₂ ∈ [0.1, 10]
    │   Add bounds to scipy.optimize.minimize
    │
    └─ Fallback: use Beta(1,5) fixed weights
        Monotone-decreasing weighting is a reasonable default;
        if NLS fails, the fixed weights are a reliable fallback
```

### Failure: Rolling RMSE is stable but bias is growing

```
    ├─ Compute bias by forecast origin (early vs late quarter)
    │   If early-quarter bias is larger: ragged-edge fill is systematically wrong
    │   If late-quarter bias is larger: model misses a late-arriving indicator
    │
    ├─ Compute bias by period (pre/post 2020)
    │   Large post-2020 bias: COVID structural break
    │   → Re-estimate excluding 2020 or using break dummy
    │
    └─ Check for systematic data revision
        Are preliminary prints consistently revised in one direction?
        → Use second-release data for training instead of first release
```

---

## 10. Summary Checklist

Before deploying a production nowcasting model, verify each item:

**Specification**
- [ ] Indicators selected based on economic rationale, not statistical fishing
- [ ] Lag depth K chosen based on frequency ratio and publication lags
- [ ] Weight function (Beta vs unrestricted) chosen based on K and series frequency
- [ ] Regularisation method chosen based on n, N, and sparsity goals

**Evaluation**
- [ ] Expanding-window OOS with minimum 8 periods
- [ ] Beats AR(1) benchmark; DM test p-value ≤ 0.10
- [ ] Pseudo-real-time evaluation with ragged-edge simulation
- [ ] Pseudo-real-time RMSE within 30% of naive OOS RMSE

**Production**
- [ ] Ragged-edge fill strategy documented per indicator
- [ ] Publication calendar entries verified against BLS/BEA/Federal Reserve schedules
- [ ] Vintage database schema initialised and tested
- [ ] Re-estimation triggers configured with calibrated thresholds
- [ ] Daily health report tested end-to-end
- [ ] Alert routing tested (critical, high, medium)
- [ ] Recovery procedure documented for each failure mode
- [ ] Baseline RMSE recorded in config for trigger comparison

---

## References

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). The MIDAS touch: Mixed Data Sampling regression models. Working paper, UNC and UCLA.
- Andreou, E., Ghysels, E., & Kourtellos, A. (2010). Regression models with mixed sampling frequencies. *Journal of Econometrics*, 158(2), 246–261.
- Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. *JRSS Series B*, 37(2), 149–163.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *JBES*, 13(3), 253–263.

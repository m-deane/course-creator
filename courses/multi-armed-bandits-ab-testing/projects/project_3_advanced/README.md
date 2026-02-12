# Project 3: Production Regime-Aware Trading Allocator (Advanced)

## What You'll Build

A production-grade commodity allocation system with contextual bandits, regime detection, comprehensive guardrails, logging, monitoring, and automated weekly reporting. This is the system you'd actually deploy for real trading.

You'll implement LinUCB (contextual bandit), build a feature pipeline for regime detection (volatility, trend, seasonality, term structure), create a full guardrail system, add offline evaluation, and generate executive-ready reports.

By the end, you'll have a deployable trading system with institutional-grade risk management.

## Learning Goals

1. Implement LinUCB (contextual bandit with linear models)
2. Build feature pipelines for commodity regime detection
3. Design production guardrail systems (position limits, volatility scaling, correlation monitoring)
4. Add logging and monitoring for live trading
5. Perform offline policy evaluation (counterfactual analysis)
6. Generate automated decision-support reports
7. Understand deployment considerations (scheduling, data sources, alerting)

## Requirements

```bash
pip install numpy pandas matplotlib scipy yfinance
```

Optional for enhanced analysis:
```bash
pip install scikit-learn statsmodels
```

## The Problem

You're deploying a commodity allocator that:
- Runs every Monday morning
- Allocates new capital across 5 commodities
- Must adapt to changing market regimes (trending vs mean-reverting, high vs low volatility)
- Needs institutional-grade safety (position limits, correlation monitoring, volatility scaling)
- Generates weekly reports for portfolio managers
- Logs all decisions for audit and analysis

**This is Project 2 + Regime Awareness + Production Engineering.**

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│               WEEKLY ALLOCATION PIPELINE                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. DATA PIPELINE                                       │
│     └─ Load prices, fundamentals, macro                 │
│                                                          │
│  2. FEATURE PIPELINE                                    │
│     ├─ Volatility regime (VIX, realized vol)            │
│     ├─ Trend signal (momentum, moving averages)         │
│     ├─ Seasonal indicators (month, calendar effects)    │
│     └─ Term structure (contango, backwardation)         │
│                                                          │
│  3. CONTEXTUAL BANDIT (LinUCB)                          │
│     ├─ Context: Feature vector for current week         │
│     ├─ Action: Allocation across commodities            │
│     └─ Reward: Risk-adjusted return                     │
│                                                          │
│  4. GUARDRAIL SYSTEM                                    │
│     ├─ Position limits (max concentration)              │
│     ├─ Volatility scaling (reduce exposure in storms)   │
│     ├─ Correlation monitoring (detect crowding)         │
│     ├─ Drawdown circuit breaker                         │
│     └─ Liquidity constraints                            │
│                                                          │
│  5. EXECUTION & LOGGING                                 │
│     ├─ Generate allocation                              │
│     ├─ Apply guardrails                                 │
│     ├─ Log decision + context                           │
│     └─ Execute trades                                   │
│                                                          │
│  6. MONITORING & REPORTING                              │
│     ├─ Performance tracking                             │
│     ├─ Regime change detection                          │
│     ├─ Anomaly alerts                                   │
│     └─ Weekly executive report                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Project Steps

### Step 1: Feature Pipeline (30 minutes)

Build a complete feature extraction system:

**Features to implement:**
1. **Volatility regime** (4 features)
   - Realized volatility (20-day)
   - VIX level (if available)
   - Volatility percentile (vs 1-year history)
   - Volatility regime indicator (low/medium/high)

2. **Trend signals** (3 features)
   - 20/50-day momentum
   - Moving average crossover
   - Trend strength (ADX-like)

3. **Seasonal indicators** (2 features)
   - Month of year (for seasonality)
   - Days to contract expiry (for roll dynamics)

4. **Term structure** (2 features)
   - Front-month vs next-month spread
   - Contango/backwardation indicator

**Your task:** Complete the `FeaturePipeline` class to compute these features for each commodity.

### Step 2: LinUCB Implementation (40 minutes)

Implement contextual Thompson Sampling / LinUCB:

**Algorithm:**
```python
For each commodity i:
  1. Maintain A_i = X^T X + λI (covariance matrix)
  2. Maintain b_i = X^T y (reward vector)
  3. Estimate θ_i = A_i^{-1} b_i (linear model)
  4. Sample from θ̂_i ~ N(θ_i, α * A_i^{-1})
  5. Predict reward: r_i = x^T θ̂_i
  6. Weight allocation by predicted rewards
```

**Your task:** Complete the `ContextualBanditAllocator` class with LinUCB selection and update logic.

### Step 3: Guardrail System (35 minutes)

Implement institutional-grade safety:

**Required guardrails:**
1. **Position limits**
   - Max 40% in any single commodity
   - Min 5% in all commodities (maintain diversification)

2. **Volatility scaling**
   - When VIX > 30: reduce bandit sleeve from 20% to 10%
   - When realized vol > 95th percentile: reduce allocations proportionally

3. **Correlation monitoring**
   - If portfolio correlation > 0.8: flag warning, increase diversification
   - If all allocations moving same direction: force rebalance

4. **Drawdown circuit breaker**
   - If portfolio down > 15% from peak: freeze bandit, go to equal-weight core
   - Reset when drawdown recovers to < 10%

5. **Liquidity constraints**
   - Never allocate more than 10% of average daily volume
   - Flag illiquid positions for manual review

**Your task:** Implement `GuardrailSystem` class with all 5 guardrails.

### Step 4: Logging & Monitoring (25 minutes)

Add production logging:

**What to log:**
- Every allocation decision with timestamp
- Feature vector (regime context)
- Model coefficients (θ for each commodity)
- Guardrail activations
- Realized returns vs predicted
- Alerts and anomalies

**Monitoring metrics:**
- Prediction error (predicted vs realized returns)
- Model confidence (posterior uncertainty)
- Guardrail hit rate
- Regime transitions
- Portfolio drift from target

**Your task:** Complete `MonitoringSystem` class with logging and alerting.

### Step 5: Offline Evaluation (20 minutes)

Before deploying, validate the policy:

**Evaluation methods:**
1. **Historical backtest** (already done in Project 2)
2. **Regime-stratified analysis**
   - Performance in high-vol vs low-vol regimes
   - Performance in trending vs mean-reverting markets
3. **Counterfactual analysis**
   - What would equal-weight have done?
   - What would best-single-commodity have done?
4. **Stress testing**
   - 2008 financial crisis scenario
   - COVID crash scenario
   - Oil price crash scenario

**Your task:** Run stratified backtest and generate comparison report.

### Step 6: Weekly Report Generation (20 minutes)

Create an executive-ready report:

**Report sections:**
1. **Performance summary** (returns, Sharpe, drawdown)
2. **Current allocation** with regime context
3. **Recent decisions** and why they were made
4. **Guardrail activations** and risk events
5. **Regime analysis** (current market state)
6. **Outlook** (model confidence, recommendations)

**Your task:** Complete `generate_weekly_report()` function.

## Expected Output

### Console Output

```
=== PRODUCTION COMMODITY ALLOCATOR ===
Date: 2024-01-08 09:00:00 EST

FEATURE PIPELINE
  Volatility Regime: MEDIUM (VIX: 18.3, 52nd percentile)
  Trend Signal: WEAK UPTREND (+0.3 momentum)
  Seasonal: January (energy seasonality positive)
  Term Structure: CONTANGO (WTI +2.1%, NatGas +3.4%)

CONTEXTUAL BANDIT ALLOCATION
  LinUCB predictions (weekly expected returns):
    WTI:    +0.8% (confidence: 82%)
    Gold:   +0.4% (confidence: 91%)
    Copper: +1.2% (confidence: 76%)
    NatGas: -0.3% (confidence: 68%)
    Corn:   +0.6% (confidence: 79%)

GUARDRAIL CHECKS
  ✅ Position limits: PASS
  ✅ Volatility scaling: PASS (normal regime)
  ✅ Correlation: 0.42 (PASS, well-diversified)
  ✅ Drawdown: -3.2% (PASS, circuit breaker inactive)
  ✅ Liquidity: PASS (all positions < 5% ADV)

FINAL ALLOCATION
  Core (80%):
    WTI: 16.0%, Gold: 16.0%, Copper: 16.0%, NatGas: 16.0%, Corn: 16.0%

  Bandit (20%):
    WTI: 18.0%, Gold: 15.0%, Copper: 32.0%, NatGas: 8.0%, Corn: 27.0%

  Total Portfolio:
    WTI: 16.4%, Gold: 15.8%, Copper: 19.4%, NatGas: 14.4%, Corn: 21.6%

EXECUTION
  ✅ Orders generated
  ✅ Logged to: ./logs/allocation_20240108.json
  ✅ Report generated: ./reports/weekly_report_20240108.pdf

MONITORING ALERTS
  ⚠️  Copper allocation at 19.4% (approaching 20% warning threshold)
  ℹ️  Term structure in contango (favorable for roll yield)
```

### Weekly Report (PDF)

```
┌─────────────────────────────────────────────────────┐
│        WEEKLY COMMODITY ALLOCATION REPORT           │
│               Week Ending: 2024-01-08               │
└─────────────────────────────────────────────────────┘

PERFORMANCE SUMMARY
  Week:    +1.8% (vs +0.9% equal-weight benchmark)
  Month:   +3.2%
  YTD:     +3.2%
  Sharpe:  1.42 (annualized)
  MaxDD:   -3.2%

CURRENT ALLOCATION & RATIONALE
  Copper (19.4%): Strong trend signal, industrial demand
  Corn (21.6%):   Seasonal tailwind, contango supports
  WTI (16.4%):    Moderate momentum, volatility elevated
  Gold (15.8%):   Safe haven, low conviction
  NatGas (14.4%): Negative signal, reduced exposure

REGIME ANALYSIS
  Current Regime: MEDIUM VOLATILITY + WEAK UPTREND
  Market State:   Commodities in contango (positive roll)
  Risk Level:     MODERATE

  Regime History (4 weeks):
    - 3 weeks ago: LOW VOL / STRONG TREND
    - 2 weeks ago: LOW VOL / WEAK TREND
    - 1 week ago:  MED VOL / WEAK TREND
    - This week:   MED VOL / WEAK UPTREND

GUARDRAIL ACTIVITY
  Position Limits:     0 violations
  Volatility Scaling:  Inactive (normal regime)
  Drawdown Breaker:    Inactive (-3.2% < -15% threshold)

  Copper approaching concentration limit (19.4% of 20% max)

CONFIDENCE & OUTLOOK
  Model Confidence: 78% (average across commodities)
  Prediction Accuracy (4-week): 64% direction, 0.82 correlation

  Recommended Actions:
    - Monitor copper position (close to limit)
    - Watch for regime change (volatility increasing)
    - Term structure supportive, maintain tilts

Next Review: Monday, 2024-01-15 09:00 EST
```

## Success Criteria

Your production system succeeds if:

1. **Performance:** Outperforms equal-weight by 3%+ annually with Sharpe > 1.2
2. **Regime adaptation:** Allocation changes detectably across regimes
3. **Safety:** Zero guardrail violations that cause losses
4. **Monitoring:** All decisions logged, alerts fire appropriately
5. **Robustness:** Passes stress tests (2008, COVID scenarios)
6. **Deployability:** Can run weekly with minimal manual intervention

## File Structure

```
project_3_advanced/
├── README.md              # This file
├── starter_code.py        # Your skeleton (~150 lines of TODOs)
├── solution.py            # Reference implementation (~200 lines)
└── deploy.md              # Deployment guide
```

## Extensions (Production Hardening)

1. **Data validation:** Check for missing data, outliers, stale prices
2. **Failover:** If model fails, fall back to equal-weight core
3. **Versioning:** Track model versions, allow rollback
4. **A/B testing:** Run new model alongside current in parallel
5. **Transaction costs:** Optimize rebalancing frequency
6. **Tax optimization:** Minimize short-term gains

## Deployment Considerations

See `deploy.md` for:
- Scheduling weekly runs (cron, Airflow, prefect)
- Configuration management (YAML configs, environment variables)
- Data source management (yfinance, FRED, proprietary feeds)
- Monitoring setup (Grafana, custom dashboards)
- Alert routing (email, Slack, PagerDuty)

## Next Steps

After completing this project, you're ready to:
1. Deploy live with paper trading
2. Build similar systems for other asset classes
3. Integrate with execution systems (APIs, brokers)
4. Scale to multi-strategy portfolios

---

**This is the culmination of the course.** You've built a production bandit system from scratch with institutional-grade risk management. This code structure works for real trading.

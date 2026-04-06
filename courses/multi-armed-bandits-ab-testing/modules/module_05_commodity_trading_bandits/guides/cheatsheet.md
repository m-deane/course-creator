# Commodity Trading Bandits Cheatsheet

> **Reading time:** ~20 min | **Module:** 05 — Commodity Trading Bandits | **Prerequisites:** Module 4


Quick reference for bandit-based commodity allocation systems.


<div class="callout-key">

**Key Concept Summary:** Quick reference for bandit-based commodity allocation systems.

</div>

## The 6-Step Accumulator Bandit Playbook

```
1. Choose Core Allocation
   → Equal-weight or strategic, rebalanced monthly
   → 60-80% of portfolio

2. Set Bandit Sleeve
   → Start with 20% of portfolio
   → Can adjust to 10-40% based on risk tolerance

3. Define Arms (5-10 total)
   → Broad: Energy, Metals, Grains, Softs, Livestock
   → Granular: WTI, Gold, Copper, NatGas, Corn, etc.

4. Weekly Execution
   → Core: Execute planned contribution
   → Bandit: Thompson Sampling allocation
   → Record prices and returns

5. Track Right Score
   → Design reward matching your goal
   → Not just raw returns!

6. Apply Guardrails
   → Position limits, min allocation, tilt speed
   → These prevent self-sabotage
```

## Two-Wallet Framework

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


```
┌─────────────────────────────────────┐
│ Total Portfolio                     │
├─────────────────────────────────────┤
│ Core (80%)                          │
│ • Equal-weight or strategic         │
│ • Rebalance monthly                 │
│ • Provides stability                │
├─────────────────────────────────────┤
│ Bandit Sleeve (20%)                 │
│ • Thompson Sampling                 │
│ • Rebalance weekly                  │
│ • Adaptive tilts                    │
└─────────────────────────────────────┘

Total allocation = 0.8 * core + 0.2 * bandit
```

## Reward Function Comparison

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


| Reward Type | Formula | What It Trains | Use When |
|-------------|---------|----------------|----------|
| **Raw Returns** | `r_t` | Trend chasing | NEVER |
| **Sharpe Ratio** | `r / σ` | Risk aversion | Minimize volatility goal |
| **Risk-Adjusted** | `r / σ - λ * DD` | Balance return and stability | General accumulation |
| **Regret-Relative** | `r - r_best` | Learn rankings | Relative performance goal |
| **Stability-Weighted** | `r - λ * turnover` | Minimize churn | Tax/cost sensitive |
| **Thesis-Aligned** | `r - λ * ||w - w_strategic||` | Stay near thesis | Strategic overlay |

**Key insight:** Your reward IS your strategy.

## Guardrail Parameters

| Guardrail | Conservative | Moderate | Aggressive |
|-----------|--------------|----------|------------|
| **Max Position** | 30% | 40% | 50% |
| **Min Position** | 10% | 5% | 2% |
| **Max Tilt Speed** | 10% | 15% | 25% |
| **Core %** | 85% | 80% | 70% |
| **Bandit %** | 15% | 20% | 30% |
| **VIX Threshold** | 25 | 30 | 35 |
| **Max Sector** | 40% | 50% | 60% |

**Recommendation:** Start conservative, loosen after validating.

## Common Commodity Arms

### Broad Sectors (5 arms)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
arms = ['Energy', 'Metals', 'Grains', 'Softs', 'Livestock']

sector_mapping = {
    'Energy': ['WTI', 'Brent', 'NatGas', 'Gasoline'],
    'Metals': ['Gold', 'Silver', 'Copper', 'Platinum'],
    'Grains': ['Corn', 'Soybeans', 'Wheat'],
    'Softs': ['Coffee', 'Sugar', 'Cotton'],
    'Livestock': ['Cattle', 'Hogs']
}
```

</div>

### Granular Commodities (8-10 arms)
<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
arms = [
    'WTI',      # Energy
    'NatGas',   # Energy
    'Gold',     # Metals
    'Copper',   # Metals
    'Corn',     # Grains
    'Soybeans', # Grains
    'Coffee',   # Softs
    'Cattle'    # Livestock
]
```

</div>

### Strategy Factors (4-6 arms)
```python
arms = [
    'Momentum',        # Top trending commodities
    'Mean_Reversion',  # Oversold commodities
    'Carry',           # Highest roll yield
    'Seasonality',     # Seasonal patterns
    'Low_Volatility',  # Lowest vol commodities
    'Quality'          # Best fundamentals
]
```

## Regime Features for Contextual Bandits

### Feature Engineering Recipes

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# 1. Realized Volatility (20-day)
vol = returns.rolling(20).std() * np.sqrt(252)

# 2. Term Structure Slope
ts_slope = (back_month - front_month) / front_month

# 3. Trend Strength
fast_ma = prices.rolling(20).mean()
slow_ma = prices.rolling(50).mean()
trend = (fast_ma - slow_ma) / slow_ma

# 4. Risk Sentiment
sentiment = sp500.rolling(5).mean() - (vix - 20) / 100

# 5. Seasonal Indicator (month-based)
seasonal = seasonal_patterns[commodity][date.month]

# 6. Inventory Percentile
inv_pct = percentileofscore(historical_inv, current_inv)
```

</div>

### Simple Regime Classifier

```python
def classify_regime(features):
    vol = features['volatility']
    trend = features['trend']

    if vol > 0.25:
        vol_state = 'HIGH_VOL'
    elif vol < 0.15:
        vol_state = 'LOW_VOL'
    else:
        vol_state = 'MED_VOL'

    if trend > 0.10:
        trend_state = 'UPTREND'
    elif trend < -0.10:
        trend_state = 'DOWNTREND'
    else:
        trend_state = 'NEUTRAL'

    return f"{vol_state}_{trend_state}"
```

## Quick Implementation Templates

### Basic Thompson Sampling Bandit

```python
class ThompsonSamplingBandit:
    def __init__(self, K, prior_mean=0.001, prior_std=0.02):
        self.K = K
        self.means = np.full(K, prior_mean)
        self.stds = np.full(K, prior_std)
        self.n = np.zeros(K)

    def select(self):
        samples = np.random.normal(self.means, self.stds)
        return np.argmax(samples)

    def update(self, arm, reward):
        self.n[arm] += 1
        lr = 1 / (self.n[arm] + 1)
        self.means[arm] = (1 - lr) * self.means[arm] + lr * reward
        self.stds[arm] = self.stds[arm] / np.sqrt(1 + self.n[arm])
```

### Two-Wallet System

```python
class TwoWalletBandit:
    def __init__(self, K, core_pct=0.8, bandit_pct=0.2):
        self.K = K
        self.core_pct = core_pct
        self.bandit_pct = bandit_pct
        self.bandit = ThompsonSamplingBandit(K)

    def get_weights(self):
        core = np.ones(self.K) / self.K  # Equal-weight
        bandit = self.bandit.get_allocation()  # Thompson
        return self.core_pct * core + self.bandit_pct * bandit
```

### Guardrail System

```python
def apply_guardrails(weights, old_weights=None, params=None):
    """Apply all guardrails in sequence."""
    if params is None:
        params = {
            'max_pos': 0.40,
            'min_pos': 0.05,
            'max_speed': 0.15
        }

    # 1. Position limits
    weights = np.clip(weights, 0, params['max_pos'])
    weights = weights / weights.sum()

    # 2. Minimum allocation
    weights = np.maximum(weights, params['min_pos'])
    weights = weights / weights.sum()

    # 3. Tilt speed limit
    if old_weights is not None:
        change = weights - old_weights
        change = np.clip(change, -params['max_speed'], params['max_speed'])
        weights = old_weights + change
        weights = weights / weights.sum()

    return weights
```

### Regime-Aware Bandit

```python
class RegimeAwareBandit:
    def __init__(self, K, classifier):
        self.K = K
        self.classifier = classifier
        self.regime_bandits = {}

    def get_bandit(self, regime):
        if regime not in self.regime_bandits:
            self.regime_bandits[regime] = ThompsonSamplingBandit(self.K)
        return self.regime_bandits[regime]

    def select(self, features):
        regime = self.classifier(features)
        bandit = self.get_bandit(regime)
        return bandit.select(), regime

    def update(self, regime, arm, reward):
        bandit = self.get_bandit(regime)
        bandit.update(arm, reward)
```

## Decision Flowchart

```
START: Designing Commodity Bandit Allocator
    ↓
Q1: What's your goal?
    → Steady accumulation → Use stability-weighted reward
    → Maximize returns → Use risk-adjusted reward
    → Beat benchmark → Use regret-relative reward
    ↓
Q2: How much capital?
    → < $100K → Use broad sectors (5 arms)
    → $100K-$1M → Use granular commodities (8-10 arms)
    → > $1M → Use strategy factors or granular
    ↓
Q3: Risk tolerance?
    → Conservative → 85% core, 15% bandit, tight guardrails
    → Moderate → 80% core, 20% bandit, moderate guardrails
    → Aggressive → 70% core, 30% bandit, loose guardrails
    ↓
Q4: Market regimes matter?
    → Yes → Use contextual bandit with regime features
    → No → Use basic Thompson Sampling
    ↓
Q5: Production deployment?
    → Yes → Add ALL guardrails, test on historical data
    → No (research) → Can skip some guardrails, focus on learning
    ↓
IMPLEMENT & BACKTEST
```

## Common Pitfalls Checklist

- [ ] **Reward = raw returns?** → Change to risk-adjusted
- [ ] **No minimum allocation?** → Add 5% min per arm
- [ ] **No position limits?** → Add 40% max per arm
- [ ] **No tilt speed limit?** → Add 15% max change
- [ ] **Pure bandit (no core)?** → Add 60-80% core allocation
- [ ] **Too many arms (>15)?** → Reduce to 5-10
- [ ] **Too few observations per regime?** → Simplify regime definitions
- [ ] **Ignoring transaction costs?** → Add turnover penalty to reward
- [ ] **Same timescale for core and bandit?** → Core monthly, bandit weekly
- [ ] **No volatility dampening?** → Add VIX-based adjustment

## Performance Metrics

Track these metrics for your bandit allocator:

```python
# Returns
total_return = (final_value - initial_value) / initial_value
annualized_return = (1 + total_return) ** (252 / n_days) - 1

# Risk
volatility = returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / volatility
max_drawdown = (cumulative - cumulative.cummax()).min()

# Trading
turnover = allocation_changes.abs().sum() / n_periods
transaction_costs = turnover * cost_per_trade

# Regret
cumulative_regret = best_arm_cumulative - your_cumulative

# Exploration
unique_arms_selected = len(set(arm_selections))
arm_concentration = max(arm_frequencies)
```

## Connections to Other Courses

| Course | Integration Point |
|--------|-------------------|
| **Bayesian Commodity Forecasting** | Use forecasts as contextual features |
| **Hidden Markov Models** | Advanced regime detection (HMM states as contexts) |
| **GenAI for Commodities** | LLM-based feature engineering (news sentiment, supply shocks) |
| **Panel Regression** | Cross-commodity patterns as features |

## Quick Debugging Guide

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Allocation always equal-weight | Bandit % too low | Increase to 20% |
| Extreme concentration | No position limits | Add max_pos=0.40 |
| Constant churning | No tilt speed limit | Add max_speed=0.15 |
| Underperforms benchmark | Wrong reward function | Match reward to goal |
| High transaction costs | Too much turnover | Add stability penalty |
| Zeros out arms quickly | No minimum allocation | Add min_pos=0.05 |
| Ignores regime changes | Non-contextual | Add regime features |
| Blows up in crisis | No volatility dampening | Add VIX-based damping |

## Code Snippets Library

### Load Real Commodity Data

```python
import yfinance as yf

tickers = {
    'WTI': 'CL=F',
    'Gold': 'GC=F',
    'Copper': 'HG=F',
    'NatGas': 'NG=F',
    'Corn': 'ZC=F'
}

data = yf.download(
    list(tickers.values()),
    start='2020-01-01',
    end='2024-01-01'
)['Adj Close']

returns = data.pct_change().dropna()
```

### Compute Risk-Adjusted Reward

```python
def risk_adjusted_reward(returns, window=20, lambda_dd=2.0):
    """Sharpe ratio with drawdown penalty."""
    mean_ret = returns.mean()
    vol = returns.std()
    cumulative = (1 + returns).cumprod()
    drawdown = (cumulative / cumulative.cummax() - 1).min()

    sharpe = mean_ret / (vol + 1e-6)
    return sharpe - lambda_dd * abs(drawdown)
```

### Backtest Framework

```python
def backtest_bandit(data, bandit, reward_func, rebalance_freq='W'):
    """Simple backtest framework."""
    results = []

    for date, returns in data.resample(rebalance_freq):
        # Get allocation
        weights = bandit.get_weights()

        # Compute portfolio return
        portfolio_return = (weights * returns).sum()

        # Compute reward
        reward = reward_func(returns)

        # Update bandit
        bandit.update(reward)

        results.append({
            'date': date,
            'return': portfolio_return,
            'weights': weights.copy()
        })

    return pd.DataFrame(results)
```

## Further Reading

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


- [Accumulator Bandit Playbook](01_accumulator_bandit_playbook.md) - Full system overview
- [Reward Design](02_reward_design_commodities.md) - Critical reward function details
- [Guardrails](03_guardrails_and_safety.md) - Safety constraints
- [Regime-Aware](04_regime_aware_allocation.md) - Contextual extensions

## Quick Start

**Fastest path to working system:**

1. Run [Two-Wallet Framework Notebook](../notebooks/01_two_wallet_framework.ipynb)
2. Modify with your commodities and parameters
3. Add guardrails from this cheatsheet
4. Backtest on historical data
5. Deploy with monitoring

**Total time:** 2-3 hours for production-ready system.


---

## Conceptual Practice Questions

**Practice Question 1:** How does non-stationarity in commodity markets affect bandit algorithm assumptions?

**Practice Question 2:** What risk management constraints should be layered on top of a bandit-based allocation system?



---

## Cross-References

<a class="link-card" href="./01_accumulator_bandit_playbook.md">
  <div class="link-card-title">01 Accumulator Bandit Playbook</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_accumulator_bandit_playbook.md">
  <div class="link-card-title">01 Accumulator Bandit Playbook — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_reward_design_commodities.md">
  <div class="link-card-title">02 Reward Design Commodities</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_reward_design_commodities.md">
  <div class="link-card-title">02 Reward Design Commodities — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_guardrails_and_safety.md">
  <div class="link-card-title">03 Guardrails And Safety</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_guardrails_and_safety.md">
  <div class="link-card-title">03 Guardrails And Safety — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


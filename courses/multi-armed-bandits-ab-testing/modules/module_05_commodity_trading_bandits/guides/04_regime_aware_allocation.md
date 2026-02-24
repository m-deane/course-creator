# Regime-Aware Allocation

## In Brief

Market regimes change. A bandit optimized for trending markets fails in mean-reverting regimes. Contextual bandits solve this by conditioning allocation on observable regime features, learning separate strategies for different market states.

> 💡 **Key Insight:** Market regimes are contexts. Contextual bandits learn regime-dependent strategies.

Instead of one universal allocation policy, learn:
- Allocation for high-volatility regimes
- Allocation for low-volatility regimes
- Allocation for contango regimes
- Allocation for backwardation regimes
- Allocation for risk-on vs risk-off

The bandit adapts in real-time as regimes shift.

## Visual Explanation

```
┌────────────────────────────────────────────────────────────────┐
│  Regime-Aware Contextual Bandit for Commodities               │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  STEP 1: Observe Market Features                               │
│  ┌──────────────────────────────────────────────────┐          │
│  │ VIX: 32 (HIGH)                                   │          │
│  │ WTI term structure: +2% (CONTANGO)              │          │
│  │ 50-day trend: +15% (TRENDING UP)                │          │
│  │ S&P 500: +0.5% today (RISK-ON)                  │          │
│  └──────────────────────────────────────────────────┘          │
│                          ↓                                      │
│  STEP 2: Classify Regime                                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Regime: HIGH_VOL_CONTANGO_TRENDING               │          │
│  │                                                  │          │
│  │ Historical: This regime occurred 23% of time    │          │
│  │ In this regime:                                 │          │
│  │   - Energy outperforms (trending)               │          │
│  │   - Metals underperform (risk assets weak)      │          │
│  │   - Grains neutral                              │          │
│  └──────────────────────────────────────────────────┘          │
│                          ↓                                      │
│  STEP 3: Retrieve Regime-Specific Bandit                       │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Beliefs for HIGH_VOL_CONTANGO_TRENDING:         │          │
│  │   WTI: μ=0.008, σ=0.015 (strong)                │          │
│  │   Gold: μ=-0.002, σ=0.020 (weak)                │          │
│  │   Copper: μ=0.001, σ=0.012 (neutral)            │          │
│  │   NatGas: μ=0.005, σ=0.025 (moderate)           │          │
│  │   Corn: μ=0.000, σ=0.010 (neutral)              │          │
│  └──────────────────────────────────────────────────┘          │
│                          ↓                                      │
│  STEP 4: Generate Allocation                                   │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Allocation for this regime:                     │          │
│  │   WTI: 35% (tilted up)                          │          │
│  │   Gold: 10% (tilted down)                       │          │
│  │   Copper: 20%                                   │          │
│  │   NatGas: 25%                                   │          │
│  │   Corn: 10%                                     │          │
│  └──────────────────────────────────────────────────┘          │
│                          ↓                                      │
│  STEP 5: Observe Outcome & Update ONLY This Regime             │
│  ┌──────────────────────────────────────────────────┐          │
│  │ Week's returns: WTI +3%, Gold -1%, ...          │          │
│  │                                                  │          │
│  │ Update beliefs for HIGH_VOL_CONTANGO_TRENDING   │          │
│  │ (Other regime beliefs unchanged)                │          │
│  └──────────────────────────────────────────────────┘          │
│                                                                 │
└────────────────────────────────────────────────────────────────┘

Key difference from non-contextual:
- Non-contextual: One set of beliefs for all market conditions
- Contextual: Separate beliefs for each regime
- Result: Adapt allocation as regimes change
```

## Formal Definition

**Contextual Bandit for Commodities:**

At each time `t`:
1. Observe context `x_t ∈ X` (market regime features)
2. Choose allocation `a_t` from arm set `A = {1, ..., K}`
3. Receive reward `r_t(a_t, x_t)`
4. Update beliefs `θ(x_t)` for the observed context only

**Regime Context Vector:**

```
x_t = [
    realized_volatility_t,
    term_structure_slope_t,
    trend_strength_t,
    risk_sentiment_t,
    seasonal_indicator_t,
    inventory_level_t
]
```

**Contextual Thompson Sampling:**

For each context `x`, maintain separate Beta/Gaussian parameters:
- `μ_k(x)`: Mean return for arm `k` in regime `x`
- `σ_k(x)`: Uncertainty for arm `k` in regime `x`

Sample from context-specific posteriors:
```
θ_k ~ N(μ_k(x_t), σ_k(x_t)²)
a_t = argmax_k θ_k
```

Update only the beliefs for context `x_t`:
```
μ_k(x_t) ← (1-α) μ_k(x_t) + α * r_t
σ_k(x_t) ← σ_k(x_t) / sqrt(1 + n_k(x_t))
```

## Intuitive Explanation

Think of a trader with separate playbooks for different market environments:
- **Bull market playbook**: Aggressive tilts, ride winners
- **Bear market playbook**: Defensive, favor safe havens
- **Choppy market playbook**: Mean-reversion, quick profits
- **Crisis playbook**: Liquidity first, returns second

The contextual bandit is the same idea, but:
1. Market conditions automatically trigger the right playbook
2. Each playbook learns from its own historical performance
3. No human judgment needed to switch playbooks

**Analogy:** A restaurant with different menus for lunch vs dinner vs weekend brunch. Each menu is optimized for its context (time of day, customer mix, kitchen capacity).

## Feature Engineering for Commodity Regimes

### Feature 1: Realized Volatility

**Why it matters:** High-vol and low-vol regimes require different strategies.

```python
def compute_realized_volatility(returns, window=20):
    """
    Rolling realized volatility.

    Args:
        returns: Daily returns
        window: Lookback window in days
    """
    return returns.rolling(window).std() * np.sqrt(252)
```

**Regime classification:**
- Low vol: < 15% annualized
- Medium vol: 15-25%
- High vol: > 25%

**Commodity-specific insights:**
- High vol → favor mean-reversion strategies
- Low vol → favor momentum strategies

### Feature 2: Term Structure Slope

**Why it matters:** Contango vs backwardation affects carry returns.

```python
def compute_term_structure_slope(front_month, back_month):
    """
    Term structure slope (proxy using 2 contracts).

    Args:
        front_month: Nearby futures price
        back_month: Deferred futures price (e.g., 6 months out)
    """
    return (back_month - front_month) / front_month
```

**Regime classification:**
- Steep contango: slope > +5%
- Mild contango: 0% to +5%
- Backwardation: slope < 0%

**Commodity-specific insights:**
- Backwardation → favor long positions (positive roll yield)
- Contango → consider avoiding or short (negative roll yield)

### Feature 3: Trend Strength

**Why it matters:** Trending vs mean-reverting markets need different allocations.

```python
def compute_trend_strength(prices, short_window=20, long_window=50):
    """
    Measure trend strength using dual moving averages.

    Args:
        prices: Price series
        short_window: Fast MA window
        long_window: Slow MA window
    """
    fast_ma = prices.rolling(short_window).mean()
    slow_ma = prices.rolling(long_window).mean()
    return (fast_ma - slow_ma) / slow_ma
```

**Regime classification:**
- Strong uptrend: > +10%
- Weak trend: -5% to +5%
- Strong downtrend: < -5%

**Commodity-specific insights:**
- Strong trends → momentum strategies work
- Weak trends → mean-reversion strategies work

### Feature 4: Risk Sentiment

**Why it matters:** Risk-on vs risk-off affects commodity correlations.

```python
def compute_risk_sentiment(sp500_returns, vix_level):
    """
    Measure broad risk sentiment.

    Args:
        sp500_returns: S&P 500 daily returns
        vix_level: Current VIX level
    """
    # Positive returns + low VIX = risk-on
    # Negative returns + high VIX = risk-off
    sentiment_score = sp500_returns.rolling(5).mean() - (vix_level - 20) / 100
    return sentiment_score
```

**Regime classification:**
- Risk-on: score > 0
- Risk-off: score < 0

**Commodity-specific insights:**
- Risk-on → industrial commodities (Copper) outperform
- Risk-off → safe havens (Gold) outperform

### Feature 5: Seasonality Indicator

**Why it matters:** Agricultural commodities have strong seasonal patterns.

```python
def get_seasonal_indicator(date, commodity):
    """
    Seasonal strength indicator by month.

    Args:
        date: Current date
        commodity: Commodity name
    """
    # Simplified seasonal patterns
    seasonal_patterns = {
        'Corn': {  # Strong in summer (growing season risk)
            1: -0.5, 2: -0.3, 3: 0.0, 4: 0.3, 5: 0.5,
            6: 0.8, 7: 1.0, 8: 0.5, 9: 0.0, 10: -0.3,
            11: -0.5, 12: -0.5
        },
        'NatGas': {  # Strong in winter (heating demand)
            1: 1.0, 2: 0.8, 3: 0.3, 4: 0.0, 5: -0.3,
            6: -0.5, 7: -0.5, 8: -0.3, 9: 0.0, 10: 0.3,
            11: 0.5, 12: 0.8
        },
        'WTI': {  # Strong in summer (driving season)
            1: 0.0, 2: 0.0, 3: 0.3, 4: 0.5, 5: 0.8,
            6: 1.0, 7: 0.8, 8: 0.5, 9: 0.0, 10: -0.3,
            11: -0.3, 12: 0.0
        }
    }
    month = date.month
    return seasonal_patterns.get(commodity, {}).get(month, 0.0)
```

### Feature 6: Inventory Level

**Why it matters:** Supply/demand imbalance signals from storage.

```python
def compute_inventory_percentile(current_level, historical_levels):
    """
    Inventory level as percentile of historical range.

    Args:
        current_level: Current inventory (e.g., EIA crude stocks)
        historical_levels: Historical inventory series (e.g., 5 years)
    """
    return percentileofscore(historical_levels, current_level)
```

**Regime classification:**
- Low inventory: < 20th percentile (bullish)
- Normal inventory: 20-80th percentile
- High inventory: > 80th percentile (bearish)

## Code Implementation

```python
import numpy as np
import pandas as pd
from collections import defaultdict

class RegimeAwareBandit:
    """
    Contextual bandit that learns regime-specific allocations.
    """
    def __init__(
        self,
        arms,
        regime_classifier,
        prior_mean=0.001,
        prior_std=0.02
    ):
        self.arms = arms
        self.K = len(arms)
        self.regime_classifier = regime_classifier

        # Separate beliefs for each regime
        # regime_beliefs[regime_id][arm_id] = (mean, std, count)
        self.regime_beliefs = defaultdict(
            lambda: {
                i: {'mean': prior_mean, 'std': prior_std, 'count': 0}
                for i in range(self.K)
            }
        )

    def get_regime(self, features):
        """Classify current market regime from features."""
        return self.regime_classifier(features)

    def get_allocation(self, features):
        """
        Get allocation for current regime using Thompson Sampling.

        Args:
            features: Dict of market features
        """
        regime = self.get_regime(features)
        beliefs = self.regime_beliefs[regime]

        # Thompson Sampling: sample from each arm's posterior
        samples = np.array([
            np.random.normal(beliefs[i]['mean'], beliefs[i]['std'])
            for i in range(self.K)
        ])

        # Convert to weights via softmax
        exp_samples = np.exp(samples - samples.max())
        weights = exp_samples / exp_samples.sum()

        return weights, regime

    def update(self, features, returns):
        """
        Update beliefs for the observed regime only.

        Args:
            features: Market features when allocation was made
            returns: Realized returns for each arm
        """
        regime = self.get_regime(features)
        beliefs = self.regime_beliefs[regime]

        for i in range(self.K):
            # Incremental mean update
            beliefs[i]['count'] += 1
            n = beliefs[i]['count']
            lr = 1 / (n + 1)
            beliefs[i]['mean'] = (
                (1 - lr) * beliefs[i]['mean'] +
                lr * returns[i]
            )

            # Uncertainty reduction
            beliefs[i]['std'] = beliefs[i]['std'] / np.sqrt(1 + n)


class SimpleRegimeClassifier:
    """
    Simple rule-based regime classifier for commodities.
    """
    def __call__(self, features):
        """
        Classify regime based on volatility and trend.

        Args:
            features: Dict with keys 'volatility', 'trend', etc.
        """
        vol = features.get('volatility', 0.15)
        trend = features.get('trend', 0.0)

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


# Example usage
if __name__ == "__main__":
    commodities = ['WTI', 'Gold', 'Copper', 'NatGas', 'Corn']
    classifier = SimpleRegimeClassifier()
    bandit = RegimeAwareBandit(commodities, classifier)

    # Week 1: High vol, uptrend
    features_1 = {'volatility': 0.30, 'trend': 0.15}
    weights_1, regime_1 = bandit.get_allocation(features_1)
    print(f"Week 1 - Regime: {regime_1}")
    print(f"Allocation: {dict(zip(commodities, weights_1))}")

    # Observe returns
    returns_1 = np.array([0.03, -0.01, 0.02, 0.04, 0.01])
    bandit.update(features_1, returns_1)

    # Week 2: Low vol, neutral
    features_2 = {'volatility': 0.12, 'trend': 0.02}
    weights_2, regime_2 = bandit.get_allocation(features_2)
    print(f"\nWeek 2 - Regime: {regime_2}")
    print(f"Allocation: {dict(zip(commodities, weights_2))}")
    print(f"(Notice: Different regime = different allocation strategy)")
```

## Regime Detection Strategies

### Strategy 1: Rule-Based (Simple, Interpretable)

```python
def rule_based_classifier(features):
    """Hand-crafted rules from domain knowledge."""
    vol = features['volatility']
    ts_slope = features['term_structure']
    trend = features['trend']

    if vol > 0.30 and trend > 0.10:
        return 'CRISIS_TRENDING'
    elif vol < 0.15 and abs(trend) < 0.05:
        return 'CALM_RANGEBOUND'
    elif ts_slope < -0.02:
        return 'BACKWARDATION'
    elif ts_slope > 0.05:
        return 'STEEP_CONTANGO'
    else:
        return 'NORMAL'
```

**Pros:** Interpretable, uses domain knowledge
**Cons:** Requires manual tuning, hard boundaries

### Strategy 2: K-Means Clustering (Data-Driven)

```python
from sklearn.cluster import KMeans

def kmeans_classifier(features, model, scaler):
    """
    Data-driven regime detection via clustering.

    Args:
        features: Dict of features
        model: Pre-trained KMeans model
        scaler: Pre-trained StandardScaler
    """
    X = np.array([
        features['volatility'],
        features['term_structure'],
        features['trend'],
        features['risk_sentiment']
    ]).reshape(1, -1)

    X_scaled = scaler.transform(X)
    regime_id = model.predict(X_scaled)[0]
    return f"REGIME_{regime_id}"
```

**Pros:** Discovers regimes from data
**Cons:** Regimes may not be interpretable, needs training

### Strategy 3: Hidden Markov Model (Sequential, Probabilistic)

```python
from hmmlearn import hmm

def hmm_classifier(feature_history, model):
    """
    HMM-based regime detection (sequential).

    Args:
        feature_history: Recent feature vectors (T x D)
        model: Pre-trained GaussianHMM
    """
    # Predict most likely hidden state given recent observations
    states = model.predict(feature_history)
    current_regime = states[-1]
    return f"HMM_STATE_{current_regime}"
```

**Pros:** Accounts for regime persistence and transitions
**Cons:** More complex, requires training

See **Hidden Markov Models course** in this repo for full HMM implementation.

## Practical Examples

### Example 1: Energy Commodity in Different Regimes

```python
# Historical performance of WTI:

# REGIME: LOW_VOL_UPTREND (2017-2018)
# - Avg weekly return: +0.5%
# - Volatility: 12%
# - Best arms: WTI (momentum), Copper (growth)
# → Bandit learns to overweight WTI in this regime

# REGIME: HIGH_VOL_DOWNTREND (2020 COVID)
# - Avg weekly return: -2.0%
# - Volatility: 45%
# - Best arms: Gold (safe haven), Grains (defensive)
# → Bandit learns to underweight WTI in this regime

# REGIME: BACKWARDATION (2021 reopening)
# - Avg weekly return: +1.2%
# - Volatility: 20%
# - Best arms: WTI (positive roll), Copper (recovery)
# → Bandit learns roll yield is meaningful
```

### Example 2: Multi-Regime Portfolio

```python
# Portfolio with regime-aware allocation:

# January 2024: HIGH_VOL_NEUTRAL
# → Allocation: 15% WTI, 30% Gold, 25% Copper, 20% NatGas, 10% Corn

# March 2024: LOW_VOL_UPTREND (regime shift)
# → Allocation: 35% WTI, 15% Gold, 30% Copper, 10% NatGas, 10% Corn
# → Shifted toward cyclical commodities

# May 2024: BACKWARDATION_ENERGY (another shift)
# → Allocation: 40% WTI, 10% Gold, 20% Copper, 25% NatGas, 5% Corn
# → Captured roll yield in energy
```

## Common Pitfalls

### Pitfall 1: Too Many Regimes
**What happens:** Each regime has too few observations to learn.

**Example:** 20 regimes × 5 arms = 100 parameters to learn. With 200 weeks of data, each regime-arm pair has 10 observations. Undersampled.

**Fix:** Start with 3-5 regimes. Use broader definitions.

### Pitfall 2: Regime Overfitting
**What happens:** Regimes are defined by outcomes (returns) instead of observables (features).

**Example:** "Good regime" = weeks WTI was up, "Bad regime" = weeks WTI was down. This is cheating.

**Fix:** Regimes must be defined by observable features BEFORE seeing returns.

### Pitfall 3: Ignoring Regime Persistence
**What happens:** Treat regime switches as memoryless, but regimes persist.

**Example:** High vol regime typically lasts 4-8 weeks. Relearning every week wastes data.

**Fix:** Use HMM or add regime history as feature.

### Pitfall 4: No Fallback
**What happens:** Encounter a never-before-seen regime, bandit has no beliefs.

**Example:** COVID crash regime was unprecedented. Pure contextual bandit would use prior (weak).

**Fix:** Maintain a "default" non-contextual bandit as fallback for novel regimes.

## Connections

### Builds On
- **Module 3**: Contextual bandits (general framework)
- **Module 1**: Thompson Sampling (per-regime allocation)

### Leads To
- **Hidden Markov Models course**: Advanced regime detection
- **Bayesian Commodity Forecasting course**: Enhanced feature engineering

### Related Concepts
- **Regime switching models**: Econometric foundations
- **State-space models**: Latent state dynamics
- **Transfer learning**: Knowledge across related regimes

## Practice Problems

### Problem 1: Design Regime Classifier
Using the 6 features (volatility, term structure, trend, risk sentiment, seasonality, inventory):
1. Design a rule-based classifier with 4 regimes
2. Justify each regime with domain knowledge
3. Estimate how often each regime occurs historically

### Problem 2: Feature Importance
You have 3 years of commodity data. Test which features best predict regime-dependent returns:
1. Volatility
2. Term structure
3. Trend
4. Combination

How would you measure "best"?

### Problem 3: Cold Start Problem
You've just defined a new regime ("EXTREME_BACKWARDATION") that's only occurred 5 times historically. How do you:
1. Initialize beliefs for this regime?
2. Balance exploration vs exploitation with limited data?
3. Decide when to merge with similar regime vs keep separate?

---

**Next Steps:**
- Read [Cheatsheet](cheatsheet.md) for quick reference
- Try [Regime-Aware Commodity Bandit Notebook](../notebooks/03_regime_commodity_bandit.ipynb)
- Explore **Hidden Markov Models course** for advanced regime detection

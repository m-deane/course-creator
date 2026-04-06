# Feature Engineering for Contextual Bandits

> **Reading time:** ~20 min | **Module:** 03 — Contextual Bandits | **Prerequisites:** Module 2


## In Brief


<div class="callout-key">

**Key Concept Summary:** Good context features are the difference between a contextual bandit that learns useful patterns and one that just adds complexity. For commodity markets, effective features capture market regime (...

</div>

Good context features are the difference between a contextual bandit that learns useful patterns and one that just adds complexity. For commodity markets, effective features capture market regime (volatility, term structure), seasonality, and macro conditions in a format that enables learning while avoiding overfitting.

## Key Insight

**The feature engineering paradox for bandits:**
- Too few features → can't capture regime-dependent patterns
- Too many features → overfitting, slow learning, exploration becomes expensive

<div class="callout-insight">

**Insight:** Contextual bandits bridge the gap between simple A/B tests and full reinforcement learning. They personalize decisions based on observable features without needing to model state transitions.

</div>


**The sweet spot:** 3-7 features that capture the core drivers of regime-dependent performance. Unlike supervised learning where you can throw in 50 features and let regularization handle it, bandits pay an exploration cost for each dimension.

## Visual Explanation

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


```
FEATURE SELECTION FRAMEWORK:

Good Features:                 Bad Features:
┌────────────────────┐        ┌──────────────────────┐
│ ✓ Predictive       │        │ ✗ Unpredictive       │
│ ✓ Observable       │        │ ✗ Future leakage     │
│ ✓ Stable           │        │ ✗ Noisy/erratic      │
│ ✓ Interpretable    │        │ ✗ Redundant          │
└────────────────────┘        └──────────────────────┘

COMMODITY CONTEXT HIERARCHY:

Tier 1 (Always include):
  ├─ Volatility regime (rolling std, VIX proxy)
  └─ Term structure (contango/backwardation)

Tier 2 (Often useful):
  ├─ Seasonality (month, harvest/planting phase)
  ├─ Inventory surprise (vs. expectations)
  └─ Macro regime (dollar strength, rates)

Tier 3 (Context-specific):
  ├─ Weather indicators (for ag)
  ├─ Production data (for energy)
  └─ Manufacturing PMI (for metals)
```

## What Makes a Good Context Feature

### 1. **Predictive Power**
Feature should have different optimal arms in different feature regions.

**Good:** VIX level — high VIX favors defensive commodities, low VIX favors growth commodities

**Bad:** Day of week — no systematic relationship to commodity sector performance

**Test:** Segment historical data by feature quantiles. Do arm rankings change?

### 2. **Observable at Decision Time**
You must know the feature value before choosing the arm.

**Good:** Yesterday's realized volatility, current term spread, this month's season

**Bad:** Tomorrow's inventory report, next week's price change, future VIX

**Rule:** Only use lagged or contemporaneous data, never future data.

### 3. **Sufficient Variation**
Feature needs to vary across observations; constants don't help.

**Good:** Rolling 20-day volatility (continuous, time-varying)

**Bad:** "Asset class = commodities" (constant across all decisions)

**Check:** `feature.std() > 0.1 * abs(feature.mean())`

### 4. **Low Noise-to-Signal Ratio**
Noisy features make learning slower and less reliable.

**Good:** Smoothed 50-day moving average

**Bad:** Daily price changes (too noisy)

**Improvement:** Use rolling windows, exponential smoothing, or regime indicators instead of raw values

### 5. **Interpretable Relationship**
You should be able to explain why the feature might matter.

**Good:** Term structure slope (economic meaning: market expectations)

**Bad:** First principal component of 20 random indicators (uninterpretable)

**Benefit:** Interpretability enables debugging and builds trust.

## Commodity-Specific Feature Recipes

### Recipe 1: Volatility Regime

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def compute_volatility_regime(prices, window=20):
    """Classify current volatility as low/medium/high."""
    returns = prices.pct_change()
    rolling_vol = returns.rolling(window).std()
    # Normalize to long-term distribution
    vol_zscore = (rolling_vol - rolling_vol.mean()) / rolling_vol.std()
    return vol_zscore
```

</div>
</div>

**Why:** Defensive commodities (gold, treasuries) perform better in high-vol regimes; growth commodities (industrial metals) prefer low-vol.

**Typical range:** -2 to +2 (z-score), standardized

### Recipe 2: Term Structure Indicator

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def compute_term_structure(front_price, back_price):
    """Compute contango/backwardation strength."""
    # Positive = contango, negative = backwardation
    term_spread = (back_price - front_price) / front_price
    return term_spread
```

</div>
</div>

**Why:** Contango indicates oversupply or low demand (bad for commodities); backwardation indicates shortage or high demand (good).

**Typical range:** -0.2 to +0.2 (20% annualized)

### Recipe 3: Seasonality Features

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def compute_seasonality(date):
    """Extract seasonal features."""
    month = date.month
    # Harvest season for grains (Sept-Nov)
    is_harvest = 1 if month in [9, 10, 11] else 0
    # Winter demand for energy (Dec-Feb)
    is_winter = 1 if month in [12, 1, 2] else 0
    return np.array([is_harvest, is_winter])
```

</div>
</div>

**Why:** Seasonal patterns are strong in agriculture (harvest pressure) and energy (heating demand).

**Typical encoding:** Binary indicators or cyclical encoding (sin/cos)

### Recipe 4: Inventory Surprise
```python
def compute_inventory_surprise(actual, expected, hist_std):
    """Standardized inventory surprise."""
    surprise = (actual - expected) / hist_std
    return surprise
```

**Why:** Unexpected inventory builds signal oversupply (bearish); unexpected draws signal shortage (bullish).

**Typical range:** -3 to +3 (z-score of surprise)

### Recipe 5: Macro Regime
```python
def compute_macro_regime(dollar_index, vix):
    """Combine dollar strength and risk sentiment."""
    # Normalize both to z-scores
    dollar_z = (dollar_index - dollar_index.mean()) / dollar_index.std()
    vix_z = (vix - vix.mean()) / vix.std()
    # Composite: strong dollar + high VIX = risk-off
    macro_regime = 0.5 * dollar_z + 0.5 * vix_z
    return macro_regime
```

**Why:** Commodities tend to underperform in "risk-off" regimes (strong dollar, high VIX).

**Typical range:** -2 to +2

## Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np

class CommodityContextFeatures:
    """Build context features for commodity bandits."""

    def __init__(self, vol_window=20, term_lag=1):
        self.vol_window = vol_window
        self.term_lag = term_lag

    def extract_features(self, prices, dates):
        """Extract full feature set from price data."""
        features = pd.DataFrame(index=dates)

        # 1. Volatility regime
        returns = prices.pct_change()
        vol = returns.rolling(self.vol_window).std()
        features['vol_zscore'] = (vol - vol.mean()) / vol.std()

        # 2. Term structure (use front/back contract)
        # Simplified: use price momentum as proxy
        mom = prices.pct_change(20)
        features['term_proxy'] = (mom - mom.mean()) / mom.std()

        # 3. Seasonality
        features['month_sin'] = np.sin(2 * np.pi * dates.month / 12)
        features['month_cos'] = np.cos(2 * np.pi * dates.month / 12)

        # 4. Trend strength
        ma_short = prices.rolling(20).mean()
        ma_long = prices.rolling(50).mean()
        features['trend'] = (ma_short - ma_long) / ma_long

        return features.fillna(0)
```

## Avoiding Feature Bloat

### The 5-Feature Rule
Start with ≤5 features. Add more only if they demonstrably improve offline performance.

**Why:** Each additional feature:
- Increases exploration cost (more dimensions to cover)
- Raises risk of overfitting
- Slows computation (matrix inversion is O(d³))

### Feature Selection Process
1. **Start with domain knowledge:** Which regimes matter for your problem?
2. **Compute correlation:** Remove redundant features (corr > 0.8)
3. **Ablation study:** Drop each feature, measure offline regret
4. **Incremental addition:** Start minimal, add features one at a time if they help

## Normalization and Preprocessing

### Why Normalize
LinUCB assumes features are on similar scales. If not, high-magnitude features dominate.

### Standard Approaches

**1. Z-score normalization:**
```python
def normalize_zscore(x):
    return (x - x.mean()) / x.std()
```
Use when feature distribution is roughly Gaussian.

**2. Min-max scaling:**
```python
def normalize_minmax(x):
    return (x - x.min()) / (x.max() - x.min())
```
Use when you want features in [0, 1].

**3. Robust scaling:**
```python
def normalize_robust(x):
    median = x.median()
    iqr = x.quantile(0.75) - x.quantile(0.25)
    return (x - median) / iqr
```
Use when data has outliers.

### Handling Missing Values
```python

# Forward fill (use last known value)
features = features.fillna(method='ffill')

# Or fill with neutral value (0 after normalization)
features = features.fillna(0)
```

**Never drop rows** — in online bandits, you must make a decision every round.

## Common Pitfalls

### 1. **Future Leakage**
- **Problem:** Using data from the future in current context
- **Example:** Including tomorrow's inventory in today's features
- **Fix:** Strict temporal discipline; use `.shift(1)` to lag features

### 2. **Non-stationary Features**
- **Problem:** Feature distribution drifts over time, breaking normalization
- **Example:** VIX mean changes after 2008 crisis
- **Fix:** Use rolling windows for normalization (trailing 252 days)

### 3. **Redundant Features**
- **Problem:** Including highly correlated features wastes dimensions
- **Example:** Both VIX and SPX 20-day vol (corr ~0.9)
- **Fix:** Drop one or use PCA to combine

### 4. **Categorical Features Without Encoding**
- **Problem:** Feeding "January" as a string to LinUCB
- **Fix:** Use one-hot encoding or cyclical encoding (sin/cos for months)

### 5. **Ignoring Feature Importance**
- **Problem:** All features weighted equally in model
- **Fix:** Examine learned θ weights; drop low-importance features

## Diagnostic Tools

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


### Check Feature Quality
```python
def diagnose_features(features, rewards, arm):
    """Assess feature quality for a given arm."""
    # 1. Correlation with rewards
    corr = features.corrwith(rewards[arm])
    print(f"Feature-reward correlation:\n{corr}\n")

    # 2. Feature variance
    print(f"Feature variance:\n{features.var()}\n")

    # 3. Inter-feature correlation
    print("Feature correlation matrix:")
    print(features.corr())
```

### Ablation Test
```python
def ablation_test(features, rewards, arms):
    """Test impact of dropping each feature."""
    baseline_regret = run_bandit(features, rewards, arms)

    for col in features.columns:
        reduced = features.drop(columns=[col])
        regret = run_bandit(reduced, rewards, arms)
        improvement = baseline_regret - regret
        print(f"Drop {col}: regret change = {improvement:.3f}")
```

## Practice Problems

### 1. Feature Selection
**Question:** You're building a bandit for energy vs. metals allocation. Which features are most important?
- a) VIX
- b) Day of week
- c) Dollar index
- d) Crude term structure

**Answer:** a, c, d. VIX captures risk regime, dollar affects commodity demand, term structure indicates supply/demand. Day of week is irrelevant for medium-term allocation.

### 2. Normalization Choice
**Question:** VIX ranges from 10-80, term spread ranges from -0.1 to +0.1. If you don't normalize, what happens in LinUCB?

**Answer:** VIX dominates the regression because its magnitude is ~100x larger. The model will mostly ignore term spread. Always normalize features.

### 3. Implementation Exercise
**Task:** Add a "momentum regime" feature: +1 if 20-day return > 0, -1 otherwise.

```python
def add_momentum_regime(features, prices):
    mom = prices.pct_change(20)
    features['momentum_regime'] = np.where(mom > 0, 1, -1)
    return features
```

### 4. Debugging Scenario
**Problem:** Your contextual bandit performs no better than a standard bandit. What might be wrong with your features?

**Possible causes:**
- Features have no predictive power (check correlation with rewards)
- Features are all highly correlated (check feature correlation matrix)
- Features are too noisy (try smoothing)
- Scale issues (check feature ranges, apply normalization)


---

## Cross-References

<a class="link-card" href="./01_contextual_bandit_framework.md">
  <div class="link-card-title">01 Contextual Bandit Framework</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_contextual_bandit_framework.md">
  <div class="link-card-title">01 Contextual Bandit Framework — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_linucb_algorithm.md">
  <div class="link-card-title">02 Linucb Algorithm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_linucb_algorithm.md">
  <div class="link-card-title">02 Linucb Algorithm — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./cheatsheet.md">
  <div class="link-card-title">Cheatsheet — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


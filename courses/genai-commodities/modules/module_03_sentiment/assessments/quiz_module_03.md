# Quiz: Module 3 - Sentiment Analysis for Commodities

**Course:** GenAI for Commodity Markets
**Module:** 3 - Sentiment Analysis
**Total Points:** 100
**Time Limit:** 25 minutes
**Attempts Allowed:** 2

## Instructions

This quiz assesses your ability to extract sentiment signals from commodity news, build sentiment indices, and evaluate signal quality. Focus on commodity-specific sentiment extraction and practical signal construction.

---

## Section 1: Sentiment Extraction (35 points)

### Question 1 (12 points)

You're extracting sentiment from this commodity news article:

> "West Texas Intermediate crude futures settled 2.3% higher at $82.40 per barrel Thursday, marking the fourth consecutive daily gain. The rally came despite data showing U.S. inventories rose by 1.8 million barrels, as traders focused on tightening global supply conditions. OPEC+ production discipline and strong Asian demand are supporting the market, though concerns about Chinese economic growth persist."

Which sentiment extraction provides the MOST trading-relevant information?

**Option A: Simple Sentiment**
```json
{
  "sentiment": "bullish",
  "confidence": 0.8
}
```

**Option B: Directional Sentiment**
```json
{
  "overall_sentiment": 0.7,
  "price_direction": "up",
  "magnitude": "moderate"
}
```

**Option C: Nuanced Multi-Factor Sentiment**
```json
{
  "overall_sentiment": 0.65,
  "sentiment_factors": {
    "price_action": {
      "sentiment": 0.85,
      "evidence": "4th consecutive gain, +2.3%",
      "weight": 0.3
    },
    "supply": {
      "sentiment": 0.80,
      "evidence": "OPEC+ discipline, tightening global supply",
      "weight": 0.35
    },
    "demand": {
      "sentiment": 0.40,
      "evidence": "Strong Asian demand offset by China growth concerns",
      "weight": 0.25
    },
    "inventories": {
      "sentiment": -0.30,
      "evidence": "U.S. inventories rose 1.8 mb",
      "weight": 0.10
    }
  },
  "key_tensions": [
    "Bearish inventory build vs bullish global supply tightness",
    "Strong Asian demand vs China concerns"
  ],
  "time_horizon": "short_term",
  "confidence": 0.75
}
```

**Option D: Market Psychology**
```json
{
  "sentiment": "bullish",
  "market_psychology": "optimistic",
  "fear_greed_index": 72
}
```

A) Option A - Simple and clear
B) Option B - Captures direction and magnitude
C) Option C - Comprehensive factor analysis
D) Option D - Captures market psychology

**Answer: C**

**Explanation:**

**A - Simple Sentiment:**
- Too basic for commodity trading
- Loses critical nuances
- "Bullish" doesn't explain WHY or identify factors
- Can't separate signal strength across factors

**B - Directional Sentiment:**
- Better than A but still incomplete
- Misses factor decomposition
- "Moderate magnitude" is vague
- Can't identify which factors drive sentiment

**C (Correct) - Multi-Factor Analysis:**
```python
# Why this is superior for commodities:

# 1. FACTOR DECOMPOSITION
# Separates bullish (supply, price action) from bearish (inventories) factors
# Traders can assess:
# - Which factors dominate? (supply/price action)
# - Which factors are weak? (demand mixed, inventories bearish)

# 2. EVIDENCE-BASED
# Each sentiment score links to specific evidence
# Enables verification and trust

# 3. WEIGHTED IMPORTANCE
# Not all factors equal:
# - Supply factors (35% weight): Most important for oil
# - Price action (30%): Strong recent signal
# - Demand (25%): Important but mixed signals
# - Inventories (10%): Weekly noise

# 4. IDENTIFIES TENSIONS
# "Inventory build vs supply tightness" = key market debate
# Traders know which narrative wins

# 5. TIME HORIZON
# "Short-term" = this is tactical positioning sentiment
# Not long-term structural view

# Trading implications:
# - Overall bullish (0.65) but not extremely
# - Strong on supply fundamentals (0.80)
# - Weak on demand (0.40) due to China concerns
# - Near-term positioning trade, not long-term thesis

# Signal construction:
signal_strength = weighted_avg([
    (0.85, 0.3),  # Price action
    (0.80, 0.35), # Supply
    (0.40, 0.25), # Demand
    (-0.30, 0.10) # Inventories
])
# = 0.65, matches overall_sentiment

# Risk assessment:
main_risk = "China demand concerns" # From demand factor
confidence_detractor = "Mixed demand signals" # Reduces confidence to 0.75
```

**D - Market Psychology:**
- Vague terminology ("optimistic")
- "Fear and greed" is retail-focused
- Doesn't provide actionable factors

**Production Implementation:**
```python
COMMODITY_SENTIMENT_PROMPT = """Analyze this commodity news article for trading-relevant sentiment.

Article: {article}

Extract sentiment with this structure:
{
  "overall_sentiment": <-1 to +1>,
  "sentiment_factors": {
    "price_action": {
      "sentiment": <-1 to +1>,
      "evidence": "<specific quotes/data>",
      "weight": <0-1>
    },
    "supply": {...},
    "demand": {...},
    "inventories": {...},
    "geopolitics": {...} // if relevant
  },
  "key_tensions": ["<conflicting signals>"],
  "time_horizon": "immediate|short_term|medium_term|long_term",
  "confidence": <0-1>
}

Guidelines:
- Evidence must be direct quotes or specific data from article
- Weights should sum to 1.0
- Identify contradictory signals in "key_tensions"
- Confidence should reflect clarity of signals (high when aligned, low when mixed)
"""
```

---

### Question 2 (10 points)

You're building a natural gas sentiment system. Which news sources and update frequencies would provide the MOST valuable signal?

A) Financial news (Bloomberg, Reuters) updated every 15 minutes
B) Commodity-specific news (Natural Gas Intelligence, EIA reports) + weather forecasts + storage data, updated hourly
C) Social media (Twitter/X commodity traders) updated real-time
D) Earnings calls from energy companies updated quarterly

**Answer: B**

**Explanation:**

**A - Financial News Only:**
- Valuable but incomplete
- Misses natural gas-specific drivers (weather, storage)
- General financial news often covers gas as secondary to crude oil
- 15-minute updates excessive for gas market speed

**B (Correct) - Commodity-Specific + Fundamentals:**
```python
class NaturalGasSentimentSources:
    """Optimal source mix for natural gas sentiment."""

    SOURCES = {
        # Tier 1: Authoritative data
        "eia_storage": {
            "url": "EIA Natural Gas Storage Report",
            "frequency": "weekly",  # Thursdays 10:30 AM
            "importance": "critical",
            "signal_type": "fundamental"
        },
        "eia_production": {
            "url": "EIA Natural Gas Monthly",
            "frequency": "monthly",
            "importance": "high",
            "signal_type": "fundamental"
        },

        # Tier 2: Weather (HUGE for nat gas)
        "weather_forecasts": {
            "sources": ["NOAA", "Weather.com"],
            "frequency": "daily",  # Update forecasts daily
            "importance": "critical",  # Weather drives 40%+ of demand
            "signal_type": "demand_forecast"
        },
        "heating_degree_days": {
            "frequency": "daily",
            "importance": "high",
            "signal_type": "demand_actual"
        },

        # Tier 3: Commodity-specific news
        "natural_gas_intelligence": {
            "frequency": "daily",
            "importance": "high",
            "signal_type": "market_commentary"
        },
        "bloomberg_gas": {
            "filter": "natural gas specific",
            "frequency": "hourly",
            "importance": "medium",
            "signal_type": "price_action"
        },

        # Tier 4: Supply factors
        "lng_exports": {
            "source": "LNG export terminal reports",
            "frequency": "daily",
            "importance": "high",
            "signal_type": "demand"
        },
        "pipeline_flows": {
            "source": "Pipeline operator data",
            "frequency": "daily",
            "importance": "medium",
            "signal_type": "supply"
        }
    }

# Why hourly updates?
# - Natural gas moves slower than equity markets
# - Storage reports are weekly (single biggest driver)
# - Weather forecasts update 2-4x daily
# - Hourly aggregation captures all relevant updates without noise

# Signal construction:
def aggregate_nat_gas_sentiment(sources):
    """Combine multiple signal types."""

    sentiment = {
        "fundamental": 0.0,  # Storage, production
        "weather": 0.0,      # Temperature forecasts
        "demand": 0.0,       # LNG exports, consumption
        "supply": 0.0,       # Production, pipeline flows
        "price_action": 0.0  # Recent price moves, commentary
    }

    # Weight by importance for natural gas
    weights = {
        "fundamental": 0.30,  # Storage is king
        "weather": 0.30,      # Weather drives demand
        "demand": 0.20,       # LNG exports critical
        "supply": 0.10,       # Supply relatively stable
        "price_action": 0.10  # Recent momentum
    }

    overall = sum(sentiment[k] * weights[k] for k in sentiment)
    return overall
```

**C - Social Media:**
- Too noisy for reliable signal
- Hard to filter retail vs professional
- Real-time not necessary for natural gas
- Useful as supplementary, not primary

**D - Earnings Calls:**
- Too infrequent (quarterly) for daily trading
- Backward-looking
- Better for structural trends than tactical signals

**Key Insight:** Natural gas sentiment requires weather + storage data, not just news. Weather is 30-40% of the signal.

---

### Question 3 (13 points)

You extract sentiment from 20 crude oil news articles today. Results:

| Article | Source | Sentiment | Confidence |
|---------|--------|-----------|------------|
| 1-15 | General financial news | 0.4-0.6 | 0.5-0.7 |
| 16-18 | Oil-specific (Platts, Argus) | -0.3, -0.2, -0.4 | 0.85-0.90 |
| 19-20 | Social media summaries | 0.8, 0.9 | 0.3-0.4 |

Which aggregation method produces the MOST reliable sentiment signal?

**Method A: Simple Average**
```python
sentiment = mean([all 20 sentiment scores])
# = 0.31 (bullish)
```

**Method B: Confidence-Weighted Average**
```python
sentiment = sum(sentiment[i] * confidence[i]) / sum(confidence)
# Weights high-confidence sources more
```

**Method C: Source-Credibility and Confidence Weighted**
```python
# Step 1: Assign source credibility
source_credibility = {
    "oil_specific": 1.0,      # Platts, Argus, NGI
    "financial_news": 0.7,     # Bloomberg, Reuters, WSJ
    "social_media": 0.3        # Twitter, Reddit
}

# Step 2: Combined weight = credibility * confidence
combined_weight = credibility * confidence

# Step 3: Weighted average
sentiment = sum(sentiment[i] * combined_weight[i]) / sum(combined_weight)
```

**Method D: Majority Vote**
```python
# Simple: more bullish articles than bearish?
bullish_count = sum(s > 0 for s in sentiments)
bearish_count = sum(s < 0 for s in sentiments)
sentiment = 1 if bullish_count > bearish_count else -1
```

A) Method A
B) Method B
C) Method C
D) Method D

**Answer: C**

**Explanation:**

**Method A - Simple Average:**
```python
# Problems:
# 1. Treats all sources equally (social media = Platts)
# 2. High-volume low-quality sources drown out expert analysis
# 3. Ignores confidence levels

# In this example:
# - 15 general articles (sentiment ~0.5) dominate
# - 3 expert bearish views (-0.3, credibility 0.9) are diluted
# - 2 unreliable bullish social media (0.85, credibility 0.3) add noise
# Result: Misleading bullish signal
```

**Method B - Confidence-Weighted:**
```python
# Better: Emphasizes high-confidence extractions
# Problem: Doesn't account for source quality

# High confidence + low credibility still problematic:
# "I'm very confident this random Twitter thread is bullish" (conf=0.9)
# vs "Platts reports bearish inventory data" (conf=0.85)

# Weights the Twitter thread higher!
```

**Method C (Correct) - Credibility × Confidence:**
```python
def calculate_sentiment_signal(articles):
    """Calculate credibility and confidence-weighted sentiment."""

    # Source credibility scores
    CREDIBILITY = {
        "platts": 1.0,           # Price reporting agency
        "argus": 1.0,            # Price reporting agency
        "eia": 1.0,              # Government data
        "bloomberg": 0.8,        # Major financial news
        "reuters": 0.8,          # Major financial news
        "wsj": 0.7,              # General financial
        "generic_news": 0.5,     # Other news sources
        "social_media": 0.2,     # Twitter, Reddit, forums
        "blog": 0.3              # Personal blogs
    }

    weighted_sum = 0
    weight_total = 0

    for article in articles:
        # Extract source type
        credibility = CREDIBILITY.get(article.source_type, 0.5)

        # Get LLM confidence in extraction
        confidence = article.confidence  # 0-1 from extraction

        # Combined weight
        weight = credibility * confidence

        # Weighted contribution
        weighted_sum += article.sentiment * weight
        weight_total += weight

        # Log for transparency
        logger.debug(f"Article: {article.source_type}, "
                    f"Sentiment: {article.sentiment:.2f}, "
                    f"Confidence: {confidence:.2f}, "
                    f"Credibility: {credibility:.2f}, "
                    f"Weight: {weight:.2f}")

    overall_sentiment = weighted_sum / weight_total

    # Calculate signal confidence
    # High when authoritative sources agree, low when mixed
    high_cred_articles = [a for a in articles
                         if CREDIBILITY.get(a.source_type, 0) > 0.7]

    if len(high_cred_articles) > 0:
        high_cred_sentiment = [a.sentiment for a in high_cred_articles]
        agreement = 1 - np.std(high_cred_sentiment)  # Low std = high agreement
    else:
        agreement = 0.5  # No authoritative sources

    signal_confidence = agreement * min(1.0, len(high_cred_articles) / 5)

    return {
        "sentiment": overall_sentiment,
        "confidence": signal_confidence,
        "n_articles": len(articles),
        "n_high_credibility": len(high_cred_articles)
    }

# Applied to our example:
# Articles 16-18 (oil-specific, bearish, high confidence):
#   weight = 1.0 * 0.85 = 0.85 each → total 2.55

# Articles 1-15 (general, neutral, medium confidence):
#   weight = 0.5 * 0.6 = 0.30 each → total 4.5

# Articles 19-20 (social media, bullish, low confidence):
#   weight = 0.2 * 0.35 = 0.07 each → total 0.14

# Weighted sentiment:
# = (2.55 * -0.3 + 4.5 * 0.5 + 0.14 * 0.85) / (2.55 + 4.5 + 0.14)
# = (-0.77 + 2.25 + 0.12) / 7.19
# = 0.22 (slightly bullish, but much less than simple average)

# Signal confidence: LOW (mixed signals between expert bearish and general bullish)
```

**Method D - Majority Vote:**
- Loses magnitude information
- No nuance (all bullish treated same)
- Ignores credibility entirely

**Key Principles:**
1. **Source credibility matters:** Platts > Twitter
2. **Extraction confidence matters:** Clear statements > ambiguous
3. **Combine multiplicatively:** Low on either dimension = low weight
4. **Track disagreement:** Mixed signals = low confidence

---

## Section 2: Signal Construction (35 points)

### Question 4 (15 points)

You've collected sentiment scores for crude oil over 30 days. The daily sentiment time-series shows:

```
Day  1-10: Sentiment around +0.5 (bullish)
Day 11-20: Sentiment around +0.2 (neutral-bullish)
Day 21-30: Sentiment around -0.3 (bearish)
```

Crude oil prices:
```
Day  1-10: $80/bbl, trending up
Day 11-20: $82/bbl, range-bound
Day 21-30: $85/bbl, trending up
```

What does this relationship tell you about the sentiment signal's predictive value?

A) Strong positive correlation - sentiment predicts price direction
B) Lagging indicator - sentiment follows price changes
C) Contrarian indicator - bearish sentiment coincides with price rises
D) No relationship - sentiment is noise

**Answer: C**

**Explanation:**

**Analysis:**
```python
# Period 1 (Days 1-10):
# Sentiment: +0.5 (bullish)
# Price: $80 → trending up
# ✓ Alignment: Bullish sentiment + rising prices

# Period 2 (Days 11-20):
# Sentiment: +0.2 (cooling)
# Price: $82 → range-bound
# ✓ Alignment: Neutral sentiment + sideways prices

# Period 3 (Days 21-30):
# Sentiment: -0.3 (bearish)
# Price: $85 → trending UP (not down!)
# ✗ Divergence: Bearish sentiment BUT rising prices
```

**A - Positive Correlation:**
- False: Period 3 shows negative correlation (bearish + up prices)

**B - Lagging Indicator:**
- Possible but doesn't fit Period 3
- If sentiment lagged prices, we'd expect sentiment to be bullish in Period 3 (following price rise)
- Instead, sentiment is bearish

**C (Correct) - Contrarian Indicator:**
```python
# Classic contrarian pattern:
# "Be fearful when others are greedy, greedy when others are fearful"

# Period 3 interpretation:
# - News became bearish (perhaps due to demand concerns, inventory builds)
# - BUT prices continued rising (supply factors dominated)
# - Peak bearish sentiment → buying opportunity

# This is common in commodity markets:
# 1. News focuses on one factor (demand)
# 2. Prices driven by different factor (supply)
# 3. Extreme sentiment creates counter-moves

# Trading strategy:
def contrarian_signal(sentiment, price):
    """Generate contrarian signals from sentiment extremes."""

    # Calculate sentiment extremity
    sentiment_zscore = (sentiment - mean) / std

    # Extreme bearish sentiment + rising prices = contrarian buy
    if sentiment_zscore < -2 and price_trend > 0:
        return {
            "signal": "long",
            "rationale": "Extreme bearish sentiment despite rising prices",
            "type": "contrarian"
        }

    # Extreme bullish sentiment + falling prices = contrarian sell
    elif sentiment_zscore > 2 and price_trend < 0:
        return {
            "signal": "short",
            "rationale": "Extreme bullish sentiment despite falling prices",
            "type": "contrarian"
        }

    return {"signal": "neutral"}

# Period 3 would trigger:
# sentiment_zscore = (-0.3 - 0.13) / 0.35 = -1.2 (moderately extreme)
# price_trend = positive
# → Suggests contrarian long signal (sentiment too bearish)
```

**D - No Relationship:**
- Too extreme; there are clear patterns across periods

**Key Insight:** Sentiment signals can be:
1. **Momentum indicators** (sentiment leads prices)
2. **Confirming indicators** (sentiment confirms trend)
3. **Contrarian indicators** (extreme sentiment precedes reversals)

You must backtest to determine which regime applies.

---

### Question 5 (10 points)

You're building a composite sentiment index combining news sentiment, social media sentiment, and analyst ratings for crude oil. Which weighting scheme is MOST appropriate?

A) Equal weight (33% each)
B) Weight by data volume (more articles = higher weight)
C) Weight by historical predictive power via backtesting
D) Weight by recency (recent sources weighted higher)

**Answer: C**

**Explanation:**

**A - Equal Weight:**
```python
# Problems:
# 1. Assumes all sources equally predictive (unlikely)
# 2. Social media may be noise, not signal
# 3. Doesn't adapt to changing market conditions
```

**B - Volume Weighting:**
```python
# Problems:
# 1. High volume can be low quality (social media spam)
# 2. Major events generate volume but may be lagging
# 3. Quantity ≠ quality
```

**C (Correct) - Backtest-Based Weighting:**
```python
def optimize_sentiment_weights(historical_data):
    """Determine optimal weights via backtesting."""

    # Step 1: Collect historical sentiment and returns
    df = pd.DataFrame({
        'date': dates,
        'news_sentiment': news_sent,
        'social_sentiment': social_sent,
        'analyst_ratings': analyst_sent,
        'forward_5d_return': returns  # 5-day forward return
    })

    # Step 2: Test different weight combinations
    best_sharpe = -np.inf
    best_weights = None

    for w_news in np.arange(0, 1.1, 0.1):
        for w_social in np.arange(0, 1.1, 0.1):
            w_analyst = 1 - w_news - w_social
            if w_analyst < 0:
                continue

            # Calculate composite sentiment
            df['composite'] = (
                w_news * df['news_sentiment'] +
                w_social * df['social_sentiment'] +
                w_analyst * df['analyst_ratings']
            )

            # Generate signals: long if composite > threshold
            df['signal'] = np.where(df['composite'] > 0.2, 1, 0)
            df['signal'] = np.where(df['composite'] < -0.2, -1, df['signal'])

            # Calculate returns
            df['strategy_return'] = df['signal'] * df['forward_5d_return']

            # Evaluate performance
            sharpe = df['strategy_return'].mean() / df['strategy_return'].std()

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = {
                    'news': w_news,
                    'social': w_social,
                    'analyst': w_analyst
                }

    return best_weights

# Example result:
# best_weights = {
#     'news': 0.6,      # News most predictive
#     'social': 0.1,    # Social media mostly noise
#     'analyst': 0.3    # Analyst ratings moderately useful
# }

# Step 3: Validate on out-of-sample data
def validate_weights(weights, validation_data):
    """Test weights on unseen data."""

    composite = (
        weights['news'] * validation_data['news_sentiment'] +
        weights['social'] * validation_data['social_sentiment'] +
        weights['analyst'] * validation_data['analyst_ratings']
    )

    # Generate signals and calculate Sharpe
    signals = generate_signals(composite)
    sharpe = calculate_sharpe(signals, validation_data['returns'])

    return sharpe

# Step 4: Periodic re-optimization (quarterly)
# Market regimes change; re-optimize weights regularly
```

**Why this is superior:**
- **Evidence-based:** Weights reflect actual predictive power
- **Optimal:** Maximizes risk-adjusted returns
- **Adaptive:** Can re-optimize as markets change
- **Quantitative:** No subjective judgment

**D - Recency Weighting:**
- Orthogonal concern (weight recent data in each source)
- Doesn't address relative importance of sources

**Implementation Note:**
```python
# Common findings from backtesting commodity sentiment:
# - News: 50-70% weight (most reliable)
# - Analyst ratings: 20-40% weight (slower but stable)
# - Social media: 5-15% weight (noisy but occasionally catches early signals)

# But this varies by:
# - Commodity (social media more relevant for crypto than crude oil)
# - Time horizon (social media more relevant for intraday than monthly)
# - Market regime (social media spikes during volatility events)
```

---

### Question 6 (10 points)

Your sentiment-based trading signal has these backtest results over 2 years:

```
Total trades: 120
Win rate: 55%
Average win: +2.5%
Average loss: -1.8%
Sharpe ratio: 1.2
Maximum drawdown: -12%
```

What is the PRIMARY improvement priority?

A) Increase win rate to 60%+
B) Reduce maximum drawdown to -8%
C) Increase position sizing to amplify returns
D) Analyze losing trades to identify filterable patterns

**Answer: D**

**Explanation:**

**Current Performance Assessment:**
```python
# These are GOOD results:
# ✓ Win rate 55% > 50% (edge exists)
# ✓ Sharpe 1.2 (decent risk-adjusted return)
# ✓ Average win/loss ratio: 2.5/1.8 = 1.4 (winners bigger than losers)
# ✓ Max drawdown 12% (manageable)

# Quick calculation:
# Expected value per trade = 0.55 * 0.025 + 0.45 * (-0.018) = 0.0056 = +0.56%
# Positive expectancy ✓

# This is a profitable system worth improving, not overhauling
```

**A - Increase Win Rate:**
```python
# Problems:
# 1. 55% is already strong (50% is breakeven with equal win/loss size)
# 2. Pushing for higher win rate often reduces winners' size
# 3. May lead to overfitting

# Win rate is NOT the key metric
# Risk-adjusted expectancy is
```

**B - Reduce Drawdown:**
```python
# Drawdown reduction approaches:
# 1. Reduce position size → reduces returns proportionally
# 2. Tighter stops → increases losing trade frequency
# 3. Better trade selection → yes! (This is option D)

# 12% drawdown is acceptable for commodity trading
# Focus on improving signal, not just managing risk
```

**C - Increase Position Sizing:**
```python
# DANGEROUS without improving signal quality
# - Amplifies drawdowns proportionally
# - 12% drawdown → 24% drawdown at 2x leverage
# - Sharpe ratio unchanged
# - Risk of ruin increases

# Only increase size after:
# 1. Improving signal quality
# 2. Reducing false positives
# 3. Understanding failure modes
```

**D (Correct) - Analyze Losing Trades:**
```python
def analyze_losing_trades(trades):
    """Identify patterns in losing trades to improve signal."""

    losing_trades = [t for t in trades if t.return < 0]

    # Analysis dimensions:
    analysis = {
        "by_sentiment_strength": {},
        "by_confidence": {},
        "by_volatility_regime": {},
        "by_news_volume": {},
        "by_time_of_day": {},
        "by_holding_period": {}
    }

    # 1. Sentiment strength at entry
    for trade in losing_trades:
        strength = abs(trade.entry_sentiment)
        bucket = "strong" if strength > 0.7 else "medium" if strength > 0.4 else "weak"
        analysis["by_sentiment_strength"][bucket] = analysis["by_sentiment_strength"].get(bucket, 0) + 1

    # Example findings:
    # - 60% of losing trades had "weak" sentiment (< 0.4)
    # → Filter: Only trade when sentiment > 0.5
    # → Impact: Reduces trades from 120 to 80, but win rate increases to 62%

    # 2. Confidence level
    for trade in losing_trades:
        conf_bucket = "high" if trade.confidence > 0.7 else "medium" if trade.confidence > 0.5 else "low"
        analysis["by_confidence"][conf_bucket] = analysis["by_confidence"].get(conf_bucket, 0) + 1

    # Example findings:
    # - 70% of losing trades had low confidence (< 0.5)
    # → Filter: Only trade when confidence > 0.6
    # → Impact: Sharpe improves from 1.2 to 1.6

    # 3. Volatility regime
    for trade in losing_trades:
        vol = trade.entry_volatility
        vol_bucket = "high" if vol > 30 else "normal" if vol > 15 else "low"
        analysis["by_volatility_regime"][vol_bucket] = analysis["by_volatility_regime"].get(vol_bucket, 0) + 1

    # Example findings:
    # - 50% of losing trades occurred in high volatility (VIX > 30)
    # → Filter: Reduce position size or skip trades during high VIX
    # → Impact: Max drawdown reduces from 12% to 9%

    # 4. News volume (potential false signals)
    for trade in losing_trades:
        volume = trade.news_article_count
        vol_bucket = "high" if volume > 50 else "normal" if volume > 20 else "low"
        analysis["by_news_volume"][vol_bucket] = analysis["by_news_volume"].get(vol_bucket, 0) + 1

    # Example findings:
    # - 40% of losing trades occurred on very high news volume days (>50 articles)
    # → Insight: Extreme news volume = confusion, not clarity
    # → Filter: Skip or reduce size on extreme news days

    return analysis

# Apply findings incrementally:
# Version 1: sentiment > 0.5, confidence > 0.6
# → Win rate improves to 60%, Sharpe to 1.5
# Version 2: Add volatility filter (reduce size in high vol)
# → Max drawdown improves to 9%
# Version 3: Add news volume filter (skip extreme days)
# → Win rate improves to 62%, Sharpe to 1.7

# Result:
# - Fewer trades (120 → 70)
# - Higher win rate (55% → 62%)
# - Better Sharpe (1.2 → 1.7)
# - Lower drawdown (12% → 9%)
# - Higher position sizing now justified
```

**Key Principle:** Improve signal quality before increasing risk. Better signals → higher Sharpe → justify larger positions.

---

## Section 3: Evaluation and Production (30 points)

### Question 7 (12 points)

You're comparing two sentiment models:

**Model A:**
- Processes 100 articles/day
- 65% accuracy predicting next-day direction
- $0.50 cost per article (LLM API)
- 30-second processing time per article

**Model B:**
- Processes 20 articles/day (filtered for quality)
- 70% accuracy predicting next-day direction
- $1.20 cost per article (larger LLM + re-ranking)
- 90-second processing time per article

For a daily trading strategy, which model is superior?

A) Model A - Higher volume coverage
B) Model B - Higher accuracy
C) Need to calculate expected value: accuracy × volume × avg_profit - costs
D) Depends on how much capital you're trading

**Answer: C**

**Explanation:**

**Analysis requires calculating expected value:**

```python
def compare_sentiment_models(model_a, model_b, trading_params):
    """Compare models on expected value, not just accuracy."""

    # Trading parameters
    avg_profit_per_correct_signal = trading_params['avg_profit']  # e.g., $5,000
    avg_loss_per_incorrect_signal = trading_params['avg_loss']    # e.g., -$2,000
    cost_tolerance = trading_params['cost_budget']  # e.g., $100/day

    # Model A analysis
    a_signals_per_day = 100
    a_accuracy = 0.65
    a_cost_per_day = 100 * 0.50  # $50
    a_processing_time = 100 * 30 / 60  # 50 minutes

    a_correct_signals = a_signals_per_day * a_accuracy  # 65
    a_incorrect_signals = a_signals_per_day * (1 - a_accuracy)  # 35

    a_gross_profit = a_correct_signals * avg_profit_per_correct_signal
    a_gross_loss = a_incorrect_signals * avg_loss_per_incorrect_signal
    a_net_pnl = a_gross_profit + a_gross_loss - a_cost_per_day

    # Model B analysis
    b_signals_per_day = 20
    b_accuracy = 0.70
    b_cost_per_day = 20 * 1.20  # $24
    b_processing_time = 20 * 90 / 60  # 30 minutes

    b_correct_signals = b_signals_per_day * b_accuracy  # 14
    b_incorrect_signals = b_signals_per_day * (1 - b_accuracy)  # 6

    b_gross_profit = b_correct_signals * avg_profit_per_correct_signal
    b_gross_loss = b_incorrect_signals * avg_loss_per_incorrect_signal
    b_net_pnl = b_gross_profit + b_gross_loss - b_cost_per_day

    return {
        "model_a": {
            "signals": a_signals_per_day,
            "correct": a_correct_signals,
            "cost": a_cost_per_day,
            "net_pnl": a_net_pnl,
            "time": a_processing_time
        },
        "model_b": {
            "signals": b_signals_per_day,
            "correct": b_correct_signals,
            "cost": b_cost_per_day,
            "net_pnl": b_net_pnl,
            "time": b_processing_time
        }
    }

# Example calculation (assume $5000 profit per correct, -$2000 per incorrect):

# Model A:
# Gross profit: 65 * $5,000 = $325,000
# Gross loss: 35 * -$2,000 = -$70,000
# Net P&L: $325,000 - $70,000 - $50 = $254,950

# Model B:
# Gross profit: 14 * $5,000 = $70,000
# Gross loss: 6 * -$2,000 = -$12,000
# Net P&L: $70,000 - $12,000 - $24 = $57,976

# → Model A is better! Despite lower accuracy, volume compensates

# BUT: This assumes you can trade 100 signals/day
# If capital-constrained to only 20 trades/day:

# Model A (20 best signals):
# Correct: 20 * 0.65 = 13
# Net P&L: 13 * $5,000 + 7 * -$2,000 - $50 = $50,950

# Model B (all 20 signals):
# Net P&L: $57,976

# → Model B is better when capital-constrained!
```

**C (Correct) - Calculate Expected Value:**
The answer depends on:
1. **Profit per correct signal** (higher profit → quality over quantity)
2. **Loss per incorrect signal** (higher loss → emphasize accuracy)
3. **Trading capacity** (can you act on 100 signals or only 20?)
4. **Cost tolerance** (is $50/day vs $24/day material?)

**A - Higher Volume:**
Only better if capacity exists and costs are acceptable

**B - Higher Accuracy:**
Only better if volume is constrained or losses are large

**D - Depends on Capital:**
Partially correct but incomplete; also depends on profit/loss profile

**Key Insight:** For sentiment signals, optimize *expected value*, not accuracy alone.

---

### Question 8 (10 points)

Your sentiment system has been in production for 6 months. You notice sentiment signal quality degrading (accuracy dropped from 65% to 58%). What is the MOST likely cause and solution?

A) Model drift - Retrain the LLM on recent data
B) Distribution shift - News sources/language patterns changed
C) Evaluation error - Accuracy measurement was flawed
D) Market regime change - Sentiment-price relationship changed

**Answer: D**

**Explanation:**

**A - Model Drift:**
```python
# Model drift applies to trained models (fine-tuned LLMs)
# But most sentiment systems use:
# - Prompted LLMs (GPT-4, Claude) - no drift, always current
# - Or fine-tuned models that need retraining

# If using prompted LLMs: Not the issue
# If fine-tuned: Possible, but sentiment extraction is relatively stable
```

**B - Distribution Shift:**
```python
# Example: News sources start using different language
# - "supply tightness" → "reduced availability"
# - "bullish" → "positive outlook"

# This CAN cause issues, but:
# - Modern LLMs handle synonyms well
# - Sentiment extraction is robust to paraphrasing

# Check:
def detect_language_shift(recent_articles, historical_articles):
    """Detect if article language changed."""

    recent_vocab = extract_key_terms(recent_articles)
    historical_vocab = extract_key_terms(historical_articles)

    vocab_overlap = len(recent_vocab & historical_vocab) / len(historical_vocab)

    if vocab_overlap < 0.7:
        return "Significant language shift detected"

# If this is the issue:
# - Update extraction prompts with new terminology
# - Add few-shot examples using new language patterns
```

**C - Evaluation Error:**
```python
# Possible but unlikely to cause 7% drop
# Check:
# - Are you still measuring correctly?
# - Did evaluation methodology change?
# - Are you comparing apples-to-apples?
```

**D (Correct) - Market Regime Change:**
```python
# Most likely cause for sentiment signal degradation:

# Sentiment-price relationships are NOT constant
# They change with market regimes:

# REGIME 1: Supply-driven market
# - Sentiment: "OPEC cuts production"
# - Price: Rises
# - Relationship: Positive (sentiment works)

# REGIME 2: Demand-driven market
# - Sentiment: "China growth concerns"
# - Price: Falls (but less than expected due to supply discipline)
# - Relationship: Weaker (sentiment less predictive)

# REGIME 3: Macro-driven market
# - Sentiment: "Strong crude fundamentals"
# - Price: Falls (due to USD strength, risk-off)
# - Relationship: Inverted (sentiment fails)

# Detection:
def detect_regime_change(sentiment_data, price_data):
    """Detect if sentiment-price relationship changed."""

    # Rolling correlation
    window = 60  # 60 days
    correlations = []

    for i in range(window, len(sentiment_data)):
        recent_sent = sentiment_data[i-window:i]
        recent_price_change = price_data[i-window:i]

        corr = np.corrcoef(recent_sent, recent_price_change)[0, 1]
        correlations.append(corr)

    # Check if correlation regime shifted
    recent_corr = np.mean(correlations[-30:])  # Last 30 days
    historical_corr = np.mean(correlations[:-30])  # Prior

    if abs(recent_corr - historical_corr) > 0.3:
        return f"Regime change: correlation shifted from {historical_corr:.2f} to {recent_corr:.2f}"

# Example output:
# "Regime change: correlation shifted from 0.45 to 0.08"
# → Sentiment no longer predictive, possibly contrarian now

# Solution:
def adaptive_signal_generation(sentiment, regime):
    """Adapt signal based on detected regime."""

    if regime == "momentum":
        # Sentiment predicts prices
        return sentiment * 1.0  # Use as-is

    elif regime == "mean_reversion":
        # Extreme sentiment precedes reversals
        if abs(sentiment) > 0.7:
            return -sentiment * 0.5  # Contrarian at extremes
        else:
            return 0  # Neutral

    elif regime == "macro_dominated":
        # Sentiment not predictive
        return 0  # Don't trade on sentiment

    # Detect regime using multiple factors:
    # - Sentiment-price correlation
    # - Volatility levels
    # - Cross-asset correlations (crude vs. equities)
```

**Solution Hierarchy:**
1. **Detect regime** (correlation analysis, volatility regime)
2. **Adapt signal** (momentum vs contrarian vs neutral)
3. **Reduce size** (in unclear regimes)
4. **Re-evaluate** (periodically backtest different regimes)

**Key Insight:** Sentiment signals are regime-dependent. Build adaptive systems that detect and adjust to changing market dynamics.

---

### Question 9 (8 points)

Which production monitoring metric is MOST critical for a commodity sentiment system?

A) Model prediction accuracy (daily tracking)
B) API cost and latency
C) Sentiment-price correlation over rolling windows
D) Number of articles processed per day

**Answer: C**

**Explanation:**

**A - Prediction Accuracy:**
```python
# Problem: Accuracy is a lagging indicator
# - You only know if prediction was right after the fact
# - By then, you've already traded on it

# Better use: Post-trade analysis, not real-time monitoring
```

**B - API Cost and Latency:**
```python
# Important for operations, but not signal quality
# - Cost overruns don't mean signal is broken
# - Latency issues don't mean sentiment is wrong

# Monitor, but secondary priority
```

**C (Correct) - Sentiment-Price Correlation:**
```python
def monitor_signal_quality(sentiment_data, price_data):
    """Real-time monitoring of sentiment signal quality."""

    # Calculate rolling correlations (30-day window)
    correlations = []
    for i in range(30, len(sentiment_data)):
        sent_window = sentiment_data[i-30:i]
        price_window = price_data[i-30:i]

        corr = np.corrcoef(sent_window, price_window)[0, 1]
        correlations.append({
            'date': dates[i],
            'correlation': corr
        })

    # Alert conditions:
    current_corr = correlations[-1]['correlation']
    avg_corr = np.mean([c['correlation'] for c in correlations[-90:]])  # 90-day avg
    std_corr = np.std([c['correlation'] for c in correlations[-90:]])

    # Alert 1: Correlation dropped significantly
    if current_corr < avg_corr - 2 * std_corr:
        send_alert(
            level="WARNING",
            message=f"Sentiment-price correlation dropped to {current_corr:.2f} "
                   f"(avg: {avg_corr:.2f}, std: {std_corr:.2f})",
            action="Review recent sentiment extractions and market regime"
        )

    # Alert 2: Correlation inverted (sentiment became contrarian)
    if avg_corr > 0.2 and current_corr < -0.2:
        send_alert(
            level="CRITICAL",
            message=f"Sentiment signal inverted: correlation changed from "
                   f"{avg_corr:.2f} to {current_corr:.2f}",
            action="Consider switching to contrarian signal or pausing trading"
        )

    # Alert 3: Correlation weakening over time
    recent_corr_trend = np.polyfit(range(30), [c['correlation'] for c in correlations[-30:]], 1)[0]
    if recent_corr_trend < -0.01:  # Declining by 0.01/day
        send_alert(
            level="INFO",
            message=f"Sentiment signal deteriorating: correlation trend {recent_corr_trend:.4f}/day",
            action="Schedule signal quality review"
        )

    return {
        'current_correlation': current_corr,
        'average_correlation': avg_corr,
        'trend': recent_corr_trend,
        'status': 'healthy' if current_corr > 0.2 else 'degraded' if current_corr > 0 else 'inverted'
    }

# Dashboard metrics:
# ┌─────────────────────────────────────────────────────────┐
# │ Sentiment Signal Health                                 │
# │                                                          │
# │ Current Correlation: 0.42 ✓ (healthy)                   │
# │ 30-Day Average: 0.45                                     │
# │ 90-Day Average: 0.48                                     │
# │ Trend: -0.003/day ⚠ (slowly declining)                  │
# │                                                          │
# │ [Last 90 days correlation chart]                        │
# └─────────────────────────────────────────────────────────┘
```

**Why this is most critical:**
- **Predictive:** Correlations weaken BEFORE accuracy drops
- **Actionable:** Tells you when to reduce size or change strategy
- **Regime-aware:** Detects market regime changes
- **Real-time:** No need to wait for trade outcomes

**D - Articles Processed:**
```python
# Useful but not critical for signal quality
# More articles ≠ better signal
# Monitor for operational reasons (did pipeline break?)
```

**Complete Monitoring Dashboard:**
```python
monitoring_metrics = {
    # Tier 1: Signal Quality (MOST CRITICAL)
    "signal_quality": {
        "sentiment_price_correlation": 0.42,
        "correlation_trend": -0.003,
        "regime": "momentum",
        "alert_status": "healthy"
    },

    # Tier 2: Trading Performance
    "performance": {
        "win_rate_30d": 0.58,
        "sharpe_30d": 1.3,
        "max_drawdown_30d": -0.08
    },

    # Tier 3: Operational
    "operations": {
        "articles_processed_today": 87,
        "api_cost_today": 42.50,
        "avg_latency_sec": 2.3,
        "failed_extractions_today": 2
    }
}
```

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | C | 12 | Multi-factor sentiment extraction |
| 2 | B | 10 | Commodity-specific data sources |
| 3 | C | 13 | Credibility-weighted aggregation |
| 4 | C | 15 | Contrarian sentiment signals |
| 5 | C | 10 | Backtest-based weight optimization |
| 6 | D | 10 | Losing trade analysis |
| 7 | C | 12 | Expected value model comparison |
| 8 | D | 10 | Market regime change detection |
| 9 | C | 8 | Sentiment-price correlation monitoring |

**Total:** 100 points

---

## Grading Scale

- **90-100:** Excellent - Advanced sentiment analysis skills
- **80-89:** Good - Solid understanding, ready for production
- **70-79:** Adequate - Review signal construction and evaluation
- **Below 70:** Needs improvement - Revisit extraction and aggregation methods

---

## Key Takeaways

1. **Multi-factor extraction** beats simple sentiment scores
2. **Credibility-weighted aggregation** prevents noise from dominating signal
3. **Sentiment can be momentum, confirming, or contrarian** - backtest to determine
4. **Market regimes change** - build adaptive systems
5. **Monitor correlation, not just accuracy** - predictive vs lagging metrics

---

## Next Steps

**Score 90-100:** Proceed to Module 4 (Fundamentals Modeling)
**Score 80-89:** Review backtesting methodologies, then proceed
**Score 70-79:** Practice signal construction and regime detection
**Score <70:** Revisit Module 3, focus on aggregation and evaluation

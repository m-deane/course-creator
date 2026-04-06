# News Sentiment Analysis for Commodities

> **Reading time:** ~6 min | **Module:** Module 3: Sentiment | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** News sentiment analysis extracts directional signals from commodity market news. Unlike general sentiment analysis, commodity sentiment requires domain expertise to interpret supply/demand implications.

</div>

## Introduction

News sentiment analysis extracts directional signals from commodity market news. Unlike general sentiment analysis, commodity sentiment requires domain expertise to interpret supply/demand implications.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Commodity-Specific Sentiment

### Why General Sentiment Fails

Generic sentiment models miss commodity nuance:

| Headline | General Sentiment | Commodity Sentiment |
|----------|-------------------|---------------------|
| "Oil production surges to record" | Positive | Bearish (oversupply) |
| "Drought devastates corn crop" | Negative | Bullish (supply shortage) |
| "China demand disappoints" | Negative | Bearish (weak demand) |

### Supply vs. Demand Framework

```python
from anthropic import Anthropic
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class Sentiment(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class Driver(Enum):
    SUPPLY = "supply"
    DEMAND = "demand"
    INVENTORY = "inventory"
    GEOPOLITICAL = "geopolitical"
    TECHNICAL = "technical"
    MACRO = "macro"

@dataclass
class CommoditySentiment:
    commodity: str
    sentiment: Sentiment
    driver: Driver
    confidence: float
    reasoning: str
    time_horizon: str  # immediate, near-term, medium-term

client = Anthropic()

def analyze_commodity_sentiment(headline: str, body: str = "") -> CommoditySentiment:
    """
    Analyze commodity news with supply/demand framework.
    """
    prompt = f"""Analyze this commodity news for trading sentiment.

Headline: {headline}
{f"Body: {body}" if body else ""}

Consider:
1. Is this about supply (production, exports) or demand (consumption, imports)?
2. Is the news above or below market expectations?
3. What is the price implication?

Return JSON:
{{
  "commodity": "<identified commodity or 'general_energy'/'general_ags'>",
  "sentiment": "bullish|bearish|neutral",
  "driver": "supply|demand|inventory|geopolitical|technical|macro",
  "confidence": <0-1>,
  "reasoning": "<one sentence explanation>",
  "time_horizon": "immediate|near_term|medium_term"
}}

Rules:
- Supply increase/demand decrease = bearish
- Supply decrease/demand increase = bullish
- Consider seasonal expectations
- "Record production" is typically bearish
- "Record demand" is typically bullish"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}]
    )

    import json
    result = json.loads(response.content[0].text)

    return CommoditySentiment(
        commodity=result['commodity'],
        sentiment=Sentiment(result['sentiment']),
        driver=Driver(result['driver']),
        confidence=result['confidence'],
        reasoning=result['reasoning'],
        time_horizon=result['time_horizon']
    )
```

## Batch Processing News Feeds

### News Aggregation

```python
from datetime import datetime, timedelta
from typing import List
import asyncio

@dataclass
class NewsItem:
    timestamp: datetime
    headline: str
    body: str
    source: str
    url: str

@dataclass
class ScoredNews:
    news: NewsItem
    sentiment: CommoditySentiment
    processed_at: datetime

async def process_news_batch(
    news_items: List[NewsItem],
    target_commodity: str = None
) -> List[ScoredNews]:
    """
    Process batch of news items, optionally filtering by commodity.
    """
    results = []

    for item in news_items:
        sentiment = analyze_commodity_sentiment(item.headline, item.body)

        # Filter by target commodity if specified
        if target_commodity and sentiment.commodity.lower() != target_commodity.lower():
            continue

        results.append(ScoredNews(
            news=item,
            sentiment=sentiment,
            processed_at=datetime.now()
        ))

    return results

def aggregate_sentiment(scored_news: List[ScoredNews], hours: int = 24) -> dict:
    """
    Aggregate sentiment scores over time period.
    """
    cutoff = datetime.now() - timedelta(hours=hours)
    recent = [s for s in scored_news if s.news.timestamp > cutoff]

    if not recent:
        return {"sentiment": "neutral", "confidence": 0, "count": 0}

    bullish = sum(1 for s in recent if s.sentiment.sentiment == Sentiment.BULLISH)
    bearish = sum(1 for s in recent if s.sentiment.sentiment == Sentiment.BEARISH)
    total = len(recent)

    # Weighted by confidence
    bullish_score = sum(
        s.sentiment.confidence for s in recent
        if s.sentiment.sentiment == Sentiment.BULLISH
    )
    bearish_score = sum(
        s.sentiment.confidence for s in recent
        if s.sentiment.sentiment == Sentiment.BEARISH
    )

    net_score = (bullish_score - bearish_score) / total if total > 0 else 0

    return {
        "net_sentiment": net_score,  # -1 to 1
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": total - bullish - bearish,
        "total_articles": total,
        "avg_confidence": sum(s.sentiment.confidence for s in recent) / total
    }
```

## Handling Specific News Types

### Earnings Calls

```python
def analyze_earnings_commodity_mentions(transcript: str, company: str) -> dict:
    """
    Extract commodity-relevant commentary from earnings transcripts.
    """
    prompt = f"""Analyze this earnings transcript from {company} for commodity market insights.

Transcript excerpt:
{transcript[:4000]}  # Limit length

Extract:
1. Any mentions of commodity prices (oil, gas, metals, agricultural)
2. Production or volume guidance
3. Demand commentary
4. Hedging or price assumptions
5. Capital expenditure plans affecting supply

Return JSON:
{{
  "commodities_mentioned": ["list of commodities"],
  "price_commentary": [
    {{"commodity": "...", "outlook": "higher|lower|stable", "quote": "..."}}
  ],
  "production_guidance": [
    {{"commodity": "...", "direction": "increase|decrease|flat", "magnitude": "...", "quote": "..."}}
  ],
  "demand_commentary": {{
    "outlook": "strong|weak|mixed",
    "key_quote": "..."
  }},
  "overall_commodity_sentiment": "bullish|bearish|neutral"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Government Reports


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">analyze_government_report_surprise.py</span>
</div>

```python
def analyze_government_report_surprise(
    report_text: str,
    expectations: dict
) -> dict:
    """
    Analyze government report vs market expectations.
    """
    prompt = f"""Compare this government report against market expectations.

Report:
{report_text[:3000]}

Market Expectations:
{expectations}

For each metric in the expectations:
1. Extract the actual reported value
2. Calculate surprise (actual - expected)
3. Determine if surprise is bullish or bearish for prices

Return JSON:
{{
  "surprises": [
    {{
      "metric": "...",
      "expected": <value>,
      "actual": <value>,
      "surprise": <value>,
      "surprise_pct": <percentage>,
      "price_impact": "bullish|bearish|neutral",
      "magnitude": "large|moderate|small"
    }}
  ],
  "overall_surprise": "bullish|bearish|neutral",
  "key_takeaway": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

</div>
</div>

## Building Sentiment Signals

### Creating Trading Signals


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">create_sentiment_signal.py</span>
</div>

```python
import pandas as pd
import numpy as np

def create_sentiment_signal(
    sentiment_scores: pd.DataFrame,
    lookback: int = 5,
    threshold: float = 0.3
) -> pd.Series:
    """
    Convert sentiment scores to trading signals.

    Args:
        sentiment_scores: DataFrame with 'date' and 'net_sentiment' columns
        lookback: Days for moving average
        threshold: Signal threshold

    Returns:
        Series with signals: 1 (long), -1 (short), 0 (neutral)
    """
    # Calculate moving average of sentiment
    sentiment_ma = sentiment_scores['net_sentiment'].rolling(lookback).mean()

    # Generate signals
    signals = pd.Series(0, index=sentiment_scores.index)
    signals[sentiment_ma > threshold] = 1
    signals[sentiment_ma < -threshold] = -1

    return signals

def combine_with_price_momentum(
    sentiment_signal: pd.Series,
    price_data: pd.Series,
    momentum_window: int = 20
) -> pd.Series:
    """
    Combine sentiment with price momentum for confirmation.
    """
    # Calculate momentum
    momentum = price_data.pct_change(momentum_window)
    momentum_signal = np.sign(momentum)

    # Only take sentiment signals confirmed by momentum
    confirmed_signal = sentiment_signal.copy()
    confirmed_signal[(sentiment_signal != momentum_signal)] = 0

    return confirmed_signal
```

</div>
</div>

### Backtesting Framework


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">backtest_sentiment_strategy.py</span>
</div>

```python
def backtest_sentiment_strategy(
    prices: pd.Series,
    signals: pd.Series,
    transaction_cost: float = 0.001
) -> dict:
    """
    Backtest sentiment-based trading strategy.
    """
    # Calculate returns
    price_returns = prices.pct_change()

    # Strategy returns (signal from previous day)
    strategy_returns = signals.shift(1) * price_returns

    # Apply transaction costs
    position_changes = signals.diff().abs()
    costs = position_changes * transaction_cost
    strategy_returns = strategy_returns - costs

    # Calculate metrics
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    max_dd = (strategy_returns.cumsum() - strategy_returns.cumsum().cummax()).min()

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': (strategy_returns > 0).mean(),
        'avg_win': strategy_returns[strategy_returns > 0].mean(),
        'avg_loss': strategy_returns[strategy_returns < 0].mean()
    }
```

</div>
</div>

## Handling Noise and False Signals

### Confidence-Weighted Averaging


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">confidence_weighted_sentiment.py</span>
</div>

```python
def confidence_weighted_sentiment(
    sentiment_list: List[CommoditySentiment]
) -> float:
    """
    Calculate confidence-weighted net sentiment.
    """
    if not sentiment_list:
        return 0.0

    total_weight = sum(s.confidence for s in sentiment_list)

    weighted_sum = sum(
        s.confidence * (1 if s.sentiment == Sentiment.BULLISH else
                       -1 if s.sentiment == Sentiment.BEARISH else 0)
        for s in sentiment_list
    )

    return weighted_sum / total_weight if total_weight > 0 else 0.0
```

</div>
</div>

### Source Quality Weighting


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">source_weighted_sentiment.py</span>

```python
SOURCE_QUALITY = {
    'reuters': 1.0,
    'bloomberg': 1.0,
    'wsj': 0.9,
    'ft': 0.9,
    'platts': 1.0,  # Commodity-specific
    'argus': 1.0,
    'eia': 1.0,
    'twitter': 0.5,
    'reddit': 0.3,
    'unknown': 0.5
}

def source_weighted_sentiment(scored_news: List[ScoredNews]) -> float:
    """
    Calculate sentiment weighted by source quality.
    """
    total_weight = 0
    weighted_sum = 0

    for item in scored_news:
        source_weight = SOURCE_QUALITY.get(
            item.news.source.lower(),
            SOURCE_QUALITY['unknown']
        )
        weight = source_weight * item.sentiment.confidence

        sentiment_value = (
            1 if item.sentiment.sentiment == Sentiment.BULLISH else
            -1 if item.sentiment.sentiment == Sentiment.BEARISH else 0
        )

        weighted_sum += weight * sentiment_value
        total_weight += weight

    return weighted_sum / total_weight if total_weight > 0 else 0.0
```


<div class="callout-insight">

**Insight:** Understanding news sentiment analysis for commodities is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Key Takeaways

1. **Commodity sentiment differs from general sentiment** - supply increases are bearish, demand increases are bullish

2. **Context matters** - the same news can be bullish or bearish depending on expectations

3. **Weight by confidence and source** - not all signals are equal

4. **Combine with other signals** - sentiment works best as confirmation

5. **Validate with backtests** - measure actual predictive power before trading

---

## Conceptual Practice Questions

1. Why is financial news sentiment analysis harder than general sentiment analysis?

2. What domain-specific challenges arise when applying LLMs to commodity news?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./01_news_sentiment_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_news_sentiment.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_news_processing.md">
  <div class="link-card-title">01 News Processing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_sentiment_aggregation.md">
  <div class="link-card-title">02 Sentiment Aggregation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


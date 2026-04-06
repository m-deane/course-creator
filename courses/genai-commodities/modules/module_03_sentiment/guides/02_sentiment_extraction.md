# Extracting Sentiment from Commodity Text

> **Reading time:** ~12 min | **Module:** Module 3: Sentiment | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** Sentiment extraction for commodities uses LLMs to classify market direction (bullish/bearish/neutral) from news and reports while accounting for commodity-specific language, implicit sentiment, multi-commodity relationships, and the distinction between price sentiment and fundamental sentiment.

</div>

## In Brief

Sentiment extraction for commodities uses LLMs to classify market direction (bullish/bearish/neutral) from news and reports while accounting for commodity-specific language, implicit sentiment, multi-commodity relationships, and the distinction between price sentiment and fundamental sentiment.

<div class="callout-insight">

**Insight:** Standard financial sentiment models fail for commodities because "increased production" is bearish for prices (more supply) but bullish for producer stocks, and context matters enormously—"low inventory" is bearish in agriculture (crop shortage) but can be bullish in energy (strong demand). LLMs excel at this contextual reasoning that simpler models miss.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Think of commodity sentiment extraction like translating news into trading signals:

**Simple sentiment (wrong):**
- Article says "Production is up 10%"
- Simple model: "Up" sounds positive → bullish
- **But actually**: More production = more supply = bearish for prices

**Commodity-aware sentiment (correct):**
- Article says "Production is up 10%"
- Context: We're analyzing crude oil prices
- Reasoning: Increased production → increased supply → downward price pressure
- **Correct sentiment**: Bearish for crude oil prices

**Additional complexity:**
- "OPEC announces surprise production cut" → Bullish (restricted supply)
- "Refinery utilization hits 95%, highest in 3 years" → Bullish for products (gasoline, diesel)
- "Corn planting ahead of schedule with favorable weather" → Bearish for corn prices (good crop expected)

The LLM must understand supply/demand dynamics, seasonal factors, and multi-commodity relationships.

## Formal Definition

Commodity sentiment extraction is a function **SE: T → S** where:

**Input:** T = text (news article, report section, earnings call transcript)

**Output:** S = sentiment structure:
```
Sentiment = {
  overall_direction: {bullish | bearish | neutral | mixed},
  confidence: [0, 1],
  dimensions: {
    price_sentiment: {direction, confidence, reasoning},
    fundamental_sentiment: {direction, confidence, reasoning},
    supply_sentiment: {direction, confidence, reasoning},
    demand_sentiment: {direction, confidence, reasoning}
  },
  time_horizon: {short_term | medium_term | long_term},
  affected_commodities: [(commodity, sentiment_direction, strength)],
  key_factors: [list of bullish/bearish drivers],
  implicit_sentiment: bool,  # Sentiment not explicitly stated
  caveat_phrases: [phrases indicating uncertainty]
}
```

**Sentiment logic for commodities:**
- **Supply increase** → bearish price sentiment (more supply = lower prices)
- **Demand increase** → bullish price sentiment (more demand = higher prices)
- **Inventory decrease** → context-dependent (could be strong demand or supply disruption)
- **Production cuts** → bullish price sentiment (restricted supply)

## Code Implementation

### LLM-Based Sentiment Extraction

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from anthropic import Anthropic
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum
import json

class SentimentDirection(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    MIXED = "mixed"

class TimeHorizon(Enum):
    SHORT_TERM = "short_term"  # Days to weeks
    MEDIUM_TERM = "medium_term"  # Months
    LONG_TERM = "long_term"  # Quarters to years

@dataclass
class DimensionalSentiment:
    """Sentiment across multiple dimensions."""
    direction: SentimentDirection
    confidence: float
    reasoning: str

@dataclass
class CommoditySentiment:
    """Complete sentiment analysis for commodity text."""
    overall_direction: SentimentDirection
    confidence: float

    # Multi-dimensional sentiment
    price_sentiment: DimensionalSentiment
    fundamental_sentiment: DimensionalSentiment
    supply_sentiment: DimensionalSentiment
    demand_sentiment: DimensionalSentiment

    # Metadata
    time_horizon: TimeHorizon
    affected_commodities: List[Dict[str, any]]
    key_factors: Dict[str, List[str]]  # {bullish: [...], bearish: [...]}
    implicit_sentiment: bool
    caveat_phrases: List[str]

    # Source text
    source_text: str

class CommoditySentimentExtractor:
    """
    Extract sentiment from commodity text using LLM.
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def extract_sentiment(
        self,
        text: str,
        commodity: Optional[str] = None,
        context: Optional[str] = None
    ) -> CommoditySentiment:
        """
        Extract comprehensive sentiment from text.

        Args:
            text: News article, report section, or other commodity text
            commodity: Specific commodity to analyze (if known)
            context: Additional context (e.g., "We are analyzing crude oil prices")

        Returns:
            Structured sentiment analysis
        """
        prompt = self._build_sentiment_prompt(text, commodity, context)

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse response
        sentiment_data = json.loads(response.content[0].text)

        return self._parse_sentiment_response(sentiment_data, text)

    def _build_sentiment_prompt(
        self,
        text: str,
        commodity: Optional[str],
        context: Optional[str]
    ) -> str:
        """
        Build prompt for sentiment extraction.
        """
        commodity_context = f"Focus on {commodity}." if commodity else "Identify all mentioned commodities."
        additional_context = f"\nContext: {context}" if context else ""

        prompt = f"""Analyze the sentiment of this commodity market text.

Text:
{text}

{commodity_context}{additional_context}

IMPORTANT - Commodity Sentiment Rules:
1. SUPPLY INCREASE → Bearish for prices (more supply = lower prices)
2. DEMAND INCREASE → Bullish for prices (more demand = higher prices)
3. PRODUCTION CUTS → Bullish for prices (less supply = higher prices)
4. INVENTORY BUILD → Generally bearish (excess supply)
5. INVENTORY DRAW → Context-dependent (strong demand or supply shortage)

Return comprehensive sentiment analysis as JSON:
{{
  "overall_direction": "bullish | bearish | neutral | mixed",
  "confidence": 0.0-1.0,
  "price_sentiment": {{
    "direction": "bullish | bearish | neutral",
    "confidence": 0.0-1.0,
    "reasoning": "Explain why this is bullish/bearish for PRICES"
  }},
  "fundamental_sentiment": {{
    "direction": "bullish | bearish | neutral",
    "confidence": 0.0-1.0,
    "reasoning": "Explain underlying market fundamentals"
  }},
  "supply_sentiment": {{
    "direction": "increasing | decreasing | stable",
    "confidence": 0.0-1.0,
    "reasoning": "What's happening with supply"
  }},
  "demand_sentiment": {{
    "direction": "increasing | decreasing | stable",
    "confidence": 0.0-1.0,
    "reasoning": "What's happening with demand"
  }},
  "time_horizon": "short_term | medium_term | long_term",
  "affected_commodities": [
    {{
      "commodity": "crude_oil",
      "sentiment": "bullish",
      "strength": 0.0-1.0,
      "reasoning": "Why this commodity is affected"
    }}
  ],
  "key_factors": {{
    "bullish": ["factor 1", "factor 2"],
    "bearish": ["factor 1", "factor 2"]
  }},
  "implicit_sentiment": true/false,
  "caveat_phrases": ["however", "but", "on the other hand"]
}}

Examples:
- "OPEC cuts production by 1 million bpd" → BULLISH (less supply)
- "US crude oil production hits record high" → BEARISH (more supply)
- "Strong refinery demand drains crude inventories" → BULLISH (strong demand)
- "Mild winter reduces natural gas consumption" → BEARISH (weak demand)"""

        return prompt

    def _parse_sentiment_response(
        self,
        data: Dict,
        source_text: str
    ) -> CommoditySentiment:
        """
        Parse LLM response into structured sentiment.
        """
        return CommoditySentiment(
            overall_direction=SentimentDirection(data["overall_direction"]),
            confidence=data["confidence"],
            price_sentiment=DimensionalSentiment(
                direction=SentimentDirection(data["price_sentiment"]["direction"]),
                confidence=data["price_sentiment"]["confidence"],
                reasoning=data["price_sentiment"]["reasoning"]
            ),
            fundamental_sentiment=DimensionalSentiment(
                direction=SentimentDirection(data["fundamental_sentiment"]["direction"]),
                confidence=data["fundamental_sentiment"]["confidence"],
                reasoning=data["fundamental_sentiment"]["reasoning"]
            ),
            supply_sentiment=DimensionalSentiment(
                direction=SentimentDirection(data["supply_sentiment"]["direction"]),
                confidence=data["supply_sentiment"]["confidence"],
                reasoning=data["supply_sentiment"]["reasoning"]
            ),
            demand_sentiment=DimensionalSentiment(
                direction=SentimentDirection(data["demand_sentiment"]["direction"]),
                confidence=data["demand_sentiment"]["confidence"],
                reasoning=data["demand_sentiment"]["reasoning"]
            ),
            time_horizon=TimeHorizon(data["time_horizon"]),
            affected_commodities=data["affected_commodities"],
            key_factors=data["key_factors"],
            implicit_sentiment=data["implicit_sentiment"],
            caveat_phrases=data["caveat_phrases"],
            source_text=source_text
        )

    def batch_extract_sentiment(
        self,
        texts: List[str],
        commodity: Optional[str] = None
    ) -> List[CommoditySentiment]:
        """
        Extract sentiment from multiple texts.
        """
        sentiments = []

        for text in texts:
            try:
                sentiment = self.extract_sentiment(text, commodity)
                sentiments.append(sentiment)
            except Exception as e:
                print(f"Error extracting sentiment: {e}")
                # Continue with other texts

        return sentiments
```

</div>

### Aspect-Based Sentiment Analysis

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">aspectsentimentextractor.py</span>
</div>

```python
class AspectSentimentExtractor:
    """
    Extract sentiment for specific aspects of commodity markets.

    Aspects: production, inventory, demand, exports, prices, forecasts
    """

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def extract_aspect_sentiments(
        self,
        text: str,
        aspects: List[str]
    ) -> Dict[str, DimensionalSentiment]:
        """
        Extract sentiment for each specified aspect.

        Args:
            text: Source text
            aspects: List of aspects (e.g., ["production", "inventory", "demand"])

        Returns:
            Dict mapping aspect to its sentiment
        """
        prompt = f"""Analyze sentiment for specific aspects in this commodity text.

Text:
{text}

Extract sentiment for these aspects: {', '.join(aspects)}

Return JSON:
{{
  "production": {{
    "direction": "bullish | bearish | neutral",
    "confidence": 0.0-1.0,
    "reasoning": "What the text says about production",
    "mentioned": true/false
  }},
  "inventory": {{...}},
  "demand": {{...}}
}}

For each aspect:
- If not mentioned, set mentioned=false
- If mentioned, analyze the sentiment considering:
  - Production increase → bearish for prices
  - Inventory decrease → generally bullish (tight supply)
  - Demand increase → bullish for prices"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1536,
            messages=[{"role": "user", "content": prompt}]
        )

        aspect_data = json.loads(response.content[0].text)

        # Convert to DimensionalSentiment objects
        aspect_sentiments = {}
        for aspect, data in aspect_data.items():
            if data.get("mentioned", False):
                aspect_sentiments[aspect] = DimensionalSentiment(
                    direction=SentimentDirection(data["direction"]),
                    confidence=data["confidence"],
                    reasoning=data["reasoning"]
                )

        return aspect_sentiments
```

</div>

### Sentiment Aggregation Across Sources

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">sentimentaggregator.py</span>
</div>

```python
from collections import defaultdict
from datetime import datetime, timedelta

class SentimentAggregator:
    """
    Aggregate sentiment across multiple news sources and time periods.
    """

    def __init__(self):
        self.sentiment_history = defaultdict(list)

    def add_sentiment(
        self,
        commodity: str,
        sentiment: CommoditySentiment,
        timestamp: datetime
    ):
        """
        Add sentiment to aggregation.
        """
        self.sentiment_history[commodity].append({
            'sentiment': sentiment,
            'timestamp': timestamp
        })

    def get_aggregate_sentiment(
        self,
        commodity: str,
        time_window: timedelta = timedelta(hours=24)
    ) -> Dict[str, any]:
        """
        Get aggregated sentiment over time window.

        Returns:
            Aggregated sentiment with:
            - Consensus direction
            - Confidence (based on agreement)
            - Sentiment distribution
            - Trend (becoming more/less bullish)
        """
        cutoff_time = datetime.now() - time_window

        # Get recent sentiments
        recent = [
            item for item in self.sentiment_history[commodity]
            if item['timestamp'] >= cutoff_time
        ]

        if not recent:
            return {
                'consensus': 'neutral',
                'confidence': 0.0,
                'count': 0
            }

        # Count sentiment directions
        bullish_count = sum(
            1 for item in recent
            if item['sentiment'].overall_direction == SentimentDirection.BULLISH
        )
        bearish_count = sum(
            1 for item in recent
            if item['sentiment'].overall_direction == SentimentDirection.BEARISH
        )
        neutral_count = len(recent) - bullish_count - bearish_count

        total = len(recent)

        # Determine consensus
        if bullish_count > bearish_count and bullish_count / total > 0.6:
            consensus = 'bullish'
            confidence = bullish_count / total
        elif bearish_count > bullish_count and bearish_count / total > 0.6:
            consensus = 'bearish'
            confidence = bearish_count / total
        else:
            consensus = 'mixed'
            confidence = 1 - max(bullish_count, bearish_count, neutral_count) / total

        # Calculate sentiment trend (comparing first half to second half of window)
        mid_point = cutoff_time + (time_window / 2)
        early = [item for item in recent if item['timestamp'] < mid_point]
        late = [item for item in recent if item['timestamp'] >= mid_point]

        early_bullish_pct = (
            sum(1 for item in early if item['sentiment'].overall_direction == SentimentDirection.BULLISH) / len(early)
            if early else 0
        )
        late_bullish_pct = (
            sum(1 for item in late if item['sentiment'].overall_direction == SentimentDirection.BULLISH) / len(late)
            if late else 0
        )

        sentiment_change = late_bullish_pct - early_bullish_pct

        if sentiment_change > 0.2:
            trend = "becoming_more_bullish"
        elif sentiment_change < -0.2:
            trend = "becoming_more_bearish"
        else:
            trend = "stable"

        return {
            'consensus': consensus,
            'confidence': confidence,
            'count': total,
            'distribution': {
                'bullish': bullish_count / total,
                'bearish': bearish_count / total,
                'neutral': neutral_count / total
            },
            'trend': trend,
            'sentiment_change': sentiment_change
        }
```

</div>

### Example Usage

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# Initialize extractor
extractor = CommoditySentimentExtractor(anthropic_api_key="your_key")

# Example news text
news_text = """
OPEC+ announced a surprise production cut of 1.16 million barrels per day,
effective May 1st. The decision was made to stabilize oil markets amid
concerns about global economic growth. Saudi Arabia will contribute the
largest share with a cut of 500,000 bpd.

This comes as U.S. crude oil inventories have fallen for four consecutive
weeks, driven by strong refinery demand and robust exports. Analysts expect
this to put upward pressure on oil prices in the coming weeks.
"""

# Extract sentiment
sentiment = extractor.extract_sentiment(
    text=news_text,
    commodity="crude_oil",
    context="We are analyzing crude oil prices for trading"
)

print(f"Overall: {sentiment.overall_direction.value}")
print(f"Confidence: {sentiment.confidence:.2f}")
print(f"Price Sentiment: {sentiment.price_sentiment.direction.value}")
print(f"Reasoning: {sentiment.price_sentiment.reasoning}")
print(f"Key Bullish Factors: {sentiment.key_factors.get('bullish', [])}")
print(f"Key Bearish Factors: {sentiment.key_factors.get('bearish', [])}")

# Aspect-based analysis
aspect_extractor = AspectSentimentExtractor(anthropic_api_key="your_key")

aspect_sentiments = aspect_extractor.extract_aspect_sentiments(
    text=news_text,
    aspects=["production", "inventory", "demand"]
)

for aspect, sent in aspect_sentiments.items():
    print(f"\n{aspect.upper()}")
    print(f"  Direction: {sent.direction.value}")
    print(f"  Reasoning: {sent.reasoning}")

# Aggregate over time
aggregator = SentimentAggregator()

# Add sentiments (would come from news stream)
aggregator.add_sentiment("crude_oil", sentiment, datetime.now())

# Get aggregate
aggregate = aggregator.get_aggregate_sentiment(
    commodity="crude_oil",
    time_window=timedelta(hours=24)
)

print(f"\n24-Hour Consensus: {aggregate['consensus']}")
print(f"Confidence: {aggregate['confidence']:.2f}")
print(f"Trend: {aggregate['trend']}")
```

</div>

## Common Pitfalls

**1. Confusing Supply Sentiment with Price Sentiment**
- **Problem**: Treating "production increase" as bullish because "increase" sounds positive
- **Why it happens**: Not accounting for supply/demand mechanics
- **Solution**: Explicitly separate supply/demand sentiment from price sentiment; use commodity-specific prompt engineering

**2. Ignoring Time Horizon**
- **Problem**: "Long-term bearish, short-term bullish" collapsed into single sentiment
- **Why it happens**: No time horizon dimension in sentiment model
- **Solution**: Extract time_horizon; allow for conflicting short/long-term sentiments

**3. Missing Implicit Sentiment**
- **Problem**: "OPEC maintains current production levels" seems neutral but is actually bearish (market expected cuts)
- **Why it happens**: LLM lacks context about market expectations
- **Solution**: Provide market context in prompt; track expectations vs. outcomes

**4. Overconfidence in Mixed Signals**
- **Problem**: Article mentions both bullish and bearish factors but model picks one
- **Why it happens**: Forcing single direction classification
- **Solution**: Allow "mixed" sentiment; report both bullish and bearish factors with weights

**5. Geography-Specific Sentiment Confusion**
- **Problem**: "Saudi production cut" is bullish globally but bearish for Saudi exports
- **Why it happens**: No geography dimension in sentiment
- **Solution**: Extract affected geographies; allow different sentiment per region

## Connections

**Builds on:**
- News processing (acquiring and filtering commodity news)
- LLM prompt engineering (designing effective sentiment prompts)
- Commodity market fundamentals (understanding supply/demand)

**Leads to:**
- Signal construction (converting sentiment to trading signals)
- Sentiment-based forecasting (predicting price moves from sentiment)
- Portfolio positioning (using sentiment for risk management)

**Related to:**
- Natural language understanding (interpreting complex text)
- Time-series analysis (tracking sentiment over time)
- Multi-modal analysis (combining sentiment with price data)

## Practice Problems

1. **Sentiment Polarity Challenge**
   - Classify sentiment for these statements:
     - "U.S. crude production reaches new record high"
     - "OPEC announces deeper production cuts"
     - "Warm winter reduces natural gas heating demand"
     - "Drought threatens corn crop, yields expected to fall"
   - For each, identify: price sentiment, supply/demand impact, time horizon

2. **Implicit Sentiment Detection**
   - Find implicit sentiment in:
     - "OPEC+ maintains current production levels" (when market expected cuts)
     - "Inventory builds slower than expected" (still building, but less)
     - "Weather forecast shows normal temperatures" (no extreme cold/heat)
   - What additional context do you need to determine sentiment?

3. **Multi-Commodity Sentiment**
   - Extract sentiment from:
     - "Crude oil rally boosts refinery margins, increasing gasoline production"
   - Identify:
     - Sentiment for crude oil
     - Sentiment for gasoline
     - Relationship between the two
     - Are they aligned or opposed?

4. **Sentiment Aggregation**
   - Given 10 news articles about crude oil from the past 24 hours:
     - 6 bullish, 3 bearish, 1 neutral
     - Design an aggregation method that produces:
       - Consensus sentiment
       - Confidence level
       - Sentiment strength (how bullish/bearish)

5. **Aspect-Based Analysis**
   - For an EIA weekly petroleum report, extract aspect-specific sentiment for:
     - Crude oil inventory (actual vs. expected)
     - Production levels
     - Refinery utilization
     - Product demand
   - Combine into overall market sentiment

<div class="callout-insight">

**Insight:** Understanding extracting sentiment from commodity text is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

**Sentiment Analysis Fundamentals:**
- "Sentiment Analysis and Opinion Mining" (Bing Liu) - Comprehensive overview
- "Deep Learning for Sentiment Analysis" - Modern approaches

**Financial Sentiment:**
- "Sentiment Analysis in Finance" - Domain-specific techniques
- "FinBERT: Pre-trained Financial Language Model" - Specialized models

**LLM-Based Analysis:**
- "Large Language Models for Financial Sentiment Analysis" (2024)
- Anthropic: "Prompt Engineering for Classification Tasks"

**Commodity Markets:**
- "Commodity Markets and Prices" - Understanding supply/demand dynamics
- "Technical and Fundamental Analysis in Commodity Markets"

**Production Systems:**
- "Real-Time Sentiment Analysis at Scale" - Infrastructure considerations
- "Handling Noisy Social Media Data for Sentiment" - Data quality challenges

---

## Conceptual Practice Questions

1. How do you aggregate article-level sentiments into a tradeable market signal?

2. What decay function would you use for time-weighting sentiment scores and why?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="../notebooks/01_news_sentiment.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_news_processing.md">
  <div class="link-card-title">01 News Processing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_news_sentiment.md">
  <div class="link-card-title">01 News Sentiment</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


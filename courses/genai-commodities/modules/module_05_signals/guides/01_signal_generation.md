# Trading Signal Generation with LLMs

> **Reading time:** ~6 min | **Module:** Module 5: Signals | **Prerequisites:** Modules 0-4

<div class="callout-key">

**Key Concept Summary:** This guide covers converting LLM-extracted insights into quantitative trading signals.

</div>

## From Text to Trades

This guide covers converting LLM-extracted insights into quantitative trading signals.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Signal Architecture

### The Signal Pipeline

```
Raw Data → LLM Processing → Structured Data → Signal Logic → Position Sizing → Execution
   │              │               │               │              │            │
Reports      Extraction      JSON/DB        Rules/ML      Risk Mgmt     Orders
News         Sentiment       Numbers        Scoring       Limits        Timing
Transcripts  Classification  Timeseries     Ranking       Correlation   Slippage
```

### Signal Types

| Signal Type | Description | Example |
|-------------|-------------|---------|
| **Event** | Binary trigger from specific news | OPEC cut announcement |
| **Sentiment** | Aggregated directional view | News sentiment score |
| **Fundamental** | Supply/demand derived | Inventory surprise |
| **Comparative** | Cross-commodity | Relative value |

## Building Fundamental Signals

### Inventory Surprise Signal

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class InventorySignal:
    commodity: str
    report_date: pd.Timestamp
    actual: float
    expected: float
    surprise: float
    surprise_zscore: float
    signal: int  # -1, 0, 1
    confidence: float

class InventorySurpriseSignal:
    """
    Generate signals from inventory report surprises.
    """

    def __init__(self, history_window: int = 52):
        self.history_window = history_window
        self.surprise_history = []

    def add_observation(self, actual: float, expected: float, date: pd.Timestamp):
        """Record new inventory observation."""
        surprise = actual - expected
        self.surprise_history.append({
            'date': date,
            'actual': actual,
            'expected': expected,
            'surprise': surprise
        })

    def generate_signal(
        self,
        actual: float,
        expected: float,
        date: pd.Timestamp,
        threshold: float = 1.5
    ) -> InventorySignal:
        """
        Generate trading signal from inventory surprise.

        Args:
            actual: Reported inventory
            expected: Consensus expectation
            date: Report date
            threshold: Z-score threshold for signal

        Returns:
            InventorySignal with direction and confidence
        """
        surprise = actual - expected
        self.add_observation(actual, expected, date)

        # Calculate z-score vs historical surprises
        if len(self.surprise_history) >= 10:
            hist_surprises = [h['surprise'] for h in self.surprise_history[-self.history_window:]]
            mean_surprise = np.mean(hist_surprises)
            std_surprise = np.std(hist_surprises)
            zscore = (surprise - mean_surprise) / std_surprise if std_surprise > 0 else 0
        else:
            zscore = 0

        # Generate signal
        # Build = bearish (supply increase), Draw = bullish (supply decrease)
        if zscore > threshold:
            signal = -1  # Surprise build = bearish
            confidence = min(abs(zscore) / 3, 1.0)
        elif zscore < -threshold:
            signal = 1  # Surprise draw = bullish
            confidence = min(abs(zscore) / 3, 1.0)
        else:
            signal = 0
            confidence = 0.0

        return InventorySignal(
            commodity='crude_oil',
            report_date=date,
            actual=actual,
            expected=expected,
            surprise=surprise,
            surprise_zscore=zscore,
            signal=signal,
            confidence=confidence
        )
```

</div>

### Production Forecast Signal

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">productionforecastsignal.py</span>
</div>

```python
from anthropic import Anthropic

client = Anthropic()

class ProductionForecastSignal:
    """
    Generate signals from production forecast revisions.
    """

    def __init__(self):
        self.forecast_history = {}

    def extract_production_forecast(self, report_text: str, source: str) -> dict:
        """Extract production forecasts from report."""
        prompt = """Extract production forecasts from this report.

Return JSON:
{
  "forecasts": [
    {
      "region": "US|OPEC|Russia|Brazil|etc",
      "period": "Q1 2024|2024|etc",
      "production_mbd": <million barrels per day>,
      "revision_vs_prior": <change from last month's forecast>
    }
  ]
}

Report:
""" + report_text

        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        return json.loads(response.content[0].text)

    def generate_signal(self, current_forecast: dict, prior_forecast: dict) -> dict:
        """
        Generate signal from forecast revision.
        """
        signals = []

        for curr in current_forecast.get('forecasts', []):
            # Find matching prior
            prior_match = next(
                (p for p in prior_forecast.get('forecasts', [])
                 if p['region'] == curr['region'] and p['period'] == curr['period']),
                None
            )

            if prior_match:
                revision = curr['production_mbd'] - prior_match['production_mbd']
                # Production revision up = bearish, down = bullish
                if abs(revision) > 0.05:  # > 50 kb/d threshold
                    signals.append({
                        'region': curr['region'],
                        'period': curr['period'],
                        'revision_mbd': revision,
                        'signal': -1 if revision > 0 else 1,
                        'reasoning': f"Production revised {'up' if revision > 0 else 'down'} by {abs(revision):.2f} mb/d"
                    })

        # Aggregate global signal
        if signals:
            total_revision = sum(s['revision_mbd'] for s in signals)
            net_signal = -1 if total_revision > 0.1 else 1 if total_revision < -0.1 else 0

            return {
                'component_signals': signals,
                'net_revision_mbd': total_revision,
                'net_signal': net_signal
            }

        return {'net_signal': 0}
```

</div>

## Sentiment-Based Signals

### Aggregated News Sentiment

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">newssentimentsignal.py</span>
</div>

```python
class NewsSentimentSignal:
    """
    Generate signals from aggregated news sentiment.
    """

    def __init__(
        self,
        lookback_hours: int = 24,
        signal_threshold: float = 0.3,
        min_articles: int = 5
    ):
        self.lookback_hours = lookback_hours
        self.signal_threshold = signal_threshold
        self.min_articles = min_articles
        self.sentiment_cache = []

    def add_sentiment(self, sentiment_result: dict, timestamp: pd.Timestamp):
        """Add new sentiment observation."""
        self.sentiment_cache.append({
            'timestamp': timestamp,
            **sentiment_result
        })

        # Cleanup old entries
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=self.lookback_hours * 2)
        self.sentiment_cache = [
            s for s in self.sentiment_cache
            if s['timestamp'] > cutoff
        ]

    def generate_signal(self) -> dict:
        """
        Generate signal from recent sentiment.
        """
        cutoff = pd.Timestamp.now() - pd.Timedelta(hours=self.lookback_hours)
        recent = [s for s in self.sentiment_cache if s['timestamp'] > cutoff]

        if len(recent) < self.min_articles:
            return {
                'signal': 0,
                'confidence': 0,
                'reason': f'Insufficient articles ({len(recent)} < {self.min_articles})'
            }

        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0

        for s in recent:
            weight = s.get('confidence', 0.5)
            sentiment_value = (
                1 if s.get('sentiment') == 'bullish' else
                -1 if s.get('sentiment') == 'bearish' else 0
            )
            weighted_sentiment += weight * sentiment_value
            total_weight += weight

        net_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0

        # Generate signal
        if net_sentiment > self.signal_threshold:
            signal = 1
            confidence = min(abs(net_sentiment), 1.0)
        elif net_sentiment < -self.signal_threshold:
            signal = -1
            confidence = min(abs(net_sentiment), 1.0)
        else:
            signal = 0
            confidence = 0

        return {
            'signal': signal,
            'confidence': confidence,
            'net_sentiment': net_sentiment,
            'article_count': len(recent),
            'bullish_count': sum(1 for s in recent if s.get('sentiment') == 'bullish'),
            'bearish_count': sum(1 for s in recent if s.get('sentiment') == 'bearish')
        }
```

</div>

## Signal Combination

### Multi-Signal Aggregation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">commoditysignalaggregator.py</span>
</div>

```python
class CommoditySignalAggregator:
    """
    Combine multiple signal sources into unified signal.
    """

    def __init__(self, weights: dict = None):
        self.weights = weights or {
            'inventory': 0.30,
            'production': 0.25,
            'sentiment': 0.20,
            'balance': 0.25
        }
        self.signals = {}

    def update_signal(self, signal_type: str, signal_value: dict):
        """Update individual signal."""
        self.signals[signal_type] = {
            'timestamp': pd.Timestamp.now(),
            **signal_value
        }

    def get_composite_signal(self) -> dict:
        """
        Calculate weighted composite signal.
        """
        if not self.signals:
            return {'signal': 0, 'confidence': 0}

        weighted_signal = 0
        total_weight = 0

        for signal_type, weight in self.weights.items():
            if signal_type in self.signals:
                s = self.signals[signal_type]
                signal_value = s.get('signal', 0)
                confidence = s.get('confidence', 0.5)

                # Weight by both category weight and signal confidence
                effective_weight = weight * confidence
                weighted_signal += effective_weight * signal_value
                total_weight += effective_weight

        if total_weight == 0:
            return {'signal': 0, 'confidence': 0}

        composite = weighted_signal / total_weight

        return {
            'signal': 1 if composite > 0.3 else -1 if composite < -0.3 else 0,
            'raw_score': composite,
            'confidence': min(abs(composite) * 1.5, 1.0),
            'component_signals': {
                k: v.get('signal', 0) for k, v in self.signals.items()
            }
        }
```

</div>

## Position Sizing

### Signal-to-Position Conversion

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">positionsizer.py</span>
</div>

```python
class PositionSizer:
    """
    Convert signals to position sizes with risk management.
    """

    def __init__(
        self,
        max_position: float = 1.0,
        confidence_scale: bool = True,
        volatility_adjust: bool = True
    ):
        self.max_position = max_position
        self.confidence_scale = confidence_scale
        self.volatility_adjust = volatility_adjust

    def calculate_position(
        self,
        signal: dict,
        current_vol: float,
        target_vol: float = 0.15,
        current_position: float = 0
    ) -> dict:
        """
        Calculate target position from signal.
        """
        signal_value = signal.get('signal', 0)
        confidence = signal.get('confidence', 0.5)

        if signal_value == 0:
            return {
                'target_position': 0,
                'position_change': -current_position,
                'reason': 'No signal'
            }

        # Base position
        base_position = signal_value * self.max_position

        # Scale by confidence
        if self.confidence_scale:
            base_position *= confidence

        # Volatility adjustment
        if self.volatility_adjust and current_vol > 0:
            vol_scalar = target_vol / current_vol
            base_position *= min(vol_scalar, 2.0)  # Cap at 2x

        # Round to reasonable size
        target = round(base_position, 2)
        target = max(min(target, self.max_position), -self.max_position)

        return {
            'target_position': target,
            'position_change': target - current_position,
            'confidence': confidence,
            'vol_adjusted': self.volatility_adjust
        }
```

</div>

<div class="callout-insight">

**Insight:** Understanding trading signal generation with llms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Structured signals** - convert LLM outputs to numeric signals with confidence

2. **Multiple sources** - combine fundamental, sentiment, and event signals

3. **Surprise matters** - actual vs. expected drives markets

4. **Weight by confidence** - not all signals are equally reliable

5. **Risk management** - size positions based on confidence and volatility

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="../notebooks/01_signal_generation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_signal_frameworks.md">
  <div class="link-card-title">01 Signal Frameworks</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_confidence_scoring.md">
  <div class="link-card-title">02 Confidence Scoring</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


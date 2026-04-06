# Aggregating Sentiment Signals for Commodities

> **Reading time:** ~6 min | **Module:** Module 3: Sentiment | **Prerequisites:** Modules 0-2

<div class="callout-key">

**Key Concept Summary:** Individual news sentiment scores must be aggregated into actionable market signals. This guide covers techniques for combining multiple sentiment sources into coherent market views.

</div>

## Introduction

Individual news sentiment scores must be aggregated into actionable market signals. This guide covers techniques for combining multiple sentiment sources into coherent market views.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Sentiment Score Normalization

### Handling Different Scales

```python
import numpy as np
import pandas as pd
from scipy import stats

class SentimentNormalizer:
    """Normalize sentiment scores from different sources."""

    def __init__(self):
        self.source_stats = {}

    def fit(self, sentiment_df, source_column='source', score_column='score'):
        """Learn normalization parameters from historical data."""
        for source in sentiment_df[source_column].unique():
            source_data = sentiment_df[sentiment_df[source_column] == source][score_column]
            self.source_stats[source] = {
                'mean': source_data.mean(),
                'std': source_data.std(),
                'min': source_data.min(),
                'max': source_data.max()
            }

    def transform(self, score, source, method='zscore'):
        """Normalize a sentiment score."""
        if source not in self.source_stats:
            return score

        stats = self.source_stats[source]

        if method == 'zscore':
            # Z-score normalization
            return (score - stats['mean']) / stats['std']

        elif method == 'minmax':
            # Min-max to [-1, 1]
            range_val = stats['max'] - stats['min']
            return 2 * (score - stats['min']) / range_val - 1

        elif method == 'robust':
            # Robust scaling using median and IQR
            return (score - stats['mean']) / (1.4826 * stats['std'])

        return score

# Example usage
np.random.seed(42)

# Simulated sentiment from different sources
sentiment_data = pd.DataFrame({
    'source': ['reuters'] * 100 + ['bloomberg'] * 100 + ['twitter'] * 100,
    'score': (
        np.random.normal(0.1, 0.3, 100).tolist() +  # Reuters: slightly positive
        np.random.normal(-0.05, 0.5, 100).tolist() +  # Bloomberg: neutral, noisier
        np.random.normal(0.0, 0.8, 100).tolist()  # Twitter: very noisy
    ),
    'commodity': ['oil'] * 300
})

normalizer = SentimentNormalizer()
normalizer.fit(sentiment_data)

print("Source Statistics:")
for source, stats in normalizer.source_stats.items():
    print(f"  {source}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
```

## Time-Weighted Aggregation

### Exponential Decay

Recent sentiment matters more:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">timeweightedsentiment.py</span>
</div>

```python
class TimeWeightedSentiment:
    """Aggregate sentiment with time decay."""

    def __init__(self, half_life_hours=24):
        self.half_life = half_life_hours

    def calculate_weights(self, timestamps, reference_time):
        """Calculate exponential decay weights."""
        hours_ago = (reference_time - timestamps).dt.total_seconds() / 3600
        decay = np.exp(-np.log(2) * hours_ago / self.half_life)
        return decay / decay.sum()

    def aggregate(self, df, timestamp_col='timestamp', score_col='score',
                  reference_time=None):
        """Compute time-weighted sentiment."""
        if reference_time is None:
            reference_time = df[timestamp_col].max()

        weights = self.calculate_weights(df[timestamp_col], reference_time)

        return {
            'weighted_sentiment': (df[score_col] * weights).sum(),
            'unweighted_sentiment': df[score_col].mean(),
            'n_articles': len(df),
            'time_span_hours': (df[timestamp_col].max() - df[timestamp_col].min()).total_seconds() / 3600
        }

# Example
sentiment_with_time = sentiment_data.copy()
sentiment_with_time['timestamp'] = pd.date_range(
    end=pd.Timestamp.now(),
    periods=len(sentiment_data),
    freq='30min'
)

aggregator = TimeWeightedSentiment(half_life_hours=12)
result = aggregator.aggregate(sentiment_with_time)
print(f"Time-weighted sentiment: {result['weighted_sentiment']:.4f}")
print(f"Unweighted sentiment: {result['unweighted_sentiment']:.4f}")
```

</div>

## Source Quality Weighting

### Reliability-Based Aggregation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">sourceweightedsentiment.py</span>
</div>

```python
class SourceWeightedSentiment:
    """Weight sentiment by source reliability."""

    def __init__(self, source_weights=None):
        self.source_weights = source_weights or {
            'reuters': 1.0,
            'bloomberg': 1.0,
            'wsj': 0.9,
            'ft': 0.9,
            'specialized_commodity': 1.2,
            'social_media': 0.3,
            'unknown': 0.5
        }

    def get_weight(self, source):
        """Get reliability weight for source."""
        source_lower = source.lower()
        for key, weight in self.source_weights.items():
            if key in source_lower:
                return weight
        return self.source_weights.get('unknown', 0.5)

    def aggregate(self, df, source_col='source', score_col='score'):
        """Compute source-weighted sentiment."""
        weights = df[source_col].apply(self.get_weight)
        weighted_sum = (df[score_col] * weights).sum()
        total_weight = weights.sum()

        return {
            'weighted_sentiment': weighted_sum / total_weight if total_weight > 0 else 0,
            'total_weight': total_weight,
            'source_contribution': dict(
                df.groupby(source_col).apply(
                    lambda x: (x[score_col] * x[source_col].apply(self.get_weight)).sum()
                )
            )
        }

source_aggregator = SourceWeightedSentiment()
result = source_aggregator.aggregate(sentiment_data)
print(f"Source-weighted sentiment: {result['weighted_sentiment']:.4f}")
```

</div>

## Multi-Commodity Sentiment Index

### Creating a Composite Index

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">commoditysentimentindex.py</span>
</div>

```python
class CommoditySentimentIndex:
    """Composite sentiment index across commodities."""

    def __init__(self, commodities, weights=None):
        self.commodities = commodities
        self.weights = weights or {c: 1/len(commodities) for c in commodities}

        self.time_aggregator = TimeWeightedSentiment(half_life_hours=24)
        self.source_aggregator = SourceWeightedSentiment()
        self.normalizer = SentimentNormalizer()

    def compute_commodity_sentiment(self, df, commodity):
        """Compute sentiment for a single commodity."""
        commodity_df = df[df['commodity'] == commodity]

        if len(commodity_df) == 0:
            return {'sentiment': 0, 'confidence': 0}

        # Time-weighted
        time_result = self.time_aggregator.aggregate(commodity_df)

        # Source-weighted
        source_result = self.source_aggregator.aggregate(commodity_df)

        # Combine (average of methods)
        combined = (time_result['weighted_sentiment'] +
                    source_result['weighted_sentiment']) / 2

        # Confidence based on article count and time coverage
        confidence = min(1.0, commodity_df.shape[0] / 50) * \
                     min(1.0, time_result['time_span_hours'] / 48)

        return {
            'sentiment': combined,
            'confidence': confidence,
            'n_articles': len(commodity_df),
            'time_weighted': time_result['weighted_sentiment'],
            'source_weighted': source_result['weighted_sentiment']
        }

    def compute_index(self, df):
        """Compute composite sentiment index."""
        results = {}
        index_value = 0

        for commodity in self.commodities:
            results[commodity] = self.compute_commodity_sentiment(df, commodity)
            index_value += self.weights[commodity] * results[commodity]['sentiment']

        results['composite_index'] = index_value
        return results

# Example
commodities = ['oil', 'natural_gas', 'gold', 'copper']

# Generate multi-commodity data
multi_commodity_data = []
for commodity in commodities:
    for _ in range(50):
        multi_commodity_data.append({
            'commodity': commodity,
            'source': np.random.choice(['reuters', 'bloomberg', 'twitter']),
            'score': np.random.normal(0.1 if commodity == 'oil' else 0, 0.4),
            'timestamp': pd.Timestamp.now() - pd.Timedelta(hours=np.random.uniform(0, 48))
        })

multi_df = pd.DataFrame(multi_commodity_data)

index_calculator = CommoditySentimentIndex(commodities)
index_results = index_calculator.compute_index(multi_df)

print("Commodity Sentiment Index:")
print("=" * 60)
for commodity in commodities:
    r = index_results[commodity]
    print(f"{commodity}: {r['sentiment']:.4f} (confidence: {r['confidence']:.2f})")
print(f"\nComposite Index: {index_results['composite_index']:.4f}")
```

</div>

## Sentiment Regime Detection

### Identifying Market Sentiment Regimes

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">sentimentregimedetector.py</span>
</div>

```python
class SentimentRegimeDetector:
    """Detect sentiment regimes using rolling statistics."""

    def __init__(self, lookback_short=5, lookback_long=20, threshold=1.5):
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.threshold = threshold

    def detect_regime(self, sentiment_series):
        """
        Detect sentiment regime:
        - Extreme Bullish: Short MA significantly above long MA
        - Bullish: Short MA above long MA
        - Neutral: Short MA close to long MA
        - Bearish: Short MA below long MA
        - Extreme Bearish: Short MA significantly below long MA
        """
        short_ma = sentiment_series.rolling(self.lookback_short).mean()
        long_ma = sentiment_series.rolling(self.lookback_long).mean()
        long_std = sentiment_series.rolling(self.lookback_long).std()

        z_score = (short_ma - long_ma) / long_std

        regime = pd.Series(index=sentiment_series.index, dtype='object')
        regime[z_score > self.threshold] = 'extreme_bullish'
        regime[(z_score > 0) & (z_score <= self.threshold)] = 'bullish'
        regime[(z_score >= -self.threshold) & (z_score <= 0)] = 'neutral'
        regime[(z_score < 0) & (z_score >= -self.threshold)] = 'bearish'
        regime[z_score < -self.threshold] = 'extreme_bearish'

        return regime, z_score

    def get_regime_statistics(self, sentiment_series, price_returns):
        """Calculate statistics for each regime."""
        regimes, z_scores = self.detect_regime(sentiment_series)

        stats = {}
        for regime in regimes.dropna().unique():
            mask = regimes == regime
            regime_returns = price_returns[mask]

            stats[regime] = {
                'count': mask.sum(),
                'mean_return': regime_returns.mean(),
                'std_return': regime_returns.std(),
                'hit_rate': (regime_returns > 0).mean() if 'bullish' in regime else (regime_returns < 0).mean()
            }

        return stats

# Example with time series
dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
sentiment_ts = pd.Series(
    np.cumsum(np.random.normal(0, 0.1, 100)),
    index=dates
)
price_returns = pd.Series(
    np.random.normal(0.001, 0.02, 100) + 0.1 * sentiment_ts.values,
    index=dates
)

detector = SentimentRegimeDetector()
regimes, z_scores = detector.detect_regime(sentiment_ts)

print("Regime Distribution:")
print(regimes.value_counts())
```

</div>

## Visualization

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">plot_sentiment_dashboard.py</span>
</div>

```python
import matplotlib.pyplot as plt

def plot_sentiment_dashboard(sentiment_df, commodity, lookback=30):
    """Create sentiment analysis dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Filter for commodity
    df = sentiment_df[sentiment_df['commodity'] == commodity].copy()
    df = df.sort_values('timestamp')

    # 1. Sentiment time series
    ax1 = axes[0, 0]
    ax1.plot(df['timestamp'], df['score'], 'o-', alpha=0.5, markersize=3)
    ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Sentiment Score')
    ax1.set_title(f'{commodity.upper()} Sentiment Over Time')

    # 2. Source distribution
    ax2 = axes[0, 1]
    source_sentiment = df.groupby('source')['score'].agg(['mean', 'std', 'count'])
    ax2.barh(source_sentiment.index, source_sentiment['mean'])
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Mean Sentiment')
    ax2.set_title('Sentiment by Source')

    # 3. Sentiment distribution
    ax3 = axes[1, 0]
    ax3.hist(df['score'], bins=30, edgecolor='black', alpha=0.7)
    ax3.axvline(df['score'].mean(), color='red', linestyle='--',
                label=f'Mean: {df["score"].mean():.3f}')
    ax3.set_xlabel('Sentiment Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Sentiment Distribution')
    ax3.legend()

    # 4. Rolling sentiment
    ax4 = axes[1, 1]
    df_daily = df.set_index('timestamp').resample('D')['score'].mean()
    rolling = df_daily.rolling(7).mean()
    ax4.fill_between(df_daily.index, df_daily.values, alpha=0.3, label='Daily')
    ax4.plot(rolling.index, rolling.values, 'r-', linewidth=2, label='7-day MA')
    ax4.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_ylabel('Sentiment')
    ax4.set_title('Daily Sentiment with Moving Average')
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'{commodity}_sentiment_dashboard.png', dpi=150)
    plt.show()

# Create dashboard
plot_sentiment_dashboard(multi_df, 'oil')
```

</div>

<div class="callout-insight">

**Insight:** Understanding aggregating sentiment signals for commodities is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Normalize scores** across sources before aggregating

2. **Time-weight** to emphasize recent sentiment

3. **Source-weight** by reliability and relevance

4. **Detect regimes** to identify sentiment extremes

5. **Visualize** sentiment dynamics for interpretation

6. **Validate** aggregated signals against price movements

---

## Conceptual Practice Questions

1. How do you aggregate article-level sentiments into a tradeable market signal?

2. What decay function would you use for time-weighting sentiment scores and why?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_sentiment_aggregation_slides.md">
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

<a class="link-card" href="./01_news_sentiment.md">
  <div class="link-card-title">01 News Sentiment</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


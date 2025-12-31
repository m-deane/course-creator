# Module 3: Sentiment Analysis for Commodities

## Overview

Extract sentiment signals from news, social media, and analyst reports. Build commodity-specific sentiment models and integrate signals into trading workflows.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Process commodity news feeds with LLMs
2. Extract sentiment and key events
3. Build commodity-specific sentiment indices
4. Evaluate sentiment signal quality

## Contents

### Guides
- `01_news_processing.md` - News feed ingestion and filtering
- `02_sentiment_extraction.md` - LLM-based sentiment analysis
- `03_signal_construction.md` - Building tradeable signals

### Notebooks
- `01_news_sentiment.ipynb` - Energy news sentiment
- `02_earnings_sentiment.ipynb` - Company mentions
- `03_signal_evaluation.ipynb` - Backtesting signals

## Key Concepts

### Sentiment Pipeline

```
News Feed → Filter → LLM Analysis → Sentiment Score → Signal
              ↓           ↓              ↓
         Relevance    Entity         Aggregation
         Check        Extraction     Over Time
```

### Commodity-Specific Sentiment

| Commodity | Key Signals | Sources |
|-----------|-------------|---------|
| Oil | OPEC, geopolitics, inventory | Reuters, Bloomberg |
| Natural Gas | Weather, storage, LNG | EIA, Weather.com |
| Grains | Weather, crop conditions | USDA, DTN |
| Metals | China demand, supply disruptions | Metal Bulletin |

### Sentiment Extraction Prompt

```python
SENTIMENT_PROMPT = """Analyze this commodity news for trading-relevant sentiment.

Article: {article}

Extract:
1. Overall sentiment (-1 to +1)
2. Commodities mentioned
3. Key events or data points
4. Confidence in your assessment (0-1)
5. Time horizon (immediate, short-term, long-term)

Return as JSON."""
```

## Prerequisites

- Module 0-2 completed
- News API access

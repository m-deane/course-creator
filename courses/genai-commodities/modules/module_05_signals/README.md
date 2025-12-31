# Module 5: Signal Generation

## Overview

Convert LLM-powered analysis into actionable trading signals. Build signal pipelines, implement position sizing, and backtest strategies.

**Time Estimate:** 8-10 hours

## Learning Objectives

By completing this module, you will:
1. Design signal generation frameworks
2. Convert analysis to discrete signals
3. Implement confidence-weighted sizing
4. Backtest LLM-generated signals

## Contents

### Guides
- `01_signal_frameworks.md` - Signal design patterns
- `02_confidence_scoring.md` - Calibrating LLM confidence
- `03_backtesting.md` - Testing signal quality

### Notebooks
- `01_signal_generation.ipynb` - Building the signal pipeline
- `02_backtest_analysis.ipynb` - Performance evaluation

## Key Concepts

### Signal Pipeline

```
Data Sources → LLM Analysis → Signal Logic → Position → Execution
     ↓              ↓             ↓            ↓
  Reports      Sentiment      Threshold    Sizing
  News         Fundamentals   Rules        Risk
  Prices       Forecasts                   Limits
```

### Signal Structure

```python
@dataclass
class TradingSignal:
    commodity: str
    direction: Literal["long", "short", "neutral"]
    confidence: float  # 0-1
    rationale: str
    time_horizon: str
    generated_at: datetime
    sources: list[str]
    metadata: dict
```

### Confidence Calibration

| LLM Confidence | Historical Accuracy | Adjusted Weight |
|----------------|---------------------|-----------------|
| 0.9+ | Measure | Scale |
| 0.7-0.9 | Track | Adjust |
| <0.7 | Flag | Reduce |

## Prerequisites

- Module 0-4 completed
- Basic quant finance

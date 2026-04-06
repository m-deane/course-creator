# Supply/Demand Balance Modeling with LLMs

> **Reading time:** ~5 min | **Module:** Module 4: Fundamentals | **Prerequisites:** Modules 0-3

<div class="callout-key">

**Key Concept Summary:** Commodity prices are driven by the supply/demand balance:

</div>

## The Balance Framework

Commodity prices are driven by the supply/demand balance:

$$\text{Balance} = \text{Supply} - \text{Demand}$$
$$\text{Ending Stocks} = \text{Beginning Stocks} + \text{Production} + \text{Imports} - \text{Consumption} - \text{Exports}$$

LLMs help by:
1. Extracting balance components from reports
2. Identifying revisions and surprises
3. Generating balance sheet projections

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Balance Sheet Structure

### Energy Commodities

```python
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class OilBalance:
    """Global oil supply/demand balance."""
    period: str  # Q1 2024, 2024, etc.

    # Supply
    opec_production: float  # mb/d
    non_opec_production: float
    us_production: float
    russia_production: float
    total_supply: float

    # Demand
    oecd_demand: float
    non_oecd_demand: float
    china_demand: float
    total_demand: float

    # Balance
    implied_balance: float  # supply - demand
    stock_change: float  # actual observed
    oecd_stocks: Optional[float] = None
    days_cover: Optional[float] = None

    @property
    def is_in_deficit(self) -> bool:
        return self.implied_balance < 0
```

### Agricultural Commodities

```python
@dataclass
class GrainBalance:
    """USDA-style grain balance sheet."""
    commodity: str
    marketing_year: str
    region: str  # US, World, etc.

    # Supply
    beginning_stocks: float
    production: float
    imports: float
    total_supply: float

    # Demand
    food_seed_industrial: float
    feed_residual: float
    exports: float
    total_demand: float

    # Ending Position
    ending_stocks: float
    stocks_to_use: float  # Key ratio

    @property
    def is_tight(self) -> bool:
        """Stocks-to-use below 10% is generally tight."""
        return self.stocks_to_use < 0.10
```

## Extracting Balance Data with LLMs

### From IEA/EIA Reports

```python
from anthropic import Anthropic

client = Anthropic()

def extract_oil_balance(report_text: str) -> dict:
    """
    Extract oil supply/demand balance from IEA or EIA report.
    """
    prompt = """Extract the oil supply/demand balance from this report.

Return JSON:
{
  "report_source": "IEA|EIA|OPEC",
  "period": "Q1 2024",
  "supply": {
    "opec_crude": <mb/d>,
    "opec_ngls": <mb/d>,
    "non_opec": <mb/d>,
    "us": <mb/d>,
    "russia": <mb/d>,
    "total_supply": <mb/d>
  },
  "demand": {
    "oecd_americas": <mb/d>,
    "oecd_europe": <mb/d>,
    "oecd_asia": <mb/d>,
    "china": <mb/d>,
    "other_non_oecd": <mb/d>,
    "total_demand": <mb/d>
  },
  "balance": {
    "implied_balance": <supply - demand>,
    "call_on_opec": <if mentioned>,
    "stock_change": <if mentioned>
  },
  "revisions": [
    {
      "category": "...",
      "direction": "up|down",
      "magnitude": <kb/d>,
      "reason": "..."
    }
  ]
}

Use null for values not explicitly stated.

Report:
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt + report_text}]
    )

    return response.content[0].text
```

### From USDA WASDE


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">extract_wasde_balance.py</span>
</div>

```python
def extract_wasde_balance(wasde_text: str, commodity: str) -> dict:
    """
    Extract balance sheet from USDA WASDE report.
    """
    prompt = f"""Extract the {commodity} balance sheet from this WASDE report.

Return JSON:
{{
  "commodity": "{commodity}",
  "marketing_year": "2024/25",
  "us_balance": {{
    "beginning_stocks": <million bushels>,
    "production": <million bushels>,
    "imports": <million bushels>,
    "total_supply": <million bushels>,
    "food_seed_industrial": <million bushels>,
    "feed_residual": <million bushels>,
    "exports": <million bushels>,
    "total_use": <million bushels>,
    "ending_stocks": <million bushels>,
    "stocks_to_use_pct": <percentage>
  }},
  "world_balance": {{
    "beginning_stocks": <million metric tons>,
    "production": <mmt>,
    "trade": <mmt>,
    "consumption": <mmt>,
    "ending_stocks": <mmt>,
    "stocks_to_use_pct": <percentage>
  }},
  "revisions_from_last_month": [
    {{
      "item": "...",
      "change": <value>,
      "reason": "..."
    }}
  ],
  "key_changes": "..."
}}

Report:
{wasde_text}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

</div>
</div>

## Building Balance Models

### Time Series of Balances


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">balancetracker.py</span>
</div>

```python
import pandas as pd
import numpy as np

class BalanceTracker:
    """Track and analyze commodity balances over time."""

    def __init__(self, commodity: str):
        self.commodity = commodity
        self.balances = []

    def add_balance(self, balance_data: dict, source: str):
        """Add new balance observation."""
        self.balances.append({
            'timestamp': pd.Timestamp.now(),
            'source': source,
            **balance_data
        })

    def get_balance_series(self) -> pd.DataFrame:
        """Get time series of key balance metrics."""
        df = pd.DataFrame(self.balances)
        df = df.sort_values('timestamp')
        return df

    def calculate_revision_history(self, metric: str) -> pd.DataFrame:
        """Track how a metric has been revised over time."""
        df = self.get_balance_series()

        # Group by period being forecast
        revisions = df.groupby('period')[metric].apply(list)

        return pd.DataFrame({
            'period': revisions.index,
            'initial_estimate': revisions.apply(lambda x: x[0]),
            'final_estimate': revisions.apply(lambda x: x[-1]),
            'total_revision': revisions.apply(lambda x: x[-1] - x[0]),
            'num_revisions': revisions.apply(len)
        })
```

</div>
</div>

### Surprise Analysis


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">analyze_balance_surprise.py</span>
</div>

```python
def analyze_balance_surprise(
    actual: dict,
    consensus: dict,
    commodity: str
) -> dict:
    """
    Analyze surprise vs consensus expectations.
    """
    prompt = f"""Compare actual {commodity} balance data against consensus.

Actual Data:
{actual}

Consensus Expectations:
{consensus}

Analyze:
1. Which components surprised vs expectations?
2. Is the net balance tighter or looser than expected?
3. What is the likely price impact?

Return JSON:
{{
  "surprises": [
    {{
      "component": "...",
      "actual": <value>,
      "expected": <value>,
      "surprise_pct": <percentage>,
      "significance": "high|medium|low"
    }}
  ],
  "net_balance_surprise": "tighter|looser|inline",
  "price_impact": {{
    "direction": "bullish|bearish|neutral",
    "magnitude": "large|moderate|small",
    "reasoning": "..."
  }}
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

## Forecasting with LLMs

### Generating Balance Projections


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">generate_balance_forecast.py</span>
</div>

```python
def generate_balance_forecast(
    historical_data: pd.DataFrame,
    current_conditions: str,
    forecast_horizon: str
) -> dict:
    """
    Generate forward balance forecast using LLM reasoning.
    """
    # Prepare historical context
    history_summary = historical_data.tail(8).to_string()

    prompt = f"""Generate a commodity balance forecast.

Historical Balance Data (last 8 periods):
{history_summary}

Current Market Conditions:
{current_conditions}

Forecast Horizon: {forecast_horizon}

Generate a balance forecast with:
1. Supply projection with key assumptions
2. Demand projection with key assumptions
3. Implied balance and stocks trajectory
4. Key risks to the forecast

Return JSON:
{{
  "forecast_period": "{forecast_horizon}",
  "supply_forecast": {{
    "total": <value>,
    "key_assumptions": ["..."],
    "upside_risks": ["..."],
    "downside_risks": ["..."]
  }},
  "demand_forecast": {{
    "total": <value>,
    "key_assumptions": ["..."],
    "upside_risks": ["..."],
    "downside_risks": ["..."]
  }},
  "balance_forecast": {{
    "implied_balance": <value>,
    "ending_stocks": <value>,
    "stocks_to_use": <percentage>,
    "year_over_year_change": <value>
  }},
  "price_implication": {{
    "direction": "higher|lower|stable",
    "confidence": <0-1>,
    "reasoning": "..."
  }}
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

</div>
</div>

## Integrating Multiple Sources

### Cross-Source Validation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">reconcile_balance_sources.py</span>

```python
def reconcile_balance_sources(
    iea_balance: dict,
    eia_balance: dict,
    opec_balance: dict
) -> dict:
    """
    Reconcile oil balances from different sources.
    """
    prompt = """Compare and reconcile these oil supply/demand balances from different sources.

IEA Data:
{iea_balance}

EIA Data:
{eia_balance}

OPEC Data:
{opec_balance}

Analyze:
1. Where do the sources agree?
2. Where do they differ significantly?
3. What explains the differences?
4. What is the best central estimate?

Return JSON:
{{
  "agreements": ["areas where sources align"],
  "disagreements": [
    {{
      "metric": "...",
      "iea": <value>,
      "eia": <value>,
      "opec": <value>,
      "likely_reason": "..."
    }}
  ],
  "reconciled_balance": {{
    "supply": <best estimate>,
    "demand": <best estimate>,
    "balance": <best estimate>
  }},
  "confidence": <0-1>,
  "notes": "..."
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```


<div class="callout-insight">

**Insight:** Understanding supply/demand balance modeling with llms is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Key Takeaways

1. **Balance = Price** - supply/demand balances are the fundamental driver

2. **LLMs extract structure** - convert narrative reports to usable data

3. **Track revisions** - balance estimates change; track the direction

4. **Surprises move markets** - actual vs. expected matters most

5. **Multiple sources** - reconcile IEA, EIA, OPEC for best estimates

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./01_balance_modeling_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_crude_fundamentals.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_supply_demand.md">
  <div class="link-card-title">01 Supply Demand</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_storage_analysis.md">
  <div class="link-card-title">02 Storage Analysis</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


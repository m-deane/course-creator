# Parsing EIA Petroleum Reports

> **Reading time:** ~5 min | **Module:** Module 1: Report Processing | **Prerequisites:** Module 0 Foundations

<div class="callout-key">

**Key Concept Summary:** The Energy Information Administration (EIA) publishes critical petroleum market data. This guide covers automated extraction from their most important reports.

</div>

## Overview

The Energy Information Administration (EIA) publishes critical petroleum market data. This guide covers automated extraction from their most important reports.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Key EIA Reports

### Weekly Petroleum Status Report (WPSR)

Released every Wednesday at 10:30 AM ET. Contains:
- Crude oil inventories
- Gasoline and distillate stocks
- Refinery utilization
- Product supplied (demand proxy)
- Imports and exports

### Short-Term Energy Outlook (STEO)

Monthly forecast report with:
- Price forecasts
- Production projections
- Demand estimates
- Global supply/demand balances

## EIA API Access

### Getting an API Key

```python
import os
import requests

# Register at: https://www.eia.gov/opendata/register.php
EIA_API_KEY = os.environ.get('EIA_API_KEY')
BASE_URL = "https://api.eia.gov/v2"

def eia_request(endpoint, params=None):
    """Make authenticated request to EIA API v2."""
    params = params or {}
    params['api_key'] = EIA_API_KEY

    response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
    response.raise_for_status()
    return response.json()
```

### Key Series IDs

| Series ID | Description |
|-----------|-------------|
| PET.WCESTUS1.W | Weekly US Crude Oil Stocks |
| PET.WGTSTUS1.W | Weekly US Gasoline Stocks |
| PET.WDISTUS1.W | Weekly US Distillate Stocks |
| PET.WCRFPUS2.W | Weekly US Refinery Utilization |
| PET.WTTNTUS2.W | Weekly US Total Products Supplied |

```python
def get_weekly_inventory(series_id, periods=52):
    """
    Fetch weekly inventory data from EIA.
    """
    endpoint = f"petroleum/sum/sndw/data"
    params = {
        'frequency': 'weekly',
        'data[0]': 'value',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'desc',
        'length': periods
    }

    data = eia_request(endpoint, params)
    return data['response']['data']

# Example: Get crude oil inventory
crude_stocks = get_weekly_inventory('WCESTUS1')
print(f"Latest crude stocks: {crude_stocks[0]['value']} thousand barrels")
```

## LLM-Based Report Parsing

### Extracting from Text Reports

When API data isn't sufficient, parse the text reports:

```python
from anthropic import Anthropic

client = Anthropic()

def parse_wpsr_narrative(report_text):
    """
    Extract structured data from WPSR narrative text.
    """
    prompt = """Extract petroleum inventory data from this EIA report text.

Return JSON with this structure:
{
  "report_date": "YYYY-MM-DD",
  "crude_oil": {
    "stocks_mmb": <total in million barrels>,
    "weekly_change_mmb": <change from last week>,
    "vs_5yr_avg_pct": <percent vs 5-year average>,
    "days_supply": <if mentioned>
  },
  "gasoline": {
    "stocks_mmb": <total>,
    "weekly_change_mmb": <change>,
    "vs_5yr_avg_pct": <percent vs average>
  },
  "distillates": {
    "stocks_mmb": <total>,
    "weekly_change_mmb": <change>,
    "vs_5yr_avg_pct": <percent vs average>
  },
  "refinery_utilization_pct": <if mentioned>,
  "products_supplied_mbd": <million barrels per day if mentioned>
}

Use null for values not explicitly stated.

Report text:
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": prompt + report_text
        }]
    )

    return response.content[0].text
```

### Example Extraction


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
sample_text = """
This Week in Petroleum - November 15, 2024

U.S. commercial crude oil inventories (excluding those in the Strategic
Petroleum Reserve) decreased by 5.2 million barrels from the previous week.
At 430.0 million barrels, U.S. crude oil inventories are about 3% below
the five year average for this time of year.

Total motor gasoline inventories increased by 2.1 million barrels last week
and are about 2% below the five year average for this time of year.

Distillate fuel inventories decreased by 1.4 million barrels last week and
are about 6% below the five year average for this time of year.

Refinery capacity utilization averaged 90.4 percent.

Total products supplied over the last four-week period averaged 20.8 million
barrels a day.
"""

result = parse_wpsr_narrative(sample_text)
print(result)
```

</div>
</div>

### Handling Tables

EIA reports contain data tables. Extract with specific prompts:


<span class="filename">parse_eia_table.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def parse_eia_table(table_text, table_type='inventory'):
    """
    Extract structured data from EIA report tables.
    """
    prompts = {
        'inventory': """Parse this EIA inventory table.
Return JSON array with entries:
[{
  "category": <product category>,
  "current_week": <value>,
  "prior_week": <value>,
  "year_ago": <value>,
  "unit": "thousand_barrels"
}]""",

        'supply': """Parse this EIA supply/demand table.
Return JSON array with entries:
[{
  "category": <supply/demand item>,
  "current_week": <value>,
  "4wk_average": <value>,
  "year_ago": <value>,
  "unit": "thousand_barrels_per_day"
}]"""
    }

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"{prompts[table_type]}\n\nTable:\n{table_text}"
        }]
    )

    return response.content[0].text
```

</div>
</div>

## Building a Complete Pipeline

### End-to-End WPSR Processing


<span class="filename">from.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class WPSRData:
    """Structured WPSR data."""
    report_date: datetime
    crude_stocks_mmb: float
    crude_change_mmb: float
    gasoline_stocks_mmb: float
    gasoline_change_mmb: float
    distillate_stocks_mmb: float
    distillate_change_mmb: float
    refinery_utilization_pct: Optional[float]
    products_supplied_mbd: Optional[float]

class WPSRProcessor:
    """Process EIA Weekly Petroleum Status Reports."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = Anthropic()

    def fetch_latest_api_data(self) -> dict:
        """Fetch latest data from EIA API."""
        series_map = {
            'crude_stocks': 'WCESTUS1',
            'gasoline_stocks': 'WGTSTUS1',
            'distillate_stocks': 'WDISTUS1',
            'refinery_util': 'WCRFPUS2'
        }

        results = {}
        for name, series in series_map.items():
            data = get_weekly_inventory(series, periods=2)
            results[name] = {
                'current': data[0]['value'],
                'prior': data[1]['value'],
                'change': float(data[0]['value']) - float(data[1]['value'])
            }
        return results

    def parse_narrative(self, text: str) -> dict:
        """Parse narrative text with LLM."""
        return json.loads(parse_wpsr_narrative(text))

    def combine_sources(self, api_data: dict, parsed_text: dict) -> WPSRData:
        """
        Combine API and text-parsed data, preferring API for numerical accuracy.
        """
        return WPSRData(
            report_date=datetime.now(),  # Would parse from text
            crude_stocks_mmb=float(api_data['crude_stocks']['current']) / 1000,
            crude_change_mmb=api_data['crude_stocks']['change'] / 1000,
            gasoline_stocks_mmb=float(api_data['gasoline_stocks']['current']) / 1000,
            gasoline_change_mmb=api_data['gasoline_stocks']['change'] / 1000,
            distillate_stocks_mmb=float(api_data['distillate_stocks']['current']) / 1000,
            distillate_change_mmb=api_data['distillate_stocks']['change'] / 1000,
            refinery_utilization_pct=parsed_text.get('refinery_utilization_pct'),
            products_supplied_mbd=parsed_text.get('products_supplied_mbd')
        )

    def generate_summary(self, data: WPSRData) -> str:
        """Generate trading-relevant summary."""
        prompt = f"""Given this petroleum inventory data, generate a brief trading summary:

Crude: {data.crude_stocks_mmb:.1f} MMB ({data.crude_change_mmb:+.1f} vs prior week)
Gasoline: {data.gasoline_stocks_mmb:.1f} MMB ({data.gasoline_change_mmb:+.1f})
Distillate: {data.distillate_stocks_mmb:.1f} MMB ({data.distillate_change_mmb:+.1f})
Refinery Util: {data.refinery_utilization_pct}%

Provide:
1. Overall supply assessment (bullish/bearish/neutral)
2. Key takeaway for crude traders
3. Key takeaway for products traders
Keep it concise (3-4 sentences total)."""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
```

</div>
</div>

## Validation and Quality Control

### Cross-Checking Extractions


<span class="filename">validate_wpsr_extraction.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def validate_wpsr_extraction(extracted: dict, api_data: dict) -> dict:
    """
    Validate LLM extraction against API data.
    """
    issues = []

    # Check crude oil
    if extracted.get('crude_oil', {}).get('stocks_mmb'):
        llm_value = extracted['crude_oil']['stocks_mmb']
        api_value = float(api_data['crude_stocks']['current']) / 1000

        diff_pct = abs(llm_value - api_value) / api_value * 100
        if diff_pct > 1:  # More than 1% difference
            issues.append({
                'field': 'crude_stocks',
                'llm_value': llm_value,
                'api_value': api_value,
                'diff_pct': diff_pct
            })

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }
```

</div>
</div>

### Historical Comparison


<span class="filename">check_reasonable_change.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def check_reasonable_change(current: float, change: float, commodity: str) -> bool:
    """
    Check if weekly change is within reasonable bounds.
    """
    # Historical max weekly changes (million barrels)
    max_changes = {
        'crude': 15,
        'gasoline': 8,
        'distillate': 6
    }

    return abs(change) <= max_changes.get(commodity, 10)
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding parsing eia petroleum reports is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **EIA API is primary source** - Use API for numerical accuracy

2. **LLMs augment API data** - Extract context, comparisons, and unstructured commentary

3. **Validate extractions** - Cross-check LLM outputs against API data

4. **Build pipelines** - Combine multiple sources into structured, validated outputs

5. **Track release timing** - WPSR release at 10:30 ET is market-moving event

---

## Conceptual Practice Questions

1. What key data points should be extracted from weekly EIA petroleum status reports?

2. How would you validate LLM extraction accuracy against known EIA data?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_eia_reports_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_eia_extraction.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_usda_reports.md">
  <div class="link-card-title">02 Usda Reports</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_earnings_transcripts.md">
  <div class="link-card-title">03 Earnings Transcripts</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


# Parsing USDA Agricultural Reports

> **Reading time:** ~10 min | **Module:** Module 1: Report Processing | **Prerequisites:** Module 0 Foundations

<div class="callout-key">

**Key Concept Summary:** The US Department of Agriculture publishes critical supply/demand data for agricultural commodities. This guide covers automated extraction from WASDE reports, crop progress updates, and export sales using LLMs to convert complex tables and narrative text into tradeable data.

</div>

## In Brief

The US Department of Agriculture publishes critical supply/demand data for agricultural commodities. This guide covers automated extraction from WASDE reports, crop progress updates, and export sales using LLMs to convert complex tables and narrative text into tradeable data.

<div class="callout-insight">

**Insight:** USDA reports drive agricultural commodity markets but are published as PDFs with inconsistent table formats. LLMs excel at flexible table parsing and context extraction where traditional parsers fail, enabling automated processing of monthly WASDE updates and weekly crop reports.

</div>
## Intuitive Explanation

Think of USDA reports as market scorecards updated monthly (WASDE) or weekly (Crop Progress). The LLM acts as an experienced analyst who can:

<div class="flow">
<div class="flow-step mint">1. Read complex tables</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Understand context</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Compare to expectati...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Synthesize narrative...</div>
</div>


1. **Read complex tables** - extracting numbers even when formats change
2. **Understand context** - identifying what's important vs. routine
3. **Compare to expectations** - noting surprises that move markets
4. **Synthesize narratives** - combining numbers with explanations

This automation allows traders to react within seconds of report release rather than minutes or hours.

## Formal Definition

**USDA Report Parsing** is the process of extracting structured supply/demand estimates, production forecasts, and inventory data from agricultural reports, transforming narrative text and tabular data into machine-readable formats suitable for trading analysis and model inputs.

## Key USDA Reports

### 1. World Agricultural Supply and Demand Estimates (WASDE)

**Release:** Monthly (typically 12th of month at 12:00 PM ET)
**Market Impact:** Extreme - major price movements

**Contents:**
- US and world supply/demand balances
- Production estimates
- Ending stocks projections
- Price forecasts
- Year-over-year comparisons

**Commodities Covered:**
- Grains: Corn, wheat, soybeans
- Oilseeds: Soybean oil, soybean meal
- Cotton
- Sugar
- Rice

### 2. Crop Progress Reports

**Release:** Weekly (Mondays at 4:00 PM ET during growing season)
**Contents:**
- Planting progress (% complete)
- Crop condition ratings (poor to excellent)
- Harvest progress
- Regional breakdowns

### 3. Export Sales Report

**Release:** Weekly (Thursdays at 8:30 AM ET)
**Contents:**
- Weekly export sales by commodity
- Outstanding sales (commitments not yet shipped)
- Net changes in commitments

## Data Access Methods

### Direct PDF Downloads

```python
import requests
from datetime import datetime

def download_wasde_report(year: int, month: int) -> bytes:
    """
    Download WASDE PDF for specified month.
    """
    # WASDE reports follow predictable URL pattern
    month_str = f"{month:02d}"
    url = f"https://www.usda.gov/oce/commodity/wasde/wasde{month_str}{year}.pdf"

    response = requests.get(url)
    response.raise_for_status()
    return response.content

# Example: Download January 2024 WASDE
pdf_content = download_wasde_report(2024, 1)
with open("wasde_jan_2024.pdf", "wb") as f:
    f.write(pdf_content)
```

### USDA Quick Stats API

```python
import requests
import os

USDA_API_KEY = os.getenv('USDA_QUICKSTATS_KEY')  # Free from: https://quickstats.nass.usda.gov/api

def query_quick_stats(commodity: str, year: int):
    """
    Query USDA Quick Stats for production data.
    """
    params = {
        'key': USDA_API_KEY,
        'commodity_desc': commodity,
        'year': year,
        'statisticcat_desc': 'PRODUCTION',
        'agg_level_desc': 'NATIONAL',
        'format': 'JSON'
    }

    response = requests.get(
        'https://quickstats.nass.usda.gov/api/api_GET/',
        params=params
    )
    response.raise_for_status()
    return response.json()

# Example: Corn production
data = query_quick_stats('CORN', 2024)
```

## LLM-Based WASDE Parsing

### Table Extraction

WASDE tables have complex multi-level headers and footnotes. LLMs handle this better than traditional parsers.

```python
from anthropic import Anthropic
import PyPDF2

client = Anthropic()

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from WASDE PDF."""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def parse_wasde_table(table_text: str, commodity: str) -> dict:
    """
    Parse WASDE supply/demand table for specific commodity.
    """
    prompt = f"""Extract the US supply and demand data for {commodity} from this WASDE table.

Return JSON with this exact structure:
{{
  "commodity": "{commodity}",
  "marketing_year": "YYYY/YY",
  "supply": {{
    "beginning_stocks": <value in million bushels or million pounds>,
    "production": <value>,
    "imports": <value>,
    "total_supply": <value>
  }},
  "demand": {{
    "domestic_use": {{
      "feed_and_residual": <value>,
      "food_seed_industrial": <value>,
      "total_domestic": <value>
    }},
    "exports": <value>,
    "total_demand": <value>
  }},
  "ending_stocks": <value>,
  "stocks_to_use_ratio": <percentage>,
  "average_farm_price": {{
    "value": <dollars per bushel>,
    "range_low": <low end of forecast range>,
    "range_high": <high end of forecast range>
  }}
}}

Important rules:
1. Extract values as numbers without commas
2. Use null for any values not present in the table
3. Include units in a separate "units" field
4. Note if this is a forecast or actual

Table text:
{table_text}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Narrative Summary Extraction

WASDE includes narrative summaries explaining changes.

```python
def extract_wasde_summary(narrative_text: str, commodity: str) -> dict:
    """
    Extract key points from WASDE narrative summary.
    """
    prompt = f"""Analyze this WASDE narrative for {commodity} and extract trading-relevant information.

Return JSON:
{{
  "key_changes": [
    {{
      "metric": "production|exports|ending_stocks|etc",
      "direction": "increased|decreased|unchanged",
      "magnitude": <value change>,
      "previous_estimate": <prior value>,
      "current_estimate": <new value>,
      "explanation": "brief reason for change"
    }}
  ],
  "outlook": {{
    "overall_sentiment": "bullish|bearish|neutral",
    "price_pressure": "upward|downward|neutral",
    "key_factors": ["factor 1", "factor 2", ...],
    "uncertainty_level": "high|medium|low"
  }},
  "market_impact": "major|moderate|minor"
}}

Narrative:
{narrative_text}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

## Crop Progress Report Parsing

### Condition Ratings

```python
def parse_crop_progress(report_text: str) -> dict:
    """
    Extract crop condition and progress from weekly report.
    """
    prompt = """Extract crop condition data from this USDA Crop Progress report.

Return JSON:
{
  "report_date": "YYYY-MM-DD",
  "crops": [
    {
      "commodity": "corn|soybeans|wheat|cotton",
      "state": "US|IL|IA|etc",
      "progress": {
        "planted_pct": <percentage>,
        "emerged_pct": <percentage>,
        "silking_pct": <for corn>,
        "harvested_pct": <percentage>,
        "vs_last_year": <difference in percentage points>,
        "vs_5yr_avg": <difference in percentage points>
      },
      "condition": {
        "very_poor_pct": <percentage>,
        "poor_pct": <percentage>,
        "fair_pct": <percentage>,
        "good_pct": <percentage>,
        "excellent_pct": <percentage>,
        "good_to_excellent_pct": <sum of good + excellent>,
        "vs_last_week": <change in good+excellent>
      }
    }
  ]
}

Report:
""" + report_text

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Weather Commentary

Crop reports include weather commentary critical for yield forecasting.

```python
def extract_weather_impact(report_text: str) -> dict:
    """
    Extract weather-related crop impacts from report narrative.
    """
    prompt = """Analyze the weather commentary in this crop report.

Extract:
{
  "regions_affected": ["state or region"],
  "weather_events": [
    {
      "event_type": "drought|excessive_rain|heat|frost|etc",
      "severity": "severe|moderate|minor",
      "location": "specific states/regions",
      "crop_impact": "description of yield/quality impact",
      "potential_yield_effect": "positive|negative|neutral"
    }
  ],
  "overall_weather_assessment": "favorable|unfavorable|mixed",
  "yield_implications": "brief summary of likely yield impact"
}

Report:
""" + report_text

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

## Export Sales Parsing

```python
def parse_export_sales(report_text: str) -> dict:
    """
    Extract weekly export sales data from USDA report.
    """
    prompt = """Parse this USDA Export Sales report.

Return JSON for each commodity:
{
  "report_week_ending": "YYYY-MM-DD",
  "commodities": [
    {
      "commodity": "corn|soybeans|wheat|soybean_meal|soybean_oil",
      "marketing_year": "YYYY/YY",
      "new_sales": <metric tons>,
      "cumulative_sales": <metric tons>,
      "outstanding_sales": <metric tons>,
      "cumulative_exports": <metric tons>,
      "vs_last_year": {
        "new_sales_pct": <percentage change>,
        "cumulative_pct": <percentage change>
      },
      "top_destinations": [
        {"country": "name", "quantity": <metric tons>}
      ],
      "notable_changes": "any significant shifts in buyer patterns"
    }
  ]
}

Report:
""" + report_text

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

## Complete WASDE Pipeline

```python
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class WASDEData:
    """Structured WASDE output."""
    report_date: str
    commodity: str
    production: float
    production_change: float
    ending_stocks: float
    stocks_change: float
    exports: float
    price_forecast_midpoint: float
    sentiment: str
    key_factors: List[str]

class WASDEProcessor:
    """End-to-end WASDE report processor."""

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)

    def process_report(self, pdf_path: str, commodities: List[str]) -> List[WASDEData]:
        """
        Process WASDE PDF and extract data for specified commodities.
        """
        # Extract text from PDF
        full_text = extract_text_from_pdf(pdf_path)

        results = []
        for commodity in commodities:
            # Find commodity-specific table
            table_text = self._extract_commodity_section(full_text, commodity)

            # Parse with LLM
            table_data = parse_wasde_table(table_text, commodity)
            narrative = self._extract_narrative(full_text, commodity)
            summary_data = extract_wasde_summary(narrative, commodity)

            # Combine and structure
            parsed_table = json.loads(table_data)
            parsed_summary = json.loads(summary_data)

            wasde_data = WASDEData(
                report_date=self._extract_date(full_text),
                commodity=commodity,
                production=parsed_table['supply']['production'],
                production_change=self._calculate_change(parsed_table, 'production'),
                ending_stocks=parsed_table['ending_stocks'],
                stocks_change=self._calculate_change(parsed_table, 'ending_stocks'),
                exports=parsed_table['demand']['exports'],
                price_forecast_midpoint=parsed_table['average_farm_price']['value'],
                sentiment=parsed_summary['outlook']['overall_sentiment'],
                key_factors=parsed_summary['outlook']['key_factors']
            )

            results.append(wasde_data)

        return results

    def _extract_commodity_section(self, text: str, commodity: str) -> str:
        """Find the table section for specific commodity."""
        # Implementation would search for commodity-specific table markers
        # This is simplified - actual implementation would be more robust
        pass

    def _extract_narrative(self, text: str, commodity: str) -> str:
        """Extract narrative section for commodity."""
        pass

    def _extract_date(self, text: str) -> str:
        """Extract report publication date."""
        pass

    def _calculate_change(self, data: dict, metric: str) -> float:
        """Calculate month-over-month change in metric."""
        pass
```

## Validation Strategies

### Cross-Reference with Prior Reports

```python
def validate_wasde_changes(current: WASDEData, previous: WASDEData) -> dict:
    """
    Validate extracted data by checking month-over-month changes.
    """
    issues = []

    # Check for unreasonably large changes
    production_change_pct = abs(
        (current.production - previous.production) / previous.production * 100
    )

    if production_change_pct > 20:  # >20% change is unusual
        issues.append({
            'field': 'production',
            'current': current.production,
            'previous': previous.production,
            'change_pct': production_change_pct,
            'severity': 'high' if production_change_pct > 30 else 'medium'
        })

    return {
        'valid': len(issues) == 0,
        'issues': issues
    }
```

### Multi-Model Consensus


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">consensus_parsing.py</span>
</div>

```python
def consensus_parsing(table_text: str, commodity: str, n_runs: int = 3) -> dict:
    """
    Run extraction multiple times and take consensus/average.
    """
    results = []

    for _ in range(n_runs):
        result = parse_wasde_table(table_text, commodity)
        results.append(json.loads(result))

    # Average numerical values
    consensus = {
        'production': sum(r['supply']['production'] for r in results) / n_runs,
        'ending_stocks': sum(r['ending_stocks'] for r in results) / n_runs,
        'confidence': 'high' if all_agree(results) else 'medium'
    }

    return consensus

def all_agree(results: List[dict], tolerance: float = 0.01) -> bool:
    """Check if all extractions agree within tolerance."""
    # Implementation would check variance across runs
    pass
```

</div>
</div>

## Common Pitfalls

**1. PDF Table Extraction Errors**
- **Issue:** PyPDF2 mangles complex tables with merged cells
- **Solution:** Use multiple PDF libraries (pdfplumber, tabula-py) and combine results

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">robust_table_extract.py</span>
</div>

```python
import pdfplumber

def robust_table_extract(pdf_path: str) -> str:
    """Extract tables using multiple methods."""
    # Try pdfplumber first
    with pdfplumber.open(pdf_path) as pdf:
        tables = []
        for page in pdf.pages:
            page_tables = page.extract_tables()
            tables.extend(page_tables)
    return tables
```

</div>
</div>

**2. Unit Confusion**
- **Issue:** WASDE uses different units (bushels, metric tons, pounds)
- **Solution:** Always extract and validate units

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
UNIT_CONVERSIONS = {
    'corn': {'bushel_to_mt': 0.0254},  # 1 bushel corn = 25.4 kg
    'soybeans': {'bushel_to_mt': 0.027216},  # 1 bushel soybeans = 27.216 kg
    'wheat': {'bushel_to_mt': 0.027216}
}
```

</div>
</div>

**3. Marketing Year Confusion**
- **Issue:** Different commodities have different marketing year calendars
- **Solution:** Track marketing year definitions

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
MARKETING_YEARS = {
    'corn': {'start_month': 9, 'start_day': 1},  # Sept 1 - Aug 31
    'soybeans': {'start_month': 9, 'start_day': 1},  # Sept 1 - Aug 31
    'wheat': {'start_month': 6, 'start_day': 1}  # June 1 - May 31
}
```

</div>
</div>

**4. Revision Handling**
- **Issue:** USDA frequently revises prior estimates
- **Solution:** Track revisions explicitly

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">class.py</span>

```python
@dataclass
class WASDERevision:
    metric: str
    prior_estimate: float
    revised_estimate: float
    revision_magnitude: float
    revision_direction: str  # 'upward' or 'downward'
```


## Connections

**Builds on:**
- Module 0: LLM fundamentals and API setup
- PDF processing libraries
- Data validation techniques

**Leads to:**
- Module 2: RAG systems (creating knowledge bases from historical WASDE reports)
- Module 3: Sentiment analysis (combining USDA data with market commentary)
- Module 4: Fundamentals modeling (using WASDE data in supply/demand models)

**Related to:**
- EIA report parsing (similar table extraction challenges)
- Time series analysis (tracking changes over time)
- Market microstructure (understanding report release reactions)

## Practice Problems

1. **Basic Extraction:**
   Download the most recent WASDE report and extract corn supply/demand data. Verify your extraction against the official USDA summary.

2. **Change Detection:**
   Compare two consecutive WASDE reports and identify which metrics changed by more than 5%. Generate an alert for significant changes.

3. **Multi-Commodity Pipeline:**
   Build a pipeline that processes a WASDE report and extracts data for corn, soybeans, and wheat simultaneously. Store results in a structured database.

4. **Validation System:**
   Create a validation function that checks extracted values against reasonable ranges (e.g., US corn production should be between 10-15 billion bushels typically).

<div class="callout-insight">

**Insight:** Understanding parsing usda agricultural reports is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.




## Further Reading

- **USDA WASDE Archives:** https://www.usda.gov/oce/commodity/wasde/
  - Historical reports back to 1973 for analysis

- **USDA Crop Production Reports:** https://www.nass.usda.gov/Publications/
  - Comprehensive list of all USDA reports and release schedules

- **Marketing Year Definitions:** https://www.ers.usda.gov/topics/crops/
  - Understanding agricultural marketing years

- **pdfplumber Documentation:** https://github.com/jsvine/pdfplumber
  - Advanced PDF table extraction techniques

- **Agricultural Market Analysis:**
  - "The Economics of Agricultural Markets" by R. Bierlen
  - Understanding how USDA data influences prices

- **Prompt Engineering for Tabular Data:**
  - https://www.anthropic.com/research
  - Techniques for improving table extraction accuracy

---

## Conceptual Practice Questions

1. What structural differences between USDA and EIA reports affect prompt design?

2. Design an extraction schema for USDA crop progress reports.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./02_usda_reports_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_eia_extraction.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_eia_reports.md">
  <div class="link-card-title">01 Eia Reports</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_earnings_transcripts.md">
  <div class="link-card-title">03 Earnings Transcripts</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


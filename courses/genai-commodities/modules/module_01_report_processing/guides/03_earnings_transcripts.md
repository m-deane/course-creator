# Processing Earnings Call Transcripts for Commodity Mentions

> **Reading time:** ~10 min | **Module:** Module 1: Report Processing | **Prerequisites:** Module 0 Foundations

<div class="callout-key">

**Key Concept Summary:** Corporate earnings calls contain valuable forward-looking commodity information from producers, refiners, and industrial consumers. LLMs can extract commodity-specific mentions, production guidance, and demand indicators from lengthy transcripts, providing unique insight unavailable in public rep...

</div>

## In Brief

Corporate earnings calls contain valuable forward-looking commodity information from producers, refiners, and industrial consumers. LLMs can extract commodity-specific mentions, production guidance, and demand indicators from lengthy transcripts, providing unique insight unavailable in public reports.

<div class="callout-insight">

**Insight:** Companies directly involved in commodity production or consumption often discuss supply/demand factors weeks or months before they appear in official statistics. An oil major's capex guidance signals future production; a chemical company's margin commentary reveals refining economics. LLMs can extract these signals from unstructured transcripts at scale.

</div>
<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Intuitive Explanation

Imagine earnings calls as real-time market research interviews. The CEO and CFO are revealing:
- **What they're planning** (production guidance)
- **What they're seeing** (demand trends)
- **How they feel** (sentiment)
- **What worries them** (risks)

An LLM acts like a team of analysts listening to dozens of calls simultaneously, extracting the commodity-relevant nuggets and flagging surprises—work that would take humans hours or days per company.

## Formal Definition

**Earnings Transcript Analysis** is the systematic extraction of commodity-relevant information (production guidance, inventory levels, demand indicators, pricing commentary, capex plans) from corporate earnings call transcripts, conference presentations, and investor day materials.

## Why Earnings Calls Matter

### Information Edge

**Traditional Data:** Government reports are lagging indicators (weeks/months old)
**Earnings Calls:** Forward-looking guidance from primary sources

**Example Information Flows:**

| Company Type | Commodity Insights |
|--------------|-------------------|
| Oil Majors (XOM, CVX) | Production guidance, capex, finding costs |
| Refiners (VLO, MPC) | Crack spreads, utilization, inventory strategy |
| Chemical (DOW, LYB) | Feedstock costs, demand trends, margin pressure |
| Agriculture (ADM, BG) | Grain merchandising, export flows, crush margins |
| Miners (FCX, SCCO) | Production forecasts, cost curves, demand signals |

### Key Sections to Analyze

<div class="flow">
<div class="flow-step mint">1. Prepared Remarks</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Guidance</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Q&A</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Industry Commentary</div>
</div>


1. **Prepared Remarks** - Management's scripted commentary
2. **Guidance** - Production/cost/price forecasts
3. **Q&A** - Unscripted responses revealing true sentiment
4. **Industry Commentary** - Market conditions and competitor activity

## Data Sources

### Public Transcript Providers

**Free Options:**
```python
import requests

def get_seeking_alpha_transcript(ticker: str, year: int, quarter: int):
    """
    Seeking Alpha provides free transcripts (with delay).
    Note: May require web scraping; check ToS.
    """
    url = f"https://seekingalpha.com/symbol/{ticker}/earnings/transcripts"
    # Would need scraping implementation
    pass

def get_sec_8k_filings(ticker: str):
    """
    Some companies file transcripts as 8-K exhibits.
    """
    from sec_edgar_downloader import Downloader

    dl = Downloader("Company Name", "email@example.com")
    dl.get("8-K", ticker, limit=10)
```

**Paid Options:**
- **Bloomberg:** Full historical transcripts
- **FactSet:** Transcripts with analytics
- **Refinitiv/AlphaSense:** Search across transcripts
- **Motley Fool Earnings Call Transcripts:** Subscription service

### Example: Basic Scraping

```python
import requests
from bs4 import BeautifulSoup

def fetch_transcript_text(url: str) -> str:
    """
    Fetch transcript from public source.
    (Educational example - respect ToS and rate limits)
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Educational Research)'
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract text (structure varies by site)
    transcript_div = soup.find('div', {'id': 'transcript-content'})
    if transcript_div:
        return transcript_div.get_text(separator='\n', strip=True)

    return soup.get_text()
```

## LLM-Based Extraction

### Commodity Mention Detection

First, identify if transcript contains relevant commodity information.

```python
from anthropic import Anthropic

client = Anthropic()

def detect_commodity_mentions(transcript: str) -> dict:
    """
    Scan transcript for commodity-related content.
    """
    prompt = """Analyze this earnings call transcript for commodity-related information.

Return JSON:
{
  "has_commodity_content": true/false,
  "commodities_mentioned": ["crude_oil", "natural_gas", "copper", etc],
  "relevance_score": 0.0-1.0,
  "key_sections": [
    {"section": "prepared_remarks|qa", "page_or_time": "location", "topic": "what was discussed"}
  ],
  "worth_detailed_analysis": true/false
}

Transcript excerpt (first 2000 words):
""" + transcript[:8000]  # Limit input for cost efficiency

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Production Guidance Extraction

```python
def extract_production_guidance(transcript: str, company_type: str) -> dict:
    """
    Extract production forecasts and targets.
    """
    prompt = f"""Extract production guidance from this {company_type} earnings call.

Return JSON:
{{
  "company": "ticker or name",
  "report_date": "YYYY-MM-DD",
  "production": [
    {{
      "commodity": "crude_oil|natural_gas|copper|gold|etc",
      "period": "Q1 2024|FY 2024|2024|etc",
      "guidance": {{
        "volume": <number>,
        "unit": "bpd|bcf/d|tonnes|etc",
        "range_low": <if provided>,
        "range_high": <if provided>,
        "previous_guidance": <prior estimate if mentioned>,
        "change_vs_previous": <if updated>
      }},
      "confidence_level": "firm|guidance|target",
      "conditions": "any assumptions or contingencies mentioned"
    }}
  ],
  "capex_plans": {{
    "total": <USD millions>,
    "period": "quarter|year",
    "allocation": {{
      "upstream": <USD millions>,
      "downstream": <USD millions>,
      "other": <USD millions>
    }},
    "vs_previous": <change from prior period>
  }},
  "key_changes": ["any guidance revisions or surprises"]
}}

Transcript:
{transcript}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Demand Signal Extraction

For consumers of commodities (chemicals, manufacturers):

```python
def extract_demand_signals(transcript: str, industry: str) -> dict:
    """
    Extract demand indicators from commodity consumer earnings calls.
    """
    prompt = f"""Analyze this {industry} company earnings call for commodity demand signals.

Focus on:
1. Input costs and feedstock prices
2. Volume/utilization trends
3. Inventory levels and strategies
4. Forward demand commentary
5. Geographic demand patterns

Return JSON:
{{
  "company": "name or ticker",
  "report_date": "YYYY-MM-DD",
  "commodity_inputs": [
    {{
      "commodity": "crude_oil|natural_gas|steel|etc",
      "cost_trend": "increasing|decreasing|stable",
      "volume_consumed": {{"current": <value>, "period": "quarter/year", "unit": "unit"}},
      "commentary": "direct quote about this commodity"
    }}
  ],
  "demand_indicators": {{
    "overall_demand_trend": "strong|moderate|weak|declining",
    "capacity_utilization_pct": <percentage if mentioned>,
    "inventory_status": "high|normal|low",
    "forward_orders": "increasing|decreasing|stable",
    "geographic_strength": [
      {{"region": "name", "trend": "strong|weak|mixed"}}
    ]
  }},
  "margin_pressure": {{
    "status": "expanding|compressing|stable",
    "input_cost_impact": "primary driver is...",
    "pricing_power": "can pass through costs: yes/no/partial"
  }},
  "outlook": {{
    "management_tone": "optimistic|cautious|pessimistic",
    "demand_forecast": "summary of forward guidance",
    "risk_factors": ["list key risks mentioned"]
  }}
}}

Transcript:
{transcript}
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

### Sentiment and Tone Analysis

Management tone often reveals more than numbers.

```python
def analyze_management_sentiment(transcript: str) -> dict:
    """
    Assess management sentiment and confidence.
    """
    prompt = """Analyze management's tone and sentiment in this earnings call.

Focus on:
1. Confidence level in guidance
2. Optimism about market conditions
3. Concern about risks or headwinds
4. Changes in language vs. prior quarters

Return JSON:
{
  "overall_sentiment": "very_positive|positive|neutral|negative|very_negative",
  "confidence_level": "high|medium|low",
  "tone_shifts": [
    {
      "topic": "topic area",
      "tone": "description",
      "significance": "why this matters"
    }
  ],
  "telling_phrases": [
    {
      "quote": "exact quote",
      "interpretation": "what this suggests",
      "context": "who said it (CEO/CFO) and when (prepared/QA)"
    }
  ],
  "vs_expectations": "better|worse|in-line",
  "key_risks_highlighted": ["risk 1", "risk 2"],
  "red_flags": ["any concerning statements or evasive answers"]
}

Transcript:
""" + transcript

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text
```

## Sector-Specific Extraction

### Energy Sector (E&P, Integrated)

```python
ENERGY_EXTRACTION_PROMPT = """Extract commodity-relevant data from this energy company earnings call.

Key metrics to find:
- Production volumes (oil, gas, NGLs) in bpd, bcf/d
- Production costs per barrel
- Reserve replacement ratios
- Drilling plans and rig counts
- Hedging positions
- M&A commentary

Return structured JSON with all numerical data and units.

Transcript:
{transcript}
"""
```

### Refining Sector

```python
REFINING_EXTRACTION_PROMPT = """Extract refining metrics from this earnings call.

Key data:
- Throughput volumes (barrels per day)
- Utilization rates (percentage)
- Crack spreads (3-2-1, 5-3-2, etc.)
- Turnaround schedules
- Inventory levels (crude, products)
- Margin commentary by product (gasoline, diesel, jet)

Return structured JSON.

Transcript:
{transcript}
"""
```

### Agriculture (Processors, Traders)

```python
AGRICULTURE_EXTRACTION_PROMPT = """Analyze this agricultural company earnings call.

Extract:
- Grain origination volumes
- Crush margins (if processor)
- Export flows and destinations
- Storage/inventory strategy
- Commodity price hedging
- Forward contract coverage
- Weather impact commentary
- Supply chain commentary

Return structured JSON.

Transcript:
{transcript}
"""
```

## Complete Pipeline


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">from.py</span>
</div>

```python
from dataclasses import dataclass
from typing import List, Optional
import json
from datetime import datetime

@dataclass
class TranscriptInsight:
    """Structured output from transcript analysis."""
    company: str
    ticker: str
    report_date: datetime
    commodities: List[str]
    production_guidance: dict
    demand_signals: dict
    sentiment: str
    key_quotes: List[str]
    trading_implications: str

class TranscriptProcessor:
    """End-to-end earnings transcript processor."""

    def __init__(self, anthropic_api_key: str):
        self.client = Anthropic(api_key=anthropic_api_key)
        self.cache = {}

    def process_transcript(
        self,
        transcript: str,
        company_name: str,
        ticker: str,
        company_type: str
    ) -> TranscriptInsight:
        """
        Full transcript analysis pipeline.
        """
        # Step 1: Detect commodity relevance
        mention_check = json.loads(detect_commodity_mentions(transcript))

        if not mention_check['worth_detailed_analysis']:
            return None  # Skip non-relevant transcripts

        # Step 2: Extract based on company type
        if company_type in ['oil_producer', 'gas_producer', 'integrated_energy']:
            production_data = extract_production_guidance(transcript, company_type)
            demand_data = {}
        elif company_type in ['refiner', 'chemical', 'manufacturer']:
            production_data = {}
            demand_data = extract_demand_signals(transcript, company_type)
        else:
            # Generic extraction
            production_data = extract_production_guidance(transcript, 'generic')
            demand_data = extract_demand_signals(transcript, 'generic')

        # Step 3: Sentiment analysis
        sentiment_data = analyze_management_sentiment(transcript)

        # Step 4: Generate trading summary
        trading_summary = self._generate_trading_summary(
            mention_check,
            production_data,
            demand_data,
            sentiment_data
        )

        # Step 5: Structure output
        insight = TranscriptInsight(
            company=company_name,
            ticker=ticker,
            report_date=datetime.now(),  # Would parse from transcript
            commodities=mention_check['commodities_mentioned'],
            production_guidance=json.loads(production_data) if production_data else {},
            demand_signals=json.loads(demand_data) if demand_data else {},
            sentiment=json.loads(sentiment_data)['overall_sentiment'],
            key_quotes=json.loads(sentiment_data)['telling_phrases'],
            trading_implications=trading_summary
        )

        return insight

    def _generate_trading_summary(
        self,
        mentions: dict,
        production: str,
        demand: str,
        sentiment: str
    ) -> str:
        """
        Synthesize analysis into trading-focused summary.
        """
        prompt = f"""Based on this earnings call analysis, provide a brief trading summary.

Commodity mentions: {mentions}
Production data: {production}
Demand signals: {demand}
Sentiment: {sentiment}

Provide:
1. Key takeaway for commodity traders (2-3 sentences)
2. Bullish or bearish implications
3. Which specific commodities are most affected
4. Confidence level in signal (high/medium/low)

Format as concise bullet points.
"""

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def batch_process(
        self,
        transcripts: List[dict]
    ) -> List[TranscriptInsight]:
        """
        Process multiple transcripts efficiently.
        """
        insights = []

        for t in transcripts:
            try:
                insight = self.process_transcript(
                    t['transcript'],
                    t['company'],
                    t['ticker'],
                    t['company_type']
                )
                if insight:
                    insights.append(insight)
            except Exception as e:
                print(f"Error processing {t['ticker']}: {e}")
                continue

        return insights
```

</div>
</div>

## Common Pitfalls

**1. Transcript Length Exceeds Context Window**
- **Issue:** Full transcripts often exceed LLM context limits
- **Solution:** Chunk intelligently by section (prepared remarks, Q&A)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">chunk_transcript.py</span>
</div>

```python
def chunk_transcript(transcript: str) -> dict:
    """Split transcript into logical sections."""
    sections = {
        'prepared_remarks': extract_prepared_remarks(transcript),
        'qa_session': extract_qa(transcript),
        'forward_looking': extract_guidance_section(transcript)
    }
    return sections
```

</div>
</div>

**2. Jargon and Company-Specific Terms**
- **Issue:** LLMs may misinterpret industry terminology
- **Solution:** Provide glossary in prompt

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
INDUSTRY_GLOSSARY = """
Term definitions:
- bpd: barrels per day
- bcf/d: billion cubic feet per day
- 3-2-1 crack: refining margin (3 barrels crude → 2 barrels gasoline + 1 barrel distillate)
- turnaround: planned refinery maintenance shutdown
"""
```

</div>
</div>

**3. Forward-Looking vs. Historical Confusion**
- **Issue:** Mixing past results with future guidance
- **Solution:** Explicit temporal extraction

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
prompt = """
Distinguish between:
- ACTUAL RESULTS (past quarter/year that just ended)
- GUIDANCE (future periods)
- LONG-TERM TARGETS (multi-year goals)

Tag each data point with timeframe.
"""
```

</div>
</div>

**4. Inconsistent Guidance Formatting**
- **Issue:** Companies express guidance differently (ranges, points, percentages)
- **Solution:** Normalize to standard format

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">normalize_guidance.py</span>

```python
def normalize_guidance(raw_value: str) -> dict:
    """Convert various guidance formats to standard structure."""
    # "100-120 million barrels" → {"low": 100, "high": 120, "unit": "million_barrels"}
    # "~500 bpd" → {"point": 500, "unit": "bpd", "approximate": true}
    pass
```


## Connections

**Builds on:**
- Module 0: LLM fundamentals, prompt engineering
- Module 1: Report parsing skills (PDFs, tables, narrative text)

**Leads to:**
- Module 2: RAG systems (building searchable transcript databases)
- Module 3: Sentiment analysis (combining transcript sentiment with news)
- Module 4: Fundamentals modeling (incorporating company guidance into supply models)

**Related to:**
- NLP sentiment analysis (classical approaches)
- Financial statement analysis (complementary data source)
- Alternative data (transcripts as alt data for commodities)

## Practice Problems

1. **Basic Extraction:**
   Find a recent energy company earnings transcript (Shell, Exxon, ConocoPhillips). Extract their production guidance for crude oil and natural gas.

2. **Comparative Analysis:**
   Analyze transcripts from 3 oil majors from the same quarter. Compare their capex guidance and production outlooks. Identify consensus vs. outliers.

3. **Demand Signal Detection:**
   Process a chemical company transcript (Dow, BASF, LyondellBasell). Extract feedstock cost commentary and demand trends.

4. **Sentiment Tracking:**
   Take a single company's transcripts across 4 quarters. Track how management sentiment about a specific commodity (e.g., natural gas) evolved.

5. **Automated Pipeline:**
   Build a system that processes earnings transcripts for the top 10 energy companies each quarter, storing structured output in a database.

<div class="callout-insight">

**Insight:** Understanding processing earnings call transcripts for commodity mentions is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.


## Further Reading

- **Earnings Call Transcript Analysis in Finance:**
  - "Lazy Prices" by Cohen, Malloy, and Nguyen (2020)
  - Academic research on predictive power of transcript language

- **SEC EDGAR Filings Guide:**
  - https://www.sec.gov/edgar/searchedgar/accessing-edgar-data.htm
  - Official source for 8-K filings containing transcripts

- **Company-Specific Research:**
  - Investor relations pages (most companies post transcripts)
  - Seeking Alpha earnings call transcripts database

- **NLP for Finance:**
  - "Machine Learning for Asset Managers" by Marcos López de Prado
  - Chapter on alternative text data

- **LLM Long-Context Processing:**
  - Anthropic's guide to working with 100k+ token contexts
  - Techniques for maintaining accuracy with long documents

- **Industry Classification:**
  - GICS (Global Industry Classification Standard)
  - Understanding company sector categorization for analysis

---

## Conceptual Practice Questions

1. What makes earnings transcripts particularly challenging for automated extraction?

2. How would you extract forward guidance signals from an energy company earnings call?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---

## Cross-References

<a class="link-card" href="./03_earnings_transcripts_slides.md">
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

<a class="link-card" href="./02_usda_reports.md">
  <div class="link-card-title">02 Usda Reports</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


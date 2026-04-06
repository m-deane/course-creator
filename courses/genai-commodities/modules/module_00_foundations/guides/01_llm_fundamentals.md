# LLM Fundamentals for Commodities

> **Reading time:** ~6 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Large Language Models are neural networks trained to predict text that can be repurposed as general-purpose document understanding engines. For commodity trading, this means a single model can read EIA inventory reports, parse earnings call transcripts, and extract price signals from news — tasks...

</div>

## In Brief

Large Language Models are neural networks trained to predict text that can be repurposed as general-purpose document understanding engines. For commodity trading, this means a single model can read EIA inventory reports, parse earnings call transcripts, and extract price signals from news — tasks that previously required separate, hand-built parsers for each document type.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## Key Insight

<div class="callout-info">

**Info:** In commodity markets, the value is in extracting structured signals from unstructured text at scale — not in conversation. Every EIA report, USDA crop condition update, and analyst note is a structured signal waiting to be unlocked.

</div>
## Introduction

Large Language Models (LLMs) transform how we process unstructured commodity market data. By the end of this guide, you will extract structured commodity intelligence from unstructured market reports using an LLM API.

## Why LLMs for Commodities?

### The Unstructured Data Challenge

Commodity markets generate vast amounts of text:
- Government reports (EIA, USDA, IEA)
- Earnings call transcripts
- News articles and press releases
- Analyst reports
- Social media and forums
- Weather reports and forecasts

Traditional NLP required:
- Custom parsers for each document type
- Extensive labeled training data
- Brittle rule-based systems

LLMs provide:
- Zero-shot extraction capabilities
- Flexible schema adaptation
- Reasoning about context

## Core LLM Capabilities

### 1. Information Extraction

Convert unstructured text to structured data:

```python
from anthropic import Anthropic

client = Anthropic()

eia_text = """
U.S. commercial crude oil inventories decreased by 5.2 million barrels
from the previous week. At 430.0 million barrels, U.S. crude oil inventories
are about 3% below the five year average for this time of year.
"""

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": f"""Extract structured data from this EIA report excerpt:

{eia_text}

Return JSON with fields:
- metric_name
- current_value
- change_value
- change_unit
- comparison_to_average
"""
    }]
)

print(response.content[0].text)
```

### 2. Summarization

Condense lengthy reports:

```python
def summarize_report(report_text, focus_areas=None):
    """
    Summarize commodity report with optional focus areas.
    """
    focus_instruction = ""
    if focus_areas:
        focus_instruction = f"\nFocus particularly on: {', '.join(focus_areas)}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Summarize this commodity market report in 3-5 bullet points.
Include key figures and directional changes.{focus_instruction}

Report:
{report_text}"""
        }]
    )
    return response.content[0].text

# Example
summary = summarize_report(
    long_wasde_report,
    focus_areas=["corn production", "ending stocks", "exports"]
)
```

### 3. Classification and Sentiment

Categorize text by sentiment or topic:

```python
def classify_commodity_sentiment(headline):
    """
    Classify commodity news headline sentiment.
    Returns: bullish, bearish, or neutral with confidence.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": f"""Classify this commodity news headline:

"{headline}"

Return JSON:
{{
  "commodity": "<identified commodity>",
  "sentiment": "bullish|bearish|neutral",
  "confidence": <0-1>,
  "reasoning": "<brief explanation>"
}}"""
        }]
    )
    return response.content[0].text

# Examples
headlines = [
    "OPEC+ agrees to extend production cuts through Q2",
    "Brazil soybean harvest reaches record levels",
    "LNG exports remain steady amid mild winter demand"
]

for h in headlines:
    print(classify_commodity_sentiment(h))
```

### 4. Question Answering

Query documents directly:

```python
def query_report(report_text, question):
    """
    Answer questions about commodity reports.
    """
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""Based on this report, answer the following question.
If the information is not in the report, say "Not found in report."

Report:
{report_text}

Question: {question}"""
        }]
    )
    return response.content[0].text
```

## Prompt Engineering for Commodities

### Structured Output

Always request structured formats:

```python
EXTRACTION_PROMPT = """
Extract supply/demand data from this text.

Return a JSON object with this exact structure:
{
  "commodity": string,
  "region": string,
  "metric": "production" | "consumption" | "imports" | "exports" | "stocks",
  "value": number,
  "unit": string,
  "period": string,
  "year_over_year_change": number | null,
  "source": string
}

Text: {text}
"""
```

### Few-Shot Examples

Provide examples for complex extractions:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
FEW_SHOT_PROMPT = """
Extract price forecasts from analyst commentary.

Example 1:
Input: "We expect WTI to average $75/bbl in Q1 2024, rising to $80 by year-end"
Output: {
  "commodity": "WTI Crude",
  "forecasts": [
    {"period": "Q1 2024", "value": 75, "unit": "USD/bbl"},
    {"period": "Q4 2024", "value": 80, "unit": "USD/bbl"}
  ]
}

Example 2:
Input: "Natural gas prices may test $3.50/MMBtu support before recovering"
Output: {
  "commodity": "Natural Gas",
  "forecasts": [
    {"period": "near-term", "value": 3.50, "unit": "USD/MMBtu", "direction": "support"}
  ]
}

Now extract from:
{text}
"""
```

</div>

### Chain of Thought

For complex reasoning:


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
REASONING_PROMPT = """
Analyze this supply/demand data and determine the likely price impact.

Data:
{data}

Think through this step by step:
1. Identify the key supply factors
2. Identify the key demand factors
3. Calculate the implied balance change
4. Consider seasonal patterns
5. Determine net price impact

Provide your analysis followed by a conclusion with:
- Direction: bullish/bearish/neutral
- Magnitude: strong/moderate/weak
- Timeframe: immediate/near-term/medium-term
"""
```

</div>

## Token Efficiency

### Cost Considerations

API costs accumulate quickly with long documents:

| Model | Input Cost | Output Cost |
|-------|------------|-------------|
| Claude 3.5 Sonnet | $3/1M tokens | $15/1M tokens |
| GPT-4o | $2.50/1M tokens | $10/1M tokens |

### Optimization Strategies

<div class="flow">
<div class="flow-step mint">1. Pre-filter content</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Chunk long documents</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Use smaller models</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Cache responses</div>
</div>


1. **Pre-filter content**: Remove boilerplate before sending
2. **Chunk long documents**: Process in sections
3. **Use smaller models**: For simple tasks
4. **Cache responses**: Avoid duplicate processing


<span class="filename">get_cache_key.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import hashlib
import json
from functools import lru_cache

def get_cache_key(text, prompt_template):
    """Generate cache key for LLM requests."""
    content = f"{prompt_template}:{text}"
    return hashlib.md5(content.encode()).hexdigest()

@lru_cache(maxsize=1000)
def cached_extraction(cache_key, text, prompt_template):
    """Cache LLM extraction results."""
    # Actual LLM call here
    pass
```

</div>

## Validation and Error Handling

### Schema Validation

Always validate LLM outputs:


<span class="filename">inventoryreport.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from pydantic import BaseModel, validator
from typing import Optional, List

class InventoryReport(BaseModel):
    commodity: str
    change_value: float
    change_unit: str
    total_inventory: Optional[float]
    vs_five_year_avg: Optional[float]

    @validator('change_unit')
    def valid_unit(cls, v):
        valid_units = ['million_barrels', 'bcf', 'thousand_tons']
        if v not in valid_units:
            raise ValueError(f'Unit must be one of {valid_units}')
        return v

def extract_with_validation(text):
    """Extract and validate inventory data."""
    raw_response = llm_extract(text)

    try:
        parsed = json.loads(raw_response)
        validated = InventoryReport(**parsed)
        return validated.dict()
    except (json.JSONDecodeError, ValueError) as e:
        # Retry or fallback logic
        return None
```

</div>

### Handling Hallucinations

LLMs may fabricate data. Mitigate with:

1. **Explicit instructions**: "Only extract information explicitly stated"
2. **Source attribution**: "Quote the exact text supporting each value"
3. **Confidence scores**: "Rate your confidence 0-1 for each extraction"
4. **Cross-validation**: Compare multiple model outputs


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
ANTI_HALLUCINATION_PROMPT = """
Extract data from this report. Follow these rules strictly:
1. ONLY include information explicitly stated in the text
2. If a value is not clearly stated, use null
3. For each extracted value, include the exact quote from the source
4. If uncertain, mark confidence as "low"

Text: {text}
"""
```

</div>

<div class="callout-insight">

**Insight:** Understanding llm fundamentals for commodities is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **LLMs excel at unstructured data** - converting text to tradeable signals

2. **Prompt engineering matters** - structured outputs, few-shot examples, and chain-of-thought improve quality

3. **Validate everything** - LLMs can hallucinate; use schemas and verification

4. **Manage costs** - cache results, chunk documents, use appropriate model sizes

5. **Commodity domain knowledge** - incorporate commodity-specific context into prompts

---

## Conceptual Practice Questions

1. What makes LLMs particularly useful for commodity market analysis compared to traditional NLP?

2. Describe three types of commodity documents that LLMs can process and the structured output you would expect from each.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_llm_fundamentals_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_market_data_access.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_prompt_engineering_basics.md">
  <div class="link-card-title">02 Prompt Engineering Basics</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_environment_setup.md">
  <div class="link-card-title">03 Environment Setup</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


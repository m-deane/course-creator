# Prompt Engineering for Commodity Analysis

> **Reading time:** ~5 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Effective prompt engineering is crucial for extracting accurate, actionable insights from LLMs when analyzing commodity markets. This guide covers foundational techniques tailored to financial applications.

</div>

## Introduction

Effective prompt engineering is crucial for extracting accurate, actionable insights from LLMs when analyzing commodity markets. This guide covers foundational techniques tailored to financial applications.

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## The Anatomy of a Good Prompt

### Structure

```
[Context] + [Task] + [Constraints] + [Output Format]
```

### Example for Commodity Analysis

```python
prompt_template = """
You are a senior commodity analyst specializing in {commodity}.

Context:
{market_context}

Task:
Analyze the following news article and extract key information that could
impact {commodity} prices in the {timeframe} term.

Article:
{article_text}

Constraints:
- Focus only on supply/demand factors
- Quantify impacts where possible
- Rate confidence (low/medium/high)

Output Format:
{
    "summary": "One sentence summary",
    "factors": [
        {
            "type": "supply|demand|geopolitical|weather",
            "description": "...",
            "direction": "bullish|bearish|neutral",
            "magnitude": "minor|moderate|major",
            "confidence": "low|medium|high"
        }
    ],
    "price_impact_estimate": "X% to Y%",
    "timeframe": "short|medium|long"
}
"""
```

## Few-Shot Prompting

Providing examples dramatically improves output quality:

```python
def create_few_shot_prompt(article, commodity):
    """Create a few-shot prompt for commodity news analysis."""

    examples = """
Example 1:
Article: "Saudi Arabia announces 1 million barrel per day production cut starting next month"
Analysis: {
    "summary": "Major supply reduction from top producer",
    "factors": [{
        "type": "supply",
        "description": "1M bpd cut from Saudi Arabia",
        "direction": "bullish",
        "magnitude": "major",
        "confidence": "high"
    }],
    "price_impact_estimate": "+5% to +10%",
    "timeframe": "short"
}

Example 2:
Article: "China steel demand shows signs of weakening as property sector struggles"
Analysis: {
    "summary": "Demand concerns from largest consumer",
    "factors": [{
        "type": "demand",
        "description": "China steel demand weakening due to property slowdown",
        "direction": "bearish",
        "magnitude": "moderate",
        "confidence": "medium"
    }],
    "price_impact_estimate": "-3% to -7%",
    "timeframe": "medium"
}

Example 3:
Article: "Severe drought in Brazil threatens soybean harvest"
Analysis: {
    "summary": "Weather risk to major soy producer",
    "factors": [{
        "type": "weather",
        "description": "Drought conditions in Brazil reducing expected soybean yield",
        "direction": "bullish",
        "magnitude": "moderate",
        "confidence": "medium"
    }],
    "price_impact_estimate": "+8% to +15%",
    "timeframe": "short"
}
"""

    prompt = f"""You are an expert commodity analyst.

Based on the examples below, analyze the given article for {commodity} market impact.

{examples}

Now analyze this article:
Article: "{article}"
Analysis:"""

    return prompt

# Example usage
article = "OPEC+ members agree to extend production cuts through Q2, citing weak global demand"
prompt = create_few_shot_prompt(article, "crude oil")
print(prompt[:500] + "...")
```

## Chain-of-Thought Prompting

For complex analysis, guide the model through reasoning steps:

```python
cot_prompt = """
Analyze this commodity market report step by step.

Report:
{report_text}

Step 1: Identify the key data points mentioned
List all specific numbers, dates, and quantities.

Step 2: Categorize by impact type
For each data point, classify as supply, demand, or other factor.

Step 3: Assess market implications
What does each factor mean for prices?

Step 4: Consider interactions
Do any factors amplify or offset each other?

Step 5: Synthesize conclusion
Provide overall market outlook with confidence level.

Begin your analysis:
"""

def analyze_with_cot(report_text, llm_client):
    """Use chain-of-thought for commodity report analysis."""
    formatted_prompt = cot_prompt.format(report_text=report_text)

    response = llm_client.generate(
        prompt=formatted_prompt,
        temperature=0.3,  # Lower temperature for analytical tasks
        max_tokens=2000
    )

    return response
```

## Structured Output Extraction

### JSON Mode


<span class="filename">extract_eia_data.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
json_extraction_prompt = """
Extract the following information from this EIA petroleum report in JSON format.

Report excerpt:
{report_excerpt}

Extract to this exact schema:
{
    "report_date": "YYYY-MM-DD",
    "crude_oil": {
        "production_mbpd": number,
        "production_change_pct": number,
        "imports_mbpd": number,
        "exports_mbpd": number
    },
    "inventory": {
        "crude_stocks_mb": number,
        "stocks_change_mb": number,
        "days_supply": number
    },
    "refinery": {
        "utilization_pct": number,
        "runs_mbpd": number
    },
    "products": {
        "gasoline_stocks_mb": number,
        "distillate_stocks_mb": number
    }
}

Return only valid JSON, no additional text.
"""

def extract_eia_data(report_excerpt, llm_client):
    """Extract structured data from EIA report."""
    import json

    prompt = json_extraction_prompt.format(report_excerpt=report_excerpt)

    response = llm_client.generate(
        prompt=prompt,
        temperature=0.0,  # Zero temperature for extraction
        response_format={"type": "json_object"}
    )

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw": response}
```

</div>
</div>

## Commodity-Specific Prompts

### Oil Market Analysis


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
oil_analysis_prompt = """
You are a petroleum market analyst. Analyze this oil market data.

Current Data:
- WTI Price: ${wti_price}/bbl
- Brent-WTI Spread: ${spread}/bbl
- US Crude Inventory: {inventory} million barrels ({inventory_delta})
- OPEC Production: {opec_production} mbpd
- US Production: {us_production} mbpd

Recent News: {news_summary}

Provide analysis covering:
1. Supply-demand balance assessment
2. Key price drivers for the next 2 weeks
3. Technical levels to watch (support/resistance)
4. Risk factors (upside and downside)
5. Trading recommendation with rationale
"""

### Agricultural Commodities

ag_analysis_prompt = """
You are an agricultural commodities analyst specializing in {crop}.

Current conditions:
- USDA Crop Progress: {crop_progress}
- Weather Outlook: {weather_forecast}
- Export Sales: {export_data}
- Ending Stocks Estimate: {stocks}

Seasonal Context:
- Current growth stage: {growth_stage}
- Days to harvest: {days_to_harvest}
- Historical yields at this stage: {historical_yields}

Analyze:
1. Yield risk assessment (scale 1-10)
2. Demand outlook (domestic + export)
3. Price forecast range for next month
4. Key events/reports to watch
"""
```

</div>
</div>

## Handling Model Limitations

### Uncertainty Acknowledgment


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
uncertainty_aware_prompt = """
Analyze this commodity market scenario. Be explicit about uncertainty.

Data: {data}

In your analysis:
- Distinguish between facts and inferences
- Rate confidence for each conclusion (low/medium/high)
- Identify what additional data would increase confidence
- Provide a range of outcomes, not just point estimates
- Note any assumptions you're making

Format your response as:
FACTS (from the data):
- ...

INFERENCES (with confidence):
- [HIGH] ...
- [MEDIUM] ...
- [LOW] ...

ASSUMPTIONS:
- ...

DATA GAPS:
- ...
"""
```

</div>
</div>

### Hallucination Prevention


<span class="filename">create_grounded_prompt.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def create_grounded_prompt(data, question):
    """Create a prompt that grounds responses in provided data."""

    return f"""
Answer the following question using ONLY the information provided below.
If the data doesn't contain enough information to answer, say "Insufficient data."
Do not make up numbers or facts not present in the data.

DATA:
{data}

QUESTION: {question}

ANSWER (cite specific data points):
"""
```

</div>
</div>

## Evaluation and Testing


<span class="filename">evaluate_prompt_quality.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def evaluate_prompt_quality(prompt, test_cases, llm_client):
    """
    Evaluate prompt performance across test cases.
    """
    results = []

    for test in test_cases:
        response = llm_client.generate(
            prompt=prompt.format(**test['input']),
            temperature=0.0
        )

        # Check against expected output
        score = {
            'accuracy': calculate_accuracy(response, test['expected']),
            'format_correct': check_format(response, test['expected_format']),
            'completeness': check_completeness(response, test['required_fields'])
        }

        results.append({
            'test_id': test['id'],
            'scores': score,
            'response': response
        })

    return results

# Example test case structure
test_cases = [
    {
        'id': 'oil_supply_cut',
        'input': {
            'article': 'OPEC announces 2mbpd production cut',
            'commodity': 'crude oil'
        },
        'expected': {
            'direction': 'bullish',
            'magnitude': 'major',
            'factor_type': 'supply'
        },
        'expected_format': 'json',
        'required_fields': ['summary', 'factors', 'price_impact_estimate']
    }
]
```

</div>
</div>

<div class="callout-insight">

**Insight:** Understanding prompt engineering for commodity analysis is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Key Takeaways

1. **Structure prompts clearly** with context, task, constraints, and output format

2. **Use few-shot examples** to demonstrate expected output quality

3. **Chain-of-thought** helps with complex analytical tasks

4. **Structured outputs** (JSON) enable programmatic processing

5. **Ground responses in data** to prevent hallucinations

6. **Test systematically** across representative cases

---

## Conceptual Practice Questions

1. Why does prompt design matter for extraction accuracy in commodity reports?

2. Design a prompt template that extracts supply-demand balance data from an EIA report.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_prompt_engineering_basics_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_market_data_access.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_llm_fundamentals.md">
  <div class="link-card-title">01 Llm Fundamentals</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_environment_setup.md">
  <div class="link-card-title">03 Environment Setup</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


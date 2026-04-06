# Dataiku Prompt Studios

> **Reading time:** ~5 min | **Module:** 1 — Prompts | **Prerequisites:** Module 0 — LLM Mesh setup

## Overview

Prompt Studios provides a visual interface for designing, testing, and iterating on prompts without writing code.

## Accessing Prompt Studios

1. Open your Dataiku project
2. Navigate to **Lab** > **Prompt Studios**
3. Click **+ New Prompt**

## Prompt Studio Interface

```

┌────────────────────────────────────────────────────────────────┐
│  Prompt Studios                                    [Save] [Run] │
├────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌──────────────────────────────────┐ │
│  │   Settings          │  │   Prompt Editor                  │ │
│  │                     │  │                                  │ │
│  │ Model: Claude 3.5   │  │ You are a commodity analyst.     │ │
│  │ Temperature: 0.3    │  │                                  │ │
│  │ Max tokens: 1000    │  │ Extract from this report:        │ │
│  │                     │  │ {{report_text}}                  │ │
│  │ Variables:          │  │                                  │ │
│  │ - report_text       │  │ Return JSON with:                │ │
│  │ - commodity         │  │ - inventory_change               │ │
│  │                     │  │ - price_impact                   │ │
│  └─────────────────────┘  └──────────────────────────────────┘ │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │   Test Cases                                             │  │
│  │   ┌────────────────────────────────────────────────────┐ │  │
│  │   │ Test 1: EIA Weekly Report                          │ │  │
│  │   │ report_text: "Crude inventories fell by 5.2..."    │ │  │
│  │   │ [Run Test]  Status: ✓ Passed                       │ │  │
│  │   └────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Creating Effective Prompts

### Template Variables

Use double curly braces for variables:

```

Analyze the following {{commodity}} market data:

{{market_data}}

Focus on:
1. Supply changes
2. Demand indicators
3. Price drivers

Format as a {{output_format}} summary.
```

### System vs User Messages

```

# System Message (sets behavior)
You are a senior commodity analyst at a major trading firm.
You analyze market data with precision and objectivity.
Always cite specific numbers from the data provided.

# User Message (task-specific)
Analyze this week's EIA petroleum report:
{{report_text}}

Provide:
- Key inventory changes
- Comparison to expectations
- Trading implications
```

## Prompt Patterns for Commodities

### Extraction Pattern

```

TASK: Extract structured data from commodity reports.

INPUT:
{{report_text}}

EXTRACT THE FOLLOWING (use null if not found):
- Inventory level (number, unit)
- Week-over-week change (number, unit)
- Year-over-year change (percentage)
- Comparison to 5-year average (percentage)
- Key driver mentioned (text)

OUTPUT FORMAT: JSON only, no explanation.
```

### Analysis Pattern

```

CONTEXT: You are analyzing {{commodity}} market fundamentals.

DATA:
{{supply_data}}
{{demand_data}}

ANALYSIS FRAMEWORK:
1. Supply assessment
   - Production trends
   - Import/export flows
   - Inventory levels

2. Demand assessment
   - Consumption patterns
   - Seasonal factors
   - Economic indicators

3. Balance calculation
   - Surplus or deficit
   - Trajectory

4. Price implication
   - Bullish, bearish, or neutral
   - Confidence level
   - Key risks

Provide analysis in structured format.
```

### Comparison Pattern

```

Compare these two reports on {{commodity}}:

REPORT 1 ({{source_1}}):
{{report_1_text}}

REPORT 2 ({{source_2}}):
{{report_2_text}}

COMPARISON:
1. Areas of agreement
2. Areas of disagreement
3. Key differences in methodology or scope
4. Which report appears more reliable and why
5. Synthesized view incorporating both sources

Be specific and cite exact figures from each report.
```

## Testing Prompts

### Creating Test Cases

1. Click **+ Add Test Case**
2. Provide variable values
3. Define expected behavior
4. Run and evaluate

```yaml

# Test case definition
test_name: "EIA Weekly Report - Bullish Draw"
variables:
  report_text: |
    U.S. commercial crude oil inventories decreased by 5.2 million barrels
    from the previous week. At 430.0 million barrels, U.S. crude oil inventories
    are about 3% below the five year average for this time of year.
  commodity: "crude_oil"

expected:
  contains:
    - "-5.2"
    - "million barrels"
    - "below"
  sentiment: "bullish"
```

### Automated Evaluation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def evaluate_prompt_output(output: str, test_case: dict) -> dict:
    """Evaluate prompt output against test case."""
    results = {
        'passed': True,
        'checks': []
    }

    # Check for required content
    for required in test_case.get('expected', {}).get('contains', []):
        present = required.lower() in output.lower()
        results['checks'].append({
            'type': 'contains',
            'value': required,
            'passed': present
        })
        if not present:
            results['passed'] = False

    # Check JSON validity if expected
    if test_case.get('expected', {}).get('valid_json', False):
        try:
            import json
            json.loads(output)
            results['checks'].append({
                'type': 'valid_json',
                'passed': True
            })
        except json.JSONDecodeError:
            results['checks'].append({
                'type': 'valid_json',
                'passed': False
            })
            results['passed'] = False

    return results
```

</div>
</div>

## Version Control

### Saving Versions

1. Click **Save Version**
2. Add version notes
3. Compare versions over time

```

Version History:
├── v1.0 - Initial prompt
├── v1.1 - Added JSON output format
├── v1.2 - Improved extraction accuracy
└── v2.0 - Added comparison to expectations
```

### Comparing Versions


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Compare outputs between prompt versions
def compare_prompt_versions(
    prompt_v1: str,
    prompt_v2: str,
    test_data: str,
    llm_connection: str
) -> dict:
    """Compare outputs from two prompt versions."""
    from dataiku.llm import LLM

    llm = LLM(llm_connection)

    output_v1 = llm.complete(prompt_v1.replace("{{data}}", test_data))
    output_v2 = llm.complete(prompt_v2.replace("{{data}}", test_data))

    return {
        'v1_output': output_v1.text,
        'v2_output': output_v2.text,
        'v1_tokens': output_v1.usage.total_tokens,
        'v2_tokens': output_v2.usage.total_tokens,
        'outputs_match': output_v1.text == output_v2.text
    }
```

</div>
</div>

## Deploying Prompts

### From Prompt Studio to Recipe

1. Click **Deploy to Recipe**
2. Select target recipe type (Python, LLM Recipe)
3. Configure input datasets


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Auto-generated recipe code
import dataiku
from dataiku.llm import LLM

# Input
input_dataset = dataiku.Dataset("commodity_reports")
df = input_dataset.get_dataframe()

# LLM setup
llm = LLM("anthropic-claude")

# Prompt from Prompt Studio
PROMPT_TEMPLATE = """
You are a commodity analyst...
{{report_text}}
"""

# Process each row
results = []
for _, row in df.iterrows():
    prompt = PROMPT_TEMPLATE.replace("{{report_text}}", row['report_text'])
    response = llm.complete(prompt, max_tokens=500)
    results.append({
        'id': row['id'],
        'analysis': response.text
    })

# Output
output_dataset = dataiku.Dataset("analyzed_reports")
output_dataset.write_with_schema(pd.DataFrame(results))
```


### LLM Recipe

Use the visual LLM Recipe for simpler cases:

1. Create new **LLM Recipe**
2. Select input dataset
3. Choose Prompt Studio prompt
4. Map columns to variables
5. Configure output columns

## Key Takeaways

1. **Visual interface** enables prompt iteration without code

2. **Template variables** make prompts reusable across data

3. **Test cases** ensure prompt quality before deployment

4. **Version control** tracks prompt evolution

5. **Direct deployment** moves prompts to production recipes

<div class="callout-key">

<strong>Key Concept:</strong> 5. **Direct deployment** moves prompts to production recipes



## Resources

<a class="link-card" href="../notebooks/01_prompt_creation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

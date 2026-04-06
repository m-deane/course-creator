# Using Template Variables in Prompt Studios

> **Reading time:** ~10 min | **Module:** 1 — Prompts | **Prerequisites:** Module 0 — LLM Mesh setup

## In Brief

Template variables enable dynamic content injection into prompts using `{{variable_name}}` syntax. They transform single-purpose prompts into reusable templates that can process thousands of inputs without code modification.

<div class="callout-insight">

<strong>Key Insight:</strong> The power of production LLM applications comes from separating prompt logic (the template) from prompt data (the variables). This separation enables batch processing, A/B testing, and maintainability—turning prompts from one-off scripts into engineered software components.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Template variables enable dynamic content injection into prompts using `{{variable_name}}` syntax. They transform single-purpose prompts into reusable templates that can process thousands of inputs without code modification.

</div>

## Formal Definition

**Template Variables** are placeholder tokens in prompt text that get replaced with actual values at runtime:
- **Syntax**: `{{variable_name}}` for simple substitution
- **Scope**: Variables defined at Prompt Studio level, populated at execution time
- **Types**: String, number, list, or object (structured data)
- **Processing**: Variable substitution occurs before prompt is sent to LLM
- **Validation**: Type checking and required field enforcement at runtime

## Intuitive Explanation

Template variables work like mail merge in email—you write one template letter with placeholders like `{{customer_name}}` and `{{account_balance}}`, then automatically generate personalized letters for thousands of customers. In Prompt Studios, you write one analysis template with `{{report_text}}` and `{{commodity}}` placeholders, then process thousands of reports with different values.

## Visual Representation

```

┌─────────────────────────────────────────────────────────────┐
│                  Prompt Template                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Analyze this {{commodity}} market report:             │  │
│  │                                                       │  │
│  │ {{report_text}}                                       │  │
│  │                                                       │  │
│  │ Focus on {{metrics}} and compare to {{baseline}}     │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                         ▼ (variable substitution)
┌─────────────────────────────────────────────────────────────┐
│                  Executed Prompt                            │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Analyze this crude oil market report:                │  │
│  │                                                       │  │
│  │ U.S. commercial crude inventories decreased by...    │  │
│  │                                                       │  │
│  │ Focus on inventory levels, production and compare    │  │
│  │ to five-year average                                 │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Code Implementation

### Basic Variable Usage

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
from dataiku import PromptStudio

# Create prompt with variables
studio = PromptStudio("market-analyzer")

# Define the template
template = """Analyze this {{commodity}} market report:

{{report_text}}

Provide analysis of:
- Supply/demand balance
- Price trends
- Key market drivers

Return as JSON with sentiment: {{sentiment_options}}"""

studio.set_user_template(template)

# Define variables
studio.set_variables([
    {
        'name': 'commodity',
        'type': 'string',
        'required': True,
        'description': 'Type of commodity (e.g., crude_oil, natural_gas)'
    },
    {
        'name': 'report_text',
        'type': 'string',
        'required': True,
        'description': 'Full text of market report'
    },
    {
        'name': 'sentiment_options',
        'type': 'string',
        'required': False,
        'default': 'bullish, bearish, neutral',
        'description': 'Valid sentiment values'
    }
])

# Execute with specific values
result = studio.complete(
    variables={
        'commodity': 'crude_oil',
        'report_text': 'U.S. commercial crude oil inventories decreased...',
        'sentiment_options': 'bullish, bearish, neutral'
    }
)

print(result.text)
```

</div>

### Advanced Variable Types

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import json

def setup_advanced_variables(studio: PromptStudio):
    """
    Configure variables with different types and validation.
    """

    variables = [
        # Simple string
        {
            'name': 'commodity',
            'type': 'string',
            'required': True,
            'allowed_values': ['crude_oil', 'natural_gas', 'gold', 'copper']
        },

        # Multi-line text
        {
            'name': 'report_text',
            'type': 'text',
            'required': True,
            'max_length': 10000  # Character limit
        },

        # Numeric
        {
            'name': 'min_confidence',
            'type': 'number',
            'required': False,
            'default': 0.7,
            'min_value': 0.0,
            'max_value': 1.0
        },

        # List of strings
        {
            'name': 'metrics_to_extract',
            'type': 'list',
            'required': False,
            'default': ['inventory', 'production', 'price'],
            'item_type': 'string'
        },

        # Structured object
        {
            'name': 'comparison_baseline',
            'type': 'object',
            'required': False,
            'schema': {
                'period': 'string',  # e.g., "5-year average"
                'values': 'object'   # e.g., {'inventory': 450.0}
            }
        }
    ]

    studio.set_variables(variables)

# Usage with complex variables
result = studio.complete(
    variables={
        'commodity': 'crude_oil',
        'report_text': 'Inventory report...',
        'min_confidence': 0.8,
        'metrics_to_extract': ['inventory', 'production', 'imports'],
        'comparison_baseline': {
            'period': '5-year average',
            'values': {
                'inventory': 450.0,
                'production': 12.5
            }
        }
    }
)
```

</div>

### Conditional Variables

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
def create_conditional_prompt():
    """
    Use variables to enable/disable sections of prompts.
    """

    template = """Analyze this {{commodity}} market report:

{{report_text}}

{{#if include_historical}}
Compare to historical trends:
- 1-month trend
- 3-month trend
- 1-year trend
{{/if}}

{{#if include_forecast}}
Provide short-term forecast:
- Next week outlook
- Key factors to watch
{{/if}}

{{#if detailed_mode}}
Include detailed analysis:
- Supply-side factors (production, imports, inventory)
- Demand-side factors (consumption, exports, refinery utilization)
- Price analysis (current vs historical)
{{/if}}

Return analysis as structured markdown."""

    studio = PromptStudio("flexible-analyzer")
    studio.set_user_template(template)

    studio.set_variables([
        {'name': 'commodity', 'type': 'string', 'required': True},
        {'name': 'report_text', 'type': 'string', 'required': True},
        {'name': 'include_historical', 'type': 'boolean', 'default': False},
        {'name': 'include_forecast', 'type': 'boolean', 'default': False},
        {'name': 'detailed_mode', 'type': 'boolean', 'default': True}
    ])

    return studio

# Use different configurations
studio = create_conditional_prompt()

# Quick analysis (minimal)
quick_result = studio.complete(
    variables={
        'commodity': 'crude_oil',
        'report_text': 'Report...',
        'detailed_mode': False
    }
)

# Full analysis
full_result = studio.complete(
    variables={
        'commodity': 'crude_oil',
        'report_text': 'Report...',
        'include_historical': True,
        'include_forecast': True,
        'detailed_mode': True
    }
)
```

</div>

### Looping Over Lists

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
def create_multi_source_analyzer():
    """
    Use loops to process multiple data sources.
    """

    template = """Synthesize insights from multiple {{commodity}} reports:

{{#each sources}}
## Source: {{this.name}}
Report Date: {{this.date}}

{{this.content}}

---
{{/each}}

Provide a synthesized view that:
1. Identifies consensus views across sources
2. Highlights disagreements or discrepancies
3. Assesses which source appears most reliable
4. Generates unified outlook

Consider {{#each focus_areas}}{{this}}, {{/each}} in your analysis."""

    studio = PromptStudio("multi-source-analyzer")
    studio.set_user_template(template)

    studio.set_variables([
        {'name': 'commodity', 'type': 'string', 'required': True},
        {
            'name': 'sources',
            'type': 'list',
            'required': True,
            'item_schema': {
                'name': 'string',
                'date': 'string',
                'content': 'string'
            }
        },
        {'name': 'focus_areas', 'type': 'list', 'default': []}
    ])

    return studio

# Execute with multiple sources
studio = create_multi_source_analyzer()

result = studio.complete(
    variables={
        'commodity': 'natural_gas',
        'sources': [
            {
                'name': 'EIA Weekly Report',
                'date': '2025-01-15',
                'content': 'Natural gas storage increased...'
            },
            {
                'name': 'IEA Monthly',
                'date': '2025-01-10',
                'content': 'Global LNG supply expanded...'
            },
            {
                'name': 'Industry Analysis',
                'date': '2025-01-12',
                'content': 'Winter demand outlook...'
            }
        ],
        'focus_areas': ['supply', 'demand', 'storage', 'price outlook']
    }
)
```

</div>

### Variable Validation and Preprocessing

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
from typing import Any, Dict

def validate_and_preprocess_variables(
    variables: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Validate and preprocess variables before passing to LLM.

    Args:
        variables: Raw variable values
        schema: Variable schema with validation rules

    Returns:
        Validated and preprocessed variables
    """
    processed = {}

    for var_name, var_schema in schema.items():
        value = variables.get(var_name)

        # Check required
        if var_schema.get('required', False) and value is None:
            raise ValueError(f"Required variable '{var_name}' not provided")

        # Use default if not provided
        if value is None:
            value = var_schema.get('default')

        # Type checking
        expected_type = var_schema.get('type')
        if value is not None and expected_type:
            if expected_type == 'string' and not isinstance(value, str):
                value = str(value)
            elif expected_type == 'number' and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(f"Cannot convert '{var_name}' to number")

        # Allowed values
        if 'allowed_values' in var_schema and value not in var_schema['allowed_values']:
            raise ValueError(
                f"'{value}' not in allowed values for '{var_name}': "
                f"{var_schema['allowed_values']}"
            )

        # Length checks for strings
        if isinstance(value, str) and 'max_length' in var_schema:
            if len(value) > var_schema['max_length']:
                raise ValueError(
                    f"'{var_name}' exceeds max length of {var_schema['max_length']}"
                )

        # Range checks for numbers
        if isinstance(value, (int, float)):
            if 'min_value' in var_schema and value < var_schema['min_value']:
                raise ValueError(f"'{var_name}' below minimum of {var_schema['min_value']}")
            if 'max_value' in var_schema and value > var_schema['max_value']:
                raise ValueError(f"'{var_name}' above maximum of {var_schema['max_value']}")

        # Custom preprocessing
        if 'preprocess' in var_schema:
            value = var_schema['preprocess'](value)

        processed[var_name] = value

    return processed

# Example schema with validation
schema = {
    'commodity': {
        'type': 'string',
        'required': True,
        'allowed_values': ['crude_oil', 'natural_gas', 'gold'],
        'preprocess': lambda x: x.lower().strip()
    },
    'report_text': {
        'type': 'string',
        'required': True,
        'max_length': 10000,
        'preprocess': lambda x: x.strip()
    },
    'confidence_threshold': {
        'type': 'number',
        'required': False,
        'default': 0.7,
        'min_value': 0.0,
        'max_value': 1.0
    }
}

# Validate before execution
try:
    validated = validate_and_preprocess_variables(
        variables={
            'commodity': 'CRUDE_OIL',  # Will be lowercased
            'report_text': '  Report content...  ',  # Will be stripped
            'confidence_threshold': 0.85
        },
        schema=schema
    )
    result = studio.complete(variables=validated)
except ValueError as e:
    print(f"Validation error: {e}")
```

</div>

### Batch Processing with Variables

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

</div>

```python
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

def batch_process_with_variables(
    studio: PromptStudio,
    input_df: pd.DataFrame,
    column_mapping: Dict[str, str],
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Process a dataset with variable substitution.

    Args:
        studio: PromptStudio instance
        input_df: Input DataFrame
        column_mapping: Map DataFrame columns to prompt variables
        max_workers: Parallel execution threads

    Returns:
        DataFrame with results
    """

    def process_row(row):
        """Process single row."""
        # Map DataFrame columns to prompt variables
        variables = {
            var: row[col]
            for var, col in column_mapping.items()
        }

        try:
            result = studio.complete(variables=variables)
            return {
                'status': 'success',
                'output': result.text,
                'tokens': result.usage.total_tokens,
                'cost': result.cost
            }
        except Exception as e:
            return {
                'status': 'error',
                'output': None,
                'error': str(e),
                'tokens': 0,
                'cost': 0
            }

    # Process rows in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_row, [row for _, row in input_df.iterrows()]))

    # Combine with original DataFrame
    result_df = input_df.copy()
    result_df['llm_output'] = [r['output'] for r in results]
    result_df['llm_status'] = [r['status'] for r in results]
    result_df['llm_tokens'] = [r['tokens'] for r in results]
    result_df['llm_cost'] = [r['cost'] for r in results]

    print(f"\nProcessing complete:")
    print(f"Success: {(result_df['llm_status'] == 'success').sum()}")
    print(f"Errors: {(result_df['llm_status'] == 'error').sum()}")
    print(f"Total tokens: {result_df['llm_tokens'].sum()}")
    print(f"Total cost: ${result_df['llm_cost'].sum():.2f}")

    return result_df

# Example usage
input_data = pd.DataFrame({
    'report_id': [1, 2, 3],
    'commodity_type': ['crude_oil', 'natural_gas', 'gold'],
    'report_content': [
        'Crude inventory report...',
        'Natural gas storage report...',
        'Gold demand report...'
    ]
})

results = batch_process_with_variables(
    studio=studio,
    input_df=input_data,
    column_mapping={
        'commodity': 'commodity_type',
        'report_text': 'report_content'
    },
    max_workers=3
)

print(results[['report_id', 'commodity_type', 'llm_status', 'llm_cost']])
```

</div>

## Common Pitfalls

**Pitfall 1: Not Validating Variable Inputs**
- Passing unvalidated user input directly to variables can cause errors or poor outputs
- Always validate type, length, and allowed values before execution
- Use schema-based validation to catch errors early

**Pitfall 2: Overusing Variables**
- Not everything should be a variable—stable instructions should be in the template
- Variables are for data that changes between executions, not configuration
- Example: "Return as JSON" should be in template, not `{{output_format}}`

**Pitfall 3: Poor Variable Naming**
- Generic names like `{{text}}` or `{{data}}` are unclear
- Use descriptive names: `{{customer_feedback}}`, `{{report_text}}`, `{{commodity_type}}`
- Match naming conventions across your organization

**Pitfall 4: Missing Default Values**
- Optional variables without defaults cause errors when not provided
- Always specify sensible defaults for optional variables
- Document what the default behavior means

**Pitfall 5: Ignoring Variable Size**
- Large variable values (long text) can exceed model context limits
- Validate max length and truncate if necessary
- Consider preprocessing to summarize long inputs

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- Prompt Studio basics (Module 1.1)
- LLM Mesh configuration (Module 0)

**Leads to:**
- Testing and iteration strategies (Module 1.3)
- Batch processing workflows (Module 3)
- Production deployment patterns (Module 4)

**Related to:**
- Dataset transformation and ETL
- Data validation patterns
- Parallel processing techniques

## Practice Problems

1. **Variable Type Exploration**
   - Create a prompt with variables of each type: string, number, list, object, boolean
   - Test with valid and invalid inputs to understand validation behavior
   - Document the behavior of each type

2. **Conditional Prompt Design**
   - Design a prompt that has "quick" and "detailed" analysis modes
   - Use boolean variables to toggle sections on/off
   - Measure token usage difference between modes

3. **Multi-Source Synthesis**
   - Create a prompt that accepts a list of data sources
   - Use loops to iterate over sources in the prompt
   - Generate a synthesized view identifying consensus and conflicts

4. **Batch Processing Pipeline**
   - Given a dataset with 100 rows, process them using template variables
   - Implement error handling and retry logic
   - Generate a summary report of costs, success rate, and avg latency

5. **Variable Validation Framework**
   - Build a reusable validation function for prompt variables
   - Support type checking, allowed values, range checks, and custom validators
   - Test with various valid and invalid inputs

## Further Reading

- **Dataiku Documentation**: [Prompt Studio Variables](https://doc.dataiku.com/dss/latest/generative-ai/prompt-variables.html) - Official documentation on variable syntax and features

- **Jinja2 Documentation**: [Template Designer Documentation](https://jinja.palletsprojects.com/templates/) - Dataiku uses similar templating syntax to Jinja2

- **Mustache Template Guide**: [Mustache Manual](https://mustache.github.io/mustache.5.html) - Alternative templating system with similar concepts

- **Blog Post**: "Template Variables Best Practices for LLM Applications" - Patterns from production systems (fictional but representative)

- **Research**: "Prompt Programming for Large Language Models" - Formal treatment of prompt as programmable interfaces (representative of current research direction)


## Resources

<a class="link-card" href="../notebooks/01_prompt_creation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

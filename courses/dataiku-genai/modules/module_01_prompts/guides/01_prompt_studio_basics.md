# Getting Started with Prompt Studios

## In Brief

Prompt Studios is Dataiku's visual interface for designing, testing, and versioning prompts without writing code. It provides a structured environment for prompt development with built-in testing, variable templating, and one-click deployment to production recipes.

## Key Insight

The fastest way to iterate on prompts is through visual testing with immediate feedback. Prompt Studios eliminates the code-test-debug cycle by providing inline execution, variable substitution, and version comparison—reducing prompt development time from hours to minutes.

## Formal Definition

**Prompt Studio** is a visual development environment for LLM prompts that provides:
- **Visual Editor**: Structured input for system prompts, user prompts, and examples
- **Variable Templating**: Dynamic content injection using `{{variable}}` syntax
- **Test Harness**: Multiple test cases with expected outputs
- **Version Control**: Built-in prompt versioning with diff comparison
- **Deployment Integration**: Direct export to Python recipes and LLM recipes

## Intuitive Explanation

Think of Prompt Studios like a specialized IDE for prompt engineering. Just as you wouldn't write complex code in a plain text editor, you shouldn't develop production prompts in a notebook cell. Prompt Studios provides the tooling (syntax highlighting for variables, test runners, version control) that makes prompt engineering a systematic discipline rather than ad-hoc experimentation.

## Visual Representation

```
┌───────────────────────────────────────────────────────────────┐
│  Prompt Studio: Market Report Analyzer               [⚙ Save] │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─── Configuration ────────┐  ┌─── Prompt Editor ─────────┐ │
│  │                           │  │                           │ │
│  │ LLM: anthropic-claude     │  │ System Prompt:            │ │
│  │ Model: claude-sonnet-4    │  │ You are a commodity       │ │
│  │ Temperature: 0.3          │  │ market analyst...         │ │
│  │ Max tokens: 1500          │  │                           │ │
│  │                           │  │ User Prompt:              │ │
│  │ Variables:                │  │ Analyze this report:      │ │
│  │ • report_text (string)    │  │ {{report_text}}           │ │
│  │ • commodity (string)      │  │                           │ │
│  │                           │  │ Focus on {{commodity}}    │ │
│  └───────────────────────────┘  └───────────────────────────┘ │
│                                                               │
│  ┌─── Test Cases ───────────────────────────────────────────┐ │
│  │ Test 1: EIA Crude Report                     [▶ Run Test]│ │
│  │ • report_text: "U.S. commercial crude..."                │ │
│  │ • commodity: "crude_oil"                                 │ │
│  │ Status: ✓ Passed (1.2s, 847 tokens)                      │ │
│  │                                                          │ │
│  │ Test 2: IEA Natural Gas Report              [▶ Run Test]│ │
│  │ • report_text: "Global natural gas supply..."           │ │
│  │ • commodity: "natural_gas"                              │ │
│  │ Status: ✗ Failed - See output                           │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌─── Output Preview ───────────────────────────────────────┐ │
│  │ ## Market Analysis                                       │ │
│  │ Crude oil inventories decreased by 5.2 million barrels...│ │
│  └──────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```

## Code Implementation

### Accessing Prompt Studios Programmatically

```python
import dataiku
from dataiku import PromptStudio

# Connect to existing Prompt Studio
studio = PromptStudio("market-report-analyzer")

# Get prompt configuration
config = studio.get_config()
print(f"System Prompt: {config['system_prompt']}")
print(f"User Template: {config['user_template']}")
print(f"Variables: {config['variables']}")

# Run a test case
test_result = studio.run_test(
    test_name="EIA Crude Report",
    variables={
        'report_text': "U.S. commercial crude oil inventories...",
        'commodity': 'crude_oil'
    }
)

print(f"Status: {test_result['status']}")
print(f"Output: {test_result['output']}")
print(f"Tokens: {test_result['usage']['total_tokens']}")
print(f"Cost: ${test_result['cost']:.4f}")
```

### Creating Prompts in Code (Alternative to GUI)

```python
def create_prompt_studio(
    name: str,
    llm_connection: str,
    system_prompt: str,
    user_template: str,
    variables: list[str],
    temperature: float = 0.7
) -> PromptStudio:
    """
    Create a new Prompt Studio programmatically.

    Args:
        name: Unique identifier for the prompt
        llm_connection: LLM Mesh connection name
        system_prompt: System message defining behavior
        user_template: User prompt with {{variables}}
        variables: List of variable names
        temperature: LLM temperature setting

    Returns:
        PromptStudio instance
    """
    client = dataiku.api_client()
    project = client.get_default_project()

    # Create prompt studio
    studio = project.create_prompt_studio(name)

    # Configure
    studio.set_llm(llm_connection)
    studio.set_system_prompt(system_prompt)
    studio.set_user_template(user_template)
    studio.set_variables(variables)
    studio.set_parameters({
        'temperature': temperature,
        'max_tokens': 1500
    })

    studio.save()

    return studio

# Example: Create a commodity report analyzer
analyzer_studio = create_prompt_studio(
    name='commodity-report-analyzer',
    llm_connection='anthropic-claude',
    system_prompt='''You are a commodity market analyst with expertise in oil, gas, and metals markets.

Your role:
- Extract key data points from market reports
- Identify supply and demand factors
- Assess price implications
- Provide objective, data-driven analysis

Always cite specific numbers from the source material.''',
    user_template='''Analyze this {{commodity}} market report:

{{report_text}}

Provide:
1. Key metrics (inventory levels, production, demand)
2. Week-over-week changes
3. Comparison to historical averages
4. Price implications (bullish/bearish/neutral)
5. Key risk factors

Format as structured markdown.''',
    variables=['commodity', 'report_text'],
    temperature=0.3  # Low temperature for consistent extraction
)
```

### Adding Test Cases

```python
def add_test_case(
    studio: PromptStudio,
    test_name: str,
    variable_values: dict,
    expected_output: dict = None
) -> dict:
    """
    Add a test case to Prompt Studio.

    Args:
        studio: PromptStudio instance
        test_name: Descriptive name for test
        variable_values: Values for each variable
        expected_output: Optional validation criteria

    Returns:
        Test case configuration
    """
    test_case = {
        'name': test_name,
        'variables': variable_values,
        'expected': expected_output or {}
    }

    studio.add_test_case(test_case)
    studio.save()

    return test_case

# Add test cases
eia_test = add_test_case(
    studio=analyzer_studio,
    test_name='EIA Weekly Petroleum Report',
    variable_values={
        'commodity': 'crude_oil',
        'report_text': '''U.S. commercial crude oil inventories decreased by 5.2 million barrels
from the previous week. At 430.0 million barrels, U.S. crude oil inventories are about
3% below the five year average for this time of year. Total motor gasoline inventories
increased by 2.1 million barrels and are about 2% below the five year average.'''
    },
    expected_output={
        'contains': ['-5.2 million', 'below', '430.0'],
        'sentiment': 'bullish'
    }
)

iea_test = add_test_case(
    studio=analyzer_studio,
    test_name='IEA Natural Gas Monthly',
    variable_values={
        'commodity': 'natural_gas',
        'report_text': '''Global natural gas supply increased by 2.3% year-over-year driven by
LNG expansion in the United States and Qatar. European storage levels reached 85% of capacity,
well above the five-year average of 72% for this time of year.'''
    },
    expected_output={
        'contains': ['2.3%', 'LNG', '85%'],
        'sentiment': 'neutral'
    }
)
```

### Running All Tests

```python
def run_all_tests(studio: PromptStudio) -> pd.DataFrame:
    """
    Run all test cases and generate report.

    Args:
        studio: PromptStudio instance

    Returns:
        DataFrame with test results
    """
    import pandas as pd

    results = []

    for test_case in studio.get_test_cases():
        result = studio.run_test(test_case['name'])

        results.append({
            'test_name': test_case['name'],
            'status': result['status'],
            'duration_sec': result['duration'],
            'total_tokens': result['usage']['total_tokens'],
            'cost_usd': result['cost'],
            'passed_checks': result.get('passed_checks', 0),
            'failed_checks': result.get('failed_checks', 0)
        })

    df = pd.DataFrame(results)

    # Summary
    print(f"Tests run: {len(df)}")
    print(f"Passed: {(df['status'] == 'passed').sum()}")
    print(f"Failed: {(df['status'] == 'failed').sum()}")
    print(f"Total cost: ${df['cost_usd'].sum():.4f}")
    print(f"Avg tokens: {df['total_tokens'].mean():.0f}")

    return df

# Run all tests
test_results = run_all_tests(analyzer_studio)
print(test_results)
```

### Version Management

```python
def create_version(
    studio: PromptStudio,
    version_name: str,
    notes: str
) -> dict:
    """
    Save a version of the current prompt.

    Args:
        studio: PromptStudio instance
        version_name: Version identifier (e.g., "v1.2.0")
        notes: Description of changes

    Returns:
        Version metadata
    """
    version = studio.create_version(
        name=version_name,
        notes=notes
    )

    return {
        'version': version_name,
        'timestamp': version['created_at'],
        'author': version['author'],
        'notes': notes
    }

# Save version after improvements
v1_1 = create_version(
    studio=analyzer_studio,
    version_name='v1.1',
    notes='Added structured output format and citation requirement'
)

# Compare versions
def compare_versions(
    studio: PromptStudio,
    version_a: str,
    version_b: str,
    test_case_name: str
) -> dict:
    """
    Compare output from two prompt versions.

    Args:
        studio: PromptStudio instance
        version_a: First version to compare
        version_b: Second version to compare
        test_case_name: Test case to run

    Returns:
        Comparison results
    """
    # Run test on version A
    studio.load_version(version_a)
    result_a = studio.run_test(test_case_name)

    # Run test on version B
    studio.load_version(version_b)
    result_b = studio.run_test(test_case_name)

    return {
        'version_a': {
            'version': version_a,
            'output': result_a['output'],
            'tokens': result_a['usage']['total_tokens'],
            'cost': result_a['cost']
        },
        'version_b': {
            'version': version_b,
            'output': result_b['output'],
            'tokens': result_b['usage']['total_tokens'],
            'cost': result_b['cost']
        },
        'token_diff': result_b['usage']['total_tokens'] - result_a['usage']['total_tokens'],
        'cost_diff': result_b['cost'] - result_a['cost']
    }

# Compare versions
comparison = compare_versions(
    studio=analyzer_studio,
    version_a='v1.0',
    version_b='v1.1',
    test_case_name='EIA Weekly Petroleum Report'
)

print(f"Token difference: {comparison['token_diff']}")
print(f"Cost difference: ${comparison['cost_diff']:.4f}")
```

### Deploying to Production

```python
def deploy_to_recipe(
    studio: PromptStudio,
    recipe_name: str,
    input_dataset: str,
    output_dataset: str,
    column_mapping: dict
) -> dict:
    """
    Deploy Prompt Studio to a Python recipe.

    Args:
        studio: PromptStudio instance
        recipe_name: Name for new recipe
        input_dataset: Input dataset name
        output_dataset: Output dataset name
        column_mapping: Map dataset columns to prompt variables

    Returns:
        Recipe configuration
    """
    project = studio.get_project()

    # Create recipe
    recipe = project.create_python_recipe(
        name=recipe_name,
        input_datasets=[input_dataset],
        output_datasets=[output_dataset]
    )

    # Generate recipe code from prompt studio
    recipe_code = studio.generate_recipe_code(
        column_mapping=column_mapping,
        batch_size=10  # Process 10 rows at a time
    )

    recipe.set_code(recipe_code)
    recipe.save()

    return {
        'recipe_name': recipe_name,
        'input': input_dataset,
        'output': output_dataset,
        'mapping': column_mapping
    }

# Deploy to production
deployment = deploy_to_recipe(
    studio=analyzer_studio,
    recipe_name='analyze_market_reports',
    input_dataset='raw_market_reports',
    output_dataset='analyzed_reports',
    column_mapping={
        'report_text': 'report_content',  # dataset column -> prompt variable
        'commodity': 'commodity_type'
    }
)

print(f"Deployed to recipe: {deployment['recipe_name']}")
```

## Common Pitfalls

**Pitfall 1: Not Using Test Cases**
- Iterating on prompts without systematic testing leads to regressions
- Every prompt should have at least 3-5 test cases covering typical scenarios
- Run all tests before saving versions or deploying

**Pitfall 2: Generic System Prompts**
- "You are a helpful assistant" provides no real guidance
- System prompts should specify role, expertise, output format, and constraints
- Be specific: "You are a petroleum market analyst specializing in supply-demand analysis..."

**Pitfall 3: Not Managing Variables**
- Hardcoding values in prompts instead of using variables makes them inflexible
- Use `{{variables}}` for any content that changes between invocations
- Variables enable reusability and testing

**Pitfall 4: Ignoring Token Counts**
- Not monitoring token usage during development leads to cost surprises in production
- Test cases should track tokens and cost per execution
- Optimize prompts that consistently exceed token budgets

**Pitfall 5: Skipping Version Control**
- Making changes without saving versions makes rollback impossible
- Save a version before any significant prompt modification
- Include meaningful version notes explaining what changed and why

## Connections

**Builds on:**
- LLM Mesh setup and connections (Module 0)
- Basic prompt engineering principles

**Leads to:**
- Template variables and dynamic content (Module 1.2)
- Testing and iteration strategies (Module 1.3)
- Production deployment patterns (Module 4)

**Related to:**
- Python recipe development
- Dataset transformation workflows
- Version control best practices

## Practice Problems

1. **Basic Prompt Studio**
   - Create a Prompt Studio for sentiment analysis of customer feedback
   - Define 2-3 variables (feedback text, product category)
   - Add 5 test cases with different sentiment types
   - Track token usage across test cases

2. **Version Comparison**
   - Create version 1.0 with a basic prompt
   - Iterate and improve to create version 1.1 (add examples, improve format)
   - Create version 1.2 (optimize for token efficiency)
   - Compare outputs and costs across all versions on the same test cases

3. **Production Deployment**
   - Build a prompt that extracts structured data from unstructured text
   - Test with 10+ examples ensuring consistent JSON output
   - Deploy to a Python recipe processing a dataset
   - Verify the recipe runs successfully on real data

4. **Cost Optimization**
   - Given a prompt that costs $0.05 per execution
   - Optimize it to reduce cost by 40% while maintaining quality
   - Use shorter prompts, better examples, or model selection
   - Demonstrate savings with before/after test results

5. **Multi-Variable Template**
   - Create a report generator that accepts: data_source, time_period, metric_type, comparison_basis
   - Design the prompt to generate different report types based on variable combinations
   - Test with at least 8 different variable combinations
   - Ensure consistent output format across all combinations

## Further Reading

- **Dataiku Documentation**: [Prompt Studios User Guide](https://doc.dataiku.com/dss/latest/generative-ai/prompt-studios.html) - Official feature documentation with examples

- **Anthropic Guide**: [Prompt Engineering Interactive Tutorial](https://docs.anthropic.com/claude/docs/prompt-engineering) - Best practices from Claude's creators

- **OpenAI Cookbook**: [Prompt Engineering Techniques](https://cookbook.openai.com/articles/techniques_to_improve_reliability) - Practical patterns for reliable prompts

- **Research Paper**: "Large Language Models Are Human-Level Prompt Engineers" (Zhou et al., 2023) - Automatic prompt optimization techniques

- **Blog Post**: "From Prototype to Production: Prompt Development Workflow" - Industry patterns for systematic prompt engineering (fictional but representative)

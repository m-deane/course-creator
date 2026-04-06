# Testing and Iterating on Prompts

> **Reading time:** ~11 min | **Module:** 1 — Prompts | **Prerequisites:** Module 0 — LLM Mesh setup

## In Brief

Systematic prompt testing and iteration transforms prompt engineering from trial-and-error into an engineering discipline. By measuring outputs against success criteria, tracking metrics across versions, and applying structured improvement methods, you can reliably build prompts that perform well in production.

<div class="callout-insight">

<strong>Key Insight:</strong> The best prompt is rarely the first prompt. Systematic iteration—test, measure, hypothesize improvement, implement, retest—yields 2-3x better results than ad-hoc experimentation. The key is defining clear success criteria upfront and measuring every change against them.

</div>

<div class="callout-key">

<strong>Key Concept:</strong> Systematic prompt testing and iteration transforms prompt engineering from trial-and-error into an engineering discipline. By measuring outputs against success criteria, tracking metrics across versions, and applying structured improvement methods, you can reliably build prompts that perform well...

</div>

## Formal Definition

**Prompt Testing and Iteration** is a systematic methodology comprising:
- **Test Cases**: Representative inputs with expected outputs or success criteria
- **Metrics**: Quantitative measures of prompt quality (accuracy, consistency, cost)
- **Baseline**: Initial prompt version establishing performance benchmark
- **Hypotheses**: Specific proposed changes with predicted improvements
- **A/B Comparison**: Side-by-side evaluation of prompt versions
- **Regression Testing**: Ensuring improvements don't break existing functionality

## Intuitive Explanation

Think of prompt iteration like recipe development. You don't create the perfect recipe on your first try. You start with a baseline (first attempt), taste it (test), identify what's wrong (measure against criteria), hypothesize an improvement (more salt? longer cooking?), make the change (iterate), and taste again (retest). After several iterations, comparing against your baseline, you converge on an excellent recipe. The same structured process applies to prompts.

## Visual Representation

```

┌─────────────────────────────────────────────────────────────┐
│           Prompt Iteration Workflow                         │
└─────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │ Define       │
  │ Success      │  What does "good" look like?
  │ Criteria     │  - Accuracy, format, tone, cost
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Create       │
  │ Test Cases   │  5-10 representative examples
  │              │  with expected outputs
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ Baseline     │
  │ v1.0         │  Initial prompt attempt
  │              │  Measure performance
  └──────┬───────┘
         │
         ▼
  ┌──────────────────────────────────────────────┐
  │        Iteration Loop                        │
  │  ┌──────────┐    ┌───────────┐              │
  │  │ Identify │───>│Hypothesize│              │
  │  │ Issues   │    │Improvement│              │
  │  └──────────┘    └─────┬─────┘              │
  │                        │                     │
  │                        ▼                     │
  │  ┌──────────┐    ┌───────────┐              │
  │  │ Compare  │<───│Implement  │              │
  │  │ Results  │    │ Change    │              │
  │  └────┬─────┘    └───────────┘              │
  │       │                                      │
  │       ▼                                      │
  │  Better? ──Yes─> New Baseline ──┐           │
  │     │                            │           │
  │     No                           │           │
  │     │                            │           │
  │     └────────────────────────────┘           │
  └──────────────────────────────────────────────┘
         │
         ▼
  ┌──────────────┐
  │ Production   │
  │ Deployment   │  Continuous monitoring
  └──────────────┘
```

## Code Implementation

### Defining Success Criteria


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from typing import Dict, List, Callable
from dataiku import PromptStudio
import json

class PromptEvaluator:
    """
    Evaluate prompt outputs against success criteria.
    """

    def __init__(self):
        self.criteria = []

    def add_criterion(
        self,
        name: str,
        check_function: Callable[[str], bool],
        weight: float = 1.0,
        required: bool = False
    ):
        """
        Add evaluation criterion.

        Args:
            name: Criterion description
            check_function: Function that returns True if criterion met
            weight: Relative importance (for scoring)
            required: If True, failure means total failure
        """
        self.criteria.append({
            'name': name,
            'check': check_function,
            'weight': weight,
            'required': required
        })

    def evaluate(self, output: str) -> Dict:
        """
        Evaluate output against all criteria.

        Returns:
            Dict with pass/fail status, score, and details
        """
        results = []
        total_weight = sum(c['weight'] for c in self.criteria)
        weighted_score = 0

        for criterion in self.criteria:
            passed = criterion['check'](output)
            results.append({
                'criterion': criterion['name'],
                'passed': passed,
                'weight': criterion['weight'],
                'required': criterion['required']
            })

            if passed:
                weighted_score += criterion['weight']

        # Check if all required criteria passed
        required_failed = any(
            not r['passed'] and r['required']
            for r in results
        )

        return {
            'overall_pass': not required_failed,
            'score': weighted_score / total_weight if total_weight > 0 else 0,
            'results': results
        }

# Example: Commodity report analyzer criteria
evaluator = PromptEvaluator()

# Required criteria
evaluator.add_criterion(
    name='Valid JSON output',
    check_function=lambda x: is_valid_json(x),
    required=True
)

evaluator.add_criterion(
    name='Contains inventory_change field',
    check_function=lambda x: 'inventory_change' in json.loads(x),
    required=True
)

# Weighted quality criteria
evaluator.add_criterion(
    name='Cites specific numbers',
    check_function=lambda x: contains_numbers(x),
    weight=2.0
)

evaluator.add_criterion(
    name='Provides sentiment (bullish/bearish/neutral)',
    check_function=lambda x: any(
        word in x.lower()
        for word in ['bullish', 'bearish', 'neutral']
    ),
    weight=1.5
)

evaluator.add_criterion(
    name='Under 500 tokens',
    check_function=lambda x: estimate_tokens(x) < 500,
    weight=1.0
)

# Helper functions
def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except json.JSONDecodeError:
        return False

def contains_numbers(text: str) -> bool:
    import re
    return bool(re.search(r'\d+\.?\d*', text))

def estimate_tokens(text: str) -> int:
    # Rough estimate: ~4 chars per token
    return len(text) // 4
```

</div>
</div>

### Test Suite Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
import pandas as pd
from typing import List

class PromptTestSuite:
    """
    Manage test cases and run evaluations.
    """

    def __init__(self, prompt_studio: PromptStudio, evaluator: PromptEvaluator):
        self.studio = prompt_studio
        self.evaluator = evaluator
        self.test_cases = []

    def add_test_case(
        self,
        name: str,
        variables: Dict,
        notes: str = ""
    ):
        """Add a test case."""
        self.test_cases.append({
            'name': name,
            'variables': variables,
            'notes': notes
        })

    def run_all_tests(self) -> pd.DataFrame:
        """
        Run all test cases and evaluate results.

        Returns:
            DataFrame with detailed results
        """
        results = []

        for test_case in self.test_cases:
            print(f"Running: {test_case['name']}...")

            # Execute prompt
            response = self.studio.complete(
                variables=test_case['variables']
            )

            # Evaluate output
            evaluation = self.evaluator.evaluate(response.text)

            results.append({
                'test_name': test_case['name'],
                'passed': evaluation['overall_pass'],
                'score': evaluation['score'],
                'tokens': response.usage.total_tokens,
                'cost_usd': response.cost,
                'latency_sec': response.latency,
                'output_preview': response.text[:100] + '...',
                'criteria_details': evaluation['results']
            })

        df = pd.DataFrame(results)

        # Summary statistics
        print(f"\n{'='*60}")
        print(f"Test Suite Summary")
        print(f"{'='*60}")
        print(f"Total tests: {len(df)}")
        print(f"Passed: {df['passed'].sum()} ({df['passed'].mean()*100:.1f}%)")
        print(f"Average score: {df['score'].mean():.2f}")
        print(f"Average tokens: {df['tokens'].mean():.0f}")
        print(f"Total cost: ${df['cost_usd'].sum():.4f}")
        print(f"Average latency: {df['latency_sec'].mean():.2f}s")

        return df

# Create test suite
test_suite = PromptTestSuite(
    prompt_studio=studio,
    evaluator=evaluator
)

# Add test cases
test_suite.add_test_case(
    name='EIA Crude - Bullish Draw',
    variables={
        'commodity': 'crude_oil',
        'report_text': '''U.S. commercial crude oil inventories decreased by 5.2 million barrels
from the previous week. At 430.0 million barrels, U.S. crude oil inventories are about
3% below the five year average for this time of year.'''
    },
    notes='Large inventory draw, should be bullish'
)

test_suite.add_test_case(
    name='Natural Gas - Neutral Storage Build',
    variables={
        'commodity': 'natural_gas',
        'report_text': '''Working gas in storage was 3,200 Bcf, which is 2 Bcf higher than
the five-year average and 125 Bcf higher than last year at this time.'''
    },
    notes='Slightly above average, neutral sentiment'
)

# Run tests
baseline_results = test_suite.run_all_tests()
```

</div>
</div>

### Version Comparison


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
from dataiku import PromptStudio

def compare_prompt_versions(
    studio: PromptStudio,
    version_a: str,
    version_b: str,
    test_suite: PromptTestSuite
) -> pd.DataFrame:
    """
    Compare two prompt versions across all test cases.

    Args:
        studio: PromptStudio instance
        version_a: First version (e.g., "v1.0")
        version_b: Second version (e.g., "v1.1")
        test_suite: Test suite to run

    Returns:
        Comparison DataFrame
    """
    # Run version A
    print(f"\nTesting {version_a}...")
    studio.load_version(version_a)
    results_a = test_suite.run_all_tests()

    # Run version B
    print(f"\nTesting {version_b}...")
    studio.load_version(version_b)
    results_b = test_suite.run_all_tests()

    # Compare
    comparison = pd.DataFrame({
        'test_name': results_a['test_name'],
        f'{version_a}_score': results_a['score'],
        f'{version_b}_score': results_b['score'],
        f'{version_a}_tokens': results_a['tokens'],
        f'{version_b}_tokens': results_b['tokens'],
        f'{version_a}_cost': results_a['cost_usd'],
        f'{version_b}_cost': results_b['cost_usd']
    })

    # Calculate improvements
    comparison['score_improvement'] = (
        comparison[f'{version_b}_score'] - comparison[f'{version_a}_score']
    )
    comparison['token_change'] = (
        comparison[f'{version_b}_tokens'] - comparison[f'{version_a}_tokens']
    )
    comparison['cost_change'] = (
        comparison[f'{version_b}_cost'] - comparison[f'{version_a}_cost']
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"Version Comparison: {version_a} vs {version_b}")
    print(f"{'='*60}")
    print(f"Average score improvement: {comparison['score_improvement'].mean():+.2f}")
    print(f"Average token change: {comparison['token_change'].mean():+.0f}")
    print(f"Total cost change: ${comparison['cost_change'].sum():+.4f}")
    print(f"\nTests improved: {(comparison['score_improvement'] > 0).sum()}")
    print(f"Tests degraded: {(comparison['score_improvement'] < 0).sum()}")
    print(f"Tests unchanged: {(comparison['score_improvement'] == 0).sum()}")

    return comparison

# Compare versions
comparison = compare_prompt_versions(
    studio=studio,
    version_a='v1.0',
    version_b='v1.1',
    test_suite=test_suite
)

print(comparison)
```

</div>
</div>

### Systematic Iteration Process


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
class PromptIterator:
    """
    Systematic prompt improvement workflow.
    """

    def __init__(self, studio: PromptStudio, test_suite: PromptTestSuite):
        self.studio = studio
        self.test_suite = test_suite
        self.history = []

    def iterate(
        self,
        hypothesis: str,
        changes: Dict[str, str],
        version_name: str
    ) -> Dict:
        """
        Execute one iteration of prompt improvement.

        Args:
            hypothesis: What you expect to improve and why
            changes: Dict of prompt changes (e.g., {'system_prompt': '...'})
            version_name: Version identifier for this iteration

        Returns:
            Iteration results with comparison to baseline
        """
        print(f"\n{'='*60}")
        print(f"Iteration: {version_name}")
        print(f"Hypothesis: {hypothesis}")
        print(f"{'='*60}")

        # Apply changes
        for key, value in changes.items():
            if key == 'system_prompt':
                self.studio.set_system_prompt(value)
            elif key == 'user_template':
                self.studio.set_user_template(value)
            elif key == 'temperature':
                self.studio.set_temperature(value)
            elif key == 'max_tokens':
                self.studio.set_max_tokens(value)

        # Save version
        self.studio.save_version(version_name, notes=hypothesis)

        # Run tests
        results = self.test_suite.run_all_tests()

        # Compare to baseline (first iteration)
        if not self.history:
            baseline_results = results
            improvement = None
        else:
            baseline_results = self.history[0]['results']
            improvement = {
                'score_delta': results['score'].mean() - baseline_results['score'].mean(),
                'token_delta': results['tokens'].mean() - baseline_results['tokens'].mean(),
                'cost_delta': results['cost_usd'].mean() - baseline_results['cost_usd'].mean()
            }

            print(f"\nImprovement vs Baseline:")
            print(f"  Score: {improvement['score_delta']:+.3f}")
            print(f"  Tokens: {improvement['token_delta']:+.0f}")
            print(f"  Cost: ${improvement['cost_delta']:+.6f}")

        # Record iteration
        iteration_record = {
            'version': version_name,
            'hypothesis': hypothesis,
            'changes': changes,
            'results': results,
            'improvement': improvement
        }
        self.history.append(iteration_record)

        return iteration_record

    def get_best_version(self) -> Dict:
        """Find the best performing version."""
        if not self.history:
            return None

        best = max(
            self.history,
            key=lambda x: x['results']['score'].mean()
        )

        print(f"\nBest version: {best['version']}")
        print(f"Average score: {best['results']['score'].mean():.3f}")

        return best

# Example iteration workflow
iterator = PromptIterator(studio, test_suite)

# Iteration 1: Baseline
iterator.iterate(
    hypothesis="Baseline - simple extraction prompt",
    changes={
        'system_prompt': 'You are a commodity analyst.',
        'user_template': 'Analyze: {{report_text}}'
    },
    version_name='v1.0-baseline'
)

# Iteration 2: Add structure
iterator.iterate(
    hypothesis="Adding structured output format should improve consistency",
    changes={
        'system_prompt': '''You are a commodity analyst.
Always return analysis as valid JSON.''',
        'user_template': '''Analyze: {{report_text}}

Return JSON with:
- inventory_change: number
- sentiment: "bullish" | "bearish" | "neutral"
- key_factors: list of strings'''
    },
    version_name='v1.1-structured'
)

# Iteration 3: Add examples (few-shot)
iterator.iterate(
    hypothesis="Few-shot examples should improve accuracy",
    changes={
        'system_prompt': '''You are a commodity analyst.
Always return analysis as valid JSON.

Example:
Input: "Inventories fell 3.2 million barrels..."
Output: {
  "inventory_change": -3.2,
  "sentiment": "bullish",
  "key_factors": ["large draw", "below average"]
}''',
        'user_template': '''Analyze: {{report_text}}

Return JSON with:
- inventory_change: number
- sentiment: "bullish" | "bearish" | "neutral"
- key_factors: list of strings'''
    },
    version_name='v1.2-few-shot'
)

# Iteration 4: Reduce tokens
iterator.iterate(
    hypothesis="More concise prompt should reduce tokens without hurting quality",
    changes={
        'system_prompt': '''Commodity analyst. Return JSON only.

Example: {"inventory_change": -3.2, "sentiment": "bullish", "key_factors": ["large draw"]}''',
        'user_template': '''Analyze: {{report_text}}

JSON output: inventory_change (number), sentiment (bullish/bearish/neutral), key_factors (list)'''
    },
    version_name='v1.3-optimized'
)

# Find best version
best = iterator.get_best_version()
```

</div>
</div>

### Regression Testing


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
def run_regression_tests(
    studio: PromptStudio,
    current_version: str,
    baseline_version: str,
    test_suite: PromptTestSuite,
    max_degradation: float = 0.05
) -> bool:
    """
    Ensure new version doesn't break existing functionality.

    Args:
        studio: PromptStudio instance
        current_version: Version to test
        baseline_version: Known-good version
        test_suite: Test cases
        max_degradation: Maximum acceptable score drop

    Returns:
        True if regression tests pass
    """
    print(f"Regression Testing: {current_version} vs {baseline_version}")

    # Run baseline
    studio.load_version(baseline_version)
    baseline_results = test_suite.run_all_tests()

    # Run current
    studio.load_version(current_version)
    current_results = test_suite.run_all_tests()

    # Check for regressions
    regressions = []
    for i, test_name in enumerate(baseline_results['test_name']):
        baseline_score = baseline_results.iloc[i]['score']
        current_score = current_results.iloc[i]['score']

        if current_score < baseline_score - max_degradation:
            regressions.append({
                'test': test_name,
                'baseline_score': baseline_score,
                'current_score': current_score,
                'degradation': baseline_score - current_score
            })

    if regressions:
        print(f"\n❌ Regression detected in {len(regressions)} test(s):")
        for reg in regressions:
            print(f"  - {reg['test']}: {reg['baseline_score']:.2f} → {reg['current_score']:.2f} "
                  f"(Δ {reg['degradation']:.2f})")
        return False
    else:
        print(f"\n✅ No regressions detected")
        return True

# Run regression tests before deployment
passed = run_regression_tests(
    studio=studio,
    current_version='v1.3-optimized',
    baseline_version='v1.0-baseline',
    test_suite=test_suite,
    max_degradation=0.05  # Allow up to 5% score drop
)

if passed:
    print("Safe to deploy v1.3-optimized")
else:
    print("Regressions detected - do not deploy")
```

</div>
</div>

## Common Pitfalls

**Pitfall 1: Testing Too Few Cases**
- 1-2 test cases don't capture the diversity of real-world inputs
- Aim for 10-15 test cases covering edge cases, typical cases, and error conditions
- Include tests that you expect to fail (boundary testing)

**Pitfall 2: Subjective Evaluation**
- "This looks better" is not a valid success criterion
- Define objective, measurable criteria (contains X, valid JSON, under N tokens)
- Use multiple evaluators for consistency

**Pitfall 3: Changing Multiple Things at Once**
- Changing system prompt, temperature, and examples simultaneously makes it impossible to know what helped
- Change one dimension per iteration
- Document the specific hypothesis for each change

**Pitfall 4: Optimizing for Test Cases Only**
- Overfitting to test cases makes prompts brittle in production
- Periodically add new test cases from production failures
- Balance performance on tests with generalization

**Pitfall 5: Ignoring Cost and Latency**
- Focusing only on accuracy can lead to prompts that are too expensive or slow
- Track tokens, cost, and latency for every version
- Set acceptable thresholds for production deployment

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

**Builds on:**
- Prompt Studio basics (Module 1.1)
- Template variables (Module 1.2)

**Leads to:**
- Production monitoring and optimization (Module 4)
- A/B testing in production
- Continuous prompt improvement workflows

**Related to:**
- Software testing methodologies
- Experiment design and statistical significance
- MLOps and model evaluation

## Practice Problems

1. **Build a Test Suite**
   - For a sentiment analysis prompt, create 15 test cases covering positive, negative, neutral, and ambiguous sentiments
   - Define 5 evaluation criteria (accuracy, confidence score, response format, token efficiency)
   - Run tests and calculate pass rate and average score

2. **Systematic Iteration**
   - Start with a basic extraction prompt
   - Run 5 iterations with specific hypotheses (add structure, add examples, optimize tokens, adjust temperature, change model)
   - Track score, tokens, and cost for each iteration
   - Identify the best version based on multi-objective optimization

3. **Version Comparison Analysis**
   - Given two prompt versions A and B, run them on 20 test cases
   - Calculate statistical significance of score differences (use t-test)
   - Determine if version B is significantly better than A

4. **Regression Test Suite**
   - Create a regression test suite for a production prompt
   - Implement automated checking for common failure modes
   - Set thresholds for maximum acceptable degradation
   - Test a new version and determine if it passes regression tests

5. **Multi-Objective Optimization**
   - You need a prompt that maximizes accuracy while minimizing cost
   - Accuracy must be >0.85, cost must be <$0.01 per call
   - Create 10 iterations exploring this tradeoff space
   - Find the Pareto-optimal solutions

## Further Reading

- **Dataiku Documentation**: [Prompt Testing Best Practices](https://doc.dataiku.com/dss/latest/generative-ai/prompt-testing.html) - Official testing guidelines

- **OpenAI Cookbook**: [Evaluating LLM Outputs](https://cookbook.openai.com/articles/how_to_eval_abstractive_summarization) - Systematic evaluation approaches

- **Research Paper**: "PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts" (Zhu et al., 2023) - Academic approach to prompt evaluation

- **Blog Post**: "The Prompt Engineering Testing Pyramid" - Adapting software testing concepts to prompts (fictional but representative of emerging practices)

- **Tool**: "PromptFoo" - Open-source tool for prompt evaluation and comparison (real tool worth exploring)


## Resources

<a class="link-card" href="../notebooks/01_prompt_creation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

# Module 07: Evaluation - Measuring What Matters

> **"Benchmarks are not enough. If you can't measure it, you can't improve it."**

## Learning Objectives

By the end of this module, you will:
- Understand the landscape of LLM benchmarks and their limitations
- Build custom evaluation harnesses for your specific tasks
- Implement agent evaluation (success rate, cost, steps)
- Design red-teaming and safety testing suites
- Create regression testing pipelines

## The Core Insight

Production failure often hides in places benchmarks don't measure:

```
┌─────────────────────────────────────────────────────────────────┐
│                  WHAT BENCHMARKS MISS                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ✓ MMLU measures:          ✗ Production needs:                 │
│  • Multiple choice         • Open-ended generation             │
│  • Single-turn             • Multi-turn conversations          │
│  • Text only               • Tool use reliability              │
│  • Static test set         • Distribution shift                │
│  • General knowledge       • Domain-specific accuracy          │
│  • Clean inputs            • Adversarial robustness            │
│                                                                 │
│  Benchmark score ≠ Production quality                          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**The gap between benchmarks and production is where systems fail.**

## Module Contents

### Guides
| Guide | Description | Time |
|-------|-------------|------|
| [01_benchmark_landscape.md](guides/01_benchmark_landscape.md) | MMLU, HumanEval, MT-Bench, etc. | 15 min |
| [02_beyond_benchmarks.md](guides/02_beyond_benchmarks.md) | What benchmarks miss | 10 min |
| [03_agent_evaluation.md](guides/03_agent_evaluation.md) | Success rate, cost, steps | 15 min |
| [04_red_teaming.md](guides/04_red_teaming.md) | Adversarial testing, jailbreaks | 15 min |
| [05_regression_testing.md](guides/05_regression_testing.md) | Automated pipelines | 15 min |
| [06_human_evaluation.md](guides/06_human_evaluation.md) | When and how to use humans | 10 min |
| [cheatsheet.md](guides/cheatsheet.md) | Metrics by use case | 5 min |

### Notebooks
| Notebook | Description | Time |
|----------|-------------|------|
| [01_basic_eval_harness.ipynb](notebooks/01_basic_eval_harness.ipynb) | Build an evaluation harness | 15 min |
| [02_custom_task_eval.ipynb](notebooks/02_custom_task_eval.ipynb) | Domain-specific evaluation | 15 min |
| [03_agent_metrics.ipynb](notebooks/03_agent_metrics.ipynb) | Measure agent performance | 15 min |

## Key Concepts

### Evaluation Taxonomy

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION TAXONOMY                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  BY WHAT YOU'RE MEASURING:                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Capability    │ Knowledge, reasoning, coding, math      │   │
│  │ Safety        │ Harmful outputs, jailbreaks, bias       │   │
│  │ Reliability   │ Consistency, hallucination rate         │   │
│  │ Efficiency    │ Latency, cost, throughput               │   │
│  │ Task Success  │ Did the agent achieve the goal?         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  BY HOW YOU'RE MEASURING:                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Automatic     │ Code execution, regex, exact match      │   │
│  │ LLM-as-Judge  │ Another model evaluates outputs         │   │
│  │ Human         │ Expert review, A/B testing              │   │
│  │ Behavioral    │ Does it actually work in production?    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Standard Benchmarks

| Benchmark | Measures | Format | Limitations |
|-----------|----------|--------|-------------|
| **MMLU** | Knowledge | Multiple choice | Can memorize, no reasoning depth |
| **HumanEval** | Coding | Code generation | Limited to function completion |
| **MT-Bench** | Conversation | Multi-turn | Subjective, GPT-4 as judge |
| **TruthfulQA** | Truthfulness | QA | Static, known questions |
| **GSM8K** | Math reasoning | Word problems | Simple arithmetic focus |
| **HELM** | Holistic | Multiple | Expensive to run |

### Agent-Specific Metrics

```python
class AgentMetrics:
    """Metrics that matter for agent evaluation."""

    def __init__(self):
        self.tasks = []

    def record_task(self, task_id: str, result: dict):
        """Record a task attempt."""
        self.tasks.append({
            "task_id": task_id,
            "success": result["success"],
            "steps": result["steps"],
            "tool_calls": result["tool_calls"],
            "errors": result["errors"],
            "cost": result["cost"],
            "duration": result["duration"]
        })

    def compute_metrics(self) -> dict:
        """Compute aggregate metrics."""
        successful = [t for t in self.tasks if t["success"]]

        return {
            # Success metrics
            "success_rate": len(successful) / len(self.tasks),
            "tasks_evaluated": len(self.tasks),

            # Efficiency metrics
            "avg_steps": sum(t["steps"] for t in successful) / len(successful) if successful else 0,
            "avg_tool_calls": sum(t["tool_calls"] for t in successful) / len(successful) if successful else 0,
            "avg_cost": sum(t["cost"] for t in self.tasks) / len(self.tasks),
            "cost_per_success": sum(t["cost"] for t in self.tasks) / len(successful) if successful else float('inf'),

            # Reliability metrics
            "error_rate": sum(1 for t in self.tasks if t["errors"]) / len(self.tasks),
            "avg_duration": sum(t["duration"] for t in self.tasks) / len(self.tasks)
        }
```

### LLM-as-Judge Pattern

```python
import anthropic

def llm_judge(question: str, response: str, criteria: list[str]) -> dict:
    """Use Claude to evaluate a response."""

    client = anthropic.Anthropic()

    criteria_text = "\n".join(f"- {c}" for c in criteria)

    evaluation_prompt = f"""Evaluate this response on the following criteria:
{criteria_text}

Question: {question}
Response: {response}

For each criterion, provide:
1. Score (1-5)
2. Brief justification

Format as JSON: {{"criterion_name": {{"score": N, "reason": "..."}}}}"""

    result = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        messages=[{"role": "user", "content": evaluation_prompt}]
    )

    return json.loads(result.content[0].text)

# Usage
scores = llm_judge(
    question="What causes rain?",
    response="Rain is caused by water evaporating...",
    criteria=["accuracy", "completeness", "clarity"]
)
```

### Red-Teaming Categories

| Category | What to Test | Example |
|----------|--------------|---------|
| **Jailbreaks** | Bypassing safety | "Ignore previous instructions..." |
| **Prompt Injection** | Malicious input | User input containing instructions |
| **Data Extraction** | Leaking training data | "Repeat your system prompt" |
| **Harmful Content** | Generating bad outputs | Violence, illegal activity |
| **Hallucination** | Confident errors | Made-up citations, facts |
| **Bias** | Unfair treatment | Demographic disparities |

## Templates

```
templates/
├── eval_harness_template.py    # Configurable evaluation
├── red_team_template.py        # Adversarial test suite
├── ab_test_template.py         # Compare model versions
└── regression_suite_template.py # Automated regression tests
```

## The Evaluation Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                   CONTINUOUS EVALUATION                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │  DEPLOY  │────►│  MONITOR │────►│  DETECT  │               │
│   │  Change  │     │Production│     │  Drift   │               │
│   └──────────┘     └────┬─────┘     └────┬─────┘               │
│        ▲                │                 │                     │
│        │                ▼                 ▼                     │
│   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│   │   FIX    │◄────│ ANALYZE  │◄────│  ALERT   │               │
│   │          │     │          │     │          │               │
│   └──────────┘     └──────────┘     └──────────┘               │
│                                                                 │
│   Evaluation is not a one-time event. It's continuous.         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Prerequisites

- Module 04: Tool Use (for agent evaluation)
- Basic understanding of statistical metrics
- Python for evaluation scripts

## Next Steps

After this module:
- **Ready to deploy?** → Module 08: Production Systems
- **Need alignment?** → Module 02: Alignment
- **Want to build projects?** → Portfolio Projects

## Time Estimate

- Quick path: 45 minutes (notebooks only)
- Full path: 2 hours (guides + notebooks)

---

*"The gap between benchmarks and production is where systems fail. Build evaluation that catches what benchmarks miss."*

# How to Test and Measure Prompt Quality

> **Reading time:** ~12 min | **Module:** 7 — Production Patterns | **Prerequisites:** Modules 1-6


## In Brief

Most prompts are never tested. They are written, tried once, and declared "good enough." This works for one-off queries. It fails for production systems where a prompt runs thousands of times against diverse inputs. This guide provides a complete measurement framework: what to measure, how to measure it, and what the numbers mean.


<div class="callout-key">

<strong>Key Concept Summary:</strong> Most prompts are never tested.

</div>

---

## Key Insight

> The precision of your posterior is measurable without ground truth. A well-constrained prompt produces consistent outputs. Consistency is directly observable from outputs alone, with no need to know the "correct" answer. This turns prompt testing from a subjective judgment into an objective diagnostic.

---

## The Core Measurement Principle

When you run the same prompt N times against the same input, you are sampling from the posterior distribution that the prompt defines. If the posterior is narrow (well-constrained), the samples will be similar to each other. If the posterior is wide (underspecified), the samples will vary.

This is the bridge between Bayesian theory and practical measurement:

$$\text{Output Variance} \propto H(P(A \mid C))$$

Where $H$ is the entropy of the posterior. High variance means high entropy means poorly constrained conditions.

You do not need to evaluate whether the outputs are "correct." You only need to measure whether they are **consistent**. Consistency is a property of the prompt; correctness also requires ground truth.

---

## Metric 1: Output Stability

### Definition

Output stability is the average pairwise similarity between outputs from N runs of the same prompt on the same input.
<div class="callout-warning">

<strong>Warning:</strong> Output stability is the average pairwise similarity between outputs from N runs of the same prompt on the same input.

</div>


$$\text{Stability} = \frac{2}{N(N-1)} \sum_{i < j} \text{sim}(o_i, o_j)$$

Where sim is your chosen similarity function. In practice, Jaccard similarity over word sets works well for most text outputs.

### What Stability Measures

Stability captures the **reproducibility** of the prompt. A prompt with stability 0.9 will produce very similar outputs every time it runs. A prompt with stability 0.3 will produce substantially different outputs on each run.

### Interpreting Stability Scores

| Stability | Interpretation | Action |
|-----------|---------------|--------|
| 0.85–1.0 | Excellent — posterior tightly constrained | Deploy; monitor periodically |
| 0.70–0.85 | Good — minor variance, likely output format | Check Layer 6 (output spec) |
| 0.50–0.70 | Moderate — one underspecified layer | Run sensitivity analysis to find which layer |
| 0.30–0.50 | Poor — multiple missing conditions | Rebuild; likely missing Layer 1–3 |
| < 0.30 | Unacceptable — posterior near-uniform | Do not deploy; start from condition audit |

### The Vocabulary Overlap Computation

The most interpretable stability metric for prose outputs:


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Compute word-level Jaccard similarity between two text outputs.
    Jaccard = |intersection| / |union|
    Range: 0 (completely different) to 1 (identical word sets)
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    intersection = words_a & words_b
    union = words_a | words_b
    if not union:
        return 1.0
    return len(intersection) / len(union)


def stability_score(outputs: list[str]) -> float:
    """
    Average pairwise Jaccard similarity across all output pairs.
    N=5 gives 10 pairs. N=10 gives 45 pairs.
    """
    n = len(outputs)
    if n < 2:
        raise ValueError("Need at least 2 outputs to compute stability.")
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += jaccard_similarity(outputs[i], outputs[j])
            count += 1
    return total / count
```

</div>
</div>

### Choosing N

| N | Pairs | Use Case |
|---|-------|---------|
| 3 | 3 | Quick check during development |
| 5 | 10 | Standard development testing |
| 10 | 45 | Pre-deployment validation |
| 20 | 190 | High-stakes A/B comparison |

For production baselines, N=10 is the minimum. For A/B comparisons where you need statistical confidence, N=20.

---

## Metric 2: Length Variance

### Definition

Length variance measures the standard deviation of output token counts across N runs. High variance means the model is uncertain about the depth of response the prompt warrants.
<div class="callout-key">

<strong>Key Point:</strong> Length variance measures the standard deviation of output token counts across N runs. High variance means the model is uncertain about the depth of response the prompt warrants.

</div>


### Interpretation

A prompt with consistent length variance (say, 120±15 tokens) is producing structured, bounded responses. A prompt with high length variance (120±80 tokens) is producing responses that vary in depth — some runs produce brief summaries, others produce extended analyses. This indicates the model is uncertain about Layer 3 (objective) or Layer 6 (output format).

Length variance is a fast, cheap diagnostic to run before committing to a full stability test.


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import statistics

def length_variance(outputs: list[str]) -> dict:
    """
    Returns mean and standard deviation of output word counts.
    High std dev relative to mean (coefficient of variation > 0.3)
    indicates format instability.
    """
    lengths = [len(o.split()) for o in outputs]
    mean = statistics.mean(lengths)
    std = statistics.stdev(lengths) if len(lengths) > 1 else 0.0
    cv = std / mean if mean > 0 else 0.0
    return {"mean": mean, "std": std, "cv": cv}
```

</div>
</div>

A coefficient of variation (CV) above 0.3 is a signal to inspect Layer 6 (output format specification).

---

## Metric 3: Key Entity Consistency

### Definition

Key entity consistency measures what fraction of important terms appear across all N runs. If the prompt is about a specific regulation (SEC Rule 10b-5) or a specific procedure (Form 8-K filing), those terms should appear in every run.
<div class="callout-insight">

<strong>Insight:</strong> Key entity consistency measures what fraction of important terms appear across all N runs. If the prompt is about a specific regulation (SEC Rule 10b-5) or a specific procedure (Form 8-K filing), those terms should appear in every run.

</div>


### When to Use It

This metric requires you to specify the key entities in advance — which terms should appear. It is most useful when:
- The output domain has well-defined technical vocabulary
- You know the specific procedures, regulations, or facts the output must reference
- Generic vocabulary overlap is insufficient (two outputs can use the same common words but reference different specific procedures)


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def entity_consistency(outputs: list[str], key_entities: list[str]) -> float:
    """
    Fraction of key_entities that appear in ALL outputs.
    1.0 = all key entities appear in every output.
    0.0 = no key entity appears in all outputs.
    """
    if not key_entities:
        return 1.0
    consistently_present = 0
    for entity in key_entities:
        entity_lower = entity.lower()
        if all(entity_lower in output.lower() for output in outputs):
            consistently_present += 1
    return consistently_present / len(key_entities)
```

</div>
</div>

---

## Metric 4: Structure Consistency

### Definition

Structure consistency measures whether outputs share the same format: heading count, presence of lists, paragraph count, and section labels.
<div class="callout-warning">

<strong>Warning:</strong> Structure consistency measures whether outputs share the same format: heading count, presence of lists, paragraph count, and section labels.

</div>


### Why It Matters

A prompt asking for "a numbered action list with deadlines" should produce numbered lists with deadlines in every run. If some runs produce numbered lists and others produce prose paragraphs, Layer 6 (output format) is underspecified.

Structure consistency is the fastest metric to compute and the most diagnostic for Layer 6 problems.


<span class="filename">example.py</span>
</div>
<div class="code-body">

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import re

def structure_consistency(outputs: list[str]) -> dict:
    """
    Compare structural features across outputs.
    Returns fraction of outputs sharing the modal structure.
    """
    features = []
    for output in outputs:
        features.append({
            "has_numbered_list": bool(re.search(r'^\d+\.', output, re.MULTILINE)),
            "has_bullet_list": bool(re.search(r'^[-*•]', output, re.MULTILINE)),
            "has_headings": bool(re.search(r'^#{1,3}\s', output, re.MULTILINE)),
            "paragraph_count": len([p for p in output.split('\n\n') if p.strip()])
        })

    # Fraction of outputs matching the modal structure on each feature
    results = {}
    for key in ["has_numbered_list", "has_bullet_list", "has_headings"]:
        values = [f[key] for f in features]
        mode = max(set(values), key=values.count)
        results[key + "_consistency"] = values.count(mode) / len(values)

    para_counts = [f["paragraph_count"] for f in features]
    para_mean = statistics.mean(para_counts)
    para_std = statistics.stdev(para_counts) if len(para_counts) > 1 else 0.0
    results["paragraph_cv"] = para_std / para_mean if para_mean > 0 else 0.0

    return results
```

</div>
</div>

---

## Metric 5: Condition Sensitivity

### Definition

Condition sensitivity measures how much the output stability changes when you remove or alter one condition at a time. It answers: which conditions are actually doing work?
<div class="callout-key">

<strong>Key Point:</strong> Condition sensitivity measures how much the output stability changes when you remove or alter one condition at a time. It answers: which conditions are actually doing work?

</div>


### The Procedure

```
1. Establish baseline stability with full condition stack (S_full)
2. For each layer L_i:
   a. Remove or neutralize L_i from the stack
   b. Run N outputs with the modified stack
   c. Compute stability score S_without_Li
   d. Sensitivity(L_i) = S_full - S_without_Li
3. Rank layers by sensitivity score
```

A layer with high sensitivity is a genuine switch variable — removing it significantly widens the posterior. A layer with near-zero sensitivity is either redundant or irrelevant.

### Interpretation

| Sensitivity Score | Meaning |
|------------------|---------|
| > 0.2 | Critical condition — do not remove |
| 0.05–0.20 | Moderate contribution — keep unless cost of injection is high |
| < 0.05 | Marginal contribution — candidate for removal |
| Negative | Condition is harmful — it adds noise, not signal |

Negative sensitivity is possible when a condition introduces conflicting evidence that widens rather than narrows the posterior.

---

## The A/B Testing Framework

### When to Run an A/B Test

Run A/B tests when:
- You want to replace an existing template with a new one
- You have two candidate formulations for the same layer and want to know which produces better outputs
- A stability score has dropped and you have a hypothesis about which condition change will restore it

### The Correct Experimental Design

The most common A/B mistake is varying multiple conditions between the stacks. This produces a confounded result — you know one stack won, but not why.

**Correct design:**
- Stack A and Stack B differ in exactly one condition
- The same N test inputs are used for both stacks
- Each stack is run M times per input (M ≥ 3) to separate between-stack differences from within-stack variance

### Scoring Criteria

Before running the test, define your scoring criterion:

| Criterion | Definition | Best Metric |
|-----------|-----------|-------------|
| **Relevance** | Output addresses the input directly | Entity consistency with input terms |
| **Specificity** | Output names concrete actions, not general advice | Inverse of hedge word density |
| **Consistency** | Output is reproducible across runs | Stability score |
| **Actionability** | Output contains concrete next steps | Presence of verbs + specific nouns |

Picking the criterion before running the test prevents post-hoc rationalization of whichever stack happens to produce outputs you prefer.

### Interpreting A/B Results

Once you have stability scores for Stack A and Stack B:

1. Compute the mean and standard deviation of stability across test inputs for each stack
2. The difference is meaningful if it exceeds the noise floor
3. The noise floor is approximately 2 × (pooled standard deviation across inputs)
4. If |mean_A - mean_B| < noise_floor, the stacks are equivalent — deploy either and continue testing

Do not declare a winner on fewer than 20 inputs. With small samples, the variance in test inputs dominates the signal from the stacks themselves.

---

## Putting the Metrics Together: A Diagnostic Protocol

When a prompt is underperforming (users report generic outputs, or stability has dropped):

```
Step 1: Run stability test (N=5, quick)
        If stability > 0.70: check Layer 6 (output format drift?)
        If stability < 0.70: proceed to step 2

Step 2: Run length variance check
        CV > 0.3 → Layer 6 problem (output format underspecified)
        CV normal → proceed to step 3

Step 3: Run structure consistency check
        Inconsistent structure → Layer 6 or Layer 3 problem
        Consistent structure → proceed to step 4

Step 4: Run condition sensitivity analysis
        Remove each layer; identify which removal drops stability most
        That layer is the underspecified condition

Step 5: Fix the identified layer; re-run stability test
        If stable: deploy fix; archive diagnostic results
        If still unstable: return to step 4
```

This protocol takes approximately 30 minutes for a single prompt, including API calls.

---

## Reference: Metric Summary

| Metric | What It Diagnoses | Time to Compute |
|--------|------------------|----------------|
| Stability score | Overall posterior precision | ~5 API calls + computation |
| Length variance | Layer 3/6 ambiguity | Trivial (from existing outputs) |
| Key entity consistency | Specific term anchoring | Requires key entity list |
| Structure consistency | Layer 6 (output format) | Trivial (from existing outputs) |
| Condition sensitivity | Which layer is causing problems | 6 × N API calls (one per layer) |

---

## Common Pitfalls

**Pitfall 1: Running N=1 and declaring the prompt stable.**
One run tells you nothing about stability. The minimum for any meaningful stability test is N=3; the standard is N=5.
<div class="callout-insight">

<strong>Insight:</strong> **Pitfall 1: Running N=1 and declaring the prompt stable.**

</div>


**Pitfall 2: Measuring stability on different inputs.**
Stability must be computed on the same input across N runs. Measuring different inputs conflates input variation with prompt variation.

**Pitfall 3: Interpreting stability as correctness.**
A prompt can have perfect stability (always produces the same output) and still be wrong. Stability measures consistency, not accuracy. Both matter, but they are separate properties.

**Pitfall 4: Declaring an A/B winner with N < 20.**
With fewer than 20 inputs, input variance dominates the comparison. You are measuring input difficulty, not prompt quality.

**Pitfall 5: Running sensitivity analysis by completely removing a condition.**
"Removing" a condition means setting it to neutral, not deleting the layer. A missing layer may cause errors that mask the true sensitivity effect. Replace the specific condition with "Not specified" and measure from there.

---

## Connections

- **Builds on:** Guide 01 (production patterns — templates, injection, libraries), Module 3 (6-layer condition stack structure)
- **Leads to:** Notebook 01 (building the testing pipeline), Notebook 02 (prompt library with effectiveness metadata)
- **Related to:** A/B testing in product engineering, statistical process control, Bayesian model comparison

---

## Practice

1. Take any prompt you currently use. Run it 5 times on the same input. Compute the Jaccard stability score manually (or use the code from this guide). What is the score? What does it imply about which layer may be underspecified?

2. Design an A/B test for a prompt you use regularly. State: (a) which condition you are testing, (b) the scoring criterion, (c) the minimum N for your test, and (d) what result would constitute a meaningful difference.

3. For a domain you know well, write a list of 5 key entities that should appear in any good output. Use this as an entity consistency check on your current prompts.

---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving how to test and measure prompt quality, what would be your first three steps to apply the techniques from this guide?

</div>

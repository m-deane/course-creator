# Guide Title

> **Reading time:** ~X min | **Module:** N — Topic | **Prerequisites:** Module N-1

---

<!-- ============================================================
     FULL FORMAT — Use for conceptual/theoretical guides that
     introduce a topic for the first time.
     ============================================================ -->

## In Brief

<div class="callout-insight">
  <strong>Insight:</strong> One-paragraph summary of the concept and why it matters.
</div>

## Formal Definition

State the formal mathematical or technical definition.

$$
f(x) = \sum_{i=1}^{n} w_i \cdot x_i
$$

where:
- $w_i$ is the weight for feature $i$
- $x_i \in \{0, 1\}$ indicates feature inclusion

## Intuitive Explanation

Explain the concept using an analogy or plain language. Reference the diagram below to make it concrete.

<!-- Insert SVG diagram here -->
<!-- Example: ![Concept Map](../resources/concept_map.svg) -->

<div class="caption">Figure 1: Description of the diagram.</div>

## Code Implementation

<div class="code-window">
  <div class="code-header">
    <div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
    <span class="filename">implementation.py</span>
  </div>

```python
# Primary implementation — use code-window for the key example
def main():
    data = load_data("dataset.csv")
    results = process(data)
    return results
```

</div>

Supporting examples use bare fenced blocks:

```python
# Purpose: Show alternative invocation pattern
results = main()
print(f"Processed {len(results)} records")
```

<div class="callout-warning">
  <strong>Warning:</strong> Highlight a common pitfall related to this implementation.
</div>

## Common Pitfalls

1. **Pitfall name** — Explanation of what goes wrong and why.

```python
# Purpose: Demonstrate the wrong approach
bad_result = process(data, leak_test=True)  # Data leakage!
```

2. **Pitfall name** — Explanation with corrected approach.

## Connections

<div class="callout-info">
  <strong>How this connects to the rest of the course:</strong>
</div>

- **Builds On:** [Prior concept](./XX_prior_guide.md) — what foundation this assumes
- **Leads To:** [Next concept](./XX_next_guide.md) — where this knowledge is applied
- **Related To:** [Parallel concept](./XX_related_guide.md) — complementary material

## Practice Problems

1. **Problem statement** — What the learner should implement or answer.

```python
# Skeleton code for the learner to complete
def solve():
    # TODO: Implement the solution
    pass
```

2. **Problem statement** — Second practice exercise.

## Further Reading

- Author, "Paper Title," *Journal*, Year. [Link](url)
- Author, *Book Title*, Publisher, Year.
- [Online Resource Title](url) — Brief description.

---

**Next:** [Companion Slides](./XX_concept_slides.md) | [Notebook](../notebooks/XX_concept_notebook.ipynb)


<!-- ============================================================
     COMPACT FORMAT — Use for practical/reference guides where
     the reader already has conceptual foundation.
     Delete everything above and use this format instead.
     ============================================================ -->

<!--
# Guide Title

> **Reading time:** ~X min | **Module:** N — Topic | **Prerequisites:** Module N-1

---

## [Topic-Driven Section 1]

Focused explanation of the first practical topic. Lead with working code.

<div class="code-window">
  <div class="code-header">
    <div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
    <span class="filename">example.py</span>
  </div>

```python
# Key implementation
result = run_pipeline(data, config)
```

</div>

## [Topic-Driven Section 2]

Focused explanation of the second practical topic.

```python
# Purpose: Quick supporting example
config = {"param": value}
```

<div class="callout-warning">
  <strong>Warning:</strong> Common mistake to avoid.
</div>

## Key Takeaways

<div class="callout-key">
  <strong>Key Takeaways:</strong>
  1. First takeaway — the most important practical lesson
  2. Second takeaway — what to watch out for
  3. Third takeaway — when to use this approach vs. alternatives
</div>

---

**Next:** [Companion Slides](./XX_concept_slides.md) | [Notebook](../notebooks/XX_concept_notebook.ipynb)
-->

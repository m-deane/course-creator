# Prompt Routing Bandits — Quick Reference

> **Reading time:** ~20 min | **Module:** 08 — Prompt Routing Bandits | **Prerequisites:** Module 7


## Core Concept


<div class="callout-key">

**Key Concept Summary:** Instead of manually testing prompts, let the system learn which prompt works best for each request type.

</div>

**Prompt routing = multi-armed bandit where arms are prompt templates**

Instead of manually testing prompts, let the system learn which prompt works best for each request type.

## The Bad Prompt Tax

<div class="callout-insight">

**Insight:** The core insight of bandit algorithms is that learning and earning are not separate phases. Every observation contributes to both understanding which option is best and generating value from the best option.

</div>


Every production LLM system pays:
- **Time tax:** Weeks of manual prompt iteration
- **Cost tax:** Wasted tokens on verbose/low-quality outputs
- **Trust tax:** User frustration from hallucinations
- **Opportunity tax:** Missing "excellent" because you settled for "good enough"

**Solution:** Adaptive prompt routing eliminates this tax.

---

## 1. Prompt Arm Design Checklist

<div class="callout-warning">

**Warning:** Bandit algorithms assume the reward distributions are stationary (or slowly changing). In commodity markets, regime shifts can make a historically optimal arm suddenly suboptimal. Always implement change detection alongside your bandit.

</div>


### Essential Prompt Arms for Commodity Trading LLM

| Prompt Type | Use When | Prevents |
|-------------|----------|----------|
| **Structured Extraction** | Data-heavy queries, tables | Verbose responses when you need numbers |
| **Evidence-Only (RAG-safe)** | Accuracy is critical, retrieval succeeds | Hallucinations |
| **Clarify-First** | Ambiguous requests | Confident guessing |
| **Quantitative Analysis** | Fundamental analysis, trend ID | Qualitative speculation |
| **Trading Signal** | Decision support needed | Hedged non-answers |
| **Scenario Analysis** | Risk assessment | Single-point estimates |

### Design Principles

- **5-8 prompts maximum** (too many = slow learning)
- **Each prompt solves ONE job** (extraction ≠ analysis ≠ signals)
- **Distinct instructions** (clear differences between prompts)
- **Observable differences** (you can measure when each works better)

---

## 2. Reward Function Template

### Structure: Primary Metric + Guardrails

```python
reward = primary_metric + Σ(guardrail_penalties)
```

### Primary Metrics (0 to 1)

| Metric | Measures | How to Calculate |
|--------|----------|------------------|
| **Task Completion** | Did it answer the question? | Automated check or LLM-as-judge |
| **Extraction Accuracy** | Correct data extraction? | Compare to ground truth |
| **Signal Quality** | Directionally correct signal? | Backtest against outcomes |
| **Research Completeness** | Covered all dimensions? | Checklist (supply/demand/price) |

### Guardrails (Penalties)

| Guardrail | Penalty | When to Apply |
|-----------|---------|---------------|
| **Hallucination Detection** | -0.3 to -0.5 per claim | Unsupported factual claims |
| **Format Compliance** | -0.3 | Invalid JSON, missing required fields |
| **Cost/Latency** | -0.1 to -0.2 | Exceeds token or time budget |
| **Citation Verification** | -0.2 | Missing or invalid sources |

### Example Composite Reward

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def commodity_llm_reward(query, response, docs, ground_truth=None):
    # Primary
    primary = task_completion_score(query, response)  # [0, 1]

    # Guardrails
    hallucination_pen = hallucination_penalty(response, docs)  # [-0.5, 0]
    format_pen = format_compliance(response)  # [-0.3, 0]
    citation_pen = citation_score(response, docs)  # [-0.2, 0.1]

    return max(0, primary + hallucination_pen + format_pen + citation_pen)
```

</div>

---

## 3. Context Features for Routing

### Key Features (Commodity LLM Systems)

| Feature | Values | Extraction Method |
|---------|--------|-------------------|
| **Task Type** | extraction, analysis, signal, scenario | Keyword matching or classifier |
| **Commodity Sector** | energy, agriculture, metals | Keyword matching |
| **Data Availability** | high, medium, low | RAG retrieval score |
| **User Preference** | concise, balanced, detailed | Inferred from history |
| **Urgency** | high, medium, low | Time of day, keywords |

### Context Vector Example (15 dimensions)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
context = [
    # Task type (one-hot, 5 features)
    1, 0, 0, 0, 0,  # extraction

    # Sector (one-hot, 4 features)
    1, 0, 0, 0,  # energy

    # Data availability (continuous, 1 feature)
    1.0,  # high

    # User preference (one-hot, 3 features)
    0, 1, 0,  # balanced

    # Urgency (continuous, 1 feature)
    0.5,  # medium

    # Intercept
    1.0
]
```

</div>

---

## 4. Algorithm Selection

| Algorithm | Use When | Code Complexity |
|-----------|----------|-----------------|
| **Thompson Sampling** | No context features, simple routing | 10 lines |
| **LinUCB** | Context features matter (task type, sector) | 30 lines |
| **Epsilon-Greedy** | Quick baseline, interpretable | 5 lines |

### Thompson Sampling (No Context)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class PromptRouter:
    def __init__(self, num_prompts):
        self.alpha = np.ones(num_prompts)  # Successes
        self.beta = np.ones(num_prompts)   # Failures

    def select(self):
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)

    def update(self, idx, reward):
        if reward > 0.7:
            self.alpha[idx] += 1
        else:
            self.beta[idx] += 1
```

</div>

### LinUCB (With Context)

```python
class ContextualPromptRouter:
    def __init__(self, num_prompts, context_dim, alpha=1.0):
        self.A = [np.identity(context_dim) for _ in range(num_prompts)]
        self.b = [np.zeros(context_dim) for _ in range(num_prompts)]
        self.alpha = alpha

    def select(self, context):
        ucb_scores = []
        for i in range(len(self.A)):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            expected = theta @ context
            uncertainty = np.sqrt(context @ A_inv @ context)
            ucb_scores.append(expected + self.alpha * uncertainty)
        return np.argmax(ucb_scores)

    def update(self, idx, context, reward):
        self.A[idx] += np.outer(context, context)
        self.b[idx] += reward * context
```

---

## 5. Prompt Routing Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     REQUEST ARRIVES                          │
│              "What are latest EIA inventories?"              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  FEATURE EXTRACTION                          │
│  • Task: extraction                                          │
│  • Sector: energy                                            │
│  • Data: high availability (RAG found 5 docs)                │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  BANDIT SELECTION                            │
│  Thompson Sampling or LinUCB                                 │
│  → Selects Prompt 2 (Evidence-Only)                          │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    LLM GENERATION                            │
│  Prompt 2 + Query + Retrieved Docs → GPT-4/Claude           │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                   REWARD CALCULATION                         │
│  • Task completion: 1.0                                      │
│  • Hallucination: 0 (all claims sourced)                     │
│  • Format: 1.0                                               │
│  Composite: 1.0                                              │
└───────────────────────────┬──────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    UPDATE BANDIT                             │
│  Prompt 2 gets positive update for this context             │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. Common Failure Modes & Fixes

| Failure Mode | Symptom | Fix |
|--------------|---------|-----|
| **Confident hallucinations** | High user satisfaction, low accuracy | Heavy hallucination penalty in reward |
| **Verbose outputs** | High cost, low task completion | Reward brevity + task completion |
| **Format drift** | Unparseable JSON responses | Format compliance penalty |
| **Cold start problems** | Bad prompts selected early | Warm-start with domain knowledge |
| **Slow convergence** | All prompts selected equally after 100 trials | Reduce number of arms (5-8 max) |

---

## 7. Production Deployment Checklist

### Before Launch

- [ ] Define 5-8 distinct prompt templates
- [ ] Implement composite reward function (primary + guardrails)
- [ ] Extract 10-15 context features
- [ ] Choose algorithm (Thompson Sampling or LinUCB)
- [ ] Set up logging (prompt used, reward, context)

### During Operation

- [ ] Monitor hallucination rate (should decrease over time)
- [ ] Track cost per query (should decrease with better routing)
- [ ] Measure task completion rate (should increase)
- [ ] Check prompt selection distribution (avoid 100% one prompt)

### Guardrails

- [ ] Hallucination detection system
- [ ] Format validation (reject invalid outputs)
- [ ] Cost ceiling (fallback to cheaper model if exceeded)
- [ ] Human-in-the-loop for high-stakes decisions

---

## 8. When NOT to Use Prompt Routing

**Don't use bandits when:**

1. **One prompt clearly dominates** across all contexts
2. **Feedback is too noisy** to measure quality reliably
3. **Cold-start is critical** (need optimal behavior immediately)
4. **Prompts are infinite** (use meta-prompting instead)
5. **Manual tuning is fast** (simple use case, rarely changes)

---

## 9. Key Metrics to Track

| Metric | Target | Red Flag |
|--------|--------|----------|
| **Hallucination Rate** | <5% | >20% |
| **Task Completion** | >80% | <50% |
| **Cost per Query** | Decreasing | Increasing |
| **User Satisfaction** | >4.0/5 | <3.0/5 |
| **Prompt Diversity** | 3+ prompts selected >10% | 1 prompt >90% |

---

## 10. Connection to Other Modules

| Module | Connection |
|--------|------------|
| **Module 2** | Thompson Sampling = prompt routing engine |
| **Module 3** | LinUCB = contextual prompt routing |
| **Module 5** | Same domain (commodities), different application |
| **Module 7** | Production deployment patterns |

---

## Quick Start Code

<div class="callout-danger">

**Danger:** Never deploy a bandit system without a kill switch and maximum allocation limits. An unconstrained bandit can allocate 100% of traffic/capital to a single arm, which creates catastrophic risk if the reward signal is noisy or delayed.

</div>


```python
# Initialize
prompts = ["Extraction...", "Analysis...", "Signal..."]
router = PromptRouter(len(prompts))

# For each request
for query in requests:
    # Select prompt
    idx = router.select()
    prompt = prompts[idx]

    # Generate
    response = llm_call(prompt, query)

    # Reward
    reward = evaluate(response)

    # Update
    router.update(idx, reward)
```

---

## Further Reading

- Original article: "Bandits for Prompts" by Shenggang Li
- Module guides: 01_fundamentals, 02_reward_design, 03_contextual_routing, 04_case_studies
- Notebooks: Hands-on implementations with commodity trading examples


---

## Conceptual Practice Questions

**Practice Question 1:** What is the primary tradeoff this approach makes compared to simpler alternatives?

**Practice Question 2:** Under what conditions would this approach fail or underperform?



---

## Cross-References

<a class="link-card" href="./01_prompt_routing_fundamentals.md">
  <div class="link-card-title">01 Prompt Routing Fundamentals</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_prompt_routing_fundamentals.md">
  <div class="link-card-title">01 Prompt Routing Fundamentals — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./02_reward_design_llm.md">
  <div class="link-card-title">02 Reward Design Llm</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_reward_design_llm.md">
  <div class="link-card-title">02 Reward Design Llm — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_contextual_prompt_routing.md">
  <div class="link-card-title">03 Contextual Prompt Routing</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_contextual_prompt_routing.md">
  <div class="link-card-title">03 Contextual Prompt Routing — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


# The Six Probability Mistakes in Prompt Engineering

> **Reading time:** ~13 min | **Module:** 6 — Probability Mistakes | **Prerequisites:** Module 1 Bayesian Frame


## In Brief

Every bad AI answer is a predictable output of a specific probability mistake. This guide catalogs the six most common mistakes, explains each one through the Bayesian lens, and provides the exact fix — with before/after examples for each.

## Key Insight

The model is not malfunctioning when it gives a bad answer. It is answering the most probable question consistent with your evidence. Your job is to make the right question the most probable one.


<div class="callout-key">
<strong>Key Concept Summary:</strong> Every bad AI answer is a predictable output of a specific probability mistake.
</div>

---

## Mistake 1: Confusing "More Detail" with "More Conditions"
<div class="callout-insight">
<strong>Insight:</strong> You get a generic answer. You try to fix it by adding more text — more context, more background, a longer paragraph. You get a slightly longer version of the same generic answer.
</div>


### What It Looks Like

You get a generic answer. You try to fix it by adding more text — more context, more background, a longer paragraph. You get a slightly longer version of the same generic answer.

### Why It Happens

There is a critical distinction between **information** (words that increase surface area) and **evidence** (conditions that constrain the posterior distribution). More information does not automatically mean more evidence.

Consider the Bayesian update:

$$P(\text{answer} \mid \text{prompt}) = \frac{P(\text{prompt} \mid \text{answer}) \cdot P(\text{answer})}{P(\text{prompt})}$$

Adding generic background text increases $P(\text{prompt})$ but does not shift which specific answer has the highest posterior. Evidence shifts the posterior by making some answer states far more likely than others.

**Information:** "I'm asking about database selection for a startup working in the logistics space with multiple clients and a need for good reporting."

**Evidence:** "Read/write ratio: 80% reads. Data shape: wide rows, sparse columns. Query pattern: ad-hoc aggregations, not point lookups. Scale: 10M rows today, 500M in 18 months. Deployment constraint: managed cloud only."

The second is shorter but contains far more discriminating conditions.

### Before / After

**Before (more information, not evidence):**
```
I'm a software engineer at a Series B startup in the supply chain space. We have
a backend system that handles inventory tracking for multiple warehouse clients.
We need a database that can handle our use case. We currently use PostgreSQL but
are wondering if we should switch. What database should we use?
```

**After (conditions as evidence):**
```
Database selection: help me evaluate switching from PostgreSQL.

Current conditions:
- Query pattern: 90% time-series aggregations over 12-month windows
- Write pattern: append-only sensor data, ~50k inserts/minute
- Read pattern: dashboards aggregating by warehouse/SKU/date
- No joins needed: all queries are on a single fact table
- Current pain: PostgreSQL query times degrade past 100M rows

What's the right database choice given these specific conditions?
```

### The Fix

Before adding text, ask: "Does this sentence change which answer is correct, or does it just describe the situation?" Only add the former.

Structure your additions as conditions, not narrative:
- Change conditions that flip the answer
- State them explicitly, not embedded in prose
- Remove text that doesn't change the answer

---

## Mistake 2: Asking for One Answer When You Need a Conditional Tree
<div class="callout-warning">
<strong>Warning:</strong> You ask a question and get a confident single answer. You follow the advice. It doesn't work — or it works for someone else's situation but not yours.
</div>


### What It Looks Like

You ask a question and get a confident single answer. You follow the advice. It doesn't work — or it works for someone else's situation but not yours.

### Why It Happens

Every question has a decision tree structure underneath it. The branches are the conditions that would change the correct answer. When you ask for a single answer, you are collapsing that tree into a single node. The model picks the most common branch in its training data — which is statistically likely to be wrong for your specific situation.

```
P(correct answer | your specific conditions)
≠
P(most common answer | no conditions specified)
```

This is not a model failure. It is a prompt design choice you made implicitly.

### Before / After

**Before (asking for one answer):**
```
Should I build my MVP as a mobile app or web app?
```

**After (asking for the conditional tree):**
```
Before giving a recommendation, map out the decision tree for MVP platform choice.

What are the 3-4 conditions that most strongly determine whether mobile or web
is better? For each condition, show what choice it points toward.

Then, given my conditions — B2B SaaS, users are operations managers at
manufacturing companies, primary use case is shift scheduling reviewed once
daily, no offline requirement — which branch am I on, and what's the answer?
```

### The Fix

Recognize questions with hidden conditional structure. Signals include:
- Questions with "or" in them (A or B?)
- Questions where the answer depends on who's asking
- Questions where different industries, roles, or contexts would answer differently
- Any question where you've seen different advisors give opposite advice

For these questions, add: "Before answering, map the conditions that would change this answer. Then tell me which branch I'm on."

---

## Mistake 3: Treating AI Like a Search Engine (Keywords Instead of Inference)
<div class="callout-key">
<strong>Key Point:</strong> Prompts that look like search queries: short, noun-heavy, no structure. The model returns a document summary or a list of definitions. You wanted reasoning.
</div>


### What It Looks Like

Prompts that look like search queries: short, noun-heavy, no structure. The model returns a document summary or a list of definitions. You wanted reasoning.

### Why It Happens

Search engines retrieve documents ranked by relevance to keywords. Language models do something fundamentally different: they perform conditional inference. Given your prompt as evidence, they generate the most probable continuation.

When you write a keyword prompt, you are asking for the completion of a search-result document. The model obliges. You get retrieval, not reasoning.

The correct mental model: you are not querying a database. You are specifying the conditions under which a smart, knowledgeable expert would generate their response. Those are very different specification problems.

**Keyword prompt:** "Python async performance optimization database queries"

**Conditional inference prompt:** "I have a Python service making sequential database calls inside an async event loop. Each call takes ~20ms. Under high load (500 concurrent users), the event loop blocks and latency spikes to 4s. Walk me through the specific async patterns that fix this — with the tradeoffs between each approach given that I cannot change my database (Postgres) or my ORM (SQLAlchemy)."

### Before / After

**Before (keyword prompt):**
```
machine learning model deployment latency reduction
```

**After (inference specification):**
```
I need to reduce inference latency for a fine-tuned BERT model in production.

Current state: 280ms p95 latency, model served on a single A10G GPU,
batch size = 1 (no batching), Python Flask API, ~200 requests/minute.

Target: under 80ms p95.

What are the highest-ROI latency reduction techniques for this specific setup?
For each technique, tell me: (1) expected latency improvement, (2) implementation
complexity, (3) what I'd lose (accuracy, throughput, etc.).
```

### The Fix

Replace keyword phrases with inference specifications:
1. State the current state (what exists)
2. State the target state (what you want)
3. State the constraints (what cannot change)
4. Ask for reasoning under those conditions, not a document about the topic

The question "what is X?" asks for a definition. The question "given conditions A, B, C — what should I do about X?" asks for conditional inference.

---

## Mistake 4: Not Specifying the Objective Function

### What It Looks Like
<div class="callout-insight">
<strong>Insight:</strong> You ask for advice. The advice seems reasonable. But when you look closely, it's optimizing for the wrong thing — speed when you wanted safety, cost when you wanted reliability, the average case when you're in the tail.
</div>


You ask for advice. The advice seems reasonable. But when you look closely, it's optimizing for the wrong thing — speed when you wanted safety, cost when you wanted reliability, the average case when you're in the tail.

### Why It Happens

Every recommendation implicitly optimizes for something. When you don't specify what to optimize for, the model defaults to the most common objective function in its training data — usually "minimize cost" or "maximize simplicity" for engineering questions, "most common treatment" for medical questions, "standard practice" for business questions.

$$\text{optimal answer} = \arg\max_{\text{answer}} \mathbb{E}[\text{value} \mid \text{answer, conditions, objective}]$$

If you don't specify the objective, the model inserts one. It is almost certainly not exactly yours.

This matters enormously in domains with competing objectives:
- **Engineering:** speed vs. correctness vs. maintainability vs. cost
- **Finance:** return vs. risk vs. liquidity vs. regulatory compliance
- **Medical:** efficacy vs. side effects vs. cost vs. patient preference
- **Legal:** winning vs. speed vs. cost vs. preserving relationship

### Before / After

**Before (no objective function):**
```
How should I structure my cloud infrastructure for a new web application?
```

**After (objective function specified):**
```
Cloud infrastructure design for a new web application.

My objective function: minimize time-to-first-deployment, accept higher
monthly cost and lower scalability ceiling in exchange. We are 2 engineers,
launch in 4 weeks, and can re-architect in 6 months if we get traction.

I am explicitly NOT optimizing for: cost efficiency, infinite scalability,
or production best practices that add operational complexity.

Given this objective, what's the right starting infrastructure?
```

### The Fix

Before any recommendation request, state explicitly:
1. What you are optimizing FOR (the primary objective)
2. What secondary objectives exist (ranked)
3. What you are explicitly NOT optimizing for
4. What the constraints are (things that cannot be traded away)

A clean format: "Optimize for [primary goal]. Acceptable to sacrifice [tradeables]. Hard constraints: [non-negotiables]."

---

## Mistake 5: Ignoring Temporal Conditions

### What It Looks Like
<div class="callout-warning">
<strong>Warning:</strong> The advice is correct — but for a different time. You're asking about what to do now, and the model answers based on what the right answer was historically, or what the right answer will be eventually, or what the right answer is in steady state — none of which is what you need.
</div>


The advice is correct — but for a different time. You're asking about what to do now, and the model answers based on what the right answer was historically, or what the right answer will be eventually, or what the right answer is in steady state — none of which is what you need.

### Why It Happens

Conditions change over time, and the right answer often depends not just on what conditions exist but on what phase or moment you're in. The Bayesian update shifts based on temporal variables:

$$P(\text{correct action} \mid \text{conditions, t=now}) \neq P(\text{correct action} \mid \text{conditions, t=steady\_state})$$

Temporal conditions include:
- **Phase of a project:** early vs. late, pre-launch vs. post-launch
- **Point in a cycle:** market top vs. bottom, beginning of quarter vs. end
- **State of a process:** before you have data vs. after you have data
- **External timing:** regulation changes, technology transitions, seasonal patterns
- **Maturity stage:** startup vs. scale-up vs. enterprise

### Before / After

**Before (no temporal conditions):**
```
Should I hire a head of sales now?
```

**After (temporal conditions specified):**
```
I'm evaluating whether to hire a head of sales. Temporal conditions:

- Current state: $180k ARR, 3 customers acquired by founders via direct outreach
- Stage: pre-product-market-fit (still iterating on ICP and pricing)
- Time to next funding: 4 months (need to hit $400k ARR milestone)
- Sales motion: still undefined — no repeatable playbook yet

Given where I am in this timeline, should I hire a head of sales now or
at a later phase? What temporal trigger would change this answer?
```

### The Fix

Add temporal conditions as a separate clause:
- "At this stage of [project/company/market]..."
- "Given that I'm currently in [phase/state]..."
- "The trigger for the next phase will be [milestone/condition]"
- "This question is for the next [time horizon], not steady state"

Also ask: "What would have to change about the timing or phase for this answer to change?"

---

## Mistake 6: Assuming the Model Shares Your Priors

### What It Looks Like
<div class="callout-key">
<strong>Key Point:</strong> You ask a question that seems specific. The model answers confidently, accurately — for the average person in the average situation. But you're not average. Your context, your constraints, your existing beliefs are different from the training distribution. The model doesn't know this, and neither do you until the advice fails.
</div>


You ask a question that seems specific. The model answers confidently, accurately — for the average person in the average situation. But you're not average. Your context, your constraints, your existing beliefs are different from the training distribution. The model doesn't know this, and neither do you until the advice fails.

### Why It Happens

Language models have priors built from their training data. Those priors represent the distribution of situations humans have written about. When you prompt without specifying your priors, the model answers from its priors — which represent the average, the common case, the mainstream assumption.

$$P(\text{answer} \mid \text{your prior, conditions}) \neq P(\text{answer} \mid \text{training prior, conditions})$$

Examples of misaligned priors:
- **Your prior:** "good software always uses microservices" (your team has deep K8s experience)
  **Training prior:** "good software architecture depends on team size and scale"
- **Your prior:** "this is a high-risk situation" (you've been burned before)
  **Training prior:** "this is a standard business decision"
- **Your prior:** "compliance is mandatory" (regulated industry)
  **Training prior:** "compliance is a consideration to weigh"

### Before / After

**Before (priors assumed, not stated):**
```
How should I handle error logging in my application?
```

**After (priors made explicit):**
```
Error logging approach for my application.

My existing priors to factor in:
- We already have Datadog APM instrumented everywhere — prefer to centralize
  there rather than add another tool
- Our on-call rotation has burned out on alert fatigue — I'm biased toward
  fewer, higher-signal alerts over comprehensive logging
- We have a SOC 2 Type II audit coming in 3 months — anything touching user
  data must have tamper-evident logs

Given these priors, what's the right error logging approach? Explicitly
note where your recommendation would differ if these priors weren't true.
```

### The Fix

Before asking for advice, ask yourself: "What do I already believe about this? What constraints exist that the average person wouldn't have? What would I push back on if the model gave me standard advice?"

Then make those priors explicit:
- "My existing constraint is..."
- "I already believe that... so only give me advice that assumes this"
- "The standard recommendation here is X, but for my situation it doesn't apply because..."
- "Where your answer would normally differ: explicitly tell me how your answer changes given my specific priors"

---


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving the six probability mistakes in prompt engineering, what would be your first three steps to apply the techniques from this guide?
</div>

## Summary

| Mistake | Root Cause | Diagnostic Signal | Fix |
|---------|-----------|-------------------|-----|
| 1. Detail ≠ Conditions | Adding text without discriminating conditions | More text, same wrong answer | Separate information from evidence |
| 2. One Answer Instead of Tree | Collapsing conditional structure | Advice wrong for your specific case | Ask for the tree first |
| 3. Keyword Prompts | Treating inference as retrieval | Generic, document-like responses | Specify the inference chain |
| 4. No Objective Function | Implicit optimization target | Advice optimizes for the wrong goal | State what you're maximizing |
| 5. Ignoring Time | Temporal conditions absent | Advice correct for a different phase | Specify the temporal context |
| 6. Misaligned Priors | Model priors ≠ your priors | Standard answer, non-standard situation | Make your priors explicit |

---

## Connections

- **Builds on:** Module 2 (switch variables — conditions that flip answers), Module 3 (condition stack), Module 4 (conditional trees)
- **Leads to:** Module 7 (production patterns — automating condition injection so mistakes can't happen)
- **Related to:** The diagnostic framework in `02_diagnostic_framework_guide.md`

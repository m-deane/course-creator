# Agent Conditioning: How Bayesian Conditions Flow Through Multi-Step Workflows

> **Reading time:** ~9 min | **Module:** 5 — Agents & Workflows | **Prerequisites:** Module 4 Conditional Trees


## In Brief

An AI agent workflow is a sequence of conditional probability computations. Each step receives some context, generates output conditioned on that context, and passes (some subset of) that output forward. The conditions that shaped step 1's reasoning are present in step 5's reasoning only if they were explicitly included in every handoff payload. Conditions that are not explicitly passed decay — they are replaced by the model's training prior.

## Learning Objectives

By the end of this guide you will be able to:

1. Explain why conditions decay across agent chains using the probability framework
2. Identify the three mechanisms that cause condition loss
3. Design structured handoff payloads that preserve the full condition stack
4. Implement context summaries as a token-efficient alternative to full condition passing
5. Distinguish between conditions that must persist vs. conditions that can be regenerated


<div class="callout-key">
<strong>Key Concept Summary:</strong> An AI agent workflow is a sequence of conditional probability computations.
</div>

---

## The Single-Turn Baseline

In a single-turn prompt, the Bayesian frame is clear:

$$P(\text{answer} \mid \text{jurisdiction}, \text{time}, \text{objective}, \text{constraints}, \text{facts}, \text{format})$$

You specify all conditions in one prompt. The model computes the posterior over answers conditioned on everything you provided. If you provided the right conditions, the answer is precise.

The condition stack from Module 3 encodes this directly:

```
Layer 1: Jurisdiction / Rule Set
Layer 2: Time / Procedural Posture
Layer 3: Objective Function
Layer 4: Constraints
Layer 5: Facts
Layer 6: Output Specification
```

In a single-turn setting, all six layers are simultaneously present in the model's context window. The posterior is fully conditioned.

---

## The Multi-Agent Problem

When work is distributed across multiple agents, each agent receives only what was passed to it. Consider a three-agent pipeline for contract analysis:
<div class="callout-warning">
<strong>Warning:</strong> When work is distributed across multiple agents, each agent receives only what was passed to it. Consider a three-agent pipeline for contract analysis:
</div>


```
Agent 1 (Classifier): "What type of contract is this?"
  Input: raw contract text
  Output: "This is a commercial lease agreement governed by California law."

Agent 2 (Risk Extractor): "What are the key risk clauses?"
  Input: Agent 1's output
  Output: List of clauses

Agent 3 (Advisor): "Should the client sign this?"
  Input: Agent 2's clause list
  Output: Recommendation
```

What conditions does Agent 3 have?

- It has a list of clauses.
- It does NOT have: the client's risk tolerance, their jurisdiction (unless Agent 1's output was passed through), their timeline, their alternative options, their objective (minimize risk? close fast? preserve optionality?).

Agent 3 will fill those missing conditions from its training prior. The training prior for "should a client sign this?" optimizes for the average client. Your client is not the average client.

---

## Condition Decay: The Three Mechanisms

### Mechanism 1: Output Summarization

Agents summarize their outputs. Summaries preserve salience, not completeness. Conditions that seem peripheral to Agent 2 — but are critical for Agent 5 — get dropped.

```
Full context passed:   1200 tokens
After Agent 2 summary: 80 tokens
Conditions lost: objective function, constraints, procedural posture
```

### Mechanism 2: Context Window Limits

As conversations grow, earlier context scrolls out of the model's effective window. System prompts from step 1 may be technically present but receive less attention weight as the conversation lengthens.

### Mechanism 3: Implicit Assumption of Shared Context

Prompts written for Agent 3 often assume Agent 3 "knows" things that were only established in Agent 1's system prompt. This is a conditioning error: the model only knows what is in its current context.

---

## Visualizing Condition Flow

### Full propagation (what you want):

```
User Question + Conditions
        │
        ▼
┌───────────────────┐
│   Agent 1         │
│  [C1,C2,C3,C4,C5,C6] ──► Output + {condition_stack: {...}}
└───────────────────┘
        │ passes condition_stack forward
        ▼
┌───────────────────┐
│   Agent 2         │
│  [C1,C2,C3,C4,C5,C6] ──► Output + {condition_stack: {...}}
└───────────────────┘
        │ passes condition_stack forward
        ▼
┌───────────────────┐
│   Agent 3         │
│  [C1,C2,C3,C4,C5,C6] ──► Final Answer
└───────────────────┘
```

### Condition decay (what typically happens):

```
User Question + Conditions
        │
        ▼
┌───────────────────┐
│   Agent 1         │
│  [C1,C2,C3,C4,C5,C6] ──► "Here is the contract type..."
└───────────────────┘
        │ passes only output text
        ▼
┌───────────────────┐
│   Agent 2         │
│  [C5 only]        ──► "Here are the risk clauses..."
└───────────────────┘
        │ passes only clause list
        ▼
┌───────────────────┐
│   Agent 3         │
│  [C5 fragment]    ──► Fills C1,C2,C3,C4 from prior
└───────────────────┘
```

---

## The Structured Handoff Solution

The fix is to treat the condition stack as a first-class data object — not as implicit context embedded in natural language outputs.
<div class="callout-warning">
<strong>Warning:</strong> The fix is to treat the condition stack as a first-class data object — not as implicit context embedded in natural language outputs.
</div>


### Step 1: Define a condition payload schema

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
condition_payload = {
    "layer_1_jurisdiction": "California, USA — commercial tenancy law",
    "layer_2_time": "Pre-signature review, 2025",
    "layer_3_objective": "Minimize tenant liability exposure, not minimize rent",
    "layer_4_constraints": [
        "Client budget ceiling: $8,000/month",
        "Move-in required before March 1",
        "No subletting needed"
    ],
    "layer_5_facts": "5-year lease, mixed-use property, force majeure clause present",
    "layer_6_output_spec": "Bullet recommendations with confidence levels"
}
```

</div>
</div>

### Step 2: Require each agent to pass conditions forward

Every agent's output schema includes the condition payload:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
agent_output = {
    "result": "...",           # The agent's actual output
    "condition_stack": {...},  # Conditions received — passed through unchanged
    "added_context": {...}     # New conditions this agent contributes
}
```

</div>
</div>

### Step 3: Inject conditions into every downstream system prompt

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
def build_agent_prompt(task: str, condition_stack: dict) -> str:
    conditions_text = "\n".join([
        f"- {k}: {v}" for k, v in condition_stack.items()
    ])
    return f"""You are operating under the following conditions that MUST shape your reasoning:

{conditions_text}

Task: {task}

Do not deviate from the objective and constraints above. If the task conflicts with these conditions, flag the conflict explicitly rather than resolving it silently."""
```

</div>
</div>

---

## Context Summaries: Token-Efficient Condition Passing
<div class="callout-key">
<strong>Key Point:</strong> In long pipelines, passing the full condition payload at every step costs tokens. The alternative is a **condition summary** — a compressed representation that preserves the high-leverage conditions.
</div>


In long pipelines, passing the full condition payload at every step costs tokens. The alternative is a **condition summary** — a compressed representation that preserves the high-leverage conditions.

### What to include in a condition summary

Include only conditions that:
1. Are not recoverable from the task description alone
2. Would change the reasoning if assumed differently
3. Constrain the solution space (not just describe it)

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
CONDITION_SUMMARY_PROMPT = """
Given the following condition stack, produce a 3-5 sentence condition summary.
Include: jurisdiction/rule set, objective function, and top 2 constraints.
Exclude: facts (these are task-specific), output format (specify per-agent).

Condition stack:
{condition_stack}

Condition summary:
"""
```

</div>
</div>

### Trade-off table

| Approach | Token cost | Condition fidelity | Best for |
|----------|-----------|-------------------|----------|
| Full payload pass-through | High | Perfect | Short pipelines, high-stakes decisions |
| Condition summary | Medium | Good (high-leverage preserved) | Medium pipelines |
| System prompt only | Low | Partial (static conditions only) | Long pipelines with stable conditions |
| No explicit passing | None | Decay to prior | Never recommended |

---

## Switch Variables as the Minimal Condition Set

From Module 2: switch variables are the conditions that flip the solution branch. In an agent pipeline, switch variables are the minimum set of conditions that must survive every handoff.

Identify them before designing the pipeline:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>
<div class="code-body">

```python
# For a contract review pipeline:
switch_variables = {
    "deal_breaker_threshold": "client will walk away if X",
    "jurisdiction": "determines which clauses are enforceable",
    "timeline_pressure": "tight timeline = accept more risk",
    "client_type": "individual vs. corporation (affects liability)")
}
```

</div>
</div>

Pass these at minimum. They are the conditions whose absence causes the largest posterior shift.

---

## The Condition Decay Diagnostic

When an agent gives a wrong or imprecise answer, run this diagnostic before debugging the model:

```
1. List the conditions the task requires
2. For each condition, ask: "Is this condition present in this agent's context?"
3. For missing conditions: was it in an earlier step? Was it passed forward?
4. Identify the first step where the condition was dropped
5. Fix the handoff at that step
```

This diagnostic usually reveals the problem in under 5 minutes. The answer is almost always: "Agent N received output text but not the condition payload."

---

## Common Pitfalls

**Pitfall 1: Assuming system prompts persist across API calls**
Each API call is stateless. A system prompt in call 1 has zero effect on call 2 unless you include it in call 2's system parameter.

**Pitfall 2: Trusting natural language to carry conditions**
"Please keep in mind the client's risk tolerance" in step 1 does not condition step 3's reasoning. Only structured data in the context window conditions reasoning.

**Pitfall 3: Treating condition passing as an optimization**
Teams add condition passing "later when performance is an issue." Do it from the start. Retrofitting structured handoffs into an existing pipeline is painful.

**Pitfall 4: Over-specifying conditions at every step**
Not every condition is relevant at every step. Over-stuffing contexts degrades attention on relevant conditions. Pass what each agent needs, not everything.

---

## Connections

- **Builds on:** Module 3 (Condition Stack), Module 2 (Switch Variables)
- **Leads to:** Module 7 (Production Patterns — dynamic condition injection at scale)
- **Related to:** Guide 02 (Claude-specific features as condition layers)

---

## Practice Problems

1. A research pipeline has 4 agents: Search, Synthesize, Critique, Report. Sketch the condition payload schema. Which conditions does each agent need? Which can be omitted at each step?

2. An agent gives a confident but wrong answer in step 4 of a 5-step pipeline. Using the decay diagnostic, walk through how you would identify which step dropped the critical condition.

3. You have a 10-step pipeline and cannot afford full payload passing due to token costs. Design a condition summary strategy: which layers go in the summary, which are in each agent's system prompt, and which are regenerated per-task?

---


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Learning Objectives" and why it matters in practice.

2. Given a real-world scenario involving agent conditioning: how bayesian conditions flow through multi-step workflows, what would be your first three steps to apply the techniques from this guide?
</div>

## Further Reading

- Anthropic: "Building effective agents" (contexts and memory patterns)
- LangChain documentation: Conversation memory types — illustrates condition persistence patterns
- Lilian Weng: "LLM-powered Autonomous Agents" — covers context management challenges

---

## Cross-References

<a class="link-card" href="../notebooks/01_condition_aware_agent.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

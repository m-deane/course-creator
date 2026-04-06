# Production Patterns: Bayesian Prompting at Scale

> **Reading time:** ~12 min | **Module:** 7 — Production Patterns | **Prerequisites:** Modules 1-6


## In Brief

A prompt that works once is a result. A prompt that works reliably across thousands of queries, multiple users, and changing context is a system. This guide covers the infrastructure layer that separates experimental prompting from production prompting.


<div class="callout-key">

<strong>Key Concept Summary:</strong> A prompt that works once is a result.

</div>

---

## Key Insight

> Prompt engineering without infrastructure is a craft. Prompt engineering with infrastructure is an engineering discipline. The difference is not the quality of individual prompts — it is whether quality survives organizational scale, personnel turnover, and load.

---

## The Production Problem

Consider what happens to a well-crafted condition stack six months after you build it:

1. A colleague modifies it for their use case — without versioning
2. Another colleague uses a truncated version because "the full one takes too long to fill out"
3. A third person writes a new one from scratch for a similar use case
4. Nobody can compare which version produces better outputs because there is no test harness

This is **prompt entropy**. The Bayesian structure erodes, the posteriors widen, and output quality degrades. The symptoms look like "AI got worse" when the real cause is "prompts got worse."

The fix is not discipline or process — it is infrastructure. Condition stacks become parameterized templates. Context becomes injected data. Quality becomes a measured metric.

---

## Pattern 1: Parameterized Condition Stack Templates
<div class="callout-warning">

<strong>Warning:</strong> A static prompt is a fully-written condition stack stored as a string. It works, but it does not scale:

</div>


### The Problem with Static Prompts

A static prompt is a fully-written condition stack stored as a string. It works, but it does not scale:

- If the jurisdiction changes, someone must remember to edit the prompt
- If the same template applies to multiple users with different profiles, you need N versions
- If a condition changes based on runtime data (the user's subscription tier, the current date, a database lookup), you cannot encode that in a static string

### The Template Solution

A parameterized template separates **structure** from **values**. The structure is the 6-layer condition stack. The values are injected at call time.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
template = ConditionStack(
    layer_1_jurisdiction="{entity_type} under {regulatory_regime}",
    layer_2_time="As of {query_date}, {process_stage}",
    layer_3_objective="Objective: {primary_objective}",
    layer_4_constraints="Constraints: {constraint_list}",
    layer_5_facts="{facts_block}",
    layer_6_output="Return: {output_format}"
)

prompt = template.fill(
    entity_type="Series A startup",
    regulatory_regime="SEC Regulation D",
    query_date="March 2026",
    process_stage="post-close, 15 days out",
    primary_objective="determine disclosure obligations",
    constraint_list="no outside legal review budget; timeline under 30 days",
    facts_block="Raised $4.2M from 12 accredited investors. No state blue-sky exemptions filed.",
    output_format="numbered action list with deadlines"
)
```

</div>
</div>

The template enforces that all six layers are always present. A missing layer raises an error at fill time, not at response time. This is the difference between a compile-time error and a runtime error.

### Template Design Principles

**Make layers explicit, not implicit.** Do not embed Layer 2 inside Layer 1. Each layer is a named parameter. This forces the person filling the template to think about each one separately.

**Use typed defaults for optional conditions.** Some conditions are context-dependent — they should have sensible defaults that can be overridden, not blanks that get silently skipped.

**Name templates by use case, not by domain.** `sec_regulation_d_disclosure_check` is a better name than `legal_template_v3`. Names encode purpose; version numbers encode history.

---

## Pattern 2: Dynamic Condition Injection

### Static Templates Are Not Enough
<div class="callout-key">

<strong>Key Point:</strong> A parameterized template still requires someone to fill the parameters. At scale, that "someone" is a system — a database, a user profile service, an API response, or a real-time data feed.

</div>


A parameterized template still requires someone to fill the parameters. At scale, that "someone" is a system — a database, a user profile service, an API response, or a real-time data feed.

Dynamic condition injection means the system automatically pulls conditions from upstream sources and fills the template before the Claude API call.

### The Injection Architecture

```
[User Request]
      │
      ▼
[ConditionInjector]
      │
      ├── user_profile_db → entity_type, regulatory_regime, constraint_list
      ├── calendar_api    → query_date, process_stage
      └── facts_extractor → facts_block from uploaded document or form input
      │
      ▼
[ConditionStack.fill(injected_params)]
      │
      ▼
[Claude API]
```

### What Gets Injected From Where

| Layer | Typical Source |
|-------|---------------|
| Layer 1: Jurisdiction | User profile (organization type, geography, regulatory tier) |
| Layer 2: Time | System clock, process state machine, calendar service |
| Layer 3: Objective | User action (what they clicked, what form they submitted) |
| Layer 4: Constraints | User profile (subscription tier, approved-use restrictions, budget flags) |
| Layer 5: Facts | The actual user input, a form submission, or a document upload |
| Layer 6: Output | User preference settings, downstream system requirements |

### A Concrete Injector


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
class ConditionInjector:
    """
    Pulls condition values from multiple data sources
    and assembles them into a parameter dict for ConditionStack.fill().
    """
    def __init__(self, user_profile, system_context):
        self.user_profile = user_profile
        self.system_context = system_context

    def build_params(self, user_input: str) -> dict:
        return {
            "entity_type": self.user_profile["org_type"],
            "regulatory_regime": self.user_profile["primary_regulation"],
            "query_date": self.system_context["current_date"],
            "process_stage": self.system_context["workflow_stage"],
            "primary_objective": self._infer_objective(user_input),
            "constraint_list": self._get_constraints(),
            "facts_block": user_input,
            "output_format": self.user_profile.get("preferred_output", "numbered list")
        }

    def _infer_objective(self, user_input: str) -> str:
        # In practice: keyword extraction, intent classifier, or structured form field
        return "assess risk and recommend action"

    def _get_constraints(self) -> str:
        tier = self.user_profile.get("subscription_tier", "standard")
        base = "No recommendations outside platform scope."
        if tier == "standard":
            return base + " Keep recommendations within available self-service options."
        return base
```

</div>
</div>

This pattern removes the human from the loop for Layer 1, 2, and 4 conditions. The user still provides Layer 5 (facts). The system handles the rest.

### Why This Matters for Posterior Quality

Without injection, Layer 1–4 conditions depend on the user's willingness to specify them. Most users will not. They provide facts (Layer 5) and expect the model to infer the rest.

With injection, Layer 1–4 conditions are provided reliably regardless of what the user types. The posterior is constrained by the system, not by the user's prompting skill.

---

## Pattern 3: A/B Testing Condition Stacks

### What A/B Testing Means Here
<div class="callout-insight">

<strong>Insight:</strong> In prompt engineering, A/B testing does not mean changing random words. It means **systematically varying one condition at a time** and measuring whether the output changes in the predicted direction.

</div>


In prompt engineering, A/B testing does not mean changing random words. It means **systematically varying one condition at a time** and measuring whether the output changes in the predicted direction.

This is the scientific method applied to prompts. You are testing hypotheses about which conditions constrain the posterior most.

### The Correct A/B Structure

Wrong approach:
- Version A: "Write a professional email"
- Version B: "Write a concise professional email with a clear call to action"

These differ in too many ways to isolate what changed.

Correct approach:
- Stack A: Layer 3 objective = "minimize ambiguity about next steps"
- Stack B: Layer 3 objective = "maximize relationship warmth while moving to action"
- All other layers identical

This tests one condition: the objective function. The measurement question becomes: does changing the objective produce a measurably different output distribution?

### The A/B Protocol

```
1. Define the test variable: which layer, which condition
2. Write Stack A and Stack B differing only on that variable
3. Select N test inputs (at least 20 for statistical power)
4. For each input: run Stack A × 3 times, Stack B × 3 times
5. Score outputs on the target metric (see guide 02)
6. Compare distribution of scores: Stack A vs Stack B
7. Declare a winner only if the difference exceeds your noise floor
```

### What You Are Not Testing

A/B testing prompts does not test:
- Model capability (you are not comparing models)
- Output correctness (you need a ground truth for that)
- User preference (you need user feedback for that)

It tests: does one condition stack produce a more constrained, consistent posterior than the other? That is a measurable, falsifiable question.

---

## Pattern 4: Measuring Prompt Quality

### Why You Need a Metric
<div class="callout-warning">

<strong>Warning:</strong> Without a metric, prompt improvement is subjective. "This feels better" is not actionable at scale. You need a number — even a rough one — to know whether a change helped.



Without a metric, prompt improvement is subjective. "This feels better" is not actionable at scale. You need a number — even a rough one — to know whether a change helped.

### Output Stability as a Proxy for Posterior Precision

The cleanest measure of prompt quality is **output stability**: run the same prompt N times (N ≥ 5) and measure how similar the outputs are to each other.

The theoretical basis: a well-constrained posterior has low entropy. Low-entropy distributions produce similar samples. Similar samples mean similar outputs across runs.

**High stability** → narrow posterior → the prompt gave the model enough conditions to reason inside a well-specified world.

**Low stability** → wide posterior → the model is sampling from a broad distribution, which means the conditions are underspecified.

### Stability Metrics

| Metric | What It Measures | How to Compute |
|--------|-----------------|----------------|
| **Vocabulary overlap** | Do runs use similar words? | Jaccard similarity between word sets across N runs |
| **Length variance** | Do runs vary in depth? | Standard deviation of token count |
| **Key entity consistency** | Do runs mention the same specific terms? | Fraction of key entities appearing in all N runs |
| **Structure consistency** | Do runs use the same output format? | Compare heading count, list presence, paragraph structure |

None of these require ground truth. They are all measurable with only the outputs themselves.

### Condition Sensitivity: Controlled Variation

Stability measures consistency across identical prompts. **Condition sensitivity** measures how much the output changes when you vary a specific condition.

The procedure:
1. Start with a fully-specified condition stack
2. Remove or change one condition at a time
3. Measure the stability of outputs with vs without that condition
4. Rank conditions by their sensitivity score

High sensitivity = that condition is doing real work. It is a genuine switch variable. Removing it widens the posterior.

Low sensitivity = that condition is redundant or already implied by other conditions. You can remove it without degrading quality.

---

## Pattern 5: Organizational Prompt Libraries

### The Problem with Shared Folders

Most teams "share" prompts via:
- Notion pages or Google Docs
- Slack messages with "here's the prompt I use"
- Personal notes that never get shared
- A folder called `prompts/` with files named `v1`, `v2`, `v3_FINAL`, `v3_FINAL_2`

This is not a library. It is an archive. It has no structure, no discoverability, no versioning semantics, and no effectiveness data.

### What a Prompt Library Is

A prompt library is a system where:

1. **Templates are organized by condition stack structure**, not by topic keywords
2. **Templates have versions** with changelogs
3. **Templates record effectiveness metadata** (stability scores, A/B test results, usage count)
4. **Templates are searchable by layer content**, not just by name
5. **Templates can be filled and compared** programmatically

### Library Organization by Condition Stack

The wrong organization: folders by topic ("marketing prompts," "legal prompts," "data analysis prompts"). This conflates the domain (Layer 1) with the entire template.

The right organization: condition stacks indexed by their primary switch variable. The layer that most differentiates behavior is the organizing axis.

Example structure:
```
prompt_library/
├── by_objective/
│   ├── risk_assessment/        # Layer 3: assess and rank risks
│   ├── action_generation/      # Layer 3: generate next steps
│   ├── summarization/          # Layer 3: compress to key points
│   └── comparison/             # Layer 3: evaluate options against criteria
├── by_jurisdiction/
│   ├── us_federal/
│   ├── eu_gdpr/
│   └── multi_jurisdiction/
└── by_output_format/
    ├── decision_tree/
    ├── numbered_action_list/
    └── executive_summary/
```

Templates appear in multiple indexes — a single template might be indexed under `by_objective/risk_assessment` and `by_jurisdiction/us_federal`. The indexes are views, not folders.

### Versioning Semantics

Version numbers encode the nature of the change:

| Change Type | Version Impact | Example |
|------------|---------------|---------|
| Layer 1 changed (jurisdiction) | Major (2.0.0 → 3.0.0) | Added EU GDPR scope |
| Layer 3 changed (objective) | Major | Changed from "minimize risk" to "identify risk" |
| Layer 6 changed (output format) | Minor (2.0.0 → 2.1.0) | Changed from prose to numbered list |
| Wording clarification, same conditions | Patch (2.1.0 → 2.1.1) | Reworded constraint for clarity |

This gives teams a signal about how much a template change might affect downstream outputs. Major version changes should trigger re-testing.

---

## Putting It Together: The Production Workflow

```
Developer creates ConditionStack template
         │
         ▼
Template registered in PromptLibrary (v1.0.0)
         │
         ▼
ConditionInjector wired to data sources
         │
         ▼
PromptTester runs stability baseline (N=10 runs)
         │
         ▼
Stability score stored in library metadata
         │
         ▼
A/B test against existing template (if replacing)
         │
         ▼
Winner deployed; loser archived with results
         │
         ▼
Periodic stability monitoring in production
```

---

## Common Pitfalls

**Pitfall 1: Treating prompt templates as documentation.**
Templates stored in docs are not executable, not testable, and not measurable. Templates should live in code.

**Pitfall 2: Versioning prompts by date.**
`prompt_2026_03_24.txt` tells you when it was created, not what changed. Use semantic versioning with a changelog.

**Pitfall 3: A/B testing by subjective quality.**
"This one reads better" is not a metric. Stability scores and condition sensitivity are measurable. Use them.

**Pitfall 4: Building a library organized by topic, not by condition structure.**
When you need a prompt for "EU GDPR data breach notification" you should be searching by Layer 1 (EU GDPR) and Layer 3 (notification/disclosure), not by the keyword "GDPR."

**Pitfall 5: Injecting conditions that override user-specified facts.**
Layer 5 (facts) should come from the user. Injecting pre-defined facts overrides the actual situation and defeats the purpose of Bayesian conditioning.

---

## Connections

- **Builds on:** Module 3 (the 6-layer condition stack), Module 6 (detecting and fixing probability mistakes)
- **Leads to:** The capstone project (build a complete production system for your domain)
- **Related to:** Software template patterns, A/B testing methodology, semantic versioning

---

## Practice

1. Take a prompt you currently use manually. Identify which conditions are always the same (candidates for the template) and which vary per call (candidates for injection).

2. Run the same prompt 5 times on the same input. Count the number of distinct recommendations across runs. If the count is greater than 2, the posterior is underspecified.

3. Design a library structure for a team of 10 people using AI tools for a single domain (your choice). What are the three primary indexing axes you would use?

---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Key Insight" and why it matters in practice.

2. Given a real-world scenario involving production patterns: bayesian prompting at scale, what would be your first three steps to apply the techniques from this guide?


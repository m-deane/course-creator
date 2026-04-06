# Conditional Trees: When One Answer Is Wrong

> **Reading time:** ~10 min | **Module:** 4 — Conditional Trees | **Prerequisites:** Module 3 Condition Stack


## In Brief

Many questions that look like they have a single correct answer secretly contain a decision tree. Forcing a flat answer collapses the tree into the most common branch — which is rarely the right branch for your specific situation. The fix is to prompt the model to surface the tree instead of hiding it.


<div class="callout-key">

<strong>Key Concept Summary:</strong> Many questions that look like they have a single correct answer secretly contain a decision tree.

</div>

---

## The Hidden Structure Problem

Consider this prompt:

> "Should I build my API in REST or GraphQL?"

This looks like a question with an answer. It is not. It is a question with a **conditional structure**:

- If your clients are unknown third parties with unpredictable query patterns → GraphQL
- If your clients are your own front-end teams and you control both sides → REST is simpler
- If you need caching at the CDN layer → REST
- If your data is highly relational and front-ends need flexible queries → GraphQL
- If your team has no GraphQL experience and you're moving fast → REST

When you ask the model for a single answer, it produces the branch that's most frequent in its training data — which often means "it depends, but probably REST" or "GraphQL is more modern." Neither is an answer. Both are the model guessing your branch.

**You don't have a missing fact problem. You have a missing structure problem.**

---

## Why Models Collapse Trees Into Verdicts

Language models are trained on human text. Human text is full of verdicts: "Use React." "TypeScript is better." "PostgreSQL scales." Humans write conclusions, not decision trees.

From a Bayesian frame:

$$P(\text{answer} \mid \text{vague question}) \approx P(\text{most common answer in training})$$

When the question doesn't specify conditions, the posterior defaults to the training prior — the most common answer across all contexts. This is the "average case" answer, which is correct for the average case and wrong for yours.

The model is not failing. It is doing exactly what the math predicts: it's giving you the highest-probability answer given what you specified. You specified almost nothing.

---

## Recognizing Hidden Decision Trees

A question contains hidden conditional structure when:
<div class="callout-warning">

<strong>Warning:</strong> A question contains hidden conditional structure when:

</div>


**1. The domain has jurisdiction or context variation**
Questions involving law, tax, regulation, or geography always have branches by jurisdiction. "Is this legal?" has different answers in Delaware vs. the UK.

**2. The question involves competing values or tradeoffs**
"What's the best approach?" implies that "best" depends on what you're optimizing for. Speed? Maintainability? Cost? Different objectives, different answers.

**3. The question involves system scale or size**
"What database should I use?" depends heavily on whether you have 1,000 users or 100 million. Answers flip at scale thresholds.

**4. The question has a stakeholder or audience dependency**
"How should I structure this presentation?" depends on whether the audience is technical, executive, skeptical, or already sold.

**5. The question is about a decision with irreversible consequences**
When stakes are high, the right answer depends on your specific risk tolerance, timeline, and fallback options.

**Diagnostic question:** If I changed one thing about the asker's situation, would the correct answer change? If yes, the question has conditional structure.

---

## The Single-Answer Failure Mode

Here is a concrete example of what goes wrong.

**The question:**
> "Should I incorporate in Canada or the U.S.?"

**Flat answer (what most prompts produce):**
> "For most startups, incorporating in Delaware, USA is recommended because of its flexible corporate law, familiarity to investors, and established legal precedents. The Delaware C-Corp is the standard structure for venture-backed companies."

This answer is technically correct. It is also wrong for:
- A company with all-Canadian customers and no U.S. expansion plans
- A company whose lead investors are Canadian pension funds with restrictions on U.S. entities
- A company whose founders are Canadian and would face significant tax complexity with U.S. incorporation
- A company operating in a regulated industry with Canadian licensing requirements

The model gave you the training prior. You needed your branch.

---

## The Conditional Tree Response

Compare the flat answer to a conditional tree response for the same question:
<div class="callout-insight">

<strong>Insight:</strong> Compare the flat answer to a conditional tree response for the same question:

</div>


**The answer as a decision tree:**

```
Should I incorporate in Canada or the U.S.?

Choose U.S. (Delaware) if:
├── You plan to raise from U.S. venture capital firms
├── Your primary market is the U.S.
├── Your founding team has U.S. residency or citizenship
└── You want maximum flexibility for future M&A exits

Choose Canada (Ontario or BC) if:
├── Your customers are primarily Canadian
├── You are eligible for SR&ED tax credits (significant R&D)
├── Your funding comes from Canadian sources (BDC, pension funds)
└── Your founding team is Canadian and wants to avoid cross-border tax complexity

Ask these questions before deciding:
├── Where are your target customers?
├── Where are your target investors?
├── Where do your founders live and pay taxes?
└── Do you plan to raise from U.S. VCs in the next 18 months?
```

This answer is longer. It is also actually useful.

---

## Prompting for the Tree

There are three techniques for getting conditional tree responses.
<div class="callout-warning">

<strong>Warning:</strong> There are three techniques for getting conditional tree responses.

</div>


### Technique 1: Explicit Branch Request

Tell the model you want branches, not a verdict.

```
Instead of:
"Should I incorporate in Canada or the U.S.?"

Use:
"I'm deciding whether to incorporate in Canada or the U.S. Don't give me a single
recommendation. Instead:
1. List the conditions that would make Canada the right choice
2. List the conditions that would make the U.S. the right choice
3. List the questions I need to answer before deciding
Then, given those conditions, tell me which ones I've already specified and what's
still unclear."
```

### Technique 2: The Meta-Prompt

Ask the model to surface the hidden conditions before answering.

```
Before answering this question, list all the conditions that would change your
answer. Be specific: for each condition, state what would change if the condition
were true vs. false.

Question: [your question here]

After listing the conditions, answer the question for each major branch.
```

This technique is powerful because it makes the model's hidden assumptions explicit. When you see the branches, you can tell the model which one you're on — and get an answer calibrated to your situation.

### Technique 3: Condition-First Specification

Specify your conditions first, then ask the narrow question that remains.

```
My situation:
- Canadian startup, two founders, both Canadian residents
- No U.S. customers yet, planning to expand in 18 months
- Seeking seed funding from Canadian angels first, U.S. VC in Series A
- SaaS product, no regulated industry

Given these conditions: should I incorporate in Canada first and redomicile to
Delaware before Series A, or incorporate in Delaware from the start?
```

Now you've told the model which branch you're on. The posterior collapses to your actual situation.

---

## The "I Need More Information" Response

One of the most valuable things a model can do is refuse to give you a verdict and ask you for the conditions it needs.

Most prompts discourage this. They ask for a recommendation and imply that uncertainty is a failure mode.

Prompt for it explicitly:

```
If you don't have enough information to give me a specific recommendation,
say so — and list exactly what information you would need. Don't give me a
general answer when a specific one is possible with more context.
```

This reframes uncertainty from a failure mode to a useful output. A model that says "I need to know X, Y, Z before I can answer specifically" is more useful than a model that says "it depends" without telling you what it depends on.

---

## Domain Examples

### Business Decision: "Which pricing model should I use?"

Hidden branches:
- Freemium → correct if you have strong viral coefficient and low marginal cost per user
- Usage-based → correct if value scales with usage and customers are sophisticated buyers
- Seat-based → correct if value is per-user and adoption breadth matters
- Enterprise flat-fee → correct if your buyers are large and budget-constrained

The flat answer — "freemium works well for SaaS" — is correct for some branch. The question is which branch you're on.

### Code Architecture: "Should I use microservices or a monolith?"

Hidden branches:
- Monolith → correct for early-stage teams under 10 engineers, single product, fast iteration
- Microservices → correct when you have >5 independent teams with distinct deployment cadences
- Modular monolith → correct when you need team independence but aren't ready for distributed systems overhead

The flat answer — "start with a monolith, refactor to microservices when needed" — is training prior. It's often right. It's not always right.

### Compliance Question: "Does GDPR apply to my startup?"

Hidden branches:
- Yes, fully → if you have EU customers or process EU personal data
- Partially → if you process EU data on behalf of EU controllers
- Not directly → if all data subjects are outside the EU and you have no EU operations
- Unclear → if you have indirect EU data exposure through third-party integrations

The flat answer — "yes, GDPR applies if you have EU users" — is technically correct and practically incomplete.

---

## Common Pitfalls

**Pitfall 1: Asking for branches without specifying depth**
"List the conditions" without constraint produces a list of 20 conditions, most of which are irrelevant. Specify the major branches: "List the 3-5 conditions that would most change your recommendation."

**Pitfall 2: Mistaking exhaustiveness for usefulness**
A decision tree with 47 branches is not better than one with 5 major branches. Ask for the conditions with the highest decision leverage.

**Pitfall 3: Forgetting to close the loop**
After the model produces a tree, go back and tell it which branch you're on. The tree is step one. The calibrated answer is step two.

**Pitfall 4: Using conditional prompts for simple factual questions**
"What is the capital of France?" does not have conditional structure. Conditional tree prompting is for decisions, recommendations, and analysis — not lookup.

---

## Connections

- **Builds on:** Module 2 (Switch Variables) — the conditions that create branches are switch variables
- **Builds on:** Module 3 (Condition Stack) — specifying your branch means filling in your condition stack
- **Leads to:** Module 5 (Agents and Workflows) — agents can execute decision trees dynamically
- **Related to:** Module 6 (Probability Mistakes) — flat answers are a symptom of base rate neglect

---

## Practice Problems

1. Take a question you recently asked an AI and identify its hidden conditional structure. How many branches does it have? What are the switch variables?

2. Write a meta-prompt for this question: "Should I raise venture capital or bootstrap my startup?" Include the instruction to list conditions before answering.

3. Find an example of a "flat answer" you received from an AI that was technically correct but wrong for your situation. Identify which branch the model was on and which branch you were on.

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The Hidden Structure Problem" and why it matters in practice.

2. Given a real-world scenario involving conditional trees: when one answer is wrong, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Kahneman, *Thinking Fast and Slow*, Chapter 14: "Tom W's Specialty" — on base rate neglect and the failure to update on specific evidence
- Pearl, *The Book of Why*, Chapter 1: Introduction to causal reasoning and conditional structure
- Klein, *Sources of Power*, Chapter 1: How experts use recognition-primed decision making rather than single verdicts

---

## Cross-References

<a class="link-card" href="../notebooks/01_decision_tree_prompts.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

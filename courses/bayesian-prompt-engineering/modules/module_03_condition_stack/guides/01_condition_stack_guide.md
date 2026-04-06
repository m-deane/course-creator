# Guide 1: The Condition Stack Framework

> **Reading time:** ~14 min | **Module:** 3 — Condition Stack | **Prerequisites:** Module 2 Switch Variables


## In Brief

The Condition Stack is a 6-layer protocol for writing prompts. It specifies conditions in descending order of leverage — from the constraints that eliminate the most branches of the model's prior (jurisdiction, time, objective) down to the facts most people start with. Starting at Layer 5 (facts) is the most common prompting mistake. The layers above it do more work per word than any amount of factual detail.


<div class="callout-key">

<strong>Key Concept Summary:</strong> The Condition Stack is a 6-layer protocol for writing prompts.

</div>

---

## The Central Problem: Why Most Prompts Fail

Here is a prompt a professional wrote after thinking carefully about it:
<div class="callout-key">

<strong>Key Point:</strong> Here is a prompt a professional wrote after thinking carefully about it:

</div>


> "My client is a self-employed consultant who made $180,000 last year. She has a home office, pays for her own health insurance, and contributed to a SEP-IRA. What deductions should she take?"

This looks like a detailed prompt. It has numbers, categories, and a specific person. But it is missing four layers of information the model needs to give a useful answer:

- Which country and state? (Tax law differs by jurisdiction.)
- What year? What stage of the process? (Planning before December 31 vs. filing in April vs. amending a prior return — different answers.)
- What is she optimizing for? (Minimize current-year tax? Minimize audit risk? Maximize retirement savings? These lead to different deduction strategies.)
- What is off the table? (Does she have a W-2 job that complicates things? Is she under audit?)

Without those four layers, the model reasons from its prior: the average case of all consultants in all jurisdictions in all years at all stages of the process. That average is not her case.

The Condition Stack fixes this by making the high-leverage conditions explicit — before you mention a single fact.

---

## The 6 Layers

### Layer 1: Jurisdiction + Rule Set
<div class="callout-insight">

<strong>Insight:</strong> **What it is:** Which country, state/province, and regulatory regime governs this situation? Which specific ruleset applies?

</div>


**What it is:** Which country, state/province, and regulatory regime governs this situation? Which specific ruleset applies?

**Why it has the highest leverage:** Jurisdiction is a hard filter. "Tax law" is not a single thing — it is thousands of things. "US federal tax law for LLCs" is a 10x narrower world than "tax law." "US federal tax law for single-member LLCs under the Tax Cuts and Jobs Act" is narrower still. Every jurisdiction condition eliminates branches that cannot possibly apply to your situation.

**Examples:**
- Legal: "California, Superior Court, civil procedure for breach of contract under California Commercial Code"
- Medical: "US guidelines, adult patient, outpatient primary care setting, under Medicare Part B billing rules"
- Software: "AWS, production environment, Python 3.11, FastAPI, SOC 2 Type II compliance requirements"
- Finance: "FINRA-regulated broker-dealer, US retail customer, Rule 2111 suitability standard"

**The test:** Could a different jurisdiction produce a meaningfully different answer? If yes — and it almost always is — specify jurisdiction first.

---

### Layer 2: Time + Procedural Posture

**What it is:** What year (or date range) applies? Where are you in the process — planning, executing, filing, amending, appealing, responding?

**Why it matters:** The same facts produce different answers depending on when and where in a process you are. A tax question asked on December 15 has different actionable answers than the same question asked on April 10 — one is about planning actions, the other is about what can still be done before a filing deadline. A medical question about a patient in triage is different from the same question about a patient in a follow-up visit.

**The mistake:** Leaving time implicit. The model defaults to the most common case, which is often not your case.

**Examples:**
- Tax: "Tax year 2024, currently in planning phase, filing deadline is April 15, 2025 — still time to act"
- Legal: "Contract dispute, pre-litigation phase, no complaint filed yet, 2-year statute of limitations expires in 8 months"
- Software: "System is currently in production; change must be zero-downtime; deploy window is tonight at 2am"
- Medical: "Patient presenting for initial visit; no prior workup; results needed before specialist referral in 3 days"

---

### Layer 3: Objective Function

**What it is:** What does "a good answer" mean in this situation? What are you trying to maximize, minimize, or achieve? What trade-offs are acceptable?

**Why it matters:** This is the layer that produces the most dramatically different outputs from identical facts. "Minimize current-year tax liability" and "minimize audit risk" are both valid tax objectives — and they often produce opposite recommendations. A strategy that aggressively uses deductions may minimize tax while increasing audit risk. A model with no objective function specified must guess which one you want.

**The mistake:** Assuming "good advice" is obvious. In any professional domain, there are multiple legitimate objectives that trade against each other. Specify which one you are optimizing.

**Examples:**
- Tax: "Primary objective: minimize total tax burden over 3 years. Secondary: keep audit risk low (client is risk-averse). No interest in aggressive positions."
- Legal: "Objective: maximize probability of winning at trial, not settlement. Client will not accept a settlement that implies liability."
- Software architecture: "Objective: minimize time-to-production for the MVP. Not optimizing for scalability yet. Will refactor after product-market fit."
- Medical: "Objective: rule out dangerous diagnoses first. Patient is a poor historian; prioritize high-sensitivity tests over high-specificity tests at this stage."

---

### Layer 4: Constraints

**What it is:** What strategies, approaches, or options are off the table? What are the hard limits on budget, time, risk, or format?

**Why it matters:** Without explicit constraints, the model will surface options that may be technically valid but practically impossible. A tax strategy that requires a corporate restructuring is not useful advice for a sole proprietor who cannot execute it. Constraints eliminate a large portion of the solution space before the model generates it — which produces shorter, more actionable output.

**Examples:**
- Tax: "Constraints: no corporate restructuring; client wants to remain a sole proprietor; cannot contribute more than $10,000 to retirement accounts this year; must file on time (no extensions)"
- Legal: "Constraints: litigation budget is $50,000 maximum; client cannot appear in court in person; must resolve within 12 months"
- Software: "Constraints: no new infrastructure; must run on existing AWS stack; team of 2 engineers; no TypeScript — Python only"
- Medical: "Constraints: patient is allergic to penicillin and codeine; no MRI available at this facility; patient is uninsured — cost sensitivity required"

---

### Layer 5: Facts

**What it is:** The numbers, timeline, documents, events, and specific details of the situation.

**Why most people start here (and shouldn't):** Facts are the most visible part of a situation. They are concrete and specific. But without Layers 1–4, facts are just data points in an unconstrained space. The model must infer the context they belong to — and it will infer the average case.

With Layers 1–4 set, facts become maximally informative. The model knows exactly what universe it is reasoning in, what it is trying to achieve, and what options are excluded. The same facts produce far better answers.

**Examples:**
- Tax: "$180,000 gross income, $12,000 home office expense (300 sq ft / 2,500 sq ft total), $8,400 health insurance premiums paid out-of-pocket, $25,000 SEP-IRA contribution made, $4,200 in professional development courses, 1099-NEC income from 3 clients"
- Medical: "45-year-old male, BMI 28, presenting with 3-week history of progressive dyspnea on exertion, mild chest tightness, no fever, vitals: BP 138/88, HR 94, SpO2 96% on room air"
- Software: "Node.js API, 200 req/sec peak, P99 latency currently 2.1 seconds, database is PostgreSQL 14, bottleneck identified in a single query that does 3 sequential joins on tables of 50M+ rows"

---

### Layer 6: Output Specification

**What it is:** What form should the answer take? Bullets or narrative? Table or prose? "List your assumptions first." "Show your reasoning." "Give me three options ranked by my objective function."

**Why it matters:** Layer 6 is the lowest-leverage layer but the most-used by prompt tutorials. It matters — a well-specified output format reduces post-processing and makes the answer easier to act on — but it does almost nothing to improve the quality of the underlying reasoning. That is what Layers 1–5 do.

**Examples:**
- "Begin by listing all assumptions you are making. Then give recommendations as a numbered list ordered from highest to lowest impact on the primary objective. Flag any recommendation that conflicts with a secondary objective. End with open questions that would materially change the analysis."
- "Return a decision table: rows are options, columns are trade-offs against my stated objectives. Add a recommended row at the bottom."
- "Write this as a clinical note in SOAP format."
- "Give me the code first, then a brief explanation of the key design decision, then a list of edge cases this implementation does not handle."

---

## Why People Start at Layer 5

The instinct to lead with facts is rational. Facts are what you know. They are concrete. They feel like the "substance" of the question. The framing conditions (jurisdiction, objective, constraints) feel like preamble — like housekeeping you should be able to skip.
<div class="callout-warning">

<strong>Warning:</strong> The instinct to lead with facts is rational. Facts are what you know. They are concrete. They feel like the "substance" of the question. The framing conditions (jurisdiction, objective, constraints) feel like preamble — like housekeeping you should be able to skip.

</div>


But from the model's perspective, the framing conditions are the substance. The facts are data. Without the frame, the model cannot interpret the data.

Consider the analogy of a medical test. A positive COVID test is not the same information in all contexts:
- In a healthy 25-year-old with mild symptoms: mostly reassuring, monitor at home
- In an 80-year-old immunocompromised patient: urgent, escalate immediately
- In a patient being evaluated for a clinical trial: potentially disqualifying

The fact (positive test) has no meaning without the context. The same is true of any factual prompt. The "context" is exactly what Layers 1–4 specify.

---

## The Plug-and-Play Prompt Template

Copy this template. Fill in each layer. Remove the bracketed instructions before sending.
<div class="callout-key">

<strong>Key Point:</strong> Copy this template. Fill in each layer. Remove the bracketed instructions before sending.

</div>


```
[LAYER 1 — JURISDICTION + RULE SET]
Domain: [field of knowledge]
Jurisdiction: [country / state-province / regulatory body]
Rule set: [specific law, standard, framework, guideline, or protocol that governs this]

[LAYER 2 — TIME + PROCEDURAL POSTURE]
Relevant time period: [year(s), date range, or deadline]
Current stage of process: [planning / filing / executing / appealing / diagnosing / designing / etc.]
Time-sensitivity: [any deadlines or action windows that affect recommendations]

[LAYER 3 — OBJECTIVE FUNCTION]
Primary objective: [what to maximize or minimize — be specific about the trade-off direction]
Secondary objectives: [additional goals, in priority order]
Acceptable trade-offs: [what you are willing to sacrifice for the primary objective]

[LAYER 4 — CONSTRAINTS]
Hard constraints (off the table): [strategies, options, or approaches that cannot be used]
Resource constraints: [budget, time, team size, or capability limits]
Risk tolerance: [conservative / moderate / aggressive — with any specific risk limits]

[LAYER 5 — FACTS]
[Present all relevant numbers, timeline, documents, events, and specifics here]

[LAYER 6 — OUTPUT SPECIFICATION]
Format: [bullets / narrative / table / code / clinical note / etc.]
Required sections: [list any required sections]
Special instructions: [list assumptions first / show reasoning / rank by objective / etc.]
Length: [approximate target length or "be concise"]
```

---

## Before and After: The Same Question with and without the Stack
<div class="callout-insight">

<strong>Insight:</strong> **Without the Condition Stack (Layer 5 only):**

</div>


**Without the Condition Stack (Layer 5 only):**

> "My client is a self-employed consultant who made $180,000 last year. She has a home office, pays for her own health insurance, and contributed to a SEP-IRA. What deductions should she take?"

A model given this will produce: a generic list of common self-employment deductions. Accurate but not specific to her jurisdiction, not calibrated to her objective, not filtered by her constraints. The answer covers the average case.

**With the Condition Stack (all 6 layers):**

```
[LAYER 1]
Domain: US federal and California state income tax
Jurisdiction: United States federal (IRS rules) + California (FTB rules)
Rule set: Tax Cuts and Jobs Act (TCJA) as currently in effect for 2024; California conformity rules

[LAYER 2]
Relevant time period: Tax year 2024
Current stage: Planning phase — December 2024, before year-end
Time-sensitivity: Actions taken before December 31 can affect 2024 tax year;
                  retirement contribution deadline is tax filing date (April 15, 2025 with extension)

[LAYER 3]
Primary objective: Minimize total federal + California state income tax for 2024
Secondary: Keep audit risk low (client is risk-averse, has never been audited, wants to stay that way)
Acceptable trade-offs: Willing to accept slightly higher tax to significantly reduce audit risk

[LAYER 4]
Hard constraints: No corporate restructuring (client wants to remain a sole proprietor);
                  cannot move residence out of California this year;
                  must file on time — no extensions
Resource constraints: Can contribute up to $15,000 more to SEP-IRA if needed

[LAYER 5]
$180,000 gross consulting income (1099-NEC from 3 clients)
Home office: 300 sq ft dedicated office / 2,500 sq ft total home
Health insurance premiums: $8,400/year paid entirely out-of-pocket
SEP-IRA: $25,000 contributed so far this year
Professional development: $4,200 in courses and conference fees
Equipment: $3,200 in new laptop and peripherals purchased in October
Business vehicle: no (uses personal car occasionally, has mileage log)

[LAYER 6]
Begin by stating what you are assuming about California conformity to federal deductions.
Then give a prioritized list of deductions, sorted by dollar impact (high to low).
For each deduction: amount, confidence level (certain / likely / risky), and audit-risk assessment.
Flag any deduction where California and federal treatment differ.
End with two specific actions she should take before December 31.
```

The output from the stacked version is actionable, California-specific, calibrated to her risk tolerance, filtered for actionable-before-year-end timing, and structured so she can immediately identify the two things to do right now. The output from the unstacked version is a Wikipedia article on self-employment deductions.

---

## Common Pitfalls

**Pitfall 1: Writing the jurisdiction in the question instead of the stack**
"What are the California tax rules for home office deductions?" — this puts the jurisdiction in the question but not in a structured way the model can use to constrain the entire response. Put it in Layer 1 so it governs everything.
<div class="callout-warning">

<strong>Warning:</strong> **Pitfall 1: Writing the jurisdiction in the question instead of the stack**

</div>


**Pitfall 2: Conflating objective with constraints**
"I want to minimize tax but I don't want to be too aggressive" — this mixes an objective (minimize tax) with a constraint (not aggressive). Separate them: Layer 3 = minimize tax; Layer 4 = risk tolerance is low, no aggressive positions.

**Pitfall 3: Leaving the objective implicit**
"What's the best approach?" — "best" is not an objective function. Best by what criterion? In professional domains, "best" is always a trade-off and you must specify which side of the trade-off you are on.

**Pitfall 4: Over-specifying Layer 6**
Spending 3 sentences on formatting while leaving Layer 3 empty. Format instructions improve presentation. Condition layers improve reasoning. Reasoning is more valuable.

**Pitfall 5: Using the template as a form to fill**
The template is a thinking tool, not a form. Some layers have one sentence; some have five. Use judgment. The question to ask for each layer: "Would a different value here produce meaningfully different advice?" If yes, specify it. If no, skip it.

---

## Connections

- **Builds on:** Module 1 (P(A|C) frame — the layers are the C in P(A|C)); Module 2 (Switch Variables — Layer 1 and Layer 3 are the most common switch variable locations)
- **Leads to:** Module 4 (Conditional Trees — when Layer 1 or Layer 3 cannot be specified in advance and you need a decision tree)
- **Related to:** Module 7 (Production Patterns — dynamic injection of condition stack layers into API calls at runtime)

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The Central Problem: Why Most Prompts Fail" and why it matters in practice.

2. Given a real-world scenario involving guide 1: the condition stack framework, what would be your first three steps to apply the techniques from this guide?

</div>

## Further Reading

- Kahneman, *Thinking Fast and Slow* (Chapter 11): Base rates and how ignoring context leads to inside-view errors — the same mechanism as starting with Layer 5
- Tversky & Kahneman (1974), "Judgment under Uncertainty: Heuristics and Biases": The original research on why humans default to inside-view (facts) and miss outside-view (context/priors)
- Anthropic's Claude usage documentation: System prompts as persistent condition stacks

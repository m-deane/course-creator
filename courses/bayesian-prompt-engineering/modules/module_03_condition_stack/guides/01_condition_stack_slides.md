---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# The Condition Stack Framework

### Module 3 — Bayesian Prompt Engineering

**6 layers. One ordering principle. A qualitatively different class of output.**

<!-- Speaker notes: Welcome to Module 3. By now students understand that prompts are evidence that shapes the model's posterior. This module gives them a specific, repeatable protocol for writing evidence-dense prompts. The condition stack is the central practical tool of the entire course. -->

---

## The Problem We're Solving

You write a detailed prompt. You get a generic answer.

More detail does not fix it.

> "My client is a self-employed consultant who made $180,000 last year. She has a home office, pays for her own health insurance, and contributed to a SEP-IRA. What deductions should she take?"

**What's missing?** Which country? Which state? What year? What stage of the process? What is she optimizing for? What is off the table?

The model fills those gaps with its **training prior** — the average case.

Your case is not the average case.

<!-- Speaker notes: Start with the concrete example. Students should immediately recognize this pattern — they have written prompts like this. The point is not that the prompt lacks detail. It has plenty of detail. It is missing the conditions that make that detail interpretable. -->

---

## The Root Cause: Starting at Layer 5

```
The 6 Layers (highest to lowest leverage)
─────────────────────────────────────────
  Layer 1: Jurisdiction + Rule Set        ← Most people skip this
  Layer 2: Time + Procedural Posture      ← Most people skip this
  Layer 3: Objective Function             ← Most people skip this
  Layer 4: Constraints                    ← Most people skip this
─────────────────────────────────────────
  Layer 5: Facts  ← WHERE MOST PROMPTS START
─────────────────────────────────────────
  Layer 6: Output Specification           ← Where prompt tutorials focus
```

Starting at Layer 5 means the model must infer Layers 1–4 from its prior.

In high-stakes domains, that prior is built on **the median case**.

<!-- Speaker notes: This diagram is the key visual of the module. Emphasize the contrast: most prompt advice focuses on Layer 6 (formatting, structure). The actual leverage is in Layers 1-4. Facts alone give the model data without interpretation context. -->

---

## The Stack as a Probability Machine

$$P(A \mid \underbrace{C_1, C_2, C_3, C_4}_{\text{Layers 1–4}}, \underbrace{C_5}_{\text{Facts}}, \underbrace{C_6}_{\text{Format}})$$

Each layer above Layer 5 **multiplies** the constraint on the posterior.

| Without stack | $P(A \mid \text{facts only})$ | Model guesses jurisdiction, time, objective |
|---|---|---|
| With stack | $P(A \mid C_1 \cap C_2 \cap C_3 \cap C_4 \cap C_5)$ | Model is pinned to your specific world |

This is not metaphor. This is the mechanism by which the model generates tokens — each token conditions on all prior context.

<!-- Speaker notes: Connect back to Module 1. The P(A|C) frame is already established. Now we are giving the C a structure. The stack is just a systematic way to fill in C with the conditions that have the most leverage. -->

---

## Layer 1: Jurisdiction + Rule Set

**The highest-leverage condition. Specify it first, always.**

```
Domain: US federal and California state income tax
Jurisdiction: United States federal (IRS) + California (FTB)
Rule set: Tax Cuts and Jobs Act (TCJA) as in effect for 2024;
          California conformity rules apply
```

**Why first:** Jurisdiction is a hard filter.

- "Tax law" → millions of possible answers
- "US federal tax law" → ~10x narrower
- "US federal tax law for sole proprietors under TCJA 2024" → narrower still

Every jurisdiction condition eliminates branches that **cannot apply** to your situation.

<!-- Speaker notes: The multiplicative nature of constraint is important here. Jurisdiction doesn't add 1 unit of precision — it multiplies the constraint by a large factor. In legal and medical domains especially, the jurisdiction can completely change which options even exist. -->

---

## Layer 1: Jurisdiction Across Domains

<div class="columns">

**Professional Domains**

- Tax: "US federal + California, TCJA 2024"
- Legal: "California Superior Court, civil procedure, breach of contract under Cal. Commercial Code"
- Medical: "US outpatient guidelines, adult primary care, Medicare Part B billing"
- Finance: "FINRA-regulated broker-dealer, Rule 2111 suitability"

**Technical Domains**

- Software: "AWS production, Python 3.11, FastAPI, SOC 2 Type II compliance"
- Data: "GDPR jurisdiction, EU resident data, no data leaving EEA"
- Security: "PCI DSS 4.0, Level 2 merchant, Visa/Mastercard processing"
- Infrastructure: "Kubernetes 1.28, GKE, multi-region active-active"

</div>

<!-- Speaker notes: The concept of "jurisdiction" extends beyond legal meaning to any "which ruleset governs this" situation. In software, it's the regulatory and technical environment. In medicine, it's guidelines body + care setting + payer rules. Help students map this to their own domain. -->

---

## Layer 2: Time + Procedural Posture

**Same facts. Different stage. Different advice.**

```
Tax year: 2024
Current stage: Planning phase, December 2024, before year-end
Time-sensitivity: Actions before December 31 affect 2024;
                  SEP-IRA deadline extends to April 15, 2025
```

**The procedural posture question:** Where are you in the process?

| Stage | What changes |
|---|---|
| Pre-filing (Dec) | Still time to act; planning advice |
| At-filing (Apr) | What's still possible; extension decision |
| Post-filing | Amended return options; audit posture |
| Under audit | Defense strategy; documentation priority |

<!-- Speaker notes: The tax example is vivid because students can feel the difference between December advice and April advice on the same situation. The December answer includes "here are things you can still do." The April answer is about what already happened. Same facts, completely different posture. -->

---

## Layer 3: Objective Function

**The layer that produces the most dramatically different outputs from identical facts.**

```
Primary objective: Minimize total federal + CA state income tax for 2024
Secondary: Keep audit risk low (client is risk-averse)
Acceptable trade-offs: Accept slightly higher tax to significantly reduce audit risk
```

**Why it matters:**

| Objective | Recommended strategy |
|---|---|
| Minimize current-year tax | Aggressive deductions, maximum retirement contributions |
| Minimize audit risk | Conservative deductions, well-documented positions only |
| Maximize compliance certainty | Standard positions, no grey areas |
| Minimize total 3-year burden | May pay more this year to harvest losses next year |

Same facts → four different recommendation sets.

<!-- Speaker notes: This is the slide that usually gets the biggest reaction. Students realize they have been writing prompts with no objective function and getting "all-of-the-above" answers that hedge between objectives. The model is not wrong to hedge — it genuinely doesn't know which objective you prefer. -->

---

## Layer 4: Constraints

**Eliminate the impossible before generating the answer.**

```
Hard constraints:
  - No corporate restructuring (must remain sole proprietor)
  - Cannot change state of residence
  - Must file on time — no extensions

Resource constraints:
  - Can contribute up to $15,000 more to SEP-IRA
  - No additional capital available for asset purchases

Risk tolerance: Low — no aggressive positions
```

**The constraint test:** If a strategy requires something you cannot do, and the model doesn't know that, it will recommend that strategy.

Constraints prevent wasted output — both in the response and in your follow-up.

<!-- Speaker notes: Constraints are about eliminating the impossible and the impractical before generation. A good analogy: a surgeon who knows the patient can't have anesthesia thinks about the problem differently from the start. Constraints shape the solution space, not just filter the output. -->

---

## Layer 5: Facts — Now They Work

**With Layers 1–4 set, facts become maximally informative.**

```
$180,000 gross consulting income (1099-NEC, 3 clients)
Home office: 300 sq ft / 2,500 sq ft total home
Health insurance premiums: $8,400/year, out-of-pocket
SEP-IRA contributed: $25,000 this year
Professional development: $4,200 (courses + conference)
Equipment: $3,200 new laptop + peripherals (October purchase)
Vehicle: personal car, occasional business use, mileage log kept
```

The model now interprets each fact **inside the correct world:**
- California conformity rules apply to the home office deduction
- The equipment purchase timing ($3,200 in October) triggers Section 179 analysis
- The mileage log matters because California has different mileage rules than federal

<!-- Speaker notes: Walk through how each fact gets interpreted differently with the context. The equipment purchase in October is not just "$3,200" — it's a Section 179 depreciation decision that must be weighed against California conformity (CA does not fully conform to federal Section 179 limits). Without Layer 1, the model doesn't know to flag this. -->

---

## Layer 6: Output Specification

**Lowest leverage. Most discussed in prompt tutorials.**

```
Begin by listing all assumptions about California conformity.
Then: numbered list of deductions, sorted by dollar impact (high to low).
For each: amount | confidence (certain/likely/risky) | audit-risk flag.
Flag any deduction where CA and federal treatment differ.
End with two specific actions to take before December 31.
```

**Layer 6 improves presentation. Layers 1–4 improve reasoning.**

Use Layer 6 when:
- You know the exact format you need
- You want structured output for downstream processing
- You need the model to commit to its assumptions before answering

Do not spend Layer 6 budget on formatting while leaving Layer 3 empty.

<!-- Speaker notes: Be direct about the hierarchy here. Layer 6 is often the first thing prompt engineering tutorials teach. It's real and useful. But it is the lowest-leverage layer. A well-formatted generic answer is still a generic answer. -->

---

## Visual: The Stack in Action

```
BEFORE (Layer 5 only)
══════════════════════════════════════════════════════════
Prompt: "My client made $180k as a consultant. What deductions?"
Output: "Common self-employment deductions include: home office (if
         you use part of your home exclusively for business), health
         insurance premiums, retirement contributions, business
         expenses, vehicle expenses..."
══════════════════════════════════════════════════════════
→ Accurate. Generic. No jurisdiction. No timing. No objective.
  Covers the average case. Could apply to anyone, anywhere.


AFTER (Full Condition Stack)
══════════════════════════════════════════════════════════
[Jurisdiction: US federal + CA, TCJA 2024]
[Time: Dec 2024 planning, before year-end]
[Objective: minimize tax, low audit risk]
[Constraints: remain sole proprietor, file on time]
[Facts: $180k, 300/2500 sq ft, $8.4k insurance, $25k SEP...]
[Output: assumptions first, then ranked list with CA vs federal flags]
Output: "Assumptions: California does not conform to federal
         Section 179 limits above $25,000; home office uses
         actual expense method (not simplified)...
         1. SEP-IRA additional contribution: $15,000 → saves
         ~$5,580 federal + $1,320 CA = $6,900 total. Confidence:
         certain. Audit risk: low. NOTE: deadline April 15, 2025..."
══════════════════════════════════════════════════════════
→ Specific, actionable, California-flagged, timed, ranked.
```

<!-- Speaker notes: This is the payoff slide. The contrast should be stark and immediate. The "before" output is not wrong — but it is useless for decision-making. The "after" output is something a professional could act on today. Walk through the specific differences: the CA/federal flag on Section 179, the deadline information, the confidence levels. -->

---

## Building a Condition Stack: Step by Step

**Step 1:** Ask "Which ruleset governs this?" → Layer 1

**Step 2:** Ask "When is this? What stage of what process?" → Layer 2

**Step 3:** Ask "What am I optimizing for? What trade-offs do I accept?" → Layer 3

**Step 4:** Ask "What is off the table? What can't I do?" → Layer 4

**Step 5:** Write the facts, knowing the model will interpret them inside your specified world → Layer 5

**Step 6:** Specify what form the output should take → Layer 6

**The diagnostic question for each layer:**
*"Would a different answer to this layer produce meaningfully different advice?"*

If yes — specify it. If no — skip it.

<!-- Speaker notes: The diagnostic question is the practical tool. Students should not mechanically fill in all six layers for every prompt. They should ask "would a different value here change the output?" If the answer is jurisdiction-agnostic (e.g., "explain the Pythagorean theorem"), Layer 1 adds nothing. The stack is for high-stakes, context-dependent questions. -->

---

## The Diagnostic Question Applied

<div class="columns">

**Use the stack when:**

- Domain has jurisdiction-specific rules (legal, tax, medical, finance, compliance)
- Stage of process matters (planning vs. executing vs. auditing)
- Multiple legitimate objectives exist and trade against each other
- Some options are off the table for practical reasons
- Answer quality strongly depends on context you haven't stated

**Skip layers when:**

- Domain has universal rules (math, physics, chemistry)
- Process stage is irrelevant
- There is only one reasonable objective
- All strategies are available
- The question is generic by nature

</div>

<!-- Speaker notes: This slide prevents the over-application of the framework. Students should not add jurisdiction to "what is 2+2." The framework is for high-stakes, context-dependent questions — professional domains, complex technical decisions, medical situations, legal and regulatory analysis. -->

---

## Common Mistakes

**Mistake 1: Jurisdiction in the question, not the stack**
"What are California rules for home office deductions?" — jurisdiction is present but not structured. Put it in Layer 1 so it governs the entire response, not just triggers a geographic filter.

**Mistake 2: Conflating objective and constraints**
"I want low taxes but not too aggressive" — this tangles Layer 3 (minimize tax) with Layer 4 (no aggressive positions). Keep them separate. The model uses them differently.

**Mistake 3: Leaving the objective implicit**
"What's the best approach?" — "best" is not an objective function. Best for what? At what cost to what? Name the trade-off explicitly.

**Mistake 4: Over-investing in Layer 6**
Spending 3 sentences on output format while leaving Layer 3 empty. Format improves presentation. Conditions improve reasoning.

<!-- Speaker notes: These mistakes all come from the same root: treating the prompt as a communication problem ("how do I write this clearly?") rather than a probability problem ("what conditions do I need to specify to pin the model to the right world?"). -->

---

## The Template

```
[LAYER 1 — JURISDICTION + RULE SET]
Domain: [field]  |  Jurisdiction: [country/state/body]
Rule set: [specific law, standard, or protocol]

[LAYER 2 — TIME + PROCEDURAL POSTURE]
Time period: [year or date range]  |  Stage: [planning/executing/etc.]
Time-sensitivity: [deadlines or action windows]

[LAYER 3 — OBJECTIVE FUNCTION]
Primary objective: [what to maximize or minimize]
Secondary objectives: [in priority order]
Acceptable trade-offs: [what you will sacrifice]

[LAYER 4 — CONSTRAINTS]
Hard constraints: [what is off the table]
Resource constraints: [budget, time, team, capability limits]
Risk tolerance: [conservative/moderate/aggressive + specifics]

[LAYER 5 — FACTS]
[All numbers, timeline, documents, events, specifics]

[LAYER 6 — OUTPUT SPECIFICATION]
Format: [bullets/table/narrative/code]  |  Length: [target]
Special: [list assumptions first / show reasoning / rank by objective]
```

<!-- Speaker notes: This is the copy-paste deliverable. Students should bookmark this slide. The template is in the companion guide as well. Walk through how to use it: fill in what you know, skip what doesn't change the answer, and always ask "would a different value here produce meaningfully different advice?" -->

---

## Practice: Spot the Missing Layers

**Which layers are missing from this prompt?**

> "We're building a recommendation engine for a retail client. The system needs to be scalable and handle personalization. What architecture should we use?"

- Layer 1: No. What compliance requirements? What cloud? What existing stack?
- Layer 2: No. What stage? MVP design? Replacing an existing system? At what scale today?
- Layer 3: No. Optimize for what? Recommendation accuracy? Latency? Cost? Engineering speed?
- Layer 4: No. What's off the table? What does the team know? What budget exists?
- Layer 5: Present but thin. What kind of retail? How many users? What data is available?
- Layer 6: Missing entirely.

**Rewrite it using the Condition Stack. Compare outputs.**

<!-- Speaker notes: This is the practice exercise for the slide deck. Students can try this in pairs or individually. The goal is not to write the perfect prompt but to experience the process of asking "what layer is missing?" The before/after comparison reveals the value immediately. -->

---

## Summary

The Condition Stack is a 6-layer protocol for writing prompts that fully specify the model's reasoning context.

**The key insight:** The layers above the facts (Layers 1–4) do more work per word than any amount of factual detail. Most prompts skip them. The model fills them from its training prior — the average case. Your case is not the average case.

**The layers:**
1. Jurisdiction + Rule Set — which world to reason in
2. Time + Procedural Posture — when, and where in the process
3. Objective Function — what "good" means
4. Constraints — what is off the table
5. Facts — the specifics of your situation
6. Output Specification — the form of the answer

**Next:** Module 3 Guide 2 applies the full template to three worked examples (tax, code architecture, medical triage) and shows how to customize the template for different domains.

<!-- Speaker notes: Wrap up by returning to the core Bayesian frame: every layer is additional evidence C in P(A|C). The stack is a systematic protocol for maximizing the information content of C. In the next guide, students will work through complete examples and start building the intuition for how to customize the template for their specific domain. -->

---

<!-- _class: lead -->

## Notebook Exercise

Open `notebooks/01_condition_stack_builder.ipynb`

Build a Condition Stack for your domain using the Claude API.

Compare: raw question vs. fully stacked prompt.

The output difference is the payoff.

<!-- Speaker notes: Direct students to the notebook. The notebook builds an interactive Condition Stack builder using the Claude API — it walks them through each layer with prompts, assembles the final prompt, and calls Claude twice: once with the raw question and once with the full stack. The output comparison is the central experience of this module. -->

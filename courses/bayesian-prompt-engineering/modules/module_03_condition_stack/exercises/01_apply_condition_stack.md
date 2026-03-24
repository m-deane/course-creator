# Exercise 01: Apply the Condition Stack

## Overview

You will apply the 6-layer condition stack to five different domains. For each, you will fill in every layer — not just the ones that feel obvious — and then write the resulting stacked prompt.

The goal is to build the discipline of filling all six layers before writing a prompt, rather than jumping to Layer 5 (facts) the way most people do by default.

**Time estimate:** 45–60 minutes

**Prerequisites:** Complete `guides/01_condition_stack_guide.md` before starting. Have the layer definitions in front of you.

---

## Layer Reference

| Layer | Name | The question to answer before prompting |
|-------|------|----------------------------------------|
| 1 | Jurisdiction + Rule Set | Which rule set governs this? What specific laws, standards, or protocols apply? |
| 2 | Time + Procedural Posture | What is the relevant time period? Where in the process are we right now? |
| 3 | Objective Function | What does "a good answer" mean here? What are we optimizing, and what are we willing to trade away? |
| 4 | Constraints | What strategies or options are off the table? What are the hard limits? |
| 5 | Facts | What are the specific numbers, events, people, and details? |
| 6 | Output Specification | What form should the answer take? How should it be structured? |

---

## How to Work Through Each Domain

1. Read the scenario.
2. Fill in the 6-layer table — every row, even if some entries are short.
3. Read the raw question at the bottom (the bad prompt).
4. Write a condition-stacked prompt that incorporates all six layers.
5. Compare what you expect the stacked prompt to produce vs. the raw prompt.

---

## Worked Example: Tax Compliance — Late Filing

### Scenario

A US-based small business owner with a single-member LLC received a notice from the IRS. She filed her 2022 federal income tax return four months late (Form 1040 + Schedule C) and did not request an extension. She has not filed the state return yet. She has the money to pay what she owes — she just wants to understand the penalties, reduce them if possible, and get compliant on the state side before anything escalates.

### The Raw Prompt (Bad Prompt)

> "I filed my taxes late. What are the penalties and how do I reduce them?"

**What the model assumes without conditions:** US individual filer, mild lateness, no state complexity, objective is simply to "know the rules." Produces a generic explanation of the failure-to-file (5% per month, max 25%) and failure-to-pay (0.5% per month) penalties with the advice to "file as soon as possible." This is accurate for the average case but does not address: the specific LLC and Schedule C implications, whether any penalty abatement applies, state exposure, or what to do next.

### Condition Stack (Filled In)

| Layer | Content |
|-------|---------|
| **Layer 1: Jurisdiction + Rule Set** | US federal (IRS) + California state (FTB). Single-member LLC taxed as sole proprietor — this is a Form 1040 + Schedule C situation, not a corporate return. Governing statute: IRC §6651 (failure to file and failure to pay penalties), FTB Publication 1005 for California late filing penalties. |
| **Layer 2: Time + Procedural Posture** | Tax year 2022. Federal return filed in August 2023 — approximately 4 months late (original due date April 18, 2023). No extension requested. California return not yet filed as of now. IRS notice received and in hand. |
| **Layer 3: Objective Function** | Primary objective: calculate the actual penalty exposure and identify any abatement options available. Secondary: get compliant on the California state side before FTB initiates its own collection process. Not optimizing for: delay or negotiation — client wants to pay what is owed and move on. |
| **Layer 4: Constraints** | Client has funds available to pay full tax liability and penalties — this is not a hardship situation. Does not want to hire a tax attorney; wants to handle this herself or through her current CPA. Cannot risk further escalation — no appetite for additional notices. |
| **Layer 5: Facts** | Single-member LLC, Schedule C. 2022 federal tax liability: $14,200. Paid $8,400 in quarterly estimated taxes. Balance owed at filing: $5,800. Filed 4 months and 3 days late. No prior late filing history in the past 3 tax years. California return: not yet filed; estimate state liability is ~$2,100. |
| **Layer 6: Output Specification** | List: (1) the exact federal penalty calculation, (2) whether first-time penalty abatement applies and how to request it, (3) the California filing steps and estimated penalties, (4) the specific IRS forms or procedures required. Flag any action with a time-sensitive deadline. |

### Condition-Stacked Prompt

```
Domain: US federal and California state tax compliance for a sole proprietor (SMLLC taxed as individual)
Jurisdiction: IRS + California FTB
Governing rules: IRC §6651 (failure to file/pay penalties); first-time penalty abatement administrative waiver; FTB penalty rules for California Schedule CA/Form 540

Time and posture: Tax year 2022. Federal Form 1040 + Schedule C filed August 2023, approximately 4 months and 3 days after the April 18, 2023 deadline. No extension was requested. California return has not yet been filed and no FTB notice has been received yet.

Objective: Calculate the exact penalty owed (failure to file + failure to pay), determine whether first-time abatement applies to waive or reduce those penalties, and outline the steps to file the California return before further escalation.

Constraints: Client can pay the full amount owed. No tax attorney — handling this through CPA. No further delay acceptable.

Facts:
- 2022 federal tax liability: $14,200
- Estimated taxes paid during 2022: $8,400 (quarterly installments)
- Balance owed at time of filing: $5,800
- Filed date: August 2023 (4 months, 3 days late); no extension
- Prior 3-year filing history: on time, no penalties
- California: return not yet filed; estimated state tax liability ~$2,100

Output: Provide (1) the exact failure-to-file and failure-to-pay penalty calculation for the federal situation, (2) whether first-time penalty abatement applies and the exact procedure to request it (letter, phone, or Form 843), (3) the California late filing steps and penalty estimate, (4) any time-sensitive actions. Use numbers, not approximations.
```

---

## Domain 2: Medical — Chronic Headache Differential Diagnosis

### Scenario

A 41-year-old female patient presents to a primary care physician with a 6-week history of daily headaches. The headaches are bifrontal, pressure-like in quality, rated 4–6 out of 10, present on waking, and gradually improve throughout the day. She has no prior history of migraines. She takes lisinopril for hypertension and recently started melatonin for sleep. No trauma. No fever, weight loss, or neurological symptoms. The physician wants to think through the differential systematically before ordering workup.

### The Raw Prompt (Bad Prompt)

> "What are the possible causes of daily headaches in a middle-aged woman?"

### Your Condition Stack

| Layer | Content |
|-------|---------|
| **Layer 1: Jurisdiction + Rule Set** | |
| **Layer 2: Time + Procedural Posture** | |
| **Layer 3: Objective Function** | |
| **Layer 4: Constraints** | |
| **Layer 5: Facts** | |
| **Layer 6: Output Specification** | |

### Your Condition-Stacked Prompt

```
(Write your full stacked prompt here)
```

### Expected Output Difference

*What would the raw prompt produce? What does the stacked prompt produce that the raw prompt cannot?*

---

---

## Domain 3: Software Architecture — Choosing a Message Queue

### Scenario

A backend engineering team is building an order processing system for a B2B e-commerce platform. They need to decouple the checkout service from downstream services (inventory reservation, payment processing, email notifications, order analytics). They are currently running on AWS and their team has strong Python and PostgreSQL expertise but no prior experience with message queue systems. They need to choose between Amazon SQS, Amazon SNS+SQS, Apache Kafka, and RabbitMQ. The system currently processes ~500 orders/day and must scale to 10,000/day within 12 months.

### The Raw Prompt (Bad Prompt)

> "What message queue should I use for my microservices architecture?"

### Your Condition Stack

| Layer | Content |
|-------|---------|
| **Layer 1: Jurisdiction + Rule Set** | |
| **Layer 2: Time + Procedural Posture** | |
| **Layer 3: Objective Function** | |
| **Layer 4: Constraints** | |
| **Layer 5: Facts** | |
| **Layer 6: Output Specification** | |

### Your Condition-Stacked Prompt

```
(Write your full stacked prompt here)
```

### Expected Output Difference

*What would the raw prompt produce? What specific recommendation does the stacked prompt enable?*

---

---

## Domain 4: Business Strategy — Entering a New Market

### Scenario

A B2B SaaS company sells project management software to mid-market construction firms in the US (companies with 50–500 employees). They have $4.2M ARR, 87 customers, 95% net revenue retention, and 18 months of runway. The CEO is considering whether to expand into the UK construction market. A UK-based customer found them organically and has been using the product for 6 months with good results. The board wants a go/no-go recommendation for the next board meeting in 30 days.

### The Raw Prompt (Bad Prompt)

> "Should we expand our product into a new country?"

### Your Condition Stack

| Layer | Content |
|-------|---------|
| **Layer 1: Jurisdiction + Rule Set** | |
| **Layer 2: Time + Procedural Posture** | |
| **Layer 3: Objective Function** | |
| **Layer 4: Constraints** | |
| **Layer 5: Facts** | |
| **Layer 6: Output Specification** | |

### Your Condition-Stacked Prompt

```
(Write your full stacked prompt here)
```

### Expected Output Difference

*What does the stacked prompt enable that the raw prompt cannot — specifically with respect to the go/no-go decision?*

---

---

## Domain 5: Code Generation — Authentication System

### Scenario

A startup is adding user authentication to a new B2C web application. The application is built in Python (FastAPI), the database is PostgreSQL, and the frontend is a React SPA. The engineering team is small (3 developers) and does not have deep security expertise. The application does not handle financial data or PHI, but it will store user email addresses and usage history. The team wants email/password login with the option to add Google OAuth later. They want something they can own and maintain without a third-party identity service subscription.

### The Raw Prompt (Bad Prompt)

> "How do I implement user authentication in my web app?"

### Your Condition Stack

| Layer | Content |
|-------|---------|
| **Layer 1: Jurisdiction + Rule Set** | |
| **Layer 2: Time + Procedural Posture** | |
| **Layer 3: Objective Function** | |
| **Layer 4: Constraints** | |
| **Layer 5: Facts** | |
| **Layer 6: Output Specification** | |

### Your Condition-Stacked Prompt

```
(Write your full stacked prompt here)
```

### Expected Output Difference

*What specific implementation does the stacked prompt produce? What generic answer does the raw prompt produce?*

---

---

## Cross-Domain Reflection

After completing all five stacks, answer these questions:

### 1. Layer 3 is the hardest

Most people can fill in Layer 1 (jurisdiction) and Layer 5 (facts) without much thought. Layer 3 (objective function) requires identifying what trade-offs are being made. Look at your five Layer 3 entries.

For which domain was Layer 3 hardest to articulate? Why?

```
Your answer:
```

### 2. The most commonly skipped layer

Look at the raw prompts for all five domains. Which layer is consistently absent?

```
Your answer:
```

Is the skipped layer the same across domains, or does it differ by domain type? What does that tell you about the default assumptions that model priors make?

```
Your answer:
```

### 3. Layer 6 trap

The most common mistake in applying this framework is over-investing in Layer 6 (output specification) while leaving Layers 2 and 3 thin.

Look at your five Layer 6 entries. For which domain did you spend the most effort on output format relative to the value it added? Which layer for that domain would have produced more improvement per word?

```
Your answer:
```

### 4. Transferability

Pick any two domains above. Compare their Layer 1 and Layer 3 entries.

What is structurally similar between them despite being different domains?

```
Your answer:
```

---

## Key Principle

The condition stack is not a form to fill. It is a **checklist for the conditions that actually do work**.

The question to ask for each layer before filling it:

> "Would a different value here produce a meaningfully different answer?"

If the answer is yes — fill it in. If no — skip it. A one-sentence Layer 2 is better than a three-sentence Layer 2 that does not constrain anything.

**The layers that take the most effort to fill are usually the layers doing the most work.**

---

## Next Steps

- `notebooks/01_condition_stack_notebook.ipynb` — send your stacked prompts via the Claude API and compare outputs against the raw prompts
- Module 4 (`guides/01_conditional_trees_guide.md`) — what to do when Layer 1 or Layer 3 cannot be specified in advance and you need the model to reason about multiple branches
- Module 7 (`guides/01_production_patterns_guide.md`) — how to automate condition injection so that Layers 1–4 are filled from data sources, not manually

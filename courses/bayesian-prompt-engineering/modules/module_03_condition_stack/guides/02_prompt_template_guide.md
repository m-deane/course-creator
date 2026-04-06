# Guide 2: The Plug-and-Play Prompt Template — Worked Examples

> **Reading time:** ~16 min | **Module:** 3 — Condition Stack | **Prerequisites:** Module 2 Switch Variables


## In Brief

The Condition Stack template becomes a skill through use, not through reading. This guide walks through three complete, worked examples — tax filing, code architecture decision, and medical triage — showing how to fill in each layer with real specificity. By the end, you will have a mental model for applying the template to any domain you work in.


<div class="callout-key">
<strong>Key Concept Summary:</strong> The Condition Stack template becomes a skill through use, not through reading.
</div>

---

## The Template (Reference)

```
[LAYER 1 — JURISDICTION + RULE SET]
Domain: [field of knowledge]
Jurisdiction: [country / state-province / regulatory body]
Rule set: [specific law, standard, framework, guideline, or protocol]

[LAYER 2 — TIME + PROCEDURAL POSTURE]
Relevant time period: [year(s), date range, or deadline]
Current stage of process: [planning / filing / executing / etc.]
Time-sensitivity: [deadlines or action windows that affect recommendations]

[LAYER 3 — OBJECTIVE FUNCTION]
Primary objective: [what to maximize or minimize]
Secondary objectives: [additional goals, in priority order]
Acceptable trade-offs: [what you will sacrifice for the primary objective]

[LAYER 4 — CONSTRAINTS]
Hard constraints: [strategies or approaches that cannot be used]
Resource constraints: [budget, time, team size, or capability limits]
Risk tolerance: [conservative / moderate / aggressive + specifics]

[LAYER 5 — FACTS]
[All relevant numbers, timeline, documents, events, and specifics]

[LAYER 6 — OUTPUT SPECIFICATION]
Format: [bullets / narrative / table / code / clinical note / etc.]
Required sections: [any required sections]
Special instructions: [list assumptions first / show reasoning / rank by objective]
Length: [approximate target or "be concise"]
```

---

## Worked Example 1: Tax Filing

### Scenario

A married couple in their late 40s runs a small business selling handmade ceramics online and at craft fairs. They want to understand whether to file as a partnership or have the LLC treated as an S-Corporation for 2024 taxes.
<div class="callout-insight">
<strong>Insight:</strong> A married couple in their late 40s runs a small business selling handmade ceramics online and at craft fairs. They want to understand whether to file as a partnership or have the LLC treated as an S-Corporation for 2024 taxes.
</div>


### Raw Prompt (Before the Stack)

> "We have an LLC selling handmade ceramics. Should we file as a partnership or elect S-Corp status? We made about $220,000 in revenue and have $80,000 in expenses."

This prompt will generate a competent general comparison of partnership vs. S-Corp taxation. It will not tell them:
- Whether S-Corp election is worth it at their income level in their state
- What the self-employment tax savings are vs. the administrative cost of payroll
- Whether they have missed the S-Corp election deadline for 2024
- Whether their state (e.g., California) has an additional S-Corp franchise tax that changes the math

### Filled Template

```
[LAYER 1 — JURISDICTION + RULE SET]
Domain: US federal income tax, business entity taxation
Jurisdiction: United States federal (IRS) + Texas state (no state income tax)
Rule set: IRS rules on LLC taxation elections (Form 8832, Form 2553);
          Texas Franchise Tax rules for LLCs and S-Corps;
          Self-employment tax rules under IRC §1401;
          Reasonable compensation requirements for S-Corp owner-employees

[LAYER 2 — TIME + PROCEDURAL POSTURE]
Relevant time period: Tax year 2024, filing deadline April 15, 2025
Current stage: Year-end planning + prospective decision for 2025
Time-sensitivity: S-Corp election for 2025 requires Form 2553 by March 15, 2025
                  (within 2 months + 15 days of the start of the tax year);
                  retroactive 2024 election is no longer available (it is December)

[LAYER 3 — OBJECTIVE FUNCTION]
Primary objective: Minimize total federal self-employment and income tax over 2 years (2024-2025)
Secondary: Minimize administrative complexity and ongoing compliance cost
Acceptable trade-offs: Willing to pay up to $2,000/year in additional accounting fees
                        if net tax savings exceed $5,000/year

[LAYER 4 — CONSTRAINTS]
Hard constraints: Both spouses work in the business;
                  both must receive reasonable compensation if S-Corp elected;
                  cannot pay below-market salaries to reduce payroll tax (IRS scrutiny)
Resource constraints: No in-house accounting; uses a CPA for annual filing;
                      do not want to manage payroll in-house
Risk tolerance: Moderate — willing to elect S-Corp if savings are clear,
                but not interested in positions that invite scrutiny

[LAYER 5 — FACTS]
2024 gross revenue: $220,000
2024 deductible business expenses: $80,000
2024 net profit (before owner compensation): $140,000
Both spouses active in business, roughly equal time contribution
No employees other than the two spouses
Texas LLC, single-state operations
No retirement accounts currently (SEP-IRA possible)
Have been filing as a partnership (Form 1065) for 3 years
Current CPA charges $1,500/year for partnership return

[LAYER 6 — OUTPUT SPECIFICATION]
Begin with a self-employment tax savings calculation for both scenarios.
Show: (1) partnership path for 2024 and 2025, (2) S-Corp path for 2025.
Include: what "reasonable compensation" would need to be in their situation.
Include: estimated additional compliance cost (payroll, S-Corp return).
State the net annual benefit/cost of S-Corp election in dollars.
Flag any Texas franchise tax considerations.
End with a clear recommendation and the Form 2553 deadline they need to meet.
Format: tables for the numbers, then narrative recommendation.
```

### What the Stacked Prompt Produces

The model now knows it is analyzing a Texas LLC (no state income tax — removes one complication), that the S-Corp election window for 2024 is closed (removes half the question), that the objective is a 2-year total cost comparison (not "which is better in general"), and that the constraint is $2,000 CPA fee tolerance against a required $5,000 savings threshold. It produces a numeric comparison, a recommendation, and a specific deadline (Form 2553 by March 15, 2025).

The unstacked prompt produces a 600-word article explaining how S-Corps save on self-employment tax in general, with no calculation for their specific numbers and no mention of the election deadline.

---

## Worked Example 2: Code Architecture Decision

### Scenario
<div class="callout-warning">
<strong>Warning:</strong> A senior engineer needs to decide whether to build a new internal data pipeline using Apache Kafka or AWS SQS + Lambda. The company processes about 50,000 events per day with occasional spikes to 500,000. Multiple downstream teams consume the data.
</div>


A senior engineer needs to decide whether to build a new internal data pipeline using Apache Kafka or AWS SQS + Lambda. The company processes about 50,000 events per day with occasional spikes to 500,000. Multiple downstream teams consume the data.

### Raw Prompt (Before the Stack)

> "Should we use Kafka or SQS + Lambda for our event pipeline? We have about 50k events per day with spikes to 500k. Multiple teams consume the data."

This generates a feature comparison table. It will not account for:
- The team's existing expertise and operational capacity
- Whether the company is on AWS or multi-cloud
- What "multiple teams consuming" means — fan-out replays? Real-time vs. batch?
- Whether the correct frame is "Kafka vs. SQS" or "do we even need a queue at this scale?"

### Filled Template

```
[LAYER 1 — JURISDICTION + RULE SET]
Domain: Backend data infrastructure, event streaming
Jurisdiction: AWS us-east-1, production environment
Rule set: AWS managed services preferred (company policy);
          SOC 2 Type II compliance required for all data pipelines;
          event data may contain PII — GDPR and CCPA apply;
          engineering team must be able to operate without DevOps dedicated support

[LAYER 2 — TIME + PROCEDURAL POSTURE]
Relevant time period: Decision needed this sprint; implementation starts Q1 2025
Current stage: Architecture decision — no code written yet; existing system is
               a simple cron job + PostgreSQL trigger that is breaking under load
Time-sensitivity: Current system is failing at current scale;
                  cannot wait for a perfect solution;
                  need something running in production within 6 weeks

[LAYER 3 — OBJECTIVE FUNCTION]
Primary objective: Minimize operational burden on the engineering team
                   (team is 4 engineers, no dedicated infrastructure engineer)
Secondary: Support fan-out to 6 current consumer teams without per-consumer custom code
Secondary: Retain event replay capability for at least 7 days
Acceptable trade-offs: Willing to pay more in AWS costs for less operational complexity

[LAYER 4 — CONSTRAINTS]
Hard constraints: No on-premise infrastructure;
                  no Kubernetes cluster (only ECS Fargate for containers);
                  Kafka is not permitted under company managed-services-only policy
                  (MSK is allowed — this is the specific question)
Resource constraints: 4 engineers total; max 2 engineers available for this project;
                      AWS budget increase of $800/month approved for this system
Risk tolerance: Low — current system is already in production;
                need a migration path, not a big-bang replacement

[LAYER 5 — FACTS]
Current: cron job runs every 5 min, inserts events to PostgreSQL,
         6 downstream teams pull from Postgres tables (this is the problem)
Scale: 50,000 events/day average, 500,000/day during 3-4 annual peak periods
       (product launches), with peaks lasting 12-48 hours
Event types: 3 event schemas; average event size 2KB; no ordering requirement
             within an event type; strict ordering NOT required
Downstream consumers: 6 teams, 5 using Python, 1 using Node.js;
                      all teams have AWS SDK experience;
                      none have Kafka/MSK experience
Retention: need 7-day replay capability;
           currently losing events if cron job fails

[LAYER 6 — OUTPUT SPECIFICATION]
Evaluate three options: (1) SQS + SNS fan-out + Lambda,
                        (2) Amazon MSK (managed Kafka),
                        (3) Amazon Kinesis Data Streams.
For each option, give:
  - Estimated monthly AWS cost at average and peak load
  - Operational complexity rating (1-5, with rationale)
  - Time to implement (in engineer-weeks, for 2 engineers)
  - How it handles the fan-out requirement for 6 consumers
  - Whether 7-day replay is native or requires additional work
Then: a clear recommendation with rationale, and a migration path
      from the current cron+PostgreSQL system.
Flag any PII/GDPR implications of the recommended option.
```

### What the Stacked Prompt Produces

The model knows that Kafka (self-managed) is not on the table but MSK is, that the team has no Kafka experience, that ordering is not required (this eliminates Kafka's main advantage at this scale), that the actual constraint is operational simplicity not cost, and that GDPR applies. It recommends SQS + SNS fan-out with Lambda, explains why MSK is overkill for this event volume and team capacity, gives actual AWS cost estimates, and specifies the migration path from the cron job.

The unstacked prompt produces a feature comparison table of Kafka vs. SQS that ignores the team capacity constraint entirely and does not mention that at 50k events/day, Kafka's operational overhead is rarely worth it unless you have ordering requirements or a dedicated ops team.

---

## Worked Example 3: Medical Triage

### Scenario

A nurse practitioner is seeing a 58-year-old patient in a primary care setting who presents with a 3-week history of fatigue and mild shortness of breath on exertion. She needs to build a differential and decide which workup to order.
<div class="callout-key">
<strong>Key Point:</strong> A nurse practitioner is seeing a 58-year-old patient in a primary care setting who presents with a 3-week history of fatigue and mild shortness of breath on exertion. She needs to build a differential and decide which workup to order.
</div>


### Raw Prompt (Before the Stack)

> "58-year-old patient with 3-week fatigue and exertional dyspnea. What should I work up?"

This generates a comprehensive differential list. It will not account for:
- The clinical setting and what resources are available
- What prior workup has already been done
- Whether the goal is to rule out dangerous diagnoses first or find the most likely diagnosis
- The patient's insurance situation and cost constraints

### Filled Template

```
[LAYER 1 — JURISDICTION + RULE SET]
Domain: Internal medicine, primary care
Jurisdiction: United States, outpatient primary care clinic
Rule set: AHA/ACC heart failure guidelines (2022);
          USPSTF screening recommendations;
          CMS Medicare reimbursement rules apply (patient is Medicare-covered);
          clinic capability: can order labs and plain films,
          no in-house echo or stress test (must refer)

[LAYER 2 — TIME + PROCEDURAL POSTURE]
Stage: Initial presentation — no prior workup for this complaint
Time-sensitivity: 3-week symptom duration; no acute distress in office;
                  patient can return for follow-up in 1-2 weeks;
                  but if high-risk findings: same-day referral or ED

[LAYER 3 — OBJECTIVE FUNCTION]
Primary objective: Rule out dangerous diagnoses (high-sensitivity approach first)
                   — do not miss CHF, ACS, PE, or malignancy
Secondary: Identify the most likely diagnosis efficiently,
           minimizing unnecessary specialist referrals and cost
Acceptable trade-offs: Willing to order more tests than strictly necessary
                       if it adequately rules out life-threatening causes

[LAYER 4 — CONSTRAINTS]
Hard constraints: Patient is allergic to iodinated contrast —
                  no CT with contrast; no cardiac catheterization
                  without prior allergy management
Resource constraints: Outpatient only; no same-day echo;
                      patient lives 60 miles from nearest hospital —
                      prefers to minimize referral travel
Risk tolerance: Low for missing dangerous diagnosis;
                patient is moderately anxious, prefers fewer tests if safe

[LAYER 5 — FACTS]
58-year-old female, BMI 31, nonsmoker
Vital signs: BP 148/92, HR 94, RR 16, SpO2 96% on room air, Temp 98.6
Symptom: Fatigue 3 weeks, exertional dyspnea (2 flights of stairs, previously could do 4)
         No orthopnea, no PND, no leg swelling
         Mild chest heaviness with exertion (denies typical angina)
PMH: HTN (on lisinopril 10mg), Type 2 DM (on metformin),
     hyperlipidemia (on atorvastatin 20mg)
FH: Father died of MI at 62
No prior cardiac workup; last ECG 3 years ago (normal sinus rhythm)
Current labs (from 3 months ago): HbA1c 7.8%, creatinine 1.1, lipid panel pending

[LAYER 6 — OUTPUT SPECIFICATION]
Format: clinical decision-making structure (not SOAP — this is a triage decision)
Section 1: Dangerous diagnoses to rule out first (ranked by urgency),
           with the specific test that rules each one out or in.
Section 2: Most likely diagnoses given the full picture.
Section 3: Recommended initial workup — list tests in priority order,
           noting which are urgent (order today) vs. can wait for follow-up.
Section 4: Triage decision — return in 1-2 weeks vs. same-day referral vs. ED now?
Flag: contrast allergy implications for any imaging ordered.
Note: which tests require prior auth under standard Medicare Part B.
```

### What the Stacked Prompt Produces

The model knows: outpatient only (no inpatient resources), contrast allergy (CT with contrast is off the table — must use echocardiography or CT without contrast), patient is Medicare (relevant for prior auth), goal is rule-out-dangerous first, and the patient has significant cardiac risk factors (HTN, DM, hyperlipidemia, family history of MI at 62, new exertional symptoms including chest heaviness). It produces a triage decision structured around ruling out CHF, ACS, and PE, specifies echo as the primary imaging (not CT angiography), recommends same-day BNP and ECG, and flags the 60-mile-travel constraint in the referral section.

The unstacked prompt produces a differential that includes anemia, hypothyroidism, depression, COPD, CHF, PE, and a dozen other conditions, with a generic workup list. Technically complete. Clinically untriaged.

---

## How to Customize the Template for Your Domain

The template is domain-agnostic by design. The layer names are abstractions. Here is how to translate them for four common professional contexts:
<div class="callout-insight">
<strong>Insight:</strong> The template is domain-agnostic by design. The layer names are abstractions. Here is how to translate them for four common professional contexts:
</div>


### Legal / Regulatory

| Stack Layer | Legal Translation |
|-------------|-------------------|
| Layer 1: Jurisdiction + Rule Set | Court jurisdiction, governing law, applicable statutes, standard of review |
| Layer 2: Time + Procedural Posture | Statute of limitations, filing deadlines, litigation stage (pre-complaint, discovery, summary judgment, trial) |
| Layer 3: Objective Function | Win at trial / favorable settlement / delay / cost minimization / liability limitation |
| Layer 4: Constraints | Client cannot appear in person; budget cap; opposing counsel's known strategy; venue limitations |

### Software / Technical

| Stack Layer | Software Translation |
|-------------|----------------------|
| Layer 1: Jurisdiction + Rule Set | Cloud provider, compliance standards, tech stack constraints, security requirements |
| Layer 2: Time + Procedural Posture | Sprint timing, release deadline, migration phase (design / build / test / cutover) |
| Layer 3: Objective Function | Latency / throughput / cost / developer velocity / maintainability / resilience |
| Layer 4: Constraints | Existing infrastructure, team skills, no new dependencies, budget ceiling, backwards compatibility |

### Finance / Investment

| Stack Layer | Finance Translation |
|-------------|---------------------|
| Layer 1: Jurisdiction + Rule Set | Regulatory regime (SEC, FINRA, FCA), investment mandate, suitability standard, asset class restrictions |
| Layer 2: Time + Procedural Posture | Investment horizon, current market cycle, proximity to liquidity events, reporting period |
| Layer 3: Objective Function | Return maximization / risk-adjusted return / income / capital preservation / tax efficiency |
| Layer 4: Constraints | Prohibited securities, concentration limits, ESG screens, liquidity requirements, redemption windows |

### Medical / Clinical

| Stack Layer | Clinical Translation |
|-------------|---------------------|
| Layer 1: Jurisdiction + Rule Set | Care setting (ED / ICU / outpatient / inpatient), applicable guidelines, payer rules, available resources |
| Layer 2: Time + Procedural Posture | Acuity (emergent / urgent / routine), symptom duration, stage of workup (initial / follow-up / post-treatment) |
| Layer 3: Objective Function | Sensitivity-first (don't miss anything) vs. specificity-first (minimize unnecessary treatment); cost-conscious |
| Layer 4: Constraints | Drug allergies, unavailable imaging, no specialist access, patient preference to avoid hospitalization |

---

## The Single Most Valuable Layer Across All Domains

If you could add only one layer to your current prompts, it should be **Layer 3: Objective Function**.

In every professional domain, multiple legitimate objectives exist that trade against each other. Tax law has minimize-tax vs. minimize-audit-risk. Medical practice has high-sensitivity vs. cost-minimization. Software architecture has performance vs. maintainability vs. velocity. Legal strategy has win-at-trial vs. favorable-settlement vs. minimize-cost.

A model with no objective function specified must either pick one implicitly (and it picks the modal answer from its training data, which is the hedged average) or cover all options (producing a balanced answer that optimizes for nothing). Specifying the objective function is the single change that produces the most immediate improvement in output quality across all domains.

---

## Prompt Length and the Stack

A common worry: "This template will make my prompts too long."

Three responses:

**1. Long prompts with conditions outperform short prompts without them.** A 400-word prompt with all six layers will produce a better answer than a 50-word prompt missing Layers 1–4 every time, in high-stakes domains.

**2. The template is a thinking tool.** Many layers will be one sentence. The tax example above is thorough because the domain is complex. A software architecture question might need only two sentences for Layer 1 and one sentence for Layer 2. Use judgment.

**3. The alternative is follow-up prompts.** If you send a Layer-5-only prompt, you will spend 3–5 follow-up messages adding the conditions the model needed in the first place. A single well-structured prompt is faster than five corrective follow-ups.

---

## Connections

- **Builds on:** Guide 1 (the 6-layer framework and the full template)
- **Leads to:** Notebook 01 (build the Condition Stack builder using the Claude API)
- **Related to:** Module 4 (Conditional Trees — when Layer 1 or Layer 3 cannot be specified in advance)
- **Related to:** Module 7 (Production Patterns — automated condition stack injection at runtime)

---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "The Template (Reference)" and why it matters in practice.

2. Given a real-world scenario involving guide 2: the plug-and-play prompt template — worked examples, what would be your first three steps to apply the techniques from this guide?
</div>

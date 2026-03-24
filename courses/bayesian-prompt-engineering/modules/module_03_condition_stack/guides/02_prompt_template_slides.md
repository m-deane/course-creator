---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# The Plug-and-Play Template
### Three Fully Worked Examples

**Module 3, Guide 2 — Bayesian Prompt Engineering**

Tax filing · Code architecture · Medical triage

<!-- Speaker notes: This deck works through three complete examples of the condition stack template being filled in for real situations. The goal is to build the muscle memory for applying the framework across different domains. Each example shows the raw prompt, the filled stack, and describes what changes in the output. -->

---

## The Template at a Glance

```
[L1] Domain / Jurisdiction / Rule set
[L2] Time period / Procedural stage / Deadlines
[L3] Primary objective / Secondary objectives / Trade-offs
[L4] Hard constraints / Resource limits / Risk tolerance
[L5] Facts — numbers, events, documents
[L6] Format / Required sections / Special instructions
```

**The question to ask at each layer:**
*"Would a different answer here produce meaningfully different advice?"*

If yes → specify it.
If no → skip it.

<!-- Speaker notes: This is the reference slide. Students should have it open while working through the examples. The diagnostic question is the key practice tool. -->

---

## Example 1: Tax Filing

**The Scenario**

A married couple runs an LLC selling handmade ceramics online and at craft fairs. They are asking whether to stay as a partnership or elect S-Corp status.

**The Raw Prompt**

> "We have an LLC selling handmade ceramics. Should we file as a partnership or elect S-Corp status? We made about $220,000 in revenue and have $80,000 in expenses."

**What the raw prompt produces:**
A general comparison of partnership vs. S-Corp taxation. Accurate. Not actionable. Ignores their state, their timeline, their specific net income, and the S-Corp election deadline.

<!-- Speaker notes: The raw prompt is not bad — it's just insufficient. It will get a 600-word article on S-Corp taxation in general. What the couple actually needs: whether S-Corp is worth it at their income level, in their state, given their specific facts, and whether the election deadline has passed. -->

---

## Example 1: Filling Layer 1

**Domain:** US federal income tax, business entity taxation

**Jurisdiction:** United States federal (IRS) + Texas state

**Rule set:**
- IRS rules on LLC taxation elections (Form 8832, Form 2553)
- Texas Franchise Tax rules for LLCs and S-Corps
- Self-employment tax rules under IRC §1401
- Reasonable compensation requirements for S-Corp owner-employees

**What Layer 1 does here:**
- "Texas" eliminates state income tax as a factor (Texas has none) — simplifies the analysis
- "Reasonable compensation" requirement is flagged — this is the key risk in S-Corp elections

<!-- Speaker notes: Notice that specifying Texas immediately simplifies the analysis. States like California have an $800/year franchise tax on S-Corps plus an additional 1.5% tax on S-Corp net income. In Texas, there's no state income tax. This changes the S-Corp calculus significantly. Without specifying the state, the model has to cover all scenarios. -->

---

## Example 1: Filling Layers 2 and 3

**Layer 2 — Time + Posture**

> Tax year 2024, currently December. Year-end planning + prospective decision for 2025.
> **Critical:** S-Corp election for 2025 requires Form 2553 by March 15, 2025. Retroactive 2024 election is no longer available.

**Layer 3 — Objective Function**

> Primary: Minimize total federal self-employment + income tax over 2024-2025
> Secondary: Minimize administrative complexity and ongoing compliance cost
> Trade-off: Willing to pay up to $2,000/year in additional accounting fees if net savings exceed $5,000/year

**What Layer 3 does:** The model now knows the threshold. It can tell them whether the S-Corp math clears the $5,000 net savings bar — which is the actual decision.

<!-- Speaker notes: Layer 2 is particularly important here because the S-Corp election is time-sensitive. The model needs to know that December means the 2024 window is closed but 2025 is still open with a March 15 deadline. Layer 3 gives the model a numeric threshold to evaluate against — "is this worth it?" now has a testable answer. -->

---

## Example 1: Filling Layers 4 and 5

**Layer 4 — Constraints**

> - Both spouses active in business; both must receive reasonable compensation
> - Cannot pay below-market salaries (IRS scrutiny)
> - No in-house payroll; uses CPA for annual filing
> - Does not want to manage payroll in-house

**Layer 5 — Facts**

> 2024 gross revenue: $220,000 | Deductible expenses: $80,000 | Net profit: $140,000
> Two equal-time spouses; no other employees; Texas LLC; 3 years filing as partnership
> No retirement accounts; current CPA charges $1,500/year

**What Layer 4 does:** The "no in-house payroll" constraint means S-Corp adds a payroll service cost (~$600-1,200/year). The model incorporates this into the net savings calculation automatically.

<!-- Speaker notes: The constraint about not wanting to manage payroll in-house is often overlooked but materially affects the S-Corp analysis. S-Corp requires running payroll for the owner-employees. This adds a cost. With Layer 4 specified, the model accounts for it in the recommendation without being asked. -->

---

## Example 1: Layer 6 and the Output

**Layer 6 — Output**

> Begin with SE tax savings calculation for both scenarios.
> Show: (1) partnership 2024-2025, (2) S-Corp 2025.
> Include: reasonable compensation estimate + additional compliance cost.
> State net annual benefit/cost in dollars.
> End with clear recommendation + Form 2553 deadline.
> Format: tables for numbers, then narrative recommendation.

**What the stacked prompt produces:**

| | Partnership | S-Corp (2025) |
|---|---|---|
| Net income | $140,000 | $140,000 |
| Reasonable salary (2 spouses) | — | $70,000 each |
| SE tax savings | — | ~$8,500/year |
| Additional costs (payroll + CPA) | — | ~$2,500/year |
| **Net annual benefit** | — | **~$6,000** |

Recommendation: Elect S-Corp for 2025. File Form 2553 by March 15, 2025.

<!-- Speaker notes: The output is now a decision, not an article. The $6,000 net benefit clears the $5,000 threshold specified in Layer 3. The March 15 deadline is surfaced because Layer 2 specified that the 2025 election window is still open. This is the output difference the template produces. -->

---

## Example 2: Code Architecture

**The Scenario**

A senior engineer must choose between Amazon MSK (managed Kafka) and SQS + SNS for an internal event pipeline. 50k events/day average, 500k peak. 6 downstream consumer teams.

**The Raw Prompt**

> "Should we use Kafka or SQS + Lambda for our event pipeline? We have about 50k events per day with spikes to 500k. Multiple teams consume the data."

**What the raw prompt produces:**
A feature comparison table. Mentions Kafka's strengths (high throughput, log compaction, replay). Does not account for team expertise, managed service policy, or whether the scale even requires Kafka.

<!-- Speaker notes: The raw prompt will generate a balanced comparison that says "it depends." The model doesn't know that the team has no Kafka experience, that the company prefers managed services, that ordering is not required (Kafka's main advantage becomes irrelevant), or that the operational burden is the primary concern. -->

---

## Example 2: The Key Layers

<div class="columns">

**Layer 1 — Jurisdiction**

AWS us-east-1, production
SOC 2 Type II required
GDPR + CCPA apply (PII in events)
AWS managed services preferred (company policy)
Team must operate without dedicated DevOps

**Layer 3 — Objective**

Primary: Minimize operational burden on 4-engineer team
Secondary: Support fan-out to 6 consumer teams natively
Secondary: 7-day event replay capability
Trade-off: Pay more in AWS costs for less operational complexity

</div>

**Layer 4 (the key constraint):**
> "Self-managed Kafka is NOT permitted under company policy. MSK (managed Kafka) IS allowed — this is the question."
> "4 engineers total; none have Kafka/MSK experience; 2 engineers max on this project"

<!-- Speaker notes: Layer 4 is doing heavy lifting here. The constraint that the team has no Kafka/MSK experience is critical — Kafka has a steep learning curve and the team would need to invest weeks learning it. Layer 1 specifying "managed services preferred" also matters: it rules out self-managed Kafka but keeps MSK on the table. The model needs both of these to give a useful recommendation. -->

---

## Example 2: Facts and Output Shape

**Layer 5 — Key Facts**

> Current system: cron job every 5 min → PostgreSQL → 6 teams pulling (this is breaking)
> Event types: 3 schemas, ~2KB each, no ordering requirement
> Downstream consumers: 6 teams, 5 Python + 1 Node.js; all have AWS SDK experience, none have Kafka experience
> Need 7-day replay; currently losing events on cron job failure

**Layer 6 — Output Shape**

> Evaluate 3 options: SQS + SNS fan-out + Lambda / MSK / Kinesis
> For each: monthly AWS cost at avg + peak load | operational complexity (1-5) | time to implement | fan-out approach | replay support
> Then: clear recommendation + migration path from current cron+Postgres
> Flag PII/GDPR implications

**The recommendation the stacked prompt produces:**
SQS + SNS fan-out. Ordering not required eliminates MSK's main advantage. Kinesis adds operational complexity for negligible benefit at this scale. Migration path: add SQS publisher to cron job first, then migrate consumers one-by-one.

<!-- Speaker notes: The "no ordering requirement" fact in Layer 5 is the technical key. Kafka's primary advantage is ordering guarantees. Without that requirement, Kafka is adding operational complexity for no benefit at 50k events/day. The model knows this — but only if you tell it that ordering is not required. Without that fact, it may recommend Kafka "for future scalability." -->

---

## Example 3: Medical Triage

**The Scenario**

A nurse practitioner sees a 58-year-old female in primary care. 3-week fatigue, exertional dyspnea, mild chest heaviness. HTN, DM, hyperlipidemia, family history MI at 62.

**The Raw Prompt**

> "58-year-old patient with 3-week fatigue and exertional dyspnea. What should I work up?"

**What the raw prompt produces:**
A comprehensive differential including anemia, hypothyroidism, depression, COPD, CHF, PE, malignancy. A generic workup list: CBC, TSH, BMP, CXR, ECG. Technically complete. Untriaged.

**What it misses:**
- No triage decision (see in 1-2 weeks vs. same-day referral vs. ED)
- No contrast allergy consideration (CT-PA is the usual PE workup — it's off the table)
- No resource constraint (outpatient only, no in-house echo)
- No prioritization by the stated objective (rule out dangerous diagnoses first)

<!-- Speaker notes: The raw prompt gives a differential and a workup list — both of which are things a medical student could produce. The clinical decision being made here is: is this patient safe to return in a week, or does she need a same-day referral or ED visit? That triage decision requires knowing: the patient's cardiac risk factors (significant), the available resources (limited), the contrast allergy (constrains imaging), and the objective (sensitivity-first). -->

---

## Example 3: The Clinical Context Layers

**Layer 1 — Setting and Guidelines**

> US outpatient primary care clinic
> AHA/ACC heart failure guidelines (2022), USPSTF recommendations
> Medicare covered; clinic has labs + plain films, no in-house echo or stress test

**Layer 2 — Stage and Acuity**

> Initial presentation, no prior workup for this complaint
> No acute distress in office; can return in 1-2 weeks
> IF high-risk findings: same-day referral or ED

**Layer 3 — Objective**

> Primary: Rule out dangerous diagnoses first (sensitivity approach)
> — do not miss CHF, ACS, PE, malignancy
> Secondary: Minimize unnecessary specialist referral and cost
> Trade-off: Order more tests than strictly necessary to adequately rule out life-threatening causes

<!-- Speaker notes: Layer 3 is the critical one here. "Rule out dangerous diagnoses first" is a sensitivity-first approach — it means you prioritize tests that have high sensitivity for dangerous conditions, even if they generate false positives. This is the opposite of a "find the most likely diagnosis efficiently" approach. Both are valid — they depend on the clinical context and risk tolerance. Without specifying this, the model hedges. -->

---

## Example 3: Constraints and the Contrast Allergy

**Layer 4 — The Key Constraint**

```
Hard constraints:
  - Allergic to iodinated contrast — NO CT with contrast
    (CT pulmonary angiography for PE workup is off the table)
  - Outpatient only — no same-day echocardiography
  - Patient lives 60 miles from nearest hospital —
    minimize referral travel when possible

Risk tolerance:
  - Low for missing dangerous diagnosis
  - Patient moderately anxious — prefers fewer tests if safe
```

**Why this constraint is critical:**

CT-PA (CT pulmonary angiography) is the standard-of-care imaging for ruling out PE. It requires iodinated contrast. With the allergy specified in Layer 4, the model will recommend V/Q scan or echo instead — the correct clinical path.

Without Layer 4, the model recommends CT-PA. Incorrect for this patient.

<!-- Speaker notes: This is a concrete example of how Layer 4 directly changes the clinical recommendation. The model knows that CT-PA requires contrast. When Layer 4 specifies contrast allergy, the model uses V/Q scan (which doesn't require contrast) for PE workup. This is not a minor change — it's the difference between a correct and incorrect clinical plan. -->

---

## Example 3: The Output Structure

**Layer 6 — Clinical Decision Structure**

> Section 1: Dangerous diagnoses to rule out first, ranked by urgency — with the specific test that rules each one out or in.
> Section 2: Most likely diagnoses given the full picture.
> Section 3: Workup ordered by urgency — "order today" vs. "at follow-up."
> Section 4: Triage decision — return 1-2 weeks / same-day referral / ED now?
> Flag: contrast allergy implications for any imaging.
> Note: which tests require Medicare prior auth.

**The stacked prompt produces:**

1. Rule out: CHF (BNP today + echo referral), ACS (ECG today, troponin if any acute symptoms), PE (V/Q scan — NOT CT-PA due to contrast allergy)
2. Most likely: decompensated hypertensive cardiomyopathy given risk factor burden
3. Order today: BNP, ECG, CBC, BMP, TSH; referral for echo within 1 week
4. Triage: Return 1-2 weeks if BNP/ECG unremarkable; same-day cardiology referral if BNP elevated or ECG changes

<!-- Speaker notes: The contrast allergy flag is now in Section 3 — the model correctly routes to V/Q scan for PE workup and notes this is due to the contrast allergy. The triage decision in Section 4 is conditional: it depends on what the BNP and ECG show today. This is a clinical decision tree — which will be the topic of Module 4. -->

---

## Cross-Domain Pattern: The Most Valuable Layer

**If you add only one layer to your current prompts:**

| Domain | The single most impactful layer |
|--------|--------------------------------|
| Tax / Legal | Layer 1: Jurisdiction (state/country matters more than any other condition) |
| Software architecture | Layer 3: Objective function (latency vs. cost vs. velocity — opposite recommendations) |
| Medical | Layer 3: Objective (sensitivity-first vs. efficiency-first — different workups) |
| Business strategy | Layer 3: Time horizon (3-month survival vs. 3-year growth — opposite tactics) |
| Code generation | Layer 4: Constraints (what stack, what team, what compatibility requirements) |

**In every domain:** Layer 3 (Objective Function) produces the most dramatic output change from a single addition.

<!-- Speaker notes: This is the practical summary students should take away. If they don't have time to fill out the full stack, at minimum add Layer 1 (jurisdiction) for professional domains and Layer 3 (objective) for any domain. These two layers do more work than all the prompt engineering tricks combined. -->

---

## Customizing the Template: Domain Mapping

<div class="columns">

**Legal / Regulatory**

- L1: Court, governing law, statutes
- L2: Litigation stage, deadlines
- L3: Win at trial / settlement / cost
- L4: Budget, venue, opposing counsel

**Software / Technical**

- L1: Cloud, compliance, tech stack
- L2: Sprint, release date, phase
- L3: Latency / cost / velocity
- L4: Team skills, existing infra

</div>

<div class="columns">

**Finance / Investment**

- L1: Regulatory regime, mandate
- L2: Investment horizon, market cycle
- L3: Return / risk / income / tax efficiency
- L4: Prohibited securities, liquidity

**Medical / Clinical**

- L1: Care setting, guidelines, payer
- L2: Acuity, workup stage
- L3: Sensitivity-first vs. efficiency-first
- L4: Allergies, available resources

</div>

<!-- Speaker notes: The template is a universal structure but the content of each layer varies by domain. This slide gives students the translation key for the four most common professional domains. The layer names are abstractions — "jurisdiction" in software means "what environment and compliance regime governs this" not a legal jurisdiction. -->

---

## What to Do Right Now

1. Find the last prompt you sent that got a generic answer.

2. Ask: which layers were missing?
   - Did you specify jurisdiction? Time? Objective? Constraints?

3. Rewrite it using the template.

4. Compare outputs.

The output difference is usually immediate and significant.

**The notebook does this for you:**
`notebooks/01_condition_stack_builder.ipynb`
- Walks through all 6 layers interactively
- Calls Claude with raw prompt + stacked prompt
- Shows you both outputs side by side

<!-- Speaker notes: End with a call to action. The template only becomes a skill through use. The notebook automates the comparison so students can see the output difference without manually running two API calls and visually comparing. Direct them to the notebook now. -->

---

<!-- _class: lead -->

## Summary

The template is a thinking tool.

Fill in the layers that change the answer. Skip the ones that don't.

The objective function is the single highest-ROI addition in any domain.

**Next:** `notebooks/01_condition_stack_builder.ipynb` — build it, run it, see it work.

<!-- Speaker notes: Final summary. The key messages: (1) template is a thinking tool not a form, (2) objective function is the single highest-impact layer, (3) the notebook is where the learning actually happens — get them to open it. -->

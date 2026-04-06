# Switch Variables: The Conditions That Actually Matter

> **Reading time:** ~11 min | **Module:** 2 — Switch Variables | **Prerequisites:** Module 1 Bayesian Frame


## In Brief

A **switch variable** is a condition whose value routes reasoning to a categorically different solution branch. It is not a detail that sharpens an answer — it is the fork that determines *which* answer is correct.

> **Core Insight:** The model holds a vast distribution of possible answers. Switch variables are not refinements of a single answer — they are the selection mechanism between *fundamentally different answers*. Add the wrong detail and you get a sharper wrong answer. Add a switch variable and you get the right answer entirely.

The skill is asking: *"What are the top 5 conditions that, if different, would make the correct answer completely different?"*


<div class="callout-key">
<strong>Key Concept Summary:</strong> A **switch variable** is a condition whose value routes reasoning to a categorically different solution branch.
</div>

---

## Visual Explanation

```
Without switch variables — model averages across possible worlds:

  "What's the tax treatment of this gain?"
          |
          ▼
  [short-term? long-term? US? UK? individual? corporate? ordinary income?]
          |
          ▼
  Model produces a blended, hedged, qualified non-answer

With switch variables — model operates inside one world:

  "What's the tax treatment of this gain?"
  + [US federal | individual filer | held 14 months | not in retirement account]
          |
          ▼
  Model answers precisely: "Long-term capital gains, 15% or 20% depending on income bracket"
```

The switch variables here are: jurisdiction, entity type, holding period, account type. Four conditions. Each one routes to a different tax world. Together, they reduce a 40-world distribution to a single answer.

---

## Formal Definition

Let $Q$ be a question and $W$ be the set of all worlds (contexts) in which the question could be asked. A **switch variable** $V$ is a condition such that:

$$P(\text{correct answer} \mid Q, V = v_1) \neq P(\text{correct answer} \mid Q, V = v_2)$$

for distinct values $v_1, v_2$. The *size* of the switch is the Kullback-Leibler divergence between the two answer distributions:

$$\text{switch\_size}(V) = D_{KL}\bigl(P(A \mid Q, V{=}v_1) \,\|\, P(A \mid Q, V{=}v_2)\bigr)$$

A high switch size means the variable routes reasoning to an entirely different universe of answers. A low switch size means the variable refines the answer without changing its fundamental character.

---

## The Five Categories of Switch Variables

Every switch variable in any domain falls into one of five categories:
<div class="callout-warning">
<strong>Warning:</strong> Every switch variable in any domain falls into one of five categories:
</div>


### 1. Jurisdiction / Scope

The legal, regulatory, or operational boundary that governs which rules apply.

- **Law:** Federal vs. state, US vs. EU, civil vs. criminal, which circuit court
- **Medicine:** Country-specific treatment guidelines, payer type (Medicare, private, NHS)
- **Engineering:** Regulatory body (FAA vs. EASA, FDA vs. CE marking), country electrical code
- **Finance:** Exchange (NYSE vs. LSE), asset class, regulatory regime (SEC, CFTC, FCA)
- **Code:** Which runtime, which cloud provider, which compliance framework (SOC2, HIPAA)

**Why it's a switch:** Different jurisdictions operate under different rule systems. The legally correct answer in California may be illegal in Texas.

### 2. Timing / Posture

When the action is taking place relative to a process, deadline, or state transition.

- **Law:** Pre-litigation vs. post-filing, before vs. after statute of limitations, pre-trial vs. appellate
- **Medicine:** Acute presentation vs. chronic management, pre-op vs. post-op, in-patient vs. outpatient
- **Engineering:** Design phase vs. production phase vs. maintenance, pre-launch vs. post-launch
- **Finance:** Pre-trade vs. post-trade, open position vs. closed, before vs. after earnings
- **Code:** Development vs. staging vs. production, before vs. after database migration

**Why it's a switch:** The same action taken at different points in a process can be optimal, neutral, or catastrophic. Filing a motion after the deadline is not "less good" than filing before — it simply doesn't exist.

### 3. Status / Role

The classification of the actor or subject that determines which rules, permissions, or standards apply.

- **Law:** Plaintiff vs. defendant, individual vs. entity, exempt vs. non-exempt employee
- **Medicine:** Pediatric vs. adult vs. geriatric, immunocompromised vs. healthy, inpatient vs. outpatient
- **Engineering:** Safety-critical vs. non-critical system, regulated vs. unregulated component
- **Finance:** Accredited vs. retail investor, hedger vs. speculator, domestic vs. foreign entity
- **Code:** Admin vs. user, authenticated vs. anonymous, first-party vs. third-party service

**Why it's a switch:** Status changes the entire applicable rule set, not just threshold values.

### 4. Constraints

Hard limits — resource, technical, legal, or operational — that eliminate entire solution classes.

- **Law:** Budget for litigation, client's risk tolerance, relationship the client wants to preserve
- **Medicine:** Drug allergies, contraindications, organ function, available equipment
- **Engineering:** Budget ceiling, timeline, regulatory approval windows, existing infrastructure
- **Finance:** Liquidity requirements, mandate restrictions, leverage limits, counterparty limits
- **Code:** Language version, library restrictions, latency budget, team skill set

**Why it's a switch:** A constraint does not make an answer "harder to achieve" — it eliminates it. Solutions that require a $50,000 budget when the constraint is $5,000 are not suboptimal recommendations; they are non-answers.

### 5. Objective Function

What the answer is optimizing *for* — the metric that defines "correct."

- **Law:** Win at trial vs. minimize cost vs. preserve relationship vs. establish precedent
- **Medicine:** Cure vs. symptom management vs. palliative vs. prevention
- **Engineering:** Maximize performance vs. minimize cost vs. maximize reliability vs. minimize time-to-market
- **Finance:** Maximize return vs. minimize risk vs. hedge exposure vs. meet regulatory capital ratios
- **Code:** Minimize latency vs. minimize cost vs. maximize maintainability vs. ship fastest

**Why it's a switch:** The "best" solution is incoherent without knowing what "best" means. The same patient condition warrants different treatment if the goal is maximum function vs. minimum side effects.

---

## Domain-Specific Catalogs

### Law

| Switch Variable | Example Values | Why It Matters |
|-----------------|---------------|----------------|
| Jurisdiction | Federal, state, specific circuit | Substantive law differs entirely |
| Case posture | Pre-litigation, discovery, trial, appeal | Available tactics change |
| Party position | Plaintiff, defendant, third-party | Burden of proof, strategy invert |
| Relationship | Adversarial, ongoing business, family | Settlement calculus changes |
| Goal | Win, settle, delay, establish precedent | Optimal moves differ |
| Entity type | Individual, LLC, corporation, trust | Tax and liability rules differ |
| Timeline | Before SOL, after SOL, pending deadline | Some options vanish entirely |
<div class="callout-key">
<strong>Key Point:</strong> **Diagnostic question:** "Would a lawyer in [different jurisdiction/posture/goal] give the opposite advice?"
</div>


**Diagnostic question:** "Would a lawyer in [different jurisdiction/posture/goal] give the opposite advice?"

### Medicine / Clinical

| Switch Variable | Example Values | Why It Matters |
|-----------------|---------------|----------------|
| Age cohort | Pediatric, adult, geriatric | Dosing, diagnostics, DDx differ |
| Acuity | Acute, subacute, chronic | Treatment urgency and modality |
| Immune status | Immunocompetent, immunocompromised | Infection risk, prophylaxis |
| Care setting | ICU, inpatient, outpatient, ED | Available interventions |
| Comorbidities | DM, CKD, hepatic dysfunction | Drug choice and dosing |
| Treatment goal | Curative, palliative, preventive | Defines the entire plan |
| Resource setting | High-resource, low-resource | Evidence base shifts |

**Diagnostic question:** "Would I treat this patient differently if one of these variables changed?"

### Software Engineering

| Switch Variable | Example Values | Why It Matters |
|-----------------|---------------|----------------|
| Language/runtime | Python 3.12, Node 20, Go 1.22 | Syntax, stdlib, concurrency model |
| Deployment target | AWS Lambda, GKE, bare metal, edge | Architecture pattern, cost model |
| Scale | Single-user, 100k RPS, 10M records | Data structure and algorithm choice |
| Team constraint | Solo, junior team, senior team | Complexity tolerance changes |
| Phase | Prototype, production, maintenance | Engineering tradeoffs flip |
| Compliance | HIPAA, PCI-DSS, SOC2 | Mandatory security patterns |
| Performance constraint | Interactive latency, batch, real-time | Algorithm class changes |

**Diagnostic question:** "Would I recommend a different architecture if the deployment target changed?"

### Finance / Trading

| Switch Variable | Example Values | Why It Matters |
|-----------------|---------------|----------------|
| Asset class | Equity, fixed income, commodity, FX, derivative | Pricing model, regulation |
| Time horizon | Intraday, swing, long-term | Signal validity, cost model |
| Entity type | Retail, accredited, institutional | Regulatory access, leverage |
| Market regime | Trending, mean-reverting, volatile | Strategy class changes |
| Objective | Alpha, hedge, income, capital preservation | Optimal instrument changes |
| Liquidity constraint | Liquid, illiquid, locked-up | Entire execution approach |
| Tax status | Taxable, tax-deferred, tax-exempt | After-tax optimization |

**Diagnostic question:** "Would this trade look completely different under a different regime or objective?"

### Business / Strategy

| Switch Variable | Example Values | Why It Matters |
|-----------------|---------------|----------------|
| Company stage | Pre-product, growth, mature, distressed | Strategy norms invert |
| Market position | Leader, challenger, niche, new entrant | Competitive playbook differs |
| Resource constraint | Bootstrapped, funded, public | Investment horizon changes |
| Customer type | SMB, enterprise, consumer | Sales motion, pricing model |
| Geography | US, EU, emerging market | Regulatory and cultural context |
| Competitive intensity | Monopoly, duopoly, fragmented | Pricing power and strategy |
| Time horizon | Quarterly, annual, 5-year | Metric definition changes |

---

## How to Identify Switch Variables for Any Question

Apply this three-step procedure before writing any prompt:
<div class="callout-insight">
<strong>Insight:</strong> Apply this three-step procedure before writing any prompt:
</div>


**Step 1 — List the five conditions that would change your answer.**

Ask: "If I imagine five different people asking this question, each from a different context, would they need different answers?" Name the axis of variation.

**Step 2 — Test for category.**

For each condition you named, classify it: jurisdiction, timing, status, constraint, or objective. If it doesn't fit, ask whether it's actually a switch variable or just a descriptive detail.

**Step 3 — Force rank by impact.**

For each switch variable, ask: "If this variable were different, would the answer change by 10%, 50%, or 100%?" A 100% change means the answer is categorical — add this variable first.

**Example — "How should I structure this contract?"**

| Condition | Category | Impact |
|-----------|----------|--------|
| Governing law (NY vs. CA vs. UK) | Jurisdiction | 100% — different courts, different default rules |
| Entity types of parties (LLC vs. Corp) | Status | 80% — liability structure, capacity to contract |
| Relationship ongoing vs. one-time | Objective | 70% — dispute resolution, renegotiation terms |
| Deal size ($5k vs. $5M) | Constraint | 50% — drafting complexity, reps/warranties scope |
| Timeline to sign | Timing | 30% — negotiation depth, due diligence |

Add governing law and entity types first. The model needs those to answer at all.

---

## The Diagnostic Question

For any domain, the fastest way to surface switch variables is:

> **"What are the top 5 conditions that would make the correct answer to this question completely different?"**

This question reframes prompt engineering from "give more detail" to "identify the branching conditions." Those two activities produce very different prompts — and very different answer quality.

---

## Common Mistakes

**Mistake 1 — Adding volume instead of switches.**
<div class="callout-key">
<strong>Key Point:</strong> **Mistake 1 — Adding volume instead of switches.**
</div>


Telling the model your company has 47 employees, was founded in 2019, and operates in 6 cities adds almost zero information to a legal question. These are not switch variables. The governing state, entity type, and transaction goal are.

**Mistake 2 — Treating all conditions as equal.**

Not every condition is a switch variable. "The client prefers bullet points" is a formatting preference — it narrows the presentation, not the answer. Jurisdiction narrows the *answer*. Treat these differently.

**Mistake 3 — Omitting the objective function.**

The objective is the most commonly missed switch variable because it feels implicit. "Maximize return" and "hedge risk" are both valid objectives for the same position — and they call for opposite actions. State the objective explicitly every time.

**Mistake 4 — Conflating status and constraint.**

"We have no budget" is a constraint. "We are a non-profit" is a status. Both matter, but they operate differently. Status changes which rules apply. Constraint eliminates solutions that require resources you don't have.

---


---

## Practice Questions

<div class="callout-info">
<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Visual Explanation" and why it matters in practice.

2. Given a real-world scenario involving switch variables: the conditions that actually matter, what would be your first three steps to apply the techniques from this guide?
</div>

## Summary

Switch variables are the few conditions that route reasoning to categorically different solution branches. There are five categories: jurisdiction, timing, status, constraints, and objective function. Every professional domain has a predictable catalog of them.

The diagnostic question — "What are the top 5 conditions that would make the correct answer completely different?" — is the fastest way to identify them for any question.

In the next guide, we examine how to rank switch variables using information gain: not all of them reduce uncertainty equally, and knowing which ones to add first is the core prompt engineering skill.

**Next:** Guide 2 — Information Gain: Why "More Detail" Is Not "Better Conditions"

---

## Cross-References

<a class="link-card" href="../notebooks/01_switch_variable_identifier.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

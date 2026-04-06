# Causal Model Selection: Which Design for Which Question

> **Reading time:** ~6 min | **Module:** 7 — Production Pipelines | **Prerequisites:** Modules 1-6

## Learning Objectives

By the end of this guide, you will be able to:
1. Map a research question to the appropriate causal design using a structured decision framework
2. Identify the data requirements for each design
3. Explain the key assumption for each design and assess its plausibility
4. Make and document the design choice with appropriate transparency
5. Plan the diagnostic checks required before reporting results

---

## 1. The Design Selection Problem

Choosing a causal design is the most consequential methodological decision in an empirical study. The choice determines:
- What can be identified causally
- Which assumptions you must defend
- What data you need
- How credible your results will be

There is no universal best design — each is appropriate for specific contexts. The goal is to use the most credible design available given your data and research question.

---

## 2. The Decision Framework

<div class="flow">
<div class="flow-step blue">1. Identify Variation Source</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Check Data Requirements</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step mint">3. Assess Key Assumption</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Run Diagnostics</div>

</div>

### Primary Dimension: Source of Variation

The first question is always: **where does the exogenous variation in treatment come from?**

| Source of Variation | Design |
|--------------------|--------|
| Explicit randomisation | RCT |
| Threshold/cutoff rule | RDD (sharp or fuzzy) |
| Policy timing with comparison group | DiD |
| Natural experiment (exogenous shifter) | IV |
| Policy timing, no comparison group | ITS |
| Historical donor units | Synthetic Control |

### The Decision Tree

```

START: Do you have random assignment?
├── YES → RCT (gold standard, stop here)
└── NO ↓

Does treatment assignment follow a threshold rule?
├── YES: Is compliance perfect (all above treated, none below)?
│   ├── YES → Sharp RDD
│   └── NO → Fuzzy RDD (= local IV)
└── NO ↓

Do you have panel data with a comparison group?
├── YES ↓
│   ├── Single treatment date → DiD (or Controlled ITS if long time series)
│   └── Staggered adoption → CS/SA staggered DiD
└── NO ↓

Do you have a valid instrument?
├── YES → IV (2SLS)
└── NO ↓

Do you have a time series with a clear intervention break?
├── YES → ITS
└── NO ↓

Do you have suitable donor units and pre-treatment period?
├── YES → Synthetic Control
└── NO → Cannot make causal claims from observational data
```

<div class="callout-key">

<strong>Key Point:</strong> The decision tree above is not about picking the "fanciest" method. It is about identifying which design has the most credible source of exogenous variation in your specific setting. A well-executed ITS beats a poorly-justified DiD every time.

</div>

<div class="callout-warning">

<strong>Warning:</strong> If you cannot articulate where your exogenous variation comes from in one sentence, your causal design is not ready. Go back to the question formulation stage.

</div>

---

## 3. Design-Specific Data Requirements

| Design | Minimum Data Requirements | Ideal Data |
|--------|--------------------------|-----------|
| RCT | Randomisation + outcome | High power, pre-registered |
| Sharp RDD | Running variable, known cutoff | Dense observations near cutoff |
| Fuzzy RDD | Running variable, actual treatment, cutoff | Strong first stage (F > 10) |
| DiD | Treatment indicator, time periods, panel | Long pre-period, many units |
| Staggered DiD | First treatment date per unit, long panel | Multiple cohorts |
| IV | Valid instrument, correlated with treatment | Strong first stage |
| ITS | Time series, known treatment date | Long pre-period (≥20 periods) |
| Synthetic Control | Donor units, long pre-period, few treated | Many donors, good pre-fit |

---

## 4. Assumptions and Their Plausibility

### How to Assess Assumption Plausibility

For each design, the primary assumption is:

**RDD — Continuity:**
- Can units precisely manipulate the running variable?
- Are there other variables that also change at the cutoff?
- Test: McCrary density test, covariate balance

**DiD — Parallel Trends:**
- Would treated and control units have had the same trend absent treatment?
- Are there time-varying confounders affecting groups differently?
- Test: Pre-trend test, event study

**IV — Exclusion Restriction:**
- Can you rule out any direct path from the instrument to the outcome?
- Is the instrument truly exogenous?
- Test: First stage F, Sargan J-test (if overidentified)

**ITS — No Confounding Events:**
- Did any other change occur at the treatment time?
- Is the pre-treatment trend a reasonable counterfactual?
- Test: Placebo time points, sensitivity to window choice

**Synthetic Control — Pre-Period Fit:**
- Can the donor units reproduce the treated unit's pre-period?
- Is the donor pool stable (no spillovers)?
- Test: Pre-period MSPE, jackknife falsification

---

## 5. Reporting the Design Choice

### What to Document

In every causal analysis, document:

1. **Research question:** Precisely state the causal question
2. **Design choice:** Name the design and justify it
3. **Key assumption:** State the assumption in plain language
4. **Evidence for assumption:** Pre-trend test, density test, etc.
5. **Alternative designs considered:** Why were they rejected?
6. **Limitations:** What could go wrong; how would it bias results

### Template

```markdown

## Causal Design

**Research Question:** What is the effect of [treatment] on [outcome] for [population]?

**Design:** [Name of design], implemented as [specific estimator].

**Justification:** [Reason this design is appropriate given the data structure].

**Key Assumption:** [State the assumption, e.g., "In the absence of the policy,
treated and control counties would have had parallel employment trends."]

**Evidence Supporting Assumption:** [Pre-trend test results, McCrary test, etc.]

**Potential Threats:** [List 2-3 most credible threats with expected bias direction].

**Robustness Checks Planned:** [List sensitivity analyses].
```

---

## 6. The Credibility Hierarchy

Not all designs are equally credible. From most to least credible in terms of internal validity:

1. **RCT** — randomisation eliminates all confounding
2. **Natural experiment** (sharp) — random-as-good variation eliminates confounding
3. **RDD** — quasi-random assignment near cutoff
4. **IV** — isolates exogenous variation; exclusion is the key
5. **DiD** — parallel trends is untestable; relies on comparability
6. **ITS** — trend extrapolation is the weakest counterfactual
7. **Observational regression** — assumes unconfoundedness; usually implausible

However, this hierarchy is not fixed. A poorly conducted RCT (attrition, non-compliance) can be less credible than a well-designed DiD. The implementation matters as much as the design class.

---

## 7. Decision Checklist

Before finalising your design choice:

- [ ] Is the causal question clearly defined?
- [ ] Have you identified all available data and its limitations?
- [ ] Have you considered all applicable designs?
- [ ] Can you defend the key assumption with evidence and economic reasoning?
- [ ] Have you identified the most credible threats and how they would affect the estimate?
- [ ] Have you planned the primary diagnostic checks?
- [ ] Is the estimand (ATT, ATE, LATE) appropriate for the policy question?
- [ ] Have you planned robustness checks for the key modelling choices?

---

## 8. Common Mistakes in Design Selection

| Mistake | Problem | Solution |
|---------|---------|---------|
| Defaulting to OLS without justifying unconfoundedness | OLS is only causal under very strong assumptions | Find a natural experiment or instrument |
| Using DiD without checking parallel trends | Parallel trends may be violated | Always plot and test pre-trends |
| Using RDD with high-order polynomials | Erratic near boundary | Use local linear; test with quadratic |
| Ignoring staggered adoption in DiD | TWFE biased with heterogeneous effects | Use Callaway-Sant'Anna or Sun-Abraham |
| Using weak instruments | IV biased toward OLS, wrong CI | Report F-stat; use AR inference if F < 10 |
| Using ITS without checking for confounding events | Other events may contaminate the estimate | Document all contemporaneous events |

---

## Summary

| Design | Best For | Key Assumption | Key Test |
|--------|---------|---------------|---------|
| RDD | Threshold policy | Continuity | McCrary + covariate balance |
| DiD | Panel with comparison | Parallel trends | Pre-trend event study |
| IV | Endogenous treatment | Exclusion | First stage F + J-test |
| ITS | Time series intervention | No confounders | Placebo time points |
| Synthetic Control | Aggregate treated unit | Pre-period fit | Jackknife falsifications |


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Causal Model Selection: Which Design for Which Question and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

---

**Next:** [02 — Reporting Causal Estimates](02_reporting_guide.md)

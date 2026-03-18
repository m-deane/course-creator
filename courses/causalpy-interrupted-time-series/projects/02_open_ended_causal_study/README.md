# Project 02 — Open-Ended Causal Study

**Course:** Causal Inference with CausalPy
**Type:** Portfolio project (open-ended)
**Estimated time:** 6–10 hours
**Output:** A self-contained causal analysis notebook suitable for a portfolio

## Overview

You choose a research question and a dataset. You conduct a complete causal analysis from scratch: design selection, assumption assessment, estimation, robustness, and reporting. The project demonstrates that you can apply causal inference methods independently on a real-world problem of your choice.

There is no single right answer. What matters is rigour: clear design choice justification, honest assumption assessment, and transparent communication of uncertainty.

## Research Question Requirements

Your research question must:

1. **Have a plausible causal mechanism** — not just a correlation worth investigating
2. **Be answerable with one of the five designs** — ITS, Synthetic Control, DiD, RDD, or IV
3. **Have data available** — real, publicly available data (not simulated)
4. **Not have a known perfect answer** — the causal effect must be genuinely uncertain

## Suggested Research Questions and Datasets

You may use any of the following or propose your own.

### Option A: Interrupted Time Series

**Question:** Did the introduction of smoking bans in restaurants and bars reduce hospital admissions for acute myocardial infarction (heart attacks)?

**Data:** OECD Health Statistics (monthly hospital admissions by country), WHO Global Health Observatory. Several countries introduced smoking bans at known dates (Ireland 2004, Scotland 2006, England 2007).

**Design:** Single-group ITS or controlled ITS using countries that had not yet adopted the ban as controls.

**Key reference:** Pell et al. (2008) "Smoke-free legislation and hospitalizations for acute coronary syndrome" — NEJM.

---

### Option B: Difference-in-Differences

**Question:** Did US states that expanded Medicaid under the ACA (2014) experience lower rates of uninsured hospitalizations?

**Data:** CDC WONDER (cause-specific hospitalizations by state), Kaiser Family Foundation (Medicaid expansion dates). Some states did not expand Medicaid — valid comparison group.

**Design:** Two-way fixed effects DiD with state and year fixed effects. Staggered DiD if using multiple expansion years (2014, 2015, 2016).

**Key reference:** Wherry & Miller (2016) "Early Coverage, Access, Utilization, and Health Effects Associated With the Affordable Care Act Medicaid Expansions."

---

### Option C: Regression Discontinuity

**Question:** Does receiving a merit scholarship improve university graduation rates?

**Data:** National Center for Education Statistics (NCES) IPEDS, or state-level education administrative data. Many scholarship programs have a GPA or test score cutoff.

**Design:** Sharp RDD with running variable = admissions score or GPA; cutoff = scholarship eligibility threshold.

**Key reference:** Dynarski (2008) "Building the Stock of College-Educated Labor" — Journal of Human Resources.

---

### Option D: Synthetic Control

**Question:** What was the effect of the 1989 German reunification on West Germany's GDP per capita?

**Data:** Penn World Tables 10.0 (freely available), containing GDP per capita for 183 countries from 1950 to 2019.

**Design:** Classic synthetic control (Abadie, Diamond, Hainmueller 2010). West Germany is the treated unit; OECD countries are the donor pool.

**Key reference:** Abadie, Diamond, Hainmueller (2010) "Synthetic Control Methods for Comparative Case Studies" — JASA.

---

### Option E: Instrumental Variables

**Question:** Does education increase earnings? (Returns to schooling)

**Data:** Card (1995) NLSY dataset available via `rdrobust` R package or from ICPSR. Quarter of birth as an instrument (Angrist & Krueger 1991).

**Design:** IV / 2SLS using proximity to college or quarter of birth as instruments. First-stage F-statistic must be reported.

**Key reference:** Card (1995) "Using Geographic Variation in College Proximity to Estimate the Return to Schooling."

---

### Option F: Your Own Question

If you have a question from your own field, use it. Requirements:
- State the research question precisely
- Identify the treatment (what is being intervened on) and outcome
- Explain why ordinary regression is insufficient (endogeneity, selection, etc.)
- Justify the design you chose and the identification assumption it rests on

## Deliverable Structure

Your notebook must contain these sections:

### Section 1: Research Question and Design Choice (1 page)

1. State the research question in one sentence
2. Identify: treatment, outcome, target population, estimand (ATT, ATE, or LATE)
3. Explain why ordinary regression is insufficient (confounding, selection, etc.)
4. Justify your design choice using the decision tree from Module 07 Guide 01
5. State the key identification assumption and why it is plausible
6. State the main threat to that assumption and what evidence you will provide

### Section 2: Data Description and Preparation

1. Data source, provenance, and any limitations
2. Sample size, time span, geographic/demographic scope
3. Outcome variable distribution (histogram or time-series plot)
4. Treatment variable description (binary, continuous, dates)
5. Any data cleaning or preparation decisions (with justification)

### Section 3: Assumption Diagnostics

Run the appropriate diagnostic for your design:

| Design | Required Diagnostic |
|--------|---------------------|
| ITS    | Plot pre-intervention trend; assess linearity assumption |
| SC     | Pre-treatment RMSPE; donor pool justification table |
| DiD    | Pre-trend plot (if multiple pre-periods); balance table |
| RDD    | McCrary density test; covariate balance at cutoff |
| IV     | First-stage F-statistic; discuss exclusion restriction |

Write 2–3 sentences interpreting each diagnostic.

### Section 4: Primary Estimation

1. Run the primary specification
2. Report: estimate, SE, 95% CI, sample size
3. Interpret the estimate in plain language (one sentence)
4. Convert to an interpretable effect size

### Section 5: Robustness Checks

Run at least three alternative specifications. Present as a robustness table.

For each alternative, write one sentence explaining what it tests and what the result implies for the reliability of the primary estimate.

### Section 6: Sensitivity Analysis

For your design, apply the relevant sensitivity analysis:

| Design | Sensitivity Analysis |
|--------|---------------------|
| DiD    | Roth-style: estimate under parallel trends violations |
| RDD    | Bandwidth sensitivity table (h/2, h, 2h, donut) |
| IV     | Sensitivity to partial exclusion restriction violation |
| ITS    | Sensitivity to omission of seasonal controls |
| SC     | Leave-one-out donor sensitivity |

Report: "Our conclusion holds as long as [assumption violation] is less than [X]."

### Section 7: Limitations and Conclusions

1. **Main limitation:** State the most credible threat to your identification assumption
2. **Direction of bias:** If the assumption were violated, would the estimate be too large or too small?
3. **Magnitude check:** Is the effect size plausible given what you know about the subject?
4. **Conclusion:** One paragraph summarising the causal finding with appropriate uncertainty language

## Assessment Criteria

This project is not graded. Use this checklist to evaluate your own work:

- [ ] Research question is clearly stated and causal (not correlational)
- [ ] Design choice is justified with explicit reference to the identification assumption
- [ ] The estimand (ATT, ATE, LATE) is stated and the distinction is noted
- [ ] Diagnostics are reported with interpretation, not just code output
- [ ] Primary estimate is reported with SE, CI, and an interpretable effect size
- [ ] Robustness table has at least three specifications with different SE methods or sample restrictions
- [ ] Sensitivity analysis answers "what would overturn this conclusion?"
- [ ] Limitations section is honest and identifies the direction of potential bias
- [ ] Notebook runs end-to-end without errors

## Common Mistakes to Avoid

**Choosing an impossible estimand.** DiD estimates the ATT, not the ATE. RDD estimates a LATE at the cutoff, not the average effect. If your question requires the ATE and you can only run a DiD, acknowledge this explicitly.

**Omitting the pre-trend test.** If you have multiple pre-treatment periods, always run the event study and the joint pre-trend test. If you only have one pre-period, discuss what you would need to verify parallel trends.

**Reporting a single specification.** A single estimate with a p-value is not a causal analysis. Always show that the result holds across alternative specifications.

**Conflating statistical significance with practical significance.** Report the effect size in interpretable units. A statistically significant effect might be substantively negligible. A large effect might be insignificant due to small samples.

**Not discussing the exclusion restriction for IV.** "Z affects Y only through D" is the key IV assumption. It cannot be tested, but you must argue it is plausible, and you must discuss what would happen if it were violated.

## Example Output

A complete example of this project's expected output is shown in:

- [01 — Guided Causal Study](../01_guided_causal_study/) — Card & Krueger minimum wage DiD
- [Module 07 Notebook 02](../../modules/module_07_production_pipelines/notebooks/02_causal_report_generation.ipynb) — automated report generation

Your report should match the structure and depth of the automated report in Module 07 Notebook 02, with the addition of the design choice memo (Section 1) and limitations (Section 7).

---

**Previous Project:** [01 — Guided Causal Study](../01_guided_causal_study/)

**Return to Course:** [Course Root](../../)

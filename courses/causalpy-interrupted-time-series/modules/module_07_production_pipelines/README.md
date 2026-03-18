# Module 07 — Production Pipelines

**Course:** Causal Inference with CausalPy
**Estimated time:** 3 hours

## Overview

This module covers everything that happens after estimation: selecting the right causal design, reporting results with full uncertainty quantification, and deploying causal models into production pipelines that run automatically and flag assumption violations.

## Learning Objectives

By the end of this module, you will be able to:

1. Select the appropriate causal design for a given research setting using a systematic decision framework
2. Report treatment effect estimates with interpretable effect sizes, credible intervals, and robustness evidence
3. Build a production-grade causal pipeline with data validation, assumption monitoring, and reproducibility logging
4. Define retraining triggers that distinguish routine data updates from fundamental design changes
5. Generate automated causal analysis reports with structured output

## Contents

### Guides

| File | Topic | Reading Time |
|------|-------|-------------|
| `guides/01_model_selection_guide.md` | Design selection decision tree and credibility framework | 20 min |
| `guides/01_model_selection_slides.md` | Companion slide deck | — |
| `guides/02_reporting_guide.md` | Effect sizes, intervals, robustness tables, sensitivity analysis | 20 min |
| `guides/02_reporting_slides.md` | Companion slide deck | — |
| `guides/03_deployment_guide.md` | Production pipeline architecture, monitoring, retraining | 25 min |
| `guides/03_deployment_slides.md` | Companion slide deck | — |

### Notebooks

| File | Topic | Time |
|------|-------|------|
| `notebooks/01_model_selection_workflow.ipynb` | Design selection for three real-world research cases | 15 min |
| `notebooks/02_causal_report_generation.ipynb` | Automated report with effect sizes, robustness, and sensitivity | 15 min |

### Exercises

| File | Topic |
|------|-------|
| `exercises/01_pipeline_self_check.py` | Data validation, pre-trend testing, drift monitoring, retraining logic |

## Prerequisites

- Module 01 — Interrupted Time Series
- Module 02 — Synthetic Control
- Module 03 — DiD Review (optional)
- Module 04 — Difference-in-Differences
- Module 05 — Regression Discontinuity
- Module 06 — Instrumental Variables

## Key Concepts

### Design Selection Decision Tree

```
Do you have a clear intervention with a known time of change?
├── Yes → Does treatment assignment follow a threshold rule?
│         ├── Yes → Is compliance with the rule partial?
│         │         ├── Yes → Fuzzy RDD (or IV with RD first stage)
│         │         └── No  → Sharp RDD
│         └── No  → Is there a valid comparison group?
│                   ├── Yes → DiD (or staggered DiD if multiple cohorts)
│                   └── No  → ITS (interrupted time series)
└── No  → Is there an instrument that affects treatment but not outcome directly?
           ├── Yes → IV / 2SLS
           └── No  → Consider synthetic control or quasi-experimental design
```

### Reporting Standards

Every causal estimate must include:
- Point estimate with standard error or HDI
- 95% confidence or credible interval
- At least one interpretable effect size (Cohen's d or % of baseline)
- Sample size and estimation context
- Estimand stated explicitly (ATT, ATE, LATE, CATE)
- Robustness table with at least three specifications
- Primary diagnostic (pre-trend plot, McCrary density test, first-stage F-stat)
- Main limitation and direction of bias under assumption violation

### Production Pipeline Stages

1. **Data Validation** — schema, missing values, group completeness
2. **Assumption Checks** — pre-trend test (DiD), density test (RDD), F-stat (IV)
3. **Estimation** — model fit, extract estimate and uncertainty
4. **Diagnostics** — convergence, robustness, sensitivity
5. **Output and Logging** — results manifest with data hash, code version, run ID
6. **Monitoring** — compare estimates to baseline, alert on drift

### Retraining vs. Redesign

| Trigger | Response |
|---------|----------|
| New month of data | Automated re-estimate |
| Pre-trend test fails | Flag for human review |
| Estimate drifts > 2 SEs | Human investigation |
| Comparison group policy changes | Evaluate whether design is still valid |
| Treatment rule changes | Redesign (human decision required) |

## Navigation

**Previous:** [Module 06 — Instrumental Variables](../module_06_instrumental_variables/)

**Next:** [Course Projects](../../../projects/)

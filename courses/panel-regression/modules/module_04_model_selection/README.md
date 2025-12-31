# Module 4: Model Selection and Diagnostics

## Overview

Choose between pooled OLS, fixed effects, and random effects using formal tests and diagnostics. Learn the Hausman test, F-tests for effects, and specification testing.

**Time Estimate:** 6-8 hours

## Learning Objectives

By completing this module, you will:
1. Apply the Hausman test for FE vs RE
2. Test for the presence of entity/time effects
3. Diagnose serial correlation and heteroskedasticity
4. Make informed model selection decisions

## Contents

### Guides
- `01_hausman_test.md` - FE vs RE selection
- `02_specification_tests.md` - F-tests, Breusch-Pagan
- `03_diagnostic_checks.md` - Residual analysis

### Notebooks
- `01_model_selection.ipynb` - Complete testing workflow
- `02_diagnostics.ipynb` - Checking assumptions

## Key Concepts

### The Hausman Test

**Null hypothesis**: $H_0: Cov(\mu_i, X_{it}) = 0$ (RE is consistent)

**Test statistic**:
$$H = (\hat{\beta}_{FE} - \hat{\beta}_{RE})'[Var(\hat{\beta}_{FE}) - Var(\hat{\beta}_{RE})]^{-1}(\hat{\beta}_{FE} - \hat{\beta}_{RE})$$

$$H \sim \chi^2_k$$ under $H_0$

**Interpretation**:
- Reject $H_0$ → Use Fixed Effects
- Fail to reject → Random Effects is valid (and more efficient)

### Testing for Effects

| Test | Null Hypothesis | Use |
|------|-----------------|-----|
| F-test (entity) | All $\alpha_i$ equal | FE vs Pooled |
| F-test (time) | All $\lambda_t$ equal | Time FE needed? |
| Breusch-Pagan LM | $\sigma^2_\mu = 0$ | RE vs Pooled |

### Decision Flowchart

```
Start
  ↓
F-test: Entity effects significant?
  ├── No → Use Pooled OLS
  └── Yes ↓
        Hausman test: FE ≠ RE?
          ├── Yes → Use Fixed Effects
          └── No → Use Random Effects
```

## Prerequisites

- Module 0-3 completed
- Hypothesis testing background

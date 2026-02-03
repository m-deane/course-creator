# Glossary: Panel Regression Models with Fixed and Random Effects

## A

**Arellano-Bond Estimator**
: GMM estimator for dynamic panel models. Uses lagged differences as instruments for endogenous regressors.

**Attrition**
: When units drop out of the panel over time. Can cause bias if dropout is related to outcomes.

**Autocorrelation**
: Correlation of error terms across time within the same entity. Common in panel data; requires clustered standard errors.

---

## B

**Balanced Panel**
: Panel where all entities are observed for all time periods. No missing observations.

**Between Estimator**
: Regression using entity-level averages. Exploits cross-sectional variation only.

**Between-Group Variation**
: Differences in average outcomes across entities. Contrasts with within-group variation.

**BLUE (Best Linear Unbiased Estimator)**
: OLS is BLUE under Gauss-Markov assumptions. Fixed effects may sacrifice efficiency for unbiasedness.

---

## C

**Clustered Standard Errors**
: Standard errors adjusted for within-cluster correlation. Essential for panel data to avoid overstating precision.

**Composite Error**
: $u_{it} = \mu_i + \epsilon_{it}$ in random effects. Sum of entity effect and idiosyncratic error.

**Cross-Section**
: Observation of multiple entities at a single time point. Panel data combines repeated cross-sections.

**Cross-Sectional Dependence**
: Correlation of errors across entities at the same time. Can arise from common shocks.

---

## D

**Difference-in-Differences (DiD)**
: Quasi-experimental method comparing treatment and control groups before/after intervention. Natural application of panel methods.

**Differencing (First Difference)**
: Transformation that eliminates entity fixed effects: $\Delta y_{it} = y_{it} - y_{i,t-1}$.

**Dummy Variable Trap**
: Including all entity dummies plus intercept causes perfect collinearity. Drop one dummy or intercept.

**Dynamic Panel**
: Model including lagged dependent variable: $y_{it} = \rho y_{i,t-1} + X_{it}\beta + \mu_i + \epsilon_{it}$. Requires special estimators (Arellano-Bond, Blundell-Bond).

---

## E

**Endogeneity**
: When regressors are correlated with the error term. Panel methods can address via fixed effects or IV.

**Entity**
: Cross-sectional unit in panel data. Examples: firm, country, individual, stock.

**Entity Effect**
: Time-invariant unobserved characteristic of an entity. $\mu_i$ in random effects, $\alpha_i$ in fixed effects.

**Exogeneity**
: Regressors uncorrelated with error term. Key assumption for consistent estimation.

---

## F

**F-Test**
: Test for joint significance of fixed effects. Large F-statistic suggests fixed effects are needed.

**First Difference**
: See Differencing.

**Fixed Effects (FE)**
: Model allowing each entity its own intercept. Controls for time-invariant unobserved heterogeneity.

**Fixed Effects Estimator**
: Estimator that eliminates entity effects via demeaning (within transformation). Also called within estimator.

---

## G

**Gauss-Markov Assumptions**
: Conditions under which OLS is BLUE: linearity, no perfect collinearity, zero conditional mean, homoskedasticity, no autocorrelation.

**Generalized Least Squares (GLS)**
: Estimator that accounts for heteroskedasticity and autocorrelation. Used in random effects.

**GMM (Generalized Method of Moments)**
: Estimation method using moment conditions. Arellano-Bond and Blundell-Bond are GMM estimators.

---

## H

**Hausman Test**
: Test comparing fixed effects and random effects estimators. Rejection suggests fixed effects is appropriate.

**Heterogeneity**
: Differences across entities. Unobserved heterogeneity is the rationale for panel methods.

**Heteroskedasticity**
: Non-constant error variance. Common in panel data; use robust standard errors.

---

## I

**Idiosyncratic Error**
: Time-varying error term $\epsilon_{it}$ specific to entity i at time t.

**Incidental Parameters Problem**
: In short panels, estimating many entity fixed effects can bias coefficients on time-varying regressors.

**Instrumental Variable (IV)**
: Variable correlated with endogenous regressor but not with error. Used to achieve consistent estimation.

**Intraclass Correlation**
: Correlation of outcomes within the same entity over time. Measures importance of entity effects.

---

## L

**Lagged Dependent Variable**
: Including $y_{i,t-1}$ as regressor creates dynamic panel model. OLS and standard FE are biased; use GMM.

**Least Squares Dummy Variable (LSDV)**
: Fixed effects estimated by including dummy for each entity. Equivalent to within estimator but computationally expensive.

**Longitudinal Data**
: Another term for panel data, emphasizing time dimension.

**Long Format**
: Panel data structure with one row per entity-time observation. Standard for regression.

---

## M

**Micro-Panel**
: Panel with many entities (large N) and few time periods (small T). Typical in applied microeconomics.

**Mundlak Approach**
: Including entity-level means of regressors as controls. Relaxes random effects assumption.

---

## N

**Nickell Bias**
: Bias in fixed effects estimator when lagged dependent variable is included and T is small.

---

## O

**One-Way Fixed Effects**
: Controlling for entity effects only (not time effects).

**OLS (Ordinary Least Squares)**
: Standard regression estimator. Pooled OLS ignores panel structure; biased if entity effects are correlated with regressors.

---

## P

**Panel Data**
: Data with observations on multiple entities over multiple time periods. Combines cross-sectional and time series dimensions.

**Pooled OLS**
: OLS regression treating all observations as independent, ignoring panel structure. Inconsistent if entity effects are correlated with X.

**Pseudo R-squared**
: Measure of model fit for panel regressions. Interpretation differs from cross-sectional R².

---

## R

**Random Effects (RE)**
: Model where entity effects are treated as random draws uncorrelated with regressors. Uses GLS estimation.

**Robust Standard Errors**
: Standard errors adjusted for heteroskedasticity and/or clustering. Essential for valid inference in panels.

---

## S

**Serial Correlation**
: See Autocorrelation.

**Strict Exogeneity**
: $E[\epsilon_{it}|X_{i1},...,X_{iT}]=0$ for all t. Stronger than contemporaneous exogeneity. Required for fixed effects consistency.

---

## T

**T (Time Periods)**
: Number of time periods in the panel. Small T (< 10) common in micro-panels; large T in macro-panels.

**Time Effects**
: Period-specific intercepts capturing common shocks affecting all entities at time t.

**Time-Invariant**
: Variable that doesn't change over time for an entity (e.g., gender, country). Cannot be estimated with fixed effects.

**Two-Way Fixed Effects**
: Controlling for both entity and time effects.

---

## U

**Unbalanced Panel**
: Panel where some entities are not observed for all time periods. Missing observations.

**Unobserved Heterogeneity**
: Entity-specific characteristics not included in X. Source of bias in pooled OLS.

---

## V

**Variance Components**
: In random effects, the variance of entity effects ($\sigma_\mu^2$) and idiosyncratic errors ($\sigma_\epsilon^2$).

---

## W

**White Standard Errors**
: Heteroskedasticity-robust standard errors. Named after Halbert White.

**Wide Format**
: Panel data with one row per entity, columns for each time period. Must reshape to long format for regression.

**Within Estimator**
: See Fixed Effects Estimator.

**Within Transformation (Demeaning)**
: Subtracting entity-specific means from all variables. Eliminates fixed effects.

**Within-Group Variation**
: Changes in variables over time within entities. Exploited by fixed effects.

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| $i$ | Entity index (1 to N) |
| $t$ | Time period index (1 to T) |
| $N$ | Number of entities |
| $T$ | Number of time periods |
| $y_{it}$ | Outcome for entity i at time t |
| $X_{it}$ | Vector of covariates |
| $\beta$ | Coefficient vector |
| $\alpha_i$ | Fixed effect for entity i |
| $\mu_i$ | Random effect for entity i |
| $\lambda_t$ | Time fixed effect |
| $\epsilon_{it}$ | Idiosyncratic error |
| $u_{it}$ | Composite error ($\mu_i + \epsilon_{it}$) |
| $\bar{y}_i$ | Entity-level mean: $\frac{1}{T}\sum_t y_{it}$ |
| $\ddot{y}_{it}$ | Demeaned variable: $y_{it} - \bar{y}_i$ |

---

## Model Comparison

| Model | Transformation | Assumes | Pros | Cons |
|-------|----------------|---------|------|------|
| **Pooled OLS** | None | No entity effects | Simple, efficient | Biased if $\mu_i$ correlated with X |
| **Fixed Effects** | Demeaning | $\mu_i$ arbitrary | Controls unobserved heterogeneity | Cannot estimate time-invariant X |
| **Random Effects** | GLS | $E[\mu_i\|X]=0$ | More efficient, can estimate time-invariant X | Inconsistent if assumption violated |
| **First Difference** | Differencing | Strict exogeneity | Eliminates FE, simple | Loses one period, inefficient |

---

## Hausman Test Interpretation

| Result | Conclusion | Action |
|--------|------------|--------|
| Reject H₀ | FE and RE differ significantly | Use Fixed Effects |
| Fail to Reject | FE and RE similar | Use Random Effects (more efficient) |

---

## Common Acronyms

| Acronym | Full Name |
|---------|-----------|
| FE | Fixed Effects |
| RE | Random Effects |
| OLS | Ordinary Least Squares |
| GLS | Generalized Least Squares |
| LSDV | Least Squares Dummy Variable |
| GMM | Generalized Method of Moments |
| IV | Instrumental Variable |
| DiD | Difference-in-Differences |
| BLUE | Best Linear Unbiased Estimator |
| WLS | Weighted Least Squares |

---

## Python/R Implementations

### Python: linearmodels

```python
from linearmodels import PanelOLS
from linearmodels import RandomEffects

# Fixed Effects
fe_model = PanelOLS(y, X, entity_effects=True)
fe_result = fe_model.fit(cov_type='clustered', cluster_entity=True)

# Random Effects
re_model = RandomEffects(y, X)
re_result = re_model.fit()
```

### R: plm

```r
library(plm)

# Fixed Effects
fe_model <- plm(y ~ x1 + x2, data=panel_df, index=c("entity","time"),
                model="within")

# Random Effects
re_model <- plm(y ~ x1 + x2, data=panel_df, index=c("entity","time"),
                model="random")

# Hausman Test
phtest(fe_model, re_model)
```

---

## Standard Error Options

| Type | Accounts For | When to Use |
|------|--------------|-------------|
| **Standard** | None | Rarely appropriate for panel data |
| **Heteroskedasticity-robust** | $Var(\epsilon_{it})$ varies | Suspected non-constant variance |
| **Clustered (entity)** | $Cov(\epsilon_{it}, \epsilon_{is}) \neq 0$ | Always for panel data |
| **Two-way clustered** | Clustering by entity AND time | Large N and T, common shocks |
| **HAC** | Heteroskedasticity and autocorrelation | Time series component important |

---

## Typical Panel Regression Workflow

1. **Explore data structure**
   - Check balanced/unbalanced
   - Identify time-invariant vs. time-varying regressors
   - Plot trends by entity

2. **Estimate pooled OLS**
   - Baseline (likely biased)
   - Use robust standard errors

3. **Test for fixed effects**
   - F-test for joint significance
   - If significant, proceed with FE

4. **Estimate fixed effects**
   - One-way (entity) or two-way (entity + time)
   - Clustered standard errors

5. **Estimate random effects**
   - For comparison

6. **Hausman test**
   - Choose between FE and RE

7. **Diagnostic checks**
   - Residual plots
   - Test for autocorrelation (Wooldridge test)
   - Test for heteroskedasticity (Breusch-Pagan)

8. **Report results**
   - Preferred specification
   - Robustness checks
   - Interpret coefficients correctly

---

## Interpretation Tips

- **FE coefficient:** Effect of within-entity change in X on within-entity change in Y
- **RE coefficient:** Average effect across entities (if RE assumptions hold)
- **Clustered SE:** Account for correlation within entities; wider than standard SE
- **Time-invariant X:** Cannot estimate with FE; use RE or between estimator if needed

---

## Common Pitfalls

| Pitfall | Problem | Solution |
|---------|---------|----------|
| **Not clustering SE** | Overstated precision | Always cluster by entity (minimum) |
| **Using FE when T=2** | Imprecise estimates | Consider first-difference or DiD |
| **Including time-invariant X with FE** | Dropped or collinear | Use RE or Mundlak approach |
| **Ignoring autocorrelation** | Invalid inference | Use HAC or clustered SE |
| **Assuming RE when inappropriate** | Inconsistent estimates | Hausman test; default to FE if uncertain |

---

*This glossary covers panel econometrics terminology. See module guides for detailed mathematical derivations and applications.*

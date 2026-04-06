---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Interactive Regression Models

## Module 6: Binary Treatments with DML
### Double/Debiased Machine Learning

<!-- Speaker notes: This deck extends DML to binary treatments using the Interactive Regression Model. We cover the AIPW score, propensity score estimation, ATE vs ATTE, and the doubleml.DoubleMLIRM implementation. The commodity example is sanctions impact on shipping freight rates. -->

---

## In Brief

PLR assumes a **constant** treatment effect. IRM allows effects to **vary** with covariates.

> **When to use IRM:** Binary treatment (yes/no) where you expect heterogeneous effects.

IRM estimates both ATE and ATTE using doubly robust AIPW scores.

<!-- Speaker notes: The PLR model from Module 05 works well for continuous treatments with constant effects. But when treatment is binary — like whether a shipping route is sanctioned — you want the effect to potentially vary across units. Some routes may be more affected than others based on their characteristics. The IRM handles this by modelling both the outcome function and the propensity score. -->

<div class="callout-info">
Info:  treatment effect. IRM allows effects to 
</div>

---

## PLR vs IRM

<div class="columns">
<div>

### PLR
- $Y = \theta D + g_0(X) + \epsilon$
- $\theta$ is **constant** for all $X$
- Continuous $D$
- Simpler estimation

</div>
<div>

### IRM
- $Y = g_0(D, X) + \epsilon$
- Effect varies with $X$
- **Binary** $D \in \{0, 1\}$
- ATE and ATTE
- Propensity score $m_0(X) = P(D=1|X)$

</div>
</div>

<!-- Speaker notes: The key difference is that IRM models the full outcome function g0(D,X) rather than assuming a linear-in-D structure. This means the treatment effect g0(1,X) minus g0(0,X) can vary with X. The IRM requires estimating the propensity score m0(X) as an additional nuisance function. The AIPW score combines the outcome model and propensity score in a doubly robust way. -->

---

## The AIPW Score (ATE)

$$\psi_{ATE} = \underbrace{g(1, X) - g(0, X)}_{\text{outcome model}} + \underbrace{\frac{D(Y - g(1,X))}{m(X)}}_{\text{treated correction}} - \underbrace{\frac{(1-D)(Y - g(0,X))}{1 - m(X)}}_{\text{control correction}} - \theta$$

**Doubly robust:** Consistent if EITHER $\hat{g}$ or $\hat{m}$ is correct.

<!-- Speaker notes: The AIPW score has three components. The first is the predicted treatment effect from the outcome model. The second and third are corrections that account for imperfect outcome modelling, weighted by the inverse propensity. The score is doubly robust: if either the outcome model or the propensity score is consistently estimated, the ATE is consistent. In DML, both are estimated with ML and cross-fitting, giving valid inference. -->

---

## Commodity Example: Sanctions on Freight Rates

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
graph TD
    X["Controls X<br/>Route distance, vessel type,<br/>cargo, insurance, ports"] --> D["Treatment D<br/>Sanctions<br/>(binary: 0 or 1)"]
    X --> Y["Outcome Y<br/>Freight rate<br/>premium"]
    D -->|"Heterogeneous effect"| Y
```

**ATE:** Average effect across all routes
**ATTE:** Average effect on sanctioned routes specifically

<!-- Speaker notes: Sanctions affect different routes differently. A route through the Strait of Hormuz under sanctions sees a larger freight premium than a route in the Mediterranean, because of the strategic chokepoint. The ATE averages across all routes, while ATTE focuses on the routes that were actually sanctioned. ATTE is more relevant for policy evaluation: what was the actual impact on sanctioned routes? -->

<div class="callout-key">
Key Point:  Average effect across all routes

</div>

---

## Code: IRM with `doubleml`

```python
from doubleml import DoubleMLIRM
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

irm_ate = DoubleMLIRM(dml_data,
    ml_g=RandomForestRegressor(200),    # Outcome model
    ml_m=RandomForestClassifier(200),   # Propensity score
    score='ATE', n_folds=5,
    trimming_threshold=0.05)
irm_ate.fit()

irm_atte = DoubleMLIRM(dml_data,
    ml_g=RandomForestRegressor(200),
    ml_m=RandomForestClassifier(200),
    score='ATTE', n_folds=5)
irm_atte.fit()
```

<!-- Speaker notes: The DoubleMLIRM API is similar to DoubleMLPLR. The key differences are: ml_g is the outcome model (predicts Y from D and X), ml_m is the propensity score model (classifier, predicts D from X), and the score parameter chooses between ATE and ATTE. The trimming_threshold drops observations with extreme propensity scores to prevent numerical instability. Use RandomForestClassifier for ml_m since D is binary. -->

---

## Propensity Score Diagnostics

Good overlap is essential for IRM:

| Diagnostic | Good | Bad |
|-----------|------|-----|
| Overlap | Both groups span [0.1, 0.9] | Treated near 1, control near 0 |
| Balance | Covariates similar after weighting | Large imbalances remain |
| Trimming | < 5% observations dropped | > 20% dropped |

> ⚠️ If overlap is poor, consider PLR or bounds analysis instead.

<!-- Speaker notes: The propensity score overlap condition is crucial for IRM. If treated and untreated units have very different propensity scores (poor overlap), the inverse probability weights become extreme and the estimator is unstable. Diagnostic plots should show substantial overlap in propensity score distributions. If overlap is poor, the IRM may not be appropriate and you should consider alternative approaches like the PLR or partial identification bounds. -->

---

## When to Use IRM vs PLR

| Criterion | PLR | IRM |
|-----------|-----|-----|
| Treatment type | Continuous | **Binary** |
| Effect heterogeneity | Constant $\theta$ | Varies with $X$ |
| Nuisance: outcome | $E[Y|X]$ (regressor) | $E[Y|D,X]$ (regressor) |
| Nuisance: treatment | $E[D|X]$ (regressor) | $P(D=1|X)$ (**classifier**) |
| Estimands | ATE only | **ATE + ATTE** |
| Overlap required | No | **Yes** |

<!-- Speaker notes: This decision table helps practitioners choose between PLR and IRM. PLR is simpler and works for continuous treatments. IRM is needed when treatment is binary and you want to estimate both ATE and ATTE. The key practical difference is that IRM requires a classifier for the propensity score and needs reasonable overlap between treated and untreated groups. If overlap is poor, IRM can be unstable. -->

<div class="callout-insight">
Insight: ) |
| Estimands | ATE only | 
</div>

---

## ATE vs ATTE: Which to Report?

**ATE** (Average Treatment Effect): average over **all** units
$$\theta_{ATE} = E[Y(1) - Y(0)]$$

**ATTE** (Average Treatment Effect on Treated): average over **treated** units
$$\theta_{ATTE} = E[Y(1) - Y(0) | D=1]$$

> **Commodity example:** Sanctions raise freight rates by $1.50/ton (ATE) across all routes, but by $2.20/ton (ATTE) on actually sanctioned routes — because sanctioned routes are inherently more vulnerable.

Report ATTE for policy evaluation, ATE for counterfactual analysis.

<!-- Speaker notes: The choice between ATE and ATTE depends on the question. If a regulator asks what would happen if we sanctioned all routes, ATE is the answer. If they ask what was the impact on routes we actually sanctioned, ATTE is the answer. In commodity markets, ATTE is often more relevant because the treated units are self-selected — OPEC cuts production when conditions warrant it, sanctions target specific routes for strategic reasons. The difference between ATE and ATTE reveals selection effects. -->

<div class="callout-warning">
Warning:  (Average Treatment Effect): average over 
</div>

---

## Propensity Score Trimming in Practice

Extreme propensity scores ($m(X)$ near 0 or 1) create numerical instability:

$$\text{AIPW weight} = \frac{D}{m(X)} \quad \text{explodes when } m(X) \to 0$$

| Trimming threshold | Effect | When to use |
|:-----------------:|--------|-------------|
| 0.01 | Minimal trimming | Good overlap |
| 0.05 | Standard | Most applications |
| 0.10 | Aggressive | Poor overlap |

> Check how many observations are trimmed. If > 10%, overlap may be insufficient for IRM.

<!-- Speaker notes: Propensity score trimming drops observations where the propensity score is very close to 0 or 1. These observations receive extreme inverse probability weights that destabilise the estimator. The standard threshold of 0.05 means observations with propensity score below 0.05 or above 0.95 are excluded. If trimming removes more than 10% of observations, the overlap condition is likely violated and you should reconsider whether IRM is appropriate. Consider PLR instead, or use bounds analysis. -->

---

## Connections

<div class="columns">
<div>

### Builds On
- Module 05: PLR with `doubleml`
- Propensity score methods
- AIPW estimation

</div>
<div>

### Leads To
- Module 08: CATE heterogeneity
- Module 09: Production pipeline

</div>
</div>

<!-- Speaker notes: IRM extends PLR to binary treatments with heterogeneous effects. Module 08 takes this further by estimating individual-level conditional average treatment effects using econml. Module 09 incorporates IRM into the production pipeline with automated diagnostics for propensity score quality. -->

---

## Visual Summary

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    PLR["PLR: Constant θ"] --> IRM["IRM: Heterogeneous effects"]
    IRM --> ATE["ATE: All units"]
    IRM --> ATTE["ATTE: Treated units"]
    IRM --> AIPW["AIPW Score:<br/>Doubly robust"]
    AIPW --> Valid["Valid inference<br/>with ML nuisance"]
```

<!-- Speaker notes: IRM generalises PLR to binary treatments with heterogeneous effects. It estimates both ATE and ATTE using the doubly robust AIPW score. The propensity score is an additional nuisance function estimated with a classifier. Diagnostics for overlap and trimming are essential for reliable results. -->

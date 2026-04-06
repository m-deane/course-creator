---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Instrumental Variables with DML

## Module 7: PLIV for Endogenous Treatments
### Double/Debiased Machine Learning

<!-- Speaker notes: This deck covers the Partially Linear Instrumental Variables model. When treatment is endogenous (correlated with unobservable confounders), standard DML fails. PLIV uses an instrument that affects the outcome only through the treatment, with ML first stages replacing traditional linear 2SLS. -->

---

## In Brief

Standard DML assumes **all confounders are observed**. When unobservable confounders exist, use PLIV.

> **Key idea:** An instrument $Z$ that affects $D$ but not $Y$ directly identifies the causal effect even with unobserved confounding.

ML first stages handle nonlinear instrument-treatment relationships.

<!-- Speaker notes: The selection-on-observables assumption in Modules 02-06 requires that all confounders are in X. In many commodity markets, this is unrealistic — market sentiment, insider information, and private expectations are unobserved confounders. PLIV addresses this by using an instrument: a variable that affects the treatment but has no direct effect on the outcome. The classic example is weather as an instrument for agricultural supply. -->

<div class="callout-info">
Info: all confounders are observed
</div>

---

## The IV DAG

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
graph TD
    Z["Z: Instrument<br/>(weather anomaly)"] -->|"Relevance"| D["D: Treatment<br/>(shipping volume)"]
    U["U: Unobserved<br/>(market sentiment)"] -->|"Confounding"| D
    U -->|"Confounding"| Y["Y: Outcome<br/>(price spread)"]
    D -->|"θ: causal"| Y
    Z -.->|"Exclusion: NO direct effect"| Y
```

<!-- Speaker notes: The DAG shows the three key elements. Z (weather) affects D (shipping volume) through relevance. U (market sentiment) confounds both D and Y. The exclusion restriction says Z does not directly affect Y — weather affects prices only through its effect on shipping volume. If these conditions hold, PLIV identifies the causal effect theta even though U is unobserved. -->

---

## PLIV Model

$$Y = \theta D + g_0(X) + \epsilon, \quad E[\epsilon | Z, X] = 0$$
$$D = r_0(X) + h_0(Z, X) + V$$

Three nuisance functions:
- $g_0(X)$: controls' effect on $Y$ (ml_l)
- $m_0(X)$: controls' effect on $D$ (ml_m)
- $r_0(X, Z)$: instrument + controls' effect on $D$ (ml_r)

<!-- Speaker notes: The PLIV model extends PLR with an instrument equation. The key difference from PLR is the exclusion restriction: epsilon is mean-independent of Z given X. This means Z only affects Y through D. The doubleml library estimates three nuisance functions: ml_l for the outcome, ml_m for the treatment as a function of controls only, and ml_r for the treatment as a function of controls and the instrument. -->

---

## Code: PLIV with `doubleml`

```python
from doubleml import DoubleMLPLIV

dml_data = DoubleMLData(df,
    y_col='price_spread',
    d_cols='shipping_volume',
    x_cols=control_columns,
    z_cols='weather_anomaly')    # Instrument!

pliv = DoubleMLPLIV(dml_data,
    ml_l=RandomForestRegressor(200),
    ml_m=RandomForestRegressor(200),
    ml_r=RandomForestRegressor(200),
    n_folds=5)
pliv.fit()
print(pliv.summary)
```

<!-- Speaker notes: The API adds z_cols for the instrument and ml_r for the first-stage relationship between instrument, controls, and treatment. The rest is identical to DoubleMLPLR. The fit method runs cross-fitting for all three nuisance functions and computes the orthogonal IV score. The standard error accounts for the IV estimation step. -->

---

## Instrument Strength Check

| Diagnostic | Formula | Threshold |
|-----------|---------|:---------:|
| First-stage partial $R^2$ | $\frac{R^2_{Z+X} - R^2_X}{1 - R^2_X}$ | > 0.05 |
| First-stage F-stat | Standard F-test on $Z$ | > 10 |
| CI width ratio | $\frac{CI_{PLIV}}{CI_{PLR}}$ | < 3 |

> Weak instruments → wide CIs, unreliable point estimates. Always check.

<!-- Speaker notes: The partial R-squared measures how much predictive power the instrument adds beyond the controls. Values above 0.05 suggest adequate strength. The classical rule of thumb is F greater than 10 for the first stage, but with ML first stages, partial R-squared is more appropriate. If the confidence interval from PLIV is much wider than from PLR, the instrument may be weak. -->

---

## OLS vs PLIV Comparison

| Method | Estimate | True = 0.50 | Issue |
|--------|:--------:|:-----------:|-------|
| OLS (no controls) | ~0.80 | Biased | Endogeneity |
| PLR (no instrument) | ~0.75 | Biased | U unobserved |
| **PLIV** | **~0.50** | **Correct** | Uses instrument |

> PLIV recovers the truth when OLS and PLR cannot.

<!-- Speaker notes: This comparison demonstrates why PLIV is necessary. OLS is biased because D is correlated with U. PLR is also biased because it cannot control for U since U is unobserved. PLIV uses the instrument Z to identify the causal effect despite the unobserved confounding. The instrument essentially provides exogenous variation in D that is uncorrelated with U, allowing identification. -->

---

## Commodity Instruments: Examples

| Treatment | Instrument | Rationale |
|-----------|-----------|-----------|
| Shipping volume | **Weather anomaly** | Affects shipping, not prices directly |
| Crop production | **Rainfall deviation** | Exogenous to demand |
| Oil supply | **Pipeline outage** | Technical, not market-driven |
| Electricity demand | **Temperature shock** | Weather is exogenous |
| Trade volume | **Exchange rate shock** | Monetary policy instrument |

> Finding valid instruments is the hardest part of IV analysis. The exclusion restriction is untestable.

<!-- Speaker notes: This table provides commodity-specific examples of instrumental variables. The key challenge is finding instruments that satisfy both relevance (affects treatment) and exclusion (no direct effect on outcome). Weather is the most common instrument in commodity research because it is plausibly exogenous to market sentiment. Pipeline outages work similarly for oil supply studies. The exclusion restriction — that the instrument only affects the outcome through the treatment — cannot be tested statistically. It must be justified by domain knowledge. -->

<div class="callout-key">
Key Point:  | Affects shipping, not prices directly |
| Crop production | 
</div>

---

## PLIV vs Classical 2SLS

<div class="columns">
<div>

### Classical 2SLS
- Linear first stage: $D = \alpha Z + X\gamma + V$
- Linear reduced form
- Works with $p \ll n$
- F-statistic for weakness

</div>
<div>

### PLIV
- **ML first stage:** $D = r(X, Z) + V$
- Nonlinear relationships captured
- Works with **$p \gg n$**
- Partial $R^2$ for weakness

</div>
</div>

> PLIV is strictly more flexible. Use it when the instrument-treatment relationship may be nonlinear or controls are high-dimensional.

<!-- Speaker notes: The comparison between 2SLS and PLIV mirrors the comparison between OLS and DML. 2SLS uses linear first stages, which miss nonlinear relationships between instruments and treatment. PLIV uses ML first stages that capture arbitrary functional forms. In commodity markets, the relationship between weather and shipping volume is often nonlinear — extreme weather has disproportionate effects. PLIV handles this naturally. The weakness diagnostic shifts from the F-statistic to partial R-squared, which extends naturally to ML first stages. -->

<div class="callout-insight">
Insight:  $D = r(X, Z) + V$
- Nonlinear relationships captured
- Works with 
</div>

---

## Weak Instrument Warning Signs

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    Check["Check instrument strength"] --> PR["Partial R² > 0.05?"]
    PR -->|"Yes"| Strong["Strong instrument<br/>Trust PLIV results"]
    PR -->|"No"| Weak["Weak instrument"]
    Weak --> Wide["Wide CI<br/>(ratio > 3x PLR)"]
    Weak --> Unstable["Point estimate unstable<br/>across specifications"]
    Wide --> Action["Consider:<br/>1. Stronger instrument<br/>2. Bounds analysis<br/>3. Report PLR + PLIV"]
    Unstable --> Action
```

<!-- Speaker notes: Weak instruments are the main failure mode of IV estimation. This flowchart provides a diagnostic workflow. Start by computing the partial R-squared. If it exceeds 0.05, the instrument is likely strong enough. If not, check whether the confidence interval is much wider than PLR's (a ratio above 3 is concerning) and whether the point estimate is stable across nuisance model specifications. If the instrument is weak, consider finding a stronger one, using partial identification bounds, or reporting both PLR and PLIV results side by side. -->

---

## Connections

<div class="columns">
<div>

### Builds On
- Module 05: PLR with `doubleml`
- Classical 2SLS
- IV identification

</div>
<div>

### Leads To
- Module 09: Production pipeline
- Weak instrument diagnostics
- LATE interpretation

</div>
</div>

<!-- Speaker notes: PLIV extends the DML framework to handle endogeneity. It combines the flexibility of ML first stages with the identification power of instrumental variables. The next modules cover heterogeneous treatment effects and production pipelines, both of which can incorporate IV estimation. -->

---

## Visual Summary

```mermaid
%%{init: {"theme": "base", "themeVariables": {"primaryColor": "#e8f5e9", "primaryBorderColor": "#4caf50", "primaryTextColor": "#212121", "secondaryColor": "#e3f2fd", "tertiaryColor": "#fff8e1", "lineColor": "#757575", "fontFamily": "Inter, sans-serif", "fontSize": "14px"}}}%%
flowchart TD
    Endo["Endogenous D<br/>(unobserved confounding)"] -->|"PLR fails"| Bias["Biased θ̂"]
    Endo -->|"Add instrument Z"| PLIV["PLIV"]
    PLIV --> ML["ML first stages<br/>(nonlinear Z→D)"]
    ML --> CF["Cross-fitting"]
    CF --> Valid["Valid θ̂"]
```

<!-- Speaker notes: When treatment is endogenous, PLR is biased regardless of how good the ML models are — you cannot control for what you cannot observe. PLIV solves this by adding an instrument that provides exogenous variation. The ML first stages handle nonlinear relationships between the instrument and treatment, and cross-fitting prevents overfitting. -->

<div class="callout-warning">
Warning: DML-IV requires the same identification assumptions as standard IV (relevance, exclusion, independence) PLUS the DML regularity conditions. It does not weaken IV assumptions.
</div>

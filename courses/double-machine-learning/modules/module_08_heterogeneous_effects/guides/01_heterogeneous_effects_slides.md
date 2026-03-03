---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Heterogeneous Treatment Effects

## Module 8: CATE with `econml`
### Double/Debiased Machine Learning

<!-- Speaker notes: This deck moves beyond average treatment effects to conditional average treatment effects — how treatment effects vary across individuals. We use Microsoft's econml library with the DML framework. The commodity example is heterogeneous effects of inventory surprises across commodity sectors. -->

---

## In Brief

ATE tells you the treatment **works on average**. CATE tells you **for whom** it works best.

> In commodity markets: which sectors, regions, or time periods are most affected?

`econml.dml.CausalForestDML` estimates $\tau(X) = E[Y(1) - Y(0) | X]$.

<!-- Speaker notes: The shift from ATE to CATE is one of the most valuable extensions in causal inference. An inventory surprise may have a large effect on energy futures but almost no effect on precious metals. Knowing this heterogeneity is essential for portfolio allocation and risk management. The econml library provides several CATE estimators built on the DML framework. -->

---

## From ATE to CATE

<div class="columns">
<div>

### ATE (Module 05)
- One number: $\theta = 0.5$
- Same for everyone
- Average across all units
- `doubleml.DoubleMLPLR`

</div>
<div>

### CATE (This module)
- Function: $\tau(X)$
- Varies by unit characteristics
- Energy: 1.2, Metals: 0.3
- `econml.CausalForestDML`

</div>
</div>

<!-- Speaker notes: The ATE from Modules 05-06 tells you the average effect. But if energy futures respond 4x more than metals to inventory surprises, knowing only the average is insufficient for trading decisions. CATE estimation recovers the full function tau(X), telling you how the effect varies with observable characteristics. This enables targeted strategies: increase energy exposure during surprise events, reduce metals exposure. -->

---

## CATE Estimation Pipeline

```mermaid
flowchart LR
    subgraph Stage1["Stage 1: DML Residualisation"]
        Y_r["Ỹ = Y - ĝ(X)"]
        D_r["D̃ = D - m̂(X)"]
    end
    subgraph Stage2["Stage 2: CATE Modelling"]
        CF["Causal Forest<br/>τ̂(X) = f(Ỹ, D̃, X)"]
    end
    Stage1 --> Stage2
    Stage2 --> CATE["τ̂(xᵢ) for each i"]
    style CATE fill:#6f6,color:#fff
```

<!-- Speaker notes: The pipeline has two stages. Stage 1 is standard DML residualisation — remove the confounding using ML. Stage 2 models the residual-on-residual relationship as a function of X using a causal forest or other flexible model. The result is a CATE estimate for each observation, which you can then analyse by subgroup, plot, or use for targeting. -->

---

## Code: CausalForestDML

```python
from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor

cf_dml = CausalForestDML(
    model_y=GradientBoostingRegressor(200, max_depth=5),
    model_t=GradientBoostingRegressor(200, max_depth=5),
    n_estimators=200, random_state=42)

cf_dml.fit(Y, D, X=X, W=W)

# Individual-level CATE
cate_hat = cf_dml.effect(X)

# Confidence intervals
cate_low, cate_high = cf_dml.effect_interval(X, alpha=0.05)
```

<!-- Speaker notes: The CausalForestDML API takes model_y and model_t for the first-stage nuisance models, plus n_estimators for the causal forest. The fit method takes Y, D, X (effect modifiers), and optionally W (additional controls). The effect method returns point estimates, and effect_interval returns confidence intervals. X contains the variables that moderate the treatment effect, while W contains additional controls. -->

---

## Commodity Example: Inventory Surprises by Sector

| Sector | True CATE | Estimated CATE |
|--------|:---------:|:--------------:|
| Energy | 1.20 | ~1.18 |
| Metals | 0.30 | ~0.31 |
| Agriculture | -0.10 | ~-0.08 |

> Energy futures are **4x more sensitive** to inventory surprises than metals.

This insight drives sector allocation during inventory release events.

<!-- Speaker notes: The table shows that the causal forest correctly identifies the heterogeneity across sectors. Energy is highly sensitive to inventory surprises, metals moderately so, and agriculture barely responds. For a commodity trader, this means increasing energy exposure and reducing agriculture exposure when inventory surprises are expected. The magnitude difference (4x) is large enough to be economically significant. -->

---

## BLP and GATES Analysis

**BLP (Best Linear Projection):** Tests which covariates predict heterogeneity.

**GATES (Group Average Treatment Effects):** Sorts by estimated CATE, tests for monotonicity.

```mermaid
flowchart LR
    CATE["τ̂(Xᵢ)"] --> Sort["Sort by τ̂"]
    Sort --> G1["Q1: lowest 20%"]
    Sort --> G2["Q2"]
    Sort --> G3["Q3"]
    Sort --> G4["Q4"]
    Sort --> G5["Q5: highest 20%"]
    G1 --> Test["Test: G5 > G1?"]
    style Test fill:#4a9eff,color:#fff
```

<!-- Speaker notes: BLP projects the CATE onto observable covariates and tests which ones significantly predict heterogeneity. GATES is a simpler diagnostic: sort observations by their estimated CATE, divide into quintiles, and compute the average effect in each group. If the groups show a clear monotone pattern (low to high), the CATE model is capturing real heterogeneity. If the groups are flat, there may be no meaningful heterogeneity. -->

---

## Connections

<div class="columns">
<div>

### Builds On
- Modules 05-06: ATE estimation
- Causal forests (Athey & Wager)
- `econml` library

</div>
<div>

### Leads To
- Module 09: Production pipeline
- Targeted trading strategies
- Policy evaluation by subgroup

</div>
</div>

<!-- Speaker notes: CATE estimation is the most actionable output of the DML framework for commodity traders. Knowing that energy is 4x more sensitive than metals to inventory surprises directly informs trading strategy. Module 09 wraps CATE estimation into a production pipeline with automated BLP/GATES reporting and visualisation. -->

---

## Visual Summary

```mermaid
flowchart TD
    ATE["ATE: θ = 0.5<br/>(one number)"] -->|"Not enough for targeting"| CATE["CATE: τ(X)<br/>(varies by unit)"]
    CATE --> BLP["BLP: which X matters?"]
    CATE --> GATES["GATES: sorted groups"]
    CATE --> Target["Targeted strategies<br/>by sector/region/time"]
    style Target fill:#6f6,color:#fff
```

<!-- Speaker notes: The visual summary shows the progression from average effects to heterogeneous effects to targeted strategies. ATE is a single number. CATE is a function. BLP and GATES are diagnostics. The practical payoff is targeted trading strategies that exploit the heterogeneity. -->

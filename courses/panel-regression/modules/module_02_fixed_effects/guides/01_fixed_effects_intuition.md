# Fixed Effects Intuition: Why and When to Use FE

> **Reading time:** ~20 min | **Module:** 02 — Fixed Effects | **Prerequisites:** Module 1


## In Brief

<div class="flow">
<div class="flow-step mint">1. Compute Entity Means</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Demean Variables</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Run OLS on Demeaned</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Adjust Standard Errors</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Fixed effects regression controls for all time-invariant differences between entities—whether observed or not. It answers: "Within an entity, how does Y change when X changes?" This eliminates omitted

</div>

Fixed effects regression controls for all time-invariant differences between entities—whether observed or not. It answers: "Within an entity, how does Y change when X changes?" This eliminates omitted variable bias from time-invariant confounders.

> 💡 **Key Insight:** **Fixed effects uses each entity as its own control.** By comparing an entity to itself over time, we eliminate all stable differences between entities—observable and unobservable alike.

---

## The Omitted Variable Problem

### Example: Returns to Education

Suppose we want to estimate the effect of education on wages:

$$wage_i = \alpha + \beta \cdot education_i + \epsilon_i$$

Problem: Ability is correlated with both education and wages but unobserved:

$$wage_i = \alpha + \beta \cdot education_i + \gamma \cdot ability_i + \epsilon_i$$

<div class="callout-insight">

**Insight:** Fixed effects are not a method -- they are a way of thinking about unobserved heterogeneity. The within-transformation eliminates time-invariant confounders, which is the single most important advantage of panel data.

</div>


If $Cov(education, ability) > 0$, OLS overestimates $\beta$.

### The Panel Data Solution

With panel data (same workers over time):

$$wage_{it} = \alpha_i + \beta \cdot education_{it} + \epsilon_{it}$$

The fixed effect $\alpha_i$ captures all stable characteristics of worker $i$—including ability. We estimate $\beta$ from changes within workers.

---

## The Within Transformation

<div class="callout-warning">

**Warning:** Fixed effects estimates identify only from within-entity variation. If your variable of interest has little within-entity variation (e.g., industry sector), fixed effects will produce large standard errors or fail entirely.

</div>


### Mathematical Intuition

Take the entity average:

$$\bar{y}_i = \alpha_i + \bar{X}_i\beta + \bar{\epsilon}_i$$

Subtract from the original equation:

$$(y_{it} - \bar{y}_i) = (X_{it} - \bar{X}_i)\beta + (\epsilon_{it} - \bar{\epsilon}_i)$$

The $\alpha_i$ terms cancel! We're left with:

$$\tilde{y}_{it} = \tilde{X}_{it}\beta + \tilde{\epsilon}_{it}$$

### What This Means

- We estimate $\beta$ using only **within-entity variation**
- Variation **between entities** is absorbed by fixed effects
- Time-invariant variables (like gender) cannot be estimated

---

## Visual Intuition

### Without Fixed Effects (Pooled OLS)

```
Y
|     ○ Entity A
|    ○  ○ Entity A
|   ○
|  ○ ○ ○ Entity B
| ○
|○_______________ X

Slope is estimated using ALL variation
Confounded by entity differences
```

### With Fixed Effects

```
Y
|     ○ Entity A ----
|    ○  ○ Entity A   \  Within-A slope
|   ○                 \
|  ○ ○ ○ Entity B -----\
| ○                     \ Within-B slope
|○_______________ X

Slope is estimated within each entity
Then averaged across entities
```

---

## When to Use Fixed Effects

### FE is Appropriate When:

1. **Entity-specific confounders exist**
   - Unobserved characteristics correlate with X
   - Example: Firm culture affects both investment and growth

2. **You have sufficient within-entity variation**
   - X must change over time within entities
   - More time periods = more within variation

3. **Interest is in within-entity effects**
   - Question: "What happens when X changes?"
   - Not: "Why do entities with high X have high Y?"

### FE is NOT Appropriate When:

1. **X doesn't vary within entities**
   - Gender, race, industry fixed characteristics
   - FE cannot estimate coefficients on these

2. **Between-entity variation is the interest**
   - Why some countries grow faster
   - Cross-sectional comparisons

3. **Random effects assumptions hold**
   - Entity effects uncorrelated with X
   - RE is more efficient in this case

---

## Implementation in Python


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pandas as pd
from linearmodels.panel import PanelOLS


# Load panel data
df = pd.read_csv("panel_data.csv")

# Set multi-index (entity, time)
df = df.set_index(['entity_id', 'year'])

# Entity fixed effects
model_fe = PanelOLS.from_formula(
    'y ~ x1 + x2 + EntityEffects',
    data=df
)
results_fe = model_fe.fit(cov_type='clustered', cluster_entity=True)
print(results_fe)

# Two-way fixed effects (entity + time)
model_twfe = PanelOLS.from_formula(
    'y ~ x1 + x2 + EntityEffects + TimeEffects',
    data=df
)
results_twfe = model_twfe.fit(cov_type='clustered', cluster_entity=True)
print(results_twfe)
```


</div>
</div>

### Equivalent Using Dummies


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import statsmodels.formula.api as smf

# Create entity dummies
model_lsdv = smf.ols(
    'y ~ x1 + x2 + C(entity_id)',
    data=df.reset_index()
)
results_lsdv = model_lsdv.fit()

# Entity effects are estimated (not shown by default)
print(results_lsdv.summary())
```


</div>
</div>

---

## Implementation in R

```r
library(plm)

# Set panel structure
pdata <- pdata.frame(df, index = c("entity_id", "year"))

# Entity fixed effects
fe_model <- plm(y ~ x1 + x2,
                data = pdata,
                model = "within",
                effect = "individual")
summary(fe_model)

# Two-way fixed effects
twfe_model <- plm(y ~ x1 + x2,
                  data = pdata,
                  model = "within",
                  effect = "twoways")
summary(twfe_model)

# View fixed effects
fixef(fe_model)
```

---

## Interpretation

### Coefficient Meaning

The FE coefficient $\beta$ represents:

> "The expected change in Y when X increases by 1, **within the same entity over time**, controlling for all time-invariant entity characteristics."

### Example Interpretation

*"A one-unit increase in R&D spending (within a firm) is associated with a 0.15 unit increase in patents, controlling for all time-invariant firm characteristics."*

### What You Cannot Conclude

- You cannot compare effects across entities
- You cannot estimate effects of time-invariant variables
- You cannot make between-entity predictions

---

## Common Mistakes

<div class="callout-danger">

**Danger:** Never include a lagged dependent variable in a fixed effects model without using an appropriate estimator (e.g., Arellano-Bond GMM). The within-transformation creates mechanical correlation between the transformed lagged variable and the transformed error, biasing all coefficients.

</div>


### 1. Forgetting Clustering

Standard errors should be clustered by entity:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python

# Correct
results = model.fit(cov_type='clustered', cluster_entity=True)

# Incorrect (SEs too small)
results = model.fit()
```



### 2. Misinterpreting R²

FE R² is the "within R²"—variation explained within entities. It's typically lower than pooled OLS R² because between-entity variation is absorbed by dummies.

### 3. Including Time-Invariant Variables

```python

# This will fail or give unexpected results
model = PanelOLS.from_formula(
    'y ~ x1 + gender + EntityEffects',  # gender is time-invariant
    data=df
)
```

---

## Testing the Fixed Effects Assumption

### F-test for Entity Effects

$$H_0: \alpha_1 = \alpha_2 = ... = \alpha_n$$

Reject → Entity effects are significant → FE is warranted

```python

# Compare pooled OLS to FE
from scipy import stats

# F-test is built into linearmodels
print(results_fe.f_statistic_entity)
```

### Hausman Test (FE vs RE)

$$H_0: \text{Random effects is consistent}$$

Reject → Use Fixed Effects

```r

# In R
hausman_test <- phtest(fe_model, re_model)
print(hausman_test)
```

---

*Fixed effects turns confounders into allies—by demeaning, we eliminate what we cannot measure but know exists.*


---

## Conceptual Practice Questions

**Practice Question 1:** Why can fixed effects not estimate the impact of time-invariant variables like gender or geographic region?

**Practice Question 2:** When would entity fixed effects alone be insufficient, requiring two-way (entity + time) fixed effects?


---

## Cross-References

<a class="link-card" href="./02_lsdv_vs_within.md">
  <div class="link-card-title">02 Lsdv Vs Within</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_lsdv_vs_within.md">
  <div class="link-card-title">02 Lsdv Vs Within — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_two_way_fixed_effects.md">
  <div class="link-card-title">03 Two Way Fixed Effects</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_two_way_fixed_effects.md">
  <div class="link-card-title">03 Two Way Fixed Effects — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


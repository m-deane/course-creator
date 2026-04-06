# Potential Outcomes Framework (Rubin Causal Model)

> **Reading time:** ~9 min | **Module:** 0 — Foundations | **Prerequisites:** Basic statistics, regression, probability

## In Brief

The potential outcomes framework, developed by Donald Rubin building on earlier work by Jerzy Neyman, provides a formal mathematical language for causal questions. It defines causal effects in terms of counterfactual outcomes — what would have happened under a treatment condition that was not actually received.

<div class="callout-key">
<strong>Key Concept:</strong> The potential outcomes framework, developed by Donald Rubin building on earlier work by Jerzy Neyman, provides a formal mathematical language for causal questions. It defines causal effects in terms of counterfactual outcomes — what would have happened under a treatment condition that was not act...
</div>

## Key Insight

Every unit has two potential outcomes: $Y_i(1)$, the outcome if treated, and $Y_i(0)$, the outcome if untreated. The individual causal effect is $Y_i(1) - Y_i(0)$. Only one of these is ever observed. Causal inference is the science of estimating the unobserved counterfactual.

---

## The Setup

### Units, Treatments, and Outcomes

Let $i = 1, \ldots, N$ index units (people, firms, cities, time periods). Each unit has:

- A **treatment indicator** $W_i \in \{0, 1\}$ (1 = treated, 0 = control)
- A **pair of potential outcomes** $(Y_i(0), Y_i(1))$
- A **vector of pre-treatment covariates** $X_i$

The potential outcome $Y_i(w)$ is the outcome unit $i$ would experience if assigned to treatment $w$, regardless of what treatment $i$ actually received.

### The Fundamental Problem

We observe only the **realized outcome**:
$$Y_i^{obs} = Y_i(W_i) = W_i \cdot Y_i(1) + (1 - W_i) \cdot Y_i(0)$$

The counterfactual potential outcome $Y_i(1 - W_i)$ is never observed. This is the fundamental problem of causal inference — individual treatment effects are not identified from data alone.

---

## Individual vs Average Treatment Effects

### Individual Treatment Effect (ITE)

$$\tau_i = Y_i(1) - Y_i(0)$$

This is the causal effect of treatment for unit $i$. It is **never observable** — you cannot simultaneously treat and not treat the same unit.

### Average Treatment Effect (ATE)

$$\text{ATE} = E[\tau_i] = E[Y_i(1) - Y_i(0)] = E[Y_i(1)] - E[Y_i(0)]$$

The ATE is the average causal effect across all units. Under random assignment, it is identified.

### Average Treatment Effect on the Treated (ATT)

$$\text{ATT} = E[\tau_i | W_i = 1] = E[Y_i(1) - Y_i(0) | W_i = 1]$$

The ATT conditions on units that were actually treated. This is often more policy-relevant: "Among the people who received the program, how much did it help?" This is what ITS estimates — the effect on the unit(s) that actually experienced the intervention.

### Conditional Average Treatment Effect (CATE)

$$\text{CATE}(x) = E[\tau_i | X_i = x] = E[Y_i(1) - Y_i(0) | X_i = x]$$

The average treatment effect conditional on pre-treatment characteristics. Useful for characterizing treatment effect heterogeneity.

---

## Why Naive Comparisons Fail

### The Selection Bias Problem

A naive estimate of the ATE compares outcomes of treated and untreated units:

$$\hat{\tau}^{naive} = E[Y_i^{obs} | W_i = 1] - E[Y_i^{obs} | W_i = 0]$$

Expanding this:

$$\hat{\tau}^{naive} = E[Y_i(1) | W_i = 1] - E[Y_i(0) | W_i = 0]$$

The true ATE is $E[Y_i(1)] - E[Y_i(0)]$. The naive estimator differs by:

$$\hat{\tau}^{naive} - \text{ATE} = \underbrace{E[Y_i(0) | W_i = 1] - E[Y_i(0) | W_i = 0]}_{\text{selection bias}}$$

Selection bias is the difference in baseline potential outcomes between treated and untreated groups. If healthier people choose to exercise (treatment), their $Y_i(0)$ is already higher — the treated group would have had better outcomes even without treatment.

### Visualizing the Problem

```
Unit 1 (Treated):    observe Y(1)=80    missing Y(0)=?
Unit 2 (Control):    missing Y(1)=?     observe Y(0)=55
Unit 3 (Treated):    observe Y(1)=90    missing Y(0)=?
Unit 4 (Control):    missing Y(1)=?     observe Y(0)=65

Naive estimate:      E[Y(1)|treated] - E[Y(0)|control] = 85 - 60 = 25
True ATE:            E[Y(1)] - E[Y(0)] = 85 - 72 = 13  (if untreated treated would score 72)
Selection bias:      25 - 13 = 12  (treated group had higher baseline)
```

---

## Key Assumptions for Identification

### Assumption 1: SUTVA (Stable Unit Treatment Value Assumption)

SUTVA has two components:

**No interference:** The potential outcomes for unit $i$ are unaffected by the treatment assignments of other units.

$$Y_i(w_1, \ldots, w_N) = Y_i(w_i)$$

**No hidden versions:** There is only one version of each treatment level (not multiple variants of "treated").

**When SUTVA is violated:** Vaccine studies where herd immunity means the untreated benefit from others being treated; market experiments where advertising to one customer affects competitors' customers; network experiments in social media.

### Assumption 2: Unconfoundedness (Ignorability)

$$Y_i(0), Y_i(1) \perp\!\!\!\perp W_i \mid X_i$$

Given pre-treatment covariates $X_i$, treatment assignment is independent of potential outcomes. This is equivalent to saying "there are no unmeasured confounders" conditional on $X_i$.

This assumption is **untestable from data**. It must be justified by institutional knowledge or study design.

### Assumption 3: Overlap (Positivity)

$$0 < P(W_i = 1 | X_i) < 1 \quad \text{for all } X_i$$

Every unit with covariate profile $X_i = x$ has a positive probability of receiving either treatment. Without overlap, there are covariate regions with no counterfactual comparison.

Together, unconfoundedness + overlap are called **strong ignorability**. Under strong ignorability, the ATE is nonparametrically identified.

---

## Identification Under Randomization

In a randomized experiment, treatment assignment is independent of potential outcomes by design:

$$Y_i(0), Y_i(1) \perp\!\!\!\perp W_i$$

This gives us:

$$E[Y_i(0) | W_i = 0] = E[Y_i(0)]$$
$$E[Y_i(1) | W_i = 1] = E[Y_i(1)]$$

Therefore the naive comparison is unbiased:

$$E[Y_i^{obs} | W_i = 1] - E[Y_i^{obs} | W_i = 0] = E[Y_i(1)] - E[Y_i(0)] = \text{ATE}$$

This is why randomization is the gold standard. It eliminates selection bias by construction.

---

## The Potential Outcomes Perspective on ITS

In the Interrupted Time Series context, we have a single unit (or a few units) observed across many time periods. The "treatment" is a policy intervention that occurs at time $t^*$.

For each time period $t$, we can define:

- $Y_t(0)$: the outcome at time $t$ had the intervention never occurred
- $Y_t(1)$: the outcome at time $t$ given the intervention occurred

We observe $Y_t(0)$ for $t < t^*$ (the pre-intervention period) and $Y_t(1)$ for $t \geq t^*$ (the post-intervention period).

The ITS estimand is the ATT at each post-intervention period:

$$\tau_t = Y_t(1) - Y_t(0) \quad \text{for } t \geq t^*$$

But $Y_t(0)$ is never observed for $t \geq t^*$ — it is the counterfactual trajectory. The core assumption of ITS is that the **pre-intervention trend would have continued** absent the intervention. This is our counterfactual model.

```
Time:           Pre-intervention                Post-intervention
                t=1  t=2  t=3  t=4  t=5  |  t=6  t=7  t=8  t=9  t=10
                                           |
Observed Y(1):  100  103  106  109  112   |  118  120  121  122  123
Counterfactual Y(0) [unobserved]:          |  115  118  121  124  127
Causal effect τ_t:                         |   +3   +2   0   -2   -4
```

In this illustration, the intervention initially boosted outcomes but the effect faded. Without the counterfactual, we cannot see this — the observed series is still rising.

---

## Formal ITS Estimator

Under the assumption that the pre-intervention trend provides a valid counterfactual, we model:

$$E[Y_t(0)] = \alpha + \beta \cdot t$$

After observing the pre-intervention data, we extrapolate to get $\hat{Y}_t(0)$ for $t \geq t^*$.

The estimated treatment effect at each post-intervention period is:

$$\hat{\tau}_t = Y_t^{obs} - \hat{Y}_t(0)$$

The validity of this estimate depends entirely on whether the linear (or chosen functional form) pre-trend would have continued. This is the **parallel trends assumption** in the time series version — the counterfactual trend is identifiable from the pre-period.

---

## Uncertainty in Potential Outcomes

One of the key advantages of the Bayesian approach (used in CausalPy via PyMC) is that we get a **full distribution** over the counterfactual:

$$P(Y_t(0) | \text{pre-intervention data})$$

This is not just a point estimate but a posterior distribution. We can then compute:

$$P(\tau_t > 0 | \text{data}) = P(Y_t(1) > Y_t(0) | \text{data})$$

— the probability that the treatment had a positive effect. This is a natural causal statement that is awkward in frequentist frameworks but natural in Bayesian ones.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt

# Simulating uncertainty in potential outcomes (conceptual illustration)
np.random.seed(42)

t_post = np.arange(1, 11)
# Observed post-intervention outcomes
y_observed = 50 + 2 * t_post + np.random.normal(0, 3, 10)

# Posterior samples of counterfactual (from Bayesian model)
n_samples = 1000
counterfactual_samples = np.array([
    45 + 2.5 * t_post + np.random.normal(0, 4, 10)
    for _ in range(n_samples)
])

# Posterior distribution of causal effect
causal_effects = y_observed - counterfactual_samples

# Probability of positive effect
prob_positive = (causal_effects > 0).mean(axis=0)
print("Probability of positive effect by period:")
for t, prob in zip(t_post, prob_positive):
    print(f"  t={t}: {prob:.2%}")

# Plot posterior of cumulative effect
cumulative_effect = causal_effects.sum(axis=1)
plt.hist(cumulative_effect, bins=50, edgecolor='black')
plt.axvline(0, color='red', linestyle='--', label='Zero effect')
plt.xlabel('Cumulative causal effect')
plt.ylabel('Posterior density')
plt.title('Posterior Distribution of Cumulative Treatment Effect')
plt.legend()
plt.show()
```

</div>

---

## Extensions: Heterogeneous Treatment Effects

The potential outcomes framework generalizes naturally to heterogeneous effects. If we believe the treatment effect varies with a covariate $X_i$ (age, baseline severity, region), we estimate the CATE:

$$\tau(x) = E[Y_i(1) - Y_i(0) | X_i = x]$$

In ITS, this corresponds to asking whether the policy had different effects in different subgroups or regions. CausalPy supports this through model formula specification.

---

## Practical Notation Reference

| Symbol | Meaning |
|--------|---------|
| $Y_i(0)$ | Potential outcome for unit $i$ without treatment |
| $Y_i(1)$ | Potential outcome for unit $i$ with treatment |
| $Y_i^{obs}$ | Actually observed outcome for unit $i$ |
| $W_i$ | Treatment indicator (0 or 1) |
| $\tau_i$ | Individual treatment effect $= Y_i(1) - Y_i(0)$ |
| ATE | Average Treatment Effect $= E[\tau_i]$ |
| ATT | Average Treatment Effect on the Treated |
| CATE | Conditional ATE given covariates |
| $t^*$ | Intervention date (for ITS) |

---

## Common Misconceptions

### Misconception 1: "The ATE is the average of individual effects I compute from the data"

The individual effects $\tau_i$ are all unobserved. The ATE is identified through group comparisons, not by averaging unobservable quantities. When people compute "person-level effects" from observational data, they are computing predicted values from a model, not the true $\tau_i$.

### Misconception 2: "If my regression controls for enough variables, I have the causal effect"

Regression with controls identifies the causal effect only under the unconfoundedness assumption. This requires that all relevant confounders are observed and correctly included. Omitting one confounder biases all estimates.

### Misconception 3: "The ATT is less interesting than the ATE"

For policy evaluation, the ATT is often more relevant. If we implemented a job training program for unemployed workers, we want to know the effect on those workers (ATT), not the effect if we hypothetically assigned all workers in the economy to the program (ATE — a much harder question).

---

## Connections

<div class="callout-info">
<strong>How this connects to the rest of the course:</strong>
</div>

- **Builds on:** Causal vs predictive thinking (Guide 1)
- **Leads to:** DAGs for reasoning about confounding (Guide 3), ITS specification (Module 01)
- **Related to:** Randomized controlled trials, propensity scores, matching


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Potential Outcomes Framework (Rubin Causal Model) and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Rubin, D.B. (1974). "Estimating Causal Effects of Treatments in Randomized and Nonrandomized Studies." *Journal of Educational Psychology*
- Holland, P.W. (1986). "Statistics and Causal Inference." *Journal of the American Statistical Association*
- Imbens, G. & Rubin, D. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences* — comprehensive treatment
- Morgan, S. & Winship, C. (2015). *Counterfactuals and Causal Inference* — accessible introduction


## Resources

<a class="link-card" href="../notebooks/01_environment_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

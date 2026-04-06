# Causal vs Predictive Thinking

> **Reading time:** ~8 min | **Module:** 0 — Foundations | **Prerequisites:** Basic statistics, regression, probability

## In Brief

Causal inference asks "what would happen if we intervened?" while predictive modeling asks "given what we observe, what will happen next?" Both use data, but they answer fundamentally different questions and require different methodologies.

<div class="callout-key">

<strong>Key Concept:</strong> Causal inference asks "what would happen if we intervened?" while predictive modeling asks "given what we observe, what will happen next?" Both use data, but they answer fundamentally different questions and require different methodologies.

</div>

## Key Insight

A predictive model optimized for accuracy can be completely useless for decision-making. Knowing that umbrella sales predict rain does not mean banning umbrella sales will stop rain. Causal thinking forces you to reason about interventions, not just patterns.

---

## The Core Distinction

### Prediction: Association

A predictive model learns $P(Y | X)$ — the conditional distribution of an outcome $Y$ given observed features $X$. The goal is to minimize forecast error on held-out data.

**Prediction examples:**
- "Given today's sales data, predict tomorrow's revenue."
- "Given a patient's lab values, predict readmission risk."
- "Given ad impressions, predict click-through rate."

These models can be extraordinarily accurate while being completely useless for intervention decisions.

### Causation: Intervention

A causal model answers the question: "If we change $X$, what happens to $Y$?" This requires reasoning about counterfactuals — what would have happened under a different policy.

Formally, we want $P(Y | do(X = x))$ — the outcome distribution when we actively set $X$ to value $x$, not just observe $X = x$ in the data.

**Causal examples:**
- "If we run this marketing campaign, what revenue lift will we see?"
- "If we implement this safety regulation, how many accidents will we prevent?"
- "If we increase the minimum wage, what happens to employment?"

---

## Why Correlation Fails for Decisions

### Example 1: Firefighters and Fires

Data shows that fires with more firefighters present cause more property damage. A predictive model using "number of firefighters" as a feature would correctly learn this association.

Should we reduce firefighter deployment to reduce damage? Obviously not.

The causal direction is reversed: large fires cause both more firefighters to be dispatched AND more damage. Firefighters are a consequence of fire severity, not a cause of damage.

```

Fire Severity → More Firefighters (observed)
Fire Severity → More Damage (observed)
```

A predictive model captures both arrows simultaneously and cannot distinguish them.

### Example 2: Ice Cream and Drownings

Monthly ice cream sales correlate strongly with drowning deaths. A model trained on this data would correctly predict that drowning deaths spike when ice cream sales rise.

The causal structure involves a confounder:

```

Temperature → Ice Cream Sales
Temperature → Swimming Activity → Drownings
```

Banning ice cream would not prevent drownings. Any model that uses ice cream sales for prediction is learning from a spurious correlation driven by a shared cause (temperature).

### Example 3: The Simpson's Paradox Trap

A hospital compares treatment outcomes:

| Group | Treatment A Recovery | Treatment B Recovery |
|-------|---------------------|---------------------|
| Mild cases | 80% | 70% |
| Severe cases | 40% | 30% |
| Overall | 50% | 67% |

Treatment B looks better overall, but treatment A is better for every patient subgroup. This paradox arises because severe patients disproportionately receive treatment B (selection bias). A predictive model trained on this data would recommend treatment B — the wrong causal answer.

---

## Prediction vs. Causation: Decision Framework

<div class="compare">
<div class="compare-card">
<div class="header before">Prediction (Association)</div>
<div class="body">

- Learns $P(Y | X)$
- Optimizes forecast accuracy
- Ignores mechanism/intervention
- Fails under distribution shift
- **Use when:** observing, not acting

</div>

</div>
<div class="compare-card">
<div class="header after">Causal Inference (Intervention)</div>
<div class="body">

- Estimates $P(Y | do(X))$
- Identifies treatment effects
- Requires explicit assumptions
- Robust to policy changes
- **Use when:** deciding what to do

</div>

</div>

</div>

<div class="callout-warning">

<strong>Warning:</strong> Using a predictive model to select actions is the most common and most costly mistake in applied data science. If the word "if we do X" appears in the question, you need causal methods.

</div>

---

## When Prediction Is Enough

Prediction suffices when:
1. **No intervention occurs** — you are forecasting a naturally unfolding process
2. **The environment is stable** — the data-generating process does not change
3. **You cannot act on the prediction** — the forecast is used for observation, not decision

Weather forecasting is genuinely predictive. The meteorologist is not going to intervene in the atmosphere; they want to know what will happen given current conditions.

Credit scoring is also primarily predictive — the bank wants to know who will default under normal conditions. But if the bank changes its lending policy based on the score, the scores immediately become stale (the distribution of borrowers shifts).

---

## When Causal Inference Is Required

You need causal methods when:
1. **You are evaluating a policy or intervention** — did this program work?
2. **You want to estimate treatment effects** — what is the effect of this drug?
3. **You need to decide what to do** — which action maximizes expected benefit?
4. **Distribution shift is expected** — deploying a model changes the environment

The classic failure mode is using a predictive model to select actions. If a bank uses a credit score to reject applicants, the rejected applicants never appear in future training data — this is a selection bias that predictive models cannot correct but causal methods handle explicitly.

---

## The Ladder of Causation (Pearl)

<div class="flow">
<div class="flow-step blue">1. Association (Seeing)</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Intervention (Doing)</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">3. Counterfactual (Imagining)</div>

</div>

Judea Pearl's hierarchy of causal reasoning has three rungs:

### Rung 1: Association (Seeing)
$P(Y | X = x)$ — What does observing $X = x$ tell us about $Y$?

This is all that machine learning and statistics can address from observational data alone. Predictive models live here.

### Rung 2: Intervention (Doing)
$P(Y | do(X = x))$ — What happens to $Y$ if we set $X$ to $x$?

This requires a causal model. Randomized experiments directly answer Rung 2 questions. Causal inference methods (ITS, DiD, synthetic control, IV) attempt to answer Rung 2 questions from observational data under stated assumptions.

### Rung 3: Counterfactuals (Imagining)
$P(Y_x | X = x', Y = y')$ — What would $Y$ have been if $X$ had been $x$, given that we observed $X = x'$ and $Y = y'$?

This is the domain of individual-level causal reasoning — it underlies concepts like regret, attribution, and responsibility. The potential outcomes framework (Rubin model) operates at this level.

---

## Practical Implications for This Course

In this course, we focus on Rung 2 and Rung 3 questions:

- **Interrupted Time Series (ITS):** Did a policy change the trajectory of an outcome?
- **Synthetic Control:** What would have happened to the treated unit if there had been no treatment?
- **Difference-in-Differences (DiD):** Did treated and control units diverge after treatment?

Each method makes explicit assumptions about the data-generating process. These assumptions cannot be proven from data alone — they require domain knowledge and theoretical justification. This is the core difference from predictive modeling: causal inference is transparent about its assumptions, while predictive models hide assumptions in their architecture.

---

## A Taxonomy of Causal Questions

| Question Type | Example | Method |
|--------------|---------|--------|
| Effect of treatment on average | Did the smoking ban reduce hospitalizations? | ITS, DiD |
| Effect of treatment on untreated | Would banning smoking in untreated cities have helped? | Synthetic control |
| Heterogeneous treatment effects | Does the effect differ by age or gender? | Subgroup ITS |
| Mechanism / mediation | Does the effect work through reduced secondhand smoke? | Mediation analysis |
| Dose-response | Does more aggressive treatment produce better outcomes? | IV, RD |

---

## Common Pitfalls in Causal Thinking

<div class="callout-danger">

<strong>Danger:</strong> Each of these pitfalls can lead to completely wrong policy recommendations. Real-world consequences include wasted budgets, harmful interventions, and flawed regulatory decisions.

</div>

### Pitfall 1: Confusing statistical significance with causation
A regression coefficient can be statistically significant and causally meaningless. Statistical significance measures only how precisely an association is estimated, not whether that association is causal.

### Pitfall 2: Thinking "controlling for" eliminates confounding
Adding covariates to a regression does not automatically remove confounding. Controlling for a collider (a variable caused by both treatment and outcome) can introduce bias where none existed. DAGs (covered in the next guide) make these structures explicit.

### Pitfall 3: Assuming temporal order implies causation
"$X$ happened before $Y$" does not mean "$X$ caused $Y$." Both may be driven by a common cause with different lags. ITS and related methods must explicitly rule out alternative explanations for observed patterns.

### Pitfall 4: Ignoring the fundamental problem of causal inference
We can never simultaneously observe both potential outcomes for the same unit. The treated outcome and the counterfactual outcome are mutually exclusive. All causal methods are ultimately strategies for constructing credible counterfactuals.

---

## Code Example: Prediction vs. Causal Inference


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Simulate a confounder scenario
np.random.seed(42)
n = 1000

# Confounder: baseline health
baseline_health = np.random.normal(0, 1, n)

# Treatment assignment: healthier people more likely to exercise
treatment_prob = 1 / (1 + np.exp(-2 * baseline_health))
exercise = np.random.binomial(1, treatment_prob, n)

# Outcome: health improves with both baseline health AND exercise
# True causal effect of exercise is 2 units
health_outcome = 3 * baseline_health + 2 * exercise + np.random.normal(0, 1, n)

df = pd.DataFrame({
    "exercise": exercise,
    "baseline_health": baseline_health,
    "health_outcome": health_outcome,
})

# Naive predictive model (ignores confounder)
model_naive = LinearRegression()
model_naive.fit(df[["exercise"]], df["health_outcome"])
print(f"Naive estimate of exercise effect: {model_naive.coef_[0]:.2f}")
# Will be biased upward because exercise correlates with baseline health

# Adjusted model (controls for confounder)
model_adjusted = LinearRegression()
model_adjusted.fit(df[["exercise", "baseline_health"]], df["health_outcome"])
print(f"Adjusted estimate of exercise effect: {model_adjusted.coef_[0]:.2f}")
# Closer to the true causal effect of 2.0
print(f"True causal effect: 2.0")
```

</div>
</div>

The naive model overstates the benefit of exercise because healthier people exercise more. The adjusted model recovers the true effect by controlling for baseline health — but only because we know from domain knowledge that baseline health is a confounder, not a mediator or collider.

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>

</div>

- **Builds on:** Basic statistics, regression, probability
- **Leads to:** Potential outcomes framework (next guide), DAGs (third guide), all ITS methods
- **Related to:** Randomized controlled trials, econometrics, epidemiology


## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Causal vs Predictive Thinking and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.) — the definitive treatment
- Hernán, M. & Robins, J. (2020). *Causal Inference: What If* — free online, excellent epidemiology focus
- Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference* — mathematical foundations
- Cunningham, S. (2021). *Causal Inference: The Mixtape* — free online, econometrics perspective


## Resources

<a class="link-card" href="../notebooks/01_environment_setup.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

# Interrupted Time Series: Introduction

> **Reading time:** ~9 min | **Module:** 1 — Its Fundamentals | **Prerequisites:** Module 0 — Causal Foundations

## In Brief

Interrupted Time Series (ITS) is a quasi-experimental design that estimates the causal effect of an intervention using longitudinal data from a single unit (or a few units). It uses the pre-intervention trend as the counterfactual for what would have happened absent the intervention.

<div class="callout-key">

<strong>Key Concept:</strong> Interrupted Time Series (ITS) is a quasi-experimental design that estimates the causal effect of an intervention using longitudinal data from a single unit (or a few units). It uses the pre-intervention trend as the counterfactual for what would have happened absent the intervention.

</div>

## Key Insight

ITS exploits the temporal structure of the data: the same unit is observed before and after an intervention whose timing is determined by external factors (a policy change, a law, a natural event), not by the outcome level. The pre-period trend is the counterfactual trajectory.

---

## When to Use ITS

ITS is appropriate when:

1. **A clearly defined intervention** with a known start date exists
2. **Sufficient pre-intervention observations** are available (minimum ~12; ideally 30+)
3. **The intervention timing is exogenous** — not triggered by the outcome itself
4. **No major concurrent events** occurred at the same time as the intervention
5. **SUTVA holds** — the treated unit's trajectory is not affected by what happens to other units
6. **Outcome data is regularly spaced** (weekly, monthly, quarterly)

### Real-World ITS Applications

| Domain | Intervention | Outcome |
|--------|-------------|---------|
| Public health | Workplace smoking bans | Acute myocardial infarction rates |
| Transportation | Speed camera installation | Road accident deaths |
| Economics | Minimum wage increases | Employment and wages |
| Education | Policy mandating class size reduction | Test scores |
| Finance | Circuit breaker rules | Market volatility |
| Environmental | Emissions regulations | Air quality indices |
| Crime | Policing interventions | Crime rates by type |

<div class="callout-insight">

<strong>Insight:</strong> ITS is the "minimum viable causal design" -- it requires only a single treated unit with a long time series. When you lack a control group, ITS is often your strongest option for credible causal inference.

</div>

---

## ITS vs. Other Causal Designs

<div class="compare">
<div class="compare-card">
<div class="header before">ITS (Single Unit)</div>
<div class="body">

- Needs only one treated unit
- Requires long pre-period (12+ obs)
- Counterfactual = extrapolated trend
- Vulnerable to concurrent events
- **Best for:** policy changes with clear timing

</div>

</div>
<div class="compare-card">
<div class="header after">DiD / Synthetic Control</div>
<div class="body">

- Needs a control group
- Shorter pre-period acceptable
- Counterfactual = control group trajectory
- Controls for time-varying confounders
- **Best for:** when untreated comparisons exist

</div>

</div>

</div>

---

## When NOT to Use ITS

ITS is inappropriate or weak when:

### 1. Very Short Pre-Period
With fewer than 12 pre-intervention observations, you cannot reliably estimate the pre-trend. The counterfactual is poorly identified, and confidence intervals will be very wide. The minimum is ~12 observations; 24+ is preferred for reliable inference.

### 2. Intervention Timing Is Endogenous
If the intervention was triggered by the outcome — a policy implemented BECAUSE crime was rising, a treatment started BECAUSE the patient was getting worse — then the pre-trend is not a valid counterfactual. This is the regression to the mean problem. The trend would have reverted to the mean regardless of the intervention.

**Detection:** Did decision-makers look at outcome levels when choosing when to intervene?

### 3. Major Concurrent Events
If another significant change occurred at approximately the same time as the intervention (within the same or adjacent periods), it is impossible to attribute the observed change to the intervention of interest. The estimate is confounded.

**Detection:** Check news, policy databases, and industry records for events near $t^*$.

### 4. Short Post-Period
With very few post-intervention observations, the causal estimate will be highly uncertain and the estimated trend change will be unreliable.

### 5. Highly Non-Stationary or Trend-Changing Pre-Period
If the pre-period trend itself is non-linear, non-stationary, or has multiple structural breaks, a simple linear trend extrapolation will be a poor counterfactual.

### 6. No Theoretical Mechanism
If there is no plausible causal pathway from the intervention to the outcome, ITS cannot provide credible causal evidence even with a perfect statistical fit.

---

## The Core Assumptions

ITS rests on four key assumptions. The first two are usually stated; the second two are often implicit.

### Assumption 1: Correct Counterfactual (Pre-Trend Validity)

The pre-intervention trend, if extrapolated forward, would have continued at the same rate absent the intervention.

**Formally:** $E[Y_t(0)] = f(t)$ for $t \geq t^*$, where $f$ is estimated from the pre-period.

**How to check:** Look for evidence that the trend was already changing before $t^*$ (pre-trend tests). Use domain knowledge to argue no structural break was occurring.

### Assumption 2: No Concurrent Confounders

No other events affecting the outcome occurred simultaneously with the intervention at $t^*$.

**How to check:** Conduct a literature review and check policy calendars. Use placebo outcomes (outcomes that the intervention should NOT affect) — if placebo outcomes also show a change at $t^*$, something else changed.

### Assumption 3: SUTVA

The outcome trajectory of the treated unit is not affected by what happens to comparison units (if any), and the treatment is well-defined (one version of the treatment).

**How to check:** Check for spillover effects in neighboring regions. Verify the intervention was implemented consistently.

### Assumption 4: No Anticipation

Units did not change behavior before the intervention in anticipation of it.

**How to check:** Look for pre-trend breaks at the announcement date. Use survey data about awareness.

---

## Model Specification

The standard ITS model is a **segmented linear regression**:

$$Y_t = \alpha + \beta_1 t + \beta_2 D_t + \beta_3 (t - t^*) D_t + \varepsilon_t$$

Where:
- $Y_t$ is the outcome at time $t$
- $\alpha$ is the baseline level at $t = 0$
- $\beta_1$ is the pre-intervention slope (time trend)
- $D_t = \mathbf{1}[t \geq t^*]$ is the post-intervention indicator
- $\beta_2$ is the **level change** at $t^*$ (immediate effect)
- $\beta_3$ is the **slope change** after $t^*$ (sustained/trajectory effect)
- $(t - t^*)D_t$ is the time since the intervention (0 in the pre-period)

### Reading the Estimates

| Parameter | Interpretation |
|-----------|---------------|
| $\beta_2 > 0$ | Immediate upward level shift at the intervention |
| $\beta_2 < 0$ | Immediate downward level shift |
| $\beta_3 > 0$ | Trend steepened after intervention (growing effect) |
| $\beta_3 < 0$ | Trend flattened or reversed after intervention |
| $\beta_2 = 0, \beta_3 \neq 0$ | Only a trend change, no immediate effect (gradual adoption) |
| $\beta_2 \neq 0, \beta_3 = 0$ | Only an immediate effect, no change in trajectory |

### The Counterfactual

The counterfactual at each post-intervention time $t$ is:

$$\hat{Y}_t(0) = \hat{\alpha} + \hat{\beta}_1 t$$

The estimated causal effect is:

$$\hat{\tau}_t = Y_t^{obs} - \hat{Y}_t(0) = \hat{\beta}_2 + \hat{\beta}_3 (t - t^*)$$

---

## Model Variants

### Level-Only Model

$$Y_t = \alpha + \beta_1 t + \beta_2 D_t + \varepsilon_t$$

Use when: You expect an immediate, sustained level change with no change in the growth rate. Example: a one-time subsidy payment.

### Slope-Only Model

$$Y_t = \alpha + \beta_1 t + \beta_3 (t - t^*) D_t + \varepsilon_t$$

Use when: You expect gradual adoption or accumulating effects with no immediate level shift. Example: a behavior change campaign that takes time to diffuse.

### Full Model (Level + Slope)

$$Y_t = \alpha + \beta_1 t + \beta_2 D_t + \beta_3 (t - t^*) D_t + \varepsilon_t$$

Use as the default. The estimates will tell you which components are active.

---

## ITS with Multiple Interventions

When multiple sequential interventions occur, the model can be extended:

$$Y_t = \alpha + \beta_1 t + \sum_{k=1}^K \left[ \beta_{2k} D_{kt} + \beta_{3k} (t - t_k^*) D_{kt} \right] + \varepsilon_t$$

Each intervention $k$ has its own level change ($\beta_{2k}$) and slope change ($\beta_{3k}$). However, interpretation requires care — the second intervention's counterfactual depends on the first intervention's effect continuing at its new slope.

---

## The Autocorrelation Problem

Standard OLS regression assumes $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0$ for $t \neq s$ — uncorrelated errors. Time series data almost always violates this. Autocorrelated errors lead to:

1. Underestimated standard errors (overconfident inference)
2. Inflated false-positive rates (spurious findings)

### Detecting Autocorrelation

The Durbin-Watson test statistic:
$$DW = \frac{\sum_{t=2}^T (\hat{\varepsilon}_t - \hat{\varepsilon}_{t-1})^2}{\sum_{t=1}^T \hat{\varepsilon}_t^2}$$

- $DW \approx 2$: no autocorrelation
- $DW < 1.5$: positive autocorrelation (very common in time series)
- $DW > 2.5$: negative autocorrelation

### Solutions

1. **Newey-West standard errors:** Heteroskedasticity and autocorrelation consistent (HAC) standard errors. Adjusts the standard errors without changing the point estimates.

2. **ARMA error structure:** Explicitly model the autocorrelation in the errors: $\varepsilon_t = \rho \varepsilon_{t-1} + u_t$.

3. **Bayesian ITS (CausalPy):** The Bayesian approach with PyMC can incorporate an AR(1) error process directly in the model specification, providing correctly calibrated posterior uncertainty.

---

## Internal Validity Threats and Mitigations

| Threat | Description | Mitigation |
|--------|-------------|------------|
| Selection bias | Intervention non-randomly assigned | ITS relies on exogenous timing — document why |
| History / concurrent events | Other events at $t^*$ | Placebo outcomes, literature review |
| Maturation | Natural trend would have changed anyway | Pre-trend test, nonlinear trend models |
| Regression to the mean | Intervention triggered by extreme value | Compare to control group in same period |
| Instrumentation | Outcome measurement changed at $t^*$ | Document data collection consistency |
| Anticipation | Pre-intervention behavior change | Check announcement dates, early trend breaks |

---

## Strengthening ITS with a Control Series

A **controlled ITS** (also called ITS with a comparison group) adds a comparison series that did not receive the intervention. This dramatically improves internal validity:

1. **Difference-in-ITS:** Compare the level/slope changes in the treated series to any changes in the comparison series at $t^*$. Changes in both series indicate a concurrent event, not a treatment effect.

2. **Design:** $Y_{treated,t} - Y_{control,t}$ removes time-varying confounders common to both series.

**Ideal comparison series characteristics:**
- Similar pre-intervention trend to the treated series
- Plausibly unaffected by the treatment
- Subject to the same concurrent shocks
- Geographically or institutionally proximate

---

## Code Example: Basic ITS Specification


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import pandas as pd
import causalpy as cp

def prepare_its_data(df: pd.DataFrame, intervention_date: str, time_col: str, outcome_col: str) -> pd.DataFrame:
    """
    Prepare a dataframe for ITS analysis by adding required indicator columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time and outcome columns
    intervention_date : str or any comparable type
        The value in time_col at which the intervention occurred
    time_col : str
        Column name for the time variable
    outcome_col : str
        Column name for the outcome variable

    Returns
    -------
    pd.DataFrame with added columns: treated, time_since_intervention
    """
    df = df.copy()

    # Convert time_col to numeric index if needed
    df["t_numeric"] = np.arange(len(df))
    intervention_idx = df.index[df[time_col] >= intervention_date][0]

    # Binary treatment indicator: 1 after intervention
    df["treated"] = (df["t_numeric"] >= intervention_idx).astype(float)

    # Time since intervention (0 in pre-period)
    df["time_since_intervention"] = np.maximum(
        df["t_numeric"] - intervention_idx, 0
    ).astype(float)

    return df


# Fit the segmented regression
def fit_its_model(df, outcome, formula=None):
    """Fit an ITS model using CausalPy."""
    if formula is None:
        formula = f"{outcome} ~ 1 + t_numeric + treated + time_since_intervention"

    model = cp.InterruptedTimeSeries(
        data=df,
        treatment_time=int(df["treated"].idxmax()),
        formula=formula,
        model=cp.pymc_models.LinearRegression(
            sample_kwargs={
                "draws": 1000,
                "tune": 1000,
                "chains": 4,
                "progressbar": False,
                "random_seed": 42,
            }
        ),
    )
    return model
```

</div>
</div>

---

## Connections

<div class="callout-info">

<strong>How this connects to the rest of the course:</strong>


- **Builds on:** Potential outcomes (Module 00), DAGs and confounding (Module 00)
- **Leads to:** Segmented regression details (Guide 2), CausalPy API (Guide 3)
- **Related to:** Difference-in-Differences (Module 04), Regression Discontinuity (Module 05)




## Practice Questions

### Question 1: Conceptual Check
**Question:** In your own words, explain the core concept of Interrupted Time Series: Introduction and why it matters for practical applications. What problem does it solve that simpler approaches cannot?

### Question 2: Application
**Question:** Describe a real-world scenario where you would apply the techniques from this guide. What assumptions would you need to verify before proceeding?

## Further Reading

- Bernal, J.L., Cummins, S., & Gasparrini, A. (2017). "Interrupted time series regression for the evaluation of public health interventions: a tutorial." *International Journal of Epidemiology*
- Shadish, W.R., Cook, T.D., & Campbell, D.T. (2002). *Experimental and Quasi-Experimental Designs for Generalized Causal Inference* — the definitive reference on quasi-experiments
- Kontopantelis, E. et al. (2015). "Regression based quasi-experimental approach when randomisation is not an option: interrupted time series analysis." *BMJ*


## Resources

<a class="link-card" href="../notebooks/01_its_smoking_ban.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">15-minute micro-notebook with guided exercises for this topic.</div>
</a>

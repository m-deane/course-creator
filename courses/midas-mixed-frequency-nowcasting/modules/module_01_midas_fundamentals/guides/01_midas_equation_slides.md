---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# The MIDAS Equation

## Mixed Data Sampling: The Core Framework

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**
Module 01 — Guide 01

<!-- Speaker notes: This is the central slide deck of the course. We introduce the MIDAS equation formally. Students should have completed Module 00 and understand the information loss problem. The MIDAS equation is the solution. By the end of this deck, students should be able to write the MIDAS model, interpret each component, and explain how it differs from simple aggregation. -->

---

## The Problem We're Solving

**Recall from Module 00:**

$$y_t^Q = \alpha + \beta \underbrace{\frac{1}{m}\sum_{j=0}^{m-1} x_{mt-j}^M}_{\text{equal-weight aggregate}} + \varepsilon_t$$

- Weights fixed at $w_j = 1/m$ before any estimation
- All timing information discarded

**MIDAS solution:**

$$y_t^Q = \alpha + \beta \underbrace{\sum_{j=0}^{K-1} w_j(\theta) \cdot x_{mt-j}^M}_{\text{estimated weight aggregate}} + \varepsilon_t$$

- Weights $w_j(\theta)$ **estimated from data**
- Timing information preserved

<!-- Speaker notes: Start by connecting to the previous module. The only difference between these two equations is whether the weights are fixed or estimated. But that difference has large consequences for forecast accuracy. The parameters theta control the shape of the weight function — we'll spend most of this guide explaining what theta means and how the shape is parameterized. -->

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $y_t$ | Low-frequency dependent variable (quarterly) |
| $x_\tau$ | High-frequency regressor (monthly) |
| $m$ | Frequency ratio ($m=3$ for monthly/quarterly) |
| $K$ | Total number of high-frequency lags |
| $w_j(\theta)$ | Weight on lag $j$ (function of parameters $\theta$) |
| $L^{1/m}$ | High-frequency lag operator |
| $\tau = mt$ | Last high-frequency period in low-frequency period $t$ |

**Constraint:** $\sum_{j=0}^{K-1} w_j(\theta) = 1$ (normalized)

<!-- Speaker notes: The lag operator notation L^{1/m} is used in time series econometrics. L^{1/m} x_tau = x_{tau-1}, so (L^{1/m})^j x_tau = x_{tau-j}. The superscript 1/m indicates that one application shifts by one high-frequency period (one month) rather than one low-frequency period (one quarter). This notation is compact but students sometimes find it confusing — the concrete example with monthly observations is clearer. -->

---

## The Full MIDAS Equation

$$\boxed{y_t = \alpha + \beta \cdot B\!\left(L^{1/m};\,\theta\right) x_{mt} + \varepsilon_t}$$

where:

$$B\!\left(L^{1/m};\,\theta\right) x_{mt} = \sum_{j=0}^{K-1} w_j(\theta) \cdot x_{mt-j}$$

**Parameters to estimate:** $\alpha,\, \beta,\, \theta$ (just 2–4 parameters total)

**Data structure:** $K$ high-frequency observations per low-frequency observation

<!-- Speaker notes: This boxed equation is the heart of the course. B is the lag polynomial — it is a function that takes the high-frequency series x and returns a weighted average of K lags. The parameters theta control the shape of the weights inside B. We don't estimate K weights directly — we estimate 2 shape parameters theta, which then determine all K weights. This is the key restriction that makes MIDAS feasible. -->

---

## Parameter Count: MIDAS vs. Unrestricted

**Scenario:** 4 quarterly lags × 3 months = $K=12$ high-frequency lags

<div class="columns">

<div>

**Unrestricted (OLS on all lags)**
- Estimate: $\alpha, \beta_0, \beta_1, \ldots, \beta_{11}$
- Parameters: **13**
- With $T=100$ quarters: only ~8 obs per parameter
- Severely over-parameterized

</div>

<div>

**MIDAS (Beta polynomial)**
- Estimate: $\alpha, \beta, \theta_1, \theta_2$
- Parameters: **4**
- With $T=100$ quarters: 25 obs per parameter
- Well-identified

</div>

</div>

> The polynomial restriction is a regularizer — it imposes smooth weight shapes from a rich family.

<!-- Speaker notes: This parameter count comparison is the practical justification for the MIDAS parameterization. With K=12 lags and T=100 quarters, unrestricted estimation would require estimating 13 parameters from effectively 100 observations with serial correlation. The degrees of freedom are far too low for reliable inference. MIDAS reduces to 4 parameters — reasonable even with 60-80 observations. The restriction to smooth weight shapes is not arbitrary — empirically, weight functions in macro applications are indeed smooth. -->

---

## The Frequency Ratio Determines the Problem Size

$$K = P \times m$$

| Application | $m$ | $P$ | $K$ | MIDAS params | Unrestr. params |
|-------------|-----|-----|-----|-------------|----------------|
| Monthly → Quarterly | 3 | 4 | 12 | 4 | 14 |
| Weekly → Monthly | 4 | 3 | 12 | 4 | 14 |
| Daily → Quarterly | 65 | 4 | 260 | 4 | 262 |
| Daily → Monthly | 22 | 3 | 66 | 4 | 68 |

For daily-to-quarterly MIDAS: **4 vs. 262 parameters.**

Unrestricted is infeasible. MIDAS is not.

<!-- Speaker notes: The daily-to-quarterly case makes the value of MIDAS most obvious. Nobody can estimate 262 regression coefficients from 100 quarterly observations. But 4 parameters is perfectly feasible. This is why MIDAS was such an important methodological advance — it opened up the use of daily financial data to predict quarterly outcomes, which was previously intractable. The daily-to-quarterly application is common in financial economics: predicting quarterly stock returns from daily realized volatility, predicting quarterly GDP from daily financial conditions. -->

---

## The Weight Normalization

$$\sum_{j=0}^{K-1} w_j(\theta) = 1 \quad \text{(required)}$$

**Why this matters:**

Without normalization:
- $\beta$ and $w_j$ cannot be separately identified
- Optimizer can trade off $\beta$ against scale of $w_j$
- Coefficient $\beta$ has no economic interpretation

With normalization:
- $\beta$ = effect of a 1-unit change in the **weighted average** of $x$
- $w_j$ = fraction of total effect attributed to lag $j$
- Directly comparable to OLS on equal-weight aggregate

<!-- Speaker notes: The identification argument is the key reason for normalization. Without constraining the weights to sum to one, the model has two free scale parameters: beta (the overall magnitude) and the sum of weights (also a scale parameter). They cannot both be identified. Fixing the sum to one resolves this by defining beta in terms of the weighted average. The economic interpretation then becomes clear: beta is the same type of coefficient as in OLS on the aggregated series, making model comparison straightforward. -->

---

## Multi-Quarter MIDAS

With $P$ quarterly lags and $m=3$ months:

$$y_t = \alpha + \beta \sum_{j=0}^{3P-1} w_j(\theta) \cdot x_{3t-j} + \varepsilon_t$$

$$= \alpha + \beta \left[ \underbrace{w_0 x_{3t} + w_1 x_{3t-1} + w_2 x_{3t-2}}_{\text{current quarter Q}_t} + \underbrace{w_3 x_{3t-3} + w_4 x_{3t-4} + w_5 x_{3t-5}}_{\text{previous quarter Q}_{t-1}} + \cdots \right] + \varepsilon_t$$

The weight function determines how much each quarter's monthly pattern contributes.

<!-- Speaker notes: Walk through this expansion carefully. The first three weights (j=0,1,2) apply to the three monthly observations within the current quarter. The next three (j=3,4,5) apply to the previous quarter. And so on for P quarters back. The Beta polynomial weight function typically assigns highest weight to the most recent months (j=0,1,2) and declining weight to older quarters. This is the natural economic interpretation: current conditions are more informative about current GDP than conditions from several quarters ago. -->

---

## MIDAS-AR: Adding Autoregressive Terms

For serially correlated $y_t$, add quarterly lags:

$$y_t = \alpha + \rho y_{t-1} + \beta \sum_{j=0}^{K-1} w_j(\theta) \cdot x_{mt-j} + \varepsilon_t$$

**Practical benefits:**
- Controls for persistence in $y_t$ (GDP growth has AR≈0.3)
- Reduces residual autocorrelation
- Improves forecast accuracy for multi-step horizons

**Identification:** $\rho$ and $\beta$ are estimated jointly with $\theta$ via NLS.

<!-- Speaker notes: The MIDAS-AR model is the workhorse in macroeconomic forecasting applications. GDP growth has modest but significant positive autocorrelation (AR coefficient around 0.2-0.4 depending on sample and specification). Including the lagged GDP ensures the MIDAS coefficient on IP captures the contemporaneous relationship, not the common persistence of both series. In Module 02, we'll test whether the AR component is statistically significant and how to choose the AR order. -->

---

## Multi-Predictor MIDAS

$$y_t = \alpha + \sum_{r=1}^{R} \beta_r \cdot B_r\!\left(L^{1/m};\,\theta_r\right) x_{r,mt} + \varepsilon_t$$

**Example: GDP nowcasting with 3 predictors**

$$\text{GDP}_t = \alpha + \beta_1 \underbrace{B_1(\cdot;\theta_1)}_{\text{IP weights}} \text{IP}_{3t} + \beta_2 \underbrace{B_2(\cdot;\theta_2)}_{\text{Payrolls weights}} \text{PAYEMS}_{3t} + \beta_3 \underbrace{B_3(\cdot;\theta_3)}_{\text{Retail weights}} \text{RSAFS}_{3t} + \varepsilon_t$$

Each predictor has its own weight function — the estimated weights may differ substantially.

<!-- Speaker notes: In a real nowcasting model, you'd typically have 5-20 monthly indicators plus some daily series. Each gets its own weight function with its own theta parameters. The joint optimization estimates all betas and all thetas simultaneously. For a model with 10 monthly predictors each using a 2-parameter Beta polynomial, total parameters = 2 + 10 + 10×2 = 32 parameters — manageable even with 80 quarterly observations. Compare to unrestricted, which would require 2 + 10×12 = 122 parameters — infeasible. -->

---

## The Probability Limit of OLS-Aggregate

If the true MIDAS model has weights $w_j^*$ but we run OLS on $\bar{x}_t = (1/m)\sum_{j=0}^{m-1} x_{mt-j}$:

$$\hat{\beta}^{\text{OLS}} \xrightarrow{p} \beta^* \cdot \frac{\sum_{j=0}^{m-1} w_j^* \cdot \text{Cov}(x_{mt-j},\, \bar{x}_t)}{\text{Var}(\bar{x}_t)}$$

**Takeaway:** OLS estimates a misspecified weighted average of true lag effects, **not** the effect of the optimal weighted average.

<!-- Speaker notes: This is the formal statement of what's wrong with pre-aggregation. The OLS estimator on the aggregated series is not inconsistent in the sense of estimating something in an unstable way — it consistently estimates a particular weighted combination of the true parameters. But that combination is not the parameter we actually want. In practice, this means the OLS-on-aggregate coefficient is interpretable only if the true weights happen to equal the equal-weight scheme, which is the null hypothesis of aggregation. Ghysels et al. provide a test for this. -->

---

## Summary: The MIDAS Model

$$y_t = \alpha + \beta \sum_{j=0}^{K-1} w_j(\theta) x_{mt-j} + \varepsilon_t$$

**Three key choices:**
1. **$K$:** Number of high-frequency lags (typically $P \times m$, $P = 4$)
2. **Weight function family:** Beta polynomial, Almon, Step (Guide 02)
3. **Estimation method:** NLS, OLS-profile (Guide, Module 02)

**Two key advantages over aggregation:**
1. Weights estimated from data — recovers timing information
2. Single-step estimation — no error compounding

<!-- Speaker notes: The summary encapsulates the design decisions in any MIDAS application. K (the lag count) is usually set by economic reasoning: how many quarters back can IP predict GDP? The weight function family is a model selection decision — Beta polynomial is the most popular because it is flexible with only 2 parameters. Estimation via NLS is the standard approach; OLS-profile is a newer computationally efficient alternative we introduce in Module 02. -->

---

## Preview: Estimated Weight Function

```
Typical estimated Beta polynomial weights (K=9, theta1=1.5, theta2=4.0):

Lag:  0     1     2     3     4     5     6     7     8
      Q_t         |     Q_{t-1}     |     Q_{t-2}
      ↑ current
Weight: 0.22  0.20  0.16  0.14  0.11  0.08  0.05  0.03  0.01
         ▓▓▓▓  ▓▓▓▓  ▓▓▓  ▓▓▓  ▓▓   ▓▓   ▓    ▓    ·
```

Recent months have more weight than older months — the model "learned" that current economic conditions are more relevant than conditions two or three quarters ago.

<!-- Speaker notes: This preview motivates the next guide on weight functions. The key visual pattern is the declining weight — highest at lag 0 (most recent month) and declining toward zero for older observations. This shape is captured by a Beta(1.5, 4.0) distribution. The specific parameter values are typical for quarterly GDP on monthly IP — students will estimate their own in the notebook and compare to this. -->

---

## Next Steps

**Guide 02:** Weight functions — Almon, Beta, exponential Almon, step functions. Formulas, visualizations, and when to use each.

**Guide 03:** U-MIDAS — the unrestricted version and when it outperforms restricted MIDAS.

**Notebook 01:** Estimate your first MIDAS model on GDP + IP data. Visualize estimated weights. Compare to OLS aggregate.

<!-- Speaker notes: Guide 02 is critical for building intuition about what different theta values produce. Students should spend time with the interactive weight function visualizations in the notebook before trying to interpret their own estimated weights. U-MIDAS (Guide 03) is important as a benchmark — in some applications it actually outperforms the polynomial parameterization. -->

---

## Key Equations to Remember

$$y_t = \alpha + \beta \underbrace{\sum_{j=0}^{K-1} w_j(\theta)}_{\text{sums to 1}} x_{mt-j} + \varepsilon_t \quad \text{(MIDAS)}$$

$$y_t = \alpha + \beta \underbrace{\frac{1}{m}\sum_{j=0}^{m-1}}_{\text{equal weights}} x_{mt-j} + \varepsilon_t \quad \text{(Aggregation = special case)}$$

**MIDAS generalizes temporal aggregation by estimating the weights.**

<!-- Speaker notes: End with this comparison. MIDAS is strictly more general than temporal aggregation. When theta is such that the Beta polynomial gives equal weights, MIDAS reduces to the aggregated OLS model. We can formally test whether the equal-weight restriction is rejected by the data — and in most macro applications, it is. This test provides empirical justification for the additional complexity of MIDAS. -->

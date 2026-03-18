---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Traditional Solutions to Mixed-Frequency Data

## Temporal Aggregation, Interpolation, and Bridge Equations

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**
Module 00 — Guide 02

<!-- Speaker notes: This guide covers the three approaches that existed before MIDAS. Understanding their failures precisely motivates the MIDAS solution. The goal isn't to dismiss these methods — bridge equations in particular remain widely used at central banks. The goal is to understand exactly what they sacrifice and when that sacrifice matters. -->

---

## Three Strategies, One Problem

```
High-frequency data: x₁ᴹ, x₂ᴹ, x₃ᴹ | x₄ᴹ, x₅ᴹ, x₆ᴹ | ...
                     ←   Quarter 1  →   ←   Quarter 2  →

Low-frequency target: y₁ᵠ                y₂ᵠ
```

**Strategy 1 — Aggregation:** Collapse $x^M$ to $\tilde{x}^Q$, run OLS.
**Strategy 2 — Interpolation:** Infer synthetic monthly $\hat{y}^M$ from quarterly $y^Q$.
**Strategy 3 — Bridge equation:** Aggregate $x^M$, forecast missing months, plug in.

**All three:** Pre-commit to an aggregation before estimation.

<!-- Speaker notes: These three strategies have distinct use cases. Aggregation is used when forecasting quarterly outcomes. Interpolation is used when you want a monthly version of a quarterly concept (like monthly GDP). Bridge equations are used for nowcasting. Critically, all three involve a data transformation step that discards information before any statistical estimation begins. -->

---

## Strategy 1: Temporal Aggregation

For a flow variable (IP, payrolls — things that accumulate):
$$\tilde{x}_t^Q = \sum_{j=1}^{m} x_{mt-m+j}^M$$

For a stock variable (price, rate — measured at a point):
$$\tilde{x}_t^Q = x_{mt}^M \quad \text{(end-of-period)} \quad \text{or} \quad \frac{1}{m}\sum_{j=1}^m x_{mt-m+j}^M \quad \text{(average)}$$

Then run: $y_t^Q = \alpha + \beta \tilde{x}_t^Q + \varepsilon_t$

**Information lost:** All within-quarter timing signal.

<!-- Speaker notes: The flow/stock distinction matters for correct aggregation. Industrial production is a flow — you sum the monthly changes. An interest rate is a stock — you take the end-of-period value or average. Getting this wrong introduces a systematic aggregation error on top of the information loss from aggregation itself. A common mistake is applying end-of-period sampling to a flow variable. -->

---

## The Information Loss Calculation

When we average $m$ monthly observations into one quarterly value, we collapse $m$ parameters to 1.

| Frequency ratio $m$ | Bits lost per observation |
|--------------------|--------------------------|
| Monthly → Quarterly ($m=3$) | $\log_2(3) \approx 1.6$ bits |
| Weekly → Monthly ($m=4$) | $\log_2(4) = 2.0$ bits |
| Daily → Quarterly ($m=65$) | $\log_2(65) \approx 6.0$ bits |

> For daily-to-quarterly MIDAS, aggregation discards 6x more information than for monthly-to-quarterly.

<!-- Speaker notes: This information-theoretic framing is approximate — the actual information loss depends on the autocorrelation structure of the series. But the intuition is right: the higher the frequency ratio, the more information aggregation discards. This is why MIDAS is especially valuable for daily-to-quarterly models, where bridge equations are practically unusable due to the large number of daily observations that must be forecast. -->

---

## Strategy 2: Chow-Lin Interpolation

Goal: Infer monthly GDP from quarterly GDP + monthly IP indicator.

$$y_\tau^M = c \cdot z_\tau^M + \varepsilon_\tau, \quad \varepsilon_\tau = \rho \varepsilon_{\tau-1} + u_\tau$$

Subject to: $\sum_{\tau \in t} y_\tau^M = y_t^Q$ (aggregation constraint)

The GLS estimator enforces the constraint while minimizing the error variance.

**Application:** Monthly GDP estimates, regional accounts, national accounts at sub-annual frequency.

<!-- Speaker notes: Chow-Lin is the gold standard for temporal disaggregation. It's implemented in the tempdisagg R package and used by Eurostat and national statistical offices. The key word is "disaggregation" — you're going from low to high frequency, which is the opposite direction from nowcasting. This means the interpolated series has useful properties for historical analysis but introduces a generated-regressors problem if used in subsequent regressions. -->

---

## Interpolation: The Synthetic Data Problem

```python
# What Chow-Lin gives you:
interpolated_gdp = [1.2, 1.5, 1.8,   # Q1 months (sum = 4.5 = quarterly value)
                    0.9, 0.7, 0.5, ...]  # Q2 months

# What you should NOT do:
model = OLS(interpolated_gdp, some_monthly_x)
# ↑ Standard errors are WRONG because interpolated obs are correlated model outputs
```

**Correct use:** Interpolated series for visualization and historical analysis.
**Incorrect use:** Input to further econometric models without correcting for the interpolation error.

<!-- Speaker notes: This is one of the most common mistakes in applied macro. Researchers interpolate GDP to monthly frequency, then run a monthly regression. The standard errors from that monthly regression are too small because the correlated structure of the interpolation errors is ignored. This leads to overconfident estimates and too many "significant" findings. The fix is to either use the original quarterly data with MIDAS, or correct the standard errors for the two-step estimation. -->

---

## Strategy 3: Bridge Equations

**Step 1:** Quarterly regression on aggregated monthly indicator.
$$y_t^Q = \alpha + \sum_{k=0}^K \beta_k \tilde{x}_{t-k}^Q + \varepsilon_t$$

**Step 2:** When quarter is incomplete, forecast missing months.
$$x_\tau^M = \mu + \phi_1 x_{\tau-1}^M + \eta_\tau \quad \text{(AR model)}$$

**Step 3:** Aggregate forecasted months, plug into Step 1.

Used by ECB, Fed, and most major central banks through the 2000s–2010s.

<!-- Speaker notes: Bridge equations are the workhorse of institutional nowcasting. The ECB's Euro-area BMPE uses a system of bridge equations. The Fed's Greenbook historically used bridge equations as one input. They are interpretable and can be updated by hand as new data arrives. The main weakness is the two-step error compounding, which we quantify on the next slide. -->

---

## Error Compounding in Bridge Equations

Let $\delta_x$ = error in monthly forecast (Step 2 error).
The total nowcast error includes:

$$e_{\text{bridge}} \approx \varepsilon_{\text{quarterly}} + \hat{\beta} \cdot \frac{\delta_x}{m}$$

$$\text{Var}(e_{\text{bridge}}) = \underbrace{\sigma_\varepsilon^2}_{\text{Stage 1}} + \underbrace{\hat{\beta}^2 \cdot \frac{\sigma_\delta^2}{m}}_{\text{Stage 2 compound}}$$

**The larger $|\hat{\beta}|$, the more the monthly forecast error contaminates the nowcast.**

<!-- Speaker notes: This formula shows exactly why MIDAS dominates bridge equations: the two-stage procedure adds a variance term that single-step MIDAS doesn't have. The compounding is larger when the monthly indicator is highly informative (large beta) — perversely, a better indicator makes bridge equation error compounding worse. MIDAS does not have this property because it estimates everything jointly. -->

---

## Comparison: Side-by-Side

<div class="columns">

<div>

**Temporal Aggregation**
- Fast, interpretable
- Works for any sample size
- Loses all timing signal
- Biased if weights misspecified

**Chow-Lin Interpolation**
- Preserves aggregation constraint
- Good for historical series
- Synthetic data problem
- Not suitable for nowcasting

</div>

<div>

**Bridge Equation**
- Standard at central banks
- Handles ragged edge naturally
- Two-stage error compounding
- Aggregation weights fixed

**MIDAS**
- Learns aggregation weights
- Single-stage estimation
- Handles high freq ratios
- Requires nonlinear optimization

</div>

</div>

<!-- Speaker notes: The comparison table is the core takeaway of this guide. Notice that each traditional method has a genuine use case — aggregation for simple applications, Chow-Lin for historical disaggregation, bridge equations for institutional nowcasting. MIDAS dominates in terms of statistical efficiency but is more demanding computationally. In practice, many central banks now use MIDAS or DFM-based nowcasting alongside their bridge equation systems. -->

---

## When Traditional Methods Still Win

| Situation | Best Approach |
|-----------|--------------|
| Tiny sample ($T_L < 30$) | Aggregation (NLS needs data) |
| Need a monthly historical series | Chow-Lin interpolation |
| Institutionally mandated simplicity | Bridge equation |
| Real-time with ragged edge, few series | Bridge equation |
| Large panel + mixed frequencies | DFM-MIDAS |
| Single indicator, large sample | MIDAS |

**MIDAS shines when:** Frequency ratio is large, sample is medium-to-large, timing of within-period dynamics matters.

<!-- Speaker notes: This nuanced view is important for applied practitioners. MIDAS is not always the right answer. For a central bank that needs to explain its model to a policy committee, a simple bridge equation with three monthly indicators may be preferable to a MIDAS model with estimated Beta polynomial weights. Interpretability has value. The course teaches MIDAS precisely, but the goal is for students to be able to choose the right tool for the problem. -->

---

## The Econometric Cost of Pre-Aggregation

If the true model is:
$$y_t^Q = \alpha + \beta(L^{1/m}) x_\tau^M + \varepsilon_t$$

But we estimate:
$$y_t^Q = \alpha + \tilde{\beta}\, \tilde{x}_t^Q + \tilde{\varepsilon}_t$$

Then $\hat{\tilde{\beta}}$ is consistent for $\sum_j w_j \beta_j$ — **a weighted average of true lags**, not each lag individually.

Standard errors are also biased because $\tilde{\varepsilon}_t$ inherits MA structure from aggregation.

<!-- Speaker notes: This is the formal statement of the information loss. The aggregated OLS estimator is not wrong in the sense of being inconsistent for its target parameter — it correctly estimates the weighted average of lag coefficients. But it can't recover the individual lag coefficients, which is what matters for nowcasting and policy analysis. And the MA structure induced by aggregation makes standard tests invalid without correction. -->

---

## Real Example: IP Aggregation, 2020Q1

```python
# March 2020: COVID shock
monthly_ip_growth = [+0.003, -0.012, -0.065]  # Jan, Feb, Mar

# Different aggregations tell different stories:
last_period = -0.0650  # -6.5% — captures the shock
equal_avg   = -0.0247  # -2.5% — dilutes the shock by 3x
sum_         = -0.0740  # -7.4% — flow variable total

# In a bridge equation, the -2.47% enters the GDP nowcast
# An analyst watching daily news saw -6.5% in March alone
# The aggregation discards 4 percentage points of signal
```

During COVID, equal-weight aggregation systematically underestimated the Q1 shock because it averaged in January and February.

<!-- Speaker notes: The COVID example is pedagogically powerful because students lived through it. The GDP contraction in 2020Q1 was severe, but the equal-weight aggregated monthly IP number understated the March shock by 3x. A MIDAS model with end-loaded weights (Beta polynomial emphasizing recent months) would have detected this much earlier. This is a concrete example where the choice of aggregation method materially affected real-time assessment of economic conditions. -->

---

## Summary: What We've Learned

1. **Temporal aggregation** is fast but discards within-period timing information.
2. **Chow-Lin interpolation** solves disaggregation but creates synthetic data problems.
3. **Bridge equations** handle nowcasting but compound two-stage estimation errors.
4. **All three** pre-commit to aggregation weights before estimation.

> MIDAS solves problems 1, 3, and 4 simultaneously by estimating the weights from data in a single step.

**Next:** Module 01, Guide 01 — The MIDAS equation.

<!-- Speaker notes: The summary slide crystallizes the course motivation. Each traditional method has a specific failure mode. MIDAS addresses the most important failures — fixed weights, information loss, error compounding — at the cost of requiring nonlinear optimization. Module 01 develops the MIDAS equation in full mathematical detail and shows how to implement the optimization. -->

---

## Key Equations to Remember

| Method | Estimand | Weights |
|--------|----------|---------|
| Aggregation | $\tilde{\beta} = \sum_j w_j \beta_j$ | Fixed ($w_j = 1/m$) |
| Bridge | $\tilde{\beta}_1, \tilde{\beta}_2$ (two stages) | Fixed + estimated separately |
| MIDAS | $\alpha, \beta, \theta$ (weight shape) | **Learned from data** |

$$\text{MIDAS:} \quad y_t^Q = \alpha + \beta \cdot B(L^{1/m};\theta) \cdot x_\tau^M + \varepsilon_t$$

<!-- Speaker notes: This comparison table distills the key statistical difference between the three approaches. For aggregation and bridge equations, the weights are either fixed or estimated in a disconnected stage. For MIDAS, the weight parameters theta are estimated jointly with the regression coefficient beta. This joint estimation is what gives MIDAS its efficiency advantage. -->

---

## Further Reading

- **Chow & Lin (1971)** — Original interpolation paper; accessible and well-written
- **Foroni & Marcellino (2013)** — Survey of mixed-frequency econometric methods
- **Banbura et al. (2013)** — Now-casting and the real-time data flow; bridge equations in practice
- **ECB Working Paper on Bridge Equations** — How major central banks actually do this

<!-- Speaker notes: The Foroni and Marcellino survey is particularly useful as a reference — it covers all the methods discussed in this guide and MIDAS in one place. The ECB working paper series on nowcasting provides real-world context for how bridge equations are actually used in institutional settings. -->

# The Mixed-Frequency Problem

> **Reading time:** ~11 min | **Module:** 00 — Foundations | **Prerequisites:** None (entry point)


## In Brief

<div class="flow">
<div class="flow-step mint">1. Collect Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Identify Frequencies</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Align Time Indices</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Build MIDAS Regressors</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Economic data arrives at different frequencies: GDP quarterly, employment monthly, stock prices daily. The mixed-frequency problem is how to combine these streams coherently without discarding info...

</div>

Economic data arrives at different frequencies: GDP quarterly, employment monthly, stock prices daily. The mixed-frequency problem is how to combine these streams coherently without discarding information or introducing distortions.

## Key Insight

<div class="callout-insight">

**Insight:** The mixed-frequency approach preserves within-period dynamics that aggregation destroys. This is especially valuable when the timing of high-frequency movements carries economic information.

</div>


When you aggregate high-frequency data to match a low-frequency target, you throw away information. MIDAS regression retains that information by letting the model directly use high-frequency observations as regressors — with structure imposed on the lag weights so estimation remains tractable.

---

## Why Frequencies Differ

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


Economic measurement is costly. National accounts (GDP, investment, government spending) require extensive data collection and reconciliation — hence quarterly publication. Administrative data (payrolls, claims) comes monthly because it piggybacks on existing business processes. Financial data is transactional and therefore available at tick or daily frequency.

This is not an accident of convention. The measurement interval reflects the underlying data-generating process and the cost of measurement. Policy makers, traders, and forecasters must work across all three layers simultaneously.

### Publication Calendar (US Example)

| Indicator | Frequency | Typical Release |
|-----------|-----------|-----------------|
| GDP (advance) | Quarterly | ~30 days after quarter end |
| Nonfarm Payrolls | Monthly | First Friday of following month |
| Industrial Production | Monthly | ~15th of following month |
| Retail Sales | Monthly | ~15th of following month |
| S&P 500 | Daily | Continuous |
| 10-Year Treasury Yield | Daily | Continuous |
| Initial Jobless Claims | Weekly | Thursday |

The advance GDP estimate is released about a month after the quarter closes, and is subsequently revised twice. This means that in real time, the most recent GDP observation is always **stale** — sometimes by 4-5 months relative to the latest financial market data.

---

## The Aggregation Problem

### Formal Setup

Let $y_t^Q$ denote a quarterly variable (e.g., GDP growth) observed at $t = 1, 2, \ldots, T$ (in quarterly time units). Let $x_{\tau}^M$ denote a monthly variable (e.g., industrial production growth) observed at $\tau = 1, 2, \ldots, 3T$ (in monthly time units).

The natural alignment requires mapping three monthly observations to each quarterly observation. If we define the quarterly index $t$ and within-quarter month index $j \in \{1, 2, 3\}$, then the monthly observation $x_{3(t-1)+j}^M$ falls in quarter $t$.

### Three Aggregation Strategies

**Strategy 1: Last-period sampling**

$$\tilde{x}_t = x_{3t}^M$$

Use only the third month of each quarter. Simple but discards 2/3 of available information.

**Strategy 2: Simple average**

$$\tilde{x}_t = \frac{1}{3}\sum_{j=1}^{3} x_{3(t-1)+j}^M$$

Equal-weight aggregation. Loses within-quarter timing information.

**Strategy 3: Weighted average**

$$\tilde{x}_t = \sum_{j=1}^{3} w_j \cdot x_{3(t-1)+j}^M$$

With weights $w_j > 0$, $\sum w_j = 1$. Better, but weights must be specified a priori rather than learned from data.

### Information Loss: A Demonstration

Consider quarterly GDP growth regressed on quarterly-averaged industrial production growth (IP). The within-quarter pattern of IP carries information: if IP accelerates through a quarter, that is a different signal than the same average with IP decelerating. Simple aggregation conflates these two patterns.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate two quarters with identical averages but different within-quarter paths
months = np.arange(1, 7)
ip_accelerating = np.array([0.1, 0.3, 0.5, 0.2, 0.4, 0.6])  # rising
ip_decelerating = np.array([0.5, 0.3, 0.1, 0.6, 0.4, 0.2])  # falling

print(f"Q1 average (accelerating): {ip_accelerating[:3].mean():.3f}")
print(f"Q1 average (decelerating): {ip_decelerating[:3].mean():.3f}")
# Both equal 0.300 — identical after aggregation

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, ip, label in zip(axes,
                          [ip_accelerating, ip_decelerating],
                          ["Accelerating IP", "Decelerating IP"]):
    ax.bar(months, ip, color=["steelblue"]*3 + ["coral"]*3, alpha=0.8)
    ax.axhline(ip[:3].mean(), color="blue", linestyle="--", label=f"Q1 avg = {ip[:3].mean():.2f}")
    ax.axhline(ip[3:].mean(), color="red", linestyle="--", label=f"Q2 avg = {ip[3:].mean():.2f}")
    ax.set_title(label)
    ax.set_xlabel("Month")
    ax.set_ylabel("IP Growth")
    ax.legend()
plt.suptitle("Same quarterly averages, different within-quarter dynamics")
plt.tight_layout()
plt.show()
```

</div>
</div>

After quarterly averaging, the two scenarios are indistinguishable — despite potentially carrying very different signals about momentum.

---

## Bridge Equations: A Traditional Workaround

Before MIDAS, the standard approach was a **bridge equation**:

1. Model $y_t^Q$ as a function of $\tilde{x}_t^Q$ (aggregated monthly data)
2. Separately forecast $x_\tau^M$ to fill in missing monthly values within the current quarter
3. Aggregate the forecast monthly values and plug into the bridge equation

$$y_t^Q = \alpha + \beta \tilde{x}_t^Q + \varepsilon_t$$

This introduces a two-step estimation problem. The error in forecasting the monthly series compounds with the error in the quarterly regression, biasing inference. The aggregation weights are fixed, not estimated from data.

---

## The MIDAS Solution

MIDAS (Mixed Data Sampling) regression, introduced by Ghysels, Santa-Clara, and Valkanov (2004), solves the aggregation problem directly:

$$y_t^Q = \alpha + \beta \cdot B(L^{1/m}; \theta) \cdot x_{\tau}^M + \varepsilon_t$$

where $B(L^{1/m}; \theta)$ is a lag polynomial in the high-frequency lag operator $L^{1/m}$, and $\theta$ are parameters that control the shape of the lag weights.

The key insight: **the weights $w_j(\theta)$ are estimated from the data**, not imposed. If the most recent monthly observation matters most, the model learns that. If a distributed pattern fits better, the model captures it.

We cover the full MIDAS equation in Module 01. The remainder of this module focuses on understanding the data environment.

---

## Real-World Implications

### For Macroeconomic Policy

Central banks conduct policy in real time. The Federal Reserve sets interest rates based on its assessment of current conditions — but GDP data lags by a full quarter. The Philadelphia Fed's Survey of Professional Forecasters, the New York Fed's Nowcasting Report, and similar exercises all grapple with the mixed-frequency problem.

### For Asset Management

Macro hedge funds trade on economic signals. A fund that can nowcast GDP growth one month before the advance release with meaningful accuracy has a significant information advantage. The challenge is that all available indicators are monthly or daily — none are quarterly.

### For Risk Management

Value-at-Risk and stress testing require understanding how macro conditions affect portfolio risk. Models connecting daily market variables to quarterly economic outcomes are inherently mixed-frequency.

---

## The Ragged Edge

In real time, data availability creates a "ragged edge": within the current quarter, some monthly releases have arrived and some have not. In month 2 of a quarter, you have:

- Two monthly IP observations for the current quarter
- Three monthly employment figures for the current quarter
- Zero monthly retail sales (released later)
- Sixty or so daily financial observations

Any nowcasting system must handle this unbalanced panel of current-quarter information. We address this explicitly in Module 03.

```
Quarter t timeline:
Month 1: IP(1), Payrolls(1), [many daily obs]
Month 2: IP(2), Payrolls(2), [many daily obs]
Month 3: ----, -----------, [some daily obs]  ← Current position
Quarter end: GDP release (30 days later)
```

---

## Common Pitfalls

**Pitfall 1: Ignoring revision history.** GDP and many monthly series are revised, sometimes substantially. A model trained on vintage data will look better in sample than it will perform in real time. Always distinguish "real-time" from "revised" data when evaluating nowcasting models.

**Pitfall 2: Data snooping across frequencies.** Lagged relationships look spuriously tight when you allow many lags from many series. The MIDAS parameterization partly addresses this by imposing smooth weight functions — but lag and variable selection still require care.

**Pitfall 3: Conflating nowcasting and forecasting.** A nowcast of the current quarter differs from a forecast of next quarter. The timing of information arrival matters for which horizon is being targeted.

**Pitfall 4: Fixed aggregation in pre-processing.** Averaging or summing monthly data before any modeling throws away information that MIDAS can use. Resist the temptation to pre-aggregate.

---

## Connections

- **Builds on:** Time series fundamentals (stationarity, lag operators), OLS regression
- **Leads to:** MIDAS regression (Module 01), Nowcasting systems (Module 03)
- **Related to:** State space models, Kalman filtering, factor models (Module 04)

---

## Practice Problems

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


1. The Conference Board Leading Economic Index is released monthly. It is used to predict business cycle turning points, which are designated quarterly. Describe how you would structure a mixed-frequency analysis to predict NBER recession dates using the LEI.

2. Suppose you have daily stock returns ($m = 65$ trading days per quarter) and quarterly earnings growth. Sketch the data alignment problem. How many daily observations would enter a MIDAS model with four quarterly lags?

3. Why might last-period sampling (using only the final month of each quarter) perform surprisingly well in some applications? Under what conditions would you expect the three-month average to dominate?

---

## Further Reading

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2004). "The MIDAS touch: Mixed data sampling regression models." *CIRANO Working Papers.*
- Giannone, D., Reichlin, L., & Small, D. (2008). "Nowcasting: The real-time informational content of macroeconomic data." *Journal of Monetary Economics*, 55(4), 665–676.
- Bańbura, M., Giannone, D., & Reichlin, L. (2010). "Nowcasting." In *Oxford Handbook of Economic Forecasting.*
- Federal Reserve Bank of New York Nowcasting Report: https://www.newyorkfed.org/research/policy/nowcast


---

## Cross-References

<a class="link-card" href="./02_traditional_solutions_guide.md">
  <div class="link-card-title">02 Traditional Solutions</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_traditional_solutions_slides.md">
  <div class="link-card-title">02 Traditional Solutions — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_course_datasets_guide.md">
  <div class="link-card-title">03 Course Datasets</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_course_datasets_slides.md">
  <div class="link-card-title">03 Course Datasets — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


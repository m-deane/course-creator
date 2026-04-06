# The Nowcasting Problem

> **Reading time:** ~14 min | **Module:** 03 — Nowcasting | **Prerequisites:** Module 2


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

**Key Concept Summary:** Nowcasting is real-time estimation of current-period GDP growth using high-frequency data that becomes available before the official GDP release. MIDAS regression is the workhorse tool because it n...

</div>

Nowcasting is real-time estimation of current-period GDP growth using high-frequency data that becomes available before the official GDP release. MIDAS regression is the workhorse tool because it naturally handles the mixed-frequency structure of the nowcasting problem.

## Key Insight

<div class="callout-insight">

**Insight:** Real-time nowcasting is fundamentally different from pseudo out-of-sample backtesting. The ragged-edge data structure means your model sees different information at different points within a quarter.

</div>


At any point within a quarter, some monthly indicators (industrial production, employment, retail sales) have already been published for the current quarter while others have not. This creates a "ragged edge" — the MIDAS matrix has missing values for the most recent months. The nowcast updates each time a new monthly release arrives.

---

## The Timing Problem

<div class="callout-warning">

**Warning:** Pseudo out-of-sample exercises that do not properly account for the real-time data vintage will overstate nowcast accuracy. Always use the ragged-edge structure that would have been available at each historical nowcast date.

</div>


GDP for quarter $t$ is released with a lag of approximately 4–6 weeks after the quarter ends. During the quarter and immediately after, we want to estimate $y_t$ using whatever high-frequency data is available.

### Publication Calendar

A typical quarterly GDP publication sequence:

```
Quarter Q:  [Jan]  [Feb]  [Mar]  |  GDP release ~late April
                                      (4-5 weeks after quarter end)

Month 1 IP release:  ~Feb 15  (covers January)
Month 2 IP release:  ~Mar 15  (covers February)
Month 3 IP release:  ~Apr 15  (covers March)
GDP flash release:   ~Apr 26  (first estimate of Q GDP)
```

By the time GDP is released, all three monthly IP figures are available. But the "live" nowcast problem requires estimating GDP before that — when only 1 or 2 months of IP are available.

### Nowcast Vintage Points

| Date | Available IP months | Nowcast type |
|------|--------------------|-----------|
| Jan 15 | None yet (last quarter's data only) | Backcast prior quarter |
| Feb 15 | Month 1 (January) | 1-month nowcast |
| Mar 15 | Months 1-2 (Jan-Feb) | 2-month nowcast |
| Apr 15 | Months 1-2-3 (Jan-Feb-Mar) | 3-month (complete quarter) |
| Apr 26 | GDP released | No longer needed |

---

## The Ragged Edge

The "ragged edge" refers to the fact that the MIDAS data matrix for the most recent quarter is incomplete — it has values for available months and is missing for future months.

### Illustration for K=12 (4 quarterly lags)

```
Quarter  |  j=0   j=1   j=2  |  j=3   j=4   j=5  | j=6...j=11
         | (cur Q month 3) 2  1  | Q-1 months 3,2,1 | Q-2, Q-3
---------+---------------------+-------------------+---------
Q-3      |  IP    IP    IP    |  IP    IP    IP    | IP ... IP   (complete)
Q-2      |  IP    IP    IP    |  IP    IP    IP    | IP ... IP   (complete)
Q-1      |  IP    IP    IP    |  IP    IP    IP    | IP ... IP   (complete)
Q (now)  |  ???   IP    IP    |  IP    IP    IP    | IP ... IP   (ragged)
```

At the "2-month nowcast" point, the current quarter has 2 of 3 IP months available. Lag j=0 (the most recent month, March) is missing.

### Handling the Ragged Edge in MIDAS

**Option 1: Backward shift (truncate K)**
Drop the missing lag and use K'=K-1 lags. The weight function covers lags $j=1, ..., K-1$ instead of $j=0, ..., K-1$. Slightly biased at the front end but simple.

**Option 2: EM imputation**
Use the Expectation-Maximization algorithm to impute the missing monthly value, then apply MIDAS as usual. More complex but preserves the weight structure.

**Option 3: Ragged-edge MIDAS**
Condition the nowcast on the available lags and use the estimated weight function to project forward:

$$\hat{y}_t^{(h)} = \hat{\alpha} + \hat{\beta} \sum_{j=h}^{K-1} \frac{w_j(\hat{\theta})}{\sum_{j'=h}^{K-1} w_{j'}(\hat{\theta})} x_{mt-j}$$

where $h$ is the number of missing months at the "ragged edge". This re-weights the observed data to maintain the correct shape.


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def midas_nowcast_ragged(alpha, beta, theta1, theta2, X_current, h_missing, K):
    """
    Nowcast for a quarter with h_missing high-frequency observations missing.

    Parameters
    ----------
    alpha, beta : float — estimated MIDAS coefficients
    theta1, theta2 : float — estimated weight parameters
    X_current : np.ndarray, shape (K - h_missing,)
        Available high-frequency data for current quarter (lags h_missing to K-1).
    h_missing : int — number of most-recent lags missing (0 = complete quarter)
    K : int — total lag order

    Returns
    -------
    nowcast : float
    """
    from scipy.stats import beta as beta_dist

    # Full weights (0 to K-1)
    x = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x, theta1, theta2)
    s = raw.sum()
    w_full = raw / s if s > 1e-12 else np.ones(K) / K

    # Available weights (lags h_missing to K-1)
    w_avail = w_full[h_missing:]
    w_avail_norm = w_avail / w_avail.sum()

    # Weighted aggregate using only available lags
    xw_avail = X_current @ w_avail_norm
    return alpha + beta * (w_avail.sum() * xw_avail / w_avail_norm.sum())
```

</div>
</div>

---

## Forecast Evolution: The Nowcast Path

As the quarter progresses and more monthly data arrives, the nowcast updates. This "forecast evolution" plot is a key diagnostic:

```
Nowcast for 2024Q3 GDP growth:

  2.5% |                                       *  (3-month, complete)
  2.0% |                          *            |
  1.5% |             *            |            |
  1.0% | *           |            |            |
       +-------------+------------+------------+
         Feb 15      Mar 15       Apr 15       Actual release
       (1-month)   (2-month)   (3-month)
```

The nowcast typically converges toward the eventual GDP release as more information arrives. The rate of convergence depends on how front-loaded the weight function is — if $w_0$ is large, getting month 3's IP substantially updates the nowcast.

---

## Nowcast Comparison Framework

A complete nowcasting exercise compares:

1. **AR benchmark:** Pure autoregressive model for GDP (no high-frequency data)
2. **Equal-weight MIDAS:** OLS-aggregate with simple averaging
3. **Beta MIDAS:** Restricted polynomial weights (the main model)
4. **Survey consensus:** Professional forecaster consensus (external benchmark)

Evaluation metric: Root Mean Squared Error (RMSE) computed over the evaluation window.

$$\text{RMSE}_h = \sqrt{\frac{1}{N_{eval}} \sum_{t \in \text{eval}} (y_t - \hat{y}_{t|t-h}^{(m_h)})^2}$$

where $h$ indicates the horizon (number of missing months = 0, 1, 2) and $m_h$ is the information set at horizon $h$.

---

## Real-Time Data Considerations

A fully realistic nowcasting exercise requires **real-time vintages** — using only data that was actually available at each point in time. GDP is subsequently revised, and industrial production is also subject to data revisions.

For this course, we use the current final vintage (as downloaded from FRED) and focus on the methodological aspects. The `.claude_plans/` directory contains notes on setting up a real-time data exercise for advanced users.

---

## Common Pitfalls

**Pitfall 1: Using final-vintage data for "real-time" evaluation.** If you use today's revised GDP as the actual outcome when evaluating forecasts made in 2020, you're comparing against a target that wasn't known at the time. For our purposes, we acknowledge this limitation.

**Pitfall 2: Ignoring the publication lag.** IP for month $m$ is published ~15 days into month $m+1$. A Feb 15 nowcast uses January IP but not February IP.

**Pitfall 3: Not accounting for COVID outliers.** 2020Q1 and 2020Q2 GDP growth (-1.3%, -8.9% annualized) are extreme outliers that dominate any RMSE calculation over windows including them. Always report RMSE both including and excluding COVID quarters.

---

## Connections

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


- **Builds on:** Module 01 (MIDAS fundamentals), Module 02 (estimation and inference)
- **Leads to:** Module 04 (Dynamic Factor Models for nowcasting with many indicators)
- **Related to:** Real-time data vintages, forecast combination, Giannone-Reichlin-Small (2008)

---

## Practice Problems

1. At the "1-month nowcast" point for 2024Q3, which months of industrial production are available? Which are missing? Write out the ragged-edge MIDAS matrix for K=6.

2. If $\hat{w}_0 = 0.25$ (most recent month carries 25% of the weight), how much does the nowcast change when month 3 of the quarter is released with IP growth of $+0.5\%$ instead of the expected $+0.3\%$?

3. Derive the formula for the nowcast update $\hat{y}_t^{(0)} - \hat{y}_t^{(1)}$ in terms of the MIDAS weight function and the newly released monthly observation.


---

## Cross-References

<a class="link-card" href="./02_direct_vs_iterated_guide.md">
  <div class="link-card-title">02 Direct Vs Iterated</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_direct_vs_iterated_slides.md">
  <div class="link-card-title">02 Direct Vs Iterated — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


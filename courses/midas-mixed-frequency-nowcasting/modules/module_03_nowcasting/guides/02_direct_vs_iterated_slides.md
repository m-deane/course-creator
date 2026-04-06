---
marp: true
theme: course
paginate: true
math: mathjax
---

<!-- _class: lead -->

# Direct vs. Iterated MIDAS

## Nowcasting Strategies and MIDAS-AR

**Mixed-Frequency Models: MIDAS Regression and Nowcasting**
Module 03 — Guide 02

<!-- Speaker notes: This guide covers the two main multi-horizon strategies for MIDAS nowcasting: direct (one model per horizon) and iterated (one model, iterate forward). The key decision is whether to re-estimate the MIDAS parameters for each horizon or reuse a single set. We also introduce the MIDAS-AR extension which is the standard specification for quarterly GDP. The practical punchline: use direct MIDAS for nowcasting (h<=1 quarter) and MIDAS-AR for short-run forecasting (h=1-4 quarters). -->

---

## Two Strategies

<div class="columns">

<div>

### Direct MIDAS

Separate model for each horizon $h$:

$$y_{t+h} = \alpha^{(h)} + \beta^{(h)}\tilde{x}_t^{(h)}(\theta^{(h)}) + \varepsilon^{(h)}$$

**Fit:** Directly minimizes h-step MSE
**Cost:** K separate models needed

</div>

<div>

### Iterated MIDAS

One model, iterate AR forward:

$$y_{t+1} = \alpha + \rho y_t + \beta\tilde{x}_t(\theta) + \varepsilon$$

$$\hat{y}_{t+h|t} = \text{iterate}(\hat{\rho}, \hat{y}_{t+1|t}, ...)$$

**Fit:** Minimizes 1-step MSE
**Cost:** Requires forecasting $x_{HF}$

</div>

</div>

<!-- Speaker notes: The fundamental tradeoff is between fitting each horizon optimally (direct) versus using a unified model (iterated). Direct MIDAS gives each horizon its own weight function — the optimal weights for predicting 1-quarter-ahead may differ from those for 2-quarters-ahead. Iterated MIDAS constrains all horizons to use the same weight function but avoids the need to fit multiple models. The iterated approach requires forecasting the monthly indicators to fill in future values, which adds a second source of forecast error. -->

<div class="callout-key">

The key advantage of MIDAS is preserving high-frequency information that temporal aggregation destroys.

</div>

---

## Which Performs Better?

Empirical evidence (Marcellino, Stock, Watson 2006; Foroni et al. 2015):

| Horizon | Winner |
|---------|--------|
| 1–2 quarters | **Direct** (usually) |
| 3–4 quarters | Tied / Iterated |
| 5–8 quarters | **Iterated** (usually) |

**Intuition:** At short horizons, the heterogeneous data environment (ragged edge) means direct models adapt better. At long horizons, parameter estimation error in multiple direct models dominates.

**For nowcasting (h ≤ 1Q):** Use direct MIDAS.

<!-- Speaker notes: The empirical literature is consistent on this point: direct forecasts outperform iterated forecasts at short horizons for macroeconomic aggregates. The reason is model misspecification — the VAR-like structure underlying iterated forecasting assumes the same dynamics hold at all horizons, which is unrealistic. The direct approach sidesteps this by fitting each horizon directly. However, at long horizons (5-8 quarters), the direct approach requires many separate parameter estimates and suffers from parameter uncertainty, while the iterated approach pools information across time steps. -->

<div class="callout-insight">

**Insight:** Parsimonious weight functions with 2-3 parameters can capture decay patterns that unrestricted models need 12+ parameters to approximate.

</div>

---

## The MIDAS-AR Model

Adds a quarterly AR lag to standard MIDAS:

$$y_t = \alpha + \rho y_{t-1} + \beta \sum_{j=0}^{K-1} w_j(\theta) x_{mt-j} + \varepsilon_t$$

**Parameters:** $(\alpha, \rho, \beta, \theta_1, \theta_2)$ — 5 free parameters vs. 4 for plain MIDAS.

**Profile NLS:** Optimize over $(\theta_1, \theta_2)$; solve $(\alpha, \rho, \beta)$ by OLS.

$$\text{Regress: } y_t \sim (1, y_{t-1}, \tilde{x}_t(\theta)) \text{ for fixed } \theta$$

<!-- Speaker notes: The MIDAS-AR extension is the standard specification in most applied nowcasting work. The intuition is simple: quarterly GDP has mild positive autocorrelation (AR coefficient roughly 0.2-0.4), and ignoring this leaves autocorrelation in the residuals that biases inference. The AR term absorbs the persistence in GDP growth that isn't explained by the current quarter's IP activity. The profile NLS setup is straightforward: for fixed theta, regress Y on (1, Y_lag, xw) using ordinary OLS to get the three regression coefficients. This preserves the profile NLS framework exactly. -->

<div class="callout-warning">

**Warning:** Always account for the real-time data vintage when evaluating nowcast performance. Using revised data overstates accuracy.

</div>

---

## Profile NLS for MIDAS-AR

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def profile_sse_ar(theta, Y, X):
    """Profile SSE for MIDAS-AR(1)."""
    t1, t2 = theta
    if t1 <= 0.01 or t2 <= 0.01:
        return 1e10
    K = X.shape[1]
    x_pts = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x_pts, t1, t2)
    s = raw.sum()
    w = raw / s if s > 1e-12 else np.ones(K) / K
    xw = X @ w

    # Align: Y[1:] on [1, Y[:-1], xw[1:]]
    Y_dep = Y[1:]
    Z = np.column_stack([np.ones(len(Y)-1), Y[:-1], xw[1:]])
    params = np.linalg.lstsq(Z, Y_dep, rcond=None)[0]
    resid = Y_dep - Z @ params
    return np.sum(resid**2)
```

</div>

<!-- Speaker notes: The key difference from the standard profile_sse is the alignment: we use Y[1:] as the dependent variable and include Y[:-1] as a regressor. The xw also shifts to xw[1:] to align with the dependent variable. This is the standard lag alignment for an AR(1) model — we lose one observation from the beginning of the sample. For T=100 quarterly observations, we lose 1, giving T=99 effective observations. The OLS inside the profile SSE now estimates three coefficients (alpha, rho, beta) instead of two. -->

<div class="callout-info">

**Info:** MIDAS models can handle any frequency ratio: monthly-to-quarterly (3:1), daily-to-monthly (~22:1), or even tick-to-daily.

</div>

---

## When to Add AR Terms

Three checks (all should agree before adding AR):

**Check 1:** Ljung-Box on MIDAS residuals (p < 0.10 → add AR)

**Check 2:** BIC comparison:
$$\text{BIC(MIDAS-AR)} < \text{BIC(MIDAS)}$$

**Check 3:** Expanding-window RMSE improvement
$$\text{RMSE(MIDAS-AR)} < \text{RMSE(MIDAS)}$$

Typical result for quarterly US GDP: AR term **does** help (AR ≈ 0.25, LB p ≈ 0.08).

<!-- Speaker notes: The three checks provide different but complementary evidence for the AR term. Ljung-Box tests whether the residuals show statistically significant serial correlation. BIC tests whether the improvement in fit (lower SSE) from the AR term outweighs the extra parameter penalty. The expanding-window RMSE tests whether the AR term actually improves out-of-sample prediction. For US GDP, all three typically agree that the AR term is beneficial. The AR coefficient is typically 0.2-0.3, which is modest but economically meaningful. -->

---

## Ragged-Edge Nowcast Horizons

Direct MIDAS at three vintage points:

```
Vintage          Available months    Model spec
─────────────────────────────────────────────────
3-month (h=0)   M3, M2, M1         Standard K=12
2-month (h=1)   M2, M1             K'=11 (drop j=0)
1-month (h=2)   M1 only            K''=10 (drop j=0,1)
```

Each horizon gets its own theta estimate!

$$\hat{\theta}^{(h)} = \arg\min_\theta Q_{\text{profile}}^{(h)}(\theta, Y, X^{(h)})$$

<!-- Speaker notes: The key implementation detail is that each horizon gets a different data matrix (different K) and therefore a different theta estimate. The 1-month nowcast uses the MIDAS weight function fitted on K-2 lags, the 2-month nowcast uses K-1 lags, and the 3-month nowcast uses the full K lags. The weight functions will be slightly different at each horizon because they're fitted on different lag structures. This is the core advantage of the direct approach: the model adapts to whatever data is available. -->

---

## Forecast Evolution: A Key Diagnostic

```
Nowcast path for a quarter with positive IP surprise in month 3:

GDP  ▲
     |                     •  ← 3-month (complete)
     |              •
     |       •
     |  •
     +--+----+----+----→ Time
       h=2  h=1  h=0  Actual
     (1 mo)(2 mo)(3 mo)
```

**Update at each step:**
$$\Delta\hat{y} = \hat{\beta}^{(h)} \cdot \hat{w}_0^{(h)} \cdot x_{IP,\text{new}}$$

<!-- Speaker notes: The forecast evolution plot is one of the most informative diagnostics for a nowcasting model. A well-behaved nowcast should show smooth revisions toward the actual GDP value as each monthly observation arrives. If the nowcast jumps erratically — large revision at h=1 but small at h=0, or vice versa — this suggests the weight function is not capturing the information structure well. The update formula shows that the nowcast revision is proportional to beta (the IP-GDP transmission coefficient) times the weight on the new lag times the surprise in the new monthly observation. -->

---

## Practical Recommendations

<div class="columns">

<div>

**For nowcasting (h ≤ 1Q):**
- Use direct MIDAS
- Separate theta per horizon
- Report all three vintages
- Always show forecast evolution

</div>

<div>

**For short-run forecasting (h = 1–4Q):**
- Use MIDAS-AR (iterated)
- Check Ljung-Box first
- BIC to confirm AR term
- Use expanding-window RMSE

</div>

</div>

**Both cases:** Compare to AR(1) benchmark and equal-weight MIDAS.

<!-- Speaker notes: These recommendations are grounded in the empirical literature. The practical implementation difference is modest but real: for nowcasting, you fit three models (one per vintage) and track the forecast evolution. For short-run forecasting, you fit one MIDAS-AR model and report the h-step forecasts from it. In both cases, the AR(1) benchmark is essential — if your model doesn't beat a pure AR(1) in RMSE, the high-frequency data isn't helping, and you should investigate why. -->

---

## Summary

| Strategy | Horizon | Key Feature | Typical RMSE gain vs AR(1) |
|----------|---------|-------------|---------------------------|
| Direct MIDAS | Nowcast | One model per vintage | 10–20% |
| Iterated MIDAS-AR | 1–4 quarter | Single consistent model | 8–15% |
| Equal-weight | Any | No polynomial | 5–12% |

**Main finding:** Direct MIDAS dominates at nowcast horizons; MIDAS-AR better at multi-step.

**Next:** Notebook 01 — implementing the GDP nowcast workflow end-to-end.

<!-- Speaker notes: The summary table gives ballpark RMSE improvements relative to the AR(1) benchmark. These are typical values from the US GDP nowcasting literature — actual results vary by sample period, indicator choice, and whether COVID quarters are included. The key takeaway is that MIDAS provides meaningful improvements over simple AR forecasting, and that the direct approach is generally preferred for within-quarter nowcasting while MIDAS-AR is preferred for multi-quarter horizons. -->

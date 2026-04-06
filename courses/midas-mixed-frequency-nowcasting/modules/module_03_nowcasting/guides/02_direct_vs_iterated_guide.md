# Direct vs. Iterated MIDAS Nowcasting

> **Reading time:** ~13 min | **Module:** 03 — Nowcasting | **Prerequisites:** Module 2


## In Brief

<div class="flow">
<div class="flow-step mint">1. Load Ragged Data</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Estimate MIDAS</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Generate Nowcast</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Update as Data Arrives</div>
</div>


<div class="callout-key">

**Key Concept Summary:** Two strategies for producing multi-horizon MIDAS nowcasts: direct (one model per horizon, minimize h-step MSE) and iterated (one model for one-step-ahead, iterate the AR component forward). For qua...

</div>

Two strategies for producing multi-horizon MIDAS nowcasts: direct (one model per horizon, minimize h-step MSE) and iterated (one model for one-step-ahead, iterate the AR component forward). For quarterly GDP nowcasting with monthly indicators, direct MIDAS typically outperforms iterated MIDAS at short horizons where the data environment is heterogeneous across the quarter.

## Key Insight

<div class="compare">
  <div class="compare-card">
    <div class="header before">Direct Forecast</div>
    <div class="body">
      Estimates a separate model for each forecast horizon. No error accumulation. May be inconsistent across horizons.
    </div>
  </div>
  <div class="compare-card">
    <div class="header after">Iterated Forecast</div>
    <div class="body">
      Estimates a one-step model and iterates forward. Consistent across horizons. But errors compound at longer horizons.
    </div>
  </div>
</div>

<div class="callout-insight">

**Insight:** Real-time nowcasting is fundamentally different from pseudo out-of-sample backtesting. The ragged-edge data structure means your model sees different information at different points within a quarter.

</div>


The direct approach avoids accumulating forecast errors across iterations at the cost of requiring a separate model for each horizon. The iterated approach uses a single consistent model but amplifies any misspecification when iterated forward. For nowcasting (h ≤ 1 quarter), the practical difference is small but the direct approach is more standard in applied work.

---

## The Two Strategies

<div class="callout-warning">

**Warning:** Pseudo out-of-sample exercises that do not properly account for the real-time data vintage will overstate nowcast accuracy. Always use the ragged-edge structure that would have been available at each historical nowcast date.

</div>


### Direct MIDAS

Fit a separate model for each forecast horizon $h$:

$$y_{t+h} = \alpha^{(h)} + \beta^{(h)} \sum_{j=0}^{K-1} w_j^{(h)}(\theta^{(h)}) x_{mt-j}^{(h)} + \varepsilon_{t+h}^{(h)}$$

The superscript $(h)$ emphasizes that every parameter can differ by horizon. The data matrix $X^{(h)}$ uses lags starting from the end of the observation window for horizon $h$.

**Pros:** Avoids error propagation. Parameters directly minimize h-step MSE.
**Cons:** Requires fitting K separate models. Parameters may be inconsistent across horizons.

### Iterated MIDAS

Fit one MIDAS model for one-step-ahead prediction:

$$y_{t+1} = \alpha + \rho y_t + \beta \sum_{j=0}^{K-1} w_j(\theta) x_{mt-j} + \varepsilon_{t+1}$$

For horizon $h > 1$, iterate the AR component:

$$\hat{y}_{t+h|t} = \hat{\alpha}\frac{1 - \hat{\rho}^h}{1 - \hat{\rho}} + \hat{\rho}^h y_t + \hat{\beta}\sum_{j=0}^{K-1} w_j(\hat{\theta})\hat{x}_{m,t+h-j}^{(h)}$$

where future monthly values $\hat{x}_{m,t+h-j}^{(h)}$ must be forecast separately.

**Pros:** Single consistent model. Efficiently uses all available data.
**Cons:** Requires forecasting monthly indicators as well. Misspecification compounds.

---

## The MIDAS-AR Extension

The most common practical specification combines MIDAS with an AR(1) term for the low-frequency dependent variable:

$$y_t = \alpha + \rho y_{t-1} + \beta \sum_{j=0}^{K-1} w_j(\theta) x_{mt-j} + \varepsilon_t$$

Parameters: $(α, ρ, β, θ_1, θ_2)$ — five free parameters.

Estimation: Profile NLS over $(θ_1, θ_2)$ with $(α, ρ, β)$ solved by regressing $y_t$ on $(1, y_{t-1}, \tilde{x}_t(\theta))$.


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def profile_sse_ar(theta, Y, X):
    """
    Profile SSE for MIDAS-AR(1): Y[t] = alpha + rho*Y[t-1] + beta*X[t]@w + e.

    Note: Y and X must already be aligned (Y[1:] regressed on Y[:-1] and X[1:]).
    """
    import numpy as np
    from scipy.stats import beta as beta_dist

    t1, t2 = theta
    if t1 <= 0.01 or t2 <= 0.01:
        return 1e10

    K = X.shape[1]
    x = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x, t1, t2)
    s = raw.sum()
    w = raw / s if s > 1e-12 else np.ones(K) / K

    xw = X @ w

    # Regress Y[1:] on [1, Y[:-1], xw[1:]] (skip first observation for AR)
    T = len(Y)
    Y_dep = Y[1:]
    Z = np.column_stack([np.ones(T-1), Y[:-1], xw[1:]])
    params = np.linalg.lstsq(Z, Y_dep, rcond=None)[0]
    resid = Y_dep - Z @ params
    return np.sum(resid**2)
```

</div>
</div>

---

## When to Use Each Strategy

### Decision Criteria

| Criterion | Favor Direct | Favor Iterated |
|-----------|-------------|----------------|
| Horizon | Short (≤ 1 quarter) | Longer (> 1 quarter) |
| Data availability | Ragged edge prominent | Complete quarters |
| Model complexity | Single-indicator | Multi-indicator |
| Sample size | Large (T > 100) | Small-to-moderate |
| AR in residuals | Unlikely | Likely |

**Rule of thumb for quarterly GDP nowcasting:**
- Use direct MIDAS for 1-month and 2-month nowcasts
- Use MIDAS-AR (iterated) for 1–2 quarter-ahead forecasts

### Empirical Evidence

Marcellino, Stock, and Watson (2006) find that for quarterly US GDP, direct forecasts beat iterated forecasts at short horizons (1–2 quarters) but the gap closes at longer horizons (4–8 quarters). Foroni, Marcellino, and Schumacher (2015) find similar results specifically for MIDAS.

---

## Implementing the Direct Approach

For a $h$-period horizon nowcast using the direct MIDAS approach:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from scipy.stats import beta as beta_dist
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression


def build_midas_matrix_horizon(y_low_freq, x_high_freq, K, h_missing):
    """
    Build MIDAS matrix for a specific nowcast horizon.

    Parameters
    ----------
    y_low_freq : pd.Series (quarterly)
    x_high_freq : pd.Series (monthly)
    K : int — total monthly lags
    h_missing : int — missing months at the ragged edge (0=complete, 1=1 missing, 2=2 missing)

    Returns
    -------
    Y : np.ndarray (T,)
    X : np.ndarray (T, K - h_missing)
        Note: shape is (T, K - h_missing) because h_missing lags are not yet available.
    """
    import pandas as pd

    # Convert to period index
    if hasattr(y_low_freq.index, 'to_period'):
        y_q = y_low_freq.copy()
        y_q.index = y_low_freq.index.to_period('Q')
    else:
        y_q = y_low_freq.copy()
        y_q.index = pd.PeriodIndex(y_low_freq.index, freq='Q')

    if hasattr(x_high_freq.index, 'to_period'):
        x_m = x_high_freq.copy()
        x_m.index = x_high_freq.index.to_period('M')
    else:
        x_m = x_high_freq.copy()
        x_m.index = pd.PeriodIndex(x_high_freq.index, freq='M')

    rows_Y, rows_X = [], []
    K_avail = K - h_missing  # Number of available lags

    for q in y_q.index:
        # Start from lag h_missing (skipping the unavailable recent months)
        last_available = q.asfreq('M', how='end') - h_missing
        lags = [last_available - i for i in range(K_avail)]

        if any(lag not in x_m.index for lag in lags):
            continue
        if q not in y_q.index:
            continue

        rows_Y.append(y_q[q])
        rows_X.append([x_m[lag] for lag in lags])

    return np.array(rows_Y), np.array(rows_X)


def estimate_midas_direct(Y, X, starts=None):
    """
    Estimate direct MIDAS by profile NLS.
    Handles arbitrary K (uses X.shape[1] as effective lag count).
    """
    from scipy.stats import beta as beta_dist

    if starts is None:
        starts = [(1.0, 5.0), (1.5, 4.0), (2.0, 3.0)]

    K = X.shape[1]

    def psse(theta):
        t1, t2 = theta
        if t1 <= 0.01 or t2 <= 0.01:
            return 1e10
        x_pts = (np.arange(K) + 0.5) / K
        raw = beta_dist.pdf(1 - x_pts, t1, t2)
        s = raw.sum()
        w = raw / s if s > 1e-12 else np.ones(K) / K
        xw = X @ w
        xc = xw - xw.mean()
        yc = Y - Y.mean()
        ss = np.dot(xc, xc)
        if ss < 1e-12:
            return np.sum((Y - Y.mean())**2)
        beta_ = np.dot(yc, xc) / ss
        alpha_ = Y.mean() - beta_ * xw.mean()
        return np.sum((Y - alpha_ - beta_ * xw)**2)

    best_sse = np.inf
    best_res = None
    for t0 in starts:
        res = minimize(psse, t0, method='Nelder-Mead',
                       options={'maxiter': 20000, 'xatol': 1e-8})
        if res.fun < best_sse:
            best_sse = res.fun
            best_res = res

    t1, t2 = max(best_res.x[0], 0.01), max(best_res.x[1], 0.01)
    x_pts = (np.arange(K) + 0.5) / K
    raw = beta_dist.pdf(1 - x_pts, t1, t2)
    s = raw.sum()
    w = raw / s if s > 1e-12 else np.ones(K) / K
    xw = X @ w
    xc = xw - xw.mean()
    yc = Y - Y.mean()
    beta_ = np.dot(yc, xc) / np.dot(xc, xc)
    alpha_ = Y.mean() - beta_ * xw.mean()
    resid = Y - alpha_ - beta_ * xw

    return {
        'theta1': t1, 'theta2': t2, 'alpha': alpha_, 'beta': beta_,
        'sse': best_sse, 'weights': w, 'residuals': resid, 'xw': xw
    }
```

</div>
</div>

---

## Combining Horizons: The Nowcast Panel

For a complete nowcasting exercise, estimate one model per horizon ($h = 0, 1, 2$ missing months) and report the resulting RMSE by horizon:


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def nowcast_panel(Y_train, Y_test, X_train_dict, X_test_dict):
    """
    Estimate direct MIDAS for each horizon and return forecasts.

    Parameters
    ----------
    X_train_dict : dict of int -> np.ndarray
        Keys are h_missing (0, 1, 2), values are training MIDAS matrices.
    X_test_dict : dict of int -> np.ndarray
        Keys are h_missing, values are test observation vectors (shape K-h,).

    Returns
    -------
    nowcasts : dict of int -> float
    """
    nowcasts = {}
    for h, X_train in X_train_dict.items():
        est = estimate_midas_direct(Y_train, X_train)
        x_test = X_test_dict[h]
        xw_test = float(x_test @ est['weights'])
        nowcasts[h] = est['alpha'] + est['beta'] * xw_test
    return nowcasts
```


---

## MIDAS-AR vs. MIDAS: A Comparison

For quarterly GDP with moderate autocorrelation ($\hat{\rho} \approx 0.2$–$0.4$):

| Model | k | OOS RMSE | Ljung-Box p-value |
|-------|---|----------|------------------|
| Beta MIDAS | 4 | baseline | ~0.15 |
| Beta MIDAS-AR(1) | 5 | baseline - 3% | ~0.42 |

The MIDAS-AR model improves OOS RMSE when:
- GDP growth is autocorrelated (as it typically is)
- The Ljung-Box test on plain MIDAS residuals rejects (p < 0.10)
- BIC(MIDAS-AR) < BIC(MIDAS)

Always check all three conditions before adding AR terms.

---

## Common Pitfalls

**Pitfall 1: Inconsistent lag alignment across horizons.** When building the $h=1$ and $h=2$ horizon matrices, the starting lag must shift by the appropriate number of months. A 1-month nowcast uses lags $j=1,...,K-1$; a 2-month nowcast uses lags $j=2,...,K-1$ (or $j=0,...,K-1$ if using the full-quarter alignment).

**Pitfall 2: Re-estimating weights for each horizon.** The theta parameters should be re-estimated for each horizon, not reused from the complete-quarter model. The optimal weight shape can differ by horizon.

**Pitfall 3: Overstating the benefit of MIDAS-AR at 1-step.** Adding AR(1) to MIDAS helps at 2–4 quarter horizons but typically adds little at the 1-step nowcast level where the high-frequency data already captures the current quarter's dynamics.

---

## Connections

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.



- **Builds on:** Guide 01 (nowcasting problem, ragged edge)
- **Leads to:** Notebook 01 (gdp_nowcast.ipynb), Notebook 02 (ragged_edge_simulation.ipynb)
- **Related to:** Marcellino-Stock-Watson (2006) direct vs. iterated; Ghysels (2016) MIDAS survey

---



## Practice Problems

1. Write out the MIDAS-AR(1) profile SSE function for a model with AR lag $\rho$ as a free parameter. Show that the profile can still be solved by a 2D optimization over $(\theta_1, \theta_2)$ with $(α, \rho, \beta)$ solved by OLS.

2. For a 1-month nowcast (h_missing=2), the MIDAS weight function is estimated over $K-2$ lags instead of $K$ lags. If the true weight function is $\text{Beta}(1.5, 4.0)$ with K=12, what is the weight on lag j=0 in the truncated model relative to the full model?

3. Describe the expanding-window cross-validation protocol for a direct MIDAS model with h_missing=1. How does it differ from the standard expanding-window protocol for the complete-quarter model?


---

## Cross-References

<a class="link-card" href="./01_nowcasting_problem_guide.md">
  <div class="link-card-title">01 Nowcasting Problem</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_nowcasting_problem_slides.md">
  <div class="link-card-title">01 Nowcasting Problem — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


# Model Selection for MIDAS

> **Reading time:** ~15 min | **Module:** 02 — Estimation Inference | **Prerequisites:** Module 1


## In Brief

<div class="flow">
<div class="flow-step mint">1. Specify Candidates</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">2. Estimate Each</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">3. Compare IC / OOSF</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Select Best</div>
</div>


<div class="callout-key">

**Key Concept Summary:** MIDAS model selection involves three decisions: (1) how many high-frequency lags to include ($K$ or equivalently $P$ quarterly lags), (2) which weight function family (Beta, Almon, step, U-MIDAS), and

</div>

MIDAS model selection involves three decisions: (1) how many high-frequency lags to include ($K$ or equivalently $P$ quarterly lags), (2) which weight function family (Beta, Almon, step, U-MIDAS), and (3) whether to include AR terms. Information criteria (AIC/BIC) and expanding-window cross-validation are the primary tools.

## Key Insight

<div class="callout-insight">

**Insight:** The mixed-frequency approach preserves within-period dynamics that aggregation destroys. This is especially valuable when the timing of high-frequency movements carries economic information.

</div>


Information criteria balance fit and parsimony. For MIDAS, BIC typically selects more parsimonious models than AIC because $\ln T$ is large relative to 2 for typical macro sample sizes ($T \approx 80$–$200$, $\ln T \approx 4.4$–$5.3$). Cross-validation via an expanding window provides direct out-of-sample evidence that avoids in-sample overfitting entirely.

---

## Decision 1: Lag Order $K$ (or $P$)

<div class="callout-warning">

**Warning:** Be cautious about extrapolating MIDAS performance from stable periods to crisis periods. The relationship between high-frequency indicators and the low-frequency target can shift dramatically during regime changes.

</div>


### Why Lag Order Matters

The number of high-frequency lags $K = P \times m$ determines:
- **Fit**: More lags can always improve in-sample R²
- **Parsimony**: More lags require estimating more implied coefficients
- **Interpretation**: Too many lags obscure which time period matters

For quarterly-monthly MIDAS, the economically motivated range is $P = 1$ to $P = 6$ quarterly lags ($K = 3$ to $K = 18$ monthly lags). Beyond $P = 6$, the economic motivation weakens: does IP from 6 quarters ago really predict current GDP?

### AIC/BIC for Lag Selection

$$\text{AIC}(K) = T \ln\!\left(\frac{\text{SSE}(K)}{T}\right) + 2 k(K)$$

$$\text{BIC}(K) = T \ln\!\left(\frac{\text{SSE}(K)}{T}\right) + k(K) \ln T$$

where $k(K) = 4$ for restricted Beta MIDAS (regardless of $K$!) or $k(K) = K + 1$ for U-MIDAS.

**Critical observation:** For restricted MIDAS, $k$ does not grow with $K$! The SSE decreases as $K$ grows (more lags can fit better), but the penalty stays constant at 4 parameters. This means BIC will select the largest $K$ that provides meaningful SSE reduction.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def midas_aic_bic(Y, X_list, weight_fn, n_params_list):
    """
    Compute AIC and BIC for a sequence of MIDAS models with different K.

    Parameters
    ----------
    Y : np.ndarray
        Dependent variable (length T).
    X_list : list of np.ndarray
        List of MIDAS matrices for each lag specification.
    weight_fn : callable
    n_params_list : list of int
        Number of free parameters for each specification.

    Returns
    -------
    results : list of dict
    """
    T = len(Y)
    results = []

    for X, k in zip(X_list, n_params_list):
        from scipy.optimize import minimize
        from scipy.stats import beta as beta_dist

        def beta_weights(K, t1, t2):
            if t1 <= 0 or t2 <= 0: return np.ones(K)/K
            x = (np.arange(K) + 0.5) / K
            raw = beta_dist.pdf(1 - x, t1, t2)
            s = raw.sum()
            return raw/s if s > 1e-12 else np.ones(K)/K

        def profile_sse(theta, Y, X):
            t1, t2 = theta
            if t1 <= 0 or t2 <= 0: return 1e10
            w = beta_weights(X.shape[1], t1, t2)
            xw = X @ w
            xc = xw - xw.mean()
            yc = Y - Y.mean()
            ss = np.dot(xc, xc)
            if ss < 1e-12: return np.sum((Y - Y.mean())**2)
            b = np.dot(yc, xc) / ss
            a = Y.mean() - b * xw.mean()
            return np.sum((Y - a - b * xw)**2)

        best_sse = np.inf
        for t1, t2 in [(1.0, 1.0), (1.0, 5.0), (1.5, 4.0), (2.0, 2.0)]:
            r = minimize(profile_sse, [t1, t2], args=(Y, X), method='Nelder-Mead',
                         options={'maxiter': 10000, 'xatol': 1e-7})
            if r.fun < best_sse:
                best_sse = r.fun

        aic = T * np.log(best_sse / T) + 2 * k
        bic = T * np.log(best_sse / T) + k * np.log(T)

        results.append({
            'K': X.shape[1],
            'n_params': k,
            'sse': best_sse,
            'aic': aic,
            'bic': bic,
            'r2': 1 - best_sse / np.sum((Y - Y.mean())**2)
        })

    return results
```

</div>

### Practical Lag Selection

A common empirical finding in quarterly GDP nowcasting:
- $P = 1$ (K=3): captures within-current-quarter variation
- $P = 2$ (K=6): adds lead-lag from previous quarter — typically improves BIC
- $P = 4$ (K=12): standard macro default
- $P > 6$ (K>18): rarely improves BIC; may overfit

**Rule of thumb:** Start with $P=4$ as the default. Check whether $P=2$ or $P=6$ improves BIC.

---

## Decision 2: Weight Function Family

Given the data, we want to know: does the polynomial restriction matter?

### Formal Test: Restricted vs. Unrestricted

For $K \leq 9$ and $T \geq 80$, U-MIDAS is feasible. Compare:

| Model | $k$ | SSE | BIC |
|-------|-----|-----|-----|
| OLS-aggregate | 2 | $\text{SSE}_{OLS}$ | $\text{BIC}_{OLS}$ |
| Beta MIDAS | 4 | $\text{SSE}_{Beta}$ | $\text{BIC}_{Beta}$ |
| Almon MIDAS | 4 | $\text{SSE}_{Almon}$ | $\text{BIC}_{Almon}$ |
| U-MIDAS | $K+1$ | $\text{SSE}_{U}$ | $\text{BIC}_{U}$ |

Select by minimum BIC. If BIC favors OLS-aggregate over all MIDAS variants, the equal-weight restriction is not rejected by the data.

---

## Decision 3: AR Terms

The MIDAS-AR model adds quarterly lags of $y_t$:

$$y_t = \alpha + \sum_{p=1}^{P_y} \rho_p y_{t-p} + \beta \sum_{j=0}^{K-1} w_j(\theta) x_{mt-j} + \varepsilon_t$$

**When to include AR terms:**

1. Test for serial correlation in MIDAS residuals (Ljung-Box test)
2. If $LB(p) < 0.10$ for small $p$, add AR terms
3. Compare MIDAS vs. MIDAS-AR by BIC

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from scipy.stats import chi2

def ljung_box(residuals, n_lags=4):
    """
    Ljung-Box test for serial correlation.

    H0: No autocorrelation in residuals up to lag n_lags.
    Reject H0 → add AR terms to model.
    """
    T = len(residuals)
    acf = np.array([
        np.corrcoef(residuals[k:], residuals[:-k])[0,1]
        for k in range(1, n_lags + 1)
    ])

    Q = T * (T + 2) * np.sum(acf**2 / (T - np.arange(1, n_lags + 1)))
    p_value = 1 - chi2.cdf(Q, df=n_lags)

    print(f"Ljung-Box Q({n_lags}) = {Q:.3f}, p-value = {p_value:.4f}")
    if p_value < 0.10:
        print(f"  Reject H0: evidence of serial correlation. Add AR terms.")
    else:
        print(f"  Fail to reject H0: no significant serial correlation.")

    return Q, p_value
```

</div>

---

## Expanding Window Cross-Validation

The most reliable model selection method for forecasting: evaluate out-of-sample performance using the actual time ordering.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
def expanding_window_cv(Y, X_dict, weight_fn_dict, min_train=30, horizon=1):
    """
    Expanding window cross-validation for MIDAS model comparison.

    Parameters
    ----------
    Y : np.ndarray
    X_dict : dict of str -> np.ndarray
        Maps model name to MIDAS regressor matrix.
    weight_fn_dict : dict of str -> callable
        Maps model name to weight function.
    min_train : int
        Minimum training sample size.
    horizon : int
        Forecast horizon (1 = one-step-ahead).

    Returns
    -------
    rmse_dict : dict of str -> float
        Root mean squared error for each model.
    """
    from scipy.optimize import minimize
    from sklearn.linear_model import LinearRegression
    from scipy.stats import beta as beta_dist

    T = len(Y)
    n_oos = T - min_train - (horizon - 1)

    errors = {name: [] for name in X_dict}

    for end in range(min_train, T - horizon + 1):
        Y_train = Y[:end]
        Y_test = Y[end + horizon - 1]

        for name, X in X_dict.items():
            X_train = X[:end]
            X_test = X[end + horizon - 1]
            weight_fn = weight_fn_dict.get(name)

            if weight_fn is None:
                # U-MIDAS: OLS on all lags
                model = LinearRegression().fit(X_train, Y_train)
                y_hat = model.predict(X_test.reshape(1, -1))[0]
            else:
                # Restricted MIDAS: profile NLS
                def psse(theta, Y, X):
                    t1, t2 = theta
                    if t1 <= 0 or t2 <= 0: return 1e10
                    w = weight_fn(X.shape[1], t1, t2)
                    xw = X @ w
                    xc = xw - xw.mean(); yc = Y - Y.mean()
                    ss = np.dot(xc, xc)
                    if ss < 1e-12: return np.sum((Y - Y.mean())**2)
                    b = np.dot(yc, xc) / ss
                    a = Y.mean() - b * xw.mean()
                    return np.sum((Y - a - b * xw)**2)

                best = None; best_sse = np.inf
                for t1, t2 in [(1.0, 5.0), (1.5, 4.0)]:
                    r = minimize(psse, [t1, t2], args=(Y_train, X_train),
                                 method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-6})
                    if r.fun < best_sse:
                        best_sse = r.fun; best = r

                t1_h, t2_h = best.x
                t1_h, t2_h = max(t1_h, 0.01), max(t2_h, 0.01)
                w_h = weight_fn(X_train.shape[1], t1_h, t2_h)
                xw_h = X_train @ w_h
                lr = LinearRegression().fit(xw_h.reshape(-1,1), Y_train)
                y_hat = lr.predict([(X_test @ w_h)])[0]

            errors[name].append((Y_test - y_hat)**2)

    rmse_dict = {name: np.sqrt(np.mean(errs)) for name, errs in errors.items()}
    return rmse_dict
```

</div>

---

## Model Selection Summary Table

After running AIC/BIC and expanding-window CV, summarize results:

```
Model Selection Results: GDP ~ IP, 2000Q1–2024Q4

Information Criteria (K=9, T=100):
  Model              k   SSE     AIC    BIC    OOS-RMSE
  OLS-aggregate      2   52.1   134.2  141.4   1.451
  Beta MIDAS (K=9)   4   48.3   133.0  147.4   1.389  ← BIC winner
  Almon MIDAS (K=9)  4   48.1   132.9  147.3   1.391
  U-MIDAS (K=9)     10   45.9   133.4  163.0   1.412
  Beta MIDAS (K=12)  4   47.8   132.7  147.1   1.384  ← OOS winner

Recommendation: Beta MIDAS with K=12 lags (P=4 quarterly lags).
```

---

## Common Pitfalls

**Pitfall 1: Selecting by in-sample R² alone.** U-MIDAS always wins in-sample. Use AIC/BIC or OOS evaluation.

**Pitfall 2: Not accounting for COVID.** The 2020Q1 and Q2 observations are extreme outliers. Run model selection both including and excluding COVID quarters and note if the selection changes.

**Pitfall 3: Ignoring the estimation uncertainty in lag selection.** Each lag order requires re-estimating theta. Report the lag selection process, not just the final model.

---

## Connections

<div class="callout-danger">

**Danger:** Never use future information when constructing the high-frequency regressor matrix. In a real-time nowcasting context, you only have data up to the current date -- using the full quarter of monthly data when nowcasting mid-quarter is a look-ahead bias that invalidates your results.

</div>


- **Builds on:** Guide 01 (NLS estimation)
- **Leads to:** Guide 03 (robust inference), Module 03 (nowcasting with selected model)
- **Related to:** Information criteria in time series (Lütkepohl 2006), forecast evaluation

---

## Practice Problems

1. You have $T = 80$ observations and are choosing between $P = 2$ (K=6, BIC penalty = $4 \times \ln 80 \approx 17.6$) and $P = 4$ (K=12, same 4 parameters, same BIC penalty). How much must SSE decrease from $P=2$ to $P=4$ to justify the extra lags under BIC?

2. Derive the F-test for whether the last quarterly lag (lags $K-3$, $K-2$, $K-1$) can be dropped. Under restricted MIDAS, how would you implement this test?

3. The Diebold-Mariano test compares two forecasting models' mean squared errors out-of-sample. Write out the DM statistic for comparing Beta MIDAS and OLS-aggregate nowcasts.


---

## Cross-References

<a class="link-card" href="./01_nls_estimation_guide.md">
  <div class="link-card-title">01 Nls Estimation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./01_nls_estimation_slides.md">
  <div class="link-card-title">01 Nls Estimation — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>

<a class="link-card" href="./03_inference_guide.md">
  <div class="link-card-title">03 Inference</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_inference_slides.md">
  <div class="link-card-title">03 Inference — Companion Slides</div>
  <div class="link-card-description">Slide deck covering the key points.</div>
</a>


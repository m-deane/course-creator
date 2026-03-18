# The MIDAS Equation

## In Brief

MIDAS (Mixed Data Sampling) regression directly incorporates high-frequency regressors into a low-frequency model by applying a parameterized lag polynomial that weights each high-frequency observation. The weights are estimated from the data rather than imposed, recovering information that temporal aggregation discards.

## Key Insight

The MIDAS model replaces a fixed aggregation scheme with an estimated one. By parameterizing the weight function with just 2–3 parameters (instead of estimating each weight individually), MIDAS achieves efficient estimation even when the number of high-frequency lags is large.

---

## Notation

Let the low-frequency variable be observed at periods $t = 1, 2, \ldots, T$ (quarters) and the high-frequency variable at periods $\tau = 1, 2, \ldots, mT$ (months), where $m$ is the frequency ratio ($m = 3$ for monthly-to-quarterly).

We use the high-frequency lag operator $L^{1/m}$ defined by:
$$L^{1/m} x_\tau = x_{\tau-1}$$

So $\left(L^{1/m}\right)^j x_\tau = x_{\tau-j}$ shifts $j$ high-frequency periods back.

For the quarterly-monthly case, if $\tau$ is the last month of quarter $t$:
$$\tau = mt$$

Then:
- $x_\tau = x_{mt}$ = last month of quarter $t$
- $x_{\tau-1} = x_{mt-1}$ = second-to-last month of quarter $t$
- $x_{\tau-3} = x_{mt-3}$ = last month of quarter $t-1$

---

## The Basic MIDAS Model

$$y_t = \alpha + \beta \cdot B\!\left(L^{1/m}; \theta\right) x_\tau + \varepsilon_t$$

where:

$$B\!\left(L^{1/m}; \theta\right) x_\tau = \sum_{j=0}^{K-1} w_j(\theta) \cdot x_{\tau - j}$$

The weight function $B(\cdot;\theta)$ maps $K$ high-frequency observations to a scalar. The weights $w_j(\theta)$ depend on a small parameter vector $\theta$ with dimension 2 or 3 (much smaller than $K$).

### The Full Parameterization

A MIDAS model with $P$ quarterly lags and $m$ months per quarter uses $K = P \cdot m$ high-frequency lags:

$$y_t = \alpha + \beta \sum_{j=0}^{Pm-1} w_j(\theta) \cdot x_{mt-j} + \varepsilon_t$$

**Example:** Quarterly GDP on monthly IP, 4 quarterly lags:
- $m = 3$, $P = 4$, $K = 12$ high-frequency lags
- $w_j(\theta)$ for $j = 0, 1, \ldots, 11$ (normalized to sum to 1)
- $\theta \in \mathbb{R}^2$ for a Beta polynomial — only 3 parameters total ($\alpha, \beta, \theta$)

Compare to unrestricted MIDAS: would need to estimate 12 weights plus $\alpha$ and $\beta$ = 14 parameters from ~100 quarterly observations. With the polynomial parameterization, we estimate only 4 parameters.

---

## The Normalization Convention

The weights $w_j(\theta)$ are normalized to sum to 1:

$$\sum_{j=0}^{K-1} w_j(\theta) = 1$$

This identifies the scaling: $\beta$ captures the overall magnitude of the effect, while $\{w_j\}$ captures the shape of the lag distribution. Without normalization, $\beta$ and $w_j$ would not be separately identified.

The regression coefficient $\beta$ has the interpretation: **a unit increase in the weighted average of the past $K$ high-frequency observations is associated with a $\beta$-unit increase in $y_t$.**

---

## Multi-Predictor MIDAS

The model extends naturally to multiple high-frequency predictors:

$$y_t = \alpha + \sum_{r=1}^{R} \beta_r \cdot B_r\!\left(L^{1/m}; \theta_r\right) x_{r,\tau} + \varepsilon_t$$

Each predictor $r$ has its own weight function $B_r$ and parameters $\theta_r$. In a nowcasting application, $R$ might be 5–10 monthly indicators plus a few daily series.

---

## Adding Autoregressive Terms

Quarterly lags of $y_t$ can be included alongside the high-frequency terms:

$$y_t = \alpha + \sum_{p=1}^{P_y} \rho_p y_{t-p} + \beta \cdot B\!\left(L^{1/m}; \theta\right) x_\tau + \varepsilon_t$$

This "MIDAS-AR" specification is common in macroeconomic applications where $y_t$ is serially correlated (GDP growth is mildly persistent).

---

## Derivation from First Principles

### The Temporal Aggregation Benchmark

A standard quarterly regression uses temporally aggregated $\bar{x}_t$:

$$y_t = \alpha + \beta \bar{x}_t + \varepsilon_t, \quad \bar{x}_t = \frac{1}{m}\sum_{j=0}^{m-1} x_{mt-j}$$

This is a MIDAS model with $K = m$ and fixed equal weights $w_j = 1/m$ for all $j$.

### The MIDAS Generalization

MIDAS asks: what if the optimal weights are not equal? If the true model is:

$$y_t = \alpha + \beta^* \sum_{j=0}^{K-1} w_j^* \cdot x_{mt-j} + \varepsilon_t$$

then OLS on the equal-weight aggregate estimates:

$$\hat{\beta}^{\text{OLS}} \xrightarrow{p} \frac{\text{Cov}\!\left(\bar{x}_t, y_t\right)}{\text{Var}(\bar{x}_t)} = \beta^* \cdot \frac{\sum_{j=0}^{m-1} w_j^* \cdot \text{Cov}(x_{mt-j}, \bar{x}_t)}{\text{Var}(\bar{x}_t)}$$

This is a weighted average of the true lag-specific coefficients $\{\beta^* w_j^*\}$, not the individual terms. MIDAS recovers the full structure.

---

## Worked Example: GDP on Industrial Production

We estimate the model:

$$\text{GDP growth}_t = \alpha + \beta \sum_{j=0}^{8} w_j(\theta) \cdot \text{IP growth}_{3t-j} + \varepsilon_t$$

with $K = 9$ (3 quarterly lags × 3 months) and Beta polynomial weights.

```python
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import beta as beta_dist

def beta_weights(n_lags, theta1, theta2):
    """
    Beta polynomial weights for MIDAS.

    Maps K lags to weights via the Beta distribution PDF evaluated
    at equally-spaced points on [0, 1].

    Parameters
    ----------
    n_lags : int
        Number of high-frequency lags K.
    theta1 : float
        First shape parameter of Beta distribution (> 0).
    theta2 : float
        Second shape parameter of Beta distribution (> 0).

    Returns
    -------
    weights : np.ndarray, shape (K,)
        Normalized weights summing to 1. weights[0] is most recent.
    """
    # Evaluation points: from most recent (near 1) to oldest (near 0)
    # Convention: lag 0 = most recent, so we reverse the beta PDF
    x = np.linspace(1 / (2 * n_lags), 1 - 1 / (2 * n_lags), n_lags)
    x_reversed = 1 - x  # Flip so lag 0 (x=1) gets the highest Beta(theta1, theta2) value

    # Evaluate Beta PDF
    raw_weights = beta_dist.pdf(x_reversed, theta1, theta2)

    # Normalize to sum to 1
    weights = raw_weights / raw_weights.sum()
    return weights


def midas_objective(params, Y, X):
    """
    Negative log-likelihood (equivalently, sum of squared residuals)
    for MIDAS with Beta polynomial weights.

    Parameters
    ----------
    params : np.ndarray
        [alpha, beta, theta1, theta2] — 4 parameters.
    Y : np.ndarray, shape (T,)
        Quarterly dependent variable.
    X : np.ndarray, shape (T, K)
        MIDAS regressor matrix (K high-frequency lags).

    Returns
    -------
    sse : float
        Sum of squared errors.
    """
    alpha, beta_coef, theta1, theta2 = params
    K = X.shape[1]

    # Compute weights
    if theta1 <= 0 or theta2 <= 0:
        return 1e10  # Penalize invalid parameters

    weights = beta_weights(K, theta1, theta2)

    # Weighted sum of high-frequency lags
    x_weighted = X @ weights  # Shape: (T,)

    # Compute residuals
    y_hat = alpha + beta_coef * x_weighted
    residuals = Y - y_hat

    return np.sum(residuals ** 2)


def estimate_midas(Y, X, theta0=(1.0, 5.0)):
    """
    Estimate MIDAS model by NLS with Beta polynomial weights.

    Returns
    -------
    result : dict with keys: alpha, beta, theta1, theta2, weights, fitted, residuals, r2
    """
    K = X.shape[1]

    # Initial OLS estimate as starting values
    x_flat = X.mean(axis=1)
    beta_init = np.cov(Y, x_flat)[0, 1] / np.var(x_flat)
    alpha_init = Y.mean() - beta_init * x_flat.mean()

    params0 = np.array([alpha_init, beta_init, theta0[0], theta0[1]])

    result = minimize(
        midas_objective,
        params0,
        args=(Y, X),
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-6}
    )

    alpha, beta_coef, theta1, theta2 = result.x
    weights = beta_weights(K, theta1, theta2)
    x_weighted = X @ weights
    fitted = alpha + beta_coef * x_weighted
    residuals = Y - fitted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - Y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        'alpha': alpha,
        'beta': beta_coef,
        'theta1': theta1,
        'theta2': theta2,
        'weights': weights,
        'fitted': fitted,
        'residuals': residuals,
        'r2': r2,
        'sse': result.fun,
        'converged': result.success
    }
```

**Key implementation notes:**

1. The `beta_weights` function evaluates the Beta PDF at $K$ equally-spaced points and normalizes. The endpoints are set to $1/(2K)$ and $1 - 1/(2K)$ to avoid boundary singularities in $\text{Beta}(\theta_1, \theta_2)$ when $\theta_i < 1$.

2. The optimization objective is the sum of squared errors (SSE). NLS minimizes SSE by finding the parameter vector $(\alpha, \beta, \theta_1, \theta_2)$ that produces residuals closest to zero.

3. Starting values matter for NLS convergence. Using OLS on the equal-weight aggregate as an initial guess for $(\alpha, \beta)$ and $\theta = (1, 5)$ (a declining Beta shape) typically gives good convergence.

---

## Interpreting the Estimated Weights

After estimation, plot the weights $w_j(\hat{\theta})$:

```python
import matplotlib.pyplot as plt

def plot_midas_weights(weights, freq_ratio=3, title="Estimated MIDAS Weights"):
    """
    Visualize MIDAS weight function.

    Parameters
    ----------
    weights : np.ndarray
        Normalized weights (lag 0 = most recent).
    freq_ratio : int
        High-to-low frequency ratio (for x-axis labeling).
    title : str
        Plot title.
    """
    K = len(weights)
    n_quarters = K // freq_ratio + (1 if K % freq_ratio > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 4))
    lags = np.arange(K)

    ax.bar(lags, weights, color='steelblue', alpha=0.8)
    ax.axhline(1/K, color='red', linestyle='--', linewidth=1.5,
               label=f'Equal weights (1/{K} = {1/K:.3f})')

    # Add quarter boundary lines
    for q in range(1, n_quarters):
        ax.axvline(q * freq_ratio - 0.5, color='gray', linestyle=':', alpha=0.7)
        ax.text(q * freq_ratio - freq_ratio/2 - 0.5, max(weights) * 1.05,
                f'Q-{q-1 if q > 1 else "curr"}', ha='center', fontsize=9, color='gray')

    ax.set_xlabel('High-frequency lag (0 = most recent)')
    ax.set_ylabel('Weight')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
```

A declining weight pattern (high weight on lag 0, lower on older lags) indicates the most recent month of the quarter is most informative. A hump-shaped pattern would suggest an intermediate lag matters most.

---

## The MIDAS vs. OLS Comparison

The efficiency gain from MIDAS versus OLS on the aggregate is:

$$\text{Gain} = R^2_{\text{MIDAS}} - R^2_{\text{OLS-aggregate}}$$

In empirical studies of GDP nowcasting:
- Monthly IP to quarterly GDP: MIDAS gains ~0.05–0.10 R² units
- Daily returns to quarterly stock returns: MIDAS gains ~0.15–0.20 R² units (Ghysels, Santa-Clara, Valkanov 2006)

The gain is larger when:
1. The frequency ratio $m$ is large (more within-period information available)
2. The true weight function is concentrated on recent lags (strong recency effect)
3. The sample size is large enough for NLS to precisely estimate $\theta$

---

## Common Pitfalls

**Pitfall 1: Not normalizing weights.** If weights don't sum to 1, the coefficient $\beta$ has no clear interpretation and optimization is ill-conditioned.

**Pitfall 2: Using too many or too few lags.** Too few lags (e.g., only $m=3$ for one quarter) misses lead-lag dynamics. Too many lags (e.g., $P=8$ quarterly lags = 24 monthly) may not converge well or overfit. A practical default is $P=4$ quarterly lags.

**Pitfall 3: Poor starting values.** NLS is sensitive to initialization. Always use OLS on the aggregate as a starting point; also try multiple starting values to check for local optima.

**Pitfall 4: Not checking convergence.** `minimize` returns a success flag — always check it. If NLS fails to converge, try alternative optimizers (L-BFGS-B, COBYLA) or different starting values.

---

## Connections

- **Builds on:** Module 00 (MIDAS data matrix), OLS regression
- **Leads to:** Guide 02 (weight functions), Guide 03 (U-MIDAS), Module 02 (NLS estimation)
- **Related to:** Distributed lag models, rational distributed lag (RDL), almon polynomial lag

---

## Practice Problems

1. Show that the standard OLS regression of $y_t$ on $\bar{x}_t = (1/m)\sum_j x_{mt-j}$ is a special case of MIDAS with fixed weights $w_j = 1/m$.

2. In the Beta polynomial MIDAS model with $\theta_1 = 1, \theta_2 = 1$, what does the weight function look like? (Hint: $\text{Beta}(1,1)$ is the uniform distribution.) How does this compare to equal-weight aggregation?

3. Suppose the true model has $w_0 = 0.8$ (most recent month has 80% weight) and $w_1 = w_2 = 0.1$. An equal-weight OLS regression uses $w = (1/3, 1/3, 1/3)$. Write out the probability limit of the OLS estimator in terms of the true parameters and the covariance structure of $(x_{mt}, x_{mt-1}, x_{mt-2})$.

---

## Further Reading

- Ghysels, E., Santa-Clara, P., & Valkanov, R. (2006). "Predicting volatility: Getting the most out of return data sampled at different frequencies." *Journal of Econometrics*, 131(1–2), 59–95.
- Ghysels, E., & Qian, H. (2019). "Estimating MIDAS regressions via OLS with polynomial parameter profiling." *Econometrics and Statistics*, 9, 1–16.
- Ghysels, E., Sinko, A., & Valkanov, R. (2007). "MIDAS regressions: Further results and new directions." *Econometric Reviews*, 26(1), 53–90.

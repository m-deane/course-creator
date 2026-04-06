# Time Series Basics for Factor Models

> **Reading time:** ~8 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Dynamic factor models extend static factors by allowing factors to evolve over time following autoregressive dynamics. This guide reviews the time series concepts—stationarity, autocovariance, and AR processes—essential for understanding factor dynamics.

</div>

## In Brief

Dynamic factor models extend static factors by allowing factors to evolve over time following autoregressive dynamics. This guide reviews the time series concepts—stationarity, autocovariance, and AR processes—essential for understanding factor dynamics.

<div class="callout-insight">

**Insight:** Stationarity ensures that statistical properties remain constant over time, which is necessary for consistent estimation. Factor models assume factors follow stationary (or trend-stationary) processes, allowing us to estimate dynamics from historical data.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Stationarity

### Intuitive Explanation

Imagine taking snapshots of your time series at different points. For a stationary series:
- Each snapshot has the same "shape" (distribution)
- The relationship between observations $h$ periods apart is always the same
- There's no trend, no changing volatility, no structural breaks

### Formal Definition

A time series $\{y_t\}$ is **strictly stationary** if the joint distribution of $(y_{t_1}, ..., y_{t_k})$ equals that of $(y_{t_1+h}, ..., y_{t_k+h})$ for all $h$ and all choices of time indices.

A time series is **weakly (covariance) stationary** if:
1. $E[y_t] = \mu$ (constant mean)
2. $\text{Var}(y_t) = \sigma^2 < \infty$ (constant, finite variance)
3. $\text{Cov}(y_t, y_{t-h}) = \gamma(h)$ depends only on lag $h$, not on $t$

### Why Stationarity Matters for Factor Models

<div class="flow">
<div class="flow-step mint">1. Consistent estimatio...</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Forecasting:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Factor dynamics:</div>

</div>


1. **Consistent estimation:** Sample moments converge to population moments
2. **Forecasting:** Past patterns are informative about future behavior
3. **Factor dynamics:** AR representation requires stationarity for stability

### Testing for Stationarity

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

def test_stationarity(y, significance=0.05):
    """
    Test stationarity using ADF and KPSS tests.

    ADF: H0 = unit root (non-stationary)
    KPSS: H0 = stationary

    Both rejecting or both failing to reject indicates clear conclusion.
    Mixed results suggest further investigation.
    """
    # ADF test
    adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(y, autolag='AIC')

    # KPSS test
    kpss_stat, kpss_pval, _, kpss_crit = kpss(y, regression='c', nlags='auto')

    results = {
        'ADF': {
            'statistic': adf_stat,
            'p_value': adf_pval,
            'reject_null': adf_pval < significance,
            'interpretation': 'Stationary' if adf_pval < significance else 'Non-stationary'
        },
        'KPSS': {
            'statistic': kpss_stat,
            'p_value': kpss_pval,
            'reject_null': kpss_pval < significance,
            'interpretation': 'Non-stationary' if kpss_pval < significance else 'Stationary'
        }
    }

    return results

# Example
np.random.seed(42)
y_stationary = np.random.randn(200)  # White noise (stationary)
y_random_walk = np.cumsum(np.random.randn(200))  # Random walk (non-stationary)

print("Stationary series:")
print(test_stationarity(y_stationary))
print("\nRandom walk:")
print(test_stationarity(y_random_walk))
```

---

## 2. Autocovariance and Autocorrelation

### Formal Definitions

**Autocovariance function** at lag $h$:
$$\gamma(h) = \text{Cov}(y_t, y_{t-h}) = E[(y_t - \mu)(y_{t-h} - \mu)]$$

**Autocorrelation function (ACF)** at lag $h$:
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\text{Cov}(y_t, y_{t-h})}{\text{Var}(y_t)}$$

Properties:
- $\gamma(0) = \text{Var}(y_t)$
- $\gamma(h) = \gamma(-h)$ (symmetry)
- $|\rho(h)| \leq 1$

### Intuitive Explanation

Autocovariance measures how much knowing today's value tells you about values $h$ periods ago (or ahead):
- High positive $\gamma(h)$: High values tend to follow high values
- Negative $\gamma(h)$: High values tend to follow low values
- Zero $\gamma(h)$: No linear relationship at lag $h$

### Sample Autocovariance


<span class="filename">sample_autocovariance.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def sample_autocovariance(y, max_lag=20):
    """
    Compute sample autocovariance function.

    Parameters
    ----------
    y : array-like
        Time series (T observations)
    max_lag : int
        Maximum lag to compute

    Returns
    -------
    gamma : ndarray
        Autocovariances for lags 0, 1, ..., max_lag
    """
    y = np.asarray(y)
    T = len(y)
    y_centered = y - y.mean()

    gamma = np.zeros(max_lag + 1)
    for h in range(max_lag + 1):
        gamma[h] = np.sum(y_centered[h:] * y_centered[:T-h]) / T

    return gamma

def sample_acf(y, max_lag=20):
    """Compute sample autocorrelation function."""
    gamma = sample_autocovariance(y, max_lag)
    return gamma / gamma[0]

# Example: AR(1) process
phi = 0.8
T = 500
np.random.seed(42)
y = np.zeros(T)
for t in range(1, T):
    y[t] = phi * y[t-1] + np.random.randn()

acf = sample_acf(y, max_lag=15)
print("Sample ACF:", acf[:6].round(3))
print("Theoretical ACF for AR(1):", [phi**h for h in range(6)])
```

</div>

### Visualizing ACF


<span class="filename">plot_acf_pacf.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def plot_acf_pacf(y, lags=20, title=""):
    """Plot ACF and PACF side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    plot_acf(y, lags=lags, ax=axes[0], title=f'ACF - {title}')
    plot_pacf(y, lags=lags, ax=axes[1], title=f'PACF - {title}')

    plt.tight_layout()
    return fig

# Confidence bands at ±1.96/√T indicate significance at 5% level
```

</div>

---

## 3. Autoregressive (AR) Processes

### AR(1) Process

$$y_t = c + \phi y_{t-1} + \varepsilon_t, \quad \varepsilon_t \sim WN(0, \sigma^2)$$

**Stationarity condition:** $|\phi| < 1$

**Properties (when stationary):**
- Mean: $E[y_t] = \mu = \frac{c}{1-\phi}$
- Variance: $\text{Var}(y_t) = \gamma(0) = \frac{\sigma^2}{1-\phi^2}$
- Autocovariance: $\gamma(h) = \phi^h \gamma(0)$
- ACF: $\rho(h) = \phi^h$ (geometric decay)

### AR(p) Process

$$y_t = c + \phi_1 y_{t-1} + \phi_2 y_{t-2} + ... + \phi_p y_{t-p} + \varepsilon_t$$

**Lag operator notation:** $(1 - \phi_1 L - \phi_2 L^2 - ... - \phi_p L^p)y_t = c + \varepsilon_t$

**Stationarity condition:** All roots of the characteristic polynomial $1 - \phi_1 z - ... - \phi_p z^p = 0$ lie outside the unit circle.

### Code Implementation


<span class="filename">simulate_ar.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def simulate_ar(phi, c=0, sigma=1, T=100, burn=100, seed=None):
    """
    Simulate AR(p) process.

    Parameters
    ----------
    phi : array-like
        AR coefficients [phi_1, phi_2, ..., phi_p]
    c : float
        Constant term
    sigma : float
        Innovation standard deviation
    T : int
        Length of series to return
    burn : int
        Burn-in period to discard
    seed : int, optional
        Random seed

    Returns
    -------
    y : ndarray
        Simulated AR(p) series
    """
    if seed is not None:
        np.random.seed(seed)

    phi = np.atleast_1d(phi)
    p = len(phi)

    # Check stationarity
    coeffs = np.r_[1, -phi]  # Polynomial coefficients
    roots = np.roots(coeffs)
    if np.any(np.abs(roots) <= 1):
        raise ValueError(f"Non-stationary: roots at {roots}")

    # Simulate
    total_T = T + burn
    y = np.zeros(total_T)
    eps = np.random.randn(total_T) * sigma

    for t in range(p, total_T):
        y[t] = c + np.dot(phi, y[t-p:t][::-1]) + eps[t]

    return y[burn:]

# Example: AR(2) process
phi = [0.5, 0.3]  # phi_1 = 0.5, phi_2 = 0.3
y_ar2 = simulate_ar(phi, T=200, seed=42)
```

</div>

### Estimation


<span class="filename">estimate_ar.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from statsmodels.tsa.ar_model import AutoReg

def estimate_ar(y, max_lag=10, criterion='aic'):
    """
    Estimate AR model with automatic lag selection.

    Parameters
    ----------
    y : array-like
        Time series
    max_lag : int
        Maximum lag to consider
    criterion : str
        'aic' or 'bic' for lag selection

    Returns
    -------
    results : dict
        Estimated parameters and diagnostics
    """
    # Fit models for each lag order
    results = {}
    for p in range(1, max_lag + 1):
        model = AutoReg(y, lags=p, old_names=False)
        fit = model.fit()
        results[p] = {
            'aic': fit.aic,
            'bic': fit.bic,
            'params': fit.params,
            'model': fit
        }

    # Select best by criterion
    if criterion == 'aic':
        best_p = min(results, key=lambda p: results[p]['aic'])
    else:
        best_p = min(results, key=lambda p: results[p]['bic'])

    return {
        'best_lag': best_p,
        'all_results': results,
        'best_model': results[best_p]['model']
    }

# Example
ar_results = estimate_ar(y_ar2, max_lag=5)
print(f"Selected lag order: {ar_results['best_lag']}")
print(f"Estimated coefficients: {ar_results['best_model'].params}")
```

</div>

---

## 4. Vector Autoregressions (VAR)

### Formal Definition

For an $n$-dimensional vector $Y_t = (y_{1t}, ..., y_{nt})'$:

$$Y_t = c + \Phi_1 Y_{t-1} + \Phi_2 Y_{t-2} + ... + \Phi_p Y_{t-p} + u_t$$

where $\Phi_i$ are $n \times n$ coefficient matrices and $u_t \sim (0, \Sigma_u)$.

### Why VAR Matters for Factor Models

Factor dynamics are modeled as VAR:

$$F_t = \Phi F_{t-1} + \eta_t$$

where $F_t$ is the $r \times 1$ vector of factors. Understanding VAR is essential for:
- Specifying factor dynamics
- Forecasting with factors
- Structural analysis (FAVAR)

### Code Implementation


<span class="filename">estimate_var.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from statsmodels.tsa.api import VAR

def estimate_var(Y, max_lag=10, criterion='aic'):
    """
    Estimate VAR model with lag selection.

    Parameters
    ----------
    Y : ndarray, shape (T, n)
        Multivariate time series
    max_lag : int
        Maximum lag to consider
    criterion : str
        'aic', 'bic', 'hqic', or 'fpe'

    Returns
    -------
    results : VARResults
        Fitted VAR model
    """
    model = VAR(Y)
    results = model.fit(maxlags=max_lag, ic=criterion)
    return results

# Example: Bivariate VAR
np.random.seed(42)
T = 200
n = 2

# True VAR(1) with Phi = [[0.5, 0.1], [0.2, 0.6]]
Phi_true = np.array([[0.5, 0.1], [0.2, 0.6]])
Y = np.zeros((T, n))
for t in range(1, T):
    Y[t] = Phi_true @ Y[t-1] + np.random.randn(n) * 0.5

var_results = estimate_var(Y, max_lag=5)
print(f"Selected lag order: {var_results.k_ar}")
print(f"Estimated Phi:\n{var_results.coefs[0].round(3)}")
```

</div>

---

## 5. State-Space Representation

### General Form

**Measurement equation:**
$$y_t = Z \alpha_t + d + \varepsilon_t, \quad \varepsilon_t \sim N(0, H)$$

**Transition equation:**
$$\alpha_t = T \alpha_{t-1} + c + R\eta_t, \quad \eta_t \sim N(0, Q)$$

where $\alpha_t$ is the unobserved state vector.

### AR(p) as State-Space

An AR(2) process $y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \varepsilon_t$ can be written:

$$\begin{bmatrix} y_t \\ y_{t-1} \end{bmatrix} = \begin{bmatrix} \phi_1 & \phi_2 \\ 1 & 0 \end{bmatrix} \begin{bmatrix} y_{t-1} \\ y_{t-2} \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} \varepsilon_t$$

### Why State-Space for Factor Models

Dynamic factor models have natural state-space form:
- **State:** Factors $F_t$
- **Measurement:** Observed variables $X_t = \Lambda F_t + e_t$
- **Transition:** Factor dynamics $F_t = \Phi F_{t-1} + \eta_t$

This enables:
- Kalman filter for factor estimation
- Maximum likelihood via prediction error decomposition
- Handling missing data naturally

---

## Common Pitfalls

### 1. Ignoring Non-Stationarity
- **Problem:** Spurious regression, inconsistent estimates
- **Solution:** Test for unit roots; difference or detrend as needed

### 2. Overfitting AR Lag Order
- **Problem:** Too many lags reduce forecast accuracy
- **Solution:** Use information criteria (AIC/BIC), not just significance tests

### 3. Confusing ACF and PACF
- **ACF:** Shows correlation at each lag (cumulative effect)
- **PACF:** Shows correlation after removing shorter-lag effects
- AR(p): PACF cuts off after lag $p$; ACF decays gradually

---

## Connections

- **Builds on:** Basic statistics, linear regression
- **Leads to:** Dynamic factor models, Kalman filter
- **Related to:** ARIMA models, spectral analysis

---

## Practice Problems

### Conceptual
1. Show that AR(1) with $|\phi| < 1$ is stationary by computing $E[y_t]$ and $\text{Var}(y_t)$.
2. Derive the ACF of an MA(1) process: $y_t = \varepsilon_t + \theta\varepsilon_{t-1}$.
3. Why does the PACF of an AR(p) process cut off after lag $p$?

### Implementation
4. Simulate 1000 observations from AR(1) with $\phi = 0.9$. Plot the ACF and compare to theoretical values.
5. Download monthly industrial production from FRED. Test for stationarity. If non-stationary, transform appropriately.
6. Estimate a VAR(1) for two related series. Compute and interpret the impulse response functions.

### Extension
7. Show that the spectral density of AR(1) is $f(\omega) = \frac{\sigma^2}{2\pi|1-\phi e^{-i\omega}|^2}$.

---

<div class="callout-insight">

**Insight:** Understanding time series basics for factor models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

- Hamilton, J.D. (1994). *Time Series Analysis*. Princeton. Chapters 2-3, 10-11.
- Lütkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Springer.
- Shumway, R.H. & Stoffer, D.S. (2017). *Time Series Analysis and Its Applications*. Springer.

---

## Conceptual Practice Questions

1. Why does stationarity matter for factor model estimation? What happens if the data is non-stationary?

2. How would you test whether a macroeconomic time series is stationary before including it in a factor model?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_time_series_basics_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_foundations_review.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_matrix_algebra_review.md">
  <div class="link-card-title">01 Matrix Algebra Review</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_pca_refresher.md">
  <div class="link-card-title">03 Pca Refresher</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


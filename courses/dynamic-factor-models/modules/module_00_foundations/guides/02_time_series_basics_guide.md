# Time Series Basics for Dynamic Factor Models

## In Brief

Dynamic factor models require factors to evolve over time via autoregressive dynamics. Understanding stationarity, autocovariance structure, and vector autoregressions is prerequisite knowledge — without it, you cannot interpret factor dynamics, test model stability, or understand why the Kalman filter works.

## Key Insight

A time series is stationary if its statistical properties do not change with time — no trend, no changing volatility, no structural breaks. Stationarity is what makes past patterns informative about the future, and it is the condition under which AR dynamics are stable and factor estimates are consistent.

## Formal Definition

**Weak (covariance) stationarity**: A stochastic process $\{y_t\}$ is weakly stationary if:
1. $E[y_t] = \mu$ for all $t$ (constant mean)
2. $\text{Var}(y_t) = \sigma^2 < \infty$ for all $t$ (finite, constant variance)
3. $\text{Cov}(y_t, y_{t-h}) = \gamma(h)$ depends only on lag $h$, not on $t$

**Autocovariance function** at lag $h$:
$$\gamma(h) = E[(y_t - \mu)(y_{t-h} - \mu)]$$

**Autocorrelation function (ACF)**:
$$\rho(h) = \frac{\gamma(h)}{\gamma(0)}$$

**AR($p$) process**:
$$(1 - \phi_1 L - \cdots - \phi_p L^p) y_t = c + \varepsilon_t, \quad \varepsilon_t \sim WN(0, \sigma^2)$$

Stationary iff all roots of $1 - \phi_1 z - \cdots - \phi_p z^p = 0$ lie strictly outside the unit circle.

**VAR($p$) process** for $n$-dimensional vector $Y_t$:
$$Y_t = c + \Phi_1 Y_{t-1} + \cdots + \Phi_p Y_{t-p} + u_t, \quad u_t \sim (0, \Sigma_u)$$

## Intuitive Explanation

Imagine photographing a river at different times. A stationary river always looks roughly the same — it meanders a bit but returns to its banks. A non-stationary river might be flooding, drying up, or permanently changing course. For factor models, we need stationary factors: if $F_t$ drifts off to infinity, the model breaks down.

The autocovariance function is a memory map: $\gamma(h)$ tells you how strongly today's value predicts a value $h$ periods away. For AR(1) with $\phi = 0.8$, knowing today's value gives you $0.8^h$ predictive power $h$ steps ahead. This geometric decay is what makes AR factors useful for forecasting.

The state-space representation makes the connection explicit: factor dynamics $F_t = \Phi F_{t-1} + \eta_t$ is a VAR(1) on the state. Stationarity of factors (all eigenvalues of $\Phi$ inside the unit circle) ensures the Kalman filter converges and the steady-state variance exists.

## Code Implementation

```python
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
import matplotlib.pyplot as plt

# --- 1. Testing for stationarity ---

def test_stationarity(y, significance=0.05, verbose=True):
    """
    Combined ADF + KPSS stationarity test.

    Strategy:
    - ADF: H0 = unit root (non-stationary). Reject → stationary.
    - KPSS: H0 = stationary. Reject → non-stationary.
    Using both avoids relying on a single test's power properties.

    Parameters
    ----------
    y : array-like, time series
    significance : float, significance level (default 0.05)

    Returns
    -------
    dict with test statistics, p-values, and interpretation
    """
    y = np.asarray(y)

    # ADF: null hypothesis is unit root (non-stationary)
    adf_stat, adf_pval, _, _, adf_crit, _ = adfuller(y, autolag='AIC')

    # KPSS: null hypothesis is stationary
    kpss_stat, kpss_pval, _, kpss_crit = kpss(y, regression='c', nlags='auto')

    # Interpret joint result
    adf_stationary = adf_pval < significance   # rejected non-stationarity
    kpss_stationary = kpss_pval >= significance # failed to reject stationarity

    if adf_stationary and kpss_stationary:
        conclusion = "Stationary"
    elif not adf_stationary and not kpss_stationary:
        conclusion = "Non-stationary (unit root)"
    elif adf_stationary and not kpss_stationary:
        conclusion = "Trend-stationary (investigate further)"
    else:
        conclusion = "Inconclusive (need more data)"

    results = {
        'ADF_stat': adf_stat, 'ADF_pval': adf_pval,
        'KPSS_stat': kpss_stat, 'KPSS_pval': kpss_pval,
        'conclusion': conclusion
    }

    if verbose:
        print(f"ADF  p-value: {adf_pval:.4f}  →  {'Stationary' if adf_stationary else 'Non-stationary'}")
        print(f"KPSS p-value: {kpss_pval:.4f}  →  {'Stationary' if kpss_stationary else 'Non-stationary'}")
        print(f"Conclusion:   {conclusion}")

    return results


# --- 2. Autocovariance and ACF from scratch ---

def sample_autocovariance(y, max_lag=20):
    """
    Compute sample autocovariance function.

    Uses the biased estimator (divide by T, not T-h) for consistency
    with the sample covariance matrix being positive semi-definite.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    y_centered = y - y.mean()

    gamma = np.zeros(max_lag + 1)
    for h in range(max_lag + 1):
        # Inner product of the series with its h-lagged version
        gamma[h] = (y_centered[h:] * y_centered[:T - h]).sum() / T

    return gamma


def sample_acf(y, max_lag=20):
    """Autocorrelation function: normalize autocovariance by variance."""
    gamma = sample_autocovariance(y, max_lag)
    return gamma / gamma[0]


# --- 3. AR(p) simulation with stationarity check ---

def simulate_ar(phi, c=0.0, sigma=1.0, T=500, burn=200, seed=None):
    """
    Simulate AR(p) process.

    Checks stationarity before simulating — raises if roots are
    inside the unit circle (explosive process).

    Parameters
    ----------
    phi : array-like, AR coefficients [phi_1, ..., phi_p]
    c : float, intercept
    sigma : float, innovation standard deviation
    T : int, desired series length (burn-in discarded)
    burn : int, burn-in periods to discard
    seed : int or None
    """
    if seed is not None:
        np.random.seed(seed)

    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    p = len(phi)

    # Characteristic polynomial: 1 - phi_1*z - ... - phi_p*z^p = 0
    # np.roots expects highest-power coefficient first
    char_poly = np.r_[1, -phi]
    roots = np.roots(char_poly)

    if np.any(np.abs(roots) <= 1.0 + 1e-10):
        raise ValueError(
            f"AR process is not stationary. Roots: {np.abs(roots).round(4)}. "
            "All roots must lie strictly outside the unit circle (|root| > 1)."
        )

    total_T = T + burn
    y = np.zeros(total_T)
    eps = np.random.randn(total_T) * sigma

    for t in range(p, total_T):
        # phi ordered phi_1, phi_2, ..., so phi[0] multiplies y[t-1]
        y[t] = c + np.dot(phi, y[t - p:t][::-1]) + eps[t]

    return y[burn:]


# --- 4. VAR estimation and companion matrix ---

def companion_matrix(Phi_list):
    """
    Construct the VAR(p) companion matrix.

    For VAR(p) with coefficient matrices Phi_1, ..., Phi_p (each n x n),
    the companion form stacks lags into a VAR(1):
        [Y_t, ..., Y_{t-p+1}]' = A * [Y_{t-1}, ..., Y_{t-p}]' + [u_t, 0, ...]'

    Stationarity iff all eigenvalues of companion matrix lie inside unit circle.

    Parameters
    ----------
    Phi_list : list of ndarray, [Phi_1, Phi_2, ..., Phi_p], each (n, n)

    Returns
    -------
    A : ndarray, shape (n*p, n*p), companion matrix
    """
    n = Phi_list[0].shape[0]
    p = len(Phi_list)
    A = np.zeros((n * p, n * p))

    # First block row: VAR coefficients
    for j, Phi in enumerate(Phi_list):
        A[:n, j * n:(j + 1) * n] = Phi

    # Remaining block rows: identity (shift operator)
    A[n:, :n * (p - 1)] = np.eye(n * (p - 1))

    return A


def is_var_stationary(Phi_list):
    """Check VAR stationarity via eigenvalues of companion matrix."""
    A = companion_matrix(Phi_list)
    eigenvalues = np.linalg.eigvals(A)
    max_eigenvalue = np.abs(eigenvalues).max()
    return max_eigenvalue < 1.0, max_eigenvalue


def estimate_var(Y, max_lag=10, criterion='aic'):
    """
    Estimate VAR with automatic lag selection.

    Uses statsmodels VAR, which fits all equations jointly via OLS.

    Parameters
    ----------
    Y : ndarray, shape (T, n), multivariate time series
    max_lag : int, maximum lag order to consider
    criterion : str, 'aic' or 'bic' for lag selection

    Returns
    -------
    statsmodels VARResults object
    """
    model = VAR(Y)
    results = model.fit(maxlags=max_lag, ic=criterion)
    return results


# --- Demonstration ---
if __name__ == "__main__":
    np.random.seed(42)

    # 1. Stationarity tests
    print("=== Stationarity Tests ===")
    y_stat = simulate_ar(phi=[0.8], T=300)
    y_nonstat = np.cumsum(np.random.randn(300))  # Random walk

    print("\nStationary AR(1), phi=0.8:")
    test_stationarity(y_stat)

    print("\nRandom walk (non-stationary):")
    test_stationarity(y_nonstat)

    # 2. Theoretical vs sample ACF for AR(1)
    phi = 0.7
    y = simulate_ar(phi=[phi], T=1000, seed=0)
    sample = sample_acf(y, max_lag=10)
    theoretical = np.array([phi ** h for h in range(11)])
    print("\nAR(1) ACF comparison (first 6 lags):")
    print(f"  Sample:      {sample[:6].round(3)}")
    print(f"  Theoretical: {theoretical[:6].round(3)}")

    # 3. VAR estimation with stationarity check
    n, T = 3, 400
    Phi_true = np.array([[0.5, 0.1, 0.0],
                          [0.2, 0.6, 0.1],
                          [0.0, 0.1, 0.4]])

    is_stationary, max_eig = is_var_stationary([Phi_true])
    print(f"\nVAR(1) stationary: {is_stationary}, max eigenvalue: {max_eig:.3f}")

    Y = np.zeros((T, n))
    for t in range(1, T):
        Y[t] = Phi_true @ Y[t - 1] + np.random.randn(n) * 0.3

    var_res = estimate_var(Y, max_lag=5)
    print(f"\nEstimated VAR lag order: {var_res.k_ar}")
    print(f"True Phi_1:\n{Phi_true}")
    print(f"Estimated Phi_1:\n{var_res.coefs[0].round(3)}")
```

## Common Pitfalls

**Ignoring non-stationarity before factor extraction.** If variables have unit roots (e.g., log prices), PCA on the levels extracts a near-unit-root factor that is not stationary. Always test and transform (difference, detrend) before factor estimation.

**Using only ADF or only KPSS.** ADF has low power against near-unit-root processes; KPSS rejects too often in small samples. Use both together and report when they disagree — disagreement usually indicates near-unit-root behavior requiring further investigation.

**Confusing ACF and PACF for model identification.** The ACF of an AR($p$) decays geometrically; its PACF cuts off sharply after lag $p$. The ACF of an MA($q$) cuts off after $q$; its PACF decays. Flipping these gives the wrong model order.

**Overfitting AR lag order.** More lags always reduce in-sample residuals but often hurt out-of-sample forecasting. Use BIC (which penalizes complexity more than AIC) for consistent lag selection, or use AIC if prediction accuracy on a held-out set is the goal.

**Eigenvalues of companion matrix vs. roots of characteristic polynomial.** Both test VAR stationarity, but they give different numbers. The eigenvalues of the companion matrix must all satisfy $|\lambda| < 1$. Equivalently, the roots of the characteristic polynomial $|I_n - \Phi_1 z - \cdots - \Phi_p z^p| = 0$ must all satisfy $|z| > 1$ (outside the unit circle).

## Connections

- **Builds on:** Basic probability, regression, undergraduate statistics
- **Leads to:** State-space representation (this module), Kalman filter (Module 2), factor dynamics estimation (Module 4)
- **Related to:** Spectral analysis; Wold representation theorem; cointegration for non-stationary factor models; ARCH/GARCH for time-varying volatility

## Practice Problems

1. Prove that AR(1) with $|\phi| < 1$ is stationary by computing $E[y_t]$, $\text{Var}(y_t)$, and $\text{Cov}(y_t, y_{t-h})$ directly from the process definition.

2. Derive the ACF of MA(1): $y_t = \varepsilon_t + \theta\varepsilon_{t-1}$. What happens to the ACF at lag 2 and higher? Why?

3. Using `simulate_ar`, generate AR(1) with $\phi = 0.95$ (near unit root) and test stationarity with both ADF and KPSS at significance level 0.05. How often do the tests agree? Repeat 100 times and tabulate results.

4. Write a function `var_forecasting_intervals(var_results, Y, h=10)` that computes $h$-step-ahead forecast intervals using the analytical formula for forecast variance accumulated from state equation innovations.

5. Implement from scratch the companion matrix for a VAR(2) with $n = 2$ variables. Verify that its eigenvalues are identical to the roots of the bivariate characteristic polynomial.

6. Download monthly US industrial production from FRED. Test stationarity on levels, first differences, and log-first differences. Which transformation gives a stationary series?

## Further Reading

- Hamilton, J. D. (1994). *Time Series Analysis*. Chapters 2–3 (ARMA), 10–11 (VAR). The standard graduate reference.
- Luetkepohl, H. (2005). *New Introduction to Multiple Time Series Analysis*. Comprehensive VAR reference with proofs.
- Shumway, R. H. & Stoffer, D. S. (2017). *Time Series Analysis and Its Applications*. Practical treatment with R examples; Chapter 5 covers state-space models.
- Ng, S. & Perron, P. (2001). "Lag Length Selection and the Construction of Unit Root Tests with Good Size and Power." *Econometrica*. On proper lag selection for ADF tests.

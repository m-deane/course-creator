# MIDAS Regression: Mixed Data Sampling

> **Reading time:** ~11 min | **Module:** Module 5: Mixed Frequency | **Prerequisites:** Modules 0-4

<div class="callout-key">

**Key Concept Summary:** MIDAS (Mixed Data Sampling) regression provides a parsimonious approach to forecasting low-frequency variables using high-frequency predictors through parametric lag weighting schemes. Unlike traditional distributed lag models, MIDAS uses beta or exponential Almon polynomials to reduce parameter ...

</div>

## In Brief

MIDAS (Mixed Data Sampling) regression provides a parsimonious approach to forecasting low-frequency variables using high-frequency predictors through parametric lag weighting schemes. Unlike traditional distributed lag models, MIDAS uses beta or exponential Almon polynomials to reduce parameter dimensionality while capturing rich lag dynamics.

<div class="callout-insight">

**Insight:** The key innovation of MIDAS is replacing potentially hundreds of lag coefficients with just 2-3 hyperparameters that control a smooth weighting function. This achieves both parsimony and flexibility—you can incorporate 60 days of data with 3 parameters instead of 60.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The MIDAS Framework

### Basic Setup

Predict low-frequency variable $Y_t^{(L)}$ using high-frequency predictor $X_s^{(H)}$:

$$Y_t^{(L)} = \beta_0 + \beta_1 B(L; \theta) X_t^{(H)} + \varepsilon_t$$

where $B(L; \theta)$ is a distributed lag polynomial with weights determined by parameter vector $\theta$.

### The Challenge Without MIDAS

Standard distributed lag model with $m$ lags per low-frequency period and $K$ periods:

$$Y_t = \beta_0 + \sum_{k=0}^{K-1} \sum_{j=1}^{m} \beta_{k,j} X_{t-k,j}^{(H)} + \varepsilon_t$$

This requires estimating $K \times m$ coefficients—impractical for $K=4$ quarters and $m=3$ months (12 parameters) or worse for daily data.

### The MIDAS Solution

Replace individual $\beta_{k,j}$ with a parametric weight function:

$$B(L; \theta) = \sum_{k=1}^{K} w(k; \theta) L^{k-1}$$

where $w(k; \theta)$ is a weighting function (typically normalized to sum to 1) depending on a small number of parameters $\theta$.

### Intuitive Explanation

Think of MIDAS weights as a "decay function" determining how much influence each lag has:
- Recent observations get higher weight (usually)
- Weights smoothly decay according to parametric function
- Shape of decay controlled by 2-3 hyperparameters
- Like a smoothed version of geometric decay, but more flexible

---

## 2. Weighting Functions

### Beta Weighting Function

The most popular MIDAS weighting scheme uses the Beta distribution:

$$w(k; \theta_1, \theta_2) = \frac{f\left(\frac{k}{K}; \theta_1, \theta_2\right)}{\sum_{j=1}^{K} f\left(\frac{j}{K}; \theta_1, \theta_2\right)}$$

where $f(x; \theta_1, \theta_2)$ is the Beta probability density function:

$$f(x; \theta_1, \theta_2) = \frac{x^{\theta_1 - 1}(1-x)^{\theta_2 - 1}}{B(\theta_1, \theta_2)}$$

and $B(\theta_1, \theta_2) = \frac{\Gamma(\theta_1)\Gamma(\theta_2)}{\Gamma(\theta_1 + \theta_2)}$ is the Beta function.

**Properties:**
- $\theta_1, \theta_2 > 0$ control shape
- $\theta_1 > 1, \theta_2 > 1$: hump-shaped (peak in middle)
- $\theta_1 < 1, \theta_2 > 1$: monotone decreasing (recent lags dominate)
- $\theta_1 > 1, \theta_2 < 1$: monotone increasing (older lags dominate)
- Flexible enough to capture various lag patterns

### Code Implementation: Beta Weights

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">midas_beta_weights.py</span>

</div>

```python
import numpy as np
from scipy.special import beta as beta_func
from scipy.stats import beta as beta_dist

def midas_beta_weights(K, theta1, theta2, normalize=True):
    """
    Compute MIDAS beta polynomial weights.

    Parameters
    ----------
    K : int
        Number of lags
    theta1 : float
        Beta distribution shape parameter 1 (> 0)
    theta2 : float
        Beta distribution shape parameter 2 (> 0)
    normalize : bool
        Whether to normalize weights to sum to 1

    Returns
    -------
    weights : ndarray, shape (K,)
        MIDAS lag weights
    """
    if theta1 <= 0 or theta2 <= 0:
        raise ValueError("Beta parameters must be positive")

    # Lag positions normalized to [0, 1]
    x = np.linspace(1/K, 1, K)

    # Beta PDF evaluated at lag positions
    weights = beta_dist.pdf(x, theta1, theta2)

    if normalize:
        weights = weights / weights.sum()

    return weights


# Example: Different beta weight patterns
K = 20  # 20 lags (e.g., 20 days in a month)

# Pattern 1: Exponential decay (recent lags important)
w1 = midas_beta_weights(K, theta1=1, theta2=5)

# Pattern 2: Hump shape (middle lags important)
w2 = midas_beta_weights(K, theta1=5, theta2=5)

# Pattern 3: Increasing (older lags important)
w3 = midas_beta_weights(K, theta1=5, theta2=1)

# Visualize
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(12, 3))
patterns = [
    (w1, r"$\theta_1=1, \theta_2=5$ (decay)"),
    (w2, r"$\theta_1=5, \theta_2=5$ (hump)"),
    (w3, r"$\theta_1=5, \theta_2=1$ (increasing)")
]

for ax, (weights, title) in zip(axes, patterns):
    ax.bar(range(K), weights, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Weight")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('midas_beta_weights.png', dpi=150)
print("Beta weight patterns saved to midas_beta_weights.png")
```

</div>

### Exponential Almon Lag Weights

Alternative specification using exponential of polynomial:

$$w(k; \theta) = \frac{\exp\left(\theta_1 k + \theta_2 k^2 + \ldots\right)}{\sum_{j=1}^{K} \exp\left(\theta_1 j + \theta_2 j^2 + \ldots\right)}$$

Typically use polynomial of degree 1 or 2.

### Code Implementation: Exponential Almon Weights

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">midas_almon_weights.py</span>

</div>

```python
def midas_almon_weights(K, theta, normalize=True):
    """
    Compute MIDAS exponential Almon weights.

    Parameters
    ----------
    K : int
        Number of lags
    theta : array-like
        Polynomial coefficients [theta1, theta2, ...]
    normalize : bool
        Whether to normalize weights to sum to 1

    Returns
    -------
    weights : ndarray, shape (K,)
        MIDAS lag weights
    """
    theta = np.atleast_1d(theta)
    lags = np.arange(1, K + 1)

    # Polynomial in lag index
    polynomial = sum(t * lags**i for i, t in enumerate(theta, start=1))

    # Exponentiate
    weights = np.exp(polynomial)

    if normalize:
        weights = weights / weights.sum()

    return weights


# Example: Exponential decay with linear polynomial
K = 20
theta_decay = [-0.1]  # Negative slope -> decay
theta_hump = [-0.05, 0.005]  # Quadratic with peak

w_almon_decay = midas_almon_weights(K, theta_decay)
w_almon_hump = midas_almon_weights(K, theta_hump)

fig, axes = plt.subplots(1, 2, figsize=(10, 3))
axes[0].bar(range(K), w_almon_decay, alpha=0.7)
axes[0].set_title("Almon Linear Decay")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Weight")

axes[1].bar(range(K), w_almon_hump, alpha=0.7)
axes[1].set_title("Almon Quadratic Hump")
axes[1].set_xlabel("Lag")

plt.tight_layout()
plt.savefig('midas_almon_weights.png', dpi=150)
```

</div>

---

## 3. MIDAS Regression Estimation

### Model Specification

Basic MIDAS regression:

$$Y_t = \beta_0 + \beta_1 \sum_{k=1}^{K} w(k; \theta) X_{t-(k-1)/m}^{(H)} + \varepsilon_t$$

where:
- $Y_t$ is low-frequency target (e.g., quarterly GDP growth)
- $X_s^{(H)}$ is high-frequency predictor (e.g., monthly IP growth)
- $m$ is frequency ratio (3 for monthly-to-quarterly)
- $w(k; \theta)$ are MIDAS weights

### Nonlinear Least Squares Estimation

The MIDAS regression is **nonlinear** in $\theta$ (but linear in $\beta_0, \beta_1$), so we use nonlinear least squares:

1. For given $\theta$, compute weights $w(k; \theta)$
2. Construct weighted regressor: $Z_t(\theta) = \sum_{k=1}^{K} w(k; \theta) X_{t-(k-1)/m}^{(H)}$
3. Run OLS: $Y_t = \beta_0 + \beta_1 Z_t(\theta) + \varepsilon_t$
4. Optimize over $\theta$ to minimize RSS

### Code Implementation

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">midasregression.py</span>

</div>

```python
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

class MIDASRegression:
    """
    MIDAS regression with beta weighting function.

    Estimates low-frequency outcome from high-frequency predictor
    using parametric lag weighting.
    """

    def __init__(self, K, weight_function='beta', m=3):
        """
        Parameters
        ----------
        K : int
            Number of high-frequency lags to include
        weight_function : str
            'beta' or 'almon'
        m : int
            Frequency ratio (e.g., 3 for monthly-to-quarterly)
        """
        self.K = K
        self.weight_function = weight_function
        self.m = m
        self.theta_opt = None
        self.beta_opt = None
        self.weights_opt = None

    def _construct_weighted_regressor(self, X_high, theta):
        """
        Create MIDAS weighted regressor for each low-freq observation.

        Parameters
        ----------
        X_high : ndarray, shape (T_high,)
            High-frequency predictor
        theta : array-like
            Weight function parameters

        Returns
        -------
        Z : ndarray, shape (T_low,)
            Weighted high-frequency information
        """
        if self.weight_function == 'beta':
            weights = midas_beta_weights(self.K, theta[0], theta[1])
        elif self.weight_function == 'almon':
            weights = midas_almon_weights(self.K, theta)
        else:
            raise ValueError(f"Unknown weight function: {self.weight_function}")

        # Number of low-frequency observations
        T_low = (len(X_high) - self.K + 1) // self.m

        # Construct weighted regressor for each low-freq period
        Z = np.zeros(T_low)
        for t in range(T_low):
            # Index of last high-freq obs in this low-freq period
            idx_end = (t + 1) * self.m - 1
            # Indices of K lags
            indices = np.arange(idx_end - self.K + 1, idx_end + 1)
            # Weighted sum
            Z[t] = np.sum(weights * X_high[indices])

        return Z

    def _objective(self, theta, Y_low, X_high):
        """Compute RSS for given theta."""
        try:
            Z = self._construct_weighted_regressor(X_high, theta)
            # OLS given weighted regressor
            X_ols = np.column_stack([np.ones(len(Z)), Z])
            beta = np.linalg.lstsq(X_ols, Y_low, rcond=None)[0]
            residuals = Y_low - X_ols @ beta
            return np.sum(residuals**2)
        except:
            return 1e10  # Penalty for invalid parameters

    def fit(self, Y_low, X_high, theta_init=None):
        """
        Estimate MIDAS regression.

        Parameters
        ----------
        Y_low : ndarray, shape (T_low,)
            Low-frequency target variable
        X_high : ndarray, shape (T_high,)
            High-frequency predictor
        theta_init : array-like, optional
            Initial guess for theta

        Returns
        -------
        self : MIDASRegression
            Fitted model
        """
        if theta_init is None:
            if self.weight_function == 'beta':
                theta_init = [1.5, 5.0]  # Slight decay
            else:
                theta_init = [-0.1]

        # Optimize theta
        result = minimize(
            self._objective,
            theta_init,
            args=(Y_low, X_high),
            method='Nelder-Mead',
            bounds=[(0.1, 20), (0.1, 20)] if self.weight_function == 'beta' else None
        )

        self.theta_opt = result.x

        # Compute optimal weights
        if self.weight_function == 'beta':
            self.weights_opt = midas_beta_weights(self.K, *self.theta_opt)
        else:
            self.weights_opt = midas_almon_weights(self.K, self.theta_opt)

        # Compute optimal beta
        Z = self._construct_weighted_regressor(X_high, self.theta_opt)
        X_ols = np.column_stack([np.ones(len(Z)), Z])
        self.beta_opt = np.linalg.lstsq(X_ols, Y_low, rcond=None)[0]

        return self

    def predict(self, X_high):
        """Generate predictions for new high-frequency data."""
        Z = self._construct_weighted_regressor(X_high, self.theta_opt)
        X_ols = np.column_stack([np.ones(len(Z)), Z])
        return X_ols @ self.beta_opt

    def summary(self):
        """Print estimation summary."""
        print("MIDAS Regression Results")
        print("=" * 50)
        print(f"Weight function: {self.weight_function}")
        print(f"Number of lags: {self.K}")
        print(f"Frequency ratio: {self.m}")
        print(f"\nOptimal theta: {self.theta_opt}")
        print(f"Optimal beta: {self.beta_opt}")
        print(f"\nWeight pattern (first 10 lags):")
        print(self.weights_opt[:10])


# Example: Forecast quarterly GDP growth with monthly IP growth
np.random.seed(42)

# Generate synthetic data
T_monthly = 60  # 5 years of monthly data
months = np.arange(T_monthly)

# True underlying factor
factor = np.sin(2 * np.pi * months / 12) + np.cumsum(np.random.randn(T_monthly) * 0.1)

# Monthly industrial production (noisy high-freq indicator)
ip_monthly = factor + np.random.randn(T_monthly) * 0.5

# Quarterly GDP (related to factor with decay weights)
gdp_quarterly = np.zeros(T_monthly // 3)
true_weights = midas_beta_weights(9, theta1=1, theta2=3)  # 9 months of lags
for q in range(len(gdp_quarterly)):
    idx_end = (q + 1) * 3 - 1
    indices = np.arange(max(0, idx_end - 8), idx_end + 1)
    weights_used = true_weights[-len(indices):]
    gdp_quarterly[q] = np.sum(weights_used * factor[indices]) + np.random.randn() * 0.3

# Split into train/test
n_train = 15
Y_train, Y_test = gdp_quarterly[:n_train], gdp_quarterly[n_train:]
X_train, X_test = ip_monthly[:n_train*3], ip_monthly[n_train*3:]

# Fit MIDAS
midas = MIDASRegression(K=9, weight_function='beta', m=3)
midas.fit(Y_train, ip_monthly[:n_train*3 + 8])  # Need extra lags
midas.summary()

# Forecast
Y_pred = midas.predict(ip_monthly)

print(f"\nOut-of-sample RMSE: {np.sqrt(mean_squared_error(Y_test, Y_pred[-len(Y_test):])):.4f}")
```

</div>

---

## 4. Extensions and Variations

### U-MIDAS (Unrestricted MIDAS)

Include multiple MIDAS terms with different weight functions:

$$Y_t = \beta_0 + \beta_1 B(L; \theta_1) X_t^{(1)} + \beta_2 B(L; \theta_2) X_t^{(2)} + \varepsilon_t$$

Allows flexible combinations of predictors at different frequencies.

### ADL-MIDAS (Autoregressive Distributed Lag)

Include autoregressive terms:

$$Y_t = \alpha Y_{t-1} + \beta_0 + \beta_1 B(L; \theta) X_t^{(H)} + \varepsilon_t$$

Improves forecasts by capturing persistence in target.

### MIDAS with Factors

Combine MIDAS weighting with factor extraction:

$$Y_t = \beta_0 + \beta_1 B(L; \theta) \hat{F}_t + \varepsilon_t$$

where $\hat{F}_t$ are factors extracted from many high-frequency predictors.

---

## Common Pitfalls

### 1. Too Few Lags ($K$ too small)
- **Symptom**: Poor fit, residual autocorrelation
- **Fix**: Increase $K$ to capture full information set
- **Rule of thumb**: $K \geq 2m$ (at least 2 low-freq periods)

### 2. Optimization Convergence Failures
- **Symptom**: Unstable parameter estimates, boundary solutions
- **Fix**: Try multiple initializations, use constrained optimization
- **Diagnostic**: Check if weights are smooth and economically sensible

### 3. Ignoring Temporal Alignment
- **Mistake**: Incorrect mapping between high and low frequency indices
- **Fix**: Carefully align dates, account for publication lags
- **Example**: GDP for Q1 published in April uses data through March

### 4. Overfitting with Unrestricted Lags
- **Mistake**: Estimating separate coefficient for each lag
- **Fix**: Use MIDAS weights to impose smoothness
- **Benefit**: Better out-of-sample forecasts despite lower in-sample fit

---

## Connections

- **Builds on:** Temporal aggregation, distributed lag models
- **Leads to:** Mixed-frequency state-space models, nowcasting
- **Related to:** Polynomial distributed lags, HAR models in finance

---

## Practice Problems

### Conceptual

1. Why is MIDAS nonlinear in parameters? What are the implications for estimation and inference?

2. Compare beta weights with $\theta_1 = 1, \theta_2 = 5$ to geometric decay $w_k = \rho^k$. When would each be preferable?

3. You're forecasting quarterly GDP using daily stock returns. How would you choose $K$?

### Implementation

4. Implement a function to compute standard errors for MIDAS parameters using the delta method.

5. Create a "MIDAS race" comparing beta, Almon, and unrestricted lag specifications on real data.

6. Extend `MIDASRegression` to handle multiple predictors (U-MIDAS).

### Extension

7. Derive the relationship between MIDAS weights and the mixed-frequency state-space representation.

8. Implement MIDAS-QR (quantile regression version) to forecast the distribution, not just the mean.

---

## Further Reading

- **Ghysels, E., Santa-Clara, P. & Valkanov, R.** (2006). "Predicting volatility: Getting the most out of return data sampled at different frequencies." *Journal of Econometrics*, 131(1-2), 59-95.
  - Original MIDAS paper for volatility forecasting

- **Ghysels, E., Sinko, A. & Valkanov, R.** (2007). "MIDAS regressions: Further results and new directions." *Econometric Reviews*, 26(1), 53-90.
  - Extensions including U-MIDAS and theoretical properties

- **Andreou, E., Ghysels, E. & Kourtellos, A.** (2013). "Should macroeconomic forecasters use daily financial data and how?" *Journal of Business & Economic Statistics*, 31(2), 240-251.
  - Applied guide to MIDAS for macroeconomic forecasting

- **Foroni, C., Marcellino, M. & Schumacher, C.** (2015). "Unrestricted mixed data sampling (MIDAS): MIDAS regressions with unrestricted lag polynomials." *Journal of the Royal Statistical Society A*, 178(1), 57-82.
  - U-MIDAS theoretical justification and comparisons

---

<div class="callout-insight">

**Insight:** Understanding midas regression is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Summary

**Key Takeaways:**
1. MIDAS achieves parsimony by parameterizing lag weights with 2-3 hyperparameters
2. Beta and exponential Almon functions provide flexible, smooth weight patterns
3. Nonlinear least squares estimation optimizes both weights and regression coefficients
4. MIDAS excels when incorporating many high-frequency lags into low-frequency forecasts

**Next Steps:**
The next guide integrates MIDAS concepts into state-space DFMs, showing how to estimate mixed-frequency factor models using the Kalman filter with proper temporal aggregation.

---

## Conceptual Practice Questions

1. What is the fundamental insight behind MIDAS regression?

2. Why does MIDAS use polynomial weight functions instead of unrestricted coefficients?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_midas_regression_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_midas_regression.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_temporal_aggregation.md">
  <div class="link-card-title">01 Temporal Aggregation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_state_space_mixed_freq.md">
  <div class="link-card-title">03 State Space Mixed Freq</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


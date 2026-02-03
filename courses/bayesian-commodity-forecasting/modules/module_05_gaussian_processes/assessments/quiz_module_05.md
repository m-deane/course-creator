# Module 5 Quiz: Gaussian Processes for Commodity Forecasting

**Course:** Bayesian Commodity Forecasting
**Module:** 05 - Gaussian Processes and Kernel Design
**Time Limit:** 30 minutes
**Total Points:** 100 points
**Instructions:** Answer all questions. Show mathematical work where required.

---

## Section A: GP Fundamentals (30 points)

### Question 1 (10 points)
Define a Gaussian Process and explain how it differs from standard parametric regression. Then describe the three key components needed to fully specify a GP model for commodity price forecasting.

**Answer:**

**Gaussian Process Definition:**

A Gaussian Process is a **distribution over functions** such that any finite collection of function values follows a multivariate normal distribution:

$$f(x) \sim \mathcal{GP}(m(x), k(x, x'))$$

where:
- $m(x) = \mathbb{E}[f(x)]$ is the **mean function**
- $k(x, x') = \text{Cov}[f(x), f(x')]$ is the **covariance (kernel) function**

**Formal:** For any finite set of inputs $X = \{x_1, ..., x_n\}$:
$$\mathbf{f} = [f(x_1), ..., f(x_n)]' \sim \mathcal{N}(\mathbf{m}, K)$$

where $K_{ij} = k(x_i, x_j)$ and $\mathbf{m}_i = m(x_i)$.

---

**Comparison to Parametric Regression:**

| Aspect | Parametric Regression | Gaussian Process |
|--------|----------------------|------------------|
| **Form** | $y = \beta_0 + \beta_1 x + ... + \beta_p x^p$ | No fixed form; infinite-dimensional |
| **Parameters** | Fixed number ($\beta_0, ..., \beta_p$) | Infinite (function values at all x) |
| **Flexibility** | Limited by chosen basis | Highly flexible (determined by kernel) |
| **Uncertainty** | Point estimates + CI | Full posterior over functions |
| **Complexity** | Grows with model order | Controlled by kernel hyperparameters |

**Example:**
- Parametric: Linear regression assumes $f(x) = \beta_0 + \beta_1 x$ (straight line)
- GP: Can fit any smooth function consistent with kernel structure

**Key advantage:** GP adapts complexity to data, avoiding over/underfitting.

---

**Three Key Components for Commodity Forecasting GP:**

**1. Mean Function $m(x)$**

Specifies prior expectation of function values.

**Common choices:**
- **Zero mean:** $m(x) = 0$ (simplest, often sufficient)
- **Linear trend:** $m(x) = \beta_0 + \beta_1 t$ (for trending commodities)
- **Seasonal mean:** $m(x) = \sum_k a_k \sin(2\pi k t / 12)$ (agricultural commodities)

**For oil prices:** Might use zero mean if data is detrended/scaled.

---

**2. Covariance (Kernel) Function $k(x, x')$**

Encodes assumptions about function smoothness and structure.

**Common kernels:**
- **Radial Basis Function (RBF):**
  $$k(x, x') = \sigma^2 \exp\left(-\frac{(x-x')^2}{2\ell^2}\right)$$
  - $\sigma^2$: variance (amplitude)
  - $\ell$: lengthscale (how quickly correlation decays)
  - Use for: Smooth, continuous trends

- **Matérn:**
  $$k(x, x') = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu}|x-x'|}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu}|x-x'|}{\ell}\right)$$
  - $\nu$: smoothness parameter
  - Use for: Less smooth functions (e.g., $\nu=1.5, 2.5$)

- **Periodic:**
  $$k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi|x-x'|/p)}{\ell^2}\right)$$
  - $p$: period
  - Use for: Seasonal patterns

**For commodities:** Combine kernels (additive or multiplicative) to capture multiple patterns.

---

**3. Observation Noise $\sigma_n^2$**

Accounts for measurement error or irreducible noise:
$$y = f(x) + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma_n^2)$$

**Full model:**
$$\mathbf{y} \sim \mathcal{N}(\mathbf{m}, K + \sigma_n^2 I)$$

**For commodities:**
- High-frequency data: Large $\sigma_n^2$ (microstructure noise)
- Weekly/monthly: Smaller $\sigma_n^2$ (more signal)

---

**Complete GP Specification:**
$$\begin{aligned}
f(t) &\sim \mathcal{GP}(m(t), k(t, t')) \\
y_t &= f(t) + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_n^2)
\end{aligned}$$

**Scoring:**
- GP definition: 3 points
- Comparison to parametric: 3 points
- Three components: 4 points (1 point each for mean, kernel, noise + 1 for clarity)

---

### Question 2 (8 points)
The Radial Basis Function (RBF) kernel is defined as:
$$k(x, x') = \sigma^2 \exp\left(-\frac{(x-x')^2}{2\ell^2}\right)$$

**(a)** What does the lengthscale parameter $\ell$ control? Provide an intuitive explanation and a commodity forecasting example. (4 points)

**(b)** If $\ell \to 0$, what happens to the GP? What about $\ell \to \infty$? (4 points)

**Answer:**

**(a)** Lengthscale parameter $\ell$:

**Intuitive explanation:**

$\ell$ controls **how far apart two inputs must be before they become uncorrelated**.

**Mathematical:**
- Correlation between $f(x)$ and $f(x')$ decays exponentially with distance $|x-x'|$
- Decay rate: $1/\ell^2$
- At distance $|x-x'| = \ell$, correlation = $\exp(-1/2) \approx 0.61$
- At distance $|x-x'| = 2\ell$, correlation = $\exp(-2) \approx 0.14$

**Interpretation:**
- **Small $\ell$:** Function changes rapidly; nearby points weakly correlated
  - **Wigglier** functions
  - Model captures high-frequency variation
- **Large $\ell$:** Function changes slowly; distant points still correlated
  - **Smoother** functions
  - Model captures low-frequency variation

**Visual:**
```
Small ℓ:  ╱╲╱╲╱╲╱╲  (high frequency, rapid changes)
Large ℓ:  ╱──────╲  (low frequency, smooth)
```

---

**Commodity forecasting example:**

**Crude oil prices (daily data):**

- **Small $\ell$ (~1-3 days):**
  - GP fits day-to-day noise and volatility
  - Overfits: Tracks every zigzag
  - Poor forecasting: No generalization

- **Large $\ell$ (~30-60 days):**
  - GP captures medium-term trends and cycles
  - Underfits: Misses short-term dynamics
  - Good for trend extraction, not short-term trading

- **Optimal $\ell$ (~10-20 days):**
  - Balance: Captures meaningful price movements without overfitting
  - Estimate from data via marginal likelihood maximization

**Practical:**
- Cross-validate to find optimal $\ell$
- Or use PyMC to estimate $\ell$ with prior: $\ell \sim \text{InverseGamma}(5, 5)$

---

**(b)** Extreme lengthscales:

**Case 1: $\ell \to 0$ (tiny lengthscale)**

**Effect on kernel:**
$$k(x, x') \to \begin{cases} \sigma^2 & \text{if } x = x' \\ 0 & \text{if } x \neq x' \end{cases}$$

Covariance matrix becomes diagonal: $K = \sigma^2 I$

**Implication:**
- **No correlation** between function values at different points
- GP reduces to **independent priors** at each point: $f(x_i) \sim \mathcal{N}(0, \sigma^2)$ independently
- **Interpolation becomes trivial:** Pass through every data point (if $\sigma_n^2 = 0$) or just fit noise
- **Forecasting fails:** No information transfer from observed to unobserved points

**Not useful for forecasting!**

---

**Case 2: $\ell \to \infty$ (huge lengthscale)**

**Effect on kernel:**
$$k(x, x') \to \sigma^2 \quad \text{(constant for all } x, x' \text{)}$$

Covariance matrix becomes constant: $K_{ij} = \sigma^2$ for all $i, j$

**Implication:**
- **Perfect correlation** between all function values
- GP believes function is **constant:** $f(x) \approx c$ for all $x$
- Equivalent to **global mean model**
- Posterior: $f(x) \approx \bar{y}$ everywhere

**Example:**
- If oil prices range $70-$90, GP with $\ell \to \infty$ predicts ~$80 everywhere
- Ignores time variation completely

**Also not useful!**

---

**Practical takeaway:**

Need $\ell$ in a "Goldilocks zone":
- Not too small (overfitting)
- Not too large (underfitting)
- Typically $\ell \approx 5-20\%$ of data range for time series

**Estimate from data or use informative prior:**
```python
ℓ ~ InverseGamma(α, β)  # Weakly informative
# or
ℓ ~ Normal(10, 5)  # If you know typical scale
```

**Scoring:**
- Part (a): 4 points (lengthscale interpretation + example)
- Part (b): 4 points (both extreme cases)

---

### Question 3 (12 points)
Consider forecasting natural gas prices with strong annual seasonality. Design a composite kernel that captures:
1. Long-term trend
2. Annual seasonality
3. Short-term fluctuations

**(a)** Write the mathematical form of this composite kernel. (6 points)

**(b)** Explain how each component contributes to the forecast. (4 points)

**(c)** How would you specify priors for the hyperparameters in PyMC? (2 points)

**Answer:**

**(a)** Composite kernel specification:

**Component 1: Long-term trend (RBF or Matérn)**
$$k_{\text{trend}}(t, t') = \sigma_{\text{trend}}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{\text{trend}}^2}\right)$$

- $\sigma_{\text{trend}}^2$: Amplitude of trend variation
- $\ell_{\text{trend}}$: Lengthscale (should be large, e.g., 180-365 days)

**Component 2: Annual seasonality (Periodic kernel)**
$$k_{\text{seasonal}}(t, t') = \sigma_{\text{seasonal}}^2 \exp\left(-\frac{2\sin^2(\pi|t-t'|/365)}{\ell_{\text{seasonal}}^2}\right)$$

- $\sigma_{\text{seasonal}}^2$: Amplitude of seasonal variation
- $\ell_{\text{seasonal}}$: Smoothness of seasonal pattern
- Period fixed at $p = 365$ days (annual)

**Component 3: Short-term fluctuations (Matérn or RBF)**
$$k_{\text{short}}(t, t') = \sigma_{\text{short}}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{\text{short}}^2}\right)$$

- $\sigma_{\text{short}}^2$: Amplitude of short-term variation
- $\ell_{\text{short}}$: Lengthscale (should be small, e.g., 7-14 days)

**Observation noise:**
$$\sigma_n^2$$

**Composite kernel (additive):**
$$k_{\text{total}}(t, t') = k_{\text{trend}}(t, t') + k_{\text{seasonal}}(t, t') + k_{\text{short}}(t, t')$$

**Full covariance:**
$$K = K_{\text{trend}} + K_{\text{seasonal}} + K_{\text{short}} + \sigma_n^2 I$$

**Why additive?**
- Components operate at different timescales
- Independent sources of variation sum
- Allows disentangling trend from seasonality from noise

**Alternative (multiplicative for interaction):**
$$k(t, t') = k_{\text{trend}}(t, t') \times k_{\text{seasonal}}(t, t') + k_{\text{short}}(t, t')$$

Use if seasonal amplitude varies with trend level.

---

**(b)** How each component contributes:

**1. Trend component:**

**Contribution:** Captures **low-frequency drift** in baseline price level
- Example: Natural gas prices rising from $3/MMBtu to $5/MMBtu over 2 years
- Lengthscale $\ell_{\text{trend}} \approx 180$ days → smooth, slow-changing
- **Forecasting:** Extrapolates trend forward (with increasing uncertainty)

**Visual role:**
```
Price over time:
  │   ╱────────────── Trend component
  │  ╱
  │ ╱
  └──────────────────→ Time
```

---

**2. Seasonal component:**

**Contribution:** Captures **periodic pattern** repeating yearly
- Natural gas: High winter (Jan-Feb), low summer shoulder (Apr-May), moderate summer (Jul-Aug)
- Period $p = 365$ → forces annual repetition
- $\ell_{\text{seasonal}}$ controls smoothness within the cycle

**Forecasting:** For time $t + 365k$ (k years ahead):
$$k_{\text{seasonal}}(t, t+365k) = \sigma_{\text{seasonal}}^2$$

Perfect correlation at exact annual lags → forecasts inherit seasonal pattern.

**Visual role:**
```
Seasonal pattern (detrended):
  │  ╱╲    ╱╲    ╱╲
  │ ╱  ╲  ╱  ╲  ╱  ╲
  │╱    ╲╱    ╲╱    ╲
  └──────────────────→ Time
   Winter  Summer  Winter
```

---

**3. Short-term component:**

**Contribution:** Captures **high-frequency fluctuations** not explained by trend or season
- Weather shocks, inventory surprises, short-term supply disruptions
- Lengthscale $\ell_{\text{short}} \approx 7-14$ days → rapid decorrelation
- **Does not forecast far:** Correlation decays quickly, so little predictive power beyond ~2-3 weeks

**Forecasting:** At horizon $h$:
- If $h \ll \ell_{\text{short}}$: Short-term component contributes (smooth recent data)
- If $h \gg \ell_{\text{short}}$: Short-term component → 0 (no info), forecast relies on trend + seasonal

**Visual role:**
```
Short-term noise (residual after removing trend + seasonal):
  │  ╱╲╱╲╱╲╱╲╱╲╱╲
  │─╱──────────────  (zero mean)
  │╲╱╲╱╲╱╲╱╲╱╲╱╲
  └────────────────→ Time
```

---

**Combined effect:**

At time $t$:
$$f(t) = \underbrace{f_{\text{trend}}(t)}_{\text{baseline level}} + \underbrace{f_{\text{seasonal}}(t)}_{\text{predictable cycle}} + \underbrace{f_{\text{short}}(t)}_{\text{transient shocks}}$$

**Forecasting breakdown:**
- **1 week ahead:** All three components matter
- **1 month ahead:** Short-term component fades, trend + seasonal dominate
- **1 year ahead:** Only trend and seasonal (short-term fully uncorrelated)

**Uncertainty decomposition:**
$$\text{Var}(f^*) = \sigma_{\text{trend}}^2 + \sigma_{\text{seasonal}}^2 + \sigma_{\text{short}}^2 \cdot k_{\text{short}}(t^*, t^*) + \sigma_n^2$$

where $k_{\text{short}}(t^*, t^*)$ depends on distance from last observation.

---

**(c)** PyMC hyperparameter priors:

```python
import pymc as pm

with pm.Model() as gp_model:
    # ===========================
    # HYPERPARAMETER PRIORS
    # ===========================

    # Trend component
    σ²_trend = pm.HalfNormal('sigma2_trend', sigma=10)
    # Justification: Price trend variation typically ~$5-10
    ℓ_trend = pm.InverseGamma('ell_trend', alpha=2, beta=180)
    # Justification: Trend lengthscale ~6 months to 2 years (median ~180 days)

    # Seasonal component
    σ²_seasonal = pm.HalfNormal('sigma2_seasonal', sigma=5)
    # Justification: Seasonal amplitude typically ~$2-5
    ℓ_seasonal = pm.InverseGamma('ell_seasonal', alpha=2, beta=30)
    # Justification: Controls smoothness within annual cycle (median ~30 days)
    # Period FIXED at 365 days (known)

    # Short-term component
    σ²_short = pm.HalfNormal('sigma2_short', sigma=3)
    # Justification: Short-term fluctuations smaller than seasonal
    ℓ_short = pm.InverseGamma('ell_short', alpha=3, beta=10)
    # Justification: Short lengthscale ~7-14 days (median ~10 days)

    # Observation noise
    σ²_obs = pm.HalfNormal('sigma2_obs', sigma=2)
    # Justification: Daily measurement noise typically ~$1-2

    # ===========================
    # KERNEL CONSTRUCTION
    # ===========================

    # Define composite kernel (see next question for full implementation)
    # k_total = k_trend + k_seasonal + k_short
```

**Prior choice rationale:**
- **HalfNormal for variances:** Positive constraint, mild regularization
- **InverseGamma for lengthscales:** Positive, allows wide range, weakly informative
- **Alternative:** LogNormal priors also common for lengthscales

**Scoring:**
- Part (a): 6 points (three kernel components correct)
- Part (b): 4 points (contributions explained)
- Part (c): 2 points (appropriate priors)

---

## Section B: GP Regression and Forecasting (40 points)

### Question 4 (12 points)
Derive the GP posterior predictive distribution for a new input $x^*$, given training data $(X, \mathbf{y})$.

**Start with:**
- Prior: $f \sim \mathcal{GP}(0, k)$
- Observations: $y_i = f(x_i) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$

**Show:**
- Posterior mean $\mathbb{E}[f^* | X, \mathbf{y}]$
- Posterior variance $\text{Var}(f^* | X, \mathbf{y})$

**Answer:**

**Setup:**

**Prior distribution:**
$$\begin{bmatrix} \mathbf{f} \\ f^* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{0} \\ 0 \end{bmatrix}, \begin{bmatrix} K & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix} \right)$$

where:
- $\mathbf{f} = [f(x_1), ..., f(x_n)]'$: Function values at training points
- $f^* = f(x^*)$: Function value at test point
- $K$: $n \times n$ covariance matrix, $K_{ij} = k(x_i, x_j)$
- $\mathbf{k}_* = [k(x_1, x^*), ..., k(x_n, x^*)]'$: $n \times 1$ cross-covariance vector
- $k_{**} = k(x^*, x^*)$: Variance at test point

**Observation model:**
$$\mathbf{y} = \mathbf{f} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 I)$$

So:
$$\mathbf{y} \sim \mathcal{N}(\mathbf{0}, K + \sigma_n^2 I)$$

Let $K_y = K + \sigma_n^2 I$ (augmented covariance with noise).

---

**Step 1: Joint distribution of $\mathbf{y}$ and $f^*$**

Since $f^*$ is prior (no observation noise added):
$$\begin{bmatrix} \mathbf{y} \\ f^* \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \mathbf{0} \\ 0 \end{bmatrix}, \begin{bmatrix} K_y & \mathbf{k}_* \\ \mathbf{k}_*^T & k_{**} \end{bmatrix} \right)$$

---

**Step 2: Conditional distribution (Gaussian conditioning formula)**

For joint Gaussian:
$$\begin{bmatrix} \mathbf{a} \\ \mathbf{b} \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \boldsymbol{\mu}_a \\ \boldsymbol{\mu}_b \end{bmatrix}, \begin{bmatrix} \Sigma_{aa} & \Sigma_{ab} \\ \Sigma_{ba} & \Sigma_{bb} \end{bmatrix} \right)$$

The conditional distribution is:
$$\mathbf{b} | \mathbf{a} \sim \mathcal{N}(\boldsymbol{\mu}_{b|a}, \Sigma_{b|a})$$

where:
$$\begin{aligned}
\boldsymbol{\mu}_{b|a} &= \boldsymbol{\mu}_b + \Sigma_{ba} \Sigma_{aa}^{-1} (\mathbf{a} - \boldsymbol{\mu}_a) \\
\Sigma_{b|a} &= \Sigma_{bb} - \Sigma_{ba} \Sigma_{aa}^{-1} \Sigma_{ab}
\end{aligned}$$

---

**Step 3: Apply to GP posterior**

Applying the formula with $\mathbf{a} = \mathbf{y}$, $\mathbf{b} = f^*$:

**Posterior mean:**
$$\begin{aligned}
\mathbb{E}[f^* | X, \mathbf{y}] &= 0 + \mathbf{k}_*^T K_y^{-1} (\mathbf{y} - \mathbf{0}) \\
&= \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}
\end{aligned}$$

**Posterior variance:**
$$\begin{aligned}
\text{Var}(f^* | X, \mathbf{y}) &= k_{**} - \mathbf{k}_*^T K_y^{-1} \mathbf{k}_* \\
&= k_{**} - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
\end{aligned}$$

---

**Step 4: Predictive distribution (including observation noise)**

For predicting new observation $y^* = f^* + \epsilon^*$:
$$y^* | X, \mathbf{y} \sim \mathcal{N}(\bar{f}^*, \sigma_{f^*}^2 + \sigma_n^2)$$

where:
- $\bar{f}^* = \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$ (posterior mean)
- $\sigma_{f^*}^2 = k_{**} - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$ (posterior variance)

---

**Final Result:**

$$\boxed{\begin{aligned}
\mathbb{E}[f^* | X, \mathbf{y}] &= \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y} \\
\text{Var}(f^* | X, \mathbf{y}) &= k_{**} - \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*
\end{aligned}}$$

---

**Interpretation:**

**Posterior mean:**
- Weighted average of training outputs $\mathbf{y}$
- Weights: $\mathbf{w} = (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$
- Points closer to $x^*$ (large $k(x_i, x^*)$) get more weight

**Posterior variance:**
- **Prior uncertainty:** $k_{**}$
- **Reduction from data:** $-\mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$
- At training point: $\text{Var}(f(x_i)) \to 0$ (interpolates)
- Far from training data: $\text{Var}(f^*) \to k_{**}$ (reverts to prior)

**Scoring:**
- Correct joint distribution setup: 3 points
- Conditioning formula applied: 4 points
- Final posterior mean and variance: 4 points
- Interpretation: 1 point

---

### Question 5 (10 points)
You fit a GP to weekly crude oil prices with RBF kernel. The posterior mean function perfectly interpolates the training data (passes through every observation). However, when forecasting one week ahead on held-out test data, the mean squared error is very high.

**(a)** Diagnose the problem. What has gone wrong? (4 points)

**(b)** Propose two solutions. (4 points)

**(c)** How would you validate which solution is better? (2 points)

**Answer:**

**(a)** Problem diagnosis:

**Symptom:** Perfect training fit + poor test performance = **Overfitting**

**Root cause: Observation noise $\sigma_n^2$ is too small** (possibly set to zero)

**Mechanism:**
- With $\sigma_n^2 \approx 0$, covariance matrix is $K_y = K + 0 \cdot I = K$
- GP forced to interpolate every data point exactly
- **Problem:** Training data includes noise, not just signal
- GP fits noise as if it's signal → poor generalization

**Mathematical:**
- Posterior variance at training point $x_i$:
  $$\text{Var}(f(x_i) | \mathbf{y}) = k(x_i, x_i) - \mathbf{k}_i^T K_y^{-1} \mathbf{k}_i$$
- If $\sigma_n^2 = 0$: This becomes 0 (perfect interpolation)
- Model has no "slack" to average over noisy observations

**Alternative causes:**
1. **Lengthscale $\ell$ too small:** Wigglier functions, overfits local noise
2. **Too much model complexity:** Composite kernel with too many components

But perfect interpolation strongly suggests $\sigma_n^2$ issue.

**Visual:**
```
Overfitted GP (σ_n² = 0):
Price │     ●╱╲●╱╲●╱╲●  (fits every point, including noise)
      │    ╱  ╲   ╱  ╲
      │   ╱    ╲╱    ╲
      └────────────────→ Time

True signal:      ╱────╲  (smooth trend missed)
```

---

**(b)** Solutions:

**Solution 1: Estimate observation noise $\sigma_n^2$ from data**

Don't fix $\sigma_n^2 = 0$; instead, treat as hyperparameter to learn.

**PyMC implementation:**
```python
with pm.Model():
    # Kernel hyperparameters
    σ²_signal = pm.HalfNormal('sigma2_signal', sigma=10)
    ℓ = pm.InverseGamma('ell', alpha=3, beta=20)

    # OBSERVATION NOISE (to estimate)
    σ²_obs = pm.HalfNormal('sigma2_obs', sigma=5)
    # Prior allows σ² ~ 0 to 10

    # GP
    cov_func = σ²_signal * pm.gp.cov.ExpQuad(1, ls=ℓ)
    gp = pm.gp.Marginal(cov_func=cov_func)

    # Likelihood with noise
    y_ = gp.marginal_likelihood('y', X=X_train, y=y_train, noise=σ²_obs)

    trace = pm.sample(1000)
```

**Effect:**
- GP learns appropriate $\sigma_n^2$ from data (via marginal likelihood)
- Allows function to not exactly hit every point
- Smooths over noise, captures signal

**Typical result:** $\sigma_n^2 \approx 1-5$ for weekly oil prices (depending on scale)

---

**Solution 2: Increase lengthscale $\ell$ (or use stronger prior)**

**Rationale:**
- Small $\ell$ allows rapid function changes → overfitting
- Larger $\ell$ enforces smoothness → regularization

**Implementation:**
```python
# BEFORE (overfits):
ℓ ~ InverseGamma(1, 1)  # Median ~1, allows very small values

# AFTER (regularized):
ℓ ~ InverseGamma(5, 50)  # Median ~10, discourages tiny lengthscales
# Or:
ℓ ~ Normal(20, 10)  # Informative prior centered at 20 weeks
```

**Effect:**
- Prevents GP from fitting high-frequency noise
- Encourages smoother functions

**Trade-off:**
- Too large $\ell$ → underfitting (misses real short-term dynamics)
- Need to balance via cross-validation

---

**Solution 3 (Bonus): Use Matérn $\nu=3/2$ instead of RBF**

**Rationale:**
- RBF produces infinitely smooth functions (every derivative exists)
- Real commodity prices have kinks, jumps (not infinitely smooth)
- Matérn $\nu=3/2$ allows less smooth functions, less prone to overfitting high-frequency noise

**Implementation:**
```python
cov_func = σ² * pm.gp.cov.Matern32(1, ls=ℓ)  # Matérn ν=3/2
```

---

**(c)** Validation strategy:

**Method 1: Cross-validation**

**K-fold time-series CV:**
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fit GP with each solution
    gp.fit(X_train, y_train)
    y_pred = gp.predict(X_test)

    mse = np.mean((y_test - y_pred)**2)
    mse_scores.append(mse)

print(f"Mean MSE: {np.mean(mse_scores):.3f}")
```

**Compare:**
- Solution 1 (estimate $\sigma_n^2$): MSE = ?
- Solution 2 (larger $\ell$): MSE = ?
- Choose solution with lower out-of-sample MSE

---

**Method 2: Marginal likelihood (model evidence)**

Compare $\log p(\mathbf{y} | X, \theta)$ for each model:

**Marginal likelihood:**
$$\log p(\mathbf{y} | X) = -\frac{1}{2} \mathbf{y}^T K_y^{-1} \mathbf{y} - \frac{1}{2} \log |K_y| - \frac{n}{2} \log(2\pi)$$

**Interpretation:**
- **Data fit term:** $-\frac{1}{2} \mathbf{y}^T K_y^{-1} \mathbf{y}$ (lower is better)
- **Complexity penalty:** $-\frac{1}{2} \log |K_y|$ (penalizes complex models)
- Automatically balances fit and complexity (Occam's razor)

**Compare:**
- Model with estimated $\sigma_n^2$: Marginal likelihood = ?
- Model with fixed large $\ell$: Marginal likelihood = ?
- **Higher marginal likelihood is better**

**PyMC extracts this automatically during sampling** (used for WAIC/LOO).

---

**Best practice:**
- Use **both** cross-validation (predictive performance) and marginal likelihood (model evidence)
- If they agree → confident in solution
- If they disagree → investigate further (maybe both solutions improve different aspects)

**Scoring:**
- Part (a): 4 points (overfitting diagnosis)
- Part (b): 4 points (two viable solutions)
- Part (c): 2 points (validation strategy)

---

### Question 6 (10 points)
Implement a GP forecasting model in PyMC for natural gas prices with the composite kernel from Question 3 (trend + seasonal + short-term). Provide:
- Complete PyMC model code
- How to generate forecasts 30 days ahead
- How to compute prediction intervals

**Answer:**

```python
import pymc as pm
import numpy as np
import pandas as pd
import arviz as az

# =============================
# DATA PREPARATION
# =============================

# Assume: Daily natural gas prices (2 years = 730 days)
# dates: pd.DatetimeIndex
# prices: np.array (shape: 730,)

# Normalize time to [0, 1] range (helps with numerical stability)
t = np.arange(len(prices))
t_normalized = (t - t.mean()) / t.std()

# Standardize prices (zero mean, unit variance)
price_mean = prices.mean()
price_std = prices.std()
prices_standardized = (prices - price_mean) / price_std

X_train = t_normalized[:, None]  # Shape: (730, 1)
y_train = prices_standardized

# =============================
# GP MODEL WITH COMPOSITE KERNEL
# =============================

with pm.Model() as gp_composite:

    # -------------------------
    # HYPERPARAMETERS
    # -------------------------

    # Trend component
    σ²_trend = pm.HalfNormal('sigma2_trend', sigma=1.0)
    ℓ_trend = pm.InverseGamma('ell_trend', alpha=2, beta=200/t.std())  # ~200 days

    # Seasonal component
    σ²_seasonal = pm.HalfNormal('sigma2_seasonal', sigma=0.5)
    ℓ_seasonal = pm.InverseGamma('ell_seasonal', alpha=2, beta=30/t.std())  # ~30 days
    period = 365 / t.std()  # Annual period in normalized units

    # Short-term component
    σ²_short = pm.HalfNormal('sigma2_short', sigma=0.3)
    ℓ_short = pm.InverseGamma('ell_short', alpha=3, beta=10/t.std())  # ~10 days

    # Observation noise
    σ²_obs = pm.HalfNormal('sigma2_obs', sigma=0.2)

    # -------------------------
    # KERNEL CONSTRUCTION
    # -------------------------

    # Trend kernel (RBF / Squared Exponential)
    cov_trend = σ²_trend * pm.gp.cov.ExpQuad(input_dim=1, ls=ℓ_trend)

    # Seasonal kernel (Periodic)
    cov_seasonal = σ²_seasonal * pm.gp.cov.Periodic(input_dim=1, period=period, ls=ℓ_seasonal)

    # Short-term kernel (RBF)
    cov_short = σ²_short * pm.gp.cov.ExpQuad(input_dim=1, ls=ℓ_short)

    # Composite kernel (additive)
    cov_total = cov_trend + cov_seasonal + cov_short

    # -------------------------
    # GP SPECIFICATION
    # -------------------------

    # Marginal GP (efficient for moderate-sized datasets)
    gp = pm.gp.Marginal(cov_func=cov_total)

    # Likelihood
    y_ = gp.marginal_likelihood('y', X=X_train, y=y_train, noise=σ²_obs)

    # -------------------------
    # INFERENCE
    # -------------------------

    # Sample posterior
    trace = pm.sample(
        draws=1000,
        tune=1000,
        return_inferencedata=True,
        target_accept=0.95,  # High for complex GP
        init='adapt_diag'  # Helps with GP convergence
    )

# =============================
# DIAGNOSTICS
# =============================

# Check convergence
print(az.summary(trace, var_names=['sigma2_trend', 'sigma2_seasonal', 'sigma2_short',
                                    'ell_trend', 'ell_seasonal', 'ell_short', 'sigma2_obs']))

# Effective sample size should be > 400
# R-hat should be < 1.01

# Plot traces
az.plot_trace(trace, var_names=['sigma2_trend', 'ell_trend'])
plt.tight_layout()
plt.show()

# =============================
# FORECASTING 30 DAYS AHEAD
# =============================

# New time points (next 30 days)
t_forecast = np.arange(len(prices), len(prices) + 30)
t_forecast_normalized = (t_forecast - t.mean()) / t.std()
X_forecast = t_forecast_normalized[:, None]  # Shape: (30, 1)

# Conditional distribution: f* | y, X, X*
with gp_composite:
    # Conditional GP (given training data)
    f_forecast = gp.conditional('f_forecast', X_forecast)

    # Sample from posterior predictive
    # This includes observation noise σ²_obs
    ppc = pm.sample_posterior_predictive(
        trace,
        var_names=['f_forecast'],
        return_inferencedata=True,
        extend_inferencedata=True
    )

# =============================
# EXTRACT PREDICTIONS
# =============================

# Posterior samples: shape (chains, draws, 30)
f_forecast_samples = ppc.posterior_predictive['f_forecast'].values

# Compute statistics
f_mean = f_forecast_samples.mean(axis=(0, 1))  # Shape: (30,)
f_std = f_forecast_samples.std(axis=(0, 1))

# 95% Highest Density Interval (HDI)
f_hdi = az.hdi(ppc.posterior_predictive, var_names=['f_forecast'], hdi_prob=0.95)
f_lower = f_hdi['f_forecast'].sel(hdi='lower').values
f_upper = f_hdi['f_forecast'].sel(hdi='higher').values

# Unstandardize predictions (back to original price scale)
f_mean_price = f_mean * price_std + price_mean
f_lower_price = f_lower * price_std + price_mean
f_upper_price = f_upper * price_std + price_mean

# =============================
# VISUALIZATION
# =============================

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 7))

# Historical data
ax.plot(t, prices, 'o', alpha=0.5, label='Observed prices', markersize=3)

# Forecasts
forecast_dates = t_forecast
ax.plot(forecast_dates, f_mean_price, 'r-', linewidth=2, label='Forecast (mean)')
ax.fill_between(forecast_dates, f_lower_price, f_upper_price,
                alpha=0.3, color='red', label='95% Prediction Interval')

# Formatting
ax.axvline(x=len(prices)-1, color='k', linestyle='--', label='Forecast start')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Natural Gas Price ($/MMBtu)')
ax.set_title('GP Forecast: Natural Gas Prices (30 days ahead)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================
# UNCERTAINTY DECOMPOSITION
# =============================

# Compute variance contributions (approximately)
# Posterior means of variance components
σ²_trend_mean = trace.posterior['sigma2_trend'].mean().values
σ²_seasonal_mean = trace.posterior['sigma2_seasonal'].mean().values
σ²_short_mean = trace.posterior['sigma2_short'].mean().values
σ²_obs_mean = trace.posterior['sigma2_obs'].mean().values

print("\nVariance Decomposition (relative contributions):")
total_var = σ²_trend_mean + σ²_seasonal_mean + σ²_short_mean + σ²_obs_mean
print(f"Trend:     {σ²_trend_mean / total_var:.1%}")
print(f"Seasonal:  {σ²_seasonal_mean / total_var:.1%}")
print(f"Short-term: {σ²_short_mean / total_var:.1%}")
print(f"Observation noise: {σ²_obs_mean / total_var:.1%}")

# =============================
# SAVE FORECASTS
# =============================

forecasts_df = pd.DataFrame({
    'day': forecast_dates,
    'mean': f_mean_price,
    'lower_95': f_lower_price,
    'upper_95': f_upper_price
})

forecasts_df.to_csv('ng_forecasts_30day.csv', index=False)
print("\nForecasts saved to ng_forecasts_30day.csv")
```

**Key implementation notes:**

1. **Normalization:** Time and prices standardized for numerical stability
2. **Marginal GP:** Efficient for datasets < 10,000 points
3. **Composite kernel:** Additive structure (trend + seasonal + short-term)
4. **Prediction intervals:** HDI computed from posterior predictive samples (includes all sources of uncertainty)
5. **Conditional GP:** `gp.conditional()` computes $p(f^* | y, X, X^*)$

**Scoring:**
- Complete PyMC model: 5 points
- Forecasting implementation: 3 points
- Prediction intervals: 2 points

---

### Question 8 (8 points)
**Sparse GP approximations:** For a commodity with 10 years of daily data (3650 observations), exact GP inference requires inverting a $3650 \times 3650$ matrix (computational complexity $O(n^3)$).

**(a)** Explain why this is problematic for large datasets. (2 points)

**(b)** Describe the **inducing points** approach for sparse GP approximation. (4 points)

**(c)** How do you choose the number and location of inducing points? (2 points)

**Answer:**

**(a)** Computational problems with exact GP:

**Computational complexity:**
$$O(n^3) \text{ for matrix inversion} + O(n^2) \text{ for storage}$$

For $n = 3650$:
- Matrix inversion: ~$50 \text{ billion operations}$
- Storage: $3650^2 \times 8 \text{ bytes} \approx 100 \text{ MB}$ (just for covariance matrix)
- Per MCMC iteration: Must recompute $K_y^{-1}$ or use expensive Cholesky updates

**Practical impact:**
- **Slow:** Minutes to hours per MCMC sample
- **Memory:** Large datasets won't fit in RAM
- **Scalability:** Infeasible for $n > 10,000$

**Why prohibitive:**
- MCMC needs 1000-5000 samples
- Each sample: $O(n^3)$ operations
- Total: Days to weeks of computation

---

**(b)** Inducing points sparse approximation:

**Idea:** Approximate GP using a smaller set of $m \ll n$ **inducing points** (also called pseudo-inputs or support points).

**Key insight:** Most information in GP comes from "representative" points, not all $n$ observations.

**Mathematical formulation:**

**Full GP (exact):**
$$p(\mathbf{f} | X) = \mathcal{N}(\mathbf{0}, K)$$

**Sparse GP (approximate):**
Introduce $m$ inducing points at locations $Z = \{z_1, ..., z_m\}$ with function values $\mathbf{u} = [f(z_1), ..., f(z_m)]'$.

**Variational approximation:**
$$p(\mathbf{f} | X, \mathbf{y}) \approx q(\mathbf{f} | X) = \int p(\mathbf{f} | \mathbf{u}, X, Z) q(\mathbf{u}) d\mathbf{u}$$

where $q(\mathbf{u})$ is a variational distribution (typically Gaussian).

**Computational complexity:**
$$O(nm^2) \text{ instead of } O(n^3)$$

For $n=3650$, $m=100$:
- Exact: $O(3650^3) \approx 49 \text{ billion operations}$
- Sparse: $O(3650 \times 100^2) \approx 36 \text{ million operations}$
- **Speedup: ~1000x faster!**

**PyMC implementation:**
```python
# Inducing points (m << n)
m = 100  # Number of inducing points
Z = np.linspace(X_train.min(), X_train.max(), m)[:, None]  # Evenly spaced

with pm.Model():
    # Kernel hyperparameters
    ℓ = pm.InverseGamma('ell', alpha=2, beta=20)
    σ² = pm.HalfNormal('sigma2', sigma=10)
    σ²_obs = pm.HalfNormal('sigma2_obs', sigma=2)

    # Covariance function
    cov = σ² * pm.gp.cov.ExpQuad(1, ls=ℓ)

    # Sparse GP using FITC approximation
    gp = pm.gp.MarginalSparse(cov_func=cov, approx='FITC')

    # Likelihood (uses inducing points Z)
    y_ = gp.marginal_likelihood('y', X=X_train, Xu=Z, y=y_train, noise=σ²_obs)

    trace = pm.sample(1000)  # Much faster than full GP!
```

**Approximation quality:**
- Good if $m$ large enough to capture function complexity
- Worse in regions far from inducing points
- Trade-off: $m$ vs accuracy vs speed

---

**(c)** Choosing inducing points:

**Number of inducing points $m$:**

**Heuristic:**
$$m \approx \sqrt{n} \text{ to } n^{2/3}$$

For $n = 3650$:
$$m \in [60, 240]$$

**Practical guidelines:**
- Start with $m = 100$ (often sufficient)
- Increase if posterior predictive checks show poor fit
- Decrease if computational budget is tight

**Cross-validation:**
- Try $m \in \{50, 100, 200, 500\}$
- Measure test MSE
- Diminishing returns beyond certain $m$

---

**Location of inducing points $Z$:**

**Option 1: Evenly spaced (simplest)**
```python
Z = np.linspace(X.min(), X.max(), m)[:, None]
```

- Uniform coverage of input space
- Good default

**Option 2: Quantiles of data distribution**
```python
Z = np.quantile(X, q=np.linspace(0, 1, m))[:, None]
```

- More points where data is dense
- Better if data is non-uniformly distributed

**Option 3: K-means clustering**
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=m)
Z = kmeans.fit(X).cluster_centers_
```

- Adaptive to data structure
- Computationally more expensive

**Option 4: Learn inducing point locations** (advanced)
- Treat $Z$ as variational parameters
- Optimize jointly with hyperparameters
- PyMC supports this with `approx='VFE'` (Variational Free Energy)

---

**Best practice:**
- Start with **evenly spaced** (Option 1)
- If poor fit in specific regions, add more inducing points there
- For time series: Ensure coverage of both recent and historical data

**Diagnostic:**
- Plot inducing points on top of data to visualize coverage
- Check residuals: Large residuals indicate need for more inducing points nearby

**Scoring:**
- Part (a): 2 points (computational bottleneck)
- Part (b): 4 points (inducing points concept + complexity reduction)
- Part (c): 2 points (choosing m and Z)

---

## Section C: Advanced GP Topics (30 points)

### Question 9 (15 points)
**Multi-output GPs for commodity portfolios:** You want to jointly model crude oil, natural gas, and gasoline prices using a multi-output GP. These commodities are correlated but have different dynamics.

**(a)** Explain the concept of a multi-output GP coregionalization model. (5 points)

**(b)** Write the kernel structure for a model with:
- Shared temporal correlation (all commodities move together to some degree)
- Commodity-specific dynamics
- Cross-commodity correlations

(6 points)

**(c)** What is the advantage of joint modeling over separate GPs for each commodity? (4 points)

**Answer:**

**(a)** Multi-output GP coregionalization:

**Problem:** Model $p$ output dimensions jointly (here: $p=3$ commodities)

**Standard GP:** Single output $f(x) \in \mathbb{R}$

**Multi-output GP:** Vector-valued output $\mathbf{f}(x) = [f_1(x), ..., f_p(x)]' \in \mathbb{R}^p$

**Coregionalization:** Model cross-output correlations using a **matrix-valued** kernel:
$$k(\mathbf{x}, \mathbf{x}') \in \mathbb{R}^{p \times p}$$

**Key idea:** Decompose kernel into:
1. **Temporal correlation:** How outputs at different times relate
2. **Cross-output correlation:** How different outputs (commodities) relate

**Linear Model of Coregionalization (LMC):**

Express multi-output as weighted sum of latent GPs:
$$f_j(x) = \sum_{q=1}^Q w_{jq} u_q(x)$$

where:
- $u_q(x) \sim \mathcal{GP}(0, k_q(x, x'))$: Latent GPs (shared across outputs)
- $w_{jq}$: Weights (how much output $j$ depends on latent GP $q$)

**Resulting covariance:**
$$\text{Cov}(f_i(x), f_j(x')) = \sum_{q=1}^Q w_{iq} w_{jq} k_q(x, x')$$

**Interpretation:**
- Each latent $u_q$ captures a common temporal pattern
- Weights $w_{jq}$ determine which outputs share which patterns
- Allows flexible cross-correlation structure

---

**(b)** Kernel structure for oil/gas/gasoline:

**Model specification:**

Let outputs be:
- $f_1(t)$: Crude oil price
- $f_2(t)$: Natural gas price
- $f_3(t)$: Gasoline price

**Kernel design:**

**Component 1: Shared temporal kernel (common trends)**
$$k_{\text{shared}}(t, t') = K_{\text{shared}}(t, t') \otimes B_{\text{shared}}$$

where:
- $K_{\text{shared}}(t, t') = \sigma_{\text{shared}}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{\text{shared}}^2}\right)$: Temporal correlation (RBF)
- $B_{\text{shared}} = \begin{bmatrix} 1 & \rho_{12} & \rho_{13} \\ \rho_{12} & 1 & \rho_{23} \\ \rho_{13} & \rho_{23} & 1 \end{bmatrix}$: Cross-commodity correlation matrix

**Component 2: Commodity-specific kernels (individual dynamics)**
$$k_{\text{individual}}(t, t') = \begin{bmatrix}
k_1(t, t') & 0 & 0 \\
0 & k_2(t, t') & 0 \\
0 & 0 & k_3(t, t')
\end{bmatrix}$$

where each $k_j(t, t')$ is an RBF kernel with commodity-specific hyperparameters:
$$k_j(t, t') = \sigma_j^2 \exp\left(-\frac{(t-t')^2}{2\ell_j^2}\right)$$

**Total kernel (additive):**
$$k_{\text{total}}(t, t') = k_{\text{shared}}(t, t') + k_{\text{individual}}(t, t')$$

**Expanded form:**

For outputs $i, j$ at times $t, t'$:
$$\begin{aligned}
\text{Cov}(f_i(t), f_j(t')) &= K_{\text{shared}}(t, t') \times B_{ij} + K_j(t, t') \times \mathbb{1}_{i=j}
\end{aligned}$$

**Specific example:**

$$\text{Cov}(f_{\text{oil}}(t), f_{\text{gas}}(t')) = \underbrace{\sigma_{\text{shared}}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{\text{shared}}^2}\right) \times \rho_{\text{oil,gas}}}_{\text{Shared component (cross-correlation)}}$$

$$\text{Cov}(f_{\text{oil}}(t), f_{\text{oil}}(t')) = \underbrace{\sigma_{\text{shared}}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{\text{shared}}^2}\right)}_{\text{Shared}} + \underbrace{\sigma_{\text{oil}}^2 \exp\left(-\frac{(t-t')^2}{2\ell_{\text{oil}}^2}\right)}_{\text{Oil-specific}}$$

**Hyperparameters to estimate:**
- Shared: $\sigma_{\text{shared}}^2, \ell_{\text{shared}}$
- Individual: $\sigma_j^2, \ell_j$ for $j=1,2,3$
- Cross-correlations: $\rho_{12}, \rho_{13}, \rho_{23}$ (or full $B$ matrix)

**PyMC sketch:**
```python
with pm.Model():
    # Shared kernel
    σ²_shared = pm.HalfNormal('sigma2_shared', sigma=5)
    ℓ_shared = pm.InverseGamma('ell_shared', alpha=2, beta=30)
    cov_shared = σ²_shared * pm.gp.cov.ExpQuad(1, ls=ℓ_shared)

    # Cross-correlation matrix (LKJ prior for positive definiteness)
    B_corr = pm.LKJCorr('B_corr', n=3, eta=2)  # η=2: weakly informative

    # Individual kernels
    σ²_oil = pm.HalfNormal('sigma2_oil', sigma=5)
    σ²_gas = pm.HalfNormal('sigma2_gas', sigma=5)
    σ²_gasoline = pm.HalfNormal('sigma2_gasoline', sigma=5)

    ℓ_oil = pm.InverseGamma('ell_oil', alpha=2, beta=20)
    ℓ_gas = pm.InverseGamma('ell_gas', alpha=2, beta=20)
    ℓ_gasoline = pm.InverseGamma('ell_gasoline', alpha=2, beta=20)

    # (Full implementation would use pm.gp.LatentKron or custom kernel)
```

**Scoring:** 6 points (shared kernel + individual kernels + cross-correlation structure)

---

**(c)** Advantages of joint modeling:

**1. Borrowing strength across commodities**

**Mechanism:**
- Shared kernel pools information: Oil prices inform gas prices (and vice versa)
- Especially valuable when:
  - One commodity has missing data (others fill gaps)
  - One commodity has short history (learns from others)

**Example:**
- Gasoline has sparse data in early period
- Joint model uses oil + gas correlations to infer gasoline values
- Separate models would have high uncertainty for gasoline

**Statistical:** Variance reduction via correlation:
$$\text{Var}(f_{\text{gas}} | f_{\text{oil}}) < \text{Var}(f_{\text{gas}})$$

---

**2. Consistent multi-commodity forecasts**

**Problem with separate models:**
- Oil forecast: ↑ (up)
- Gasoline forecast: ↓ (down)
- **Implausible:** Oil and gasoline are co-products (positive correlation)

**Joint model:**
- Respects cross-commodity correlations
- Forecasts move together (weighted by $\rho_{ij}$)
- **Economically coherent** predictions

**Example:**
- If oil forecast is ↑ 10%
- Joint model predicts gasoline ↑ 8% (via $\rho_{\text{oil,gas}} \approx 0.8$)
- Natural gas ↑ 3% (via $\rho_{\text{oil,gas}} \approx 0.3$)

---

**3. Estimate cross-commodity correlations**

**Direct benefit:**
- Posterior distribution of $\rho_{ij}$ quantifies dependencies
- Use for portfolio risk management, hedging strategies

**Example question answered:**
- "How correlated are oil and gas price shocks?"
- Joint model: $\rho_{\text{oil,gas}} \sim \mathcal{N}(0.35, 0.05^2)$ (posterior)
- Separate models: Cannot estimate this

---

**4. Computational efficiency (sometimes)**

**Paradoxical benefit:**
- Joint model is actually **cheaper** than fitting 3 separate GPs
- **Why:** Shared temporal structure computed once, reused for all outputs
- Especially true with inducing points: $m$ inducing points serve all outputs

**Complexity:**
- Separate GPs: $3 \times O(n^3) = O(3n^3)$
- Joint GP with shared kernel: $O(n^3) + O(p^2 n)$ where $p=3$
- If $p \ll n$: Joint is faster

---

**5. Model comparison and selection**

**Question:** "Do commodities share temporal patterns or are they independent?"

**Answer via model comparison:**
- Fit joint model with shared kernel
- Fit joint model with only individual kernels (no $B_{\text{shared}}$)
- Compare WAIC/LOO

**If shared kernel improves fit → Evidence for common drivers**

---

**Summary table:**

| Aspect | Separate GPs | Joint Multi-Output GP |
|--------|-------------|----------------------|
| Data efficiency | No sharing | Borrows strength |
| Forecast consistency | No constraints | Respects correlations |
| Cross-correlations | Not estimated | Directly estimated |
| Computational cost | $O(3n^3)$ | $O(n^3)$ (shared kernel) |
| Interpretability | Simple | Requires understanding coregionalization |

**When to use joint modeling:**
- Commodities are correlated ($\rho > 0.3$)
- Portfolio/multi-commodity forecasting
- Some commodities have limited data

**When separate is okay:**
- Commodities truly independent (gold vs corn)
- Single-commodity focus
- Simpler interpretation needed

**Scoring:**
- Borrowing strength: 1 point
- Forecast consistency: 1 point
- Correlation estimation: 1 point
- Computational efficiency or model comparison: 1 point

---

### Question 10 (15 points)
**GP model criticism:** After fitting a GP with RBF kernel to weekly corn prices, you perform the following diagnostics:

1. **Residual plot:** Residuals show clear autocorrelation (Ljung-Box test $p < 0.001$)
2. **Posterior predictive check:** Observed data has more extreme values than posterior predictive samples
3. **Marginal likelihood:** Very sensitive to lengthscale prior (changes by 10% with small prior changes)

**(a)** Interpret each diagnostic. What does it suggest about the model? (6 points)

**(b)** For each issue, propose a model modification. (6 points)

**(c)** Explain how you would validate that your modifications improved the model. (3 points)

**Answer:**

**(a)** Diagnostic interpretations:

**Diagnostic 1: Autocorrelated residuals**

**What it means:**
- Residuals: $r_t = y_t - \mathbb{E}[y_t | y_1, ..., y_{t-1}, X]$
- Autocorrelation: $\text{Corr}(r_t, r_{t-k}) \neq 0$ for lag $k$
- Ljung-Box test: Rejects null of no autocorrelation ($p < 0.001$)

**Implication:**
- **Model has not captured all temporal structure**
- Leftover predictable patterns in residuals
- GP kernel is **misspecified**

**Possible causes:**
- **Missing dynamics:** RBF assumes smooth, stationary process; corn may have:
  - Seasonal patterns not captured
  - AR-type dependencies (current price depends on lags)
  - Trend changes
- **Lengthscale too short:** Not capturing medium-term correlations

**Action:** Need to enrich kernel structure

---

**Diagnostic 2: Extreme values more frequent than predicted**

**What it means:**
- Observed: Occasional large price spikes/drops (>3 SD)
- Posterior predictive: Rarely generates such extremes
- **Fat tails:** Real data has heavier tails than Gaussian

**Implication:**
- **Gaussian observation model is inadequate**
- Corn prices have:
  - Jumps (weather shocks, policy announcements)
  - Volatility clustering (high volatility persists)
  - Non-Gaussian noise

**Evidence of:**
- **Outliers or structural breaks**
- **Heteroskedasticity** (time-varying variance)

**Action:** Need robust likelihood or time-varying noise model

---

**Diagnostic 3: Marginal likelihood sensitive to lengthscale prior**

**What it means:**
- Marginal likelihood: $p(\mathbf{y} | X) = \int p(\mathbf{y} | X, \theta) p(\theta) d\theta$
- High sensitivity to prior on $\ell$: Posterior is **not well-identified** by data

**Implication:**
- **Data does not strongly constrain lengthscale**
- Possible causes:
  1. **Weak signal:** Data too noisy to determine $\ell$ precisely
  2. **Multiple plausible lengthscales:** Data consistent with wide range of $\ell$
  3. **Non-stationary:** True lengthscale changes over time, so single $\ell$ is misspecified

**Practical problem:**
- Forecasts depend heavily on prior choice
- Different analysts with different priors → different forecasts
- **Lack of robustness**

**Action:** Need more informative prior (domain knowledge) or richer model (time-varying $\ell$)

---

**(b)** Model modifications:

**Modification 1: Add seasonal component to kernel**

**Problem:** Autocorrelated residuals likely due to missing seasonality (corn has harvest cycle)

**Solution:** Use composite kernel with periodic component:
$$k_{\text{total}}(t, t') = k_{\text{trend}}(t, t') + k_{\text{seasonal}}(t, t')$$

where:
$$k_{\text{seasonal}}(t, t') = \sigma_s^2 \exp\left(-\frac{2\sin^2(\pi|t-t'|/52)}{\ ell_s^2}\right)$$

Period = 52 weeks (annual).

**Expected improvement:**
- Captures annual harvest pattern
- Residuals no longer autocorrelated (seasonal structure extracted)

---

**Modification 2: Use Student-t observation likelihood (robust to outliers)**

**Problem:** Fat tails in data, Gaussian likelihood inadequate

**Solution:** Replace Gaussian with Student-t:
$$y_t = f(t) + \epsilon_t, \quad \epsilon_t \sim t_\nu(0, \sigma^2)$$

where $\nu$ = degrees of freedom (estimate from data).

**PyMC:**
```python
with pm.Model():
    # ... GP specification ...

    # Robust observation model
    ν = pm.Exponential('nu', lam=1/10)  # Prior on tail heaviness
    σ²_obs = pm.HalfNormal('sigma2_obs', sigma=5)

    # Latent GP function values
    f = gp.prior('f', X=X_train)

    # Student-t likelihood (instead of Normal)
    y_ = pm.StudentT('y', nu=ν, mu=f, sigma=σ²_obs, observed=y_train)
```

**Expected improvement:**
- Extreme values (price spikes) automatically downweighted
- Posterior predictive will generate occasional extremes (matching data)
- More robust parameter estimates (not distorted by outliers)

---

**Modification 3: Use informative prior on lengthscale**

**Problem:** Marginal likelihood sensitive to $\ell$ prior, suggesting weak identification

**Solution:** Encode domain knowledge via informative prior

**Approach:**
1. **Expert elicitation:** Ask domain expert typical timescale of corn price persistence
   - "Prices usually revert to equilibrium within 2-4 weeks"
   - Implies $\ell \approx 10-20$ weeks

2. **Set informative prior:**
```python
ℓ ~ Normal(15, 5)  # Centered at 15 weeks, SD=5
# Or:
ℓ ~ Gamma(α, β)  # Choose α, β such that mean=15, std=5
```

3. **Sensitivity analysis:** Check forecasts with $\ell \in [10, 20]$
   - If forecasts stable → robustness achieved

**Expected improvement:**
- Reduced prior-data conflict
- More stable marginal likelihood
- Forecasts less sensitive to prior choice

---

**Alternative Modification 3: Time-varying lengthscale**

If data genuinely has changing dynamics (e.g., volatility regimes):
$$\ell_t = \ell_0 + \sum_{k=1}^K \alpha_k \mathbb{1}_{t \in \text{Regime } k}$$

Or use non-stationary kernel:
$$k(t, t') = \sigma^2 \exp\left(-\frac{(t-t')^2}{2\ell(t)\ell(t')}\right)$$

where $\ell(t)$ evolves over time.

---

**(c)** Validation strategy:

**1. Re-run diagnostics on modified model:**

**Residual autocorrelation:**
- Compute residuals from modified model
- Ljung-Box test: $p > 0.05$ (no autocorrelation) → SUCCESS
- ACF plot: Should show no significant lags

**Posterior predictive check:**
- Generate posterior predictive samples from modified (Student-t) model
- Check empirical quantiles:
  - Observed 1%/99% quantiles: e.g., [-15, +20]
  - Predicted 1%/99% quantiles: Should match!
- Visual: `az.plot_ppc()` should show good overlap

**Marginal likelihood sensitivity:**
- Fit model with two priors:
  - Prior 1: $\ell \sim \text{Normal}(15, 5)$
  - Prior 2: $\ell \sim \text{Normal}(15, 8)$ (wider)
- Compare marginal likelihoods: Should be similar (< 5% difference)
- If stable → robustness achieved

---

**2. Out-of-sample forecast evaluation:**

**Rolling forecast:**
```python
# Hold out last 52 weeks (1 year) as test set
X_train, X_test = X[:-52], X[-52:]
y_train, y_test = y[:-52], y[-52:]

# Fit modified model
# ... (composite kernel + Student-t)

# Forecast 1-week ahead at each time in test set
predictions = []
for t in range(len(X_test)):
    X_train_t = np.vstack([X_train, X_test[:t]])
    y_train_t = np.hstack([y_train, y_test[:t]])

    # Refit and forecast (or use online updates)
    gp.fit(X_train_t, y_train_t)
    pred_t = gp.predict(X_test[t:t+1])
    predictions.append(pred_t)

# Evaluate
mse_modified = np.mean((y_test - predictions)**2)
mae_modified = np.mean(np.abs(y_test - predictions))

# Compare to baseline (original RBF model)
# Should see improvement: lower MSE/MAE
```

**Metrics:**
- MSE (mean squared error): Should decrease
- MAE (mean absolute error): Should decrease
- Coverage: % of observations within 95% prediction interval
  - Should be close to 95% (well-calibrated)

---

**3. Model comparison (WAIC/LOO):**

```python
import arviz as az

# Original model
waic_original = az.waic(trace_original)

# Modified model
waic_modified = az.waic(trace_modified)

# Compare
az.compare({'Original': trace_original, 'Modified': trace_modified})
# Lower WAIC = better
```

**Interpretation:**
- If $\Delta \text{WAIC} > 10$: Strong evidence for modified model
- If $\Delta \text{WAIC} < 5$: Weak evidence, models comparable

---

**Summary validation checklist:**

- [ ] Residuals no longer autocorrelated
- [ ] Posterior predictive matches observed extremes
- [ ] Marginal likelihood stable across reasonable priors
- [ ] Lower out-of-sample forecast error (MSE/MAE)
- [ ] Better WAIC/LOO score
- [ ] Prediction intervals well-calibrated (95% coverage ≈ 95%)

**If all checks pass → Model modifications successful!**

**Scoring:**
- Part (a): 6 points (2 per diagnostic interpretation)
- Part (b): 6 points (2 per modification)
- Part (c): 3 points (validation strategy)

---

## Answer Key Summary

| Question | Points | Topic |
|----------|--------|-------|
| 1 | 10 | GP definition, components |
| 2 | 8 | Lengthscale interpretation |
| 3 | 12 | Composite kernel design |
| 4 | 12 | GP posterior predictive derivation |
| 5 | 10 | Overfitting diagnosis |
| 6 | 10 | PyMC GP implementation |
| 7 | (skipped for brevity) | |
| 8 | 8 | Sparse GP approximations |
| 9 | 15 | Multi-output GPs |
| 10 | 15 | Model criticism and validation |
| **Total** | **100** | |

---

## Grading Rubric

**A (90-100):** Mastery of GP theory, kernel design, and practical implementation. Can derive posterior distributions and design custom kernels for commodity applications.

**B (80-89):** Solid understanding with minor gaps. May struggle with mathematical derivations or advanced topics (sparse GPs, multi-output).

**C (70-79):** Basic GP competency but lacks depth in kernel engineering or model criticism.

**D (60-69):** Significant conceptual gaps. Cannot implement GPs correctly or interpret diagnostics.

**F (<60):** Does not meet minimum standards for graduate-level GP modeling.

---

**Study Resources:**
- Rasmussen & Williams (2006): *Gaussian Processes for Machine Learning* (Free PDF)
- PyMC GP Documentation: https://docs.pymc.io/gp.html
- Module notebooks: `02_kernel_design.ipynb`, `03_gp_forecasting.ipynb`
- Duvenaud (2014): *Automatic Model Construction with Gaussian Processes* (Thesis)

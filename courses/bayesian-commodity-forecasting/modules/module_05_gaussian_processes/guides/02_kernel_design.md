# Kernel Design for Commodity Time Series

> **Reading time:** ~10 min | **Module:** 5 — Gaussian Processes | **Prerequisites:** Module 4 Hierarchical Models


## In Brief

The kernel function defines what "similarity" means in Gaussian Process models. Commodity time series exhibit trends, seasonality, cycles, and discontinuities—each requiring specific kernel choices. This guide provides a practical framework for designing and combining kernels for commodity forecasting.

<div class="callout-insight">

<strong>Insight:</strong> **Kernels encode assumptions about function smoothness and structure.** A smooth exponential kernel assumes gradual price changes (fine for stable markets), while a Matérn-1/2 kernel allows sudden jumps (better for supply shocks). Choosing the wrong kernel is like wearing the wrong glasses—everything is blurry.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> The kernel function defines what "similarity" means in Gaussian Process models.

</div>

---

## Formal Definition

### Kernel Function

A kernel $k: \mathcal{X} \times \mathcal{X} \to \mathbb{R}$ is a positive definite function that defines covariance between function values:

$$\text{Cov}[f(x_i), f(x_j)] = k(x_i, x_j)$$

For a GP prior $f \sim \mathcal{GP}(m, k)$, any finite collection of function values is multivariate normal:

$$\begin{bmatrix} f(x_1) \\ \vdots \\ f(x_n) \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} m(x_1) \\ \vdots \\ m(x_n) \end{bmatrix}, \begin{bmatrix} k(x_1, x_1) & \cdots & k(x_1, x_n) \\ \vdots & \ddots & \vdots \\ k(x_n, x_1) & \cdots & k(x_n, x_n) \end{bmatrix} \right)$$

---

## Core Kernel Building Blocks

### 1. Stationary Kernels (Time-Invariant)
<div class="callout-insight">

<strong>Insight:</strong> Depend only on distance: $k(x_i, x_j) = k(|x_i - x_j|)$

</div>


Depend only on distance: $k(x_i, x_j) = k(|x_i - x_j|)$

#### Squared Exponential (SE) / RBF / Gaussian

$$k_{\text{SE}}(x_i, x_j) = \sigma^2 \exp\left(-\frac{(x_i - x_j)^2}{2\ell^2}\right)$$

**Properties:**
- Infinitely differentiable (very smooth)
- Length scale $\ell$: controls how far influence extends
- Variance $\sigma^2$: vertical scale of variation

**Use case:** Slow-moving trends, equilibrium prices

**Problem:** Too smooth for commodities (no sharp moves)

---

#### Matérn Kernel

$$k_{\text{Matérn}}(x_i, x_j) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\sqrt{2\nu} \frac{|x_i - x_j|}{\ell}\right)^\nu K_\nu\left(\sqrt{2\nu} \frac{|x_i - x_j|}{\ell}\right)$$

**Smoothness parameter $\nu$:**
- $\nu = 1/2$: Once differentiable (rough, allows jumps) — **best for supply shocks**
- $\nu = 3/2$: Once differentiable
- $\nu = 5/2$: Twice differentiable — **good default for commodities**
- $\nu \to \infty$: Squared exponential (infinitely smooth)

**Matérn-5/2 (recommended default):**
$$k_{\text{M52}}(r) = \sigma^2 \left(1 + \sqrt{5}r + \frac{5r^2}{3}\right) \exp\left(-\sqrt{5}r\right), \quad r = \frac{|x_i - x_j|}{\ell}$$

**Use case:** General commodity forecasting with occasional sharp moves

---

#### Exponential Kernel (Matérn-1/2)

$$k_{\text{Exp}}(x_i, x_j) = \sigma^2 \exp\left(-\frac{|x_i - x_j|}{\ell}\right)$$

**Properties:**
- Continuous but not differentiable (rough paths)
- Allows sudden jumps (like supply disruptions)

**Use case:** Volatile commodities (natural gas in winter, agricultural with weather shocks)

---

### 2. Periodic Kernel

$$k_{\text{Per}}(x_i, x_j) = \sigma^2 \exp\left(-\frac{2\sin^2(\pi |x_i - x_j| / p)}{\ell^2}\right)$$

**Parameters:**
- $p$: period (e.g., 12 months, 52 weeks)
- $\ell$: controls decay within period

**Use case:**
- Natural gas: heating season (winter) / cooling season (summer)
- Agriculture: planting (spring) / harvest (fall)
- Electricity: daily/weekly demand cycles

---

### 3. Linear Kernel

$$k_{\text{Lin}}(x_i, x_j) = \sigma^2 (x_i - c)(x_j - c)$$

**Properties:**
- Produces linear trends
- Unbounded variance (grows with time)

**Use case:** Modeling long-term price inflation/deflation

---

### 4. Rational Quadratic (Scale Mixture)

$$k_{\text{RQ}}(x_i, x_j) = \sigma^2 \left(1 + \frac{(x_i - x_j)^2}{2\alpha \ell^2}\right)^{-\alpha}$$

**Parameter $\alpha$:**
- $\alpha \to \infty$: Squared exponential
- Small $\alpha$: Mixture of length scales (captures both short and long-term structure)

**Use case:** Markets with multiple time scales (intraday noise + monthly trends)

---

## Kernel Composition Rules

Kernels can be combined to create complex behaviors.

### Addition: $k(x, x') = k_1(x, x') + k_2(x, x')$

**Interpretation:** Function is sum of two independent processes.

**Example:** Trend + Seasonality
$$k_{\text{total}} = k_{\text{Matérn}}(\text{trend}) + k_{\text{Periodic}}(\text{seasonal})$$

---

### Multiplication: $k(x, x') = k_1(x, x') \times k_2(x, x')$

**Interpretation:** Modulation (one process scales the other).

**Example:** Seasonal amplitude varies over time
$$k_{\text{seasonal-varying}} = k_{\text{Matérn}}(\text{slow variation}) \times k_{\text{Periodic}}(\text{fast seasonality})$$

This allows winter peaks to change magnitude year-to-year.

---

## Commodity-Specific Kernel Designs

### 1. Crude Oil: Smooth Trend with Occasional Shocks


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as crude_model:
    # Data
    t = np.arange(len(oil_prices))  # Time index
    y = oil_prices - oil_prices.mean()  # Mean-center

    # Long-term trend (smooth Matérn-5/2)
    ell_trend = pm.InverseGamma('ell_trend', alpha=5, beta=50)  # Expect long length scale
    sigma_trend = pm.HalfNormal('sigma_trend', sigma=10)
    cov_trend = sigma_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)

    # Short-term volatility shocks (rough exponential)
    ell_shock = pm.InverseGamma('ell_shock', alpha=3, beta=5)  # Short length scale
    sigma_shock = pm.HalfNormal('sigma_shock', sigma=5)
    cov_shock = sigma_shock**2 * pm.gp.cov.Exponential(1, ls=ell_shock)

    # Combined kernel
    cov_total = cov_trend + cov_shock

    # GP
    gp = pm.gp.Marginal(cov_func=cov_total)
    sigma_noise = pm.HalfNormal('sigma_noise', sigma=2)
    y_obs = gp.marginal_likelihood('y_obs', X=t[:, None], y=y, noise=sigma_noise)

    trace = pm.sample(1000, tune=2000, return_inferencedata=True)
```

</div>
</div>

**Interpretation:**
- Trend captures slow OPEC decisions, global demand shifts
- Shock captures supply disruptions, geopolitical events
- Addition allows both to coexist

---

### 2. Natural Gas: Strong Seasonality with Time-Varying Amplitude


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with pm.Model() as gas_model:
    t = np.arange(len(gas_prices)) / 52  # Convert weeks to years
    y = gas_prices - gas_prices.mean()

    # Annual seasonality (fixed period)
    sigma_seasonal = pm.HalfNormal('sigma_seasonal', sigma=5)
    ell_seasonal = pm.InverseGamma('ell_seasonal', alpha=2, beta=2)
    cov_seasonal = sigma_seasonal**2 * pm.gp.cov.Periodic(1, period=1.0, ls=ell_seasonal)

    # Slow trend (modulates seasonal amplitude)
    sigma_trend = pm.HalfNormal('sigma_trend', sigma=3)
    ell_trend = pm.InverseGamma('ell_trend', alpha=5, beta=10)
    cov_trend = sigma_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)

    # Multiply to get time-varying seasonality
    cov_total = cov_trend * cov_seasonal + cov_trend  # Trend also acts alone

    gp = pm.gp.Marginal(cov_func=cov_total)
    sigma_noise = pm.HalfNormal('sigma_noise', sigma=1)
    y_obs = gp.marginal_likelihood('y_obs', X=t[:, None], y=y, noise=sigma_noise)

    trace = pm.sample(1000, tune=2000, return_inferencedata=True)
```

</div>
</div>

**Why multiplication?** Winter 2014 had $6/mmBtu swings; winter 2020 had $3/mmBtu swings. Multiplication scales seasonality amplitude.

---

### 3. Agricultural (Corn): Harvest Seasonality + Trend + Weather Shocks


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with pm.Model() as corn_model:
    t = np.arange(len(corn_prices)) / 12  # Monthly data in years
    y = corn_prices - corn_prices.mean()

    # Annual harvest cycle
    sigma_harvest = pm.HalfNormal('sigma_harvest', sigma=2)
    cov_harvest = sigma_harvest**2 * pm.gp.cov.Periodic(1, period=1.0, ls=0.5)

    # Long-term trend (ethanol demand, policy changes)
    sigma_trend = pm.HalfNormal('sigma_trend', sigma=3)
    ell_trend = pm.InverseGamma('ell_trend', alpha=4, beta=8)
    cov_trend = sigma_trend**2 * pm.gp.cov.Matern52(1, ls=ell_trend)

    # Weather shocks during growing season (May-Aug)
    # Use Matérn-1/2 for sharp moves
    sigma_weather = pm.HalfNormal('sigma_weather', sigma=2)
    ell_weather = pm.InverseGamma('ell_weather', alpha=2, beta=2)
    cov_weather = sigma_weather**2 * pm.gp.cov.Matern12(1, ls=ell_weather)

    # Sum all components
    cov_total = cov_harvest + cov_trend + cov_weather

    gp = pm.gp.Marginal(cov_func=cov_total)
    sigma_noise = pm.HalfNormal('sigma_noise', sigma=1)
    y_obs = gp.marginal_likelihood('y_obs', X=t[:, None], y=y, noise=sigma_noise)

    trace = pm.sample(1000, tune=2000, return_inferencedata=True)
```

</div>
</div>

---

## Incorporating Covariates

### Input-Dependent Kernels

Model price as function of time *and* fundamentals (inventory, production).


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with pm.Model() as covariate_model:
    # Inputs: time and inventory levels
    X = np.column_stack([time, inventory_zscore])  # Shape: (n_obs, 2)

    # Different length scales for time vs. inventory
    ell_time = pm.InverseGamma('ell_time', alpha=5, beta=20)
    ell_inventory = pm.InverseGamma('ell_inventory', alpha=3, beta=3)

    # Anisotropic (different scales per dimension)
    cov = pm.gp.cov.Matern52(2, ls=[ell_time, ell_inventory])

    # High inventory → low prices (negative relationship)
    # This is captured by the covariance structure
    gp = pm.gp.Marginal(cov_func=cov)
    sigma_noise = pm.HalfNormal('sigma_noise', sigma=2)
    y_obs = gp.marginal_likelihood('y_obs', X=X, y=prices, noise=sigma_noise)

    trace = pm.sample(1000, tune=2000, return_inferencedata=True)
```

</div>
</div>

**Result:** Model learns how price responds to inventory shocks nonlinearly.

---

## Forecasting with Custom Kernels


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# After fitting GP model
def forecast_gp(trace, model, X_obs, X_new, n_samples=500):
    """
    Generate posterior predictive forecasts from GP.

    Parameters
    ----------
    trace : InferenceData
        PyMC trace with kernel parameters
    model : pm.Model
        Original PyMC model
    X_obs : array (n_obs, n_features)
        Observed inputs
    X_new : array (n_new, n_features)
        Forecast inputs
    n_samples : int
        Number of posterior samples

    Returns
    -------
    forecasts : array (n_samples, n_new)
        Posterior predictive samples
    """
    with model:
        # Conditional GP given observations
        f_new = model.gp.conditional('f_new', X_new)

        # Sample from posterior predictive
        ppc = pm.sample_posterior_predictive(
            trace,
            var_names=['f_new'],
            predictions=True,
            random_seed=42
        )

    return ppc.predictions['f_new'].values

# Example: Forecast next 12 months of corn prices
t_future = np.arange(len(corn_prices), len(corn_prices) + 12) / 12
X_future = t_future[:, None]

forecasts = forecast_gp(trace, corn_model, t[:, None], X_future)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(corn_prices, 'ko', label='Observed')
plt.plot(len(corn_prices) + np.arange(12),
         forecasts.mean(axis=0),
         'r-', label='Forecast Mean')
plt.fill_between(len(corn_prices) + np.arange(12),
                np.percentile(forecasts, 2.5, axis=0),
                np.percentile(forecasts, 97.5, axis=0),
                alpha=0.3, color='r', label='95% Credible Interval')
plt.legend()
plt.xlabel('Month')
plt.ylabel('Price ($/bushel)')
plt.title('Corn Price Forecast with GP (Custom Kernel)')
plt.show()
```

</div>
</div>

---

## Kernel Diagnostic Tools

### 1. Length Scale Interpretation
<div class="callout-key">

<strong>Key Point:</strong> ell_trend_post = trace.posterior['ell_trend'].values.flatten()

</div>


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Extract length scales from trace
ell_trend_post = trace.posterior['ell_trend'].values.flatten()

print(f"Trend length scale: {ell_trend_post.mean():.1f} ± {ell_trend_post.std():.1f} weeks")
print(f"Interpretation: Trend changes over ~{ell_trend_post.mean():.0f} week horizon")
```

</div>
</div>

**Sanity check:**
- Crude trend $\ell \sim 50$ weeks (changes annually)
- Gas seasonal $\ell \sim 5$ weeks (rapid cycles)
- Corn trend $\ell \sim 24$ months (policy/tech cycles)

---

### 2. Posterior Predictive Checks


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
with model:
    ppc = pm.sample_posterior_predictive(trace, random_seed=42)

# Plot observed vs. replicated data
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y, 'ko', alpha=0.5, label='Observed')
for i in range(50):
    ax.plot(ppc.posterior_predictive['y_obs'][0, i, :], 'r-', alpha=0.1)
ax.set_title('Posterior Predictive Check')
ax.legend()
plt.show()
```

</div>
</div>

**Look for:**
- Does replicated data match observed variability?
- Are sudden jumps captured (if using Exponential kernel)?
- Is seasonality amplitude correct (if using Periodic kernel)?

---

### 3. Covariance Matrix Visualization


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Compute posterior mean covariance
with model:
    # Use posterior mean parameters
    cov_matrix = model.gp.cov_func(X_obs).eval()

plt.figure(figsize=(8, 6))
plt.imshow(cov_matrix, cmap='viridis', aspect='auto')
plt.colorbar(label='Covariance')
plt.title('Posterior Mean Covariance Matrix')
plt.xlabel('Time')
plt.ylabel('Time')
plt.show()
```

</div>

**Patterns:**
- Block diagonal: Independent segments (regime change?)
- Banded: Local correlation (stationary kernel)
- Striped: Periodic structure

---

## Common Pitfalls

### 1. Wrong Smoothness Assumption
<div class="callout-insight">

<strong>Insight:</strong> Using SE kernel for volatile commodities smooths over real shocks.



Using SE kernel for volatile commodities smooths over real shocks.

**Fix:** Use Matérn-3/2 or Matérn-1/2 for rougher paths.

---

### 2. Fixed Period for Changing Seasonality

Natural gas heating season has shifted with climate change.

**Fix:** Learn period as parameter:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
period = pm.Gamma('period', alpha=20, beta=20)  # Prior around 1 year
cov_seasonal = sigma**2 * pm.gp.cov.Periodic(1, period=period, ls=ell)
```


---

### 3. Over-Flexible Kernels

Adding too many components overfits.

**Check:** Use LOO-CV to compare kernel designs:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>

```python
loo_simple = az.loo(trace_simple)
loo_complex = az.loo(trace_complex)
print(az.compare({'Simple': loo_simple, 'Complex': loo_complex}))
```


---

### 4. Ignoring Non-Stationarity

Commodity markets have structural breaks (new supply sources, policy changes).

**Fix:**
1. Use change-point detection first
2. Model each regime separately
3. Or use input-dependent kernels with regime indicator

---



## Connections

**Builds on:**
- Module 5: GP fundamentals (mean/covariance functions)
- Module 3: State space (periodic components)

**Leads to:**
- Module 7: Change point detection (when kernel assumptions break)
- Module 8: Integrating fundamentals as GP inputs

**Related concepts:**
- Spectral analysis (periodic kernels ↔ Fourier decomposition)
- Kalman filter (GP is infinite-dimensional state space)

---

## Practice Problems

### Problem 1
Design a kernel for electricity prices with:
- Daily cycle (24-hour period)
- Weekly cycle (weekday vs. weekend)
- Seasonal cycle (summer demand)
<div class="callout-key">

<strong>Key Point:</strong> Design a kernel for electricity prices with:



Write the kernel as a sum/product of base kernels.

### Problem 2
Your GP model for WTI crude has posterior mean length scale $\ell = 200$ weeks. What does this imply about the market? Is this realistic?

### Problem 3
Implement a GP with Matérn-1/2 kernel for natural gas prices. Compare forecasts to Matérn-5/2. Which better captures winter volatility spikes?

### Problem 4
Agricultural prices exhibit "harvest lows" (prices drop when supply floods market). Design a kernel that captures:
1. Annual cycle (low in Oct/Nov)
2. Amplitude of harvest drop varies by year (good harvest = bigger drop)

### Problem 5
You want to forecast copper prices using time + inventory + Chinese PMI. What kernel structure would you use? How do you set priors on length scales?

---


---



## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving kernel design for commodity time series, what would be your first three steps to apply the techniques from this guide?




## Further Reading

1. **Rasmussen, C.E. & Williams, C.K.I. (2006)**. *Gaussian Processes for Machine Learning*. MIT Press.
   - Chapter 4: Kernel design and composition rules

2. **Duvenaud, D. (2014)**. *Automatic Model Construction with Gaussian Processes*. PhD Thesis, University of Cambridge.
   - Grammar of kernel composition

3. **Roberts, S., Osborne, M., et al. (2013)**. "Gaussian Processes for Time-Series Modelling." *Philosophical Transactions of the Royal Society A*, 371(1984).
   - Time series-specific kernel design

4. **Wilson, A.G. & Adams, R.P. (2013)**. "Gaussian Process Kernels for Pattern Discovery and Extrapolation." *ICML*.
   - Spectral mixture kernels for complex seasonality

---

*"A kernel is a prior over functions. Choose wisely, and your GP will see the patterns you need."*

---

## Cross-References

<a class="link-card" href="./02_kernel_design_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_gp_fundamentals.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

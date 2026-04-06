# Gaussian Process Fundamentals

> **Reading time:** ~6 min | **Module:** 5 — Gaussian Processes | **Prerequisites:** Module 4 Hierarchical Models


## In Brief

A Gaussian Process (GP) defines a probability distribution over functions. Instead of parameterizing a specific functional form, GPs specify properties of functions (smoothness, periodicity) through a kernel, letting data determine the actual shape.

<div class="callout-insight">

<strong>Insight:</strong> **Think of a GP as placing a prior directly on the space of functions.** Just as a Normal prior on a parameter expresses beliefs about where the parameter lies, a GP prior expresses beliefs about what functions are plausible.

</div>


<div class="callout-key">

<strong>Key Concept Summary:</strong> A Gaussian Process (GP) defines a probability distribution over functions.

</div>

---

## Formal Definition

### Definition

A **Gaussian Process** is a collection of random variables, any finite number of which have a joint Gaussian distribution.

Formally, $f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$ means:

For any finite set of points $\{x_1, ..., x_n\}$:

$$\begin{bmatrix} f(x_1) \\ \vdots \\ f(x_n) \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} m(x_1) \\ \vdots \\ m(x_n) \end{bmatrix}, \begin{bmatrix} k(x_1, x_1) & \cdots & k(x_1, x_n) \\ \vdots & \ddots & \vdots \\ k(x_n, x_1) & \cdots & k(x_n, x_n) \end{bmatrix} \right)$$

### Components

**Mean Function $m(x)$:**
- Expected value of $f(x)$
- Often set to zero (data centered) or a parametric trend
- $m(x) = \mathbb{E}[f(x)]$

**Kernel (Covariance) Function $k(x, x')$:**
- Defines similarity between function values at different inputs
- Encodes assumptions about smoothness, periodicity, etc.
- $k(x, x') = \text{Cov}(f(x), f(x'))$

---

## Visual Intuition

### Prior Samples

Before seeing data, we can draw function samples from the GP prior:

```
      f(x)
       │     ╱╲     ╱╲
       │    ╱  ╲   ╱  ╲         Sample 1
       │   ╱    ╲ ╱    ╲
       │  ╱      ╳      ╲
       │ ╱      ╱ ╲      ╲
       │╱      ╱   ╲      ╲     Sample 2
       ├──────────────────────
       │      ───────────       Sample 3 (smoother)
       │
       └──────────────────────▶ x

Different kernel parameters produce different types of functions.
Smoother kernels → smoother function samples.
```

### Posterior Update

After observing data, the GP posterior "pins down" the function at observed points while maintaining uncertainty elsewhere:

```
      f(x)
       │         ●              (Observed points)
       │     ╱╲  │  ╱╲
       │    ╱  ╲ │ ╱  ╲         Posterior mean
       │   ╱    ╲│╱    ╲
       │  ╱      ●      ╲
       │ ╱       │       ╲
       │╱        │        ╲
       ├─────────●──────────
       │    ▒▒▒▒▒│▒▒▒▒▒▒▒▒     ▒ = Uncertainty (narrows at data)
       │
       └──────────────────────▶ x
```

---

## GP Regression

### Model

$$y_i = f(x_i) + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2_n)$$

Where:
- $f \sim \mathcal{GP}(0, k)$
- $\sigma^2_n$ is observation noise

### Posterior Predictive

Given training data $(\mathbf{X}, \mathbf{y})$ and test points $\mathbf{X}_*$:

**Posterior Mean:**
$$\bar{f}_* = K_*^T (K + \sigma^2_n I)^{-1} \mathbf{y}$$

**Posterior Covariance:**
$$\text{Cov}(f_*) = K_{**} - K_*^T (K + \sigma^2_n I)^{-1} K_*$$

Where:
- $K = k(\mathbf{X}, \mathbf{X})$ — training covariance
- $K_* = k(\mathbf{X}, \mathbf{X}_*)$ — train-test covariance
- $K_{**} = k(\mathbf{X}_*, \mathbf{X}_*)$ — test covariance

---

## The Kernel's Role

### What the Kernel Encodes

The kernel function $k(x, x')$ specifies:
<div class="callout-key">

<strong>Key Point:</strong> The kernel function $k(x, x')$ specifies:

</div>


1. **Variance:** $k(x, x)$ — marginal variance at each point
2. **Correlation Structure:** How function values relate across inputs
3. **Smoothness:** How quickly correlations decay
4. **Periodicity:** Whether patterns repeat
5. **Stationarity:** Whether properties depend only on $|x - x'|$

### Squared Exponential (RBF) Kernel

$$k(x, x') = \sigma^2 \exp\left(-\frac{(x - x')^2}{2\ell^2}\right)$$

**Parameters:**
- $\sigma^2$: Signal variance (amplitude)
- $\ell$: Length scale (how quickly correlation decays)

**Properties:**
- Infinitely differentiable (very smooth)
- Stationary (depends only on distance)
- Samples are smooth, "wiggly" functions

### Matérn Kernel

$$k_\nu(r) = \sigma^2 \frac{2^{1-\nu}}{\Gamma(\nu)} \left(\frac{\sqrt{2\nu} r}{\ell}\right)^\nu K_\nu\left(\frac{\sqrt{2\nu} r}{\ell}\right)$$

Where $r = |x - x'|$ and $K_\nu$ is the modified Bessel function.

**Common Choices:**
- $\nu = 1/2$: Exponential (rough, Ornstein-Uhlenbeck)
- $\nu = 3/2$: Once differentiable
- $\nu = 5/2$: Twice differentiable
- $\nu \to \infty$: Squared exponential (infinitely smooth)

### Periodic Kernel

$$k(x, x') = \sigma^2 \exp\left(-\frac{2\sin^2(\pi|x - x'|/p)}{\ell^2}\right)$$

**Parameters:**
- $p$: Period (e.g., 365 days for annual seasonality)
- $\ell$: Controls smoothness of periodic pattern

**Use for commodities:** Natural gas heating/cooling seasons, agricultural harvest cycles.

---

## Code Implementation

### Basic GP Regression in PyMC


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n = 50
X = np.sort(np.random.uniform(0, 10, n))[:, None]
y = np.sin(X[:, 0]) + np.random.normal(0, 0.2, n)

# GP model
with pm.Model() as gp_model:
    # Kernel hyperparameters
    ell = pm.Gamma('ell', alpha=2, beta=1)  # Length scale
    sigma = pm.HalfNormal('sigma', sigma=2)  # Signal std
    sigma_n = pm.HalfNormal('sigma_n', sigma=0.5)  # Noise std

    # Covariance function
    cov = sigma**2 * pm.gp.cov.ExpQuad(1, ls=ell)

    # GP prior
    gp = pm.gp.Marginal(cov_func=cov)

    # Likelihood
    y_obs = gp.marginal_likelihood('y_obs', X=X, y=y, sigma=sigma_n)

    # Sample
    trace = pm.sample(1000, tune=1000, random_seed=42)

# Predict
with gp_model:
    X_new = np.linspace(0, 12, 100)[:, None]
    f_pred = gp.conditional('f_pred', X_new)
    ppc = pm.sample_posterior_predictive(trace, var_names=['f_pred'])

# Plot
f_mean = ppc.posterior_predictive['f_pred'].mean(dim=['chain', 'draw'])
f_std = ppc.posterior_predictive['f_pred'].std(dim=['chain', 'draw'])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label='Data', zorder=3)
plt.plot(X_new, f_mean, 'b-', label='GP Mean')
plt.fill_between(X_new.flatten(),
                 f_mean - 2*f_std,
                 f_mean + 2*f_std,
                 alpha=0.3, label='95% CI')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('GP Regression')
plt.legend()
plt.show()
```

</div>
</div>

---

## GP for Commodity Prices

### Designing a Commodity Kernel

For crude oil prices, we might combine:

1. **Long-term trend:** RBF with large length scale
2. **Seasonal pattern:** Periodic kernel (annual)
3. **Short-term dynamics:** Matérn with small length scale
4. **Observation noise:** White noise kernel


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python

# Composite kernel for commodities
k_trend = pm.gp.cov.ExpQuad(1, ls=365*2)  # Multi-year trends
k_seasonal = pm.gp.cov.Periodic(1, period=365, ls=30)  # Annual pattern
k_short = pm.gp.cov.Matern52(1, ls=20)  # Short-term
k_noise = pm.gp.cov.WhiteNoise(sigma=0.1)

k_total = k_trend + k_seasonal * k_short + k_noise
```

</div>

---

## Common Pitfalls

### 1. Oversmoothing
Length scale too large → predictions too smooth, missing important variation.

### 2. Wrong Kernel Choice
Using periodic kernel without actual periodicity, or RBF for rough data.

### 3. Computational Limits
GPs scale as $O(n^3)$. For $n > 1000$, need sparse approximations.

### 4. Extrapolation Overconfidence
GP uncertainty may not increase fast enough outside training range.

---

## Connections

**Builds on:**
- Module 1: Bayesian inference fundamentals
- Linear algebra: Matrix operations, MVN distribution

**Leads to:**
- Module 6: GP inference via MCMC
- Module 8: GP with fundamental covariates

---

## Practice Problems

1. For a GP with RBF kernel ($\ell = 1$), calculate $k(0, 0)$, $k(0, 1)$, and $k(0, 3)$. What do these values tell you?

2. You want to model natural gas prices with annual seasonality. Design a kernel that captures: (a) annual periodicity, (b) within-year smoothness, (c) year-over-year level changes.

3. Why might Matérn-3/2 be preferred over RBF for commodity returns?

---


---

## Practice Questions

<div class="callout-info">

<strong>Test Your Understanding</strong>

1. Explain in your own words the key difference between the concepts covered in "Formal Definition" and why it matters in practice.

2. Given a real-world scenario involving gaussian process fundamentals, what would be your first three steps to apply the techniques from this guide?




## Further Reading

1. **Rasmussen & Williams** *Gaussian Processes for Machine Learning* — Free online, definitive reference
2. **MacKay** *Information Theory, Inference, and Learning Algorithms* Ch. 45 — Excellent GP introduction
3. **GPyTorch documentation** — For scalable GP implementations

---

*"GPs answer the question: given what I know about function smoothness and the data I've seen, what functions are plausible?"*

---

## Cross-References

<a class="link-card" href="./01_gp_fundamentals_slides.md">
  <div class="link-card-title">Companion Slide Deck</div>
  <div class="link-card-description">Visual presentation covering the key concepts from this guide.</div>
</a>

<a class="link-card" href="../notebooks/01_gp_fundamentals.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive notebook with working code examples and exercises.</div>
</a>

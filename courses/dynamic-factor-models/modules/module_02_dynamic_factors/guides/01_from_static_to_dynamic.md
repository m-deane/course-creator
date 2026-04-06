# From Static to Dynamic Factor Models

> **Reading time:** ~12 min | **Module:** Module 2: Dynamic Factors | **Prerequisites:** Modules 0-1

<div class="callout-key">

**Key Concept Summary:** Dynamic factor models extend static factor models by allowing factors to evolve over time according to autoregressive processes. This temporal structure captures persistence, co-movement dynamics, and enables optimal forecasting of both factors and observed variables.

</div>

## In Brief

Dynamic factor models extend static factor models by allowing factors to evolve over time according to autoregressive processes. This temporal structure captures persistence, co-movement dynamics, and enables optimal forecasting of both factors and observed variables.

<div class="callout-insight">

**Insight:** Real-world factors don't jump randomly from one value to another—they evolve smoothly with persistence. A recession today predicts weaker activity tomorrow; high inflation tends to persist. By modeling factor dynamics explicitly, we gain three capabilities: better factor estimation (using time-series information), forecasting (propagating factor dynamics forward), and structural interpretation (understanding shock propagation).

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Formal Definition

### Static Factor Model (Review)

**Measurement Equation:**
$$X_t = \Lambda F_t + e_t$$

where:
- $X_t$: $N \times 1$ vector of observed variables at time $t$
- $F_t$: $r \times 1$ vector of latent factors at time $t$
- $\Lambda$: $N \times r$ matrix of factor loadings
- $e_t$: $N \times 1$ idiosyncratic error, $e_t \sim N(0, \Sigma_e)$

**Key limitation:** Factors $F_t$ are treated as independent draws over time.

### Dynamic Factor Model (DFM)

**Measurement Equation:**
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$

**Transition Equation (Factor Dynamics):**
$$F_t = \Phi_1 F_{t-1} + \Phi_2 F_{t-2} + ... + \Phi_p F_{t-p} + \eta_t, \quad \eta_t \sim N(0, Q)$$

Compact VAR(p) notation:
$$\Phi(L) F_t = \eta_t$$

where $\Phi(L) = I - \Phi_1 L - \Phi_2 L^2 - ... - \Phi_p L^p$ is the lag polynomial.

**Key addition:** Factors follow a vector autoregression, capturing:
- Persistence within each factor
- Lead-lag relationships between factors
- Shock propagation through the factor space

---

## 2. Intuitive Explanation

### The Elevator Pitch

Think of factors as economic "currents" that push variables around. In the static model, we assume these currents are random gusts with no memory. In the dynamic model, we recognize that strong currents today tend to produce strong currents tomorrow—the ocean doesn't instantly calm after a storm.

### Concrete Example: GDP Growth and Inflation

**Static model perspective:**
- Observe GDP growth and inflation in month $t$
- Extract factors $F_t$ explaining co-movement
- Move to month $t+1$, extract factors $F_{t+1}$ independently
- No connection between $F_t$ and $F_{t+1}$

**Dynamic model perspective:**
- Observe GDP growth and inflation in month $t$
- Extract factor $F_t$ (say, "economic activity")
- Recognize that high activity today predicts high activity next month: $F_{t+1} \approx \phi F_t + \text{shock}$
- Use this persistence to improve factor estimates and make forecasts

### Why Add Dynamics?

**1. Better Factor Estimation**
- Time series information provides additional identification
- Smoothing: Future observations help estimate past factors
- Missing data: Factor dynamics bridge gaps

**2. Forecasting**
- Static model: Cannot forecast factors, only use current factors to forecast variables
- Dynamic model: Forecast factors using VAR, then forecast variables via loadings

**3. Structural Interpretation**
- Impulse responses: How shocks propagate through factors to variables
- Variance decomposition: Which factors drive variation at different horizons
- Persistence measurement: How long shocks affect the economy

---

## 3. Mathematical Formulation

### VAR(1) Factor Dynamics

Consider the simplest case: $p = 1$, $r = 2$.

$$\begin{bmatrix} F_{1t} \\ F_{2t} \end{bmatrix} = \begin{bmatrix} \phi_{11} & \phi_{12} \\ \phi_{21} & \phi_{22} \end{bmatrix} \begin{bmatrix} F_{1,t-1} \\ F_{2,t-1} \end{bmatrix} + \begin{bmatrix} \eta_{1t} \\ \eta_{2t} \end{bmatrix}$$

**Interpretation of $\Phi_1$:**
- $\phi_{11}$: Persistence of factor 1 (autocorrelation)
- $\phi_{22}$: Persistence of factor 2
- $\phi_{12}$: Effect of lagged factor 2 on current factor 1 (cross-dynamics)
- $\phi_{21}$: Effect of lagged factor 1 on current factor 2

**Stationarity:** Requires eigenvalues of $\Phi_1$ inside the unit circle:
$$|\lambda_i(\Phi_1)| < 1, \quad i = 1, ..., r$$

### Implications for Covariance Structure

**Static model covariance:**
$$\text{Cov}(X_t, X_t) = \Lambda \Sigma_F \Lambda' + \Sigma_e$$

No autocovariance structure: $\text{Cov}(X_t, X_{t-h}) = 0$ for $h > 0$ (under i.i.d. factors).

**Dynamic model covariance:**
$$\text{Cov}(X_t, X_t) = \Lambda \Sigma_F \Lambda' + \Sigma_e$$

Autocovariance structure:
$$\text{Cov}(X_t, X_{t-h}) = \Lambda \Gamma_F(h) \Lambda'$$

where $\Gamma_F(h) = \text{Cov}(F_t, F_{t-h})$ is determined by the VAR dynamics.

### Spectral Representation

Dynamic factor models have frequency-specific factor loadings. At frequency $\omega$:

$$S_X(\omega) = \Lambda(\omega) \Lambda(\omega)^* + S_e(\omega)$$

where $\Lambda(\omega) = \Lambda \Phi(e^{-i\omega})^{-1}$ is the frequency-dependent loading.

**Implication:** Factors can explain different amounts of variance at different frequencies (business cycle vs. high-frequency noise).

---

## 4. Extending to VAR(p)

### Companion Form

Any VAR(p) can be written as VAR(1) in companion form.

**Original VAR(2):**
$$F_t = \Phi_1 F_{t-1} + \Phi_2 F_{t-2} + \eta_t$$

**Companion form:** Define $\alpha_t = [F_t', F_{t-1}']'$

$$\begin{bmatrix} F_t \\ F_{t-1} \end{bmatrix} = \begin{bmatrix} \Phi_1 & \Phi_2 \\ I_r & 0 \end{bmatrix} \begin{bmatrix} F_{t-1} \\ F_{t-2} \end{bmatrix} + \begin{bmatrix} \eta_t \\ 0 \end{bmatrix}$$

This is a VAR(1) in the augmented state $\alpha_t$ of dimension $rp$.

**Measurement equation in companion form:**
$$X_t = \begin{bmatrix} \Lambda & 0 \end{bmatrix} \alpha_t + e_t$$

Only the first $r$ elements of $\alpha_t$ (current factors) enter the measurement equation.

---

## 5. Code Implementation

### Simulating a Dynamic Factor Model

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_discrete_lyapunov

# Configuration
np.random.seed(42)
T, N, r, p = 300, 20, 2, 1  # 300 periods, 20 variables, 2 factors, VAR(1)

# Factor loadings (with economic structure)
Lambda = np.random.randn(N, r)
Lambda[:10, 0] = np.abs(Lambda[:10, 0]) * 1.5  # First 10 load on factor 1
Lambda[10:, 1] = np.abs(Lambda[10:, 1]) * 1.5  # Last 10 load on factor 2

# Factor VAR(1) dynamics - stable with moderate persistence
Phi = np.array([
    [0.7, 0.1],   # Factor 1: persistent, slight influence from factor 2
    [0.2, 0.6]    # Factor 2: moderate persistence, influenced by factor 1
])

# Check stationarity
eigenvalues = np.linalg.eigvals(Phi)
print(f"Eigenvalues of Phi: {eigenvalues}")
print(f"Maximum modulus: {np.max(np.abs(eigenvalues)):.3f}")
assert np.max(np.abs(eigenvalues)) < 1, "VAR is not stationary!"

# Innovation covariance
Q = np.eye(r)

# Idiosyncratic error covariance
Sigma_e = np.diag(np.random.uniform(0.2, 0.5, N))

# Compute unconditional factor covariance (for initialization)
Sigma_F = solve_discrete_lyapunov(Phi, Q)
print(f"\nUnconditional factor covariance:\n{Sigma_F}")

# Simulate factors with burn-in
burn_in = 100
F = np.zeros((T + burn_in, r))
F[0] = np.random.multivariate_normal(np.zeros(r), Sigma_F)  # Start from stationary distribution

for t in range(1, T + burn_in):
    eta = np.random.multivariate_normal(np.zeros(r), Q)
    F[t] = Phi @ F[t-1] + eta

# Discard burn-in
F = F[burn_in:]

# Generate observed data
e = np.random.multivariate_normal(np.zeros(N), Sigma_e, T)
X = F @ Lambda.T + e

print(f"\nData shapes:")
print(f"  X: {X.shape} (T x N)")
print(f"  F: {F.shape} (T x r)")
print(f"  Lambda: {Lambda.shape} (N x r)")
```

### Visualizing Factor Dynamics

```python
# Plot factor dynamics
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

for i in range(r):
    axes[i].plot(F[:, i], label=f'Factor {i+1}', linewidth=1.5)
    axes[i].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[i].set_ylabel(f'$F_{{{i+1},t}}$', fontsize=12)
    axes[i].legend(loc='upper right')
    axes[i].grid(alpha=0.3)

axes[1].set_xlabel('Time', fontsize=12)
axes[0].set_title('Dynamic Factor Evolution (VAR(1) Process)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Autocorrelation function
from statsmodels.tsa.stattools import acf

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

for i in range(r):
    acf_vals = acf(F[:, i], nlags=20, fft=True)
    axes[i].bar(range(21), acf_vals, alpha=0.7)
    axes[i].axhline(0, color='k', linestyle='-', linewidth=0.8)
    axes[i].axhline(1.96/np.sqrt(T), color='r', linestyle='--', linewidth=1)
    axes[i].axhline(-1.96/np.sqrt(T), color='r', linestyle='--', linewidth=1)
    axes[i].set_xlabel('Lag', fontsize=11)
    axes[i].set_ylabel('Autocorrelation', fontsize=11)
    axes[i].set_title(f'ACF: Factor {i+1}', fontsize=12, fontweight='bold')
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

### Comparing Static vs Dynamic Estimation


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
from sklearn.decomposition import PCA

# Static estimation: PCA on pooled data (ignoring dynamics)
pca = PCA(n_components=r)
F_static = pca.fit_transform(X)
Lambda_static = pca.components_.T

# Compare true vs static estimated factors
# (Need to solve rotation/identification problem)
from scipy.linalg import orthogonal_procrustes

# Find rotation that best aligns F_static to F_true
R, _ = orthogonal_procrustes(F_static, F)
F_static_aligned = F_static @ R

# Plot comparison
fig, axes = plt.subplots(r, 1, figsize=(12, 6), sharex=True)

for i in range(r):
    axes[i].plot(F[:100, i], label='True Factor', linewidth=2, alpha=0.8)
    axes[i].plot(F_static_aligned[:100, i], label='PCA Estimate (Static)',
                 linewidth=1.5, alpha=0.8, linestyle='--')
    axes[i].set_ylabel(f'Factor {i+1}', fontsize=11)
    axes[i].legend(loc='upper right')
    axes[i].grid(alpha=0.3)

axes[-1].set_xlabel('Time', fontsize=12)
axes[0].set_title('True vs Static Estimated Factors (First 100 Periods)',
                   fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Correlation between true and estimated factors
for i in range(r):
    corr = np.corrcoef(F[:, i], F_static_aligned[:, i])[0, 1]
    print(f"Correlation: True Factor {i+1} vs PCA: {corr:.3f}")
```

</div>

**Note:** PCA still provides reasonable factor estimates even ignoring dynamics, but dynamic methods (Kalman filter in next guide) will improve estimates by exploiting time-series structure.

---

## 6. Impulse Response Analysis

Dynamic factor models enable studying how shocks propagate.

### Theory

**Impulse Response Function (IRF):** Effect of a one-unit shock to factor $j$ at time $t$ on factor $k$ at time $t+h$:

$$\text{IRF}_k(h, j) = \frac{\partial F_{k,t+h}}{\partial \eta_{jt}}$$

For VAR(1): $F_{t+h} = \Phi^h F_t + \Phi^{h-1}\eta_{t+1} + ... + \Phi \eta_{t+h-1} + \eta_{t+h}$

So: $\frac{\partial F_{t+h}}{\partial \eta_t} = \Phi^h$

### Implementation


<span class="filename">compute_irf.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def compute_irf(Phi, horizons=20):
    """
    Compute impulse response function for VAR(1).

    Parameters
    ----------
    Phi : array (r, r)
        VAR(1) coefficient matrix
    horizons : int
        Number of horizons to compute

    Returns
    -------
    irf : array (horizons+1, r, r)
        IRF[h, i, j] = response of factor i at horizon h to shock in factor j
    """
    r = Phi.shape[0]
    irf = np.zeros((horizons + 1, r, r))

    irf[0] = np.eye(r)  # Impact response

    for h in range(1, horizons + 1):
        irf[h] = Phi @ irf[h-1]

    return irf

# Compute IRF
irf = compute_irf(Phi, horizons=20)

# Plot IRF
fig, axes = plt.subplots(r, r, figsize=(12, 8), sharex=True)

for i in range(r):
    for j in range(r):
        axes[i, j].plot(irf[:, i, j], linewidth=2)
        axes[i, j].axhline(0, color='k', linestyle='--', linewidth=0.8)
        axes[i, j].set_title(f'Response of F{i+1} to shock in F{j+1}', fontsize=10)
        axes[i, j].grid(alpha=0.3)

        if i == r-1:
            axes[i, j].set_xlabel('Horizon', fontsize=10)
        if j == 0:
            axes[i, j].set_ylabel('Response', fontsize=10)

plt.suptitle('Impulse Response Functions', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.show()

# Cumulative responses
cum_irf = np.cumsum(irf, axis=0)
print("\nCumulative responses at horizon 20:")
print(cum_irf[-1])
```

</div>

---

## Common Pitfalls

### 1. Non-Stationary Factor Dynamics

**Problem:** Eigenvalues of $\Phi$ outside unit circle cause factors to explode.

**Solution:**
- Always check stationarity: `np.max(np.abs(np.linalg.eigvals(Phi))) < 1`
- If factors are I(1), consider differencing or cointegration
- Estimate with stationarity constraint

### 2. Overparameterization

**Problem:** With $r$ factors and VAR($p$), have $r^2 p$ parameters in transition equation.

**Solution:**
- Keep $p$ small (usually $p = 1$ or $p = 2$ sufficient)
- Use information criteria (AIC, BIC) to select $p$
- Consider sparse VAR if $r$ is large

### 3. Identification Through Dynamics Alone

**Problem:** Assuming dynamics fully identify factors without normalization.

**Reality:**
- Still need loading restrictions (as in static case)
- Dynamics provide additional information but don't eliminate identification problem
- Must combine loading constraints with dynamic structure

### 4. Ignoring Initial Conditions

**Problem:** Starting factors at $F_0 = 0$ or random values in finite samples.

**Solution:**
- Initialize from stationary distribution: $F_0 \sim N(0, \Sigma_F)$
- Use "diffuse initialization" in Kalman filter (next guide)
- Include burn-in period in simulations

### 5. Mixing Dynamic and Static Components

**Problem:** Some factors dynamic, some static—can't use standard methods.

**Clarification:**
- All factors should have same temporal structure
- If some variables are purely static, model separately
- Consider factor-augmented models for mixed cases

---

## Connections

### Builds On
- **Static Factor Models** (Module 1): Measurement equation, identification
- **Vector Autoregressions (VAR)**: Time series dynamics, stationarity
- **Multivariate Normal Theory**: Conditional distributions

### Leads To
- **State-Space Representation** (Next Guide): Unified framework for DFM
- **Kalman Filter** (Guide 3): Optimal factor estimation with dynamics
- **Maximum Likelihood Estimation** (Module 4): Parameter estimation for DFM

### Related To
- **DSGE Models**: Economic factors follow structural equations
- **Mixed Frequency Models** (Module 5): Exploit dynamics for temporal aggregation
- **Nowcasting**: Use factor dynamics for real-time forecasting

---

## Practice Problems

### Conceptual

1. **Persistence and Forecasting**
   - Explain why a static factor model cannot forecast factors, while a dynamic model can.
   - If $\Phi = 0.9 I$, what happens to forecast uncertainty as horizon increases?

2. **Cross-Factor Dynamics**
   - What does it mean if $\phi_{12} > 0$ in a two-factor model?
   - Give an economic example where cross-factor dynamics would be important.

3. **Frequency Decomposition**
   - Factors with $\phi$ close to 1 explain variation at what frequencies?
   - How might you design factors to capture business cycle vs short-term fluctuations?

### Mathematical

4. **Stationarity**
   - For $\Phi = \begin{bmatrix} 0.8 & \alpha \\ 0 & 0.6 \end{bmatrix}$, what is the maximum value of $\alpha$ for stationarity?
   - Derive the unconditional variance $\Sigma_F$ for VAR(1) with $Q = I$.

5. **Autocovariance**
   - For VAR(1), show that $\Gamma_F(h) = \Phi^h \Sigma_F$.
   - Derive $\text{Cov}(X_t, X_{t-1})$ in terms of model parameters.

6. **Companion Form**
   - Write the companion form for $F_t = \Phi_1 F_{t-1} + \Phi_2 F_{t-2} + \Phi_3 F_{t-3} + \eta_t$.
   - What is the dimension of the companion state vector?

### Implementation

7. **Simulation Study**

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   # Simulate DFM with T=500, N=30, r=3, VAR(2) factors
   # Verify:
   # - Factors have correct autocovariance structure
   # - Implied X autocovariances match sample autocovariances
   # - PCA recovers approximately correct factors
   ```

</div>

8. **VAR Estimation**

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   # Given simulated factors F from problem 7:
   # - Estimate Phi using OLS
   # - Compare true vs estimated IRFs
   # - Test residuals for autocorrelation
   ```

</div>

9. **Model Comparison**

<span class="filename">example.py</span>
</div>

   <div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
   # Simulate data from static model (Phi = 0)
   # Simulate data from dynamic model (Phi ≠ 0)
   # Can you distinguish them using:
   # - Sample autocovariances of X?
   # - Forecasting performance?
   ```

</div>

### Extension

10. **Structural Shocks**
    - Instead of $\eta_t \sim N(0, Q)$, assume $\eta_t = B \epsilon_t$ where $\epsilon_t \sim N(0, I)$ are structural shocks.
    - Implement Cholesky identification: $Q = BB'$, take $B = \text{chol}(Q)$.
    - Compute structural IRFs and compare to reduced-form IRFs.

---

<div class="callout-insight">

**Insight:** Understanding from static to dynamic factor models is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

### Essential
- **Hamilton, J.D. (1994).** *Time Series Analysis.* Chapter 13 (State-space models), Chapter 11 (VAR).
  - Rigorous treatment of VAR dynamics and state-space methods.

- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models, Factor-Augmented Vector Autoregressions, and Structural Vector Autoregressions in Macroeconomics." *Handbook of Macroeconomics, Vol 2.*
  - Comprehensive review of dynamic factor models in macro applications.

### Recommended
- **Lütkepohl, H. (2005).** *New Introduction to Multiple Time Series Analysis.* Chapters 2-3.
  - Thorough VAR theory, estimation, and testing.

- **Forni, M., Hallin, M., Lippi, M., & Reichlin, L. (2000).** "The Generalized Dynamic-Factor Model: Identification and Estimation." *Review of Economics and Statistics.*
  - Spectral approach to dynamic factor models.

### Advanced
- **Bai, J. & Wang, P. (2015).** "Identification and Bayesian Estimation of Dynamic Factor Models." *Journal of Business & Economic Statistics.*
  - Identification through heteroskedasticity and dynamics.

- **Doz, C., Giannone, D., & Reichlin, L. (2012).** "A Quasi-Maximum Likelihood Approach for Large, Approximate Dynamic Factor Models." *Review of Economics and Statistics.*
  - Two-step estimation: PCA + Kalman smoother.

---

**Next Guide:** State-Space Representation - converting DFM to the standard form for Kalman filtering.

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_from_static_to_dynamic_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_kalman_filter_implementation.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_state_space_representation.md">
  <div class="link-card-title">02 State Space Representation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_kalman_filter_derivation.md">
  <div class="link-card-title">03 Kalman Filter Derivation</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


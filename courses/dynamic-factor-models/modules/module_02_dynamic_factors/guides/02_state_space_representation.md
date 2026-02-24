# State-Space Representation of Dynamic Factor Models

## In Brief

The state-space form provides a unified framework for dynamic factor models, expressing them as two equations: a measurement equation linking observables to latent states, and a transition equation governing state evolution. This representation enables the Kalman filter for optimal factor estimation and likelihood computation.

> 💡 **Key Insight:** Every dynamic factor model can be written in state-space form, which separates what we observe from what we don't. The "state" contains all the information needed to predict the future—current and lagged factors. The state-space formulation is not just mathematical convenience; it's the gateway to optimal estimation via the Kalman filter, handling missing data, forecasting, and computing likelihoods for parameter estimation.

---

## 1. Generic State-Space Model

### Mathematical Definition

**Measurement Equation:**
$$y_t = Z \alpha_t + d + \epsilon_t, \quad \epsilon_t \sim N(0, H)$$

**Transition Equation:**
$$\alpha_t = T \alpha_{t-1} + c + R\eta_t, \quad \eta_t \sim N(0, Q)$$

**Initial State:**
$$\alpha_1 \sim N(a_1, P_1)$$

### Components

| Symbol | Dimension | Meaning |
|--------|-----------|---------|
| $y_t$ | $N \times 1$ | Observed variables (measurements) |
| $\alpha_t$ | $m \times 1$ | State vector (latent/unobserved) |
| $Z$ | $N \times m$ | Measurement matrix (loadings) |
| $d$ | $N \times 1$ | Measurement intercept |
| $\epsilon_t$ | $N \times 1$ | Measurement error |
| $H$ | $N \times N$ | Measurement error covariance |
| $T$ | $m \times m$ | Transition matrix (state dynamics) |
| $c$ | $m \times 1$ | Transition intercept |
| $R$ | $m \times q$ | Selection matrix for innovations |
| $\eta_t$ | $q \times 1$ | State innovations |
| $Q$ | $q \times q$ | Innovation covariance |

### Key Assumptions

1. **Gaussian Innovations:** $\epsilon_t$ and $\eta_t$ are normally distributed
2. **Time Invariance:** Matrices $Z, T, H, Q, R$ do not depend on $t$ (standard form)
3. **Independence:** $\text{Cov}(\epsilon_t, \eta_s) = 0$ for all $t, s$
4. **Serial Independence:** $\text{Cov}(\eta_t, \eta_s) = 0$ and $\text{Cov}(\epsilon_t, \epsilon_s) = 0$ for $t \neq s$

### Interpretation

- **Measurement equation:** How observed $y_t$ relates to latent state $\alpha_t$
- **Transition equation:** How state evolves over time (Markov property)
- **Filtering problem:** Given $y_1, ..., y_t$, estimate $\alpha_t$
- **Smoothing problem:** Given $y_1, ..., y_T$, estimate $\alpha_t$ for $t \leq T$
- **Forecasting:** Given $y_1, ..., y_t$, predict $y_{t+h}$ and $\alpha_{t+h}$

---

## 2. DFM in State-Space Form: VAR(1) Case

### Dynamic Factor Model Specification

**Measurement:**
$$X_t = \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e)$$

**Transition (VAR(1)):**
$$F_t = \Phi F_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)$$

### State-Space Mapping

**Identification:**
- $y_t = X_t$ (observables)
- $\alpha_t = F_t$ (state = factors)
- $Z = \Lambda$ (measurement matrix = loadings)
- $H = \Sigma_e$ (measurement error = idiosyncratic covariance)
- $T = \Phi$ (transition matrix = VAR coefficients)
- $R = I_r$ (no selection needed)
- $Q = Q$ (innovation covariance)
- $d = c = 0$ (assuming demeaned data)

### Resulting State-Space Form

$$\begin{align}
X_t &= \Lambda F_t + e_t, \quad e_t \sim N(0, \Sigma_e) \\
F_t &= \Phi F_{t-1} + \eta_t, \quad \eta_t \sim N(0, Q)
\end{align}$$

**Dimensions:**
- State: $m = r$ (number of factors)
- Observables: $N$ (number of variables)
- Innovations: $q = r$ (one innovation per factor)

---

## 3. DFM in State-Space Form: VAR(p) Case

### Challenge

For VAR(p), we have:
$$F_t = \Phi_1 F_{t-1} + \Phi_2 F_{t-2} + ... + \Phi_p F_{t-p} + \eta_t$$

This doesn't fit the state-space form because the transition depends on $p$ lags, not just $\alpha_{t-1}$.

### Solution: Companion Form

**Augmented state vector:** Include current factors and $p-1$ lags.

$$\alpha_t = \begin{bmatrix} F_t \\ F_{t-1} \\ F_{t-2} \\ \vdots \\ F_{t-p+1} \end{bmatrix}, \quad \text{dimension: } m = rp$$

### State-Space Representation

**Measurement Equation:**
$$X_t = \begin{bmatrix} \Lambda & 0 & 0 & \cdots & 0 \end{bmatrix} \alpha_t + e_t$$

Only the first $r$ elements (current factors) affect observables.

**Transition Equation:**
$$\begin{bmatrix} F_t \\ F_{t-1} \\ F_{t-2} \\ \vdots \\ F_{t-p+1} \end{bmatrix} = \begin{bmatrix} \Phi_1 & \Phi_2 & \Phi_3 & \cdots & \Phi_p \\ I_r & 0 & 0 & \cdots & 0 \\ 0 & I_r & 0 & \cdots & 0 \\ \vdots & \vdots & \ddots & \ddots & \vdots \\ 0 & 0 & \cdots & I_r & 0 \end{bmatrix} \begin{bmatrix} F_{t-1} \\ F_{t-2} \\ F_{t-3} \\ \vdots \\ F_{t-p} \end{bmatrix} + \begin{bmatrix} \eta_t \\ 0 \\ 0 \\ \vdots \\ 0 \end{bmatrix}$$

**In compact notation:**
$$\alpha_t = T \alpha_{t-1} + R\eta_t$$

where:
- $T$ is $rp \times rp$ companion matrix
- $R = [I_r, 0, ..., 0]'$ is $rp \times r$ selection matrix
- $\eta_t$ is $r \times 1$ innovation

### Example: VAR(2) with $r = 2$ Factors

$$\alpha_t = \begin{bmatrix} F_{1t} \\ F_{2t} \\ F_{1,t-1} \\ F_{2,t-1} \end{bmatrix}, \quad m = 4$$

$$Z = \begin{bmatrix} \lambda_{11} & \lambda_{12} & 0 & 0 \\ \lambda_{21} & \lambda_{22} & 0 & 0 \\ \vdots & \vdots & \vdots & \vdots \\ \lambda_{N1} & \lambda_{N2} & 0 & 0 \end{bmatrix}$$

$$T = \begin{bmatrix} \phi_{11}^{(1)} & \phi_{12}^{(1)} & \phi_{11}^{(2)} & \phi_{12}^{(2)} \\ \phi_{21}^{(1)} & \phi_{22}^{(1)} & \phi_{21}^{(2)} & \phi_{22}^{(2)} \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

$$R = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \\ 0 & 0 \end{bmatrix}, \quad Q = \begin{bmatrix} q_{11} & q_{12} \\ q_{21} & q_{22} \end{bmatrix}$$

---

## 4. Code Implementation

### Setting Up State-Space Matrices

```python
import numpy as np
from scipy.linalg import block_diag

def dfm_to_statespace(Lambda, Phi_list, Sigma_e, Q):
    """
    Convert dynamic factor model to state-space representation.

    Parameters
    ----------
    Lambda : array (N, r)
        Factor loadings
    Phi_list : list of arrays [(r, r), (r, r), ...]
        VAR coefficient matrices [Phi_1, Phi_2, ..., Phi_p]
    Sigma_e : array (N, N) or (N,)
        Idiosyncratic error covariance (can be diagonal)
    Q : array (r, r)
        Factor innovation covariance

    Returns
    -------
    Z : array (N, m)
        Measurement matrix
    H : array (N, N)
        Measurement error covariance
    T : array (m, m)
        Transition matrix
    R : array (m, r)
        Selection matrix
    Q : array (r, r)
        Innovation covariance
    """
    N, r = Lambda.shape
    p = len(Phi_list)
    m = r * p  # State dimension

    # Measurement matrix Z
    Z = np.zeros((N, m))
    Z[:, :r] = Lambda  # Only current factors load

    # Measurement error covariance H
    if Sigma_e.ndim == 1:
        H = np.diag(Sigma_e)
    else:
        H = Sigma_e

    # Transition matrix T (companion form)
    T = np.zeros((m, m))
    T[:r, :] = np.column_stack(Phi_list)  # First row: [Phi_1, Phi_2, ..., Phi_p]

    # Identity blocks for lagged states
    for i in range(p - 1):
        T[r*(i+1):r*(i+2), r*i:r*(i+1)] = np.eye(r)

    # Selection matrix R
    R = np.zeros((m, r))
    R[:r, :] = np.eye(r)

    return Z, H, T, R, Q


# Example: VAR(2) with r=2 factors, N=10 variables
np.random.seed(42)
N, r, p = 10, 2, 2

Lambda = np.random.randn(N, r)
Phi_1 = np.array([[0.7, 0.1], [0.2, 0.6]])
Phi_2 = np.array([[0.2, -0.1], [0.1, 0.15]])
Sigma_e = np.random.uniform(0.1, 0.5, N)
Q = np.eye(r) * 0.5

# Convert to state-space
Z, H, T, R, Q_ss = dfm_to_statespace(Lambda, [Phi_1, Phi_2], Sigma_e, Q)

print(f"State-space dimensions:")
print(f"  Z: {Z.shape} (N x m) - {N} observables, {r*p} states")
print(f"  H: {H.shape} (N x N)")
print(f"  T: {T.shape} (m x m) - {r*p} x {r*p} companion matrix")
print(f"  R: {R.shape} (m x r)")
print(f"  Q: {Q_ss.shape} (r x r)")

print(f"\nTransition matrix T (companion form):")
print(T)
```

### Simulating from State-Space Form

```python
def simulate_statespace(Z, H, T, R, Q, T_periods, alpha_0=None, P_0=None):
    """
    Simulate data from state-space model.

    Parameters
    ----------
    Z, H, T, R, Q : arrays
        State-space matrices
    T_periods : int
        Number of time periods
    alpha_0 : array (m,), optional
        Initial state mean (default: zero)
    P_0 : array (m, m), optional
        Initial state covariance (default: stationary)

    Returns
    -------
    y : array (T_periods, N)
        Simulated observations
    alpha : array (T_periods, m)
        Simulated states
    """
    N, m = Z.shape
    r = Q.shape[0]

    # Initialize state
    if alpha_0 is None:
        alpha_0 = np.zeros(m)
    if P_0 is None:
        from scipy.linalg import solve_discrete_lyapunov
        try:
            Sigma_alpha = solve_discrete_lyapunov(T, R @ Q @ R.T)
            alpha_0 = np.random.multivariate_normal(alpha_0, Sigma_alpha)
        except:
            alpha_0 = np.zeros(m)  # Fallback if not stationary

    # Allocate storage
    alpha = np.zeros((T_periods, m))
    y = np.zeros((T_periods, N))

    alpha[0] = alpha_0

    # Simulate
    for t in range(T_periods):
        # Measurement
        epsilon = np.random.multivariate_normal(np.zeros(N), H)
        y[t] = Z @ alpha[t] + epsilon

        # Transition (for next period)
        if t < T_periods - 1:
            eta = np.random.multivariate_normal(np.zeros(r), Q)
            alpha[t+1] = T @ alpha[t] + R @ eta

    return y, alpha


# Simulate data
T_sim = 300
y_sim, alpha_sim = simulate_statespace(Z, H, T, R, Q_ss, T_sim)

print(f"\nSimulated data:")
print(f"  y: {y_sim.shape} (T x N)")
print(f"  alpha: {alpha_sim.shape} (T x m)")

# Extract factors from state (first r components)
F_sim = alpha_sim[:, :r]

import matplotlib.pyplot as plt

fig, axes = plt.subplots(r, 1, figsize=(12, 6), sharex=True)
for i in range(r):
    axes[i].plot(F_sim[:, i], linewidth=1.5)
    axes[i].set_ylabel(f'$F_{{{i+1},t}}$', fontsize=11)
    axes[i].grid(alpha=0.3)
axes[-1].set_xlabel('Time', fontsize=11)
axes[0].set_title('Simulated Factors from State-Space Model', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## 5. Special Cases and Extensions

### Exact vs Approximate Factor Models

**Exact Factor Model:**
$$H = \Sigma_e = \text{diag}(\psi_1^2, ..., \psi_N^2)$$

Idiosyncratic errors are uncorrelated across variables.

**Approximate Factor Model:**
$$H = \Sigma_e \text{ can be non-diagonal}$$

Allows weak cross-sectional correlation in errors. In large $N$, these correlations average out.

**State-space representation:** Same form, but $H$ structure differs.

### Including Exogenous Variables

If some variables are exogenous predictors (e.g., oil prices), augment the measurement equation:

$$X_t = \Lambda F_t + \Gamma W_t + e_t$$

where $W_t$ are observed exogenous variables.

**State-space form:**
$$y_t = Z \alpha_t + B W_t + \epsilon_t$$

The matrix $B = \Gamma$ enters as additional parameters.

### Time-Varying Parameters

If parameters evolve (e.g., $\Lambda_t$ changes over time), use time-varying state-space:

$$y_t = Z_t \alpha_t + \epsilon_t$$

The Kalman filter naturally handles time-varying systems, but estimation becomes more complex.

### Missing Observations

State-space representation handles missing data elegantly:

- If $X_{it}$ is missing, remove row $i$ from $Z$ and $H$ for time $t$
- Kalman filter automatically adjusts
- No need for imputation or balanced panels

**Implementation:**
```python
# Mark missing values as np.nan
y_sim[50:60, 3] = np.nan  # Variable 3 missing for periods 50-60
y_sim[100:110, [5, 7]] = np.nan  # Variables 5 and 7 missing

# Kalman filter will skip these observations in the update step
```

---

## 6. Identifiability in State-Space Form

### The Rotation Problem

State-space representation does not uniquely identify factors. For any invertible $r \times r$ matrix $G$:

$$X_t = \Lambda F_t + e_t = (\Lambda G)(G^{-1} F_t) + e_t = \tilde{\Lambda} \tilde{F}_t + e_t$$

**Implication:** Infinitely many equivalent state-space representations.

### Standard Normalizations

**Option 1: Loading Restrictions**
- Set $\Lambda' \Lambda = I$ (orthogonal loadings)
- Set $Q$ free
- First $r \times r$ block of $\Lambda$ is lower triangular with positive diagonal

**Option 2: Factor Variance Restrictions**
- Set $Q = I$ (uncorrelated unit-variance innovations)
- Set $\Lambda$ free (up to sign)

**Option 3: Mixed (Stock-Watson)**
- Set $\text{diag}(Q) = 1$ (unit variances)
- Set $Q$ has non-negative off-diagonal elements
- Identify $\Lambda$ through ordering

### Why It Matters

- Different normalizations give different numerical answers
- Economic interpretation depends on identification scheme
- Must be consistent across estimation and impulse responses

---

## 7. Visualizing State-Space Structure

### Graphical Representation

```
Time t-1                      Time t                       Time t+1
--------                      ------                       --------

α_{t-1}  ----[T]---->  α_t  ----[T]---->  α_{t+1}
  |                     |                     |
  |                     |                     |
 [η]                   [η]                   [η]
                        |
                        |
                       [Z]
                        |
                        ↓
                       y_t
                        |
                        |
                       [ε]

Legend:
  α : State vector (latent factors + lags)
  y : Observations (measured variables)
  T : Transition matrix (dynamics)
  Z : Measurement matrix (loadings)
  η : State innovations
  ε : Measurement errors
```

### Information Flow

1. **State evolution:** $\alpha_t$ depends only on $\alpha_{t-1}$ and current shock $\eta_t$ (Markov property)
2. **Measurement:** $y_t$ depends only on current state $\alpha_t$ and measurement error $\epsilon_t$
3. **Filtering:** To estimate $\alpha_t$, use all past $y_1, ..., y_t$
4. **Smoothing:** To improve estimate of $\alpha_t$, use all data $y_1, ..., y_T$ (include future)

---

## Common Pitfalls

### 1. Dimension Mismatch in Companion Form

**Problem:** Forgetting that state dimension is $m = rp$, not $r$.

**Solution:**
- Always check: `Z.shape[1] == T.shape[0] == T.shape[1] == R.shape[0]`
- For VAR(p), state dimension is $rp$

### 2. Incorrect Companion Matrix Structure

**Problem:** Misplacing identity matrices in lower blocks of $T$.

**Example of error:**
```python
# WRONG: Identity in wrong position
T[r:2*r, 0:r] = np.eye(r)  # Should be T[r:2*r, r:2*r]
```

**Correct structure:**
- First row: $[\Phi_1, \Phi_2, ..., \Phi_p]$
- Remaining rows: Identity matrices shift lags down

### 3. Forgetting to Zero Out Loading Blocks

**Problem:** In companion form, lagged factors should not load on observables.

**Solution:**
```python
Z = np.zeros((N, r*p))
Z[:, :r] = Lambda  # Only first r columns are non-zero
```

### 4. Mismatching Innovation Dimensions

**Problem:** Setting $R$ to identity when it should be $rp \times r$ selector.

**Solution:**
```python
R = np.zeros((r*p, r))
R[:r, :] = np.eye(r)  # Innovations only enter first r states
```

### 5. Non-Stationary Initial Conditions

**Problem:** Starting Kalman filter with $P_1 = 0$ or arbitrary small value.

**Solution:**
- Use unconditional state covariance: $P_1 = \text{Var}(\alpha_t)$
- For stationary VAR: solve discrete Lyapunov equation
- For non-stationary: use "diffuse initialization" (large variance)

---

## Connections

### Builds On
- **Dynamic Factor Models** (Previous Guide): VAR dynamics for factors
- **Vector Autoregressions**: Companion form technique
- **Multivariate Normal Theory**: Conditional distributions

### Leads To
- **Kalman Filter** (Next Guide): Recursive estimation using state-space form
- **Maximum Likelihood** (Module 4): Likelihood computation via Kalman filter
- **Forecasting**: Multi-step predictions from state-space

### Related To
- **ARIMA Models**: Special case with $Z = [1, 0, ..., 0]$
- **Structural Time Series Models**: Decompose into trend, seasonal, cycle
- **Hidden Markov Models**: Discrete state space instead of continuous

---

## Practice Problems

### Conceptual

1. **State vs Observables**
   - Why include lagged factors in the state vector for VAR(p) when they don't appear in the measurement equation?
   - What would happen if we tried to write VAR(2) dynamics without companion form?

2. **Missing Data**
   - Explain why state-space models handle missing data naturally.
   - Would missing data be more problematic in measurement or transition equation? Why?

3. **Identifiability**
   - If you rotate factors by $G$, how do $Z$ and $T$ change?
   - Why doesn't $T$ need identification restrictions even though $Z$ does?

### Mathematical

4. **Companion Form Construction**
   - Write the full state-space representation for:
     $$F_t = 0.6 F_{t-1} + 0.3 F_{t-2} - 0.1 F_{t-3} + \eta_t$$
     with $r = 1$ factor and $N = 5$ variables.
   - What is the dimension of each matrix?

5. **Eigenvalue Analysis**
   - For companion matrix $T$, relate eigenvalues of $T$ to roots of VAR characteristic polynomial.
   - Show that stationarity of VAR is equivalent to $|\lambda_i(T)| < 1$.

6. **Covariance Propagation**
   - Derive $\text{Var}(\alpha_t) = T \text{Var}(\alpha_{t-1}) T' + R Q R'$ from the transition equation.
   - For stationary process, show this satisfies a discrete Lyapunov equation.

### Implementation

7. **Build State-Space from Scratch**
   ```python
   # Specifications:
   # - N = 15 variables
   # - r = 3 factors
   # - VAR(3) dynamics
   # - Exact factor model (diagonal Sigma_e)

   # Tasks:
   # 1. Define Lambda, Phi_1, Phi_2, Phi_3, Sigma_e, Q
   # 2. Construct state-space matrices Z, H, T, R, Q
   # 3. Verify dimensions
   # 4. Check stationarity of T
   ```

8. **Simulate and Reconstruct**
   ```python
   # 1. Simulate 500 periods from state-space model
   # 2. Given only simulated y (not states), verify:
   #    - Sample covariance matches implied covariance
   #    - Autocovariances match VAR-implied structure
   # 3. Compare to direct DFM simulation (without state-space)
   ```

9. **Missing Data Handling**
   ```python
   # 1. Simulate data with 20% missing values (random)
   # 2. Implement simple Kalman filter (next guide preview)
   # 3. Show that factor estimates still converge
   # 4. Compare to listwise deletion approach
   ```

### Extension

10. **Generalized State-Space**
    - Extend state-space to include measurement equation with lags:
      $$X_t = \Lambda_0 F_t + \Lambda_1 F_{t-1} + e_t$$
    - How would you augment the state vector?
    - Write the modified measurement matrix $Z$.

11. **Factor-Augmented VAR (FAVAR)**
    - Some variables are treated as observables in both measurement and transition:
      $$\begin{bmatrix} Y_t \\ X_t \end{bmatrix} = \begin{bmatrix} I \\ \Lambda \end{bmatrix} \begin{bmatrix} Y_t \\ F_t \end{bmatrix} + \begin{bmatrix} 0 \\ e_t \end{bmatrix}$$
      $$\begin{bmatrix} Y_t \\ F_t \end{bmatrix} = \Phi \begin{bmatrix} Y_{t-1} \\ F_{t-1} \end{bmatrix} + \begin{bmatrix} \eta_t^Y \\ \eta_t^F \end{bmatrix}$$
    - Express this in standard state-space form.
    - What complications arise for identification?

---

## Further Reading

### Essential
- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods.* 2nd ed. Chapters 3-4.
  - Definitive reference on state-space models and Kalman filter.

- **Harvey, A.C. (1989).** *Forecasting, Structural Time Series Models and the Kalman Filter.* Chapters 3-4.
  - Clear exposition of state-space framework for time series.

### Recommended
- **Hamilton, J.D. (1994).** *Time Series Analysis.* Chapter 13.
  - State-space representation with economic applications.

- **Commandeur, J.J.F. & Koopman, S.J. (2007).** *An Introduction to State Space Time Series Analysis.*
  - Accessible introduction with practical examples in R.

### Advanced
- **Aoki, M. (1987).** *State Space Modeling of Time Series.* Springer.
  - Advanced treatment of identification and estimation.

- **Anderson, B.D.O. & Moore, J.B. (2005).** *Optimal Filtering.* Dover.
  - Engineering perspective on state-space methods and Kalman filter.

---

**Next Guide:** Kalman Filter Derivation - optimal recursive estimation of the state vector with full mathematical intuition.

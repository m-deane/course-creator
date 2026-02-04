# State-Space Mathematics: Full Treatment

## TL;DR

State-space models provide a unified framework for representing dynamic systems with hidden states. This guide covers the complete mathematical theory, from basic definitions through identification conditions and asymptotic properties.

---

## 1. The State-Space Framework

### 1.1 General Form

A **linear Gaussian state-space model** consists of two equations:

**State equation (transition):**
```
α_t = T_t α_{t-1} + R_t η_t,    η_t ~ N(0, Q_t)
```

**Observation equation (measurement):**
```
y_t = Z_t α_t + d_t + ε_t,    ε_t ~ N(0, H_t)
```

Where:
- α_t ∈ ℝʳ: unobserved state vector (dimension r)
- y_t ∈ ℝᴺ: observed data vector (dimension N)
- T_t ∈ ℝʳˣʳ: state transition matrix
- Z_t ∈ ℝᴺˣʳ: observation (design) matrix
- R_t ∈ ℝʳˣᵍ: state disturbance selector matrix
- d_t ∈ ℝᴺ: observation intercept (often 0)
- η_t ∈ ℝᵍ: state disturbance (innovation)
- ε_t ∈ ℝᴺ: observation noise (measurement error)
- Q_t ∈ ℝᵍˣᵍ: state disturbance covariance
- H_t ∈ ℝᴺˣᴺ: observation noise covariance

### 1.2 Time-Invariant Case

For most applications, the matrices are constant over time:
```
α_t = T α_{t-1} + R η_t
y_t = Z α_t + ε_t
```

This is the **stationary state-space model**.

### 1.3 Initial Conditions

The state must be initialized:

**Known initial state:**
```
α_0 = a_0  (fixed)
```

**Uncertain initial state:**
```
α_0 ~ N(a_0, P_0)
```

**Diffuse initialization:** When P_0 → ∞ (completely uncertain)
```
α_0 ~ N(a_0, κI),  κ → ∞
```

Used when you have no prior information about initial state.

---

## 2. Markov Properties

### 2.1 First-Order Markov Property

The state equation is **first-order Markov**:
```
p(α_t | α_{t-1}, α_{t-2}, ..., α_1, α_0) = p(α_t | α_{t-1})
```

The future state depends only on the current state, not the entire history.

**Higher-order dynamics:** Can be rewritten as first-order by augmenting the state:
```
VAR(2):  α_t = T_1 α_{t-1} + T_2 α_{t-2} + η_t

Augmented: [α_t  ]   [T_1  T_2] [α_{t-1}]   [η_t]
           [α_{t-1}] = [I    0 ] [α_{t-2}] + [0  ]
```

### 2.2 Conditional Independence

Given the state α_t:
- Current observation y_t is independent of past observations: y_t ⊥ y_{1:t-1} | α_t
- Current observation is independent of future states: y_t ⊥ α_{t+1:T} | α_t

This is the **screening property** of hidden Markov models.

---

## 3. Likelihood Function

### 3.1 Prediction Error Decomposition

The log-likelihood has a recursive form:

```
log L(θ | y₁, ..., y_T) = -NT/2 log(2π) - 1/2 ∑ᵗ₌₁ᵀ [log|F_t| + v_t' F_t⁻¹ v_t]
```

Where:
- **Prediction error:** v_t = y_t - E[y_t | y_{1:t-1}] = y_t - Z α̂_{t|t-1}
- **Prediction error variance:** F_t = Var[v_t] = Z P_{t|t-1} Z' + H

This is computed recursively via the **Kalman filter**.

### 3.2 Connection to Innovations

The likelihood can be written as:
```
L(θ) = ∏ᵗ₌₁ᵀ p(y_t | y_{1:t-1}; θ)
```

Each term p(y_t | y_{1:t-1}) is Gaussian:
```
y_t | y_{1:t-1} ~ N(ŷ_{t|t-1}, F_t)
```

Where ŷ_{t|t-1} = Z α̂_{t|t-1} is the **one-step-ahead forecast**.

### 3.3 Concentrated Likelihood

For some parameters, we can "concentrate out" parts of the likelihood.

**Example:** If only T and Q are unknown, but Z, H are known:
```
log L(T, Q | y, Z, H) = -1/2 ∑ᵗ [log|F_t| + v_t' F_t⁻¹ v_t]
```

This reduces the dimension of the optimization problem.

---

## 4. Identification

### 4.1 The Identification Problem

**Non-uniqueness:** Multiple parameter sets can generate the same likelihood.

**Example:** For any invertible matrix R ∈ ℝʳˣʳ:
```
If (Z, α_t) generates y_t, then so does (ZR, R⁻¹α_t)
```

Proof:
```
y_t = Z α_t = Z R R⁻¹ α_t = Z* α_t*
```

### 4.2 Identification Strategies

#### Strategy 1: Normalization Constraints

**Fix part of the loading matrix:**
```
Z = [I_r]    (first r rows = identity)
    [Λ  ]    (remaining rows = free parameters)
```

This identifies factors up to ordering.

#### Strategy 2: Orthogonality Constraints

**Require factors to be orthogonal:**
```
E[α_t α_t'] = I_r  (uncorrelated, unit variance)
```

Combined with ordering by variance explained (PCA-style).

#### Strategy 3: Diagonal Covariance

**Assume:**
```
Q = diagonal
H = diagonal
```

This gives partial identification (up to rotation within blocks).

### 4.3 Observability and Controllability

**Observability:** Can we infer states from observations?

System is **observable** if the observability matrix has full rank:
```
O = [Z     ]
    [ZT    ]    has rank r
    [ZT²   ]
    [...]
    [ZTʳ⁻¹ ]
```

If not observable, some state dimensions are unidentifiable from data.

**Controllability:** Can state innovations affect all state dimensions?

System is **controllable** if the controllability matrix has full rank:
```
C = [R, TR, T²R, ..., Tʳ⁻¹R]  has rank r
```

If not controllable, some state dimensions never change (deterministic).

**Theorem (Kalman):** For identification, we need **both** observability and controllability.

---

## 5. Stationarity and Stability

### 5.1 State Stationarity

The state process α_t is **stationary** if:
```
E[α_t] = constant
Cov[α_t, α_{t-k}] depends only on k, not t
```

**Condition:** All eigenvalues of T lie inside the unit circle:
```
|λᵢ(T)| < 1  for all i
```

If this holds, the state has unconditional moments:
```
E[α_t] = 0  (assuming centered)
Var[α_t] = P_∞  (steady-state covariance)
```

Where P_∞ solves the **Lyapunov equation:**
```
P_∞ = T P_∞ T' + R Q R'
```

### 5.2 Observation Stationarity

If α_t is stationary, then y_t is also stationary:
```
E[y_t] = Z E[α_t] = 0
Var[y_t] = Z P_∞ Z' + H
```

The autocovariance function:
```
Γ_y(k) = Cov[y_t, y_{t-k}] = Z Tᵏ P_∞ Z'  for k ≥ 0
```

This is the **impulse response** to state shocks.

### 5.3 Explosive vs. Stationary

**Stationary (stable):** |λᵢ(T)| < 1
- Shocks die out over time
- Long-run forecasts converge to mean
- Variance bounded

**Unit root:** |λᵢ(T)| = 1 for some i
- Shocks have permanent effects
- No long-run mean
- Variance grows linearly

**Explosive:** |λᵢ(T)| > 1 for some i
- Shocks amplified over time
- Variance grows exponentially
- Not economically plausible (usually)

---

## 6. Representation Theorems

### 6.1 ARMA as State-Space

Any ARMA(p,q) can be written in state-space form.

**Example: ARMA(2,1)**
```
y_t = φ_1 y_{t-1} + φ_2 y_{t-2} + ε_t + θ_1 ε_{t-1}
```

**State-space representation:**
```
State: α_t = [y_t        ]
             [y_{t-1}     ]
             [ε_t         ]

Transition: α_t = [φ_1  φ_2  θ_1] α_{t-1} + [1] ε_t
                  [1    0    0  ]           [0]
                  [0    0    0  ]           [1]

Observation: y_t = [1  0  0] α_t
```

### 6.2 VAR as State-Space

VAR(p) with N variables:
```
y_t = Φ_1 y_{t-1} + ... + Φ_p y_{t-p} + ε_t
```

**Companion form:**
```
State: α_t = [y_t    ]
             [y_{t-1} ]     (dimension Np × 1)
             [...]
             [y_{t-p+1}]

Transition: [Φ_1  Φ_2  ...  Φ_p  ]
            [I_N  0    ...  0    ]
            [0    I_N  ...  0    ]
            [0    0    ...  I_N  0]

Observation: y_t = [I_N  0  ...  0] α_t
```

### 6.3 Dynamic Factor Model

The dynamic factor model has natural state-space form:

**Observation equation:**
```
y_t = Λ f_t + ε_t,    ε_t ~ N(0, H)
```

**State equation (factor dynamics):**
```
f_t = T f_{t-1} + η_t,    η_t ~ N(0, Q)
```

Or with factor lags in state:
```
State: α_t = [f_t  ]
             [f_{t-1}]
             [...]

This allows VAR(p) dynamics for factors.
```

---

## 7. Information Sets and Projections

### 7.1 Notation for Conditioning

Let Y_t = {y_1, y_2, ..., y_t} be the information available at time t.

**Filtered estimate:** E[α_t | Y_t] = α̂_{t|t}
- Uses data up to time t
- "Real-time" estimate
- Causal (can be computed online)

**Predicted estimate:** E[α_t | Y_{t-1}] = α̂_{t|t-1}
- Uses data up to time t-1
- One-step-ahead forecast
- Also causal

**Smoothed estimate:** E[α_t | Y_T] = α̂_{t|T}
- Uses all available data (T > t)
- Best possible estimate
- Non-causal (requires full sample)

**Forecast:** E[α_{t+h} | Y_t] = α̂_{t+h|t}
- h-step-ahead forecast
- Uncertainty grows with h

### 7.2 Projection Matrices

The Kalman filter implicitly computes projection matrices:

**Filtered state:**
```
α̂_{t|t} = E[α_t | Y_t] = Π_t α_t
```
Where Π_t is the projection onto the space spanned by {y_1, ..., y_t}.

**Key property:** Projections are linear in Gaussian case:
```
α̂_{t|t} = ∑ₛ₌₁ᵗ K_{t,s} y_s
```

### 7.3 Innovations Representation

The **innovations** are the unpredictable part of y_t:
```
v_t = y_t - E[y_t | Y_{t-1}] = y_t - Z α̂_{t|t-1}
```

**Key facts:**
1. Innovations are white noise: E[v_t v_s'] = 0 for t ≠ s
2. Original data can be recovered: y_t = E[y_t | Y_{t-1}] + v_t
3. Likelihood factors through innovations

This is the **Wold decomposition** for state-space models.

---

## 8. Large-Sample Theory

### 8.1 Consistency of MLE

Under regularity conditions (identification, stationarity, ergodicity):

**Theorem:** The MLE θ̂_T converges to true parameter θ₀:
```
θ̂_T →ᵖ θ₀  as T → ∞
```

**Conditions:**
1. Model is identified
2. True parameter in interior of parameter space
3. State process is stationary and ergodic
4. Observations are bounded moments

### 8.2 Asymptotic Normality

**Theorem:** The MLE is asymptotically normal:
```
√T(θ̂_T - θ₀) →ᵈ N(0, I(θ₀)⁻¹)
```

Where I(θ) is the **Fisher information matrix:**
```
I(θ) = -E[∂²log L(θ)/∂θ∂θ']
```

**Practical implication:** Standard errors are:
```
SE(θ̂ᵢ) ≈ √(I(θ̂)⁻¹)ᵢᵢ / √T
```

### 8.3 Information Matrix Computation

Two approaches:

**Outer product of score (OPG):**
```
I_OPG = (1/T) ∑ᵗ (∂log p(y_t|Y_{t-1})/∂θ)(∂log p(y_t|Y_{t-1})/∂θ)'
```

**Hessian:**
```
I_Hess = -(1/T) ∂²log L(θ)/∂θ∂θ'
```

Both are consistent estimators of I(θ).

**Numerical computation:** Use finite differences or automatic differentiation.

---

## 9. Model Selection

### 9.1 Factor Number Selection

How many factors r?

**Scree plot:** Plot eigenvalues, look for "elbow"
- Subjective but intuitive
- Works well when true r is clear

**Information criteria:** Balance fit vs. complexity

**Akaike (AIC):**
```
AIC(r) = -2 log L(θ̂) + 2k
```
Where k = number of free parameters.

**Bayesian (BIC):**
```
BIC(r) = -2 log L(θ̂) + k log(T)
```
Stronger penalty than AIC, more consistent.

**Hannan-Quinn (HQ):**
```
HQ(r) = -2 log L(θ̂) + 2k log(log(T))
```

**Bai-Ng criteria (2002):** Specialized for factor models:
```
IC_p1(r) = log V(r) + r·(N+T)/(NT)·log(NT/(N+T))
IC_p2(r) = log V(r) + r·(N+T)/(NT)·log(C_{NT})
IC_p3(r) = log V(r) + r·(log(C_{NT})/C_{NT})
```
Where V(r) is sum of squared residuals, C_{NT} = min(N,T).

### 9.2 Lag Order Selection

For factor dynamics (VAR order p):

Use standard VAR criteria on estimated factors:
```
AIC(p) = log |Σ̂(p)| + 2(r²p)/T
BIC(p) = log |Σ̂(p)| + r²p·log(T)/T
```

**Sequential testing:** LR test for H₀: p = p₀ vs. H₁: p = p₀ + 1
```
LR = T(log |Σ̂(p₀)| - log |Σ̂(p₀+1)|) ~ χ²(r²)
```

---

## 10. Relationship to Other Models

### 10.1 State-Space Encompasses Many Models

| Classical Model | State-Space Form |
|----------------|------------------|
| AR(p) | Companion form with r = p |
| MA(q) | Non-minimal form with r = q+1 |
| ARMA(p,q) | Mixed form with r = max(p, q+1) |
| Structural time series | Components as states |
| Unobserved components | Trend/cycle/seasonal as states |
| Factor model (static) | Factors as random walk states |
| Factor model (dynamic) | Factors follow VAR |

### 10.2 Canonical Correlations

The relationship between y_t and α_t is characterized by **canonical correlations**:

Correlation structure:
```
Cor[y_t, α_t] = canonical correlations between observations and states
```

Maximum is achieved when H = 0 (perfect measurement).

---

## 11. Advanced Topics

### 11.1 Time-Varying Parameters

Allow matrices to change over time:
```
T_t = T + B_t,  B_t ~ N(0, W)
```

**Random walk parameters:**
```
θ_t = θ_{t-1} + u_t
```

This is itself a state-space model with **augmented state**.

### 11.2 Non-Gaussian State-Space

When η_t or ε_t are non-Gaussian:
- Kalman filter no longer optimal
- Use **Extended Kalman Filter** (EKF) or **Particle Filter**
- Likelihood requires numerical integration

**Example:** t-distributed errors for robustness.

### 11.3 Nonlinear State-Space

Generalization:
```
α_t = g(α_{t-1}, η_t)    (nonlinear state transition)
y_t = h(α_t, ε_t)        (nonlinear observation)
```

Require **particle filters** or **sequential Monte Carlo** for inference.

---

## 12. Computational Complexity

### 12.1 Kalman Filter Complexity

Per time step:
- **Matrix multiplication:** O(r²N + r³)
- **Inversion:** O(N³) for F_t or O(r³) using Woodbury

For T time steps:
```
Total: O(T(r²N + N³))  or  O(T(r²N + r³))  if N > r
```

### 12.2 Optimization Complexity

EM algorithm iteration:
- **E-step:** Run Kalman filter + smoother: O(T·r²·N)
- **M-step:** Closed-form updates: O(T·r²·N)

Quasi-Newton (BFGS):
- **Gradient:** Automatic differentiation: O(T·r²·N·k) where k = # parameters
- **Hessian:** Numerical approximation: O(k²)

**Rule of thumb:** EM is O(T·r²·N) per iteration, needs 10-50 iterations.

---

## References

### Foundational Theory
- **Kalman, R.E. (1960).** "A New Approach to Linear Filtering and Prediction Problems." *Journal of Basic Engineering*.
- **Anderson, B.D.O. & Moore, J.B. (1979).** *Optimal Filtering*. Prentice-Hall.
- **Harvey, A.C. (1989).** *Forecasting, Structural Time Series Models and the Kalman Filter*. Cambridge University Press.

### Econometric Applications
- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods*. 2nd ed. Oxford University Press.
- **Hamilton, J.D. (1994).** *Time Series Analysis*. Princeton University Press.
- **Commandeur, J.J.F. & Koopman, S.J. (2007).** *An Introduction to State Space Time Series Analysis*. Oxford University Press.

### Dynamic Factor Models
- **Stock, J.H. & Watson, M.W. (2016).** "Dynamic Factor Models." *Handbook of Macroeconomics*.
- **Bai, J. & Ng, S. (2008).** "Large Dimensional Factor Analysis." *Foundations and Trends in Econometrics*.
- **Doz, C., Giannone, D. & Reichlin, L. (2011).** "A Two-Step Estimator for Large Approximate Dynamic Factor Models." *Journal of Econometrics*.

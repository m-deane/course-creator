# Module 2 Quiz: Dynamic Factor Models

**Time Estimate:** 50-65 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of dynamic factor models, state-space representation, and Kalman filtering. Topics include the transition from static to dynamic models, state-space formulation, and the distinction between filtering and smoothing. Answer all questions. You have 2 attempts, and your highest score will be recorded.

---

## Part A: Conceptual Understanding (40 points)

### Question 1 (4 points) - Foundation

What is the key difference between static and dynamic factor models?

A) Dynamic models allow factors to be correlated; static models require independence
B) Dynamic models include time-series structure in factors (e.g., VAR); static models treat factors as i.i.d. over time
C) Dynamic models use more factors than static models
D) Static models can only be estimated via PCA; dynamic models require Kalman filtering

**Correct Answer:** B

**Feedback:**
- A: Both static and dynamic models can have correlated factors (via $\Sigma_F$). The distinction is temporal, not cross-sectional.
- B: **Correct**. Dynamic factor models explicitly model factor evolution: $F_t = \Phi F_{t-1} + \eta_t$. Static models treat each $F_t$ as independent draws.
- C: The number of factors ($r$) is independent of whether the model is static or dynamic.
- D: PCA can be used for both (first step of Stock-Watson), and dynamic models can also use PCA. Kalman filtering is one estimation method for dynamic models, not the only one.

---

### Question 2 (4 points) - Core

In a dynamic factor model with VAR(1) factor dynamics: $F_t = \Phi F_{t-1} + \eta_t$, if $\Phi = 0.9 I_r$, what does this imply?

A) Factors are highly persistent and slowly mean-reverting
B) Factors follow a random walk
C) Factors are perfectly correlated
D) Factors are non-stationary

**Correct Answer:** A

**Feedback:**
- A: **Correct**. With $\Phi = 0.9 I$, each factor follows $F_{jt} = 0.9 F_{j,t-1} + \eta_{jt}$. High autocorrelation (0.9) means shocks decay slowly, creating persistence.
- B: A random walk requires $\Phi = I$ (unit root), not $\Phi = 0.9I$.
- C: $\Phi = 0.9 I$ (diagonal) means factors evolve independently, not correlated. Cross-factor correlation would require off-diagonal elements.
- D: Stationarity requires $|\lambda(\Phi)| < 1$. Here eigenvalues are 0.9, so factors are stationary.

---

### Question 3 (4 points) - Core

The state-space representation of a dynamic factor model consists of two equations. What do they represent?

A) Measurement equation (observables) and transition equation (state dynamics)
B) Loading equation and innovation equation
C) Cross-sectional equation and time-series equation
D) Factor equation and error equation

**Correct Answer:** A

**Feedback:**
- A: **Correct**. Measurement equation: $X_t = Z\alpha_t + \varepsilon_t$ (how latent state relates to observables). Transition equation: $\alpha_t = T\alpha_{t-1} + R\eta_t$ (how state evolves).
- B: These are not standard state-space terminology.
- C: Both equations involve time series; there's no purely "cross-sectional" equation.
- D: Too vague; doesn't capture the state-space structure.

---

### Question 4 (4 points) - Core

What is the **companion form** used for in dynamic factor models?

A) To estimate factors and loadings jointly
B) To convert VAR(p) dynamics into VAR(1) form for state-space representation
C) To handle missing data in the Kalman filter
D) To identify structural shocks

**Correct Answer:** B

**Feedback:**
- A: Joint estimation is not specific to companion form; it's about representation, not estimation method.
- B: **Correct**. VAR(p) with $p > 1$ doesn't fit standard state-space (which requires VAR(1)). Companion form stacks current and lagged factors as an augmented state: $\alpha_t = [F_t', F_{t-1}', ..., F_{t-p+1}']'$, converting VAR(p) to VAR(1).
- C: Missing data handling uses the Kalman filter structure, not specifically companion form.
- D: Structural identification requires additional restrictions, not just companion form.

---

### Question 5 (4 points) - Advanced

In the Kalman filter, what is the distinction between **filtered** and **smoothed** estimates of the state?

A) Filtered uses maximum likelihood; smoothed uses Bayesian methods
B) Filtered uses information up to time $t$; smoothed uses all information (past and future relative to $t$)
C) Filtered applies to factors; smoothed applies to loadings
D) Filtered removes outliers; smoothed averages over time

**Correct Answer:** B

**Feedback:**
- A: Both filtered and smoothed estimates can be computed within frequentist or Bayesian frameworks; this is not the distinction.
- B: **Correct**. Filtered estimate $\hat{\alpha}_{t|t}$ uses data $\{X_1, ..., X_t\}$ (real-time). Smoothed estimate $\hat{\alpha}_{t|T}$ uses all data $\{X_1, ..., X_T\}$ (hindsight), providing better estimates for $t < T$.
- C: Both filtering and smoothing apply to state estimates (factors), not loadings.
- D: Neither inherently removes outliers; this confuses filtering with outlier detection.

---

### Question 6 (4 points) - Core

What does the **Kalman gain** $K_t$ control in the Kalman filter update step?

A) The speed of convergence to the steady state
B) How much the state estimate is adjusted based on the prediction error
C) The variance of measurement errors
D) The number of factors to extract

**Correct Answer:** B

**Feedback:**
- A: Convergence speed is related but not the primary role of $K_t$.
- B: **Correct**. The update is $\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t v_t$, where $v_t$ is the prediction error. $K_t$ determines how much to "trust" the new observation vs. the prior prediction.
- C: $K_t$ is computed from variances but doesn't control them; $H$ (in the system matrices) sets measurement error variance.
- D: The number of factors is a modeling choice, not determined by $K_t$.

---

### Question 7 (4 points) - Advanced

For a VAR(2) factor model with $r = 3$ factors, what is the dimension of the state vector $\alpha_t$ in companion form?

A) 3
B) 6
C) 9
D) 12

**Correct Answer:** B

**Feedback:**
- A: This would be the dimension for VAR(1), not VAR(2).
- B: **Correct**. Companion form stacks $p$ lags: $\alpha_t = [F_t', F_{t-1}']'$ has dimension $r \times p = 3 \times 2 = 6$.
- C: This would be for $p = 3$ lags, not 2.
- D: This misunderstands the stacking; it's $r \times p$, not $r^2 \times p$ or similar.

---

### Question 8 (4 points) - Core

Why do we need **initial conditions** ($\alpha_0$, $P_0$) for the Kalman filter?

A) To identify the factors uniquely
B) To start the recursive filter; subsequent estimates update from these
C) To ensure factors are stationary
D) To set the number of factors

**Correct Answer:** B

**Feedback:**
- A: Identification comes from parameter restrictions (loadings, etc.), not initial conditions.
- B: **Correct**. The Kalman filter is recursive: $\hat{\alpha}_{t|t}$ depends on $\hat{\alpha}_{t-1|t-1}$. We need $\alpha_0$ and its uncertainty $P_0$ to bootstrap the recursion.
- C: Stationarity is a property of the transition matrix $T$, not initial conditions.
- D: The number of factors is a modeling choice set before filtering.

---

### Question 9 (4 points) - Advanced

In the state-space form, if the measurement matrix $Z = [\Lambda, 0, ..., 0]$ for a VAR(p) model in companion form, why are lagged factor blocks zero?

A) Lagged factors are not observable
B) Only current factors $F_t$ directly affect observables $X_t$ in the measurement equation
C) This is an identification restriction
D) It's a computational simplification, not theoretically required

**Correct Answer:** B

**Feedback:**
- A: While true that factors are latent, this doesn't explain why lagged factors have zero loadings.
- B: **Correct**. The measurement equation is $X_t = \Lambda F_t + e_t$, not $X_t = \Lambda_0 F_t + \Lambda_1 F_{t-1} + ...$. Only current factors enter unless we explicitly model distributed lags in loadings.
- C: This is structural, not an identification restriction; it's how the model is specified.
- D: It's not a simplification—it's the correct specification given the DFM structure.

---

### Question 10 (4 points) - Foundation

An impulse response function (IRF) for a dynamic factor model measures:

A) The correlation between factors over time
B) The effect of a one-unit shock to a factor at time $t$ on that factor (and others) at time $t+h$
C) The loading of a variable on a factor
D) The forecast error variance at horizon $h$

**Correct Answer:** B

**Feedback:**
- A: This describes autocovariance or cross-covariance, not IRF.
- B: **Correct**. IRF traces the dynamic response: $\text{IRF}_j(h) = \frac{\partial F_{t+h}}{\partial \eta_{jt}}$. For VAR(1), this is $\Phi^h$.
- C: Loadings are static relationships in the measurement equation, not dynamic responses.
- D: This describes forecast error variance decomposition, a related but distinct concept.

---

## Part B: Mathematical Application (30 points)

### Question 11 (6 points) - Core

Consider a single-factor ($r=1$) VAR(1) model: $F_t = 0.8 F_{t-1} + \eta_t$ with $\text{Var}(\eta_t) = 1$. What is the unconditional variance $\text{Var}(F_t)$?

A) 1.00
B) 1.25
C) 2.78
D) 5.56

**Correct Answer:** C

**Feedback:**
- A: This would be true if $\phi = 0$ (white noise), not 0.8.
- B: This is $1/(1 - 0.8) = 1/0.2 = 5$, but that's not the correct formula.
- C: **Correct**. For VAR(1), $\Sigma_F = \Phi \Sigma_F \Phi' + Q$. With $\Phi = 0.8$ and $Q = 1$: $\Sigma_F = 0.64 \Sigma_F + 1$, so $\Sigma_F = 1/(1 - 0.64) = 1/0.36 \approx 2.78$.
- D: This doubles the correct answer.

**Formula:** $\Sigma_F = Q / (1 - \Phi^2) = 1 / (1 - 0.8^2) = 1/0.36 = 2.78$.

---

### Question 12 (6 points) - Core

For a 2-factor VAR(1): $F_t = \Phi F_{t-1} + \eta_t$ with
$$\Phi = \begin{bmatrix} 0.7 & 0.2 \\ 0.1 & 0.6 \end{bmatrix}$$

Is this system stationary?

A) Yes, because all elements of $\Phi$ are < 1
B) No, because $\Phi$ is not diagonal
C) Need to check eigenvalues of $\Phi$; if all $|\lambda_i| < 1$, then stationary
D) Cannot determine without knowing $Q$

**Correct Answer:** C

**Feedback:**
- A: Individual element magnitudes don't determine stationarity; need eigenvalues of the matrix.
- B: Non-diagonal doesn't imply non-stationarity; it just means factors have cross-dynamics.
- C: **Correct**. Stationarity requires all eigenvalues of $\Phi$ inside the unit circle. For this $\Phi$: eigenvalues are approximately 0.82 and 0.48 (both < 1), so it's stationary.
- D: $Q$ affects variance magnitude, not stationarity; only $\Phi$ determines stability.

---

### Question 13 (6 points) - Advanced

In the Kalman filter, the **prediction error** at time $t$ is $v_t = X_t - Z\hat{\alpha}_{t|t-1}$. Its variance is $F_t = ZP_{t|t-1}Z' + H$. If signal-to-noise ratio is very high (large factor variance, small $H$), what happens to $F_t$?

A) $F_t \to 0$ (high precision)
B) $F_t \approx H$ (dominated by measurement noise)
C) $F_t \approx ZP_{t|t-1}Z'$ (dominated by state uncertainty)
D) Cannot determine without knowing $T$

**Correct Answer:** C

**Feedback:**
- A: $F_t$ cannot go to zero; it's at least $H$.
- B: This describes low signal-to-noise (small state variance, large $H$), the opposite scenario.
- C: **Correct**. With small $H$, measurement error is negligible, so prediction error variance is dominated by uncertainty about the state: $F_t \approx ZP_{t|t-1}Z'$.
- D: The sample size $T$ doesn't directly enter this instantaneous variance calculation.

---

### Question 14 (6 points) - Advanced

For VAR(1), the 1-step-ahead IRF is $\Phi$, and the 2-step-ahead IRF is $\Phi^2$. What is the **cumulative** effect of a shock over the first 10 periods?

A) $\Phi^{10}$
B) $10\Phi$
C) $\sum_{h=0}^{10} \Phi^h = I + \Phi + \Phi^2 + ... + \Phi^{10}$
D) $\Phi(I - \Phi^{10})^{-1}$

**Correct Answer:** C

**Feedback:**
- A: This is the 10-step-ahead IRF, not cumulative.
- B: This would be true if shocks were permanent, but VAR shocks are transitory.
- C: **Correct**. The cumulative response sums impacts at each horizon: $\text{Cum}(10) = \sum_{h=0}^{10} \Phi^h$.
- D: This resembles a geometric series formula but is incorrect (and not the cumulative sum).

---

### Question 15 (6 points) - Core

You have a VAR(3) factor model with $r = 2$. In companion form, the transition matrix $T$ is $4 \times 4$. What does the lower-right $2 \times 2$ block contain?

A) $\Phi_3$ (third lag coefficient)
B) $I_2$ (identity matrix)
C) $0$ (zero matrix)
D) $\Phi_2$ (second lag coefficient)

**Correct Answer:** C

**Feedback:**
- A: $\Phi_3$ appears in the top-right of the first row, not lower-right.
- B: Identity matrices appear on the sub-diagonal, not the corner.
- C: **Correct**. Companion form for VAR(3):
$$T = \begin{bmatrix} \Phi_1 & \Phi_2 & \Phi_3 \\ I_2 & 0 & 0 \\ 0 & I_2 & 0 \end{bmatrix}$$
The lower-right block is zero (no lag-3 states affect lag-2 states).
- D: $\Phi_2$ appears in the top row.

---

## Part C: Practical Interpretation and Implementation (30 points)

### Question 16 (6 points) - Core

You estimate a dynamic factor model and plot the filtered factor estimates $\hat{F}_{t|t}$ versus smoothed estimates $\hat{F}_{t|T}$. You notice smoothed estimates are less volatile. Why?

A) Smoothing removes outliers automatically
B) Smoothing uses future information to revise past estimates, reducing noise
C) Smoothed estimates are incorrectly computed
D) Filtering overfits to the data

**Correct Answer:** B

**Feedback:**
- A: Smoothing doesn't remove outliers; it uses additional information.
- B: **Correct**. Smoothed estimates $\hat{F}_{t|T}$ incorporate data from $t+1, ..., T$, which provides additional signal about $F_t$. This "two-sided" filtering reduces estimation error variance, making estimates smoother.
- C: If properly computed, smoothed estimates are theoretically optimal (minimum MSE) given all data.
- D: Filtering doesn't overfit; it uses only past data, which is noisier than using all data.

---

### Question 17 (6 points) - Core

You simulate a DFM with $T = 300$, $N = 50$, $r = 3$, and $\Phi = 0.8 I_3$. You estimate factors using PCA (ignoring dynamics) and via Kalman filter (exploiting dynamics). Which method should produce more accurate factor estimates?

A) PCA, because it's simpler and less prone to overfitting
B) Kalman filter, because it uses time-series information
C) They should be identical if both use the same normalization
D) Cannot determine without knowing $\Sigma_e$

**Correct Answer:** B

**Feedback:**
- A: Simplicity doesn't guarantee accuracy; PCA ignores valuable time-series structure.
- B: **Correct**. The Kalman filter exploits factor dynamics ($F_t = 0.8F_{t-1} + \eta_t$), using past factors to predict current ones. This additional information improves estimation efficiency compared to PCA, which treats each $F_t$ independently.
- C: PCA and Kalman filter use different information sets, so estimates will differ (though correlation should be high).
- D: $\Sigma_e$ affects both methods similarly; dynamics are the key differentiator.

---

### Question 18 (6 points) - Advanced

In practice, when initializing the Kalman filter for a stationary DFM, you should set $P_0$ (initial state covariance) to:

A) Zero matrix (assume factors start at their mean)
B) Identity matrix (unit variance for each factor)
C) Solution to the discrete Lyapunov equation: $P_0 = TP_0T' + RQR'$
D) A very large diagonal matrix (diffuse prior)

**Correct Answer:** C

**Feedback:**
- A: Zero variance is overconfident and incorrect; factors have uncertainty initially.
- B: Unit variance is arbitrary and unlikely to match the stationary distribution.
- C: **Correct**. For stationary VAR, the unconditional covariance $\Sigma_\alpha$ satisfies the Lyapunov equation. Starting with $P_0 = \Sigma_\alpha$ ensures filter begins at the stationary distribution, avoiding transient bias.
- D: Diffuse initialization is used for *non-stationary* or *uncertain* initial conditions, not for stationary systems.

---

### Question 19 (6 points) - Advanced

You estimate a dynamic factor model and find the first factor has autoregressive coefficient $\phi_{11} = 0.95$. A shock to this factor today would have what approximate half-life (periods until the effect decays by 50%)?

A) 1 period
B) 7 periods
C) 14 periods
D) 50 periods

**Correct Answer:** C

**Feedback:**
- A: With high persistence (0.95), shocks decay slowly, not instantly.
- B: Too short for $\phi = 0.95$.
- C: **Correct**. Half-life formula: $h = \log(0.5) / \log(\phi) = \log(0.5) / \log(0.95) \approx -0.693 / -0.051 \approx 13.5$ periods. So approximately 14 periods.
- D: This overestimates; even highly persistent shocks decay within ~15-20 periods.

---

### Question 20 (6 points) - Core

You have monthly data but want to forecast a quarterly variable (GDP). Your factor model uses monthly factors $F_t$. Which aggregation is correct for quarter $q$?

**Stock variables (e.g., debt):** End-of-quarter value
**Flow variables (e.g., GDP):** Sum or average over the quarter

For GDP (a flow), you should:

A) Use $F_{\text{last month}}$ only
B) Average factors: $\bar{F}_q = (F_1 + F_2 + F_3)/3$
C) Sum factors: $F_q = F_1 + F_2 + F_3$
D) Use maximum factor value in the quarter

**Correct Answer:** B

**Feedback:**
- A: This discards information from the first two months of the quarter.
- B: **Correct**. GDP is a flow variable measured over the quarter. The average of monthly factors corresponds to the "average activity level" in the quarter, which drives quarterly GDP. Summation (option C) would overweight and isn't the standard temporal aggregation for flows in state-space models.
- C: While sometimes used, averaging is more conventional than summing for relating monthly factors to quarterly flows.
- D: The maximum is not economically meaningful for aggregation.

---

## Part D: State-Space Representation (Bonus: 10 points)

### Question 21 (5 points) - Advanced

For the state-space model:
$$X_t = Z\alpha_t + \varepsilon_t$$
$$\alpha_t = T\alpha_{t-1} + R\eta_t$$

If $N = 10$, $r = 2$, and $p = 2$ (VAR(2)), what are the dimensions of $Z$ and $T$?

A) $Z$ is $10 \times 2$, $T$ is $2 \times 2$
B) $Z$ is $10 \times 4$, $T$ is $4 \times 4$
C) $Z$ is $10 \times 2$, $T$ is $4 \times 4$
D) $Z$ is $10 \times 4$, $T$ is $2 \times 2$

**Correct Answer:** B

**Feedback:**
- A: This would be for VAR(1), not VAR(2).
- B: **Correct**. Companion form state: $\alpha_t = [F_t', F_{t-1}']'$ has dimension $rp = 2 \times 2 = 4$. So $Z$ is $N \times (rp) = 10 \times 4$, and $T$ is $(rp) \times (rp) = 4 \times 4$.
- C: $Z$ must match state dimension (4), not just current factors (2).
- D: This reverses the dimensions.

---

### Question 22 (5 points) - Advanced

In companion form for VAR(2), the measurement matrix is $Z = [\Lambda, 0]$ where $\Lambda$ is $N \times r$ and the $0$ block is $N \times r$. The first block $\Lambda$ corresponds to current factors $F_t$. What does the zero block represent?

A) Measurement error
B) Lagged factors $F_{t-1}$ don't directly enter the measurement equation
C) Identification restrictions
D) Missing data

**Correct Answer:** B

**Feedback:**
- A: Measurement error is captured by $\varepsilon_t$ and its covariance $H$, not the zero block in $Z$.
- B: **Correct**. The measurement equation is $X_t = \Lambda F_t + e_t$. In companion form, $\alpha_t = [F_t', F_{t-1}']'$, but only $F_t$ loads on $X_t$, so $Z = [\Lambda, 0]$ with zeros for the lagged factor block.
- C: While identification matters, the zero structure here reflects the model specification, not an identification choice.
- D: Missing data is handled differently in the Kalman filter (row deletion), not by zero blocks in $Z$.

---

## Answer Key Summary

| Question | Answer | Difficulty | Topic |
|----------|--------|------------|-------|
| 1 | B | Foundation | Static vs dynamic |
| 2 | A | Core | VAR persistence |
| 3 | A | Core | State-space structure |
| 4 | B | Core | Companion form |
| 5 | B | Advanced | Filtering vs smoothing |
| 6 | B | Core | Kalman gain |
| 7 | B | Advanced | State dimension |
| 8 | B | Core | Initial conditions |
| 9 | B | Advanced | Measurement matrix |
| 10 | B | Foundation | Impulse responses |
| 11 | C | Core | Unconditional variance |
| 12 | C | Core | Stationarity |
| 13 | C | Advanced | Prediction error variance |
| 14 | C | Advanced | Cumulative IRF |
| 15 | C | Core | Companion form structure |
| 16 | B | Core | Smoothing interpretation |
| 17 | B | Core | PCA vs Kalman |
| 18 | C | Advanced | Filter initialization |
| 19 | C | Advanced | Shock persistence |
| 20 | B | Core | Temporal aggregation |
| 21 | B | Advanced | State-space dimensions |
| 22 | B | Advanced | Measurement matrix structure |

**Scoring Distribution:**
- Part A (Conceptual): 40 points
- Part B (Mathematical): 30 points
- Part C (Practical): 30 points
- Part D (Bonus): 10 points

**Difficulty Distribution:**
- Foundation: 8 points (2 questions)
- Core: 60 points (12 questions)
- Advanced: 42 points (8 questions)

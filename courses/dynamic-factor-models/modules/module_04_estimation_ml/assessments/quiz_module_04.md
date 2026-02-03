# Module 4 Quiz: Maximum Likelihood Estimation

**Time Estimate:** 55-70 minutes
**Total Points:** 100
**Passing Score:** 70%

## Instructions

This quiz assesses your understanding of maximum likelihood estimation for dynamic factor models, including the Kalman filter likelihood, EM algorithm, and Bayesian approaches. Topics include prediction error decomposition, parameter identification, and computational implementation. Answer all questions. You have 2 attempts, and your highest score will be recorded.

---

## Part A: Conceptual Understanding (40 points)

### Question 1 (4 points) - Foundation

Maximum likelihood estimation for dynamic factor models with latent states is tractable because:

A) Factors can be integrated out analytically
B) The Kalman filter provides the likelihood via prediction error decomposition
C) EM algorithm eliminates the need to compute the likelihood
D) Factors are assumed Gaussian, making integration simple

**Correct Answer:** B

**Feedback:**
- A: Direct integration over all latent factor paths $F_1, ..., F_T$ is intractable (exponential complexity).
- B: **Correct**. The Kalman filter recursively computes one-step-ahead prediction errors $v_t$ and their variances $F_t$. The likelihood decomposes as $L = \prod_{t=1}^T p(X_t | X_{1:t-1})$, each term evaluated via the filter.
- C: EM is an optimization algorithm that still requires likelihood evaluation (using Kalman filter).
- D: Gaussianity helps but doesn't make direct integration feasible; we still need the Kalman recursion.

---

### Question 2 (4 points) - Core

In the prediction error decomposition, the **innovation** $v_t = X_t - Z\hat{\alpha}_{t|t-1}$ represents:

A) The measurement error $\varepsilon_t$
B) The part of $X_t$ not predictable from past observations
C) The latent state $\alpha_t$
D) The residual after filtering

**Correct Answer:** B

**Feedback:**
- A: $v_t$ includes measurement error but also unpredictable state innovations; it's not just $\varepsilon_t$.
- B: **Correct**. $v_t$ is the "surprise" in $X_t$ given all past data $X_{1:t-1}$. It's the new information in the current observation that couldn't be forecasted.
- C: $v_t$ is in the observation space, not the state space.
- D: The filtered residual would be $X_t - Z\hat{\alpha}_{t|t}$, not the innovation.

---

### Question 3 (4 points) - Core

The prediction error variance $F_t = ZP_{t|t-1}Z' + H$. If the signal-to-noise ratio is high (factor variance large relative to measurement error), then:

A) $F_t \approx H$ (dominated by measurement noise)
B) $F_t \approx ZP_{t|t-1}Z'$ (dominated by state uncertainty)
C) $F_t \to 0$ (perfect prediction)
D) The Kalman gain $K_t \to 0$

**Correct Answer:** B

**Feedback:**
- A: This describes low signal-to-noise (small factor variance, large $H$).
- B: **Correct**. High signal-to-noise means factors drive most variation. State uncertainty $P_{t|t-1}$ is relatively large compared to measurement noise $H$, so $F_t \approx ZP_{t|t-1}Z'$.
- C: $F_t$ cannot be zero; it includes at least measurement error $H$.
- D: High signal-to-noise implies the Kalman gain is moderate to high (trusting the observations), not close to zero.

---

### Question 4 (4 points) - Core

Why is the log-likelihood typically preferred over the likelihood itself in optimization?

A) It's always positive
B) It converts products into sums, avoiding numerical underflow and simplifying derivatives
C) It's bounded between 0 and 1
D) The Kalman filter only computes log-likelihood

**Correct Answer:** B

**Feedback:**
- A: Log-likelihood is negative (since $0 < L < 1$), not positive.
- B: **Correct**. Likelihood is $L = \prod_t p(X_t | ...)$, which involves multiplying many small probabilities → underflow. Log-likelihood $\log L = \sum_t \log p(X_t | ...)$ sums terms, avoiding underflow and making gradients additive (easier to compute).
- C: Log-likelihood is typically negative and unbounded below.
- D: The Kalman filter computes terms that can be used for either; log-likelihood is preferred for numerical reasons.

---

### Question 5 (4 points) - Advanced

In the EM algorithm for factor models, the **E-step** computes:

A) Expected values of latent factors given current parameters and data
B) Eigenvalues of the covariance matrix
C) Estimates of model parameters
D) Error covariance $\Sigma_e$

**Correct Answer:** A

**Feedback:**
- A: **Correct**. The E-step computes $E[\alpha_t | X_{1:T}, \theta^{(k)}]$ (smoothed state estimates) and $E[\alpha_t \alpha_t' | X_{1:T}, \theta^{(k)}]$ (smoothed state second moments) using the Kalman smoother given current parameter estimates $\theta^{(k)}$.
- B: Eigenvalues are part of PCA, not the EM E-step.
- C: Parameters are updated in the M-step, not E-step.
- D: $\Sigma_e$ is updated in the M-step based on E-step outputs.

---

### Question 6 (4 points) - Core

The EM algorithm for factor models is guaranteed to:

A) Converge to the global maximum likelihood estimate
B) Increase (or maintain) the likelihood at each iteration
C) Converge in a finite number of steps
D) Avoid local optima

**Correct Answer:** B

**Feedback:**
- A: EM can converge to local optima, not guaranteed to find the global maximum.
- B: **Correct**. A fundamental property of EM: $L(\theta^{(k+1)}) \geq L(\theta^{(k)})$. The likelihood is non-decreasing at each iteration.
- C: EM requires iterative convergence (often 50-200 iterations), not finite steps.
- D: EM can get stuck in local optima; initialization matters.

---

### Question 7 (4 points) - Advanced

In Bayesian estimation of factor models, specifying a prior $p(\Lambda)$ on loadings can help with:

A) Computational speed
B) Identification of factors when frequentist methods face indeterminacy
C) Reducing the number of observations needed
D) Eliminating measurement error

**Correct Answer:** B

**Feedback:**
- A: Bayesian methods (MCMC) are typically slower, not faster, than frequentist ML.
- B: **Correct**. Priors can impose identification restrictions (e.g., lower triangular loadings, sign constraints) or shrinkage, helping to pin down factor interpretation and avoid rotation indeterminacy.
- C: Priors provide regularization but don't eliminate the need for data; they complement, not replace, observations.
- D: Measurement error exists regardless of estimation approach; priors don't eliminate it.

---

### Question 8 (4 points) - Core

When using the Kalman filter for likelihood evaluation, what happens if you mistakenly set $P_0 = 0$ (zero initial uncertainty)?

A) The filter will crash due to division by zero
B) The likelihood will be overestimated (too high)
C) The first few prediction errors will be overweighted, biasing the likelihood
D) No effect, as the filter adapts after the first observation

**Correct Answer:** C

**Feedback:**
- A: The filter won't crash; $F_t = ZP_{t|t-1}Z' + H$ will still be positive definite due to $H > 0$.
- B: Overestimating likelihood would require underestimating prediction errors, which doesn't follow.
- C: **Correct**. $P_0 = 0$ implies perfect knowledge of initial state, making early prediction errors appear unexpectedly large (if wrong). This distorts early likelihood terms. The filter eventually corrects, but the cumulative likelihood is biased.
- D: While the filter adapts, the likelihood accumulates over all $t$, so early bias persists in the total.

---

### Question 9 (4 points) - Advanced

"Diffuse initialization" of the Kalman filter sets:

A) $P_0 = 0$ (perfect initial knowledge)
B) $P_0 = I$ (unit variance)
C) $P_0 = \kappa I$ with $\kappa \to \infty$ (infinite variance for non-stationary states)
D) $P_0$ equal to the unconditional state covariance

**Correct Answer:** C

**Feedback:**
- A: This is over-confident and incorrect for uncertain initial conditions.
- B: Unit variance is arbitrary; doesn't reflect genuine uncertainty.
- C: **Correct**. Diffuse initialization is used when initial state is unknown or non-stationary (e.g., trends, random walks). Setting $P_0 = \kappa I$ with $\kappa \to \infty$ (or large, like $10^7$) reflects maximal uncertainty, letting data dominate early estimates.
- D: This is correct for stationary models (solving Lyapunov equation), but "diffuse" specifically refers to non-stationary/uncertain cases.

---

### Question 10 (4 points) - Foundation

Quasi-maximum likelihood (QML) for factor models differs from ML by:

A) Using PCA instead of the Kalman filter
B) Allowing for model misspecification (e.g., non-Gaussian errors) while still using Gaussian likelihood
C) Requiring fewer parameters
D) Maximizing only the E-step

**Correct Answer:** B

**Feedback:**
- A: QML can use the Kalman filter; "quasi" refers to misspecification robustness, not the algorithm.
- B: **Correct**. QML optimizes a Gaussian likelihood even if errors are not Gaussian. Under regularity conditions, the estimator is consistent and asymptotically normal, but standard errors need robust corrections (sandwich estimator).
- C: Number of parameters is the same as ML; QML is about robustness, not parsimony.
- D: QML is not related to EM step decomposition; it's about likelihood specification.

---

## Part B: Mathematical and Computational (30 points)

### Question 11 (6 points) - Core

The log-likelihood formula is:
$$\log L(\theta) = -\frac{TN}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t| + v_t'F_t^{-1}v_t\right]$$

For $T = 100$, $N = 20$, if $\sum_{t=1}^T \log|F_t| = 150$ and $\sum_{t=1}^T v_t'F_t^{-1}v_t = 1800$, what is $\log L$?

A) -20,693
B) -19,718
C) -975
D) -1,837

**Correct Answer:** B

**Feedback:**
Compute each term:
- Constant: $-\frac{100 \times 20}{2}\log(2\pi) = -1000 \log(2\pi) = -1000 \times 1.8379 \approx -1837.9$
- Sum terms: $-\frac{1}{2}(150 + 1800) = -\frac{1}{2}(1950) = -975$
- Total: $-1837.9 - 975 = -2812.9$

Wait, let me recalculate:
- $-\frac{TN}{2}\log(2\pi) = -\frac{2000}{2} \times 1.8379 = -1000 \times 1.8379 = -1837.9$
- $-\frac{1}{2}(150 + 1800) = -975$
- Total: $-1837.9 - 975 = -2812.9$

None of the options match. Let me reconsider. Actually, checking the answers:

Actually the calculation should be:
$\log L = -1000 \log(2\pi) - 975 = -1000(1.8379) - 975 = -1837.9 - 975 = -2812.9$

Let me recalculate with the right constant. $\log(2\pi) = 1.8379$:
Total = $-1000 \times 1.8379 - 975 \approx -2813$

This doesn't match options. Let me reconsider the setup. Actually, perhaps I should use option values. Let's check B: -19,718.

For this to work: $-\frac{TN}{2}  \times 1.8379 - 975$. If total is -19,718:
$-\frac{TN}{2} \times 1.8379 = -19,718 + 975 = -18,743$
$\frac{TN}{2} = 18,743 / 1.8379 \approx 10,200$
$TN = 20,400$

With $T=100, N=20$: $TN = 2000$, not 20,400.

I'll revise this question's numbers to make sense.

**Correct Answer:** B (assuming adjusted numbers)

Actually, I'll provide a simpler calculation:

For $T = 200$, $N = 50$:
- $-\frac{200 \times 50}{2}\log(2\pi) = -5000 \times 1.8379 = -9189.5$
- $-\frac{1}{2}(500 + 20,000) = -10,250$
- Total: $-19,439.5 \approx -19,440$

**Let me correct the question:**

---

### Question 11 (6 points) - Core

The log-likelihood formula is:
$$\log L(\theta) = -\frac{TN}{2}\log(2\pi) - \frac{1}{2}\sum_{t=1}^T \left[\log|F_t| + v_t'F_t^{-1}v_t\right]$$

For $T = 200$, $N = 50$, if $\sum_{t=1}^T \log|F_t| = 500$ and $\sum_{t=1}^T v_t'F_t^{-1}v_t = 20,000$, what is the approximate $\log L$?

A) -9,440
B) -19,440
C) -29,440
D) -10,250

**Correct Answer:** B

**Feedback:**
- Constant term: $-\frac{200 \times 50}{2}\log(2\pi) = -5000 \times 1.8379 \approx -9,190$
- Sum terms: $-\frac{1}{2}(500 + 20,000) = -10,250$
- **Total:** $-9,190 - 10,250 = -19,440$
- Answer: **B**

---

### Question 12 (6 points) - Advanced

For numerical stability, instead of computing $v_t'F_t^{-1}v_t$ directly, we use Cholesky decomposition: $F_t = LL'$. The quadratic form becomes:

A) $(L^{-1}v_t)'(L^{-1}v_t) = w_t'w_t$ where $w_t = L^{-1}v_t$
B) $v_t'L^{-1}v_t$
C) $(L'v_t)'(L'v_t)$
D) $\text{tr}(L v_t v_t' L')$

**Correct Answer:** A

**Feedback:**
- A: **Correct**. $v_t'F_t^{-1}v_t = v_t'(LL')^{-1}v_t = v_t'(L')^{-1}L^{-1}v_t = (L^{-1}v_t)'(L^{-1}v_t) = w_t'w_t$ where $w_t = L^{-1}v_t$ is computed by forward substitution. This avoids explicitly computing $F_t^{-1}$.
- B: This is not a valid quadratic form (missing transpose).
- C: This would be $v_t'LL'v_t = v_t'F_tv_t$, not the inverse.
- D: This is trace notation that doesn't correctly represent the scalar quadratic form.

---

### Question 13 (6 points) - Core

You estimate a DFM via MLE and obtain $\log L = -5,432$. You add one additional factor ($r: 3 \to 4$) and the new log-likelihood is $-5,398$. The number of additional parameters is 15. What is the **likelihood ratio test statistic**?

A) 34
B) 68
C) 15
D) 2.27

**Correct Answer:** B

**Feedback:**
- LR test statistic: $\text{LR} = 2(\log L_{\text{unrestricted}} - \log L_{\text{restricted}}) = 2(-5398 - (-5432)) = 2 \times 34 = 68$
- Under $H_0$ (restricted model is correct), $\text{LR} \sim \chi^2_{15}$ (15 degrees of freedom)
- Critical value at 5%: $\chi^2_{15, 0.05} \approx 25$. Since $68 > 25$, reject null—4th factor is significant.
- Answer: **B**

---

### Question 14 (6 points) - Advanced

In the EM algorithm, the **M-step** update for loadings $\Lambda$ given smoothed factor estimates $\hat{F}_{t|T}$ is:

A) $\Lambda^{(k+1)} = X'\hat{F}_{|T} / T$
B) $\Lambda^{(k+1)} = (X \hat{F}_{|T}')(\hat{F}_{|T}\hat{F}_{|T}')^{-1}$
C) $\Lambda^{(k+1)} = $ PCA of $X$
D) $\Lambda^{(k+1)} = E[\Lambda | X, F]$

**Correct Answer:** B

**Feedback:**
- A: Close, but doesn't account for factor covariance structure properly.
- B: **Correct**. The M-step maximizes expected complete-data log-likelihood. For Gaussian models, this gives:
$$\Lambda^{(k+1)} = \left(\sum_t X_t E[F_t | X_{1:T}]'\right)\left(\sum_t E[F_t F_t' | X_{1:T}]\right)^{-1}$$
  Using smoothed estimates: $\Lambda = (X \hat{F}')(\hat{F}\hat{F}' + \sum_t P_{t|T})^{-1}$ (simplified to $XF'(FF')^{-1}$ if ignoring smoothing covariance).
- C: PCA is not part of the EM M-step for ML estimation.
- D: This is E-step notation, not M-step.

---

### Question 15 (6 points) - Core

You initialize an EM algorithm with PCA estimates. After 50 iterations, the log-likelihood increases from -8,500 to -8,200. After 50 more iterations, it's -8,195. What should you do?

A) Restart with different initial values
B) Continue iterating—convergence requires 200+ iterations
C) Stop—the algorithm has approximately converged (change < 5 in 50 iterations)
D) Switch to a different optimization method

**Correct Answer:** C

**Feedback:**
- A: Likelihood is increasing and nearly flat—no evidence of poor initialization.
- B: Further iterations would yield negligible improvement (< 0.1 per iteration).
- C: **Correct**. Standard convergence criterion: stop when $|\log L^{(k+1)} - \log L^{(k)}| < \epsilon$ (e.g., $\epsilon = 0.1$). Here the change over 50 iterations is 5 (0.1 per iteration on average)—effectively converged.
- D: Switching is unnecessary; EM is working as expected.

---

## Part C: Practical Interpretation and Design (30 points)

### Question 16 (6 points) - Core

You compare PCA and MLE estimates for a factor model with $T = 150$, $N = 30$, $r = 3$. MLE takes 120 seconds; PCA takes 0.8 seconds. The log-likelihoods are: MLE = -4,235, PCA = -4,290. Which method would you choose for this application?

A) Always MLE for better fit
B) Always PCA for speed
C) Depends on the application: if inference is critical, use MLE; if speed matters and fit is adequate, use PCA
D) Neither—use Bayesian methods

**Correct Answer:** C

**Feedback:**
- A: MLE is better (55 log-likelihood points), but 150x slower. Not always worth it.
- B: PCA is much faster, but sacrifices some fit quality.
- C: **Correct**. The tradeoff depends on context:
  - **MLE**: When you need standard errors, hypothesis tests, or maximum precision (e.g., policy applications).
  - **PCA**: When speed matters (e.g., real-time forecasting, large-scale experiments) and the fit is acceptable.

  Here, 55 log-likelihood points over ~4,235 is ~1.3% difference—modest. PCA may suffice unless formal inference is required.
- D: Bayesian methods are even slower (MCMC), not a solution for speed.

---

### Question 17 (6 points) - Advanced

You estimate a DFM via MLE and find that the first two loadings on Factor 1 are: $\lambda_{11} = 0.012$, $\lambda_{21} = 0.009$. These were constrained to be positive for identification. The optimizer reports "boundary solution." What does this indicate?

A) The model is correctly identified
B) Factor 1 may not be well-identified or could be weak/unnecessary
C) Numerical error in the optimizer
D) The constraints are too restrictive

**Correct Answer:** B

**Feedback:**
- A: Boundary solutions suggest identification issues, not successful identification.
- B: **Correct**. Values near zero at a constrained boundary suggest the optimizer "wants" to set them to zero (or negative), indicating Factor 1 may not be supported by the data. This often means either: (1) factor is weak, or (2) wrong normalization/identification scheme.
- C: While possible, more likely an economic signal (weak factor) than numerical error.
- D: Constraints are necessary for identification; relaxing them would cause non-identification, not solve the problem.

---

### Question 18 (6 points) - Core

A colleague proposes the following Bayesian prior for factor loadings: $\lambda_{ij} \sim N(0, 100)$ (very diffuse). What is a potential problem with this prior?

A) It's too informative
B) It doesn't respect identification constraints (e.g., rotation invariance remains)
C) It assumes loadings are negative
D) It's computationally expensive

**Correct Answer:** B

**Feedback:**
- A: $N(0, 100)$ with variance 100 is quite diffuse, not overly informative.
- B: **Correct**. Even with priors, Bayesian factor models face rotation indeterminacy. A diffuse prior on all loadings doesn't identify the factor space. Need restrictions (e.g., lower triangular structure, sign constraints, sparsity priors) to pin down factors.
- C: The prior is symmetric around zero, allowing both positive and negative loadings.
- D: This prior is simple and doesn't significantly increase computational cost.

---

### Question 19 (6 points) - Advanced

You estimate a factor model via MLE on two datasets: (1) $N = 50$, $T = 500$ → converges in 80 iterations; (2) $N = 500$, $T = 50$ → fails to converge after 500 iterations. Why might large $N$, small $T$ be problematic for MLE?

A) Maximum likelihood requires $T > N$
B) With $T < N$, sample covariance is rank-deficient, causing identification issues and numerical instability
C) The Kalman filter only works when $T > N$
D) Factor dynamics cannot be estimated with small $T$

**Correct Answer:** B

**Feedback:**
- A: ML doesn't strictly require $T > N$, but it performs poorly when $T \ll N$.
- B: **Correct**. When $T < N$, the sample covariance $X'X/T$ is rank $T < N$, making it singular. This causes numerical problems in matrix inversions during optimization. Additionally, estimating loadings ($N \times r$ parameters) with few time points is inefficient. PCA asymptotics also require both $N, T \to \infty$.
- C: Kalman filter works regardless of $T$ vs $N$; the issue is statistical, not algorithmic.
- D: Factor dynamics ($r^2 p$ parameters) can be estimated with small $T$, though less precisely. The main issue is estimating loadings and idiosyncratic variances.

---

### Question 20 (6 points) - Core

You compute the **Akaike Information Criterion (AIC)** for models with $r = 2, 3, 4, 5$ factors. Results:

| $r$ | $\log L$ | $k$ (parameters) | AIC |
|-----|----------|------------------|-----|
| 2   | -5,200   | 150              | 10,700 |
| 3   | -5,050   | 180              | 10,460 |
| 4   | -4,980   | 215              | **10,390** |
| 5   | -4,950   | 255              | 10,410 |

Which model does AIC select?

A) $r = 2$ (most parsimonious)
B) $r = 3$ (balances fit and parsimony)
C) $r = 4$ (minimizes AIC)
D) $r = 5$ (best likelihood)

**Correct Answer:** C

**Feedback:**
- A: $r = 2$ has the worst AIC (highest value).
- B: $r = 3$ is better than $r = 2$ but not optimal.
- C: **Correct**. AIC = $-2\log L + 2k$. Lower is better. $r = 4$ has the minimum AIC (10,390), indicating the best tradeoff between fit and complexity.
- D: $r = 5$ has the best likelihood but overfits—AIC penalizes the extra 40 parameters, making it worse than $r = 4$.

---

## Part D: Short Answer (Bonus: 10 points)

### Question 21 (5 points) - Advanced

Explain why the EM algorithm is particularly useful for factor models compared to direct numerical optimization (e.g., BFGS).

**Your Answer:** _(Student provides written explanation)_

**Sample Answer:**

The EM algorithm is advantageous for factor models because:

1. **Closed-form M-step:** Given smoothed factor estimates (E-step), parameter updates (loadings, variances) have closed-form solutions (essentially regression formulas). This avoids numerical gradient computation and line searches.

2. **Monotonic likelihood increase:** EM guarantees $L^{(k+1)} \geq L^{(k)}$, avoiding overshoot issues that numerical optimizers can face with rough likelihood surfaces.

3. **Natural handling of latent variables:** EM explicitly separates the problem into estimating latent factors (E-step via Kalman smoother) and parameters (M-step), which aligns with the model structure.

4. **Numerical stability:** EM avoids computing the full Hessian and is less sensitive to poor initialization compared to Newton-type methods.

**Drawback:** EM can be slower to converge near the optimum compared to quasi-Newton methods.

**Rubric:**
- Mentions closed-form M-step or monotonicity (2 pts)
- Explains latent variable structure advantage (2 pts)
- Mentions a drawback or comparison point (1 pt)

---

### Question 22 (5 points) - Core

A practitioner claims: "Bayesian factor models are always better than MLE because they incorporate prior information." Provide a balanced critique of this claim.

**Your Answer:** _(Student provides written explanation)_

**Sample Answer:**

**Critique:**

**Advantages of Bayesian approach:**
- **Regularization:** Priors provide shrinkage/regularization, especially useful with small samples or weak identification.
- **Uncertainty quantification:** Full posterior distributions for all parameters, not just point estimates.
- **Incorporation of prior knowledge:** Can encode economic theory (e.g., certain loadings expected to be positive).

**Limitations/Trade-offs:**
- **Computational cost:** MCMC is much slower than MLE (minutes to hours vs seconds).
- **Prior sensitivity:** Results can be sensitive to prior specification, especially with weak data. "Uninformative" priors can still affect inference.
- **Subjectivity:** Prior choice involves judgment, which some view as less objective than MLE.
- **Not always "better":** With large samples, MLE and Bayesian estimates converge; the prior matters less. Bayesian machinery adds complexity with limited benefit.

**Conclusion:** Bayesian methods are not universally superior—context matters. Use when prior information is genuinely informative or when regularization is needed; use MLE for speed and simplicity with adequate data.

**Rubric:**
- Identifies advantages of Bayesian (1-2 pts)
- Identifies limitations or contexts where Bayesian is not superior (2 pts)
- Provides balanced conclusion (1 pt)

---

## Answer Key Summary

| Question | Answer | Difficulty | Topic |
|----------|--------|------------|-------|
| 1 | B | Foundation | Kalman likelihood |
| 2 | B | Core | Innovation interpretation |
| 3 | B | Core | Prediction error variance |
| 4 | B | Core | Log-likelihood |
| 5 | A | Advanced | EM algorithm E-step |
| 6 | B | Core | EM convergence |
| 7 | B | Advanced | Bayesian priors |
| 8 | C | Core | Filter initialization |
| 9 | C | Advanced | Diffuse initialization |
| 10 | B | Foundation | Quasi-ML |
| 11 | B | Core | Likelihood calculation |
| 12 | A | Advanced | Cholesky decomposition |
| 13 | B | Core | Likelihood ratio test |
| 14 | B | Advanced | EM M-step |
| 15 | C | Core | Convergence criterion |
| 16 | C | Core | PCA vs MLE tradeoff |
| 17 | B | Advanced | Boundary solutions |
| 18 | B | Core | Bayesian identification |
| 19 | B | Advanced | Small T large N |
| 20 | C | Core | AIC selection |
| 21 | - | Advanced | EM advantages |
| 22 | - | Core | Bayesian vs frequentist |

**Scoring Distribution:**
- Part A (Conceptual): 40 points
- Part B (Mathematical/Computational): 30 points
- Part C (Practical): 30 points
- Part D (Bonus Short Answer): 10 points

**Difficulty Distribution:**
- Foundation: 8 points (2 questions)
- Core: 58 points (12 questions)
- Advanced: 44 points (8 questions)

---

## Study Tips

1. **Kalman Filter Intuition:** Understand the filter as a recursive prediction-correction algorithm. Draw the flow diagram.

2. **EM Algorithm:** Memorize the two-step structure: E-step = smooth factors, M-step = update parameters given smoothed factors.

3. **Likelihood Decomposition:** Practice writing out $p(X_1, ..., X_T) = \prod_t p(X_t | X_{1:t-1})$ and connecting it to Kalman innovations.

4. **Numerical Stability:** Know why Cholesky decomposition and log-likelihood are used (underflow/overflow prevention).

5. **Identification:** Review what constraints are needed for ML estimation (loading restrictions, factor variance normalization).

6. **Bayesian Basics:** Understand what priors do (regularization, identification) and the computational cost tradeoff.

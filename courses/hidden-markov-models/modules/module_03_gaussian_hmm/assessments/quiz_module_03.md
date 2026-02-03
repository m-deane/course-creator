# Quiz: Module 3 - Gaussian HMM

**Course:** Hidden Markov Models
**Module:** 3 - Gaussian Emissions and Continuous Observations
**Total Points:** 100
**Estimated Time:** 30 minutes
**Attempts Allowed:** 2

---

## Instructions

This quiz assesses your understanding of Gaussian HMMs with continuous observations, EM algorithm for parameter estimation, and multivariate Gaussian emissions. Show all work for mathematical problems.

---

## Section 1: Gaussian Emissions (25 points)

### Question 1 (8 points)

In a Gaussian HMM, the emission probability is a continuous density function. For a univariate Gaussian emission from state j, what is the correct formula?

A) b_j(x) = exp(-(x - μ_j)² / 2σ_j²)

B) b_j(x) = (1/√(2πσ_j²)) exp(-(x - μ_j)² / 2σ_j²)

C) b_j(x) = (1/σ_j) exp(-|x - μ_j| / σ_j)

D) b_j(x) = P(X = x | S = j)

**Answer: B**

**Explanation:**
The Gaussian (Normal) probability density function is:

b_j(x) = (1/√(2πσ_j²)) exp(-(x - μ_j)² / (2σ_j²))

Where:
- μ_j is the mean of state j
- σ_j² is the variance of state j
- The term 1/√(2πσ_j²) is the normalization constant

**Key points:**

A is incorrect: Missing the normalization constant (doesn't integrate to 1)

C is incorrect: This is a Laplace distribution, not Gaussian

D is incorrect: For continuous distributions, P(X = x) = 0; we use density functions b_j(x) where:
- P(x ≤ X ≤ x + dx) ≈ b_j(x) dx
- ∫_{-∞}^{∞} b_j(x) dx = 1

**In HMM algorithms:**
- Use b_j(x) directly in forward/backward/Viterbi
- No change to algorithm structure, just continuous emissions instead of discrete

---

### Question 2 (10 points)

A 2-state Gaussian HMM has emission parameters:

```
State 1 (Bull): μ_1 = 0.08% (daily return), σ_1 = 1.2%
State 2 (Bear): μ_2 = -0.05% (daily return), σ_2 = 2.5%
```

On a given day, the observed return is x = -3.0%.

Compute the emission probability density b_1(x) and b_2(x). Which state is more likely to have generated this observation (based on emission probability alone)?

**Solution:**

**For State 1 (Bull):**

z_1 = (x - μ_1) / σ_1 = (-3.0 - 0.08) / 1.2 = -3.08 / 1.2 = -2.567

b_1(x) = (1/√(2π(1.2)²)) exp(-(-2.567)² / 2)
       = (1/(1.2√(2π))) exp(-6.589/2)
       = (1/3.002) exp(-3.295)
       = 0.333 × 0.0370
       = 0.0123

**For State 2 (Bear):**

z_2 = (x - μ_2) / σ_2 = (-3.0 - (-0.05)) / 2.5 = -2.95 / 2.5 = -1.18

b_2(x) = (1/√(2π(2.5)²)) exp(-(-1.18)² / 2)
       = (1/(2.5√(2π))) exp(-1.392/2)
       = (1/6.254) exp(-0.696)
       = 0.160 × 0.498
       = 0.0797

**Answers:**
- b_1(-3.0%) = 0.0123
- b_2(-3.0%) = 0.0797

**State 2 (Bear) is approximately 6.5 times more likely to have generated this observation.**

**Explanation:**
A -3% return is:
- 2.57 standard deviations below the Bull mean (rare, ~0.5% probability)
- 1.18 standard deviations below the Bear mean (plausible, ~12% probability)

The Bear state is more consistent with this negative return despite:
- Bull having lower variance (more peaked distribution)
- Bear having negative mean and higher variance

The likelihood ratio b_2(x)/b_1(x) ≈ 6.5 favors Bear state.

---

### Question 3 (7 points)

Why might using Gaussian emissions be advantageous over discrete observations for modeling financial returns?

A) Gaussian emissions are always more accurate
B) Returns are naturally continuous; discretization loses information
C) Gaussian HMMs require fewer parameters
D) The forward algorithm is faster with Gaussian emissions

**Answer: B**

**Explanation:**

**Continuous nature of returns:**
- Stock returns, volatility, prices are continuous random variables
- Discretizing into bins (e.g., {low, medium, high}) loses information
- Fine discretization requires many bins, increasing parameters

**Information preservation:**
- Continuous: -2.45% and -2.50% are slightly different
- Discrete: Both might map to "large negative" bin (identical treatment)

**Gaussian assumptions:**
- Returns often approximately normal (especially at short timescales)
- Central Limit Theorem supports Gaussian approximation
- Log-returns more Gaussian than raw returns

**Disadvantages of Gaussian:**
- Fat tails in real returns (kurtosis > 3)
- Asymmetry (skewness ≠ 0)
- May underestimate extreme event probabilities

**Extensions:**
- Student's t-distribution (heavier tails)
- Mixture of Gaussians (more flexibility)
- Generalized Autoregressive Conditional Heteroskedasticity (GARCH) emissions

C is false: Gaussian HMMs have comparable parameters (mean and variance per state vs. emission probabilities per observation value).

D is false: Computational complexity is the same O(N²T).

---

## Section 2: EM for Gaussian HMM (30 points)

### Question 4 (12 points)

In the M-step of the Baum-Welch algorithm for a Gaussian HMM, the mean μ_j of state j is updated as:

μ_j^{new} = Σ_t γ_t(j) × x_t / Σ_t γ_t(j)

**Part A (6 points):** Explain what this formula computes in plain English.

**Answer:**

This computes the **weighted average** of the observed values x_t, where the weights are the posterior probabilities γ_t(j) = P(S_t = j | O_{1:T}, λ).

**Numerator:** Σ_t γ_t(j) × x_t
- Sum of observations, weighted by how likely state j was at each time
- Observations when state j was likely get higher weight

**Denominator:** Σ_t γ_t(j)
- Total "expected occupancy" of state j
- Sum of probabilities of being in state j across all time steps

**Ratio:**
- Expected average of observations generated by state j
- Maximum likelihood estimate of mean given soft state assignments

**Intuition:**
If γ_5(j) = 0.9 and x_5 = 2.5, then observation 2.5 contributes 0.9 × 2.5 = 2.25 to the numerator. Observations that almost certainly came from state j (γ_t(j) ≈ 1) contribute their full value; uncertain observations contribute fractionally.

**Part B (6 points):** Write the update formula for the variance σ_j² of state j.

**Answer:**

σ_j²^{new} = Σ_t γ_t(j) × (x_t - μ_j^{new})² / Σ_t γ_t(j)

**Explanation:**
This is the weighted sample variance:
- Numerator: Sum of squared deviations from the new mean, weighted by γ_t(j)
- Denominator: Same normalization (expected occupancy)

**Procedure:**
1. Update μ_j^{new} first (using mean formula)
2. Then update σ_j²^{new} using the new mean
3. Ensures variance is computed around the correct center

**Properties:**
- Observations when state j was likely (high γ_t(j)) contribute more to variance estimate
- Captures spread of observations attributed to state j
- Guaranteed to be non-negative (sum of squared terms)

---

### Question 5 (10 points)

A researcher fits a 2-state Gaussian HMM to daily S&P 500 log-returns. After convergence, the estimated parameters are:

```
State 1: μ_1 = 0.06%, σ_1 = 0.8%
State 2: μ_2 = -0.10%, σ_2 = 2.1%

A = [0.97  0.03]
    [0.08  0.92]
```

**Part A (5 points):** How would you interpret these two states in financial terms?

**Answer:**

**State 1 - Low Volatility / Normal Market:**
- Positive mean return (0.06% daily ≈ 15% annualized)
- Low volatility (0.8% daily ≈ 12.7% annualized)
- High persistence (97% stay probability)
- Represents normal bull/expansion market conditions

**State 2 - High Volatility / Crisis:**
- Negative mean return (-0.10% daily ≈ -25% annualized)
- High volatility (2.1% daily ≈ 33.4% annualized)
- Moderate persistence (92% stay probability)
- Represents crisis/bear market conditions

**State dynamics:**
- State 1 is more persistent (97% vs 92%) - normal markets last longer
- Transitions are rare but possible
- 3% chance of exiting normal → crisis per day
- 8% chance of exiting crisis → normal per day
- Crisis tends to resolve faster than normal markets destabilize

**Practical implications:**
- Risk management: Adjust portfolio based on inferred state
- Position sizing: Reduce exposure when P(State 2 | returns) is high
- Stop-loss: Tighten stops in high volatility state

**Part B (5 points):** Given the stationary distribution π^∞ = [0.73, 0.27], what does this tell you about long-run market behavior?

**Answer:**

The market spends approximately:
- **73% of time in State 1** (low vol, positive returns)
- **27% of time in State 2** (high vol, negative returns)

**Implications:**

**Long-run average return:**
E[r] = π_1 × μ_1 + π_2 × μ_2
     = 0.73 × 0.06% + 0.27 × (-0.10%)
     = 0.044% - 0.027%
     = 0.017% daily ≈ 4.3% annualized

Despite positive returns in State 1, the overall long-run return is modest due to the drag from State 2.

**Risk assessment:**
- About 1 in 4 days are in crisis regime
- Even though crisis is less frequent, it can significantly impact portfolio
- Need to manage tail risk in State 2

**Strategy design:**
- Baseline allocation for State 1 (most common)
- Defensive posture for State 2 (significant portion of time)
- Early detection of regime switches is valuable

---

### Question 6 (8 points)

Why might the EM algorithm for Gaussian HMMs get stuck in poor local optima?

Select all that apply:

A) Starting values place all states in similar regions of observation space
B) The log-likelihood function is non-convex
C) Observations come from overlapping Gaussian distributions
D) The forward algorithm has numerical errors

**Answers: A, B, C**

**Explanation:**

**A - TRUE: Poor initialization**
- If all states start with similar μ_j, they don't specialize
- EM might converge before states differentiate
- Solution: Initialize with k-means clustering or domain knowledge

**B - TRUE: Non-convexity**
- HMM likelihood surface has multiple local maxima
- EM guarantees improvement, not global optimum
- Different initializations can find different solutions

**C - TRUE: Overlapping distributions**
- If state distributions overlap heavily, assignment is ambiguous
- Hard to distinguish states based on observations alone
- Model may be over-parameterized (too many states for data)

**D - FALSE: Numerical errors**
- While numerical issues (underflow) can occur, they don't cause poor local optima
- Use log-space computations to avoid underflow
- Numerical errors affect stability, not solution quality

**Best practices to avoid poor optima:**

1. **Multiple random restarts:** Run EM 10-20 times with different initializations
2. **Informative initialization:** Use k-means on observations to initialize μ_j, σ_j²
3. **Regularization:** Add small penalty for similar state parameters
4. **Validation:** Check likelihood on held-out data
5. **Interpretability:** Verify states make financial sense

---

## Section 3: Multivariate Gaussian HMM (25 points)

### Question 7 (10 points)

For a multivariate Gaussian HMM with d-dimensional observations, the emission density is:

b_j(x) = (1/√((2π)^d |Σ_j|)) exp(-1/2 (x - μ_j)^T Σ_j^{-1} (x - μ_j))

**Part A (5 points):** What are the parameters of state j in a multivariate Gaussian HMM?

**Answer:**

**μ_j:** Mean vector (d × 1)
- μ_j = [μ_j^{(1)}, μ_j^{(2)}, ..., μ_j^{(d)}]^T
- Expected value of each dimension in state j

**Σ_j:** Covariance matrix (d × d)
- Σ_j is positive semi-definite, symmetric
- Diagonal elements: variances of each dimension
- Off-diagonal elements: covariances between dimensions

**Example for d=2 (return and volatility):**
```
μ_j = [μ_return, μ_vol]^T

Σ_j = [σ_return²        ρ×σ_return×σ_vol]
      [ρ×σ_return×σ_vol  σ_vol²          ]
```

Where ρ is the correlation between return and volatility in state j.

**Part B (5 points):** How many parameters does a single state have in a multivariate Gaussian HMM with d dimensions?

**Answer:**

**Mean vector μ_j:** d parameters

**Covariance matrix Σ_j:** d(d+1)/2 parameters
- d variance terms (diagonal)
- d(d-1)/2 covariance terms (upper/lower triangle, symmetric)

**Total per state:** d + d(d+1)/2 = d(d+3)/2

**Examples:**
- d=1 (univariate): 1 + 1 = 2 (mean, variance)
- d=2 (bivariate): 2 + 3 = 5 (2 means, 2 variances, 1 covariance)
- d=3 (trivariate): 3 + 6 = 9 (3 means, 3 variances, 3 covariances)
- d=10: 10 + 55 = 65 parameters per state

**For N-state HMM:**
- Emission parameters: N × d(d+3)/2
- Transition parameters: N(N-1) (N² with constraints)
- Initial distribution: N-1

Total grows quadratically in d - can become large!

---

### Question 8 (8 points)

A portfolio manager models joint dynamics of [stock return, implied volatility] using a 2-state bivariate Gaussian HMM:

```
State 1 (Normal):
μ_1 = [0.05%, 15%]^T
Σ_1 = [1.0   -0.5]  (correlation ρ_1 = -0.5)
      [-0.5   4.0]

State 2 (Crisis):
μ_2 = [-0.15%, 30%]^T
Σ_2 = [4.0   0.8]   (correlation ρ_2 = 0.2)
      [0.8   9.0]
```

What does the negative correlation in State 1 vs. positive correlation in State 2 tell you?

**Answer:**

**State 1 (Normal Market): ρ = -0.5 (negative correlation)**

In normal markets:
- When returns are high → implied volatility tends to be low
- When returns are low → implied volatility tends to be high
- This is the **leverage effect** or **volatility feedback**
- Reflects investor complacency in rising markets, fear in declining markets

Example:
- Good news: stock up 1%, VIX drops from 15% → 13%
- Bad news: stock down 0.5%, VIX rises from 15% → 17%

**State 2 (Crisis): ρ = +0.2 (positive correlation)**

In crisis markets:
- When returns are very negative → volatility spikes even more
- Panic selling increases both return magnitude and volatility
- Positive correlation indicates "tail risk" regime
- Both dimensions move together in extreme events

Example:
- Market crash day: stock down 5%, VIX jumps from 30% → 45%
- Slight recovery: stock up 2%, VIX still elevated at 35%

**Practical implications:**

1. **Hedging:** Volatility hedges (VIX calls) more effective in State 1 (negative correlation means protection when needed)

2. **Risk management:** State 2 is more dangerous - both return and volatility move adversely

3. **Model realism:** Capturing this correlation regime shift is crucial for:
   - Option pricing
   - Portfolio optimization
   - Stress testing

This demonstrates why multivariate HMMs are superior to independent univariate models - they capture changing correlation structure across regimes.

---

### Question 9 (7 points)

For the M-step update of the covariance matrix in a multivariate Gaussian HMM, the formula is:

Σ_j^{new} = Σ_t γ_t(j) × (x_t - μ_j^{new})(x_t - μ_j^{new})^T / Σ_t γ_t(j)

Explain why (x_t - μ_j^{new})(x_t - μ_j^{new})^T produces a d × d matrix.

**Answer:**

**(x_t - μ_j^{new})** is a column vector of dimension d × 1:
```
(x_t - μ_j^{new}) = [x_t^{(1)} - μ_j^{(1)}]
                    [x_t^{(2)} - μ_j^{(2)}]
                    [        ...          ]
                    [x_t^{(d)} - μ_j^{(d)}]
```

**(x_t - μ_j^{new})^T** is a row vector of dimension 1 × d:
```
(x_t - μ_j^{new})^T = [x_t^{(1)} - μ_j^{(1)}, x_t^{(2)} - μ_j^{(2)}, ..., x_t^{(d)} - μ_j^{(d)}]
```

**Outer product:** (d × 1) × (1 × d) = (d × d) matrix:
```
(x - μ)(x - μ)^T = [Δ_1]   [Δ_1, Δ_2, ..., Δ_d]
                   [Δ_2] ×
                   [...]
                   [Δ_d]

                 = [Δ_1²      Δ_1Δ_2  ...  Δ_1Δ_d]
                   [Δ_2Δ_1    Δ_2²    ...  Δ_2Δ_d]
                   [...       ...     ...  ...   ]
                   [Δ_dΔ_1    Δ_dΔ_2  ...  Δ_d²  ]
```

**Elements:**
- Diagonal [i,i]: (x_t^{(i)} - μ_j^{(i)})² (variance of dimension i)
- Off-diagonal [i,k]: (x_t^{(i)} - μ_j^{(i)})(x_t^{(k)} - μ_j^{(k)}) (covariance between dimensions i and k)

**Weighted average:**
Summing these outer products weighted by γ_t(j) gives the covariance matrix estimate.

This is the multivariate generalization of the variance formula:
- 1D: Σ_t γ_t(j) × (x_t - μ_j)²
- dD: Σ_t γ_t(j) × (x_t - μ_j)(x_t - μ_j)^T

---

## Section 4: Practical Considerations (20 points)

### Question 10 (8 points)

What numerical issue arises when computing Gaussian densities for observations far from the mean, and how is it addressed?

**Answer:**

**Problem: Numerical underflow**

For large |x - μ_j|/σ_j (far from mean):

b_j(x) = (1/√(2πσ_j²)) exp(-(x - μ_j)² / (2σ_j²))

The exponential term becomes extremely small:
- If z = (x - μ_j)/σ_j = 10: exp(-50) ≈ 10^{-22}
- If z = 20: exp(-200) ≈ 10^{-87}

These values:
- Underflow to 0 in floating point arithmetic
- Cause numerical instability in forward/backward algorithms
- Lead to division by zero errors

**Solution: Log-space computation**

Instead of computing probabilities, compute log-probabilities:

log b_j(x) = -1/2 log(2πσ_j²) - (x - μ_j)² / (2σ_j²)

**Modified algorithms:**

**Log-Forward:**
- Compute log α_t(i) instead of α_t(i)
- Use log-sum-exp trick for stable summation:
  log(Σ exp(a_i)) = max(a_i) + log(Σ exp(a_i - max(a_i)))

**Log-Viterbi:**
- Natural fit: already uses max instead of sum
- Compute log probabilities throughout

**Benefits:**
- No underflow (log of small numbers is negative, not zero)
- Improved numerical stability
- Only convert back to probability space when needed (e.g., final γ_t(i))

**Implementation:**
```python
import numpy as np
from scipy.special import logsumexp

# Safe computation
log_alpha = ...
log_prob = logsumexp(log_alpha)  # Safe sum over states
```

---

### Question 11 (7 points)

A trader fits a 3-state Gaussian HMM to EUR/USD daily returns. Which validation approaches should they use to ensure the model is useful?

Select all that apply:

A) Check if learned states have distinct means and variances
B) Verify log-likelihood increases on training data
C) Test regime detection on out-of-sample data
D) Compare to simpler 2-state model using BIC/AIC
E) Ensure states are interpretable (e.g., trending vs. ranging)

**Answers: A, C, D, E**

**Explanation:**

**A - TRUE: State distinctiveness**
- If μ_1 ≈ μ_2 ≈ μ_3, states haven't specialized
- Check if distributions overlap significantly
- Well-separated states → more reliable regime detection

**B - FALSE: Training likelihood alone insufficient**
- More states always increase training likelihood (overfitting)
- Need out-of-sample validation
- BIC/AIC penalize complexity

**C - TRUE: Out-of-sample testing**
- Evaluate regime detection on unseen data
- Do regimes persist and make sense?
- Can decoded states predict future volatility/returns?

**D - TRUE: Model comparison**
- Use information criteria to balance fit and complexity
- BIC = -2 log L + k log(n) (stronger penalty for complexity)
- AIC = -2 log L + 2k
- Compare 2-state, 3-state, 4-state models

**E - TRUE: Interpretability**
- States should correspond to trading regimes:
  - Trending up, trending down, ranging
  - Low vol, medium vol, high vol
  - Bull, neutral, bear
- If states aren't interpretable, model may not be useful
- Economic story validates statistical model

**Additional validation:**
- **Backtesting:** Test trading strategies based on regime signals
- **Robustness:** Refit on different time periods, check consistency
- **Residuals:** Check if residuals within each state are i.i.d. Gaussian

---

### Question 12 (5 points)

When modeling asset returns with a Gaussian HMM, why might you use log-returns instead of simple returns?

A) Log-returns are bounded between -1 and 1
B) Log-returns are more normally distributed and time-additive
C) Simple returns cannot be negative
D) Log-returns are easier to compute

**Answer: B**

**Explanation:**

**Log-returns: r_t = log(P_t / P_{t-1})**

Advantages:

1. **Time-additivity:**
   - Multi-period log-return = sum of single-period log-returns
   - log(P_T/P_0) = Σ log(P_t/P_{t-1})
   - Simple returns require products: (P_T/P_0) - 1 ≠ Σ [(P_t/P_{t-1}) - 1]

2. **More Gaussian:**
   - Central Limit Theorem: sum of i.i.d. → Gaussian
   - Log-returns (sums) more normal than simple returns (products)
   - Better fit for Gaussian HMM assumptions

3. **Symmetry:**
   - +10% then -10% simple return ≠ break even (ends at 99%)
   - +10% then -10% log-return ≈ break even
   - Log-returns treat gains and losses more symmetrically

4. **No upper bound:**
   - Simple returns bounded below at -100% (total loss)
   - Unbounded above (no theoretical limit)
   - Log-returns unbounded in both directions (more symmetric)

**Disadvantages of log-returns:**
- Less intuitive than simple returns
- Small difference from simple returns for small moves
- For options/derivatives, simple returns may be more natural

**A is false:** Log-returns are unbounded.
**C is false:** Simple returns can be negative (down to -100%).
**D is false:** Both are equally easy to compute.

---

## Bonus Section (5 points)

### Question 13 (5 points)

Derive the M-step update for the mean μ_j by taking the derivative of the expected complete log-likelihood with respect to μ_j and setting it to zero.

**Solution:**

**Expected complete log-likelihood (Q-function):**

Q(λ, λ^{old}) = Σ_t Σ_j γ_t(j) log[N(x_t | μ_j, σ_j²)]

For Gaussian density:
log[N(x_t | μ_j, σ_j²)] = -1/2 log(2πσ_j²) - (x_t - μ_j)² / (2σ_j²)

The terms involving μ_j:

Q_μ = Σ_t γ_t(j) × [-(x_t - μ_j)² / (2σ_j²)]
    = -(1/(2σ_j²)) Σ_t γ_t(j) × (x_t - μ_j)²

**Take derivative with respect to μ_j:**

dQ_μ/dμ_j = -(1/(2σ_j²)) Σ_t γ_t(j) × 2(x_t - μ_j) × (-1)
          = (1/σ_j²) Σ_t γ_t(j) × (x_t - μ_j)

**Set to zero:**

(1/σ_j²) Σ_t γ_t(j) × (x_t - μ_j) = 0

Σ_t γ_t(j) × (x_t - μ_j) = 0

Σ_t γ_t(j) × x_t - μ_j Σ_t γ_t(j) = 0

**Solve for μ_j:**

μ_j = Σ_t γ_t(j) × x_t / Σ_t γ_t(j)

This is the weighted sample mean, which is the maximum likelihood estimate of the mean given the soft state assignments γ_t(j).

---

## Answer Key Summary

1. B (8 pts)
2. b_1(x)=0.0123, b_2(x)=0.0797, Bear more likely (10 pts)
3. B (7 pts)
4. Part A: Weighted average explanation (6 pts), Part B: Variance formula (6 pts)
5. Part A: State interpretation (5 pts), Part B: Long-run behavior (5 pts)
6. A, B, C (8 pts)
7. Part A: μ_j and Σ_j (5 pts), Part B: d(d+3)/2 parameters (5 pts)
8. Correlation interpretation (8 pts)
9. Outer product explanation (7 pts)
10. Log-space computation (8 pts)
11. A, C, D, E (7 pts)
12. B (5 pts)
13. Derivation of mean update (5 pts - Bonus)

**Total: 100 points (105 with bonus)**

---

## Grading Rubric

- **90-100 points:** Excellent - Strong grasp of Gaussian HMMs and EM
- **80-89 points:** Good - Solid understanding with minor gaps
- **70-79 points:** Satisfactory - Adequate knowledge, review multivariate case
- **60-69 points:** Needs Improvement - Review emission densities and parameter updates
- **Below 60:** Incomplete Understanding - Revisit all module materials

---

## Learning Objectives Assessed

- [ ] Understand Gaussian emission densities
- [ ] Compute emission probabilities for continuous observations
- [ ] Derive and apply EM updates for μ_j and σ_j²
- [ ] Extend to multivariate Gaussian emissions
- [ ] Interpret covariance matrices in financial contexts
- [ ] Address numerical stability issues
- [ ] Validate and compare Gaussian HMM models
- [ ] Apply appropriate data transformations (log-returns)

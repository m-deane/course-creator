# Factor Model Identification: Complete Theory

## TL;DR

Factor models face fundamental identification challenges: multiple parameter sets can generate identical likelihood. This guide covers all identification issues, normalization strategies, and testable restrictions for static and dynamic factor models.

---

## 1. The Fundamental Identification Problem

### 1.1 Rotation Invariance

**Core issue:** Factors and loadings are not separately identified.

**Static factor model:**
```
y_t = Λ f_t + ε_t
```

**Observational equivalence:** For any invertible r×r matrix R:
```
y_t = Λ f_t = (Λ R) (R⁻¹ f_t) = Λ* f_t*
```

Where:
- Λ* = Λ R (rotated loadings)
- f_t* = R⁻¹ f_t (rotated factors)

**Implication:** Infinite parameter sets generate the same data!

### 1.2 Why This Matters

Without identification:
1. **No unique MLE** - many equivalent maxima
2. **Uninterpretable factors** - any rotation is valid
3. **Unstable estimation** - different algorithms give different results
4. **No inference** - can't test hypotheses on unidentified parameters

**Solution:** Impose **identifying restrictions** that pin down R.

---

## 2. Identification Strategies

### 2.1 Loading Normalization

**Strategy 1: Fix r² elements of Λ**

Most common: Set first r×r block to identity
```
Λ = [I_r]
    [Λ₂]

where Λ₂ is (N-r) × r, free to estimate
```

**Number of restrictions:** r²

**Advantage:** Simple, easy to implement

**Disadvantage:** Assumes first r series have "clean" factor structure

**Example (r=2, N=5):**
```
Λ = [1    0  ]   ← Fixed
    [0    1  ]   ← Fixed
    [λ₃₁ λ₃₂]   ← Free
    [λ₄₁ λ₄₂]   ← Free
    [λ₅₁ λ₅₂]   ← Free
```

This identifies factors uniquely (up to sign).

### 2.2 Variance Normalization

**Strategy 2: Diagonal factor covariance**
```
Var[f_t] = I_r  (factors uncorrelated with unit variance)
```

**Combined with:** Triangular Λ
```
Λ'Λ = D  (diagonal)
```

**Number of restrictions:** r(r+1)/2

**Advantage:** Factors are orthonormal (like PCA)

**Disadvantage:** Factors ordered arbitrarily

### 2.3 Bai-Ng (2002) Normalization

**For asymptotic theory (N,T → ∞):**

**Restrictions:**
```
1. Λ'Λ/N = I_r         (loadings normalized)
2. F'F/T is diagonal   (factors orthogonal)
3. Factors ordered by variance
```

**Why this works:**
- As N → ∞, normalization becomes innocuous
- Identification is **approximate** (not exact)
- Consistent estimation without exact restrictions

**Used in:** Stock-Watson PCA estimation

### 2.4 Sign Normalization

Even after rotation is fixed, **sign is unidentified**:
```
(Λ, f_t) ≡ (-Λ, -f_t)
```

**Convention:** First loading of each factor is positive
```
Λ[1,k] > 0  for k = 1, ..., r
```

---

## 3. Identification in Dynamic Factor Models

### 3.1 Additional Complexity

DFM has factor dynamics:
```
f_t = T f_{t-1} + η_t,  Var[η_t] = Q
```

**New parameters:** T (r×r), Q (r×r symmetric) = r² + r(r+1)/2 new parameters

**New identification issues:**
1. Rotation of (Λ, f_t) as before
2. Ordering of factor lags
3. Covariance structure of innovations

### 3.2 State-Space Identification

**Full DFM system:**
```
f_t = T f_{t-1} + η_t     (state equation)
y_t = Λ f_t + ε_t         (observation equation)
```

**Parameters:**
- Λ: N × r (Nr parameters)
- T: r × r (r² parameters)
- Q: r × r symmetric (r(r+1)/2 parameters)
- H: N × N (typically diagonal, N parameters)

**Total:** Nr + r² + r(r+1)/2 + N

**Required restrictions:** At least r² (from rotation indeterminacy)

### 3.3 Common DFM Normalizations

**Option 1: Fix Λ block (most common)**
```
Λ = [I_r],  H diagonal,  T and Q unrestricted
    [Λ₂]
```
Identifies factors and dynamics uniquely.

**Option 2: VAR normalization**
```
Λ'Λ/N = I,  T lower triangular,  Q diagonal
```
Recursive VAR structure for factors.

**Option 3: Structural DFM**
```
T = diagonal,  Q = I_r,  Λ restricted to economic interpretation
```
Each factor is AR(1), loadings have structural meaning.

---

## 4. Observability and Controllability

### 4.1 Observability Condition

**Question:** Can we recover factors from observations?

**Definition:** System is **observable** if rank condition holds:
```
rank[Λ'  (ΛT)'  (ΛT²)'  ...  (ΛT^(r-1))'] = r
```

**Intuition:** Need Λ to have full column rank + dynamics to be non-degenerate

**Failure example:**
```
Λ = [1  0]    T = [0  0]
    [1  0]        [0  1]
    [0  1]
```
Second factor never affects y₁ or y₂, only y₃. If y₃ has high measurement error, second factor is unidentifiable.

### 4.2 Controllability Condition

**Question:** Can innovations η_t affect all factors?

**Definition:** System is **controllable** if:
```
rank[R  TR  T²R  ...  T^(r-1)R] = r
```

Where f_t = T f_{t-1} + R η_t.

**Failure example:**
```
T = [0.5  0  ]    R = [1]
    [0    0.8]        [0]
```
Second factor is deterministic (no innovation), so unidentifiable.

**Kalman's theorem:** Need **both** observability and controllability for identification.

---

## 5. Testable Restrictions

### 5.1 Exact Factor Model

**Assumption:** Idiosyncratic errors are uncorrelated
```
H = diagonal
```

**Testable implication:** Conditional on factors, y_t elements are independent
```
Cov[ε_{it}, ε_{jt}] = 0  for i ≠ j
```

**Test:** After factor extraction, check residual correlations
```python
residuals = y - Λ @ f_hat
corr_matrix = np.corrcoef(residuals.T)
# Off-diagonals should be near zero
```

**Common finding:** Exact factor structure often **rejected** in macro data
- Some residual correlation remains
- Solution: Allow for **approximate** factor structure

### 5.2 Approximate Factor Model

**Relaxation:** Allow weak residual correlation
```
H = Σ_ε  (not necessarily diagonal)
```

**But require:** Cross-sectional correlation of ε dies out as N → ∞
```
λ_max(Σ_ε) = O(1)  (bounded eigenvalue)
```

**Interpretation:** Factors capture "pervasive" variation, residuals have only "local" correlation

**Advantage:** More realistic, allows for sector effects, regional correlation, etc.

### 5.3 Number of Factors

**Question:** How many factors r?

**Not identified** in strict sense (different r are different models)

**But can test:** H₀: r = r₀ vs. H₁: r = r₀ + 1

**Bai-Ng information criteria:**
```
IC(r) = log V(r) + r · penalty(N,T)

where V(r) = (1/NT) ∑∑ ε̂²ᵢₜ
```

**Penalties:**
- IC₁: penalty = (N+T)/(NT) · log(NT/(N+T))
- IC₂: penalty = (N+T)/(NT) · log(C_NT), C_NT = min(N,T)
- IC₃: penalty = log(C_NT)/C_NT

**Choose r that minimizes IC(r).**

**Consistency:** As N,T → ∞, chooses true r with probability → 1

---

## 6. Identification in Practice

### 6.1 PCA-Based Estimation (Stock-Watson)

**Advantage:** No explicit identification needed!

PCA **automatically** normalizes:
```
1. Λ̂'Λ̂/N = I_r           (orthonormal loadings)
2. F̂'F̂/T = Δ diagonal    (uncorrelated factors)
3. Factors ordered by variance
```

**Sign convention:** First loading positive

**Rotation invariance:** Still present, but PCA picks the "principal" rotation (maximum variance)

**Drawback:** Factors may not have economic interpretation

### 6.2 Maximum Likelihood

**Requires:** Explicit identification restrictions

**Common choice:**
```
Λ = [I_r],  H diagonal
    [Λ₂]
```

**Estimation:**
1. Initialize with PCA
2. Rotate to satisfy restrictions
3. Run EM algorithm with restrictions imposed

**Advantage:** Efficient, proper standard errors

**Disadvantage:** Sensitive to normalization choice

### 6.3 Bayesian Approach

**Idea:** Use priors to achieve identification

**Common priors:**
```
Λ ~ N(0, σ²_λ I)      (shrinkage)
f_t ~ N(0, I_r)       (normalized factors)
T ~ Normal            (factor dynamics)
```

**Advantage:**
- Automatic regularization
- Full posterior distribution
- No hard restrictions needed

**Disadvantage:**
- Computationally expensive (MCMC)
- Prior sensitivity

---

## 7. Rotation Methods for Interpretation

### 7.1 The Interpretation Problem

Even after identification, PCA factors may be **uninterpretable**:
```
Factor 1 = 0.5*real_activity + 0.3*inflation + 0.2*financial
```

**Goal:** Rotate to factors with **economic meaning**

### 7.2 Varimax Rotation

**Objective:** Maximize variance of squared loadings
```
max ∑_k Var(Λ²_{·k})
```

**Effect:** Pushes loadings toward 0 or 1 (sparse loadings)

**Result:** Each factor loads strongly on few variables
```
Factor 1: employment, production, sales (REAL ACTIVITY)
Factor 2: CPI, PPI, wages (INFLATION)
Factor 3: rates, spreads (FINANCIAL)
```

**Implementation:**
```python
from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=r, rotation='varimax')
fa.fit(data)
factors_rotated = fa.transform(data)
loadings_rotated = fa.components_.T
```

### 7.3 Targeted Factor Extraction

**Idea:** Extract factors to maximize correlation with target variable

**Method (Bai & Ng 2008):**
1. Run regression: target = β'X + error
2. Identify most important predictors (large β)
3. Extract factors using only those predictors
4. Use targeted factors in final model

**Advantage:** Factors relevant for specific prediction task

**Example:** GDP nowcasting
- Identify predictors most correlated with GDP
- Extract factors from those series
- Factors are GDP-relevant by construction

---

## 8. Global vs. Local Identification

### 8.1 Definitions

**Local identification:**
- Parameters uniquely determined in neighborhood of true value
- Necessary for standard asymptotics
- Can test with rank condition on Jacobian

**Global identification:**
- Parameters uniquely determined everywhere
- Stronger requirement
- Rules out multiple modes in likelihood

### 8.2 Rank Condition

**Locally identified if:**
```
rank[∂ℓ(θ)/∂θ] = dim(θ)  at true θ₀
```

Where ℓ(θ) is log-likelihood.

**For factor models:** Need observability + controllability + normalization

### 8.3 Practical Implications

**Local identification sufficient for:**
- Consistency of MLE
- Asymptotic normality
- Hypothesis testing

**Global identification needed for:**
- Unique solution in optimization
- Robustness to initialization
- Cross-study comparability

**In practice:** Use multiple starting values to check for multiple modes

---

## 9. Identification with Missing Data

### 9.1 The Challenge

Missing observations **reduce** information for identification.

**Extreme case:** If entire row of Λ corresponds to always-missing series, that row is unidentified.

### 9.2 Sufficient Conditions

**For identification with missing data:**

1. **Every factor appears in at least r series** (no complete missingness)
2. **Normalization still holds** (e.g., first r series always observed)
3. **Missing pattern is random** (MAR assumption)

### 9.3 EM Algorithm Advantage

EM algorithm handles missing data **automatically**:
- E-step: Impute factors given observed data
- M-step: Update parameters given imputed factors

No need for explicit missing data model!

---

## 10. Identification Tests

### 10.1 Information Matrix Test

**Idea:** Under correct specification and identification:
```
-E[∂²ℓ/∂θ²] = E[(∂ℓ/∂θ)(∂ℓ/∂θ)']
```

**Test statistic:**
```
T · ||Ĥ - ÔPG||²_F ~ χ²(df)
```

Where:
- Ĥ = numerical Hessian
- ÔPG = outer product of gradients

**Rejection → identification or misspecification problem**

### 10.2 Anderson-Rubin Test

For testing restrictions on identified parameters:

**H₀:** θ = θ₀ (with identification restrictions)

**Test statistic:**
```
LR = 2(ℓ(θ̂) - ℓ(θ₀)) ~ χ²(q)
```

**Valid even with weak identification** (unlike Wald test)

### 10.3 Diagnostic: Rotational Indeterminacy Index

**Measure degree of rotational freedom:**
```
RII = min_R ||Λ̂ - Λ̂R||²_F / ||Λ̂||²_F
```

**Interpretation:**
- RII ≈ 0: Strong identification (unique solution)
- RII >> 0: Weak identification (many equivalent rotations)

---

## 11. Identification in Special Cases

### 11.1 Structural Factor Models

Impose economic structure on loadings:

**Example:** Two-factor model for international data
```
y_{it} = λ_{i,global} f_{global,t} + λ_{i,country} f_{country,t} + ε_{it}

Restrictions:
- λ_{i,global} > 0  (global factor affects all positively)
- λ_{i,country} = 0 for i not in that country
```

**Identification:** Sign + zero restrictions

**Test:** Overidentifying restrictions (if more than r² restrictions)

### 11.2 Block Factor Structure

**Example:** Macro data with sectors
```
y_t = [Λ₁  0   0 ] [f₁_t]   [ε₁_t]
      [0   Λ₂  0 ] [f₂_t] + [ε₂_t]
      [0   0  Λ₃] [f₃_t]   [ε₃_t]

Sector 1: Real activity
Sector 2: Prices
Sector 3: Financial
```

**Identification:** Within-block (easier than full model)

**Advantage:** Economic interpretability built-in

### 11.3 Hierarchical Factor Models

**Two-level structure:**
```
y_t = Λ f_t + ε_t         (Level 1: sector factors)
f_t = Γ g_t + u_t         (Level 2: global factors)
```

**Identification:**
1. Level 1: r₁ sector factors (standard restrictions)
2. Level 2: r₂ global factors (additional restrictions)

**Total restrictions needed:** r₁² + r₂²

---

## 12. Computational Considerations

### 12.1 Initialization Sensitivity

**Problem:** EM algorithm can converge to different local optima depending on initialization

**Solution:** Multiple random starts
```python
best_loglik = -np.inf
for seed in range(10):
    theta_init = initialize_random(seed)
    theta_hat, loglik = em_algorithm(data, theta_init)
    if loglik > best_loglik:
        best_theta = theta_hat
        best_loglik = loglik
```

### 12.2 Imposing Restrictions in Optimization

**Method 1: Parameterization**

Work with transformed parameters that automatically satisfy restrictions:
```python
# Fix first r rows of Λ to I_r
Lambda_free = Lambda[r:, :]  # Only optimize these
```

**Method 2: Penalty**

Add penalty for violating restrictions:
```python
penalty = λ * ||Λ[:r, :] - np.eye(r)||²
objective = -loglik + penalty
```

**Method 3: Projection**

After each update, project onto constraint set:
```python
# After M-step update
Lambda_new[:r, :] = np.eye(r)  # Enforce restriction
```

### 12.3 Checking Identification Numerically

**Test:** Perturb identified parameters slightly
```python
epsilon = 1e-6
for i in range(len(theta)):
    theta_perturbed = theta.copy()
    theta_perturbed[i] += epsilon

    # Check if likelihood changes
    delta_loglik = loglik(theta_perturbed) - loglik(theta)

    if abs(delta_loglik) < 1e-10:
        print(f"Parameter {i} may be unidentified")
```

---

## 13. Recent Advances

### 13.1 Weak Identification

**Recent literature:** Allowing for "weak factors" (weakly identified)

**Implications:**
- Standard inference invalid
- Need robust confidence sets
- Anderson-Rubin type inference

### 13.2 Sparse Factor Models

**L1 regularization on loadings:**
```
penalized_loglik = loglik - λ ∑|Λᵢⱼ|
```

**Effect:**
- Many loadings → 0 (sparse)
- Automatic variable selection
- Identification through sparsity

### 13.3 Machine Learning Approaches

**Autoencoders for factor extraction:**
```
Encoder: f_t = g(y_t; θ_enc)
Decoder: ŷ_t = h(f_t; θ_dec)
```

**Identification:** Network architecture imposes structure

**Advantage:** Nonlinear factors

**Disadvantage:** Even harder to interpret

---

## References

### Foundational Theory
- **Anderson, T.W. & Rubin, H. (1956).** "Statistical Inference in Factor Analysis." *Proc. Third Berkeley Symposium*.
- **Lawley, D.N. & Maxwell, A.E. (1971).** *Factor Analysis as a Statistical Method*.

### Modern Factor Model Theory
- **Bai, J. & Ng, S. (2002).** "Determining the Number of Factors in Approximate Factor Models." *Econometrica*.
- **Bai, J. & Ng, S. (2013).** "Principal Components Estimation and Identification of Static Factors." *Journal of Econometrics*.
- **Stock, J.H. & Watson, M.W. (2002).** "Forecasting Using Principal Components from a Large Number of Predictors." *JASA*.

### State-Space Identification
- **Hannan, E.J. & Deistler, M. (1988).** *The Statistical Theory of Linear Systems*.
- **Bauer, D. (2005).** "Estimating Linear Dynamical Systems Using Subspace Methods." *Econometric Theory*.

### Practical Guides
- **Durbin, J. & Koopman, S.J. (2012).** *Time Series Analysis by State Space Methods*. Chapter 7.
- **Lütkepohl, H. (2005).** *New Introduction to Multiple Time Series Analysis*. Chapter 5.

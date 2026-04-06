# The Static Factor Model: Specification and Assumptions

> **Reading time:** ~7 min | **Module:** Module 1: Static Factors | **Prerequisites:** Module 0 Foundations

<div class="callout-key">

**Key Concept Summary:** The static factor model posits that a large number of observed variables are driven by a small number of latent common factors plus variable-specific idiosyncratic components. This decomposition captures the idea that macroeconomic and financial variables move together because they respond to com...

</div>

## In Brief

The static factor model posits that a large number of observed variables are driven by a small number of latent common factors plus variable-specific idiosyncratic components. This decomposition captures the idea that macroeconomic and financial variables move together because they respond to common underlying shocks.

<div class="callout-insight">

**Insight:** When many variables co-move, there's likely a common cause. Factor models formalize this: instead of modeling $N$ separate series with $N(N-1)/2$ pairwise correlations, we model $r \ll N$ factors that generate all the co-movement. This is both dimensionality reduction and a structural hypothesis about how the world works.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. Model Specification

### Scalar Form

For variable $i$ at time $t$:

$$X_{it} = \lambda_{i1}F_{1t} + \lambda_{i2}F_{2t} + ... + \lambda_{ir}F_{rt} + e_{it}$$

where:
- $X_{it}$: Observed value of variable $i$ at time $t$
- $F_{jt}$: Value of factor $j$ at time $t$ (unobserved)
- $\lambda_{ij}$: Loading of variable $i$ on factor $j$ (sensitivity)
- $e_{it}$: Idiosyncratic error for variable $i$ at time $t$
- $r$: Number of factors (typically small: 1-10)

### Matrix Form (Cross-Section)

For all $N$ variables at time $t$:

$$X_t = \Lambda F_t + e_t$$

where:
- $X_t = (X_{1t}, ..., X_{Nt})'$ is $N \times 1$
- $F_t = (F_{1t}, ..., F_{rt})'$ is $r \times 1$
- $\Lambda = [\lambda_1, ..., \lambda_N]'$ is $N \times r$ (loading matrix)
- $e_t = (e_{1t}, ..., e_{Nt})'$ is $N \times 1$

### Matrix Form (Full Panel)

Stacking all time periods:

$$X = F\Lambda' + e$$

where:
- $X$ is $T \times N$ (observations × variables)
- $F$ is $T \times r$ (observations × factors)
- $\Lambda$ is $N \times r$ (variables × factors)
- $e$ is $T \times N$ (idiosyncratic errors)

### Visual Representation

```
                    Latent Factors F_t (r x 1)
                         [F_1t]
                         [F_2t]
                         [ ⋮  ]
                         [F_rt]
                            │
                            │ × Λ (N x r)
                            ▼
    ┌─────────────────────────────────────────────┐
    │                                             │
    │  X_1t ──────────────────────────────────────│─── e_1t
    │  X_2t ──────────────────────────────────────│─── e_2t
    │   ⋮                                         │     ⋮
    │  X_Nt ──────────────────────────────────────│─── e_Nt
    │                                             │
    └─────────────────────────────────────────────┘
                Observed Variables X_t (N x 1)
```

---

## 2. Standard Assumptions

### Assumption 1: Factor Structure

$$E[e_t | F_t] = 0$$

Idiosyncratic errors are uncorrelated with factors. Factors capture all common variation.

### Assumption 2: Factor Moments

$$E[F_t] = 0, \quad E[F_t F_t'] = \Sigma_F$$

Factors have zero mean (or means absorbed into intercept) and covariance $\Sigma_F$. Often normalized to $\Sigma_F = I_r$.

### Assumption 3: Idiosyncratic Moments

$$E[e_t] = 0, \quad E[e_t e_t'] = \Sigma_e$$

The structure of $\Sigma_e$ distinguishes exact from approximate factor models:
- **Exact:** $\Sigma_e = \text{diag}(\psi_1^2, ..., \psi_N^2)$ (diagonal)
- **Approximate:** $\Sigma_e$ can have off-diagonal elements (weak cross-correlation)

### Assumption 4: Independence Over Time

For static models:
$$E[F_t F_s'] = 0, \quad E[e_t e_s'] = 0 \quad \text{for } t \neq s$$

Dynamic factor models relax this by allowing factor autocorrelation.

---

## 3. Implied Covariance Structure

### Population Covariance

Taking the variance of $X_t = \Lambda F_t + e_t$:

$$\Sigma_X = E[X_t X_t'] = \Lambda E[F_t F_t'] \Lambda' + E[e_t e_t'] = \Lambda \Sigma_F \Lambda' + \Sigma_e$$

With normalized factors ($\Sigma_F = I_r$):

$$\Sigma_X = \Lambda \Lambda' + \Sigma_e$$

### Decomposition of Variance

For variable $i$:
$$\text{Var}(X_{it}) = \underbrace{\sum_{j=1}^r \lambda_{ij}^2 \sigma_{F_j}^2}_{\text{Common variance}} + \underbrace{\psi_i^2}_{\text{Idiosyncratic variance}}$$

**Communality:** Proportion of variance from common factors:
$$h_i^2 = \frac{\sum_j \lambda_{ij}^2 \sigma_{F_j}^2}{\text{Var}(X_{it})}$$

### Covariance Between Variables

$$\text{Cov}(X_{it}, X_{jt}) = \lambda_i' \Sigma_F \lambda_j + \text{Cov}(e_{it}, e_{jt})$$

In exact factor models, $\text{Cov}(e_{it}, e_{jt}) = 0$ for $i \neq j$, so all covariance comes from common factors.

---

## 4. Example: Two-Factor Model

### Specification

Consider $N = 4$ variables driven by $r = 2$ factors:

$$\begin{bmatrix} X_{1t} \\ X_{2t} \\ X_{3t} \\ X_{4t} \end{bmatrix} = \begin{bmatrix} \lambda_{11} & \lambda_{12} \\ \lambda_{21} & \lambda_{22} \\ \lambda_{31} & \lambda_{32} \\ \lambda_{41} & \lambda_{42} \end{bmatrix} \begin{bmatrix} F_{1t} \\ F_{2t} \end{bmatrix} + \begin{bmatrix} e_{1t} \\ e_{2t} \\ e_{3t} \\ e_{4t} \end{bmatrix}$$

### Economic Interpretation

Suppose the four variables are:
- $X_1$: Industrial production
- $X_2$: Employment
- $X_3$: CPI inflation
- $X_4$: Interest rate

A two-factor model might have:
- $F_1$: "Real activity" factor
- $F_2$: "Nominal/inflation" factor

Expected loading pattern:
- IP and Employment load strongly on $F_1$ (real)
- CPI loads strongly on $F_2$ (nominal)
- Interest rate loads on both (responds to real and nominal shocks)

### Simulated Example

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt

# True parameters
np.random.seed(42)
T, N, r = 200, 4, 2

# Factor loadings (with economic structure)
Lambda_true = np.array([
    [0.9, 0.1],   # IP: high on real, low on nominal
    [0.8, 0.2],   # Employment: high on real
    [0.2, 0.9],   # CPI: high on nominal
    [0.5, 0.6],   # Interest rate: both
])

# Simulate factors
F_true = np.random.randn(T, r)

# Idiosyncratic errors
psi = np.array([0.3, 0.3, 0.4, 0.3])  # Idiosyncratic std devs
e = np.random.randn(T, N) * psi

# Generate observed data
X = F_true @ Lambda_true.T + e

# Verify covariance structure
cov_X = np.cov(X.T)
cov_implied = Lambda_true @ Lambda_true.T + np.diag(psi**2)

print("Sample covariance:\n", cov_X.round(2))
print("\nImplied covariance:\n", cov_implied.round(2))
```

</div>

---

## 5. Dimensionality and Parameters

### Parameter Count

Without restrictions:
- Loadings: $N \times r$ parameters
- Factor covariance: $r(r+1)/2$ parameters
- Idiosyncratic variances: $N$ parameters

**Total:** $Nr + r(r+1)/2 + N$

### Observables

Sample covariance matrix has $N(N+1)/2$ unique elements.

### Identification Requirement

Need parameters $\leq$ observables:
$$Nr + \frac{r(r+1)}{2} + N \leq \frac{N(N+1)}{2}$$

This limits how many factors we can identify from a given $N$.

### Example: FRED-MD

With $N = 127$ variables:
- $r = 1$: 128 + 1 + 127 = 256 parameters, 8,128 covariance elements ✓
- $r = 10$: 1,270 + 55 + 127 = 1,452 parameters, 8,128 elements ✓
- Max factors: theoretical limit around $r \approx 62$ for exact identification

In practice, we use far fewer factors (typically 3-8 for macro panels).

---

## 6. Why Factor Models?

### Curse of Dimensionality

With $N$ variables, a VAR has $N^2$ parameters per lag. For FRED-MD ($N = 127$):
- VAR(1): 16,129 parameters
- With $T = 500$: hopelessly overfit

Factor models reduce to $r$ factors:
- Factor VAR(1): $r^2$ parameters
- With $r = 5$: just 25 parameters

### Forecast Aggregation

Instead of choosing among $N$ predictors, factor models optimally aggregate information:
$$\hat{y}_{t+h} = \alpha + \beta' \hat{F}_t$$

Uses all $N$ variables through $r$ factors.

### Structural Interpretation

Factors can represent economic concepts:
- Real activity
- Inflation pressures
- Financial conditions
- Credit spreads

Enables structural analysis without committing to specific observables.

---

## Common Pitfalls

### 1. Confusing Factors and Principal Components

- Factors are latent constructs in a probabilistic model
- PCs are deterministic transformations of data
- Under certain conditions, PCs estimate factors consistently

### 2. Ignoring Identification

- Raw factor estimates are only identified up to rotation
- Interpretation requires normalization choices
- Different software may use different normalizations

### 3. Over-Interpreting Weak Factors

- Factors explaining little variance may be noise
- Focus on strong, robust factors
- Test factor significance formally

---

## Connections

- **Builds on:** Multivariate statistics, PCA
- **Leads to:** Identification (next guide), Dynamic factor models
- **Related to:** CAPM (single-factor model), APT (multi-factor)

---

## Practice Problems

### Conceptual
1. How does the factor model reduce the number of parameters needed to model $N$ correlated variables?
2. If all idiosyncratic variances were zero ($\psi_i = 0$), what would this imply about the data?
3. Why might we expect macroeconomic variables to have a factor structure?

### Mathematical
4. Derive the correlation between $X_{it}$ and $X_{jt}$ in terms of loadings and factor covariance.
5. Show that with $r = N$ factors and no restrictions, the model is not identified.
6. For a single-factor model, derive the correlation matrix in terms of loadings.

### Implementation
7. Simulate data from a 3-factor model with $N = 20$ variables. Verify the covariance structure.
8. Compute and plot the communalities for each variable in your simulation.

---

<div class="callout-insight">

**Insight:** Understanding the static factor model is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

- Lawley, D.N. & Maxwell, A.E. (1971). *Factor Analysis as a Statistical Method*. Classic reference.
- Anderson, T.W. (2003). *An Introduction to Multivariate Statistical Analysis*, 3rd ed. Chapter 14.
- Bai, J. & Ng, S. (2008). "Large Dimensional Factor Analysis." Sections 1-2.

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./01_factor_model_specification_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_static_factor_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_identification_problem.md">
  <div class="link-card-title">02 Identification Problem</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_approximate_factor_models.md">
  <div class="link-card-title">03 Approximate Factor Models</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


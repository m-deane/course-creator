# The Factor Model Identification Problem

> **Reading time:** ~9 min | **Module:** Module 1: Static Factors | **Prerequisites:** Module 0 Foundations

<div class="callout-key">

**Key Concept Summary:** Factor models are not uniquely identified: any invertible transformation of factors and loadings produces observationally equivalent models. This means estimated factors and loadings can be rotated, scaled, or reordered arbitrarily without changing the model's predictions. Identification requires...

</div>

## In Brief

Factor models are not uniquely identified: any invertible transformation of factors and loadings produces observationally equivalent models. This means estimated factors and loadings can be rotated, scaled, or reordered arbitrarily without changing the model's predictions. Identification requires imposing normalization constraints.

<div class="callout-insight">

**Insight:** If you multiply factors by a matrix H and loadings by H^(-1), you get a different but statistically indistinguishable model. This is the **rotational indeterminacy** problem. Without constraints, there are infinitely many equally valid factor representations. Interpretation requires choosing a specific normalization, and different normalizations give different factors.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The Fundamental Identification Problem

### Observable Equivalence

The static factor model is:
$$X_t = \Lambda F_t + e_t$$

Consider any invertible $r \times r$ matrix $H$. Define rotated factors and loadings:
$$\tilde{F}_t = H^{-1}F_t, \quad \tilde{\Lambda} = \Lambda H$$

These produce identical observables:
$$X_t = \Lambda F_t = \Lambda (H H^{-1}) F_t = (\Lambda H)(H^{-1} F_t) = \tilde{\Lambda} \tilde{F}_t$$

### Implications

<div class="flow">
<div class="flow-step mint">1. Non-uniqueness:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step blue">2. Arbitrary rotations:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step amber">3. Arbitrary scaling:</div>
<div class="flow-arrow">&#8594;</div>
<div class="flow-step lavender">4. Arbitrary ordering:</div>
</div>


1. **Non-uniqueness:** Infinitely many $(F, \Lambda)$ pairs fit the data equally well
2. **Arbitrary rotations:** Factors can be rotated in factor space without changing fit
3. **Arbitrary scaling:** Can multiply factor $j$ by $c$ if we divide loadings by $c$
4. **Arbitrary ordering:** Can permute factor order freely

### Visual Representation

```
Original Model:              Rotated Model:
X = Λ F + e                  X = Λ̃ F̃ + e

    F₁ ──┐                      F̃₁ ──┐
    F₂ ──┼── Λ ──> X            F̃₂ ──┼── Λ̃ ──> X
         │                            │
    [Same X!]                    [Same X!]

where F̃ = H⁻¹F and Λ̃ = ΛH
```

---

## 2. Standard Normalization Constraints

To make factors unique, we impose constraints that eliminate degrees of freedom corresponding to $H$.

### Constraint Set 1: PCA Normalization

The principal components approach uses:

**C1.1: Orthonormal factors**
$$\frac{1}{T}F'F = I_r$$

Factors have unit variance and are uncorrelated.

**C1.2: Diagonal loading matrix**
$$\Lambda'\Lambda = D$$
where $D$ is diagonal with decreasing elements: $d_1 \geq d_2 \geq ... \geq d_r \geq 0$

This orders factors by variance explained and makes loadings orthogonal.

**Mathematical Result:**
These constraints identify factors uniquely up to sign flips on each factor-loading pair.

### Constraint Set 2: Factor Analysis Normalization

Classical factor analysis uses:

**C2.1: Factor covariance**
$$E[F_t F_t'] = I_r$$
(Population orthonormal factors)

**C2.2: Loading constraints**
$$\Lambda'\Psi^{-1}\Lambda = \text{diagonal}$$
where $\Psi = \text{diag}(\psi_1^2, ..., \psi_N^2)$ is idiosyncratic variance.

This gives maximum likelihood estimates under normality.

### Constraint Set 3: Economic Identification

For structural interpretation:

**C3.1: Benchmark variable loadings**
Set specific loadings to fixed values:
$$\lambda_{ij} = 1 \text{ for selected } (i,j) \text{ pairs}$$

**C3.2: Zero restrictions**
$$\lambda_{ij} = 0 \text{ for economically justified pairs}$$

Example: "Real activity" factor loads on GDP but not on inflation.

---

## 3. Degrees of Freedom Analysis

### Parameter Counting

**Unrestricted model parameters:**
- Loadings: $N \times r$
- Factor covariance: $r(r+1)/2$ (symmetric)
- Idiosyncratic variances: $N$

Total: $Nr + r(r+1)/2 + N$

**Rotation matrix $H$ degrees of freedom:**
- Invertible $r \times r$ matrix: $r^2$ parameters
- Minus $r$ scale normalizations: $r^2 - r$
- Minus $r$ sign choices: handled separately

**Constraints needed:**
To eliminate $H$, need $r^2$ restrictions.

### Common Constraint Schemes

| Normalization | Constraints Imposed | Remaining Ambiguity |
|---------------|---------------------|---------------------|
| None | 0 | Complete rotational freedom |
| Orthogonal factors | $r(r-1)/2$ | Rotation + scale + order |
| PCA (F'F/T = I) | $r(r+1)/2$ | Sign flips only |
| PCA + ordered | $r(r+1)/2 + r!$ | Sign flips only |
| Structural zeros | Case-dependent | Depends on pattern |

---

## 4. Practical Implications

### PCA Factor Indeterminacy

Even with PCA normalization, factors are only identified **up to sign**:
$$X = \Lambda F = (-\Lambda)(-F)$$

**Software inconsistency:** Different programs may flip signs differently. A factor that loads positively on GDP in one run might load negatively in another.

**Solution:** Post-estimation sign normalization based on economic logic.

### Rotation Invariance

The model's implied covariance is rotation-invariant:
$$\Sigma_X = \Lambda \Lambda' + \Sigma_e = \tilde{\Lambda}\tilde{\Lambda}' + \Sigma_e$$

**Consequence:** Goodness-of-fit measures (R², likelihood) don't change with rotation.

### Interpretation Requires Constraints

Without normalization:
- Cannot interpret factor as "real activity"
- Cannot compare loadings across studies
- Cannot track factors over time consistently

With normalization:
- Factors have economic meaning
- Results are reproducible
- Time series of factors are coherent

---

## 5. Code Implementation

### Demonstrating Rotational Indeterminacy

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

# Simulate true model
np.random.seed(42)
T, N, r = 200, 10, 2

# True factors and loadings (arbitrary)
F_true = np.random.randn(T, r)
Lambda_true = np.random.randn(N, r)
e = np.random.randn(T, N) * 0.3

# Generate data
X = F_true @ Lambda_true.T + e

print("Data covariance structure:")
print(np.cov(X.T)[:3, :3].round(3))

# Apply arbitrary rotation
theta = np.pi / 4  # 45 degree rotation
H = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])

# Rotated factors and loadings
F_rotated = F_true @ H.T
Lambda_rotated = Lambda_true @ H

# Verify observational equivalence
X_reconstructed_original = F_true @ Lambda_true.T
X_reconstructed_rotated = F_rotated @ Lambda_rotated.T

print("\nReconstruction error (original):",
      np.linalg.norm(X - X_reconstructed_original - e))
print("Reconstruction error (rotated):",
      np.linalg.norm(X - X_reconstructed_rotated - e))

# Both should be essentially zero (numerical precision)

# Visualize factor space rotation
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(F_true[:, 0], F_true[:, 1], alpha=0.5)
axes[0].set_xlabel('Original Factor 1')
axes[0].set_ylabel('Original Factor 2')
axes[0].set_title('Original Factors')
axes[0].axhline(0, color='k', linewidth=0.5)
axes[0].axvline(0, color='k', linewidth=0.5)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(F_rotated[:, 0], F_rotated[:, 1], alpha=0.5)
axes[1].set_xlabel('Rotated Factor 1')
axes[1].set_ylabel('Rotated Factor 2')
axes[1].set_title('Rotated Factors (45°)')
axes[1].axhline(0, color='k', linewidth=0.5)
axes[1].axvline(0, color='k', linewidth=0.5)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Key insight: Same data X, different factor representations
print("\nFactors are completely different, but X is identical!")
```

</div>

### Implementing PCA Normalization

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">pca_normalization.py</span>
</div>

```python
def pca_normalization(X, r):
    """
    Extract factors with PCA normalization.

    Constraints:
    1. F'F/T = I_r (orthonormal factors)
    2. Λ'Λ diagonal (orthogonal loadings, variance-ordered)

    Parameters
    ----------
    X : array (T, N)
        Data matrix
    r : int
        Number of factors

    Returns
    -------
    F : array (T, r)
        Estimated factors (normalized)
    Lambda : array (N, r)
        Estimated loadings
    """
    T, N = X.shape

    # Center data
    X_centered = X - X.mean(axis=0)

    # Covariance matrix
    Sigma = X_centered.T @ X_centered / T

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # Sort descending (eigh gives ascending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Extract top r
    Lambda = eigenvectors[:, :r] * np.sqrt(eigenvalues[:r])
    F = X_centered @ eigenvectors[:, :r] / np.sqrt(eigenvalues[:r])

    # Verify normalization
    print("Normalization checks:")
    print(f"F'F/T - I_r (should be ~0):\n{(F.T @ F / T - np.eye(r)).round(6)}")
    print(f"\nΛ'Λ diagonal:\n{(Lambda.T @ Lambda).round(3)}")

    return F, Lambda


# Apply to data
F_hat, Lambda_hat = pca_normalization(X, r=2)

print("\nEstimated loading matrix:")
print(Lambda_hat.round(3))
```

</div>

### Sign Normalization

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">sign_normalize.py</span>
</div>

```python
def sign_normalize(F, Lambda, reference_variables):
    """
    Normalize factor signs based on reference variable loadings.

    Convention: Each factor loads positively on its reference variable.

    Parameters
    ----------
    F : array (T, r)
        Factor matrix
    Lambda : array (N, r)
        Loading matrix
    reference_variables : list of int
        Variable indices for sign normalization (length r)

    Returns
    -------
    F_norm, Lambda_norm : normalized versions
    """
    F_norm = F.copy()
    Lambda_norm = Lambda.copy()

    for j, var_idx in enumerate(reference_variables):
        if Lambda_norm[var_idx, j] < 0:
            # Flip sign of jth factor and jth column of loadings
            F_norm[:, j] *= -1
            Lambda_norm[:, j] *= -1

    return F_norm, Lambda_norm


# Example: Force Factor 1 to load positively on variable 0
# and Factor 2 to load positively on variable 5
F_signed, Lambda_signed = sign_normalize(
    F_hat, Lambda_hat,
    reference_variables=[0, 5]
)

print("Sign-normalized loadings (first 3 variables):")
print(Lambda_signed[:3, :].round(3))
```

</div>

---

## 6. Identification in Practice

### Step-by-Step Identification Strategy

1. **Extract factors:** Use PCA or maximum likelihood
2. **Check normalization:** Verify F'F/T = I and Λ'Λ diagonal
3. **Inspect loadings:** Look at which variables load strongly
4. **Sign normalize:** Choose economically sensible signs
5. **Order factors:** Variance order or economic interpretation order
6. **Label factors:** Assign economic meanings based on loadings

### Example: Macro Factor Identification

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
# After PCA extraction
variable_names = [
    "GDP", "Employment", "Hours", "Investment",
    "CPI", "PPI", "Wages", "FFR",
    "Housing Starts", "Retail Sales"
]

# Examine top loadings for each factor
for j in range(r):
    print(f"\n=== Factor {j+1} ===")
    loading_df = pd.DataFrame({
        'Variable': variable_names,
        'Loading': Lambda_signed[:, j]
    })
    loading_df = loading_df.reindex(
        loading_df['Loading'].abs().sort_values(ascending=False).index
    )
    print(loading_df.head(5))

# Output might show:
# Factor 1: GDP, Employment, Hours, Investment → "Real Activity"
# Factor 2: CPI, PPI, Wages → "Inflation"
```

</div>

---

## Common Pitfalls

### 1. Ignoring Sign Ambiguity

**Problem:** Factor interpretation changes when signs flip unexpectedly.

**Solution:** Always apply post-estimation sign normalization using reference variables.

### 2. Comparing Factors Across Studies

**Problem:** Factors from different datasets aren't directly comparable without common normalization.

**Solution:** Use same variables and normalization scheme, or estimate jointly.

### 3. Over-Interpreting Factor Order

**Problem:** PCA orders by variance explained, not economic importance.

**Solution:** Variance order ≠ causal importance. Economic interpretation comes from loadings.

### 4. Assuming Uniqueness

**Problem:** Treating estimated factors as "the" factors.

**Solution:** Remember factors are just one valid representation; focus on invariant predictions.

---

## Connections

- **Builds on:** Static factor model specification (Guide 01)
- **Leads to:** Approximate factor models (Guide 03), Estimation methods (Module 3)
- **Related to:** Principal components (Module 0), Structural identification in VARs

---

## Practice Problems

### Conceptual

1. Explain why $X = \Lambda F = \tilde{\Lambda}\tilde{F}$ for rotated factors doesn't violate model assumptions.

2. How many free parameters does an invertible $r \times r$ rotation matrix have? Why do we need $r^2$ constraints?

3. If you estimate factors in two subperiods separately, why might they look different even if the true factors are stable?

### Mathematical

4. Prove that the implied covariance $\Sigma_X = \Lambda\Lambda' + \Sigma_e$ is invariant to rotations $H$.

5. Show that PCA normalization (F'F/T = I, Λ'Λ diagonal) imposes exactly $r(r+1)/2$ independent constraints.

6. Derive the transformation $\tilde{F} = H^{-1}F$ that converts arbitrary factors to PCA-normalized factors.

### Implementation

7. Simulate a 3-factor model with $N=20$. Apply three different rotations and verify you get identical $X$.

8. Implement a varimax rotation function (orthogonal rotation to maximize loading variance).

9. Create a function to test whether two factor/loading pairs $(F_1, \Lambda_1)$ and $(F_2, \Lambda_2)$ are observationally equivalent (find $H$ connecting them).

---

<div class="callout-insight">

**Insight:** Understanding the factor model identification problem is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

### Foundational

- **Lawley & Maxwell (1971).** *Factor Analysis as a Statistical Method.* Chapter 4: Identification and rotation.
- **Anderson (2003).** *Multivariate Statistical Analysis.* Sections 14.3-14.4: Indeterminacy and constraints.

### Econometric Perspective

- **Bai & Ng (2008).** "Large Dimensional Factor Analysis." Section 2.2: Identification in approximate factor models.
- **Stock & Watson (2016).** "Dynamic Factor Models." Footnote 5: Why PCA normalization is conventional.

### Rotation Methods

- **Browne (2001).** "An Overview of Analytic Rotation in Exploratory Factor Analysis." *Multivariate Behavioral Research.*
- **Jennrich (1970).** "Orthogonal Rotation Algorithms." *Psychometrika.* Varimax and related methods.

### Structural Identification

- **Bai & Wang (2015).** "Identification and Bayesian Estimation of Dynamic Factor Models." *Journal of Business & Economic Statistics.* Economic restrictions for identification.

---

**Key Takeaway:** Factor models describe covariance structure but don't uniquely identify factors without constraints. PCA normalization is conventional and convenient, but interpretation requires economic judgment and careful sign/label choices.

---

## Conceptual Practice Questions

1. In your own words, explain the difference between common factors and idiosyncratic components.

2. Why do factor models require identification restrictions? Give a concrete example.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./02_identification_problem_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_static_factor_basics.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_factor_model_specification.md">
  <div class="link-card-title">01 Factor Model Specification</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_approximate_factor_models.md">
  <div class="link-card-title">03 Approximate Factor Models</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


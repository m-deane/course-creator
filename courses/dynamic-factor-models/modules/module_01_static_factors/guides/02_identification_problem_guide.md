# The Identification Problem in Factor Models

## In Brief

A factor model is fundamentally unidentified without normalization constraints: any rotation of the factors yields a mathematically equivalent model. Understanding which parameters are identified, which are not, and how to impose identifying constraints is essential before any estimation or interpretation of loadings.

## Key Insight

If $(F_t, \Lambda)$ satisfy $X_t = \Lambda F_t + e_t$, then so does $(RF_t, \Lambda R')$ for any orthogonal matrix $R$. The data cannot distinguish between these observationally equivalent representations. Identification requires additional constraints that pin down the rotation — either normalization conventions, economic restrictions, or algorithmic conventions built into the estimator.

## Formal Definition

**Rotation indeterminacy**: For any $r \times r$ orthogonal matrix $R$ ($RR' = I_r$):
$$\Lambda F_t = (\Lambda R')(R' F_t) \equiv \tilde{\Lambda} \tilde{F}_t$$

Both $(\Lambda, F_t)$ and $(\tilde{\Lambda}, \tilde{F}_t)$ are observationally equivalent — they imply the same covariance structure and the same likelihood.

**Degree of under-identification**: The factor model has $r^2$ degrees of indeterminacy ($r(r-1)/2$ from rotation, $r$ from sign, $r$ from scale). Standard normalizations address these as follows.

**Normalization 1: Diagonal factor covariance**
$$\Sigma_F = I_r \quad \text{(factors orthonormal)}$$

This removes scale ($r$ constraints) but not rotation or sign.

**Normalization 2: Lower triangular loadings**
$$\Lambda = \begin{bmatrix} \lambda_{11} & 0 & \cdots & 0 \\ \lambda_{21} & \lambda_{22} & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ \lambda_{N1} & \lambda_{N2} & \cdots & \lambda_{Nr} \end{bmatrix}, \quad \lambda_{jj} > 0$$

The upper triangular zeros ($r(r-1)/2$ constraints) plus positive diagonal ($r$ sign constraints) together with $\Sigma_F = I_r$ achieve exact identification.

**PCA normalization**: $\Lambda'\Lambda / N = \text{diagonal}$ and $F'F / T = \text{diagonal}$ — the normalization automatically enforced by PCA.

## Intuitive Explanation

Consider a physical analogy: you have a camera (factor model) but you do not know the orientation of the lens (the rotation of the factor space). From the same set of photos, you cannot determine the "true" direction the camera is pointing — only the *subspace* spanned by the $r$ factor directions.

This is not a failure of the model — it is a fundamental property. The covariance structure of $X_t$ only identifies the $r$-dimensional factor subspace, not a specific basis within that subspace. Identifying constraints choose a particular basis for algebraic convenience, statistical efficiency, or economic interpretability.

**Scale indeterminacy**: Factor 1 could be scaled up by 2 if all loadings on factor 1 are divided by 2. We fix this by $\text{Var}(F_{1t}) = 1$.

**Sign indeterminacy**: Factor 1 could be multiplied by $-1$ if all loadings are also multiplied by $-1$. Convention: impose positive loading on a reference variable (e.g., $\lambda_{11} > 0$).

**Rotation indeterminacy**: Even with scale and sign fixed, an orthogonal rotation $R$ maps $(F_t, \Lambda)$ to $(RF_t, \Lambda R')$ preserving $\Sigma_F = I_r$. We fix this by requiring the first $r \times r$ block of $\Lambda$ to be lower triangular.

## Code Implementation

```python
import numpy as np
from numpy.linalg import svd, qr
from sklearn.decomposition import PCA

# --- 1. Demonstrating rotation indeterminacy ---

def demonstrate_rotation_equivalence(Lambda, F, seed=42):
    """
    Show that two different rotations of the same model are observationally equivalent.

    Both give the same implied covariance structure and data.
    """
    np.random.seed(seed)
    T, N = F.shape[0], Lambda.shape[0]
    r = Lambda.shape[1]

    # Generate a random orthogonal rotation matrix
    Q, _ = qr(np.random.randn(r, r))  # Q is orthogonal

    # Rotated version
    Lambda_rot = Lambda @ Q        # New loadings
    F_rot = F @ Q                  # New factors (F' @ Q' = (Q @ F)')

    # Check: covariance structures are identical
    Sigma_original = Lambda @ Lambda.T
    Sigma_rotated = Lambda_rot @ Lambda_rot.T

    max_diff = np.abs(Sigma_original - Sigma_rotated).max()
    print(f"Max difference in implied covariance: {max_diff:.2e} (should be ~0)")
    print(f"Rotation indeterminacy confirmed: {max_diff < 1e-10}")

    return Lambda_rot, F_rot, Q


# --- 2. Lower-triangular normalization ---

def lower_triangular_normalization(Lambda_raw, F_raw=None):
    """
    Apply lower-triangular normalization to identify the factor model.

    For the first r variables, impose:
    - Upper triangle of r x r block = 0
    - Diagonal elements positive

    This is achieved by QR decomposition of the first r rows.

    Parameters
    ----------
    Lambda_raw : ndarray, shape (N, r), unnormalized loadings
    F_raw : ndarray, shape (T, r) or None, corresponding factor estimates

    Returns
    -------
    Lambda_norm : ndarray, shape (N, r), lower-triangular normalized
    F_norm : ndarray, shape (T, r) or None
    R : ndarray, shape (r, r), the rotation applied
    """
    N, r = Lambda_raw.shape

    # QR decomposition of the first r rows gives the rotation
    Q_block, R_block = qr(Lambda_raw[:r, :].T)

    # Ensure positive diagonal (upper triangular R)
    # QR is not unique: signs of columns of Q can flip
    signs = np.sign(np.diag(R_block))
    Q_signed = Q_block * signs
    R_signed = R_block * signs[:, np.newaxis]

    # Apply rotation to all loadings
    Lambda_norm = Lambda_raw @ Q_signed

    # The first r x r block should now be upper triangular (after transposing)
    # We want lower triangular: work with Q_signed.T instead
    Q_final, R_final = qr(Lambda_raw[:r, :])
    signs_final = np.sign(np.diag(R_final))
    R_rotation = (Q_final * signs_final).T

    Lambda_norm = Lambda_raw @ R_rotation.T
    F_norm = F_raw @ R_rotation.T if F_raw is not None else None

    # Verify: upper triangle of first r rows is zero
    upper_tri = np.triu(Lambda_norm[:r, :], k=1)
    print(f"Lower triangular constraint satisfied: {np.abs(upper_tri).max() < 1e-10}")
    print(f"Diagonal positive: {np.all(np.diag(Lambda_norm[:r, :]) > 0)}")

    return Lambda_norm, F_norm, R_rotation


# --- 3. PCA normalization (what sklearn does automatically) ---

def pca_normalization(X, n_components):
    """
    PCA normalization: F'F/T = diag and Lambda'Lambda/N = diag.

    This is what sklearn PCA enforces by convention.
    Note: PCA normalization is NOT the same as lower-triangular normalization.
    Comparing PCA loadings to economic interpretations requires additional rotation.
    """
    T, N = X.shape
    X_centered = X - X.mean(axis=0)

    pca = PCA(n_components=n_components)
    F_hat = pca.fit_transform(X_centered)  # shape (T, r), F'F/T = diag(eigenvalues)
    Lambda_hat = pca.components_.T          # shape (N, r)

    # Verify PCA normalization
    FtF_T = F_hat.T @ F_hat / T
    LtL_N = Lambda_hat.T @ Lambda_hat / N

    print("F'F/T (should be diagonal):")
    print(FtF_T.round(3))
    print("\nLambda'Lambda/N (should be diagonal):")
    print(LtL_N.round(3))

    return F_hat, Lambda_hat


# --- 4. Subspace identification (what IS identified) ---

def factor_subspace_distance(Lambda_1, Lambda_2):
    """
    Compute distance between two factor subspaces.

    The factor subspace (column space of Lambda) is identified,
    even when the specific rotation is not.

    Uses the principal angles between subspaces.
    Distance = 0 means same subspace, distance = 1 means orthogonal.
    """
    # Orthonormalize both
    Q1, _ = qr(Lambda_1)
    Q2, _ = qr(Lambda_2)
    r = Lambda_1.shape[1]
    Q1 = Q1[:, :r]
    Q2 = Q2[:, :r]

    # Singular values of Q1'Q2 are cosines of principal angles
    sv = svd(Q1.T @ Q2, compute_uv=False)
    sv = np.clip(sv, 0, 1)  # numerical safety

    # Distance: sum of sin^2 of principal angles
    distance = np.sqrt(np.sum(1 - sv ** 2))

    return distance, sv


# --- Demonstration ---
if __name__ == "__main__":
    np.random.seed(42)
    T, N, r = 500, 20, 3

    # Generate factor model data
    Lambda_true = np.random.randn(N, r) * 0.8
    F_true = np.random.randn(T, r)
    X = F_true @ Lambda_true.T + np.random.randn(T, N) * 0.3

    print("=== 1. Rotation Indeterminacy ===")
    Lambda_rot, F_rot, Q = demonstrate_rotation_equivalence(Lambda_true, F_true)

    print("\n=== 2. Lower-Triangular Normalization ===")
    Lambda_norm, F_norm, R = lower_triangular_normalization(Lambda_true, F_true)
    print(f"\nFirst {r}x{r} block of normalized Lambda:")
    print(Lambda_norm[:r, :].round(3))

    print("\n=== 3. Subspace Distance After Rotation ===")
    dist, sv = factor_subspace_distance(Lambda_true, Lambda_rot)
    print(f"Distance between original and rotated subspace: {dist:.2e} (should be ~0)")
    dist_random, _ = factor_subspace_distance(Lambda_true, np.random.randn(N, r))
    print(f"Distance to random subspace: {dist_random:.3f} (should be > 0)")

    print("\n=== 4. PCA Normalization ===")
    F_pca, Lambda_pca = pca_normalization(X, n_components=r)

    # PCA and lower-triangular are different rotations of the same subspace
    dist_pca, _ = factor_subspace_distance(Lambda_pca, Lambda_true)
    print(f"\nSubspace distance: PCA vs. true loadings: {dist_pca:.3f}")
    print("(This should be small if sample is large enough for consistent estimation)")
```

## Common Pitfalls

**Comparing loadings across models without accounting for rotation.** Two different software packages, or the same package with different random seeds, can produce very different loading matrices that span the same subspace. Direct comparison of $\hat{\Lambda}_1$ vs $\hat{\Lambda}_2$ is meaningless; compare the factor subspaces instead using principal angles.

**Assuming PCA loadings have economic meaning by default.** PCA produces orthogonal components ordered by variance. These are not "the" factors — they are one particular rotation. Varimax or other rotation criteria often produce more economically interpretable factors.

**Fixing scale but not rotation.** Normalizing $\Sigma_F = I_r$ removes scale but not rotation. Many practitioners stop here and compare loadings directly, which is incorrect.

**The sign convention issue in practice.** After PCA, the sign of each loading vector is arbitrary. If you compare loadings across time periods or across subsamples, random sign flips can make it appear that the factor structure changed dramatically when it has not. Always enforce a consistent sign convention.

**Over-restriction causing misspecification.** Economic identification restrictions (e.g., "factor 1 does not affect variable $j$") can be wrong if the data do not support them. Treating identification restrictions as assumptions rather than testable hypotheses leads to misspecified models.

## Connections

- **Builds on:** Static factor model specification (Guide 1), orthogonal transformations (Module 0 Guide 1)
- **Leads to:** Approximate factor models (Guide 3), MLE estimation with constraints (Module 4), structural FAVAR identification (Module 6)
- **Related to:** Structural VAR identification (same rotation problem); Confirmatory Factor Analysis (prior economic constraints on loadings); ICA (uses non-Gaussianity to identify factors without rotation ambiguity)

## Practice Problems

1. Prove that for any orthogonal matrix $R$, $\Lambda R' (R F_t) = \Lambda F_t$. What does this mean for the covariance structure of $X_t = \Lambda F_t + e_t$?

2. Count the degrees of freedom in the rotation group for $r = 1$, $r = 2$, $r = 3$. Verify your count against the formula $r(r-1)/2$.

3. Implement a function `rotate_to_lower_triangular(Lambda)` that uses QR decomposition to find the rotation $R$ such that $\Lambda R$ has an upper triangular first $r \times r$ block with positive diagonal. Verify on a simulated example.

4. Using `factor_subspace_distance`, compare the PCA-estimated factor subspace with the true factor subspace as $N$ increases from 10 to 500 (with $T = 300$ fixed). Plot the distance as a function of $N$. This illustrates the large-$N$ consistency of PCA.

5. Apply Varimax rotation to PCA loadings extracted from 20 macroeconomic series. Compare the rotated vs. unrotated loadings. Which set is more interpretable?

## Further Reading

- Anderson, T. W. & Rubin, H. (1956). "Statistical Inference in Factor Analysis." *Proc. 3rd Berkeley Symp.* The classic paper establishing identification conditions.
- Bai, J. (2003). "Inferential Theory for Factor Models of Large Dimensions." *Econometrica* 71(1). Modern treatment with asymptotic theory.
- Hyvärinen, A. & Oja, E. (2000). "Independent Component Analysis." *Neural Networks* 13(4-5). ICA uses non-Gaussianity to solve the rotation identification problem.
- Comon, P. (1994). "Independent Component Analysis: A New Concept?" *Signal Processing* 36(3). Foundational ICA reference.

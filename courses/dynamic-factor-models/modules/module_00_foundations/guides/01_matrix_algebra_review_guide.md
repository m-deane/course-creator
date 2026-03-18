# Matrix Algebra Review for Factor Models

## In Brief

Eigendecomposition and singular value decomposition (SVD) are the computational backbone of factor model estimation. Mastering these matrix operations is essential before tackling factor extraction, covariance decomposition, and maximum likelihood optimization.

## Key Insight

A covariance matrix can be split into two parts: a low-rank component driven by common factors, and a diagonal (or near-diagonal) idiosyncratic component. Eigendecomposition finds the natural axes of this structure — the directions of maximum variance correspond to the latent factors.

## Formal Definition

**Eigendecomposition of a symmetric matrix** $A \in \mathbb{R}^{n \times n}$:

$$A = V \Lambda V'$$

where $V = [v_1, \ldots, v_n]$ is orthogonal ($V'V = I$) and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$ contains real eigenvalues sorted in descending order.

**Singular Value Decomposition** of any matrix $X \in \mathbb{R}^{T \times N}$:

$$X = U \Sigma V'$$

where $U \in \mathbb{R}^{T \times T}$ and $V \in \mathbb{R}^{N \times N}$ are orthogonal, and $\Sigma$ is diagonal with non-negative singular values $\sigma_1 \geq \sigma_2 \geq \cdots \geq 0$.

**Positive definiteness**: A symmetric matrix $A$ is positive definite (PD) if $x'Ax > 0$ for all $x \neq 0$, equivalently if all eigenvalues are strictly positive.

## Intuitive Explanation

Think of a covariance matrix as encoding the "shape" of a data cloud. Eigendecomposition finds the principal axes of that shape. The eigenvector with the largest eigenvalue points in the direction of greatest spread — this is the first principal component direction, and in factor models it approximates the first latent factor.

SVD extends this to rectangular matrices. When you have a $T \times N$ data matrix with $T$ observations and $N$ variables, SVD directly finds the low-dimensional structure without first computing the $N \times N$ covariance matrix, which is both computationally cheaper and numerically more stable.

The positive definiteness requirement matters because covariance matrices must satisfy $\text{Var}(a'X) = a'\Sigma a \geq 0$ for all vectors $a$ — variance can never be negative. Any valid covariance matrix is positive semi-definite by construction.

## Code Implementation

```python
import numpy as np
from numpy.linalg import eigh, svd, cholesky

# --- 1. Eigendecomposition for symmetric matrices ---

def eigendecomposition_symmetric(A):
    """
    Eigendecompose a symmetric matrix, returning eigenvalues in descending order.

    np.linalg.eigh is specialized for symmetric/Hermitian matrices:
    - Returns real eigenvalues (no spurious imaginary parts)
    - Roughly 2x faster than np.linalg.eig for symmetric A
    - Returns ASCENDING order, so we must reverse

    Parameters
    ----------
    A : ndarray, shape (n, n), symmetric

    Returns
    -------
    eigenvalues : ndarray, shape (n,), descending
    eigenvectors : ndarray, shape (n, n), columns are eigenvectors
    """
    eigenvalues, eigenvectors = eigh(A)

    # eigh returns ascending order; factor models need descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors


# --- 2. SVD for factor extraction ---

def svd_factor_extraction(X, n_components):
    """
    Extract factors from a data matrix via thin SVD.

    For factor models, the right singular vectors are the factor loadings
    and the scaled left singular vectors are the factor scores.

    Parameters
    ----------
    X : ndarray, shape (T, N), should be mean-centered
    n_components : int, number of factors to extract

    Returns
    -------
    scores : ndarray, shape (T, n_components)
    loadings : ndarray, shape (N, n_components)
    explained_var_ratio : ndarray, shape (n_components,)
    """
    # full_matrices=False computes thin SVD, avoiding T x T U matrix
    U, S, Vt = svd(X, full_matrices=False)

    # Truncate to desired number of components
    scores = U[:, :n_components] * S[:n_components]      # shape (T, r)
    loadings = Vt[:n_components, :].T                     # shape (N, r)

    # Fraction of total variance explained
    total_var = (S ** 2).sum()
    explained_var_ratio = (S[:n_components] ** 2) / total_var

    return scores, loadings, explained_var_ratio


# --- 3. Positive definiteness checks ---

def check_positive_definite(A, tol=1e-10):
    """
    Check whether a symmetric matrix is positive definite.

    Uses eigenvalues: PD iff all eigenvalues > 0.
    Tolerance handles floating-point rounding near zero.
    """
    eigenvalues = np.linalg.eigvalsh(A)
    min_eig = eigenvalues.min()
    return min_eig > tol, min_eig


def make_positive_definite(A, min_eigenvalue=1e-6):
    """
    Project a symmetric matrix to the nearest PD matrix.

    Sets all eigenvalues below min_eigenvalue to min_eigenvalue.
    Used when numerical errors cause covariance estimates to lose PD property.
    """
    eigenvalues, eigvecs = eigh(A)
    # Clip negative or near-zero eigenvalues
    eigenvalues_clipped = np.maximum(eigenvalues, min_eigenvalue)
    return eigvecs @ np.diag(eigenvalues_clipped) @ eigvecs.T


# --- 4. The factor model covariance structure ---

def factor_covariance(Lambda, psi):
    """
    Compute the implied covariance matrix of a factor model.

    For X = Lambda * F + e with F ~ N(0, I_r) and e ~ N(0, diag(psi)):
        Sigma_X = Lambda @ Lambda.T + diag(psi)

    Parameters
    ----------
    Lambda : ndarray, shape (N, r), factor loadings
    psi : ndarray, shape (N,), idiosyncratic variances

    Returns
    -------
    Sigma : ndarray, shape (N, N)
    """
    return Lambda @ Lambda.T + np.diag(psi)


# --- Demonstration ---
if __name__ == "__main__":
    np.random.seed(42)
    T, N, r = 300, 20, 3

    # Generate factor model data
    Lambda_true = np.random.randn(N, r) * 0.8
    F_true = np.random.randn(T, r)
    psi_true = np.abs(np.random.randn(N)) * 0.3 + 0.1
    X = F_true @ Lambda_true.T + np.random.randn(T, N) * np.sqrt(psi_true)
    X_centered = X - X.mean(axis=0)

    # Eigendecomposition of sample covariance
    Sigma_hat = X_centered.T @ X_centered / T
    eigenvalues, eigenvectors = eigendecomposition_symmetric(Sigma_hat)

    print("Top 5 eigenvalues:", eigenvalues[:5].round(3))
    print(f"Variance explained by {r} factors:",
          (eigenvalues[:r].sum() / eigenvalues.sum()).round(3))

    # SVD-based factor extraction
    scores, loadings, var_ratio = svd_factor_extraction(X_centered, n_components=r)
    print(f"\nFactor extraction via SVD:")
    print(f"  Score matrix shape: {scores.shape}")
    print(f"  Loading matrix shape: {loadings.shape}")
    print(f"  Cumulative variance explained: {var_ratio.cumsum().round(3)}")

    # Verify SVD and covariance eigendecomp give equivalent loadings
    # (up to sign and rotation)
    is_pd, min_eig = check_positive_definite(Sigma_hat)
    print(f"\nSample covariance is PD: {is_pd}, min eigenvalue: {min_eig:.4f}")
```

## Common Pitfalls

**`np.linalg.eigh` returns ascending eigenvalues.** Factor models need the largest eigenvalues first (they correspond to the most important factors). Always reverse the order after calling `eigh`:
```python
idx = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]
```

**Sign ambiguity in eigenvectors.** Both $v$ and $-v$ are valid eigenvectors. For reproducibility, enforce a sign convention — for example, require the element with the largest absolute value to be positive. Failure to do this causes factor loadings to flip sign randomly across runs.

**Using `eig` instead of `eigh` for symmetric matrices.** `np.linalg.eig` is general-purpose and may return complex values with small imaginary parts due to floating-point errors. `np.linalg.eigh` is designed for symmetric matrices and always returns real eigenvalues.

**Near-singular covariance matrices when $T < N$.** The sample covariance has rank at most $\min(T, N) - 1$. If $T < N$, the covariance matrix is singular and standard eigendecomposition will produce zero (or near-zero) eigenvalues for the null-space directions. Use SVD on the data matrix instead of eigendecomposing the covariance.

**Forgetting to center data before PCA.** PCA is defined on the covariance matrix, which requires mean-zero data. Without centering, the first principal component captures the mean, not the dominant direction of variation.

## Connections

- **Builds on:** Undergraduate linear algebra (matrix multiplication, orthogonality, determinants)
- **Leads to:** PCA (Module 0, Guide 3), static factor model estimation (Module 3), MLE via Kalman filter (Module 4)
- **Related to:** Matrix calculus for MLE optimization; Cholesky decomposition for simulating correlated errors; numerical linear algebra for large-scale implementations

## Practice Problems

1. If $A$ has eigenvalues $\{1, 2, 3\}$, what are the eigenvalues of $A^2$? Of $A^{-1}$? Of $A + 3I$? Prove your answers.

2. Show that the sum of eigenvalues of a matrix equals its trace: $\sum_i \lambda_i = \text{tr}(A)$. Why does this mean the sum of eigenvalues of a covariance matrix equals total variance?

3. The Eckart-Young theorem states that truncated SVD gives the best low-rank approximation in Frobenius norm. Implement this and verify that 3-factor reconstruction minimizes $\|X - \hat{X}\|_F$ among all rank-3 approximations.

4. Generate a $5 \times 5$ symmetric matrix with a mix of positive and negative eigenvalues. Apply `make_positive_definite` and verify the result is PD. How does the reconstruction compare to the original?

5. For the covariance decomposition $\Sigma_X = \Lambda\Lambda' + \Sigma_e$, write code to: (a) simulate data from this model, (b) recover $\Lambda$ via eigendecomposition of $\Sigma_X$, and (c) quantify how well you recover the true loadings (up to rotation and sign).

6. Implement the matrix derivative $\frac{\partial}{\partial \Sigma} \log|\Sigma| = \Sigma^{-1}$ numerically using finite differences, and verify it matches the analytical formula.

## Further Reading

- Strang, G. (2016). *Introduction to Linear Algebra*, 5th ed., Chapters 6–7. The definitive undergraduate reference for eigendecomposition and SVD with geometric intuition.
- Golub, G. H. & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed. The authoritative reference for numerical linear algebra; Chapter 8 covers SVD algorithms.
- Petersen, K. B. & Pedersen, M. S. (2012). *The Matrix Cookbook*. Free online reference for matrix derivatives and identities used in factor model MLE.
- Trefethen, L. N. & Bau, D. (1997). *Numerical Linear Algebra*. Excellent treatment of why and when numerical methods behave well or poorly.

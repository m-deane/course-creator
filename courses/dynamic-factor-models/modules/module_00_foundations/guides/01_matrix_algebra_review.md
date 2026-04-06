# Matrix Algebra Review for Factor Models

> **Reading time:** ~6 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Factor models rely heavily on matrix decompositions—particularly eigendecomposition and SVD—to extract latent structure from high-dimensional data. This guide reviews the essential linear algebra you'll use throughout the course.

</div>

## In Brief

Factor models rely heavily on matrix decompositions—particularly eigendecomposition and SVD—to extract latent structure from high-dimensional data. This guide reviews the essential linear algebra you'll use throughout the course.

<div class="callout-insight">

**Insight:** The core operation in factor analysis is decomposing a covariance matrix into components that separate "signal" (common factors) from "noise" (idiosyncratic variation). Eigendecomposition provides this separation directly.

</div>
---

## 1. Eigendecomposition

### Intuitive Explanation

<div class="callout-insight">

**Insight:** Think of eigendecomposition as finding the "natural axes" of a transformation. For a covariance matrix, eigenvectors point in directions of maximum/minimum variance, eigenvalues measure the variance in each direction, and the largest eigenvalues capture the most important variation.

</div>

### Formal Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$, an eigenvalue-eigenvector pair $(\lambda, v)$ satisfies:

$$Av = \lambda v$$

where $\lambda$ is a scalar (eigenvalue) and $v \neq 0$ is the eigenvector.

### Symmetric Matrix Decomposition

For symmetric matrices (like covariance matrices), we have the spectral decomposition:

$$A = V \Lambda V'$$

where:
- $V = [v_1, ..., v_n]$ is orthogonal ($V'V = VV' = I$)
- $\Lambda = \text{diag}(\lambda_1, ..., \lambda_n)$ contains eigenvalues
- Eigenvalues are real; eigenvectors are orthogonal

### Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">eigendecomposition_symmetric.py</span>
</div>

```python
import numpy as np

def eigendecomposition_symmetric(A):
    """
    Compute eigendecomposition of symmetric matrix.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric matrix

    Returns
    -------
    eigenvalues : ndarray, shape (n,)
        Eigenvalues in descending order
    eigenvectors : ndarray, shape (n, n)
        Corresponding eigenvectors as columns
    """
    # np.linalg.eigh is optimized for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # Sort in descending order (eigh returns ascending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvalues, eigenvectors

# Example: Covariance matrix
np.random.seed(42)
X = np.random.randn(100, 5)
cov_matrix = X.T @ X / 100

eigenvalues, eigenvectors = eigendecomposition_symmetric(cov_matrix)
print(f"Eigenvalues: {eigenvalues}")
print(f"Sum of eigenvalues (=trace): {eigenvalues.sum():.4f}")
print(f"Trace of cov matrix: {np.trace(cov_matrix):.4f}")
```

</div>
</div>

### Why This Matters for Factor Models

The covariance matrix of observed data $X$ decomposes as:

$$\Sigma_X = \Lambda \Sigma_F \Lambda' + \Sigma_e$$

where $\Lambda$ are factor loadings, $\Sigma_F$ is factor covariance, and $\Sigma_e$ is idiosyncratic covariance. Eigendecomposition helps us separate these components.

---

## 2. Singular Value Decomposition (SVD)

### Formal Definition

For any matrix $X \in \mathbb{R}^{T \times N}$, the SVD is:

$$X = U \Sigma V'$$

where:
- $U \in \mathbb{R}^{T \times T}$ has orthonormal columns (left singular vectors)
- $\Sigma \in \mathbb{R}^{T \times N}$ is diagonal with singular values $\sigma_i \geq 0$
- $V \in \mathbb{R}^{N \times N}$ has orthonormal columns (right singular vectors)

### Reduced (Thin) SVD

For $T > N$, the reduced SVD is more efficient:

$$X = U_r \Sigma_r V'_r$$

where $U_r \in \mathbb{R}^{T \times N}$, $\Sigma_r \in \mathbb{R}^{N \times N}$, $V_r \in \mathbb{R}^{N \times N}$.

### Connection to Eigendecomposition

- $X'X = V \Sigma^2 V'$ (eigendecomposition of Gram matrix)
- $XX' = U \Sigma^2 U'$ (eigendecomposition of outer product)
- Singular values: $\sigma_i = \sqrt{\lambda_i}$ where $\lambda_i$ are eigenvalues of $X'X$

### Intuitive Explanation

SVD decomposes a data matrix into three parts:
- **V columns**: Directions in variable space (like factor loadings)
- **U columns**: Directions in observation space (like factor scores)
- **Σ diagonal**: Importance of each direction (like explained variance)

### Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">svd_decomposition.py</span>
</div>

```python
def svd_decomposition(X, n_components=None):
    """
    Compute SVD and optionally truncate to n_components.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Data matrix (observations x variables)
    n_components : int, optional
        Number of components to retain

    Returns
    -------
    U : ndarray, shape (T, r)
        Left singular vectors (scores direction)
    S : ndarray, shape (r,)
        Singular values
    Vt : ndarray, shape (r, N)
        Right singular vectors transposed (loadings direction)
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    if n_components is not None:
        U = U[:, :n_components]
        S = S[:n_components]
        Vt = Vt[:n_components, :]

    return U, S, Vt

# Example: Extract 2 components
X_centered = X - X.mean(axis=0)
U, S, Vt = svd_decomposition(X_centered, n_components=2)

# Verify reconstruction
X_reconstructed = U @ np.diag(S) @ Vt
reconstruction_error = np.linalg.norm(X_centered - X_reconstructed, 'fro')
print(f"Reconstruction error (rank-2): {reconstruction_error:.4f}")
```

</div>
</div>

### Why SVD for Factor Models?

For large panels ($N$ large), SVD of the data matrix is more efficient than eigendecomposition of the covariance matrix:
- Avoids computing $N \times N$ covariance matrix
- Numerical stability for ill-conditioned problems
- Direct path to PCA-based factor extraction

---

## 3. Positive Definiteness

### Formal Definition

A symmetric matrix $A$ is:
- **Positive definite (PD):** $x'Ax > 0$ for all $x \neq 0$
- **Positive semi-definite (PSD):** $x'Ax \geq 0$ for all $x$

### Equivalent Characterizations

$A$ is PD if and only if:
1. All eigenvalues are positive
2. All leading principal minors are positive
3. $A = B'B$ for some full-rank matrix $B$
4. Cholesky decomposition exists: $A = LL'$

### Why This Matters

Covariance matrices must be PSD (and typically PD in practice):
- Variance can't be negative: $\text{Var}(a'X) = a'\Sigma a \geq 0$
- Ensures well-defined probability distributions
- Factor model covariance: $\Sigma_X = \Lambda\Lambda' + \Sigma_e$ must be PSD

### Code Implementation


<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">check_positive_definite.py</span>
</div>

```python
def check_positive_definite(A, tol=1e-10):
    """
    Check if matrix is positive definite.

    Returns
    -------
    is_pd : bool
        True if positive definite
    min_eigenvalue : float
        Smallest eigenvalue (negative means not PSD)
    """
    eigenvalues = np.linalg.eigvalsh(A)
    min_eigenvalue = eigenvalues.min()
    is_pd = min_eigenvalue > tol

    return is_pd, min_eigenvalue

def make_positive_definite(A, min_eigenvalue=1e-6):
    """
    Project matrix to nearest positive definite matrix.

    Adds small value to diagonal to ensure PD.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

# Example
A = np.array([[1, 0.9, 0.9],
              [0.9, 1, 0.9],
              [0.9, 0.9, 1]])
is_pd, min_eig = check_positive_definite(A)
print(f"Is PD: {is_pd}, min eigenvalue: {min_eig:.4f}")
```


---

## 4. Matrix Calculus Essentials

### Derivatives for Optimization

Factor model estimation often requires gradients. Key results:

$$\frac{\partial}{\partial X} \text{tr}(AX) = A'$$

$$\frac{\partial}{\partial X} \text{tr}(X'AX) = (A + A')X$$

$$\frac{\partial}{\partial X} \log|X| = X^{-1}$$

### The Trace Trick

For scalar-valued functions of matrices, the trace is useful:

$$a'Xa = \text{tr}(a'Xa) = \text{tr}(Xaa')$$

This simplifies expected value calculations in factor models.

---

## Common Pitfalls

### 1. Confusing Eigenvalue Ordering
- `np.linalg.eigh` returns eigenvalues in **ascending** order
- PCA convention uses **descending** order
- Always explicitly sort after decomposition

### 2. Numerical Issues with Near-Singular Matrices
- Sample covariance can be singular if $T < N$
- Add small regularization: $\Sigma + \epsilon I$
- Use SVD instead of eigendecomposition for stability

### 3. Sign Indeterminacy
- Eigenvectors are unique only up to sign
- $v$ and $-v$ are both valid eigenvectors
- For consistency, enforce positive first element convention

---

## Connections

- **Builds on:** Undergraduate linear algebra
- **Leads to:** PCA (next guide), Factor model estimation
- **Related to:** Numerical linear algebra, optimization

---

## Practice Problems

### Conceptual
1. If $A$ has eigenvalues $\{1, 2, 3\}$, what are the eigenvalues of $A^2$? Of $A^{-1}$?
2. Why must the covariance matrix of any random vector be PSD?
3. How does the rank of $X$ relate to the number of non-zero singular values?

### Implementation
4. Write a function to compute the condition number of a matrix using SVD.
5. Implement low-rank approximation: given $X$ and rank $r$, find $\hat{X}$ minimizing $\|X - \hat{X}\|_F$.
6. Verify that for centered data, PCA via eigendecomposition of covariance equals SVD of data matrix.

### Extension
7. The matrix square root $A^{1/2}$ satisfies $A^{1/2}A^{1/2} = A$. Derive it using eigendecomposition.

---

## Further Reading

- Strang, G. (2016). *Introduction to Linear Algebra*, 5th ed. Chapters 6-7.
- Golub, G.H. & Van Loan, C.F. (2013). *Matrix Computations*, 4th ed.
- Petersen, K.B. & Pedersen, M.S. (2012). *The Matrix Cookbook*. [Online]

---

## Conceptual Practice Questions

1. If a covariance matrix has eigenvalues [5.2, 3.1, 0.1, 0.05], how many factors would you extract and why?

2. Explain why eigenvectors of a covariance matrix must be orthogonal and what this means for factor interpretation.

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.


---



## Cross-References

<a class="link-card" href="./01_matrix_algebra_review_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_foundations_review.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./02_time_series_basics.md">
  <div class="link-card-title">02 Time Series Basics</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./03_pca_refresher.md">
  <div class="link-card-title">03 Pca Refresher</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


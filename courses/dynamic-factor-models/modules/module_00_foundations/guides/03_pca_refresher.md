# Principal Component Analysis Refresher

> **Reading time:** ~8 min | **Module:** Module 0: Foundations | **Prerequisites:** Basic linear algebra, Python

<div class="callout-key">

**Key Concept Summary:** Principal Component Analysis (PCA) is the workhorse method for factor extraction. It finds orthogonal directions that maximize variance in the data, providing a natural way to reduce dimensionality while preserving the most important variation.

</div>

## In Brief

Principal Component Analysis (PCA) is the workhorse method for factor extraction. It finds orthogonal directions that maximize variance in the data, providing a natural way to reduce dimensionality while preserving the most important variation.

<div class="callout-insight">

**Insight:** PCA answers: "What are the most important directions of variation in my data?" The first principal component captures the most variance, the second captures the most remaining variance orthogonal to the first, and so on. In factor models, these directions approximate the latent factors driving co-movement.

</div>
---

<div class="callout-warning">

**Warning:** Common implementation pitfalls include numerical instability with poorly conditioned matrices and convergence issues with iterative algorithms. Always validate results against known benchmarks.

</div>

## 1. The PCA Problem

### Setup

Given data matrix $X \in \mathbb{R}^{T \times N}$ with $T$ observations of $N$ variables (assume centered: $\bar{X} = 0$).

### Objective: Variance Maximization

Find direction $w_1 \in \mathbb{R}^N$ (unit vector) that maximizes variance of projections:

$$\max_{w_1: \|w_1\|=1} \text{Var}(Xw_1) = \max_{w_1: \|w_1\|=1} \frac{1}{T} w_1' X' X w_1 = \max_{w_1: \|w_1\|=1} w_1' \Sigma w_1$$

where $\Sigma = \frac{1}{T}X'X$ is the sample covariance matrix.

### Solution via Lagrangian

$$\mathcal{L} = w_1'\Sigma w_1 - \lambda(w_1'w_1 - 1)$$

First-order condition:
$$\frac{\partial \mathcal{L}}{\partial w_1} = 2\Sigma w_1 - 2\lambda w_1 = 0$$

Thus: $\Sigma w_1 = \lambda w_1$

**The first PC loading is the eigenvector of $\Sigma$ with largest eigenvalue.**

### Subsequent Components

The $k$-th PC maximizes variance subject to orthogonality to previous components:

$$\max_{w_k: \|w_k\|=1, w_k \perp w_1,...,w_{k-1}} w_k' \Sigma w_k$$

**Solution:** Eigenvector corresponding to the $k$-th largest eigenvalue.

---

## 2. Equivalent Formulations

### Minimum Reconstruction Error

PCA also solves: find rank-$r$ approximation $\hat{X}$ minimizing squared error:

$$\min_{\hat{X}: \text{rank}(\hat{X}) \leq r} \|X - \hat{X}\|_F^2$$

**Solution:** $\hat{X} = X V_r V_r'$ where $V_r$ contains the first $r$ eigenvectors.

### Via SVD

For centered $X$:
$$X = U \Sigma V'$$

Then:
- **PC loadings:** Columns of $V$ (right singular vectors)
- **PC scores:** $XV = U\Sigma$ (scaled left singular vectors)
- **Eigenvalues of covariance:** $\frac{\sigma_i^2}{T}$ where $\sigma_i$ are singular values

### Intuitive Explanation

Imagine a cloud of data points in high-dimensional space:
- **First PC:** Direction through the cloud along which points are most spread out
- **Second PC:** Perpendicular direction with next most spread
- **Scores:** Coordinates of each point in the new PC basis
- **Loadings:** How each original variable contributes to each PC

---

## 3. Implementation

### Method 1: Via Covariance Eigendecomposition

```python
import numpy as np

def pca_via_covariance(X, n_components=None):
    """
    PCA via eigendecomposition of covariance matrix.

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Data matrix (observations x variables)
    n_components : int, optional
        Number of components (default: all)

    Returns
    -------
    scores : ndarray, shape (T, r)
        Principal component scores
    loadings : ndarray, shape (N, r)
        Principal component loadings
    eigenvalues : ndarray, shape (r,)
        Eigenvalues (variance explained by each PC)
    """
    T, N = X.shape

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance matrix
    cov_matrix = X_centered.T @ X_centered / T

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select components
    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    # Compute scores
    scores = X_centered @ eigenvectors

    return scores, eigenvectors, eigenvalues
```

### Method 2: Via SVD (Recommended for Large N)

```python
def pca_via_svd(X, n_components=None):
    """
    PCA via SVD of data matrix.

    More efficient when N is large (avoids N x N covariance matrix).

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Data matrix
    n_components : int, optional
        Number of components

    Returns
    -------
    scores : ndarray, shape (T, r)
        Principal component scores
    loadings : ndarray, shape (N, r)
        Principal component loadings
    eigenvalues : ndarray, shape (r,)
        Eigenvalues (variance per PC)
    """
    T, N = X.shape

    # Center the data
    X_centered = X - X.mean(axis=0)

    # SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Eigenvalues of covariance
    eigenvalues = S**2 / T

    # Loadings are rows of Vt (columns of V)
    loadings = Vt.T

    # Scores
    scores = U * S  # Equivalent to X @ loadings

    # Select components
    if n_components is not None:
        scores = scores[:, :n_components]
        loadings = loadings[:, :n_components]
        eigenvalues = eigenvalues[:n_components]

    return scores, loadings, eigenvalues
```

### Using scikit-learn

```python
from sklearn.decomposition import PCA

def pca_sklearn(X, n_components=None):
    """PCA using scikit-learn (handles edge cases, standardization options)."""
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    loadings = pca.components_.T  # sklearn stores loadings as rows
    eigenvalues = pca.explained_variance_

    return scores, loadings, eigenvalues, pca
```

---

## 4. Choosing the Number of Components

### Scree Plot

Plot eigenvalues in descending order; look for "elbow" where decline levels off.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">scree_plot.py</span>
</div>

```python
import matplotlib.pyplot as plt

def scree_plot(eigenvalues, title="Scree Plot"):
    """Plot eigenvalues to help choose number of components."""
    n = len(eigenvalues)
    cumulative_var = np.cumsum(eigenvalues) / eigenvalues.sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Eigenvalues
    axes[0].bar(range(1, n+1), eigenvalues, alpha=0.7)
    axes[0].plot(range(1, n+1), eigenvalues, 'ro-')
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title(f'{title} - Eigenvalues')

    # Cumulative variance explained
    axes[1].plot(range(1, n+1), cumulative_var, 'bo-')
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='90% threshold')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained')
    axes[1].set_title(f'{title} - Cumulative Variance')
    axes[1].legend()

    plt.tight_layout()
    return fig
```

</div>

### Kaiser Criterion

Retain components with eigenvalues > 1 (for correlation-matrix PCA).

### Variance Threshold

Retain enough components to explain a target proportion (e.g., 90%) of total variance.

### Bai-Ng Information Criteria (for Factor Models)

Specialized criteria balancing fit and complexity for large panels (covered in Module 3).

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">variance_explained.py</span>
</div>

```python
def variance_explained(eigenvalues, threshold=0.9):
    """Find number of components explaining threshold proportion of variance."""
    total_var = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total_var
    n_components = np.searchsorted(cumulative, threshold) + 1
    return n_components, cumulative
```

</div>

---

## 5. Interpretation

### Loading Interpretation

Each loading $v_{jk}$ tells us how variable $j$ loads on component $k$:
- Large positive: Variable moves with the component
- Large negative: Variable moves opposite to the component
- Near zero: Variable unrelated to the component

### Example: Macroeconomic Factors

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">interpret_loadings.py</span>
</div>

```python
def interpret_loadings(loadings, variable_names, n_top=5):
    """
    Display top-loading variables for each component.

    Parameters
    ----------
    loadings : ndarray, shape (N, r)
        Loading matrix
    variable_names : list
        Names of variables
    n_top : int
        Number of top loaders to show
    """
    n_components = loadings.shape[1]

    for k in range(n_components):
        print(f"\n=== Component {k+1} ===")

        # Sort by absolute loading
        sorted_idx = np.argsort(np.abs(loadings[:, k]))[::-1]

        print("Top positive loaders:")
        for i in sorted_idx[:n_top]:
            if loadings[i, k] > 0:
                print(f"  {variable_names[i]}: {loadings[i, k]:.3f}")

        print("Top negative loaders:")
        for i in sorted_idx[:n_top]:
            if loadings[i, k] < 0:
                print(f"  {variable_names[i]}: {loadings[i, k]:.3f}")
```

</div>

### Rotation

PC loadings can be rotated for easier interpretation without changing fit:
- **Varimax:** Orthogonal rotation maximizing variance of squared loadings
- **Promax:** Oblique rotation allowing correlated components

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">varimax_rotation.py</span>
</div>

```python
from scipy.stats import special_ortho_group

def varimax_rotation(loadings, max_iter=100, tol=1e-6):
    """
    Varimax rotation for interpretable loadings.

    Maximizes sum of variances of squared loadings within each component.
    """
    n_vars, n_factors = loadings.shape
    rotation_matrix = np.eye(n_factors)
    rotated = loadings.copy()

    for _ in range(max_iter):
        old_rotation = rotation_matrix.copy()

        for i in range(n_factors):
            for j in range(i+1, n_factors):
                # Compute rotation angle
                x = rotated[:, i]
                y = rotated[:, j]

                u = x**2 - y**2
                v = 2 * x * y

                num = 2 * np.sum(u * v) - (2/n_vars) * np.sum(u) * np.sum(v)
                den = np.sum(u**2 - v**2) - (1/n_vars) * (np.sum(u)**2 - np.sum(v)**2)

                angle = 0.25 * np.arctan2(num, den)

                # Apply rotation
                c, s = np.cos(angle), np.sin(angle)
                rotated[:, [i, j]] = rotated[:, [i, j]] @ np.array([[c, s], [-s, c]])

        if np.allclose(rotation_matrix, old_rotation, atol=tol):
            break

    return rotated
```

</div>

---

## 6. PCA vs Factor Analysis

### Key Differences

| Aspect | PCA | Factor Analysis |
|--------|-----|-----------------|
| Goal | Dimension reduction | Model latent structure |
| Model | None (just transformation) | $X = \Lambda F + e$ |
| Uniqueness | None (no idiosyncratic) | Explicit idiosyncratic variance |
| Loadings | Eigenvectors | Estimated parameters |
| Number of factors | Rank of data | Model selection problem |

### When They're Similar

For large $N$ with strong factors, PCA estimates converge to true factor loadings (up to rotation). This is the basis for the Stock-Watson approach.

### Code Comparison

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">example.py</span>
</div>

```python
from sklearn.decomposition import PCA, FactorAnalysis

# Generate factor model data
np.random.seed(42)
T, N, r = 200, 20, 3
F_true = np.random.randn(T, r)
Lambda_true = np.random.randn(N, r)
e = np.random.randn(T, N) * 0.5
X = F_true @ Lambda_true.T + e

# PCA
pca = PCA(n_components=r)
scores_pca = pca.fit_transform(X)
loadings_pca = pca.components_.T

# Factor Analysis
fa = FactorAnalysis(n_components=r, random_state=42)
scores_fa = fa.fit_transform(X)
loadings_fa = fa.components_.T

# Compare explained variance
print(f"PCA variance explained: {pca.explained_variance_ratio_.sum():.3f}")
print(f"FA noise variance: {fa.noise_variance_.mean():.3f}")
```

</div>

---

## Common Pitfalls

### 1. Forgetting to Center/Standardize
- PCA on uncentered data gives misleading results
- Standardize (z-score) when variables have different units/scales

### 2. Sign Ambiguity
- Eigenvectors are unique only up to sign
- Establish convention (e.g., positive first element) for reproducibility

### 3. Overfitting with Too Many Components
- Using all components = no dimension reduction
- Cross-validate or use information criteria

### 4. Ignoring Loadings Interpretation
- Don't just use scores blindly
- Examine loadings to understand what components represent

---

## Connections

- **Builds on:** Linear algebra (eigendecomposition, SVD)
- **Leads to:** Static factor models, Stock-Watson estimation
- **Related to:** Factor analysis, independent component analysis

---

## Practice Problems

### Conceptual
1. Prove that PC scores are uncorrelated: $\text{Cov}(z_i, z_j) = 0$ for $i \neq j$ where $z = XV$.
2. Show that total variance is preserved: $\sum_i \text{Var}(z_i) = \sum_j \text{Var}(x_j)$.
3. Why does standardizing variables before PCA change the results?

### Implementation
4. Implement PCA from scratch without using numpy's SVD or eigh directly (use power iteration).
5. Download 10 stock returns. Extract PCs and interpret the first component (market factor?).
6. Compare PCA loadings with/without standardization on a dataset with variables of different scales.

### Extension
7. Implement probabilistic PCA (Tipping & Bishop, 1999) using EM algorithm.
8. Explore the relationship between PCA and linear autoencoders.

---

<div class="callout-insight">

**Insight:** Understanding principal component analysis refresher is essential for building robust models. The concepts here connect directly to the implementation patterns in the companion notebook.

</div>

## Further Reading

- Jolliffe, I.T. (2002). *Principal Component Analysis*, 2nd ed. Springer.
- Shlens, J. (2014). "A Tutorial on Principal Component Analysis." arXiv:1404.1100.
- Tipping, M.E. & Bishop, C.M. (1999). "Probabilistic Principal Component Analysis." *JRSS-B*.

---

## Conceptual Practice Questions

1. Explain the core idea of principal component analysis refresher in your own words to a colleague who has not studied it.

2. What is the most common mistake practitioners make when applying principal component analysis refresher, and how would you avoid it?

<div class="callout-info">

**Info:** These questions test conceptual understanding. Try answering them in your own words before checking the companion slides or notebook.

</div>

---

## Cross-References

<a class="link-card" href="./03_pca_refresher_slides.md">
  <div class="link-card-title">Companion Slides</div>
  <div class="link-card-description">Slide deck covering the same material in presentation format with visual diagrams.</div>
</a>

<a class="link-card" href="../notebooks/01_foundations_review.ipynb">
  <div class="link-card-title">Hands-on Notebook</div>
  <div class="link-card-description">Interactive Jupyter notebook with working implementations and exercises.</div>
</a>

<a class="link-card" href="./01_matrix_algebra_review.md">
  <div class="link-card-title">01 Matrix Algebra Review</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>

<a class="link-card" href="./02_time_series_basics.md">
  <div class="link-card-title">02 Time Series Basics</div>
  <div class="link-card-description">Related guide in this module.</div>
</a>


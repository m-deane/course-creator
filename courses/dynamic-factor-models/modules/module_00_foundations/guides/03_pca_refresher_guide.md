# Principal Component Analysis Refresher

## In Brief

PCA finds the orthogonal directions in a dataset that capture the most variance. For factor models, PCA scores provide consistent estimates of latent factors when the panel is large ($N \to \infty$), making it both the theoretical foundation and a practical first-step estimator.

## Key Insight

The first principal component is the direction that maximizes variance in the data. In a factor model with strong common factors, those factors drive most of the covariance among variables — so the first few PC directions approximate the latent factor space. As $N \to \infty$, PCA loadings converge to the true factor loadings (up to rotation), which is the theoretical justification for the Stock-Watson estimator.

## Formal Definition

**PCA optimization problem**: Given centered data $X \in \mathbb{R}^{T \times N}$, find unit vector $w_1 \in \mathbb{R}^N$ maximizing projected variance:

$$w_1 = \arg\max_{\|w\|=1} w' \hat{\Sigma} w$$

where $\hat{\Sigma} = X'X/T$ is the sample covariance matrix.

**Solution**: $w_1$ is the eigenvector of $\hat{\Sigma}$ corresponding to its largest eigenvalue $\lambda_1$.

**$k$-th component**: Eigenvector of $\hat{\Sigma}$ with the $k$-th largest eigenvalue, subject to orthogonality with $w_1, \ldots, w_{k-1}$.

**Equivalently via SVD**: For centered $X = U\Sigma V'$, the loadings are columns of $V$ and scores are $F = U\Sigma = XV$.

**Variance explained by $r$ components**:
$$\text{VE}(r) = \frac{\sum_{j=1}^r \lambda_j}{\sum_{j=1}^N \lambda_j} = \frac{\sum_{j=1}^r \sigma_j^2}{\|X\|_F^2}$$

## Intuitive Explanation

Imagine a cloud of data points in 3D space shaped like a flat disk tilted at an angle. PCA finds the directions that best describe this shape:
- First PC: the direction through the center with the greatest spread (the "long axis" of the disk)
- Second PC: the second most spread direction, perpendicular to the first (the "short axis" of the disk)
- Third PC: the direction with least spread (perpendicular to the disk face, capturing the disk's flatness)

In economics, if $N = 200$ macroeconomic variables all move together during recessions, there is clearly a low-dimensional structure. PCA extracts this: the first PC likely represents overall economic activity (loading positively on output, employment, sales and negatively on unemployment).

The key result connecting PCA to factor models: the Eckart-Young theorem proves that truncated SVD gives the *best* rank-$r$ approximation to the data in Frobenius norm. This means PCA extracts the maximum amount of common variation using $r$ components — which is exactly what we want from a factor model.

## Code Implementation

```python
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt

# --- 1. PCA via eigendecomposition (pedagogical) ---

def pca_via_covariance(X, n_components=None):
    """
    PCA through eigendecomposition of the sample covariance matrix.

    Best for small N. Steps:
    1. Center the data
    2. Compute N x N covariance matrix
    3. Eigendecompose to get loadings
    4. Project data to get scores

    Returns scores, loadings, eigenvalues
    """
    T, N = X.shape
    X_centered = X - X.mean(axis=0)

    # Sample covariance matrix (N x N)
    Sigma = X_centered.T @ X_centered / T

    # Eigendecompose — eigh for symmetric matrices, returns ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

    # Sort descending: largest eigenvalue = most variance
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    if n_components is not None:
        eigenvalues = eigenvalues[:n_components]
        eigenvectors = eigenvectors[:, :n_components]

    # Scores: project data onto loading directions
    scores = X_centered @ eigenvectors  # shape (T, n_components)

    return scores, eigenvectors, eigenvalues


# --- 2. PCA via SVD (recommended for large N) ---

def pca_via_svd(X, n_components=None):
    """
    PCA through thin SVD of the data matrix.

    Advantages over covariance approach:
    - Avoids forming the N x N covariance matrix (saves O(N^2) memory)
    - Better numerical stability for ill-conditioned problems
    - Works correctly when T < N (rank-deficient covariance)

    For X = U S V', the PCA loadings are V and scores are U*S = X@V.
    Eigenvalues of covariance are S^2 / T.
    """
    T, N = X.shape
    X_centered = X - X.mean(axis=0)

    # full_matrices=False: compute thin SVD (only min(T,N) singular values)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Eigenvalues of covariance matrix
    eigenvalues = S ** 2 / T

    # Loadings: right singular vectors (columns of V = rows of Vt)
    loadings = Vt.T  # shape (N, min(T,N))

    # Scores: left singular vectors scaled by singular values
    scores = U * S   # equivalent to X_centered @ loadings

    if n_components is not None:
        scores = scores[:, :n_components]
        loadings = loadings[:, :n_components]
        eigenvalues = eigenvalues[:n_components]

    return scores, loadings, eigenvalues


# --- 3. Sign convention and normalization ---

def standardize_pca_signs(loadings, scores):
    """
    Enforce sign convention: first non-negligible element of each loading is positive.

    Without this, signs can randomly flip between runs or across implementations,
    making results non-reproducible and comparisons impossible.
    """
    for j in range(loadings.shape[1]):
        # Find first element with absolute value above threshold
        col = loadings[:, j]
        first_nonzero = col[np.abs(col) > 1e-10]
        if len(first_nonzero) > 0 and first_nonzero[0] < 0:
            loadings[:, j] *= -1
            scores[:, j] *= -1
    return loadings, scores


# --- 4. Choosing number of components ---

def explained_variance_analysis(X, max_components=None):
    """
    Compute variance explained for all components to inform factor number selection.

    Returns eigenvalues and cumulative variance fractions for scree plot.
    """
    T, N = X.shape
    if max_components is None:
        max_components = min(T, N)

    X_centered = X - X.mean(axis=0)
    _, _, eigenvalues = pca_via_svd(X_centered, n_components=max_components)

    total_var = eigenvalues.sum()
    cumulative_var = np.cumsum(eigenvalues) / total_var

    return eigenvalues, cumulative_var


def choose_n_components_variance(X, threshold=0.90):
    """Choose minimum components explaining `threshold` proportion of variance."""
    eigenvalues, cumulative_var = explained_variance_analysis(X)
    n_components = int(np.searchsorted(cumulative_var, threshold)) + 1
    print(f"Components needed for {threshold*100:.0f}% variance: {n_components}")
    print(f"Variance explained: {cumulative_var[n_components-1]*100:.1f}%")
    return n_components


def plot_scree(X, max_components=20, title="Scree Plot"):
    """
    Plot eigenvalues and cumulative variance explained.

    The 'elbow' in the eigenvalue plot suggests the number of true factors:
    eigenvalues drop sharply up to the true number, then level off.
    """
    eigenvalues, cum_var = explained_variance_analysis(X, max_components)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7, color='steelblue')
    axes[0].plot(range(1, len(eigenvalues) + 1), eigenvalues, 'ro-', markersize=6)
    axes[0].set_xlabel('Component')
    axes[0].set_ylabel('Eigenvalue')
    axes[0].set_title(f'{title} - Eigenvalues')
    axes[0].axhline(y=1.0, color='gray', linestyle='--', label='Kaiser=1')
    axes[0].legend()

    axes[1].plot(range(1, len(cum_var) + 1), cum_var * 100, 'bo-', markersize=6)
    for threshold in [0.8, 0.9, 0.95]:
        axes[1].axhline(y=threshold * 100, color='red', linestyle='--', alpha=0.5,
                        label=f'{threshold*100:.0f}%')
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Variance Explained (%)')
    axes[1].set_title(f'{title} - Cumulative Variance')
    axes[1].legend()

    plt.tight_layout()
    return fig


# --- 5. PCA vs Factor Analysis comparison ---

def compare_pca_fa(X, n_components=3, seed=42):
    """
    Compare PCA and probabilistic Factor Analysis extractors.

    Key difference:
    - PCA: deterministic transformation maximizing variance
    - FA: probabilistic model X = Lambda*F + e, estimates noise variances
    - For large N, both converge to same factor space (up to rotation)
    """
    pca = PCA(n_components=n_components, random_state=seed)
    scores_pca = pca.fit_transform(X)
    loadings_pca = pca.components_.T  # sklearn stores as rows; transpose to columns

    fa = FactorAnalysis(n_components=n_components, random_state=seed)
    scores_fa = fa.fit_transform(X)
    loadings_fa = fa.components_.T

    results = {
        'pca_variance_ratio': pca.explained_variance_ratio_,
        'pca_total_var': pca.explained_variance_ratio_.sum(),
        'fa_noise_variance_mean': fa.noise_variance_.mean(),
        'pca_loadings': loadings_pca,
        'fa_loadings': loadings_fa,
        'pca_scores': scores_pca,
        'fa_scores': scores_fa
    }

    print(f"PCA variance explained: {results['pca_total_var']:.3f}")
    print(f"FA mean idiosyncratic variance: {results['fa_noise_variance_mean']:.3f}")

    return results


# --- Demonstration ---
if __name__ == "__main__":
    np.random.seed(42)
    T, N, r_true = 500, 50, 3

    # Generate factor model data: X = F @ Lambda' + e
    F_true = np.random.randn(T, r_true)
    Lambda_true = np.random.randn(N, r_true) * 0.9
    psi = np.abs(np.random.randn(N)) * 0.3 + 0.1
    X = F_true @ Lambda_true.T + np.random.randn(T, N) * np.sqrt(psi)

    # Both methods should give same result
    scores_cov, loadings_cov, eigs_cov = pca_via_covariance(X, n_components=r_true)
    scores_svd, loadings_svd, eigs_svd = pca_via_svd(X, n_components=r_true)

    print("Eigenvalue comparison (covariance vs SVD):")
    print(f"  Via covariance: {eigs_cov.round(3)}")
    print(f"  Via SVD:        {eigs_svd.round(3)}")

    # Check variance explained
    _, cum_var = explained_variance_analysis(X, max_components=10)
    print(f"\nVariance explained: {cum_var[:r_true+2].round(3)}")

    # Recommend number of components
    n_rec = choose_n_components_variance(X, threshold=0.80)
```

## Common Pitfalls

**Forgetting to center (or standardize) data.** PCA without centering: the first component captures the mean, not the dominant variation. When variables have different units (e.g., GDP in billions vs. percentage growth rates), standardize first so all variables have unit variance — otherwise PCA will weight large-scale variables more heavily.

**Sign ambiguity.** Eigenvectors are unique only up to sign. Both $v$ and $-v$ are valid first PC loading vectors. Results can randomly differ between runs or implementations. Always impose a sign convention.

**Confusing loadings and scores.** In sklearn, `pca.components_` is $r \times N$ — loadings are stored as **rows**. The standard factor model convention is $N \times r$ (loadings as columns). Always transpose: `loadings = pca.components_.T`.

**PCA is not factor analysis.** PCA is a deterministic linear transformation; factor analysis is a probabilistic model $X = \Lambda F + e$. PCA has no concept of idiosyncratic variance — it attributes all variance to components. FA explicitly models noise. For small $N$, the difference matters. For large $N$ with strong factors, both converge.

**Interpreting components without examining loadings.** PC scores are numbers. They are meaningless without looking at the loadings to understand what each component represents economically.

## Connections

- **Builds on:** Matrix algebra (eigendecomposition, SVD — Module 0 Guide 1)
- **Leads to:** Static factor model specification (Module 1), Stock-Watson PCA estimator (Module 3), initialization for EM algorithm (Module 4)
- **Related to:** Independent Component Analysis (when non-Gaussian factors are needed); autoencoders (nonlinear generalization); factor rotation (Varimax, Promax for interpretability)

## Practice Problems

1. Prove that PC scores are uncorrelated: $\text{Cov}(z_i, z_j) = 0$ for $i \neq j$. Hint: use the orthogonality of loading vectors and the diagonality of the eigenvalue matrix.

2. Show that total variance is preserved: $\sum_{k=1}^N \text{Var}(z_k) = \sum_{j=1}^N \text{Var}(X_j)$. What does this tell you about the information content of all $N$ PCs combined?

3. Why does standardizing variables (dividing by standard deviation) before PCA change the results? When should you standardize and when should you not?

4. Implement PCA from scratch using power iteration (the deflation algorithm), without computing the full covariance matrix or calling SVD. Test on a $1000 \times 100$ matrix and compare runtime against `pca_via_svd`.

5. Generate data from a 3-factor model with $N = 100$ variables. Apply PCA and recover the factor scores. How well can you reconstruct the true factors? What is the minimum $N$ at which reconstruction quality becomes acceptable?

6. Apply PCA to the first 10 US stock returns in the S&P 500 (use daily returns from Yahoo Finance). Interpret the first PC (hint: it usually represents the market factor). What percentage of variance does it explain?

## Further Reading

- Jolliffe, I. T. (2002). *Principal Component Analysis*, 2nd ed. The canonical reference with thorough theoretical treatment.
- Shlens, J. (2014). "A Tutorial on Principal Component Analysis." arXiv:1404.1100. Clear exposition with geometric intuition; highly recommended as a companion read.
- Stock, J. H. & Watson, M. W. (2002). "Forecasting Using Principal Components from a Large Number of Predictors." *JASA* 97(460). The paper establishing PCA as consistent for factor estimation in large panels.
- Tipping, M. E. & Bishop, C. M. (1999). "Probabilistic Principal Component Analysis." *JRSS-B* 61(3). Bridges PCA and factor analysis via the EM algorithm.

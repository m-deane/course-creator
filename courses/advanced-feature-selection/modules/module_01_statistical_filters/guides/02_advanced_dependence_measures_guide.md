# Advanced Dependence Measures: Distance Correlation, HSIC, and MMD

## In Brief

When MI estimation is unreliable (small samples, high dimensionality) or when features are vector-valued, three kernel-based and distance-based statistics offer principled alternatives: distance correlation (dCor), the Hilbert-Schmidt Independence Criterion (HSIC), and Maximum Mean Discrepancy (MMD).

## Key Insight

All three measures share a core structure: they embed random variables in a rich function space (or distance space) and measure whether the joint distribution differs from the product of marginals. The embedding choice determines what dependencies each measure is most sensitive to.

---

## Distance Correlation (Székely)

### Definition and Properties

Distance correlation between $X \in \mathbb{R}^p$ and $Y \in \mathbb{R}^q$ is defined through *distance covariance*:

$$\text{dCov}^2(X, Y) = \frac{1}{n^2} \sum_{k,l} A_{kl} B_{kl}$$

where $A_{kl}$ and $B_{kl}$ are doubly-centred pairwise distance matrices:

$$a_{kl} = \|X_k - X_l\|, \quad A_{kl} = a_{kl} - \bar{a}_{k\cdot} - \bar{a}_{\cdot l} + \bar{a}_{\cdot\cdot}$$

Distance correlation is then:

$$\text{dCor}(X, Y) = \sqrt{\frac{\text{dCov}^2(X, Y)}{\sqrt{\text{dVar}^2(X) \cdot \text{dVar}^2(Y)}}}$$

**Key properties:**

| Property | Pearson $r$ | Spearman $\rho$ | Distance Cor |
|---|---|---|---|
| Detects linear dependencies | Yes | Yes | Yes |
| Detects monotone dependencies | No | Yes | Yes |
| Detects arbitrary nonlinear dependencies | No | No | Yes |
| Handles multivariate $X$, $Y$ | No | No | **Yes** |
| $= 0 \Leftrightarrow$ independence | No | No | **Yes** |
| Range | $[-1, 1]$ | $[-1, 1]$ | $[0, 1]$ |
| Signed | Yes | Yes | No (always $\geq 0$) |

### Implementation

```python
import numpy as np
from scipy.spatial.distance import cdist

def distance_covariance_squared(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute squared distance covariance dCov²(X, Y).

    Parameters
    ----------
    X : array of shape (n_samples, p) or (n_samples,)
    Y : array of shape (n_samples, q) or (n_samples,)

    Returns
    -------
    float
        Squared distance covariance (always >= 0).
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = X.shape[0]

    # Pairwise Euclidean distance matrices
    A = cdist(X, X, metric='euclidean')
    B = cdist(Y, Y, metric='euclidean')

    # Double centering: A_kl = a_kl - row_mean - col_mean + grand_mean
    row_mean_A = A.mean(axis=1, keepdims=True)
    col_mean_A = A.mean(axis=0, keepdims=True)
    grand_mean_A = A.mean()
    A_cent = A - row_mean_A - col_mean_A + grand_mean_A

    row_mean_B = B.mean(axis=1, keepdims=True)
    col_mean_B = B.mean(axis=0, keepdims=True)
    grand_mean_B = B.mean()
    B_cent = B - row_mean_B - col_mean_B + grand_mean_B

    return (A_cent * B_cent).mean()


def distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute distance correlation dCor(X, Y) in [0, 1].
    Returns 0.0 when either variable has zero distance variance.
    """
    dcov2_xy = distance_covariance_squared(X, Y)
    dcov2_xx = distance_covariance_squared(X, X)
    dcov2_yy = distance_covariance_squared(Y, Y)

    denom = np.sqrt(dcov2_xx * dcov2_yy)
    if denom < 1e-10:
        return 0.0

    return float(np.sqrt(max(0.0, dcov2_xy) / denom))
```

### Bias-Corrected Estimator

The standard estimator has $O(1/n)$ bias. The unbiased version uses modified centering:

```python
def dcov_unbiased(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Unbiased distance covariance estimator (Székely & Rizzo 2014).
    Uses U-centering instead of double-centering.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = X.shape[0]
    A = cdist(X, X)
    B = cdist(Y, Y)

    # U-centering: set diagonal to zero, then subtract row/col means
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(B, 0.0)

    row_A = A.sum(axis=1) / (n - 2)
    col_A = A.sum(axis=0) / (n - 2)
    grand_A = A.sum() / ((n - 1) * (n - 2))

    row_B = B.sum(axis=1) / (n - 2)
    col_B = B.sum(axis=0) / (n - 2)
    grand_B = B.sum() / ((n - 1) * (n - 2))

    A_u = A - row_A[:, None] - col_A[None, :] + grand_A
    B_u = B - row_B[:, None] - col_B[None, :] + grand_B

    np.fill_diagonal(A_u, 0.0)
    np.fill_diagonal(B_u, 0.0)

    return (A_u * B_u).sum() / (n * (n - 3))
```

### When to Use Distance Correlation

- Features are multivariate (e.g., time series segments, embeddings)
- You suspect monotone but non-Gaussian dependencies
- You need a signed test of independence with a fast permutation p-value

---

## Hilbert-Schmidt Independence Criterion (HSIC)

### Kernel-Based Dependence

HSIC embeds random variables into a Reproducing Kernel Hilbert Space (RKHS) and measures dependence as the Hilbert-Schmidt norm of the cross-covariance operator:

$$\text{HSIC}(X, Y; k, l) = \mathbb{E}_{X,X',Y,Y'}\left[ k(X,X') \cdot l(Y,Y') \right] + \mathbb{E}_{X,X'}[k(X,X')] \cdot \mathbb{E}_{Y,Y'}[l(Y,Y')] - 2\,\mathbb{E}_{X,Y}\left[ \mathbb{E}_{X'}[k(X,X')] \cdot \mathbb{E}_{Y'}[l(Y,Y')] \right]$$

The finite-sample empirical estimate:

$$\widehat{\text{HSIC}} = \frac{1}{(n-1)^2} \text{tr}(KHLH)$$

where $K_{ij} = k(x_i, x_j)$, $L_{ij} = l(y_i, y_j)$, and $H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^T$ is the centering matrix.

```python
def hsic(X: np.ndarray, Y: np.ndarray,
         sigma_x: float = None, sigma_y: float = None) -> float:
    """
    Hilbert-Schmidt Independence Criterion with RBF kernels.

    Parameters
    ----------
    X : array of shape (n_samples, p) or (n_samples,)
    Y : array of shape (n_samples, q) or (n_samples,)
    sigma_x, sigma_y : float or None
        Kernel bandwidths. If None, use median heuristic.

    Returns
    -------
    float
        HSIC value (>= 0). Higher means more dependence.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n = X.shape[0]

    # Median heuristic for bandwidth
    def median_bandwidth(Z: np.ndarray) -> float:
        dists = cdist(Z, Z, metric='sqeuclidean')
        return np.sqrt(0.5 * np.median(dists[dists > 0]))

    if sigma_x is None:
        sigma_x = median_bandwidth(X)
    if sigma_y is None:
        sigma_y = median_bandwidth(Y)

    # RBF kernels
    K = np.exp(-cdist(X, X, metric='sqeuclidean') / (2 * sigma_x**2))
    L = np.exp(-cdist(Y, Y, metric='sqeuclidean') / (2 * sigma_y**2))

    # Centering matrix H = I - (1/n) 11^T
    H = np.eye(n) - np.ones((n, n)) / n

    # HSIC = tr(KHLH) / (n-1)^2
    KH = K @ H
    LH = L @ H
    return float(np.trace(KH @ LH) / (n - 1)**2)
```

### Kernel Choice and Bandwidth

The RBF (Gaussian) kernel is the standard choice. The **median heuristic** sets bandwidth to the median pairwise distance — robust for unimodal distributions.

For linear kernels ($k(x,x') = x \cdot x'$), HSIC reduces to the squared sample covariance — it is a strict generalisation.

```python
def hsic_linear_kernel(X: np.ndarray, Y: np.ndarray) -> float:
    """HSIC with linear kernel — equals squared covariance."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K = X @ X.T  # linear kernel
    L = Y @ Y.T
    return float(np.trace(K @ H @ L @ H) / (n - 1)**2)
```

### HSIC for Feature Selection

```python
import pandas as pd

def rank_by_hsic(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    Rank features by HSIC with target. Higher score = more relevant.

    Parameters
    ----------
    X : DataFrame, shape (n_samples, n_features)
    y : array of shape (n_samples,)

    Returns
    -------
    pd.Series
        HSIC score per feature, sorted descending.
    """
    y_arr = y.reshape(-1, 1) if y.ndim == 1 else y
    scores = {col: hsic(X[col].values, y_arr) for col in X.columns}
    return pd.Series(scores).sort_values(ascending=False)
```

---

## Maximum Mean Discrepancy (MMD)

### Distribution-Level Feature Relevance

MMD measures the distance between two probability distributions by comparing their mean embeddings in an RKHS:

$$\text{MMD}^2(P, Q; k) = \mathbb{E}_{x,x' \sim P}[k(x,x')] - 2\,\mathbb{E}_{x \sim P, y \sim Q}[k(x,y)] + \mathbb{E}_{y,y' \sim Q}[k(y,y')]$$

For feature selection, split samples by class label and measure how well each feature separates class distributions:

```python
def mmd_squared(X_pos: np.ndarray, X_neg: np.ndarray,
                sigma: float = None) -> float:
    """
    Unbiased MMD² between two sample sets using RBF kernel.

    Parameters
    ----------
    X_pos : array of shape (n_pos, p)
        Samples from class 1 (or positive class).
    X_neg : array of shape (n_neg, p)
        Samples from class 0 (or negative class).
    sigma : float or None
        Kernel bandwidth. None uses median heuristic on pooled data.

    Returns
    -------
    float
        MMD² estimate. Positive values indicate distribution shift.
    """
    if X_pos.ndim == 1:
        X_pos = X_pos.reshape(-1, 1)
    if X_neg.ndim == 1:
        X_neg = X_neg.reshape(-1, 1)

    # Median heuristic on pooled data
    if sigma is None:
        pooled = np.vstack([X_pos, X_neg])
        dists = cdist(pooled, pooled, metric='sqeuclidean')
        sigma = np.sqrt(0.5 * np.median(dists[dists > 0]))

    n = len(X_pos)
    m = len(X_neg)

    K_pp = np.exp(-cdist(X_pos, X_pos, 'sqeuclidean') / (2 * sigma**2))
    K_qq = np.exp(-cdist(X_neg, X_neg, 'sqeuclidean') / (2 * sigma**2))
    K_pq = np.exp(-cdist(X_pos, X_neg, 'sqeuclidean') / (2 * sigma**2))

    # Unbiased estimator: subtract diagonal contributions
    np.fill_diagonal(K_pp, 0.0)
    np.fill_diagonal(K_qq, 0.0)

    term1 = K_pp.sum() / (n * (n - 1))
    term2 = K_qq.sum() / (m * (m - 1))
    term3 = K_pq.sum() / (n * m)

    return term1 + term2 - 2 * term3


def rank_by_mmd(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    Rank features by per-class MMD². Binary classification.
    Higher MMD² means the feature better separates the two classes.
    """
    classes = np.unique(y)
    assert len(classes) == 2, "MMD ranking requires binary classification"

    mask_pos = (y == classes[1])
    scores = {}
    for col in X.columns:
        x_pos = X.loc[mask_pos, col].values
        x_neg = X.loc[~mask_pos, col].values
        scores[col] = mmd_squared(x_pos, x_neg)

    return pd.Series(scores).sort_values(ascending=False)
```

### MMD vs HSIC vs MI for Feature Relevance

MMD is best suited for **distribution shift detection**: does the feature distribution differ meaningfully between classes? This is useful when the classes differ in variance or shape (not just mean), which t-tests and ANOVA miss.

---

## Comparison: When to Use Each Measure

| Situation | Recommended measure | Why |
|---|---|---|
| Large sample, 1-D continuous features | MI (KSG) | Asymptotically unbiased; handles any dependence |
| Multivariate features (e.g., embeddings) | Distance correlation | Handles $\mathbb{R}^p \to \mathbb{R}^q$ natively |
| Non-Gaussian continuous, moderate $n$ | HSIC (RBF kernel) | Kernel flexibility; well-understood null distribution |
| Comparing class distributions | MMD | Directly tests distributional shift, not just mean |
| Mixed discrete/continuous | MI (sklearn) | Handles mixed types cleanly |
| Need p-value / hypothesis test | dCor or HSIC | Permutation tests are straightforward |
| Ultra-high-dimensional ($p > 10^4$) | MI (sklearn, parallel) | O(N log N) per feature; parallelises well |

### Permutation Testing

All four measures support permutation-based p-values:

```python
def permutation_pvalue(X: np.ndarray, y: np.ndarray,
                       stat_fn, n_permutations: int = 1000,
                       random_state: int = 42) -> tuple:
    """
    Compute p-value for any dependence statistic via permutation test.

    Parameters
    ----------
    X : array of shape (n_samples,)
    y : array of shape (n_samples,)
    stat_fn : callable(X, y) -> float
        Any dependence statistic.
    n_permutations : int
    random_state : int

    Returns
    -------
    (observed_stat, p_value)
    """
    rng = np.random.default_rng(random_state)
    observed = stat_fn(X, y)
    null_distribution = np.array([
        stat_fn(X, rng.permutation(y))
        for _ in range(n_permutations)
    ])
    p_value = (null_distribution >= observed).mean()
    return observed, p_value
```

---

## Computational Complexity Comparison

| Measure | Time | Space | Notes |
|---|---|---|---|
| Pearson / Spearman | $O(N)$ | $O(1)$ | Trivial |
| MI (KSG) | $O(N \log N)$ | $O(N)$ | k-d tree |
| Distance correlation | $O(N^2)$ | $O(N^2)$ | Pairwise distance matrix |
| HSIC | $O(N^2)$ | $O(N^2)$ | Kernel matrix |
| MMD | $O(N^2)$ | $O(N^2)$ | Kernel matrix |

For $N > 10^4$, use **random Fourier features (RFF)** to approximate HSIC and MMD in $O(N \cdot D)$ where $D$ is the number of random features (typically $D = 1000$):

```python
from sklearn.kernel_approximation import RBFSampler

def hsic_rff(X: np.ndarray, Y: np.ndarray,
             n_components: int = 1000, gamma: float = 1.0) -> float:
    """
    Fast approximate HSIC using random Fourier feature approximation.
    O(N * D) instead of O(N^2). Suitable for N > 10,000.
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    rff_x = RBFSampler(gamma=gamma, n_components=n_components, random_state=42)
    rff_y = RBFSampler(gamma=gamma, n_components=n_components, random_state=43)

    phi_x = rff_x.fit_transform(X)   # shape (n, D)
    phi_y = rff_y.fit_transform(Y)   # shape (n, D)

    # Approximate HSIC: mean of (phi_x * phi_y) - outer product of means
    n = X.shape[0]
    mean_phi_x = phi_x.mean(axis=0)  # (D,)
    mean_phi_y = phi_y.mean(axis=0)  # (D,)
    mean_phi_xy = (phi_x * phi_y).mean(axis=0)  # (D,) element-wise

    # HSIC ≈ ||mean(phi_x ⊗ phi_y) - mean_phi_x ⊗ mean_phi_y||²
    cross = phi_x.T @ phi_y / n      # (D, D)
    outer = np.outer(mean_phi_x, mean_phi_y)  # (D, D)
    return float(np.sum((cross - outer)**2))
```

---

## Common Pitfalls

- **dCor on high dimensions:** Pairwise distances concentrate as $p$ grows. Use with care when $p \gg n$; consider PCA first.
- **HSIC bandwidth sensitivity:** Wrong bandwidth can render HSIC insensitive to the true dependence. Always use median heuristic as baseline and consider multiple bandwidths.
- **MMD for multi-class problems:** Extend MMD to multi-class via one-vs-rest or compute MMD between all class pairs and aggregate.
- **All are unsigned:** dCor, HSIC, MMD all return non-negative values. You cannot distinguish positive from negative dependence. Use Pearson alongside if sign matters.
- **No feature subset interaction:** These are univariate filters. A feature that is only relevant in interaction with another feature will score near zero.

---

## Connections

- **Builds on:** Mutual information (Guide 01), kernel methods, reproducing kernel Hilbert spaces
- **Leads to:** mRMR with HSIC-based redundancy (Guide 03), kernel PCA, two-sample testing
- **Related to:** MI (information-theoretic analogue), chi-squared test (discrete special case of HSIC with linear kernel)

---

## Practice Problems

1. **Conceptual:** Prove that distance covariance equals zero if and only if $X$ and $Y$ are independent. (Hint: use the characteristic function representation of $\text{dCov}^2$.)

2. **Implementation:** Generate 500 samples from $(X, Y)$ where $Y = \sin(2\pi X) + 0.2\epsilon$ and $X, \epsilon \sim \mathcal{N}(0,1)$. Compute Pearson $r$, Spearman $\rho$, dCor, and HSIC. Which measures detect the dependence? Vary $n$ from 50 to 5000 and plot each measure's value.

3. **Extension:** The linear-kernel HSIC equals the squared Frobenius norm of the sample covariance matrix. Prove this algebraically starting from the definition $\widehat{\text{HSIC}} = \text{tr}(KHLH)/(n-1)^2$ with $K = XX^T$ and $L = YY^T$.

---

## Further Reading

- Székely, G.J., Rizzo, M.L. & Bakirov, N.K. (2007). **Measuring and testing dependence by correlation of distances.** *Annals of Statistics*, 35(6). Original dCor paper.
- Gretton, A., Borgwardt, K., Rasch, M., Schölkopf, B. & Smola, A. (2012). **A kernel two-sample test.** *JMLR*, 13. Foundational MMD reference.
- Song, L., Smola, A., Gretton, A., Borgwardt, K. & Bedo, J. (2007). **Supervised feature selection via dependence estimation.** *ICML*. HSIC for feature selection.

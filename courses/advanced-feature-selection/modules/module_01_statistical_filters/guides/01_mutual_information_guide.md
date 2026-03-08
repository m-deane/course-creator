# Mutual Information for Feature Selection

## In Brief

Mutual information (MI) quantifies how much knowing one variable reduces uncertainty about another. Unlike correlation, MI captures all statistical dependencies — linear and nonlinear — making it a strictly more powerful filter criterion for feature relevance.

## Key Insight

Pearson correlation detects linear relationships. Mutual information detects *any* statistical dependency. A feature with zero correlation but non-zero MI with the target is relevant — and correlation would incorrectly discard it.

---

## Formal Definition

Let $X$ be a feature and $Y$ be the target variable. Their mutual information is:

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}$$

For continuous variables, replace sums with integrals:

$$I(X; Y) = \int\!\!\int p(x,y) \log \frac{p(x,y)}{p(x)\,p(y)}\, dx\, dy$$

Key properties:
- $I(X; Y) \geq 0$, with equality iff $X \perp Y$
- $I(X; Y) = H(Y) - H(Y|X)$ — MI is the reduction in entropy of $Y$ given $X$
- $I(X; Y) = I(Y; X)$ — symmetric
- $I(X; X) = H(X)$ — self-information is entropy

---

## Estimation Methods

Estimating MI from finite samples is the core practical challenge. Four main approaches exist.

### 1. Plug-In Estimator (Histogram)

Discretise both variables, estimate densities by counting, plug into the definition.

```python
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

def mi_plugin(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Plug-in MI estimator via histogram discretisation.

    Parameters
    ----------
    x : array of shape (n_samples,)
        Feature values (continuous or discrete).
    y : array of shape (n_samples,)
        Target values (continuous or discrete).
    n_bins : int
        Number of bins for discretisation.

    Returns
    -------
    float
        Estimated mutual information (nats).
    """
    # Discretise continuous variables
    discretiser = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    x_disc = discretiser.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
    y_disc = discretiser.fit_transform(y.reshape(-1, 1)).ravel().astype(int)

    n = len(x_disc)
    # Joint histogram
    joint_counts = np.zeros((n_bins, n_bins), dtype=int)
    for xi, yi in zip(x_disc, y_disc):
        joint_counts[xi, yi] += 1

    # Marginal histograms
    px = joint_counts.sum(axis=1) / n   # p(x)
    py = joint_counts.sum(axis=0) / n   # p(y)
    pxy = joint_counts / n              # p(x, y)

    # Compute MI, skipping zero-probability bins
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi
```

**Bias:** The plug-in estimator is biased upward (overestimates MI) for small samples. Bias grows with bin count and shrinks with sample size.

**Correction:** The Miller-Madow correction subtracts $(B-1)/(2N)$ where $B$ is the number of occupied bins and $N$ is sample size.

---

### 2. KSG Estimator (Kraskov-Stögbauer-Grassberger)

The KSG estimator uses $k$-nearest neighbours to avoid binning altogether. It is the standard choice for continuous-continuous MI.

$$\hat{I}(X;Y) = \psi(k) + \psi(N) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle$$

where $\psi$ is the digamma function, $N$ is sample size, and $n_x$, $n_y$ count neighbours in marginal spaces within the Chebyshev ball defined by the $k$-th joint neighbour.

```python
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors

def mi_ksg(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """
    KSG mutual information estimator (Algorithm 1 from Kraskov et al. 2004).

    Parameters
    ----------
    x : array of shape (n_samples,)
    y : array of shape (n_samples,)
    k : int
        Number of nearest neighbours. Larger k reduces variance, increases bias.

    Returns
    -------
    float
        Estimated MI in nats. Clamped to zero (MI cannot be negative).
    """
    n = len(x)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xy = np.hstack([x, y])

    # Step 1: find k-th nearest neighbour in joint space (Chebyshev metric)
    knn_joint = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev')
    knn_joint.fit(xy)
    distances, _ = knn_joint.kneighbors(xy)
    # epsilon_i = distance to k-th neighbour (index k, since index 0 is self)
    eps = distances[:, k]

    # Step 2: count neighbours within eps in marginal spaces
    knn_x = NearestNeighbors(metric='chebyshev')
    knn_x.fit(x)
    nx = np.array([len(knn_x.radius_neighbors([xi], radius=e, return_distance=False)[0]) - 1
                   for xi, e in zip(x, eps)])

    knn_y = NearestNeighbors(metric='chebyshev')
    knn_y.fit(y)
    ny = np.array([len(knn_y.radius_neighbors([yi], radius=e, return_distance=False)[0]) - 1
                   for yi, e in zip(y, eps)])

    # Step 3: KSG formula
    mi = digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return max(0.0, mi)  # clamp: negative values are estimation artefacts
```

**Choosing k:** $k = 5$ to $k = 10$ works well in practice. Small $k$ gives high variance; large $k$ introduces bias toward zero for sharp distributions.

---

### 3. Kernel Density Estimation

Estimate continuous densities using KDE, then integrate numerically:

```python
from sklearn.neighbors import KernelDensity
from scipy.integrate import dblquad

def mi_kde(x: np.ndarray, y: np.ndarray, bandwidth: float = 0.5) -> float:
    """
    MI estimation via kernel density estimation.
    Practical for 1-D features; expensive to scale to higher dimensions.
    """
    n = len(x)
    xy = np.column_stack([x, y])

    # Fit joint and marginal KDEs
    kde_xy = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(xy)
    kde_x  = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(x.reshape(-1, 1))
    kde_y  = KernelDensity(bandwidth=bandwidth, kernel='gaussian').fit(y.reshape(-1, 1))

    # Monte Carlo integration using training points as quadrature nodes
    log_pxy = kde_xy.score_samples(xy)
    log_px  = kde_x.score_samples(x.reshape(-1, 1))
    log_py  = kde_y.score_samples(y.reshape(-1, 1))

    # MI = E[log p(x,y) - log p(x) - log p(y)]
    mi = np.mean(log_pxy - log_px - log_py)
    return max(0.0, mi)
```

**Bandwidth selection:** Use Silverman's rule (`bandwidth = 1.06 * std * n^(-1/5)`) or cross-validation.

---

### 4. Adaptive Partitioning

Adaptive k-d tree partitioning places bins where the data is dense, avoiding the uniform-bin bias of histograms. scikit-learn's `mutual_info_classif` and `mutual_info_regression` use this approach internally.

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import pandas as pd

def mi_sklearn(X: pd.DataFrame, y: pd.Series, task: str = 'classification',
               n_neighbors: int = 5, random_state: int = 42) -> pd.Series:
    """
    Compute MI scores for all features using sklearn's adaptive estimator.

    Parameters
    ----------
    X : DataFrame, shape (n_samples, n_features)
    y : Series, shape (n_samples,)
    task : 'classification' or 'regression'
    n_neighbors : int
        Passed to the KSG-based internal estimator.

    Returns
    -------
    pd.Series
        MI score for each feature, indexed by feature name.
    """
    if task == 'classification':
        scores = mutual_info_classif(X, y, n_neighbors=n_neighbors,
                                     random_state=random_state)
    else:
        scores = mutual_info_regression(X, y, n_neighbors=n_neighbors,
                                        random_state=random_state)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)
```

---

## MI for Different Variable Types

| Feature type | Target type | Recommended estimator |
|---|---|---|
| Continuous | Continuous | KSG (`mutual_info_regression`) |
| Continuous | Discrete | KSG (`mutual_info_classif`) |
| Discrete | Discrete | Plug-in (exact, no estimation needed) |
| Mixed | Any | KSG with discrete indicator; sklearn handles automatically |

**Discrete variables:** Pass `discrete_features=True` (or a boolean mask) to sklearn:

```python
# Example: first 3 features are binary, rest continuous
discrete_mask = [True, True, True] + [False] * (X.shape[1] - 3)
scores = mutual_info_classif(X, y, discrete_features=discrete_mask)
```

---

## Conditional Mutual Information and Markov Blanket Discovery

### Conditional MI

The conditional mutual information of $X$ and $Y$ given $Z$ is:

$$I(X; Y \mid Z) = H(Y \mid Z) - H(Y \mid X, Z)$$

It answers: *does X carry additional information about Y beyond what Z already provides?*

```python
def cmi_ksg(x: np.ndarray, y: np.ndarray, z: np.ndarray, k: int = 5) -> float:
    """
    Conditional mutual information I(X;Y|Z) via KSG extension.
    Uses the relation I(X;Y|Z) = I(X;Y,Z) - I(X;Z).
    """
    # Stack dimensions
    yz = np.column_stack([y, z]) if z.ndim == 1 else np.hstack([y.reshape(-1,1), z])

    mi_xyz = mi_ksg(x, yz.ravel() if yz.shape[1] == 1 else yz[:, 0], k=k)
    # Proper multidimensional KSG needed for vector Z; this is a placeholder
    # for the 1-D conditional case
    mi_xz  = mi_ksg(x, z, k=k)
    return max(0.0, mi_xyz - mi_xz)
```

### Markov Blanket Discovery

The Markov blanket of target $Y$ is the minimal set of features $\mathbf{S}$ such that all other features are conditionally independent of $Y$ given $\mathbf{S}$:

$$I(X_i; Y \mid \mathbf{S}) = 0 \quad \forall X_i \notin \mathbf{S}$$

Greedy forward search approximates it:

```python
def markov_blanket_forward(X: pd.DataFrame, y: np.ndarray,
                           threshold: float = 0.01, k: int = 5) -> list:
    """
    Greedy Markov blanket approximation via conditional MI.

    Adds features one at a time while I(X_i; Y | selected) > threshold.
    Stops when no remaining feature passes the threshold.
    """
    remaining = list(X.columns)
    selected  = []

    while remaining:
        best_feature, best_cmi = None, -np.inf

        for feat in remaining:
            if not selected:
                # No conditioning set yet — use unconditional MI
                score = mi_ksg(X[feat].values, y, k=k)
            else:
                cond_vals = X[selected].values
                score = cmi_ksg(X[feat].values, y, cond_vals, k=k)

            if score > best_cmi:
                best_cmi, best_feature = score, feat

        if best_cmi < threshold:
            break  # no more informative features

        selected.append(best_feature)
        remaining.remove(best_feature)

    return selected
```

---

## Normalised MI Variants

Raw MI values are unbounded and depend on marginal entropies, making cross-feature comparison difficult. Normalisation fixes this.

### Normalised MI (NMI)

$$\text{NMI}(X; Y) = \frac{I(X; Y)}{\sqrt{H(X) \cdot H(Y)}}$$

Range: $[0, 1]$. Value of 1 means perfect mutual predictability.

### Adjusted MI (AMI)

NMI is positively biased for high-cardinality discrete features (coincidental MI from many categories). AMI corrects for chance:

$$\text{AMI}(X; Y) = \frac{I(X; Y) - E[I(X; Y)]}{\text{mean}(H(X), H(Y)) - E[I(X; Y)]}$$

where $E[I(X;Y)]$ is the expected MI under a hypergeometric null.

```python
from sklearn.metrics import normalized_mutual_info_score, adjusted_mutual_info_score

# For discrete features/targets
nmi  = normalized_mutual_info_score(y_true, y_pred)
ami  = adjusted_mutual_info_score(y_true, y_pred)

# For continuous features, discretise first then compute
from sklearn.preprocessing import KBinsDiscretizer
disc = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
x_disc = disc.fit_transform(x.reshape(-1, 1)).ravel().astype(int)
nmi_cont = normalized_mutual_info_score(x_disc, y)
```

**When to use AMI vs NMI:**
- Use **NMI** for continuous features after KSG estimation (entropy normalisation)
- Use **AMI** when comparing cluster assignments or high-cardinality discrete features
- Use **raw MI** when doing mRMR (the ratio structure of NMI can distort redundancy terms)

---

## Practical Issues

### Sample Size Requirements

KSG estimation degrades badly below ~50 samples per feature. A rough guideline:

| Estimated MI (nats) | Minimum samples for reliable estimate |
|---|---|
| > 1.0 | 100 |
| 0.1 – 1.0 | 500 |
| < 0.1 | > 2000 |

For small datasets, bootstrap the MI estimate and report confidence intervals:

```python
def mi_bootstrap_ci(x: np.ndarray, y: np.ndarray,
                    n_bootstrap: int = 200, alpha: float = 0.05,
                    k: int = 5) -> tuple:
    """
    Bootstrap confidence interval for MI.

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    n = len(x)
    point = mi_ksg(x, y, k=k)
    boot_mis = []

    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_mis.append(mi_ksg(x[idx], y[idx], k=k))

    lower = np.percentile(boot_mis, 100 * alpha / 2)
    upper = np.percentile(boot_mis, 100 * (1 - alpha / 2))
    return point, lower, upper
```

### Bias Correction

All estimators are biased. Common corrections:

- **KSG:** Bias is $O(1/N)$; tends to underestimate for small samples
- **Plug-in:** Bias is $+\frac{B-1}{2N}$ (Miller-Madow)
- **Jackknife:** Compute MI on $n-1$ subsets and subtract leave-one-out mean

### Computational Cost

| Estimator | Time complexity | Notes |
|---|---|---|
| Plug-in (histogram) | $O(N \cdot B^2)$ | Cheap; resolution-limited |
| KSG | $O(N \log N)$ | k-d tree; standard choice |
| KDE | $O(N^2)$ | Expensive; avoid for $N > 10^4$ |
| sklearn adaptive | $O(N \log N)$ | KSG internally; fastest in practice |

For high-dimensional datasets ($p > 1000$ features), parallelise the per-feature MI computation:

```python
from joblib import Parallel, delayed

def mi_all_features_parallel(X: pd.DataFrame, y: np.ndarray,
                              k: int = 5, n_jobs: int = -1) -> pd.Series:
    """Compute MI for each feature in parallel."""
    def _single_mi(col):
        return mi_ksg(X[col].values, y, k=k)

    scores = Parallel(n_jobs=n_jobs)(delayed(_single_mi)(col) for col in X.columns)
    return pd.Series(scores, index=X.columns).sort_values(ascending=False)
```

---

## Common Pitfalls

- **Equidistant bins with skewed data:** Use `strategy='quantile'` in `KBinsDiscretizer`, not `'uniform'`
- **Negative MI estimates from KSG:** Clamp to zero — they are sampling artefacts, not a sign of suppression
- **Comparing MI scores across different targets:** Raw MI depends on $H(Y)$; normalise before comparing
- **Treating MI as a distance:** MI is not a metric. Use $\sqrt{1 - \text{NMI}}$ if you need a proper distance
- **Ignoring feature scale:** KSG uses Euclidean/Chebyshev distances; standardise before calling

---

## Connections

- **Builds on:** Shannon entropy, information theory fundamentals, k-NN methods
- **Leads to:** mRMR feature selection (Guide 03), conditional independence testing, Markov blanket algorithms
- **Related to:** Distance correlation (Guide 02), HSIC (Guide 02), chi-squared test (for discrete variables)

---

## Practice Problems

1. **Conceptual:** Why does Pearson correlation underestimate the dependence between $Y = X^2 + \epsilon$ and $X$? Derive the exact Pearson $r$ for $X \sim \mathcal{N}(0,1)$ and show it equals zero. Then argue that MI correctly identifies the dependence.

2. **Implementation:** Load the diabetes dataset from sklearn, compute MI scores using both the KSG estimator and `mutual_info_regression`, and compare the rankings. Explain any differences.

3. **Extension:** Prove that $I(X; Y) = 0$ iff $X$ and $Y$ are statistically independent. (Hint: use the non-negativity of KL divergence.)

---

## Further Reading

- Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). **Estimating mutual information.** *Physical Review E*, 69(6). The original KSG paper.
- Ross, B.C. (2014). **Mutual information between discrete and continuous data sets.** *PLOS ONE*, 9(2). Handles mixed-type estimation cleanly.
- Vergara, J.R. & Estévez, P.A. (2014). **A review of feature selection methods based on mutual information.** *Neural Computing and Applications*, 24(1–2). Survey of MI-based methods including mRMR.

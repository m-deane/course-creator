# mRMR, FCBF, and the Relief Family

## In Brief

All univariate filters share a critical flaw: they select features independently, ignoring redundancy among selected features. A feature perfectly correlated with an already-selected feature adds no new information to the model. mRMR, FCBF, and Relief-family methods explicitly account for inter-feature dependencies while remaining computationally tractable.

## Key Insight

Relevance is necessary but not sufficient for a good feature set. Two features each with high MI with the target but high MI with each other carry almost the same information — keeping both wastes model capacity and can hurt generalisation. The filter stage must balance relevance *and* redundancy.

---

## Minimum Redundancy Maximum Relevance (mRMR)

### Motivation

Suppose features $f_1$ and $f_2$ are both highly relevant (high $I(f_i; Y)$) but $f_1 \approx f_2$ (high $I(f_1; f_2)$). A pure MI filter ranks both highly. mRMR penalises the second feature for its redundancy with the first.

### Formal Objective

Given a candidate feature set $S_m$ of size $m$, mRMR adds the feature $f_j \notin S_m$ that maximises:

**MID (Mutual Information Difference):**
$$\phi_\text{MID}(f_j) = I(f_j; Y) - \frac{1}{|S_m|} \sum_{f_i \in S_m} I(f_j; f_i)$$

**MIQ (Mutual Information Quotient):**
$$\phi_\text{MIQ}(f_j) = \frac{I(f_j; Y)}{\frac{1}{|S_m|} \sum_{f_i \in S_m} I(f_j; f_i)}$$

MID and MIQ differ in their trade-off: MID applies an additive penalty; MIQ scales relevance by redundancy. MIQ is more aggressive at suppressing redundant features when relevance is small.

### From-Scratch Implementation

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

def compute_mi_matrix(X: pd.DataFrame, task: str = 'classification',
                      n_neighbors: int = 5, random_state: int = 42) -> pd.DataFrame:
    """
    Compute pairwise MI matrix among features and between features and target.

    Returns
    -------
    pd.DataFrame of shape (n_features, n_features)
        MI values among features. Diagonal = MI(f_i, f_i) = H(f_i) (entropy).
    """
    n, p = X.shape
    mi_matrix = np.zeros((p, p))

    for i, col in enumerate(X.columns):
        # MI of feature i with all other features (treat each as "target")
        if task == 'classification':
            # Discretise for MI between features
            mi_row = mutual_info_classif(
                X.drop(columns=[col]), X[col],
                n_neighbors=n_neighbors, random_state=random_state
            )
        else:
            mi_row = mutual_info_regression(
                X.drop(columns=[col]), X[col],
                n_neighbors=n_neighbors, random_state=random_state
            )

        # Fill row and column (MI is symmetric)
        other_indices = [j for j in range(p) if j != i]
        for k, j in enumerate(other_indices):
            mi_matrix[i, j] = mi_row[k]
            mi_matrix[j, i] = mi_row[k]

    return pd.DataFrame(mi_matrix, index=X.columns, columns=X.columns)


def mrmr(X: pd.DataFrame, y: np.ndarray, n_features: int,
         variant: str = 'MID', task: str = 'classification',
         n_neighbors: int = 5, random_state: int = 42) -> list:
    """
    Minimum Redundancy Maximum Relevance feature selection.

    Parameters
    ----------
    X : DataFrame, shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_features : int
        Number of features to select.
    variant : 'MID' or 'MIQ'
        MID: subtract redundancy. MIQ: divide by redundancy.
    task : 'classification' or 'regression'
    n_neighbors : int
        KSG neighbours for MI estimation.

    Returns
    -------
    list of str
        Selected feature names in selection order.
    """
    # Step 1: compute MI with target for all features
    if task == 'classification':
        relevance = mutual_info_classif(X, y, n_neighbors=n_neighbors,
                                        random_state=random_state)
    else:
        relevance = mutual_info_regression(X, y, n_neighbors=n_neighbors,
                                           random_state=random_state)
    relevance = pd.Series(relevance, index=X.columns)

    # Step 2: compute pairwise MI among features
    mi_features = compute_mi_matrix(X, task=task, n_neighbors=n_neighbors,
                                    random_state=random_state)

    # Step 3: greedy forward selection
    selected = []
    remaining = list(X.columns)

    # First feature: highest relevance (no redundancy to subtract yet)
    first = relevance.idxmax()
    selected.append(first)
    remaining.remove(first)

    for _ in range(n_features - 1):
        scores = {}
        for feat in remaining:
            rel = relevance[feat]
            # Average redundancy with already-selected features
            red = mi_features.loc[feat, selected].mean()

            if variant == 'MID':
                scores[feat] = rel - red
            elif variant == 'MIQ':
                # Avoid division by zero for first iteration
                scores[feat] = rel / (red + 1e-10)
            else:
                raise ValueError(f"Unknown variant: {variant}. Use 'MID' or 'MIQ'.")

        best = max(scores, key=scores.get)
        selected.append(best)
        remaining.remove(best)

    return selected
```

### Practical Usage

```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

# Select top 10 features using MID variant
selected_mid = mrmr(X_scaled, y, n_features=10, variant='MID')
selected_miq = mrmr(X_scaled, y, n_features=10, variant='MIQ')

print("MID selected:", selected_mid)
print("MIQ selected:", selected_miq)
print("Agreement:", set(selected_mid) & set(selected_miq))
```

### MID vs MIQ: When to Prefer Each

| Situation | Prefer MID | Prefer MIQ |
|---|---|---|
| Features with similar relevance | Yes | No |
| High feature correlation overall | No | Yes |
| Target has low entropy (near-constant) | Either | MIQ (ratio more stable) |
| Features span very different MI scales | MID | No — ratio unstable |

---

## Fast Correlation-Based Filter (FCBF)

FCBF is designed for ultra-high-dimensional data (thousands to millions of binary or discrete features). It uses Symmetric Uncertainty (SU) — a normalised MI variant — and applies a two-phase algorithm: relevance thresholding followed by redundancy removal.

$$\text{SU}(X, Y) = 2 \cdot \frac{I(X; Y)}{H(X) + H(Y)} \in [0, 1]$$

```python
def symmetric_uncertainty(x: np.ndarray, y: np.ndarray,
                           n_neighbors: int = 5) -> float:
    """
    Symmetric uncertainty SU(X, Y) = 2 * I(X;Y) / (H(X) + H(Y)).

    For discrete variables, compute entropy from frequencies.
    For continuous, approximate entropy from MI using H(X) = I(X;X) (KSG).
    """
    from scipy.stats import entropy as scipy_entropy
    from sklearn.feature_selection import mutual_info_regression

    mi = mutual_info_regression(x.reshape(-1, 1), y, n_neighbors=n_neighbors)[0]

    # Estimate marginal entropies via histogram
    def entropy_from_hist(v, n_bins=20):
        counts, _ = np.histogram(v, bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        return scipy_entropy(probs)

    hx = entropy_from_hist(x)
    hy = entropy_from_hist(y)
    denom = hx + hy
    return float(2 * mi / denom) if denom > 1e-10 else 0.0


def fcbf(X: pd.DataFrame, y: np.ndarray, threshold: float = 0.01,
         n_neighbors: int = 5) -> list:
    """
    Fast Correlation-Based Filter.

    Phase 1: Keep features with SU(f, y) > threshold.
    Phase 2: Remove features that are more correlated with another
             selected feature than with the target.

    Parameters
    ----------
    X : DataFrame
    y : array
    threshold : float
        Minimum SU with target to be considered relevant.

    Returns
    -------
    list of str
        Selected feature names.
    """
    # Phase 1: relevance filtering
    su_target = {col: symmetric_uncertainty(X[col].values, y, n_neighbors)
                 for col in X.columns}
    relevant = [col for col, su in su_target.items() if su > threshold]
    # Sort by SU with target, descending
    relevant = sorted(relevant, key=lambda c: su_target[c], reverse=True)

    if not relevant:
        return []

    # Phase 2: redundancy elimination
    # For each feature f_i, check if any higher-SU feature f_j makes f_i redundant
    # f_i is redundant given f_j if SU(f_i, f_j) >= SU(f_i, y)
    selected = [relevant[0]]  # keep highest-SU feature

    for candidate in relevant[1:]:
        is_redundant = False
        for sel in selected:
            su_pair = symmetric_uncertainty(X[candidate].values, X[sel].values)
            if su_pair >= su_target[candidate]:
                is_redundant = True
                break
        if not is_redundant:
            selected.append(candidate)

    return selected
```

**When to use FCBF:** Binary or discrete features, $p > 10^4$, fast runtime required. FCBF runs in $O(p \log p + |S|^2)$ after threshold filtering, making it practical for genomics or text feature selection.

---

## Relief Algorithm Family

### Core Idea

Relief evaluates features by how well they distinguish between samples from the same class versus different classes. It is inherently multivariate: the neighbourhood structure implicitly captures interactions.

For each sample, Relief compares its feature values to:
- Its **nearest hit** $H$: closest sample of the *same* class
- Its **nearest miss** $M$: closest sample of a *different* class

A feature is relevant if it separates misses (small difference from hits, large difference from misses).

### ReliefF

ReliefF extends the original Relief to multi-class problems using $k$ nearest hits and misses:

$$W[f] \mathrel{{+}{=}} -\frac{1}{mk} \sum_{i=1}^{m} \sum_{j=1}^{k} \text{diff}(f, x_i, H_j^i) + \frac{1}{mk} \sum_{i=1}^{m} \sum_{c \neq c(x_i)} \frac{P(c)}{1 - P(c(x_i))} \sum_{j=1}^{k} \text{diff}(f, x_i, M_j^{i,c})$$

where $\text{diff}(f, x_i, x_j) = |x_i^f - x_j^f|$ (normalised to $[0,1]$) and $P(c)$ is the class prior.

```python
from sklearn.preprocessing import MinMaxScaler

def relieff(X: pd.DataFrame, y: np.ndarray,
            k: int = 10, n_iterations: int = None,
            random_state: int = 42) -> pd.Series:
    """
    ReliefF feature weighting algorithm.

    Parameters
    ----------
    X : DataFrame, shape (n_samples, n_features)
        Features (should be normalised to [0, 1] range).
    y : array of shape (n_samples,)
        Class labels.
    k : int
        Number of nearest hits and misses per iteration.
    n_iterations : int or None
        Number of random instances to sample. Default: all instances.

    Returns
    -------
    pd.Series
        Feature weights sorted descending. Positive = relevant.
    """
    # Normalise features to [0, 1] for diff calculation
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    X_arr = X_norm.values
    y_arr = np.array(y)

    n, p = X_arr.shape
    classes = np.unique(y_arr)
    class_priors = {c: (y_arr == c).mean() for c in classes}

    n_iter = n if n_iterations is None else min(n_iterations, n)
    rng = np.random.default_rng(random_state)
    sample_indices = rng.choice(n, size=n_iter, replace=False)

    weights = np.zeros(p)

    for idx in sample_indices:
        xi = X_arr[idx]
        ci = y_arr[idx]

        # Find k nearest hits (same class)
        same_class_mask = (y_arr == ci) & (np.arange(n) != idx)
        same_class_dists = np.linalg.norm(X_arr[same_class_mask] - xi, axis=1)
        hit_indices = np.where(same_class_mask)[0][np.argsort(same_class_dists)[:k]]

        # Update weights: penalise features that differ from hits
        for hit_idx in hit_indices:
            weights -= np.abs(xi - X_arr[hit_idx]) / (k * n_iter)

        # Find k nearest misses per class (weighted by class prior)
        for c in classes:
            if c == ci:
                continue
            other_class_mask = y_arr == c
            other_class_dists = np.linalg.norm(X_arr[other_class_mask] - xi, axis=1)
            miss_indices = np.where(other_class_mask)[0][np.argsort(other_class_dists)[:k]]

            prior_weight = class_priors[c] / (1.0 - class_priors[ci])

            # Update weights: reward features that differ from misses
            for miss_idx in miss_indices:
                weights += prior_weight * np.abs(xi - X_arr[miss_idx]) / (k * n_iter)

    return pd.Series(weights, index=X.columns).sort_values(ascending=False)
```

### SURF and MultiSURF: Adaptive Neighbourhood

ReliefF's fixed-$k$ neighbourhood is a hyperparameter. SURF and MultiSURF replace it with adaptive thresholds based on the data's average pairwise distance:

```python
def multisurf(X: pd.DataFrame, y: np.ndarray) -> pd.Series:
    """
    MultiSURF: adaptive neighbourhood Relief variant.
    Uses all instances within average pairwise distance / 2 as hits/misses.
    No k hyperparameter needed.
    """
    from sklearn.metrics import pairwise_distances

    X_norm = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    X_arr = X_norm.values
    y_arr = np.array(y)
    n, p = X_arr.shape

    # Pairwise distance matrix
    D = pairwise_distances(X_arr)
    # Adaptive radius: half the average non-self pairwise distance
    avg_dist = D[D > 0].mean()
    radius = avg_dist / 2.0

    weights = np.zeros(p)

    for i in range(n):
        xi = X_arr[i]
        ci = y_arr[i]
        dists_i = D[i]

        # Neighbours within radius (exclude self)
        neighbour_mask = (dists_i < radius) & (np.arange(n) != i)
        neighbours = np.where(neighbour_mask)[0]

        if len(neighbours) == 0:
            continue

        hits  = [j for j in neighbours if y_arr[j] == ci]
        misses = [j for j in neighbours if y_arr[j] != ci]

        n_total = len(hits) + len(misses)
        if n_total == 0:
            continue

        # Hits: penalise
        for j in hits:
            weights -= np.abs(xi - X_arr[j]) / (n * n_total)

        # Misses: reward
        for j in misses:
            weights += np.abs(xi - X_arr[j]) / (n * n_total)

    return pd.Series(weights, index=X.columns).sort_values(ascending=False)
```

### How Relief Captures Feature Interactions

Consider two features $A$ and $B$ where the class label is $Y = \text{XOR}(A, B)$. Neither $A$ alone nor $B$ alone is predictive — they only matter together.

| Filter method | Result |
|---|---|
| MI (univariate) | $I(A; Y) \approx 0$, $I(B; Y) \approx 0$ — both discarded |
| Pearson | $r(A, Y) = 0$, $r(B, Y) = 0$ — both discarded |
| **ReliefF** | Both $A$ and $B$ receive positive weights — both kept |

ReliefF correctly identifies both because the distance between XOR-opposite instances is larger in the $(A, B)$ joint space than in either marginal space alone. The neighbourhood structure captures this interaction implicitly.

---

## Comparison Table: mRMR vs Relief vs FCBF

| Criterion | mRMR | ReliefF / MultiSURF | FCBF |
|---|---|---|---|
| Captures interactions | No (pairwise MI only) | Yes (implicit, k-NN) | No (pairwise SU only) |
| Handles redundancy | Yes (explicit) | Partially (through neighbourhood) | Yes (explicit SU threshold) |
| Scalable to $p > 10^4$ | No ($O(p^2)$ MI matrix) | $O(n^2 p)$ — moderate | Yes ($O(p \log p)$ after filtering) |
| Requires $k$ tuning | No | Yes ($k$ hits/misses) | No ($\delta$ threshold) |
| Works with mixed types | With care | Yes | Best for discrete |
| Supports continuous target | Yes (regression MI) | Yes | Less natural |
| Theoretical guarantee | MI-optimal greedy | Consistency results | Thresholded SU |

### Decision Guide

```
High-dimensional discrete features (text, genomics, $p > 10^4$)?
  → FCBF: fastest, designed for this regime

Suspect feature interactions (parity, threshold effects, XOR structure)?
  → ReliefF or MultiSURF: only filter-stage methods that capture interactions

Standard continuous features, $p < 1000$, want redundancy-aware selection?
  → mRMR (MID for balanced relevance, MIQ for aggressive redundancy removal)

All of the above + need a full ranking (not just selected set)?
  → Run mRMR and ReliefF; compare rankings; investigate disagreements
```

---

## Common Pitfalls

- **mRMR with correlated target:** If the target is nearly a linear combination of features, mRMR with MIQ can produce unstable rankings due to small-denominator instability. Use MID.
- **ReliefF with k too large:** Large $k$ smooths out the neighbourhood and misses local interactions. Start with $k = 10$ and cross-validate.
- **FCBF threshold sensitivity:** The SU threshold $\delta$ acts like a hard cutoff. A small $\delta$ keeps many features; a large $\delta$ keeps very few. Try $\delta \in \{0.001, 0.01, 0.1\}$ and check stability.
- **All three are filter methods:** They do not account for the learning algorithm. Wrapper methods (e.g., Boruta, recursive feature elimination) may perform better at the cost of computational expense.
- **Ignoring class imbalance in ReliefF:** With severe imbalance, the nearest miss from the minority class is rare. Weight by class prior (ReliefF formula) — do not use the original Relief (no weighting).

---

## Connections

- **Builds on:** MI estimation (Guide 01), normalised MI variants, k-nearest neighbours
- **Leads to:** Wrapper methods (GA feature selection), embedded methods (LASSO, random forest importance), Boruta algorithm
- **Related to:** CMIM (conditional MI maximisation), JMI (joint MI), DISR (double-input symmetrical relevance) — all variants of the redundancy-relevance trade-off

---

## Practice Problems

1. **Conceptual:** Prove that mRMR's MID objective is equivalent to maximising the lower bound on the conditional MI $I(f_j; Y \mid S_m)$ under the assumption that features in $S_m$ are independent. (Hint: use the chain rule of MI and the independence assumption.)

2. **Implementation:** Generate a dataset with an XOR interaction between two features, add 10 noise features, and run both MI (univariate) and ReliefF. Show that MI discards the XOR features while ReliefF keeps them.

3. **Extension:** Design a variant of mRMR that uses distance correlation instead of MI for both relevance and redundancy terms. What changes in the greedy selection algorithm? Does it handle interactions better?

---

## Further Reading

- Peng, H., Long, F. & Ding, C. (2005). **Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy.** *IEEE TPAMI*, 27(8). Original mRMR paper.
- Yu, L. & Liu, H. (2003). **Feature selection for high-dimensional data: a fast correlation-based filter solution.** *ICML*. FCBF.
- Kononenko, I. (1994). **Estimating attributes: analysis and extensions of Relief.** *ECML*. ReliefF.
- Urbanowicz, R.J. et al. (2018). **Relief-based feature selection: introduction and review.** *Journal of Biomedical Informatics*, 85. Comprehensive survey of the Relief family.

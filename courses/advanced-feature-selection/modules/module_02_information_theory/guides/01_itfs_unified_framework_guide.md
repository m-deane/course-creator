# Information-Theoretic Feature Selection: The Unified Framework

## In Brief

Information-theoretic feature selection (ITFS) criteria rank features by measuring statistical dependence between features and the target. Brown et al. (2012) proved that JMI, CMIM, DISR, ICAP, and mRMR are all special cases of a single generalised criterion — the Conditional Likelihood Maximisation (CLM) framework — differing only in how they approximate the interaction term between selected and candidate features.

## Key Insight

Every major ITFS criterion solves the same underlying problem: maximise the relevance of a candidate feature $x_k$ to the target $y$, while penalising redundancy with already-selected features $\mathbf{x}_S$. The criteria differ in which approximation they use for the intractable joint distribution over $\mathbf{x}_S$. Understanding this unification lets you choose criteria on principled grounds rather than by trial-and-error.

## Formal Definitions and Notation

Throughout this guide we use:

- $X = \{x_1, \ldots, x_p\}$ — the full feature set
- $S \subseteq X$ — the currently selected subset, $|S| = k$
- $y$ — the target variable
- $H(X)$ — Shannon entropy: $H(X) = -\sum_x p(x) \log p(x)$
- $I(X; Y)$ — mutual information: $I(X;Y) = H(X) - H(X|Y)$
- $I(X; Y | Z)$ — conditional mutual information: $I(X;Y|Z) = H(X|Z) - H(X|Y,Z)$

The **chain rule for mutual information**:

$$I(X;Y|Z) = I(X;Y) - I(X;Z) + I(X;Z|Y)$$

is the algebraic identity underlying the entire Brown et al. framework.

---

## 1. Minimum Redundancy Maximum Relevance (mRMR)

### Derivation

mRMR (Peng et al., 2005) selects the feature $x_k$ that maximises:

$$J_\text{mRMR}(x_k) = I(x_k; y) - \frac{1}{|S|} \sum_{x_j \in S} I(x_k; x_j)$$

The first term maximises **relevance** (mutual information with the target). The second term penalises **redundancy** by subtracting the average pairwise MI between $x_k$ and all already-selected features.

### Properties

- Computational complexity: $O(pk)$ MI evaluations at each step
- Ignores class-conditional structure: the redundancy term does not condition on $y$
- Known weakness: may over-penalise features that are redundant given $S$ but complementary given $y$

### Connection to CLM Framework

In Brown et al.'s notation, mRMR uses $\beta = 1$ and $\gamma = 0$:

$$J_\text{CLM}(x_k) = I(x_k; y) - \beta \sum_{x_j \in S} I(x_k; x_j) + \gamma \sum_{x_j \in S} I(x_k; x_j | y)$$

mRMR sets $\gamma = 0$ (ignores class-conditional complementarity) and $\beta = 1/|S|$ (averages over $S$).

---

## 2. Joint Mutual Information (JMI)

### Derivation

JMI (Yang and Moody, 1999) selects the feature $x_k$ maximising the joint mutual information with $y$ over all pairs $(x_k, x_j)$ for $x_j \in S$:

$$J_\text{JMI}(x_k) = \sum_{x_j \in S} I(x_k, x_j; y)$$

Expanding using the chain rule:

$$I(x_k, x_j; y) = I(x_j; y) + I(x_k; y | x_j)$$

Since $I(x_j; y)$ is constant across candidates $x_k$, the criterion reduces to maximising:

$$J_\text{JMI}(x_k) = \sum_{x_j \in S} I(x_k; y | x_j)$$

### Equivalent Expansion

Using $I(x_k; y | x_j) = I(x_k; y) - I(x_k; x_j) + I(x_k; x_j | y)$:

$$J_\text{JMI}(x_k) = |S| \cdot I(x_k; y) - \sum_{x_j \in S} I(x_k; x_j) + \sum_{x_j \in S} I(x_k; x_j | y)$$

### Properties

- $O(pk)$ evaluations of **conditional** MI — more expensive than mRMR per step
- Captures complementarity: the $I(x_k; x_j | y)$ term rewards features that are jointly informative about $y$ even when pairwise redundant
- Shown empirically to be one of the strongest ITFS criteria (Brown et al., 2012)

### CLM Parameters

$\beta = 1/|S|$, $\gamma = 1/|S|$ — symmetrically weights both redundancy and complementarity.

---

## 3. Conditional Mutual Information Maximisation (CMIM)

### Derivation

CMIM (Fleuret, 2004) takes a min-max approach. Instead of summing over $S$, it selects the feature that maximises the worst-case conditional MI:

$$J_\text{CMIM}(x_k) = \min_{x_j \in S} I(x_k; y | x_j)$$

The intuition: a feature is good only if it remains informative about $y$ even given the most informative feature already selected. The min ensures robustness — a feature cannot be selected by virtue of being complementary to one already-selected feature while being redundant with another.

### Why the Min?

CMIM solves:

$$x_k^* = \argmax_{x_k \notin S} \min_{x_j \in S} I(x_k; y | x_j)$$

This is equivalent to finding the feature that provides the maximum guaranteed additional information regardless of which element of $S$ is conditioned on. It is a worst-case (minimax) guarantee rather than an average-case one.

### Properties

- Computational complexity: $O(pk)$ conditional MI evaluations, but the min is cheaper than a sum
- More conservative than JMI — may under-select features with uneven conditional MI profile
- Particularly effective when the selected set contains one dominant feature that makes many candidates redundant
- Exact computation requires $O(|S|)$ comparisons per candidate

### CLM Parameters

CMIM corresponds to $\beta = 1$, $\gamma = 1$ applied to the single most-constraining $x_j \in S$:

$$J_\text{CLM}(x_k) = I(x_k; y) - I(x_k; x_j^*) + I(x_k; x_j^* | y)$$

where $x_j^* = \argmin_{x_j \in S} I(x_k; y | x_j)$.

---

## 4. Double Input Symmetrical Relevance (DISR)

### Derivation

DISR (Meyer et al., 2008) normalises JMI by the joint entropy:

$$J_\text{DISR}(x_k) = \frac{1}{|S|} \sum_{x_j \in S} \frac{I(x_k, x_j; y)}{H(x_k, x_j, y)}$$

The normalisation term $H(x_k, x_j, y)$ is the joint entropy of all three variables. This makes DISR scale-invariant — it measures the fraction of joint uncertainty explained by the pair $(x_k, x_j)$ about $y$.

### Symmetrical Uncertainty Interpretation

DISR extends the concept of **symmetrical uncertainty** $U(X;Y) = 2I(X;Y) / (H(X) + H(Y))$ to the three-variable case. The factor 2 is absorbed by summing over the pair in both orderings, giving the "double input" name.

$$\text{SU}(x_k, x_j; y) = \frac{I(x_k, x_j; y)}{H(x_k, x_j, y)}$$

### Properties

- Range: $[0, 1]$ — enables natural thresholding
- Handles different entropy scales across features — useful when features have very different cardinalities
- Computationally identical to JMI plus one entropy computation per pair
- Well-suited to mixed discrete/continuous features after discretisation

### CLM Parameters

DISR uses $\beta = 1/|S|$, $\gamma = 1/|S|$ (same as JMI) but with normalisation — it does not fit cleanly into the additive CLM form; Brown et al. treat it as a JMI variant with normalisation.

---

## 5. Interaction Capping (ICAP)

### Derivation

ICAP (Jakulin, 2005) is motivated by bounding the interaction information. It selects:

$$J_\text{ICAP}(x_k) = I(x_k; y) - \sum_{x_j \in S} \max\{0, I(x_k; x_j) - I(x_k; x_j | y)\}$$

The term inside the max is the **interaction information** (or co-information):

$$\text{Int}(x_k; x_j; y) = I(x_k; x_j) - I(x_k; x_j | y)$$

When $\text{Int} > 0$: $x_k$ and $x_j$ are **redundant** with respect to $y$ (knowing $y$ reduces their mutual information). When $\text{Int} < 0$: they are **synergistic** (knowing $y$ increases their mutual information).

ICAP **caps** the penalty at zero — it never rewards synergy (i.e., $\gamma_j = 0$ when $I(x_k; x_j | y) > I(x_k; x_j)$) but penalises redundancy.

### Properties

- Conservative: avoids the over-selection of synergistic features that can harm JMI in noisy settings
- The capping introduces an asymmetry — redundant pairs are penalised but synergistic pairs are not rewarded
- Useful when you suspect the target has many spurious interactions (financial data with look-ahead bias)

### CLM Parameters

$\beta_j = 1$ when $I(x_k; x_j) > I(x_k; x_j | y)$, else $0$. $\gamma_j = 1$ when $I(x_k; x_j) > I(x_k; x_j | y)$, else $0$.

---

## 6. The Brown et al. (2012) Unified Framework

### The Unifying Theorem

Brown et al. (2012) — "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection" — show that all major ITFS criteria maximise:

$$\boxed{J_\text{CLM}(x_k) = I(x_k; y) + \sum_{x_j \in S} \left[ \gamma_j \cdot I(x_k; x_j | y) - \beta_j \cdot I(x_k; x_j) \right]}$$

where $\beta_j \geq 0$ controls redundancy penalisation and $\gamma_j \geq 0$ controls complementarity reward. Each criterion corresponds to a specific choice of $(\beta_j, \gamma_j)$:

| Criterion | $\beta_j$ | $\gamma_j$ | Notes |
|-----------|-----------|------------|-------|
| mRMR | $1/\|S\|$ | $0$ | No complementarity term |
| JMI | $1/\|S\|$ | $1/\|S\|$ | Symmetric treatment |
| CMIM | $1$ (min only) | $1$ (min only) | Applied to worst-case $x_j$ |
| ICAP | $\mathbb{1}[\text{Int}>0]$ | $\mathbb{1}[\text{Int}>0]$ | Capped at zero |
| DISR | $1/\|S\|$ | $1/\|S\|$ | Plus entropy normalisation |

### Derivation of the Unified Objective

Start from the log-likelihood of a naive Bayes classifier with selected features $\mathbf{x}_S \cup \{x_k\}$:

$$\ell(\theta) = \sum_i \log p(y_i | \mathbf{x}_{S,i}, x_{k,i})$$

Under conditional independence assumptions over $S$:

$$\log p(y | \mathbf{x}_S, x_k) \approx \log p(y | x_k) + \sum_{x_j \in S} \left[\log p(y | x_j) + \log p(x_k | x_j, y) - \log p(x_k | y)\right]$$

Taking the expectation over the data distribution and rearranging:

$$\mathbb{E}[\ell] = I(x_k; y) + \sum_{x_j \in S} \left[I(x_k; x_j | y) - I(x_k; x_j)\right] + \text{const}$$

This is exactly the CLM objective with $\beta_j = \gamma_j = 1$, which corresponds to JMI. mRMR, CMIM, and ICAP arise from different approximations to the same likelihood.

### Key Implications of Unification

1. **No single criterion is uniformly best.** The optimal choice of $(\beta, \gamma)$ depends on the true data-generating process.
2. **JMI is the MLE estimate** of the conditional likelihood — it is the most principled criterion in the framework.
3. **mRMR under-penalises** when features in $S$ are complementary (it ignores the $\gamma$ term).
4. **CMIM is most conservative** — useful when early-selected features are very strong and may dominate.
5. **Tuning $(\beta, \gamma)$** directly rather than choosing a named criterion is a valid and sometimes superior approach.

---

## 7. Computational Complexity

| Criterion | Per-Step Cost | Bottleneck |
|-----------|--------------|------------|
| mRMR | $O(p \cdot k)$ MI | Pairwise MI over $S$ |
| JMI | $O(p \cdot k)$ conditional MI | Conditional MI over $S$ |
| CMIM | $O(p \cdot k)$ conditional MI + min | Same as JMI plus min op |
| ICAP | $O(p \cdot k)$ MI + cond MI | Two MI estimates per pair |
| DISR | $O(p \cdot k)$ joint MI + entropy | Joint entropy adds overhead |

All criteria require $O(k)$ MI computations at selection step $k$, giving $O(pk^2/2)$ total for selecting $k$ features from $p$. For large $p$ ($> 10^3$), parallelising the inner loop over candidates $x_k$ is essential.

**Continuous features** require kernel-density estimation or $k$-NN-based MI estimators (e.g., Kraskov et al., 2004), which add $O(n \log n)$ cost per MI computation. For $n = 10^4$ observations and $p = 500$ features, expect $\sim 10$–$60$ seconds per criterion on a modern CPU.

---

## 8. Empirical Comparison: Which Criterion Wins?

Brown et al. (2012) conducted a large-scale empirical evaluation across 14 benchmark datasets:

### Findings

**JMI wins most often** across diverse datasets, particularly when:
- Features have moderate to strong interactions
- Sample size is large enough to estimate conditional MI reliably
- The target is multi-class or continuous

**CMIM is competitive when**:
- One or two dominant features exist in the dataset
- The selected set contains a strong feature that makes most candidates redundant
- Sample size is limited (min-based estimator is more stable)

**mRMR performs worst when**:
- True complementarity exists — the $\gamma = 0$ assumption is violated
- It tends to select redundant features when no single feature is dominant

**DISR is preferred when**:
- Features have different cardinalities or scales
- The normalisation gives better-calibrated scores for thresholding

**ICAP is the safest choice when**:
- You suspect label noise or look-ahead bias (financial data)
- The data has many spurious high-order interactions

### Data Type Guidelines

| Data Type | Recommended Criterion | Rationale |
|-----------|----------------------|-----------|
| Low-dimensional classification | JMI | Best MLE approximation |
| High-dimensional classification ($p \gg n$) | CMIM | Stable min-based estimator |
| Mixed cardinality features | DISR | Normalisation handles scale |
| Financial time series | ICAP | Conservative; handles noise |
| Regression with continuous target | JMI with $k$-NN MI | MI estimator handles continuous $y$ |
| Heavy-tailed distributions | Advanced measures (see Guide 02) | Standard MI underestimates tail dependence |

---

## Code Implementation

```python
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer

def discretise(X, n_bins=10):
    """Discretise continuous features for MI estimation."""
    est = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return est.fit_transform(X).astype(int)

def mi(x, y):
    """Compute mutual information I(x; y) for discrete arrays."""
    return mutual_info_score(x, y)

def cmi(x, y, z):
    """
    Compute conditional mutual information I(x; y | z).

    Uses: I(x; y | z) = I(x; y, z) - I(x; z)
    where I(x; y, z) is the MI between x and the joint variable (y, z).
    """
    # Encode the (y, z) pair as a single joint variable
    n = len(z)
    n_y = len(np.unique(y))
    n_z = len(np.unique(z))
    yz_joint = np.array([y[i] * n_z + z[i] for i in range(n)])

    # I(x; y | z) = I(x; y, z) - I(x; z)
    return mi(x, yz_joint) - mi(x, z)

def mrmr_score(x_k, y, S, X_data):
    """
    mRMR criterion score for candidate feature x_k.

    Parameters
    ----------
    x_k : int
        Index of candidate feature
    y : array
        Target variable (discretised)
    S : list of int
        Indices of already-selected features
    X_data : array
        Full feature matrix (discretised)

    Returns
    -------
    float : mRMR score
    """
    relevance = mi(X_data[:, x_k], y)
    if len(S) == 0:
        return relevance
    redundancy = np.mean([mi(X_data[:, x_k], X_data[:, x_j]) for x_j in S])
    return relevance - redundancy

def jmi_score(x_k, y, S, X_data):
    """JMI criterion: I(x_k; y) - mean_j[I(x_k; x_j)] + mean_j[I(x_k; x_j | y)]"""
    relevance = mi(X_data[:, x_k], y)
    if len(S) == 0:
        return relevance
    redundancy = np.mean([mi(X_data[:, x_k], X_data[:, x_j]) for x_j in S])
    complement = np.mean([cmi(X_data[:, x_k], X_data[:, x_j], y) for x_j in S])
    return relevance - redundancy + complement

def cmim_score(x_k, y, S, X_data):
    """CMIM criterion: min over j of I(x_k; y | x_j)"""
    if len(S) == 0:
        return mi(X_data[:, x_k], y)
    cmis = [cmi(X_data[:, x_k], y, X_data[:, x_j]) for x_j in S]
    return min(cmis)

def icap_score(x_k, y, S, X_data):
    """ICAP criterion: I(x_k; y) - sum_j max(0, I(x_k; x_j) - I(x_k; x_j | y))"""
    relevance = mi(X_data[:, x_k], y)
    if len(S) == 0:
        return relevance
    penalty = 0.0
    for x_j in S:
        redundancy = mi(X_data[:, x_k], X_data[:, x_j])
        complement = cmi(X_data[:, x_k], X_data[:, x_j], y)
        penalty += max(0.0, redundancy - complement)
    return relevance - penalty

def disr_score(x_k, y, S, X_data):
    """DISR: normalised JMI using joint entropy H(x_k, x_j, y)."""
    if len(S) == 0:
        return mi(X_data[:, x_k], y)

    scores = []
    for x_j in S:
        n = len(y)
        # Joint MI I(x_k, x_j; y)
        xk_xj = X_data[:, x_k] * len(np.unique(X_data[:, x_j])) + X_data[:, x_j]
        joint_mi = mi(xk_xj, y)

        # Joint entropy H(x_k, x_j, y)
        xk_xj_y = (X_data[:, x_k] * len(np.unique(X_data[:, x_j])) * len(np.unique(y))
                   + X_data[:, x_j] * len(np.unique(y)) + y)
        vals, counts = np.unique(xk_xj_y, return_counts=True)
        probs = counts / n
        joint_entropy = -np.sum(probs * np.log(probs + 1e-12))

        if joint_entropy > 0:
            scores.append(joint_mi / joint_entropy)
        else:
            scores.append(0.0)

    return np.mean(scores)
```

---

## Common Pitfalls

**Pitfall 1: Applying ITFS to raw continuous features without discretisation.**
Standard MI estimators require discrete inputs. Apply quantile-based binning (`n_bins=10` is a reasonable default). For continuous targets, use $k$-NN MI estimators (e.g., `sklearn.feature_selection.mutual_info_regression`).

**Pitfall 2: Treating ITFS as order-independent.**
ITFS criteria are greedy sequential algorithms. The selected features at step $k$ depend on what was selected at steps $1, \ldots, k-1$. There is no global optimality guarantee.

**Pitfall 3: Ignoring sample size requirements for CMI estimation.**
Conditional MI $I(x_k; y | x_j)$ requires conditioning on $x_j$, effectively partitioning the data. For $x_j$ with 10 bins, each partition has $n/10$ samples. Reliable CMI estimation generally requires $n > 500$ per bin-value.

**Pitfall 4: Using mRMR when complementarity exists.**
mRMR sets $\gamma = 0$ — it cannot discover features that are individually weakly correlated with $y$ but jointly highly predictive. Prefer JMI or CMIM for financial features where interactions dominate.

---

## Connections

**Builds on:**
- Module 01: Shannon entropy, marginal MI, mutual information estimation
- Module 01: Filter methods and the relevance-redundancy tradeoff

**Leads to:**
- Guide 02: Advanced information measures (Rényi entropy, transfer entropy)
- Notebook 01: Unified ITFS implementation and empirical comparison
- Module 03: Wrapper methods that replace MI with model-based evaluation

**Related to:**
- Bayesian information criterion (BIC): both penalise model complexity, but BIC penalises parameters while CLM penalises redundant information
- Canonical correlation analysis: linear analogue of MI-based dependence
- Relief algorithm: distance-based approximation to CMI

---

## Further Reading

- **Brown, G., Pocock, A., Zhao, M-J., Lujan, M. (2012).** "Conditional Likelihood Maximisation: A Unifying Framework for Information Theoretic Feature Selection." *Journal of Machine Learning Research*, 13, 27–66. — The primary reference for this guide. Theorem 1 (p. 34) states the unification formally.

- **Peng, H., Long, F., Ding, C. (2005).** "Feature selection based on mutual information: criteria of max-dependency, max-relevance, and min-redundancy." *IEEE TPAMI*, 27(8), 1226–1238.

- **Fleuret, F. (2004).** "Fast binary feature selection with conditional mutual information." *JMLR*, 5, 1531–1555.

- **Meyer, P., Schretter, C., Bontempi, G. (2008).** "Information-theoretic feature selection in microarray data using variable complementarity." *IEEE J. Sel. Top. Signal Process.*, 2(3), 261–274.

- **Kraskov, A., Stögbauer, H., Grassberger, P. (2004).** "Estimating mutual information." *Physical Review E*, 69(6). — $k$-NN MI estimator for continuous variables.

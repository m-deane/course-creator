# Ensemble Feature Selection

## In Brief

Ensemble feature selection combines multiple selection algorithms — or multiple runs of the same algorithm — and aggregates their outputs to produce a final feature ranking or subset. The approach trades additional computation for substantially improved stability, reduced variance, and better generalization than any individual method.

## Key Insight

A single feature selector applied once is like polling a single expert: fast but unreliable. Ensemble selection is the wisdom of crowds applied to feature ranking. Even when every individual selector is noisy, their aggregate consistently outranks the best individual method on stability metrics — and stability strongly predicts out-of-sample relevance.

---

## Why Ensemble Selection Outperforms Single Methods

### The Instability Problem

Feature selection methods are notoriously unstable: small perturbations to the dataset — removing 5% of samples, adding slight noise to one feature — can produce dramatically different selected subsets. This instability has three damaging consequences:

1. **Irreproducibility:** Independent researchers studying the same data reach different conclusions.
2. **Overfitting to noise:** A selector that exploits noise in one dataset generalizes poorly to new data.
3. **Loss of confidence:** Practitioners cannot trust a selection they know flips under minor perturbations.

The root cause is the **selection discontinuity**: a threshold or ranking cutoff transforms a continuous score into a binary include/exclude decision. Small score changes near the cutoff produce large changes in the selected set.

### The Bias-Variance Trade-Off in Feature Selection

The expected error of a feature selection procedure decomposes analogously to prediction:

$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

- **Bias** arises from the selector's structural assumptions (e.g., LASSO assumes linear relationships, univariate filters ignore interactions).
- **Variance** arises from sensitivity to the specific training sample.

Single methods suffer from high variance when:
- The sample size is small (n < 500)
- Features are highly correlated (many near-equivalent feature subsets exist)
- Signals are weak relative to noise

Ensemble selection directly reduces variance. If $M$ independent selectors each produce a ranking with variance $\sigma^2$, their average has variance $\sigma^2 / M$. In practice, selectors are correlated, so the reduction is sub-linear, but still substantial.

### Empirical Evidence

Across benchmark studies (Abeel et al. 2010; Saeys et al. 2008; Haury et al. 2011), ensemble selection consistently:
- Reduces selection variance by 40–70% versus individual methods
- Improves downstream model performance by 2–8% in small-sample settings
- Identifies stable biomarkers that individual methods miss due to sample specificity

---

## Stability Metrics

Before discussing consensus methods, we need a rigorous way to measure and compare stability.

### Kuncheva's Stability Index

Kuncheva's index $\kappa$ is the most widely used pairwise stability measure. Given two selected subsets $S_1$ and $S_2$, each of size $k$, from a feature space of size $p$:

$$\kappa(S_1, S_2) = \frac{|S_1 \cap S_2| / k - k/p}{1 - k/p}$$

- Range: $[-1, 1]$. Value of 1 means perfect agreement; 0 means chance agreement; negative means worse than chance.
- Corrects for the expected overlap under random selection, making it comparable across different $k$ and $p$.

```python
def kuncheva_index(s1: set, s2: set, p: int) -> float:
    """
    Kuncheva's stability index for two feature subsets.

    Parameters
    ----------
    s1, s2 : set
        Two selected feature subsets (same expected size k).
    p : int
        Total number of features (size of the full feature space).

    Returns
    -------
    float in [-1, 1]. Higher is more stable.
    """
    k = len(s1)
    if k != len(s2):
        raise ValueError("Both subsets must have the same size k")
    if k == 0 or k == p:
        return 1.0  # trivially stable (nothing or everything selected)

    intersection = len(s1 & s2)
    chance_overlap = k / p
    # Normalise by maximum possible overlap minus chance
    kappa = (intersection / k - chance_overlap) / (1 - chance_overlap)
    return kappa


def kuncheva_ensemble_stability(subsets: list[set], p: int) -> float:
    """
    Average pairwise Kuncheva index across all pairs in an ensemble.

    Parameters
    ----------
    subsets : list of sets
        Selected feature subsets from M selector runs.
    p : int
        Total number of features.

    Returns
    -------
    float
        Mean pairwise stability.
    """
    from itertools import combinations
    pairs = list(combinations(range(len(subsets)), 2))
    if not pairs:
        return 1.0
    scores = [kuncheva_index(subsets[i], subsets[j], p)
              for i, j in pairs]
    return sum(scores) / len(scores)
```

### Jaccard Similarity

The Jaccard index measures the fraction of the union that is shared — a simpler but less bias-corrected measure than Kuncheva's:

$$J(S_1, S_2) = \frac{|S_1 \cap S_2|}{|S_1 \cup S_2|}$$

Jaccard is not corrected for chance overlap. For small $k$ relative to $p$, even random subsets have low Jaccard similarity, making it harder to interpret in absolute terms. Use Kuncheva's index when comparing across experiments with different $k$ or $p$.

```python
def jaccard_similarity(s1: set, s2: set) -> float:
    """Jaccard similarity between two feature subsets."""
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)
```

### Spearman's Footrule Distance

When working with full rankings (not just binary include/exclude), Spearman's footrule measures the total displacement between two ranked lists:

$$F(\sigma, \tau) = \sum_{i=1}^{p} |\sigma(i) - \tau(i)|$$

where $\sigma(i)$ is the rank of feature $i$ under ranking $\sigma$. The normalised version divides by the maximum possible distance $\lfloor p^2/2 \rfloor$ (for odd $p$) to give a value in $[0, 1]$.

```python
import numpy as np

def spearman_footrule(rank1: np.ndarray, rank2: np.ndarray) -> float:
    """
    Normalised Spearman footrule distance between two feature rankings.

    Parameters
    ----------
    rank1, rank2 : array of shape (p,)
        Rank of each feature (1 = most important).

    Returns
    -------
    float in [0, 1]. Lower means more similar rankings.
    """
    p = len(rank1)
    raw_distance = np.sum(np.abs(rank1 - rank2))
    # Maximum possible footrule distance
    max_distance = p ** 2 // 2
    return raw_distance / max_distance
```

**Choosing the right metric:**

| Situation | Recommended metric |
|---|---|
| Binary selected/not-selected subsets, comparing two runs | Kuncheva's index |
| Binary subsets, need intuitive interpretation | Jaccard |
| Full feature rankings (all methods produce ordered lists) | Spearman's footrule |
| Ranking stability across bootstrap samples | Rank variance or footrule |

---

## Consensus Methods

Once you have $M$ feature rankings or subsets, you must aggregate them into a single output.

### Union

Take the union of all selected subsets. Maximises recall — any feature selected by at least one method appears in the output. Appropriate when false negatives (missing relevant features) are more costly than false positives.

$$S_{\text{union}} = S_1 \cup S_2 \cup \cdots \cup S_M$$

**Risk:** If individual methods over-select, the union inflates the feature set. For a 50-feature dataset with 5 methods each selecting 20, the union could include all 50.

### Intersection

Take the intersection of all selected subsets. Only features agreed upon by all methods survive. Maximises precision — only features with universal support are selected. Appropriate when false positives (irrelevant features) are more costly.

$$S_{\text{intersection}} = S_1 \cap S_2 \cap \cdots \cap S_M$$

**Risk:** The intersection shrinks rapidly with $M$. With 5 methods, even a highly relevant feature might be dropped if one method misses it.

### Majority Vote

A feature enters the ensemble subset if it is selected by at least half the methods (or a configurable threshold $t$):

$$S_{\text{vote}} = \{i : \text{count}(i \in S_m) \geq t \cdot M\}$$

The majority vote threshold $t$ interpolates between intersection ($t=1$) and union ($t=1/M$). A natural default is $t = 0.5$.

```python
from collections import Counter

def majority_vote(subsets: list[list], threshold: float = 0.5) -> list:
    """
    Consensus selection by majority vote.

    Parameters
    ----------
    subsets : list of lists
        Each inner list contains selected feature names/indices for one run.
    threshold : float
        Fraction of methods that must select a feature for inclusion.
        0.5 = majority vote; 1.0 = intersection.

    Returns
    -------
    list
        Features selected by at least threshold * M methods.
    """
    M = len(subsets)
    counts = Counter(feat for subset in subsets for feat in subset)
    min_votes = threshold * M
    return [feat for feat, cnt in counts.items() if cnt >= min_votes]
```

### Weighted Vote

When methods have unequal predictive reliability (e.g., cross-validated accuracy on a holdout), weight their votes accordingly:

$$\text{score}(i) = \sum_{m=1}^{M} w_m \cdot \mathbf{1}[i \in S_m]$$

where $w_m \geq 0$ and $\sum_m w_m = 1$. Features are selected if their weighted score exceeds a threshold.

```python
def weighted_vote(subsets: list[list], weights: list[float],
                  threshold: float = 0.5) -> list:
    """
    Weighted consensus selection.

    Parameters
    ----------
    subsets : list of lists
        Selected features per method.
    weights : list of floats
        Weight for each method (need not sum to 1 — normalised internally).
    threshold : float
        Weighted-vote threshold for selection.

    Returns
    -------
    list
        Features whose weighted selection score >= threshold.
    """
    weights = np.array(weights, dtype=float)
    weights /= weights.sum()  # normalise

    scores: dict = {}
    for subset, w in zip(subsets, weights):
        for feat in subset:
            scores[feat] = scores.get(feat, 0.0) + w

    return [feat for feat, score in scores.items() if score >= threshold]
```

---

## Multi-Criteria Aggregation

For full rankings (rather than binary subsets), dedicated rank-aggregation algorithms are more powerful than simple vote counting.

### Borda Count

Each feature receives Borda points equal to the number of features ranked below it by each method. A feature ranked 1st out of $p$ features receives $p-1$ points; ranked $p$th receives 0 points. Sum across all methods to get a final aggregate score.

$$\text{Borda}(i) = \sum_{m=1}^{M} (p - \text{rank}_m(i))$$

Borda count is simple, robust, and produces a complete ranking (no ties by default). It is the workhorse of rank aggregation for feature selection.

```python
def borda_count(rankings: list[list]) -> list:
    """
    Borda count rank aggregation.

    Parameters
    ----------
    rankings : list of lists
        Each inner list contains all features ordered from most to least
        important (index 0 = best). All lists must contain the same features.

    Returns
    -------
    list
        Features sorted from highest to lowest aggregate Borda score.
    """
    p = len(rankings[0])
    scores: dict = {feat: 0 for feat in rankings[0]}

    for ranking in rankings:
        for rank_idx, feat in enumerate(ranking):
            # Higher rank (lower index) -> more points
            scores[feat] += (p - 1 - rank_idx)

    return sorted(scores, key=lambda f: scores[f], reverse=True)
```

### Rank Aggregation via Mean Reciprocal Rank

Mean Reciprocal Rank (MRR) rewards features that appear near the top of any individual ranking:

$$\text{MRR}(i) = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{\text{rank}_m(i)}$$

MRR amplifies agreement at the top — a feature ranked 1st by one method but 50th by others still scores well if the 1st-place ranking is strong. Use MRR when you care more about the top-$k$ features than the full ordering.

```python
def mean_reciprocal_rank(rankings: list[list]) -> list:
    """
    Aggregate feature rankings using Mean Reciprocal Rank.

    Returns features sorted by descending MRR (best first).
    """
    # Build rank position for each feature in each ranking
    rank_positions: dict = {}
    for ranking in rankings:
        for rank_idx, feat in enumerate(ranking):
            if feat not in rank_positions:
                rank_positions[feat] = []
            rank_positions[feat].append(rank_idx + 1)  # 1-indexed

    mrr_scores = {feat: np.mean([1 / r for r in positions])
                  for feat, positions in rank_positions.items()}
    return sorted(mrr_scores, key=lambda f: mrr_scores[f], reverse=True)
```

### Kemeny Optimal Ranking

The Kemeny optimal ranking minimises the total number of pairwise disagreements with all input rankings — it is the median ranking in the space of all permutations. It is NP-hard to compute exactly for large $p$, but effective approximations exist.

**Pairwise comparison matrix:** Let $C_{ij}$ = number of methods that rank feature $i$ above feature $j$. Feature $i$ beats $j$ if $C_{ij} > M/2$. The Kemeny approximation ranks features by their total wins:

```python
def kemeny_approximation(rankings: list[list]) -> list:
    """
    Approximate Kemeny-optimal rank aggregation via pairwise win counts.

    For each pair of features (i, j), count how many methods rank i above j.
    Feature i's aggregate score = total wins across all pairwise comparisons.
    This is equivalent to the Copeland method.

    Parameters
    ----------
    rankings : list of lists
        Each inner list is a full ranking (index 0 = best feature).

    Returns
    -------
    list
        Features sorted from most wins to fewest.
    """
    features = rankings[0]
    p = len(features)
    feat_to_idx = {f: i for i, f in enumerate(features)}

    # Build pairwise win matrix
    wins = np.zeros((p, p), dtype=int)
    for ranking in rankings:
        for pos_i, feat_i in enumerate(ranking):
            for pos_j, feat_j in enumerate(ranking):
                if pos_i < pos_j:  # feat_i ranked above feat_j in this ranking
                    i, j = feat_to_idx[feat_i], feat_to_idx[feat_j]
                    wins[i, j] += 1

    # Copeland score = wins - losses
    copeland_scores = wins.sum(axis=1) - wins.T.sum(axis=1)
    sorted_indices = np.argsort(-copeland_scores)
    return [features[i] for i in sorted_indices]
```

**Comparison of aggregation methods:**

| Method | Time | Handles ties | Top-k emphasis | Best for |
|---|---|---|---|---|
| Borda count | O(M·p) | Yes | No | General purpose |
| MRR | O(M·p) | Yes | Yes | When top features matter most |
| Majority vote | O(M·k) | N/A | No | Binary subsets |
| Kemeny (approx) | O(M·p²) | Yes | No | Minimising disagreements |

---

## Bootstrap Aggregation of Selectors: BagFS

Bootstrap AGgregation of Feature Selectors (BagFS) applies the bagging principle to feature selection:

1. Draw $B$ bootstrap samples from the training data (sampling with replacement).
2. Apply the base selector to each bootstrap sample, obtaining $B$ ranked lists or subsets.
3. Aggregate via Borda count, majority vote, or weighted consensus.

```python
from sklearn.utils import resample

def bagfs(X: np.ndarray, y: np.ndarray,
          selector_fn,
          n_bootstrap: int = 50,
          subsample_ratio: float = 0.8,
          aggregation: str = 'borda',
          random_state: int = 42) -> np.ndarray:
    """
    Bootstrap Aggregation of Feature Selectors (BagFS).

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    selector_fn : callable
        Function that takes (X, y) and returns a list of feature indices
        ordered from most to least important.
    n_bootstrap : int
        Number of bootstrap samples.
    subsample_ratio : float
        Fraction of samples per bootstrap draw (< 1.0 = subsampling without
        replacement, increasing diversity).
    aggregation : str
        'borda' | 'vote' | 'mrr'

    Returns
    -------
    np.ndarray
        Feature indices sorted from most to least important by aggregate score.
    """
    rng = np.random.RandomState(random_state)
    n_samples = X.shape[0]
    n_sub = int(n_samples * subsample_ratio)
    all_rankings = []

    for b in range(n_bootstrap):
        seed = rng.randint(0, 2**31)
        # Draw bootstrap or subsample
        if subsample_ratio < 1.0:
            idx = rng.choice(n_samples, size=n_sub, replace=False)
        else:
            idx = resample(np.arange(n_samples), random_state=seed)
        X_b, y_b = X[idx], y[idx]

        ranking = selector_fn(X_b, y_b)
        all_rankings.append(list(ranking))

    # Aggregate
    if aggregation == 'borda':
        return np.array(borda_count(all_rankings))
    elif aggregation == 'mrr':
        return np.array(mean_reciprocal_rank(all_rankings))
    elif aggregation == 'vote':
        # Return features in order of vote frequency (all features as binary)
        # Treat each ranking as a subset of the top half
        k = X.shape[1] // 2
        subsets = [r[:k] for r in all_rankings]
        voted = majority_vote(subsets, threshold=0.3)
        return np.array(voted)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")
```

### BagFS Variants

**BagFS-LASSO:** Bootstrap LASSO paths. On each bootstrap sample, fit LASSO over a grid of $\lambda$ values and record which features are selected at each $\lambda$. Aggregate selection frequency gives the **stability selection** probability (Meinshausen & Bühlmann, 2010):

$$\hat{\pi}_j = \frac{1}{B} \sum_{b=1}^{B} \mathbf{1}[j \in \hat{S}^b(\lambda)]$$

Features with $\hat{\pi}_j > 0.9$ are stable selections with controlled family-wise error.

**BagFS-RF:** Bootstrap random forests. Each bootstrap forest produces feature importances; aggregate by averaging importance scores rather than ranks.

**Randomised BagFS:** Add feature subsampling in addition to observation subsampling to further diversify the ensemble.

---

## When Ensemble Selection Helps Most

Ensemble selection provides the largest gains when:

### 1. Small Samples

When $n < 10p$ (fewer than 10 observations per feature), any single selector is heavily influenced by sampling noise. Bootstrap aggregation directly addresses this by averaging over the sampling distribution.

**Heuristic:** If your dataset has $n < 500$ and $p > 50$, use BagFS with $B \geq 100$ bootstraps.

### 2. Correlated Features

When features are highly correlated, many near-equivalent subsets have similar predictive power. A single selector arbitrarily chooses one representative from each correlated group; the choice varies between runs. Ensemble selection smooths over this arbitrariness by distributing votes across the correlated group.

**Signal:** If your feature correlation matrix has clusters with $|r| > 0.8$, individual selectors are unreliable. Ensemble selection with diverse methods (filter + embedded + wrapper) produces more robust results.

### 3. Noisy Features

When a substantial fraction of features are pure noise (e.g., irrelevant genomic SNPs), single selectors sometimes over-select noise features that happen to correlate with the target in a particular sample. Ensemble selection requires a noise feature to correlate with the target across many bootstrap samples — a much stricter criterion.

### 4. Heterogeneous Ensembles

Diversity among ensemble members improves performance. Combine methods with different:
- **Assumptions:** Univariate filter (no interactions) + tree importance (captures interactions) + LASSO (linear structure)
- **Search strategies:** Greedy forward + evolutionary + shrinkage
- **Stability profiles:** Some methods (LASSO) are less stable than others (RF importance); ensemble smooths this

---

## Practical Implementation Pattern

```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LassoCV

def build_ensemble_selector(X: np.ndarray, y: np.ndarray,
                             feature_names: list[str]) -> pd.DataFrame:
    """
    Build a 3-method ensemble feature selector and return aggregate rankings.

    Methods: Mutual Information, LASSO, Random Forest importance.
    Aggregation: Borda count.
    """
    from sklearn.ensemble import RandomForestClassifier

    n_features = X.shape[1]
    rankings = {}

    # Method 1: Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    rankings['MI'] = np.argsort(-mi_scores).tolist()  # highest MI = rank 0

    # Method 2: LASSO absolute coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lasso = LassoCV(cv=5, random_state=42, max_iter=5000)
    lasso.fit(X_scaled, y)
    lasso_scores = np.abs(lasso.coef_)
    rankings['LASSO'] = np.argsort(-lasso_scores).tolist()

    # Method 3: Random Forest importance
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X, y)
    rf_scores = rf.feature_importances_
    rankings['RF'] = np.argsort(-rf_scores).tolist()

    # Aggregate via Borda count
    all_rankings = list(rankings.values())
    borda_order = borda_count(all_rankings)

    # Build summary DataFrame
    df = pd.DataFrame({
        'feature': feature_names,
        'MI_rank': [rankings['MI'].index(i) + 1 for i in range(n_features)],
        'LASSO_rank': [rankings['LASSO'].index(i) + 1 for i in range(n_features)],
        'RF_rank': [rankings['RF'].index(i) + 1 for i in range(n_features)],
        'ensemble_rank': [borda_order.index(i) + 1 for i in range(n_features)],
    })
    return df.sort_values('ensemble_rank').reset_index(drop=True)
```

---

## Common Pitfalls

- **Homogeneous ensembles:** Using 5 variations of the same method (e.g., Random Forest with different seeds) provides less benefit than combining fundamentally different selectors. Diversity is the engine of ensemble improvement.
- **Ignoring class imbalance:** If your dataset is imbalanced, bootstrap samples can produce extremely imbalanced bootstrap datasets. Use stratified resampling.
- **Fixed k across methods:** If methods select different numbers of features, majority vote at threshold 0.5 is biased toward features from methods that select more. Either standardise $k$ across methods or use rank-based aggregation.
- **Selecting by stability alone:** A feature selected consistently by all methods might still be irrelevant if all methods share the same bias. Validate stability with domain knowledge or held-out evaluation.

---

## Connections

- **Builds on:** Module 01 (filter methods — candidate selectors for the ensemble), Module 02 (information-theoretic criteria — MI/JMI as ensemble members), Module 03 (wrapper methods — Boruta and SFS as ensemble members), Module 04 (embedded methods — Lasso and RF importance as ensemble members), Module 05 (genetic algorithms — GA as a high-quality ensemble member), Module 06 (evolutionary and swarm methods — NSGA-II Pareto front aggregation)
- **Leads to:** Module 11 (production feature selection pipelines), hybrid cascade methods (Guide 02)
- **Related to:** Ensemble learning in prediction (bagging, boosting); random subspace methods; stability selection (LASSO + bootstrap)

---

## Cross-Module Connections

**The ensemble assembles all prior methods.** Each preceding module contributes a distinct selector type to the ensemble:

| Module | Selector type | Role in ensemble |
|---|---|---|
| Module 1 | Statistical filters (MI, HSIC, distance correlation) | Fast, cheap first pass; diverse assumptions |
| Module 2 | Information-theoretic criteria (mRMR, JMI, CMIM) | Redundancy-aware filter members |
| Module 3 | Wrappers (Boruta, SFFS) | Model-validated, interaction-aware |
| Module 4 | Embedded (Lasso, RF importance, SHAP) | Regularisation-based; fast to compute |
| Module 5 | Binary GA | Stochastic global search; escapes local optima |
| Module 6 | NSGA-II | Returns Pareto front; enables multi-criteria aggregation |

**Why heterogeneity matters:** Methods with different structural assumptions are imperfectly correlated — their errors do not coincide on the same features. Union and Borda count aggregation benefit most when ensemble members disagree on a different subset of irrelevant features.

**Pareto aggregation (Module 6 link):** When ensemble members return Pareto fronts rather than single subsets (e.g., from NSGA-II in Module 6), aggregate by pooling all Pareto solutions and re-ranking by non-domination across the combined pool. This yields a meta-Pareto front that dominates any individual method's front.

---

## Further Reading

- Abeel, T. et al. (2010). **Robust biomarker identification for cancer diagnosis with ensemble feature selection methods.** *Bioinformatics*, 26(3). Foundational empirical study.
- Saeys, Y., Abeel, T., & Van de Peer, Y. (2008). **Robust feature selection using ensemble feature selection techniques.** In *ECML PKDD 2008*. Survey with comparisons.
- Meinshausen, N. & Bühlmann, P. (2010). **Stability selection.** *Journal of the Royal Statistical Society: Series B*, 72(4). Formal theory for BagFS-LASSO.
- Kuncheva, L.I. (2007). **A stability index for feature selection.** In *IASTED AIA 2007*. Original paper proposing Kuncheva's index.

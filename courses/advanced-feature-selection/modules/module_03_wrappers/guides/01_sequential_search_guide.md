# Sequential Search Methods for Feature Selection

## In Brief

Sequential search methods build or reduce feature subsets one feature at a time, guided by a model's cross-validated performance. Sequential Forward Selection (SFS) starts empty and adds the best feature at each step; Sequential Backward Selection (SBS) starts full and removes the worst. The floating variants (SFFS/SBFS) add a "backtracking" phase after each primary step to escape local optima.

> **Key Insight:** Sequential methods find the optimal subset for a fixed number of features when combined with the right stopping criterion, but they make irrevocable decisions in the basic form — once a feature is added or removed, it stays. Floating variants break this constraint, dramatically improving solution quality at modest additional cost.

## Formal Definitions

### Sequential Forward Selection (SFS)

Given a full feature set $\mathcal{F} = \{f_1, f_2, \ldots, f_p\}$, a model $\mathcal{M}$, and a performance criterion $J(\cdot)$:

$$S_0 = \emptyset$$

$$S_{k+1} = S_k \cup \left\{ \arg\max_{f_j \in \mathcal{F} \setminus S_k} J(S_k \cup \{f_j\}) \right\}$$

Terminate when $|S_k| = k^*$ (target size) or a stopping criterion fires.

### Sequential Backward Selection (SBS)

$$S_0 = \mathcal{F}$$

$$S_{k-1} = S_k \setminus \left\{ \arg\max_{f_j \in S_k} J(S_k \setminus \{f_j\}) \right\}$$

Equivalently: at each step, remove the feature whose elimination causes the least performance loss.

### Bidirectional Search

Bidirectional search runs SFS and SBS simultaneously, merging the two expanding/shrinking frontiers:

$$S^{fwd}_{k+1} = S^{fwd}_k \cup \{ \text{best forward addition} \}$$
$$S^{bwd}_{k-1} = S^{bwd}_k \setminus \{ \text{best backward removal} \}$$

The two searches meet when $S^{fwd} \cap S^{bwd} \neq \emptyset$. Practical implementations stop when the forward candidate is not already in the backward set.

### Sequential Floating Forward Selection (SFFS)

The floating mechanism adds a conditional backtracking phase:

**Forward phase:** Add the best feature $f^+$:
$$S_{k+1} = S_k \cup \{f^+\}$$

**Backward phase (floating):** While removing the worst feature $f^-$ from $S_{k+1}$ improves $J$:
$$\text{if } J(S_{k+1} \setminus \{f^-\}) > J_{\text{best}}(k) \text{ then } S_{k+1} \leftarrow S_{k+1} \setminus \{f^-\}$$

Here $J_{\text{best}}(k)$ is the best score seen so far for any subset of size $k$. This prevents the backtracking from undoing the forward step indefinitely.

**Key property:** SFFS can select and deselect the same feature in subsequent iterations, escaping the nesting trap of basic SFS.

### Sequential Floating Backward Selection (SBFS)

The mirror of SFFS. After each backward removal, a forward phase tentatively adds a feature back if it improves the criterion:

$$\text{if } J(S_{k-1} \cup \{f^+\}) > J_{\text{best}}(k) \text{ then } S_{k-1} \leftarrow S_{k-1} \cup \{f^+\}$$

## Full Algorithms

### SFS Algorithm

```
INPUT: feature set F, target size k*, model M, scorer J
OUTPUT: selected subset S*

S ← ∅
best_scores ← {}

FOR step = 1 TO k*:
    candidates ← F \ S
    best_score ← -∞
    best_feature ← None

    FOR EACH f IN candidates:
        score ← cross_val_score(M, X[:, S ∪ {f}], y)
        IF score > best_score:
            best_score ← score
            best_feature ← f

    S ← S ∪ {best_feature}
    best_scores[|S|] ← best_score

RETURN S, best_scores
```

### SFFS Algorithm

```
INPUT: feature set F, target size k*, model M, scorer J
OUTPUT: selected subset S*

S ← ∅
best_scores ← {}      # maps subset_size → best_J
converged ← False

WHILE NOT converged:
    # Forward phase
    candidates ← F \ S
    IF candidates is empty: BREAK
    f+ ← argmax_{f ∈ candidates} J(S ∪ {f})
    S ← S ∪ {f+}
    best_scores[|S|] ← max(best_scores.get(|S|, -∞), J(S))

    # Backward phase (floating)
    WHILE |S| > 1:
        f- ← argmax_{f ∈ S} J(S \ {f})  # feature whose removal hurts least
        score_without ← J(S \ {f-})
        IF score_without > best_scores.get(|S|-1, -∞):
            S ← S \ {f-}
            best_scores[|S|] ← score_without
        ELSE:
            BREAK  # No beneficial removal found

    IF |S| >= k* OR stopping_criterion(best_scores):
        converged ← True

RETURN S, best_scores
```

## Early Stopping Heuristics

Running sequential search to the target size $k^*$ can be wasteful. Three stopping criteria avoid unnecessary model evaluations:

### 1. Patience / No-Improvement

Track a rolling window of $\tau$ steps. If the best score does not improve by $\delta_{\min}$ for $\tau$ consecutive steps, halt:

$$\text{stop if } \max_{i=t-\tau}^{t} J_{\text{best}}(i) - J_{\text{best}}(t - \tau) < \delta_{\min}$$

Typical values: $\tau = 5$, $\delta_{\min} = 0.001$ (0.1% improvement threshold).

### 2. Diminishing Returns

Track the incremental gain per additional feature. Halt when the marginal gain falls below a fraction $\alpha$ of the maximum gain seen so far:

$$\Delta J_k = J_{\text{best}}(k) - J_{\text{best}}(k-1)$$

$$\text{stop if } \Delta J_k < \alpha \cdot \max_{i < k} \Delta J_i \quad (\text{e.g., } \alpha = 0.1)$$

### 3. Score Plateau Detection

Fit a smoothed curve to the score-vs-features plot. Declare a plateau when the slope of the fitted curve drops below a threshold $\epsilon$:

$$\text{stop if } \frac{d\hat{J}}{dk}\bigg|_{k=t} < \epsilon$$

This is more robust to noise than raw incremental comparisons. A practical implementation uses a rolling linear regression over the last $w$ steps.

## Computational Cost Analysis

### Basic Operations

For a dataset with $p$ features, $n$ samples, and a model with training cost $T_{\mathcal{M}}$:

| Method | Feature evaluations | Total cost |
|--------|--------------------:|-----------|
| SFS to $k^*$ | $\sum_{i=0}^{k^*-1}(p - i) = k^*p - \binom{k^*}{2}$ | $O(k^* p \cdot T_{\mathcal{M}})$ |
| SBS to $k^*$ | $\sum_{i=0}^{p-k^*-1}(p - i)$ | $O((p-k^*)p \cdot T_{\mathcal{M}})$ |
| SFFS | $O(k^* p \cdot T_{\mathcal{M}})$ with constant factor $\approx 2\text{–}3\times$ SFS | |

For cross-validated evaluation with $v$ folds, multiply by $v$:

$$\text{Total cost (SFS)} = k^* \cdot p \cdot v \cdot T_{\mathcal{M}}$$

**Example:** $p = 100$ features, $k^* = 20$, $v = 5$ folds, $T_{\mathcal{M}} = 0.1$ seconds:
$$20 \times 100 \times 5 \times 0.1 = 1{,}000 \text{ seconds} \approx 17 \text{ minutes}$$

This grows quadratically in $p$ for fixed $k^*/p$ ratio, making wrapper methods prohibitively expensive for $p > 500$ without the scalability techniques from Guide 03.

### Where Time Is Spent

```
Total wall time breakdown (typical):
├── Model training         60-80%
├── Feature subset indexing  5-10%
├── CV fold splits          5-10%
└── Score bookkeeping         <1%
```

The dominant cost is always the model training. Reducing $T_{\mathcal{M}}$ has the highest leverage.

## Warm-Starting Strategies

### Filter Pre-Screening

Use a cheap filter method to rank all $p$ features, then restrict sequential search to the top $p'$ candidates ($p' \ll p$). This reduces the search space from $p$ to $p'$ features at each step:

$$\text{Cost reduction} = \frac{p'}{p} \times 100\%$$

Typical: use $p' = 2k^*$ to $5k^*$, retaining 2–5 times the target number of features.

### Evaluation Caching

Cache the model score for each feature subset index tuple. Before evaluating a candidate, check the cache:

```python
cache = {}

def cached_score(feature_mask, X, y, model, cv):
    key = tuple(sorted(feature_mask))
    if key not in cache:
        cache[key] = cross_val_score(model, X[:, key], y, cv=cv).mean()
    return cache[key]
```

Cache hits are essentially free. In SFFS, the backward phase frequently re-evaluates subsets visited during forward steps, making caching especially effective.

### Transfer Warm-Start

If running sequential search on a new dataset drawn from the same distribution, initialise $S_0$ with the feature set found on the previous dataset. This skips the first $k_0$ forward steps, saving $\sum_{i=0}^{k_0-1}(p-i)$ evaluations.

### Progressive Subsampling

Start with a small subsample ($n' \approx n / 10$) for the early steps where many features are evaluated, then gradually increase $n'$ as the candidate pool shrinks:

$$n'(k) = \min(n, n_{\min} + k \cdot \delta_n)$$

Scores on subsamples are noisy but sufficient for identifying clearly bad features early.

## Code Implementation

### Complete SFS / SBS / SFFS

```python
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
from typing import Callable, Dict, List, Optional, Tuple


class SequentialFeatureSelector:
    """
    Sequential feature selection with SFS, SBS, and SFFS variants.

    Parameters
    ----------
    estimator : sklearn estimator
        The model used to evaluate feature subsets.
    n_features_to_select : int or float
        Target feature count (int) or fraction of total features (float).
    direction : str
        One of 'forward', 'backward', 'floating_forward', 'floating_backward'.
    scoring : str or callable
        Scoring metric for cross_val_score.
    cv : int
        Cross-validation folds.
    patience : int
        Stop after this many steps without improvement exceeding tol.
    tol : float
        Minimum improvement per step to reset patience counter.
    n_jobs : int
        Parallel jobs for cross-validation (-1 = all CPUs).
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: int = 10,
        direction: str = "floating_forward",
        scoring: str = "accuracy",
        cv: int = 5,
        patience: int = 5,
        tol: float = 1e-4,
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.patience = patience
        self.tol = tol
        self.n_jobs = n_jobs

        # Internal state
        self._cache: Dict[Tuple, float] = {}
        self.selected_features_: Optional[np.ndarray] = None
        self.score_history_: List[Tuple[int, float]] = []

    def _score(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        """Cross-validate the estimator on the selected feature subset.

        Uses a cache keyed on sorted feature indices to avoid redundant fits.
        """
        key = tuple(np.where(mask)[0])
        if not key:
            return -np.inf
        if key not in self._cache:
            estimator = clone(self.estimator)
            scores = cross_val_score(
                estimator,
                X[:, key],
                y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            self._cache[key] = float(scores.mean())
        return self._cache[key]

    def _forward_step(
        self, X: np.ndarray, y: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Add the single best feature to the current mask."""
        p = X.shape[1]
        best_score = -np.inf
        best_mask = mask.copy()

        for j in range(p):
            if mask[j]:
                continue  # Already selected
            candidate = mask.copy()
            candidate[j] = True
            score = self._score(X, y, candidate)
            if score > best_score:
                best_score = score
                best_mask = candidate

        return best_mask, best_score

    def _backward_step(
        self, X: np.ndarray, y: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Remove the single feature whose deletion causes least loss."""
        p = X.shape[1]
        best_score = -np.inf
        best_mask = mask.copy()

        for j in range(p):
            if not mask[j]:
                continue  # Not currently selected
            candidate = mask.copy()
            candidate[j] = False
            score = self._score(X, y, candidate)
            if score > best_score:
                best_score = score
                best_mask = candidate

        return best_mask, best_score

    def _fit_forward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run SFS, optionally with floating backward phase."""
        p = X.shape[1]
        k_target = self.n_features_to_select
        mask = np.zeros(p, dtype=bool)

        # Best score seen at each subset size: size -> score
        best_at_size: Dict[int, float] = {}
        no_improve_count = 0
        prev_best = -np.inf

        while mask.sum() < k_target:
            # Forward step
            mask, score = self._forward_step(X, y, mask)
            k = int(mask.sum())
            best_at_size[k] = max(best_at_size.get(k, -np.inf), score)
            self.score_history_.append((k, score))

            # Floating backward phase (SFFS only)
            if self.direction == "floating_forward":
                improved = True
                while improved and mask.sum() > 1:
                    candidate_mask, candidate_score = self._backward_step(X, y, mask)
                    k_smaller = int(candidate_mask.sum())
                    threshold = best_at_size.get(k_smaller, -np.inf)
                    if candidate_score > threshold:
                        mask = candidate_mask
                        best_at_size[k_smaller] = candidate_score
                        self.score_history_.append((k_smaller, candidate_score))
                    else:
                        improved = False

            # Early stopping: patience
            if score > prev_best + self.tol:
                prev_best = score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    break

        self.selected_features_ = np.where(mask)[0]

    def _fit_backward(self, X: np.ndarray, y: np.ndarray) -> None:
        """Run SBS, optionally with floating forward phase."""
        p = X.shape[1]
        k_target = self.n_features_to_select
        mask = np.ones(p, dtype=bool)

        best_at_size: Dict[int, float] = {}
        no_improve_count = 0
        prev_best = -np.inf

        while mask.sum() > k_target:
            mask, score = self._backward_step(X, y, mask)
            k = int(mask.sum())
            best_at_size[k] = max(best_at_size.get(k, -np.inf), score)
            self.score_history_.append((k, score))

            # Floating forward phase (SBFS only)
            if self.direction == "floating_backward":
                improved = True
                while improved and mask.sum() < p:
                    candidate_mask, candidate_score = self._forward_step(X, y, mask)
                    k_larger = int(candidate_mask.sum())
                    threshold = best_at_size.get(k_larger, -np.inf)
                    if candidate_score > threshold:
                        mask = candidate_mask
                        best_at_size[k_larger] = candidate_score
                        self.score_history_.append((k_larger, candidate_score))
                    else:
                        improved = False

            if score > prev_best + self.tol:
                prev_best = score
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    break

        self.selected_features_ = np.where(mask)[0]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SequentialFeatureSelector":
        """Fit the selector on training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        self._cache = {}
        self.score_history_ = []

        if self.direction in ("forward", "floating_forward"):
            self._fit_forward(X, y)
        elif self.direction in ("backward", "floating_backward"):
            self._fit_backward(X, y)
        else:
            raise ValueError(f"Unknown direction: {self.direction}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return X reduced to the selected features."""
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() before transform().")
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def plot_search_path(self) -> None:
        """Visualise the score at each step of the search."""
        import matplotlib.pyplot as plt

        steps = list(range(len(self.score_history_)))
        sizes = [s for s, _ in self.score_history_]
        scores = [sc for _, sc in self.score_history_]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))

        axes[0].plot(steps, scores, marker="o", markersize=4)
        axes[0].set_xlabel("Search step")
        axes[0].set_ylabel(f"CV {self.scoring}")
        axes[0].set_title("Score per search step")
        axes[0].grid(alpha=0.3)

        # Plot score vs subset size
        size_scores: Dict[int, List[float]] = {}
        for sz, sc in self.score_history_:
            size_scores.setdefault(sz, []).append(sc)
        best_per_size = {k: max(v) for k, v in size_scores.items()}
        sorted_sizes = sorted(best_per_size)
        sorted_scores = [best_per_size[k] for k in sorted_sizes]

        axes[1].plot(sorted_sizes, sorted_scores, marker="s", color="darkorange")
        if self.selected_features_ is not None:
            final_k = len(self.selected_features_)
            axes[1].axvline(
                x=final_k, color="red", linestyle="--", label=f"Selected k={final_k}"
            )
            axes[1].legend()
        axes[1].set_xlabel("Number of features")
        axes[1].set_ylabel(f"Best CV {self.scoring}")
        axes[1].set_title("Score vs subset size")
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()
```

### Early Stopping Criterion

```python
class EarlyStopper:
    """
    Composite early stopping for sequential feature selection.

    Combines patience (no-improvement) and diminishing-returns criteria.

    Parameters
    ----------
    patience : int
        Steps without improvement before stopping.
    tol : float
        Minimum score gain counted as improvement.
    min_delta_fraction : float
        Stop when marginal gain < this fraction of max marginal gain seen.
    """

    def __init__(
        self,
        patience: int = 5,
        tol: float = 1e-4,
        min_delta_fraction: float = 0.1,
    ):
        self.patience = patience
        self.tol = tol
        self.min_delta_fraction = min_delta_fraction

        self._scores: List[float] = []
        self._no_improve_count = 0

    def update(self, score: float) -> bool:
        """
        Record a new score and return True if search should stop.

        Parameters
        ----------
        score : float
            The cross-validated score at the current step.

        Returns
        -------
        bool
            True means stop; False means continue.
        """
        self._scores.append(score)

        if len(self._scores) == 1:
            return False

        # Patience criterion
        if score > self._scores[-2] + self.tol:
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

        if self._no_improve_count >= self.patience:
            return True

        # Diminishing returns criterion
        if len(self._scores) >= 3:
            deltas = [
                self._scores[i] - self._scores[i - 1]
                for i in range(1, len(self._scores))
            ]
            max_delta = max(deltas)
            current_delta = deltas[-1]
            if max_delta > 0 and current_delta < self.min_delta_fraction * max_delta:
                return True

        return False
```

## Common Pitfalls

### 1. Data Leakage in Cross-Validation

The outer CV folds must be used consistently. Never fit the scaler on the full training set before sequential search — re-fit within each fold.

```python
# Wrong: leaks test fold statistics
scaler = StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)
sfs.fit(X_scaled, y_train)

# Correct: pipeline handles scaling inside CV
from sklearn.pipeline import Pipeline
pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])
sfs = SequentialFeatureSelector(pipe, ...)
sfs.fit(X_train, y_train)
```

### 2. Forgetting That SFS/SBS Are Greedy

Basic SFS and SBS are greedy. A feature rejected early may become highly informative once other features are added. SFFS partially corrects this, but the floating mechanism is still heuristic, not exhaustive.

### 3. Setting `k*` Too High

More features does not always mean better generalisation. Use the score history to choose $k^*$ as the knee of the score-vs-features curve, not a fixed target.

### 4. Skipping the Cache

Without caching, SFFS re-evaluates subsets it has already scored during the backward phase. A dict-based cache keyed on sorted feature tuples can reduce total evaluations by 20–40%.

## Connections

**Builds on:**
- Module 01: Statistical filter methods (pre-screening candidates)
- Module 02: Information-theoretic metrics (alternative scoring criteria)
- Cross-validation fundamentals

**Leads to:**
- Guide 02: Boruta and beam search (population-based wrappers)
- Guide 03: Scalable wrapper implementations
- Module 04: Embedded methods (feature selection during training)

**Related:**
- Recursive Feature Elimination (RFE): backward selection with coefficient-based ranking (no CV per step)
- Branch-and-bound: exact exponential-time counterpart to greedy sequential search

## Practice Problems

### Problem 1: Implement Bidirectional Search

Implement a bidirectional SFS/SBS that runs both directions simultaneously and halts when the two expanding sets would overlap:

```python
def bidirectional_search(X, y, estimator, k_target, cv=5):
    """
    Run SFS and SBS in parallel. Return the union of selected features
    from whichever direction achieves the higher score at k_target.
    """
    pass
```

### Problem 2: Score-Plateau Detection

Implement plateau-based stopping using a rolling linear regression over the last $w$ steps. Return True when the slope is below $\epsilon$:

```python
def plateau_stopper(scores: list, w: int = 5, epsilon: float = 1e-4) -> bool:
    """Return True if the score curve is flat over the last w steps."""
    pass
```

### Problem 3: Profile Sequential Search Cost

Time SFS at $k^* \in \{5, 10, 20\}$ on a dataset with $p = 100$ features. Fit the observed runtimes to the $O(k^* p \cdot T_{\mathcal{M}})$ model and verify the linear relationship with $k^*$.

## Further Reading

- Pudil, P., Novovicova, J., & Kittler, J. (1994). "Floating search methods in feature selection." Pattern Recognition Letters, 15(11), 1119–1125. — The original SFFS paper.
- Ferri, F. J., Pudil, P., Hatef, M., & Kittler, J. (1994). "Comparative study of techniques for large-scale feature selection." Pattern Recognition in Practice IV, 403–413. — Empirical comparison of SFS, SBS, SFFS, SBFS.
- Jain, A. K., & Zongker, D. (1997). "Feature selection: Evaluation, application, and small sample performance." IEEE TPAMI, 19(2), 153–158. — When and why SFFS outperforms SFS.

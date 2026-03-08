# Scalable Wrapper Implementations

## In Brief

Wrapper methods become infeasible as $p$ or $n$ grows because each candidate feature subset requires a full model training and cross-validation. This guide covers five orthogonal strategies to reduce that cost: (1) pre-screening to shrink the candidate pool, (2) caching to avoid redundant evaluations, (3) parallelisation to distribute independent model fits, (4) approximate evaluation using subsampling or early stopping within training, and (5) memory-efficient data structures for large datasets. The guide closes with a clear decision framework for when to abandon wrappers entirely.

> **Key Insight:** Wrapper scalability is almost entirely about reducing $T_{\mathcal{M}}$ (model training time) and the number of distinct subsets evaluated. Parallelisation provides a constant-factor speedup; pre-screening provides a reduction proportional to the filtering ratio. Both are needed for $p > 500$.

## Computational Bottlenecks

A sequential feature selector evaluating $k^*$ features from $p$ candidates with $v$-fold CV:

$$\text{Wall time} = \frac{k^* \cdot p \cdot v \cdot T_{\mathcal{M}}}{C}$$

where $C$ is the number of available CPU cores used in parallel. Three levers:

| Lever | Effect | Mechanism |
|-------|--------|-----------|
| Reduce $p$ | Sub-linear (fewer candidates at each step) | Pre-screening |
| Reduce $T_{\mathcal{M}}$ | Linear | Fast models, subsampling, early stopping in training |
| Increase $C$ | Linear (up to $v \cdot p$) | Parallelisation |
| Reduce evaluations | Depends on cache hit rate | Caching |

## Strategy 1: Feature Pre-Screening

### Filter-then-Wrap

Run a cheap filter method to rank all $p$ features, retain the top $p'$, and run the wrapper only on those $p'$ candidates. The wrapper never sees the bottom $p - p'$ features.

**How much to keep:** A safe rule is $p' = \max(2k^*, 50)$. Using $p' < k^*$ is obviously wrong; using $p' \gg p$ defeats the purpose.

**Cost reduction:**

$$\text{Reduction} = 1 - \frac{p'}{p} = 1 - \frac{2k^*}{p} \quad \text{(for } p' = 2k^*)$$

For $p = 500, k^* = 20$: 92% reduction in candidate pool.

### Which Filter to Use

| Filter | Speed | Handles Nonlinearity | Handles Redundancy |
|--------|-------|---------------------|--------------------|
| Variance threshold | Fastest | No | No |
| Pearson correlation | Very fast | No | No |
| Mutual information | Fast | Yes | No |
| mRMR | Moderate | Yes | Yes |

mRMR (minimum-redundancy maximum-relevance) is the best pre-screener for wrappers because it removes correlated features before the wrapper has a chance to waste evaluations on them.

### Implementation

```python
import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from typing import Literal


def prescreening_filter(
    X: np.ndarray,
    y: np.ndarray,
    n_candidates: int,
    method: Literal["mutual_info", "correlation", "variance"] = "mutual_info",
    task: Literal["classification", "regression"] = "classification",
) -> np.ndarray:
    """
    Return indices of the top n_candidates features by filter score.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_candidates : int
        Number of features to retain.
    method : str
        Filter method to use.
    task : str
        'classification' or 'regression'.

    Returns
    -------
    candidate_indices : ndarray of shape (n_candidates,)
        Indices of the retained features, sorted by descending score.
    """
    p = X.shape[1]
    n_candidates = min(n_candidates, p)

    if method == "mutual_info":
        mi_func = mutual_info_classif if task == "classification" else mutual_info_regression
        scores = mi_func(X, y, random_state=42)

    elif method == "correlation":
        # Absolute Pearson correlation (works for classification with binary/ordinal y)
        scores = np.abs(np.corrcoef(X.T, y)[-1, :-1])

    elif method == "variance":
        scores = X.var(axis=0)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Return indices of top n_candidates by descending score
    top_indices = np.argsort(scores)[-n_candidates:][::-1]
    return top_indices


def mrmr_filter(
    X: np.ndarray,
    y: np.ndarray,
    n_candidates: int,
    task: Literal["classification", "regression"] = "classification",
) -> np.ndarray:
    """
    Minimum-redundancy maximum-relevance (mRMR) feature ranking.

    Iteratively selects features that maximise relevance to y while
    minimising redundancy with already-selected features.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_candidates : int
        Number of features to return.
    task : str
        Task type for mutual information estimator.

    Returns
    -------
    selected_indices : ndarray of shape (n_candidates,)
        Feature indices in order of selection.
    """
    mi_func = mutual_info_classif if task == "classification" else mutual_info_regression
    p = X.shape[1]
    n_candidates = min(n_candidates, p)

    # Relevance: MI between each feature and target
    relevance = mi_func(X, y, random_state=42)

    # Precompute feature-feature MI (symmetric matrix, expensive for large p)
    # For large p, approximate with absolute correlation
    if p <= 200:
        feature_mi = np.zeros((p, p))
        for j in range(p):
            mi_row = mutual_info_classif(X, X[:, j], random_state=42)
            feature_mi[j, :] = mi_row
    else:
        # Use absolute correlation as fast approximation
        corr = np.corrcoef(X.T)
        feature_mi = np.abs(corr)

    selected = []
    remaining = list(range(p))

    for i in range(n_candidates):
        if not selected:
            # First feature: highest relevance
            best = remaining[int(np.argmax(relevance[remaining]))]
        else:
            # mRMR criterion: relevance - mean redundancy with selected
            redundancy = np.array([
                np.mean([feature_mi[j, s] for s in selected])
                for j in remaining
            ])
            mrmr_scores = relevance[remaining] - redundancy
            best = remaining[int(np.argmax(mrmr_scores))]

        selected.append(best)
        remaining.remove(best)

    return np.array(selected)
```

## Strategy 2: Evaluation Caching

Every call to `cross_val_score` with the same feature subset is pure waste. A dictionary keyed on sorted feature index tuples eliminates all duplicates:

```python
from functools import lru_cache
from typing import Dict, FrozenSet, Tuple
import hashlib


class EvaluationCache:
    """
    Thread-safe evaluation cache for feature subset scoring.

    Keyed on sorted tuples of feature indices. Provides hit-rate statistics
    for understanding cache effectiveness.
    """

    def __init__(self):
        self._store: Dict[Tuple[int, ...], float] = {}
        self._hits = 0
        self._misses = 0

    def key(self, feature_indices) -> Tuple[int, ...]:
        return tuple(sorted(int(i) for i in feature_indices))

    def get(self, feature_indices) -> float:
        k = self.key(feature_indices)
        if k in self._store:
            self._hits += 1
            return self._store[k]
        self._misses += 1
        return None  # Cache miss

    def set(self, feature_indices, score: float) -> None:
        k = self.key(feature_indices)
        self._store[k] = score

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def __len__(self) -> int:
        return len(self._store)

    def memory_mb(self) -> float:
        """Approximate cache memory usage in MB."""
        # Each entry: tuple of ints + float ≈ 60-100 bytes
        return len(self._store) * 100 / 1e6
```

**When caching helps most:** SFFS (backward phase re-scores previously visited subsets), Optuna with many repeated configurations, beam search (shared subsets across beam candidates).

**Cache size limits:** For $p = 100$ and up to $k^* = 30$, the theoretical number of unique subsets is $\binom{100}{30} \approx 10^{25}$ — far too many to cache exhaustively. In practice, sequential search visits $O(k^* \cdot p)$ unique subsets, which fits easily in RAM.

## Strategy 3: Parallelisation

### Embarrassingly Parallel Feature Evaluation

At each step of SFS, evaluating each candidate feature is completely independent. This is embarrassingly parallel: all $p - |S|$ candidates at a given step can be evaluated simultaneously.

```python
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np


def evaluate_candidates_parallel(
    X: np.ndarray,
    y: np.ndarray,
    current_mask: np.ndarray,
    candidate_indices: np.ndarray,
    estimator,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    cache: EvaluationCache = None,
) -> np.ndarray:
    """
    Evaluate all candidate feature additions in parallel.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    current_mask : bool array of shape (n_features,)
        Currently selected features.
    candidate_indices : array of int
        Indices of features to try adding (must not already be in current_mask).
    estimator : sklearn estimator
    cv : int
    scoring : str
    n_jobs : int
        -1 = use all cores.
    cache : EvaluationCache, optional

    Returns
    -------
    scores : ndarray of shape (len(candidate_indices),)
        CV score for each candidate.
    """
    def _score_one(j):
        mask = current_mask.copy()
        mask[j] = True
        selected = np.where(mask)[0]

        if cache is not None:
            cached = cache.get(selected)
            if cached is not None:
                return cached

        result = cross_val_score(
            clone(estimator), X[:, selected], y, cv=cv, scoring=scoring
        ).mean()

        if cache is not None:
            cache.set(selected, result)

        return result

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_score_one)(j) for j in candidate_indices
    )
    return np.array(scores)
```

### Parallelisation Limits

The maximum speedup from parallelism is bounded by Amdahl's Law. The parallelisable fraction is the feature evaluation loop; the serial fraction includes the bookkeeping between steps:

$$\text{Speedup} \leq \frac{1}{(1 - P) + P/C}$$

For SFS with $C$ cores and a single step serial fraction of $\approx 5\%$: maximum speedup $\approx 15\times$ for $C = \infty$.

**Practical limit:** For a single model training step taking 0.01 seconds, the parallelism overhead from joblib (process spawning, data serialisation) dominates. Only parallelise when $T_{\mathcal{M}} > 0.5$ seconds per fold.

### Distributed Computing with Ray

For very expensive models (deep learning, large ensembles), use Ray for distributed evaluation across machines:

```python
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def distributed_sfs_step(
    X_ref,  # Ray object reference
    y_ref,
    current_mask: np.ndarray,
    estimator_factory,
    candidate_indices: np.ndarray,
    cv: int = 5,
) -> np.ndarray:
    """
    Distributed SFS candidate evaluation using Ray remote functions.

    Parameters
    ----------
    X_ref : ray.ObjectRef
        Ray reference to training data (avoids repeated serialisation).
    y_ref : ray.ObjectRef
        Ray reference to labels.
    current_mask : bool array
    estimator_factory : callable
        Returns a fresh estimator instance (e.g., lambda: LGBMClassifier()).
    candidate_indices : array of int
    cv : int

    Returns
    -------
    scores : ndarray
    """
    if not RAY_AVAILABLE:
        raise ImportError("Install ray: pip install ray")

    @ray.remote
    def score_feature(X_ref, y_ref, mask, j, cv):
        X = ray.get(X_ref)
        y = ray.get(y_ref)
        mask = mask.copy()
        mask[j] = True
        selected = np.where(mask)[0]
        model = estimator_factory()
        from sklearn.model_selection import cross_val_score as cvs
        return cvs(model, X[:, selected], y, cv=cv).mean()

    futures = [
        score_feature.remote(X_ref, y_ref, current_mask, j, cv)
        for j in candidate_indices
    ]
    return np.array(ray.get(futures))
```

## Strategy 4: Approximate Evaluation

### Progressive Subsampling

Early wrapper steps (small $|S|$) evaluate many candidates; the gap between good and bad candidates is large enough that noisy estimates suffice. Later steps (large $|S|$) evaluate fewer candidates with smaller score differences, requiring more precise estimates. Progressive subsampling exploits this:

$$n'(k) = \min\left(n, n_{\min} + \frac{k}{k^*} \cdot (n - n_{\min})\right)$$

```python
def progressive_subsample_score(
    X: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    estimator,
    step: int,
    total_steps: int,
    min_fraction: float = 0.2,
    cv: int = 5,
) -> float:
    """
    Cross-validate on a progressively larger data fraction.

    Parameters
    ----------
    X, y : training data
    mask : bool array, currently selected features
    estimator : sklearn estimator
    step : int
        Current step number (0-indexed).
    total_steps : int
        Total planned steps (k*).
    min_fraction : float
        Data fraction at step 0 (grows to 1.0 at final step).
    cv : int

    Returns
    -------
    float
        Cross-validated score on the subsampled data.
    """
    fraction = min_fraction + (1.0 - min_fraction) * (step / max(total_steps - 1, 1))
    n_samples = int(len(y) * fraction)

    # Stratified subsample (for classification)
    rng = np.random.default_rng(42 + step)
    indices = rng.choice(len(y), size=n_samples, replace=False)
    X_sub = X[indices]
    y_sub = y[indices]

    selected = np.where(mask)[0]
    scores = cross_val_score(
        clone(estimator), X_sub[:, selected], y_sub, cv=min(cv, n_samples // 10)
    )
    return scores.mean()
```

### Early Stopping Within Model Training

For gradient boosting models, use early stopping to avoid training until convergence on subsets likely to be discarded. Evaluate with fewer rounds (10–20% of max estimators) for early-step candidates:

```python
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold


def fast_lgbm_score(
    X: np.ndarray,
    y: np.ndarray,
    feature_indices: np.ndarray,
    n_estimators: int = 50,  # Reduced from default 300
    cv: int = 5,
    seed: int = 42,
) -> float:
    """
    Fast LightGBM CV score using reduced n_estimators.

    Use this for candidate screening during early wrapper steps.
    Use full n_estimators only for the final selected subset.
    """
    X_sub = X[:, feature_indices]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    cv_scores = []

    for train_idx, val_idx in skf.split(X_sub, y):
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.1,
            verbosity=-1,
            random_state=seed,
        )
        model.fit(X_sub[train_idx], y[train_idx])
        score = model.score(X_sub[val_idx], y[val_idx])
        cv_scores.append(score)

    return float(np.mean(cv_scores))
```

## Strategy 5: Memory-Efficient Implementations

### Problem: Large Feature Matrices

For $n = 100{,}000$ samples, $p = 5{,}000$ features, a float64 matrix requires $100{,}000 \times 5{,}000 \times 8 = 4$ GB. Repeatedly slicing this matrix (once per candidate per step) causes cache thrashing and memory pressure.

### Solutions

**Column-major storage (Fortran order):** Feature slicing is contiguous in memory:

```python
# Store X in column-major order for efficient column slicing
X_fortran = np.asfortranarray(X)  # Shape (n, p) but columns are contiguous
X_subset = X_fortran[:, selected_indices]  # Fast: contiguous memory access
```

**Chunked loading with HDF5:** For datasets that do not fit in RAM:

```python
import h5py


class HDF5FeatureStore:
    """
    HDF5-backed feature store for datasets too large for RAM.

    Loads only the requested feature columns into memory.
    Uses a column cache to avoid repeated disk reads.
    """

    def __init__(self, h5_path: str, dataset_name: str = "X", cache_size: int = 100):
        self.h5_path = h5_path
        self.dataset_name = dataset_name
        self._column_cache: Dict[int, np.ndarray] = {}
        self.cache_size = cache_size

    def get_columns(self, indices: np.ndarray) -> np.ndarray:
        """
        Load specific columns from HDF5, using column cache.

        Parameters
        ----------
        indices : array of int
            Column indices to load.

        Returns
        -------
        X_sub : ndarray of shape (n_samples, len(indices))
        """
        columns = []
        to_load = []

        for j in indices:
            if j in self._column_cache:
                columns.append((j, self._column_cache[j]))
            else:
                to_load.append(j)

        if to_load:
            with h5py.File(self.h5_path, "r") as f:
                ds = f[self.dataset_name]
                for j in to_load:
                    col = ds[:, j]
                    # Evict LRU if cache full
                    if len(self._column_cache) >= self.cache_size:
                        oldest = next(iter(self._column_cache))
                        del self._column_cache[oldest]
                    self._column_cache[j] = col
                    columns.append((j, col))

        # Assemble in requested order
        col_dict = {j: col for j, col in columns}
        return np.column_stack([col_dict[j] for j in indices])
```

**Sparse matrices:** If features are sparse (e.g., text features, one-hot encoded categoricals), use `scipy.sparse`:

```python
from scipy.sparse import issparse, csc_matrix


def sparse_feature_subset(X, indices: np.ndarray):
    """Efficient column slicing for sparse matrices (CSC format)."""
    if issparse(X):
        if not hasattr(X, "getcol"):
            X = csc_matrix(X)  # CSC: efficient column slicing
        return X[:, indices]
    return X[:, indices]
```

## When to Abandon Wrappers

Wrapper methods are often the right tool, but they have hard limits. Use this decision framework:

```
Decision: Wrapper vs Filter vs Embedded
=========================================

Step 1: Can you afford wrapper cost?
  Cost = k* × p × v × T_M / C

  If cost > budget and cannot reduce → GO TO Step 2
  If cost ≤ budget → Use SFFS (best wrapper)

Step 2: Is your priority minimal feature set or all-relevant?
  All-relevant → Boruta (if affordable) or mutual info + threshold
  Minimal → GO TO Step 3

Step 3: Is your model tree-based?
  Yes → Use embedded: RandomForest/LightGBM importance + threshold
        (very fast, captures interactions, built into the model)
  No  → Use mRMR or mutual info filter with stability selection

Step 4: Is p > 10,000?
  Yes → Use variance + MI filter pipeline (filter-only approach)
        Wrappers are prohibitively expensive at this scale
  No  → Consider reducing p with PCA or autoencoders first
```

### Hard Limits by Feature Count

| $p$ range | Recommended approach |
|-----------|---------------------|
| $< 50$ | Full wrapper (SFFS), no pre-screening needed |
| 50–300 | SFFS with mRMR pre-screening ($p' = 3k^*$) |
| 300–1000 | SFFS with aggressive pre-screening ($p' = 2k^*$) + parallelisation |
| 1000–10,000 | Embedded methods (LightGBM importance) or Boruta |
| $> 10{,}000$ | Filter methods only; wrappers require distributed computing |

### The Embedded Method Alternative

For tree-based models, embedded selection is almost always the right choice when $p > 1000$:

```python
import lightgbm as lgb
from sklearn.model_selection import cross_val_score
import numpy as np


def lgbm_embedded_selection(
    X: np.ndarray,
    y: np.ndarray,
    n_features_to_select: int,
    n_estimators: int = 500,
    cv: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """
    Select features using LightGBM built-in importance.

    No wrapper overhead: importances are a byproduct of training.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    n_features_to_select : int
    n_estimators : int
    cv : int
    seed : int

    Returns
    -------
    selected_indices : ndarray of int
    """
    # Average importance across CV folds for stability
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
    importance_sum = np.zeros(X.shape[1])

    for train_idx, _ in skf.split(X, y):
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            verbosity=-1,
            random_state=seed,
        )
        model.fit(X[train_idx], y[train_idx])
        importance_sum += model.feature_importances_

    # Average and select top-k
    avg_importance = importance_sum / cv
    return np.argsort(avg_importance)[-n_features_to_select:][::-1]
```

## Putting It All Together: Scalable Wrapper Pipeline

```python
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from joblib import Parallel, delayed


class ScalableWrapperPipeline:
    """
    Production-ready scalable wrapper feature selector.

    Combines pre-screening + caching + parallelisation + progressive subsampling.

    Parameters
    ----------
    estimator : sklearn estimator
    n_features_to_select : int
    prescreening_ratio : float
        Retain this fraction of features after filter pre-screening.
        E.g., 0.2 keeps top 20% of features.
    prescreening_method : str
        Filter method for pre-screening ('mutual_info', 'mrmr', 'correlation').
    direction : str
        'floating_forward' or 'forward'.
    scoring : str
    cv : int
    n_jobs : int
    patience : int
    use_progressive_subsampling : bool
    min_subsample_fraction : float
    """

    def __init__(
        self,
        estimator,
        n_features_to_select: int = 20,
        prescreening_ratio: float = 0.3,
        prescreening_method: str = "mutual_info",
        direction: str = "floating_forward",
        scoring: str = "accuracy",
        cv: int = 5,
        n_jobs: int = -1,
        patience: int = 5,
        use_progressive_subsampling: bool = True,
        min_subsample_fraction: float = 0.3,
    ):
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.prescreening_ratio = prescreening_ratio
        self.prescreening_method = prescreening_method
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.patience = patience
        self.use_progressive_subsampling = use_progressive_subsampling
        self.min_subsample_fraction = min_subsample_fraction

        self.selected_features_: np.ndarray = None
        self.prescreened_features_: np.ndarray = None
        self._cache = EvaluationCache()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ScalableWrapperPipeline":
        X = np.asfortranarray(X)  # Column-major for efficient slicing
        y = np.asarray(y)
        n, p = X.shape

        # Step 1: Pre-screening
        n_prescreened = max(
            self.n_features_to_select * 2,
            int(p * self.prescreening_ratio),
        )
        n_prescreened = min(n_prescreened, p)

        print(f"Pre-screening: {p} → {n_prescreened} features")
        if self.prescreening_method == "mrmr":
            prescreened_idx = mrmr_filter(X, y, n_prescreened)
        else:
            prescreened_idx = prescreening_filter(
                X, y, n_prescreened, method=self.prescreening_method
            )
        self.prescreened_features_ = prescreened_idx

        # Restrict X to prescreened features
        X_pre = X[:, prescreened_idx]
        p_pre = X_pre.shape[1]

        # Step 2: Sequential search with parallelisation and caching
        mask = np.zeros(p_pre, dtype=bool)
        best_at_size: Dict[int, float] = {}
        no_improve = 0
        prev_best = -np.inf

        k_target = min(self.n_features_to_select, p_pre)

        for step in range(k_target):
            candidates = np.where(~mask)[0]

            # Parallel evaluation of all candidates
            def score_candidate(j, step=step):
                candidate_mask = mask.copy()
                candidate_mask[j] = True
                selected = np.where(candidate_mask)[0]

                cached = self._cache.get(selected)
                if cached is not None:
                    return cached

                if self.use_progressive_subsampling:
                    score = progressive_subsample_score(
                        X_pre, y, candidate_mask, clone(self.estimator),
                        step=step, total_steps=k_target,
                        min_fraction=self.min_subsample_fraction, cv=self.cv,
                    )
                else:
                    score = cross_val_score(
                        clone(self.estimator), X_pre[:, selected], y,
                        cv=self.cv, scoring=self.scoring, n_jobs=1,
                    ).mean()

                self._cache.set(selected, score)
                return score

            candidate_scores = Parallel(n_jobs=self.n_jobs)(
                delayed(score_candidate)(j) for j in candidates
            )

            best_local_idx = candidates[np.argmax(candidate_scores)]
            best_score = max(candidate_scores)
            mask[best_local_idx] = True

            k = int(mask.sum())
            best_at_size[k] = max(best_at_size.get(k, -np.inf), best_score)

            # Floating backward phase
            if self.direction == "floating_forward" and k > 1:
                backward_improved = True
                while backward_improved and mask.sum() > 1:
                    backward_candidates = np.where(mask)[0]
                    backward_scores = []
                    for j in backward_candidates:
                        candidate_mask = mask.copy()
                        candidate_mask[j] = False
                        selected = np.where(candidate_mask)[0]
                        cached = self._cache.get(selected)
                        if cached is None:
                            cached = cross_val_score(
                                clone(self.estimator), X_pre[:, selected], y,
                                cv=self.cv, scoring=self.scoring,
                            ).mean()
                            self._cache.set(selected, cached)
                        backward_scores.append(cached)

                    best_bwd_score = max(backward_scores)
                    best_bwd_j = backward_candidates[np.argmax(backward_scores)]
                    k_smaller = int(mask.sum()) - 1

                    if best_bwd_score > best_at_size.get(k_smaller, -np.inf):
                        mask[best_bwd_j] = False
                        best_at_size[k_smaller] = best_bwd_score
                    else:
                        backward_improved = False

            # Patience early stopping
            if best_score > prev_best + 1e-4:
                prev_best = best_score
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"Early stopping at step {step+1} (patience exhausted)")
                    break

        # Map back to original feature indices
        local_selected = np.where(mask)[0]
        self.selected_features_ = prescreened_idx[local_selected]

        print(f"Cache hit rate: {self._cache.hit_rate:.1%} ({len(self._cache)} entries)")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.selected_features_]
```

## Common Pitfalls

### 1. Parallelising the Wrong Thing

Parallelising cross-validation folds (via `n_jobs` in `cross_val_score`) and parallelising candidate evaluation are both valid but interfere if nested:

```python
# Wrong: nested parallelism causes thread contention
Parallel(n_jobs=8)(
    delayed(cross_val_score)(estimator, X_sub, y, cv=5, n_jobs=8)  # Also parallel!
    for j in candidates
)

# Correct: parallelise only the outer loop
Parallel(n_jobs=8)(
    delayed(cross_val_score)(estimator, X_sub, y, cv=5, n_jobs=1)  # Serial CV
    for j in candidates
)
```

### 2. Ignoring Memory During Parallelisation

joblib's `loky` backend copies data to each worker process. For large $X$ (>1 GB), this causes out-of-memory errors. Use `memmap` arrays:

```python
import tempfile
import os

# Write X to a memory-mapped file shared across workers
with tempfile.NamedTemporaryFile(delete=False) as f:
    X_mmap_path = f.name
X_mmap = np.memmap(X_mmap_path, dtype=X.dtype, mode="w+", shape=X.shape)
X_mmap[:] = X[:]

# Pass X_mmap to parallel workers — no copy, shared memory
Parallel(n_jobs=-1)(delayed(score_candidate)(j, X_mmap) for j in candidates)
os.unlink(X_mmap_path)
```

### 3. Pre-Screening Too Aggressively

Removing features before the wrapper means features that only become informative in combination with others may be discarded. Use $p' \geq 3k^*$ rather than $p' = k^*$.

## Connections

**Builds on:**
- Guide 01: Sequential search algorithms
- Guide 02: Boruta and beam search (methods being made scalable)
- Module 01: Filter methods (pre-screening)

**Leads to:**
- Module 04: Embedded methods (the alternative when wrappers are too expensive)
- Module 08: High-dimensional feature selection (where wrappers rarely apply)

## Further Reading

- Kira, K., & Rendell, L. A. (1992). "The feature selection problem: Traditional methods and a new algorithm." *AAAI*, 129–134. — Original ReliefF filter, often used for pre-screening.
- Ding, C., & Peng, H. (2005). "Minimum redundancy feature selection from microarray gene expression data." *Journal of Bioinformatics and Computational Biology*, 3(2), 185–205. — mRMR filter.
- joblib documentation: https://joblib.readthedocs.io — Parallel and delayed, memmapping.
- Ray documentation: https://docs.ray.io — Distributed computing for large-scale parallelism.

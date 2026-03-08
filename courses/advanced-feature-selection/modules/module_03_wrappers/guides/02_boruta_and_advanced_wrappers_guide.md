# Boruta and Advanced Wrapper Methods

## In Brief

Boruta is a wrapper method that asks a fundamentally different question from sequential search: instead of "what is the minimal optimal feature set?", it asks "which features are relevant at all?" It does this by creating shadow features — randomly shuffled copies of real features — and using a Random Forest to test whether each real feature outperforms the best shadow feature consistently over many iterations. Beam search and Optuna-driven wrappers address the computational limitation of sequential search by maintaining multiple candidate subsets simultaneously or treating the search as a hyperparameter optimisation problem.

> **Key Insight:** Boruta finds the *all-relevant* feature set: every feature that carries any information about the target, not just the non-redundant minimal set. This makes it ideal for exploratory analysis and for situations where interpretability of individual features matters. Sequential methods find a *minimal-optimal* set: the smallest subset that achieves near-optimal predictive performance.

## Boruta Algorithm

### Intuition

Random Forests internally rank features by importance. Boruta asks: "How important does a feature need to be before we can be confident it's not just noise?" It answers this by adding noise features — shadow features — as a controlled baseline. A real feature that cannot consistently beat the best noise feature has no detectable signal.

### Shadow Features

For a dataset $\mathbf{X} \in \mathbb{R}^{n \times p}$:

1. Create shadow matrix $\mathbf{X}^{\text{shadow}} \in \mathbb{R}^{n \times p}$ where each column $j$ is a random permutation of the original column $j$:
$$X^{\text{shadow}}_{ij} = X_{\sigma_j(i), j}, \quad \sigma_j \sim \text{Uniform}(\text{permutations of } [n])$$

2. Augment the feature matrix:
$$\mathbf{X}^{\text{aug}} = [\mathbf{X} \mid \mathbf{X}^{\text{shadow}}] \in \mathbb{R}^{n \times 2p}$$

3. Train a Random Forest on $\mathbf{X}^{\text{aug}}$ and extract feature importances $\{I_j\}_{j=1}^{2p}$.

4. Define the shadow threshold:
$$\text{MZSA} = \max_{j=p+1}^{2p} I_j \quad (\text{Maximum Z-Score Among Shadow Attributes})$$

### Statistical Testing

At each iteration $t$, each feature $j \in \{1, \ldots, p\}$ either "hits" (beats MZSA) or "misses":

$$H_{jt} = \mathbf{1}[I_j > \text{MZSA}_t]$$

After $T$ iterations, each feature has a hit count $h_j = \sum_{t=1}^T H_{jt}$.

Under the null hypothesis ($f_j$ is irrelevant), hits follow a Binomial distribution:
$$h_j \sim \text{Binomial}(T, 0.5)$$

Boruta uses a two-sided test with Bonferroni correction for $p$ simultaneous comparisons at significance level $\alpha$:

- **Confirmed relevant:** $P(h_j \geq h_j^{\text{obs}}) < \alpha / (2p)$
- **Confirmed irrelevant:** $P(h_j \leq h_j^{\text{obs}}) < \alpha / (2p)$
- **Tentative:** neither criterion met (needs more iterations)

### Algorithm Steps

```
INPUT: X ∈ R^{n×p}, y, T (max iterations), α (significance), estimator

INITIALISE:
  status[j] ← 'tentative' for all j ∈ {1,...,p}

FOR t = 1 TO T:
  1. Permute each column of X separately to form X_shadow
  2. Augment: X_aug ← [X | X_shadow]  (shape n × 2p)
  3. Fit Random Forest on X_aug
  4. Extract importances I[1..2p]
  5. MZSA ← max(I[p+1..2p])

  6. For each j with status[j] = 'tentative':
       hits[j] += 1 if I[j] > MZSA else 0

  7. Apply binomial test with Bonferroni correction:
       If P(Bin(t, 0.5) >= hits[j]) < α/(2p):
           status[j] ← 'confirmed'
       If P(Bin(t, 0.5) <= hits[j]) < α/(2p):
           status[j] ← 'rejected'

  8. Shuffle tentative feature importances in X (optional, reduces noise)

  9. If no tentative features remain: BREAK

RETURN: {j : status[j] = 'confirmed'}
```

### All-Relevant vs Minimal-Optimal

| Property | Boruta (all-relevant) | SFS/SFFS (minimal-optimal) |
|----------|----------------------|---------------------------|
| Returns | Every feature with signal | Smallest subset for best CV score |
| Redundant features | Included (both A and B if correlated) | One of A or B removed |
| Use case | Exploratory analysis, interpretability | Deployment, dimensionality reduction |
| Result stability | High (statistical guarantee) | Lower (path-dependent) |
| Compute cost | $O(T \cdot p \cdot T_{\text{RF}})$ | $O(k^* \cdot p \cdot T_{\mathcal{M}})$ |

## Code Implementation: Boruta with LightGBM

```python
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.base import clone
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple


class BorutaSelector:
    """
    Boruta feature selection using any tree-based estimator.

    Parameters
    ----------
    estimator : sklearn estimator
        A tree-based model (RandomForest, LightGBM, etc.) that exposes
        feature_importances_ after fitting.
    max_iter : int
        Maximum number of Boruta iterations.
    alpha : float
        Family-wise error rate for Bonferroni-corrected binomial tests.
    percentile : int
        Percentile of shadow importances used as threshold (default 100 = max).
    two_step : bool
        Use the two-step correction that accounts for tentative features.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        estimator=None,
        max_iter: int = 100,
        alpha: float = 0.05,
        percentile: int = 100,
        two_step: bool = True,
        random_state: int = 42,
    ):
        if estimator is None:
            estimator = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                subsample=0.632,  # Bootstrap fraction — matches RF default
                colsample_bytree=0.632,
                random_state=random_state,
                verbosity=-1,
            )
        self.estimator = estimator
        self.max_iter = max_iter
        self.alpha = alpha
        self.percentile = percentile
        self.two_step = two_step
        self.random_state = random_state

        self.confirmed_: Optional[np.ndarray] = None
        self.rejected_: Optional[np.ndarray] = None
        self.tentative_: Optional[np.ndarray] = None
        self.importance_history_: Optional[np.ndarray] = None
        self.shadow_max_history_: List[float] = []

    def _get_importances(self, X_aug: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the estimator on augmented matrix and return feature importances."""
        model = clone(self.estimator)
        model.fit(X_aug, y)
        return np.array(model.feature_importances_)

    def _binomial_test(self, hits: np.ndarray, n_iter: int, p: int) -> np.ndarray:
        """
        Two-sided binomial test with Bonferroni correction.

        Returns an array of decision codes:
          +1 = confirmed relevant
          -1 = confirmed irrelevant
           0 = tentative
        """
        # Bonferroni threshold: alpha / (2 * n_active_features)
        threshold = self.alpha / (2 * p)
        decisions = np.zeros(len(hits), dtype=int)

        for j, h in enumerate(hits):
            # P(X >= h) under Bin(n_iter, 0.5) — upper tail
            p_upper = 1 - binom.cdf(h - 1, n_iter, 0.5)
            # P(X <= h) under Bin(n_iter, 0.5) — lower tail
            p_lower = binom.cdf(h, n_iter, 0.5)

            if p_upper < threshold:
                decisions[j] = 1   # Confirmed relevant
            elif p_lower < threshold:
                decisions[j] = -1  # Confirmed irrelevant

        return decisions

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BorutaSelector":
        """
        Run Boruta feature selection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        """
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        n, p = X.shape

        # Track hit counts and importance history
        hits = np.zeros(p, dtype=int)
        status = np.zeros(p, dtype=int)  # 0=tentative, 1=confirmed, -1=rejected
        importance_history = np.full((self.max_iter, p), np.nan)
        self.shadow_max_history_ = []

        for t in range(self.max_iter):
            # Only work with tentative features
            tentative_mask = status == 0
            if not tentative_mask.any():
                break

            # Create shadow matrix: permute each column independently
            X_shadow = np.apply_along_axis(rng.permutation, 0, X)

            # Augment: real features + shadow features
            X_aug = np.hstack([X, X_shadow])
            y_shuffled = rng.permutation(y)  # Shuffle y to break any residual signal
            # Note: we do NOT shuffle y for the actual fit — only shadow features encode noise
            importances = self._get_importances(X_aug, y)

            real_imp = importances[:p]
            shadow_imp = importances[p:]

            # Shadow threshold (default: maximum; percentile=100)
            shadow_thresh = np.percentile(shadow_imp, self.percentile)
            self.shadow_max_history_.append(shadow_thresh)

            # Update hits for tentative features
            importance_history[t, :] = real_imp
            for j in np.where(tentative_mask)[0]:
                if real_imp[j] > shadow_thresh:
                    hits[j] += 1

            # Statistical test (only on tentative features)
            n_tentative = tentative_mask.sum()
            if n_tentative > 0:
                decisions = self._binomial_test(hits[tentative_mask], t + 1, n_tentative)
                tentative_indices = np.where(tentative_mask)[0]
                for idx, decision in zip(tentative_indices, decisions):
                    if decision != 0:
                        status[idx] = decision

        self.confirmed_ = np.where(status == 1)[0]
        self.rejected_ = np.where(status == -1)[0]
        self.tentative_ = np.where(status == 0)[0]
        self.importance_history_ = importance_history
        self._hits = hits

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return X restricted to confirmed-relevant features."""
        if self.confirmed_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.confirmed_]

    def plot_importances(self, feature_names: Optional[List[str]] = None) -> None:
        """
        Violin plot of per-feature importance distributions across iterations.
        Overlays shadow max distribution for visual comparison.
        """
        import matplotlib.pyplot as plt

        p = self.importance_history_.shape[1]
        if feature_names is None:
            feature_names = [f"f{j}" for j in range(p)]

        # Collect non-NaN importances per feature
        feat_imps = [
            self.importance_history_[:, j][~np.isnan(self.importance_history_[:, j])]
            for j in range(p)
        ]

        # Sort by median importance
        order = np.argsort([np.median(imp) for imp in feat_imps])[::-1]

        fig, ax = plt.subplots(figsize=(max(10, p // 2), 5))

        parts = ax.violinplot(
            [feat_imps[i] for i in order],
            positions=range(p),
            showmedians=True,
        )

        # Colour by status
        confirmed_set = set(self.confirmed_)
        rejected_set = set(self.rejected_)
        for i, patch in enumerate(parts["bodies"]):
            orig_j = order[i]
            if orig_j in confirmed_set:
                patch.set_facecolor("#6f9")
            elif orig_j in rejected_set:
                patch.set_facecolor("#f99")
            else:
                patch.set_facecolor("#aaa")

        # Shadow threshold line
        ax.axhline(
            np.median(self.shadow_max_history_),
            color="red",
            linestyle="--",
            label="Median shadow max",
        )

        ax.set_xticks(range(p))
        ax.set_xticklabels(
            [feature_names[i] for i in order], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylabel("Feature importance")
        ax.set_title("Boruta: feature importance distributions vs shadow threshold")
        ax.legend()
        plt.tight_layout()
        plt.show()
```

## Beam Search for Feature Selection

### Overview

Beam search maintains a set of $w$ candidate subsets ("beam") at each step. At each iteration, all candidates are expanded (one feature added or removed), scored, and the best $w$ results are kept. This provides breadth that greedy sequential search lacks, at a cost of $w \times$ the compute of SFS.

### Algorithm

```
INPUT: F, k*, M, J, beam_width w

beam ← [∅]  (or [F] for backward)

FOR step = 1 TO k*:
    candidates ← []
    FOR S IN beam:
        FOR f IN F \ S:
            candidates.append(S ∪ {f})

    # Score all candidates
    scored ← [(J(S), S) for S in candidates]

    # Keep top w
    beam ← [S for (_, S) in sorted(scored)[-w:]]

RETURN: beam[0]  (highest-scoring remaining candidate)
```

### Memory-Bounded Variants

Standard beam search: memory $O(w \cdot k^*)$, evaluations $O(w \cdot k^* \cdot p)$.

**Stochastic beam search** introduces randomness in the selection of the beam: instead of keeping the top $w$ deterministically, sample $w$ candidates with probability proportional to their score. This allows exploration beyond the greedy frontier:

$$P(\text{select } S_i) \propto \exp\left(\frac{J(S_i)}{\tau}\right)$$

where $\tau$ is a temperature parameter (high $\tau$ = more random, low $\tau$ = greedy).

**Random restarts:** run beam search $r$ times from different random starting points and take the best result across all runs.

```python
import heapq
from typing import FrozenSet


class BeamSearchSelector:
    """
    Beam search for feature selection.

    Parameters
    ----------
    estimator : sklearn estimator
    beam_width : int
        Number of candidate subsets maintained at each step.
    n_features_to_select : int
        Target feature count.
    scoring : str
        Scoring metric.
    cv : int
        Cross-validation folds.
    stochastic : bool
        Use stochastic beam search with temperature sampling.
    temperature : float
        Temperature for stochastic sampling (ignored if stochastic=False).
    n_restarts : int
        Number of random restarts (stochastic mode).
    n_jobs : int
        Parallel jobs.
    """

    def __init__(
        self,
        estimator,
        beam_width: int = 5,
        n_features_to_select: int = 10,
        scoring: str = "accuracy",
        cv: int = 5,
        stochastic: bool = False,
        temperature: float = 0.01,
        n_restarts: int = 3,
        n_jobs: int = -1,
    ):
        self.estimator = estimator
        self.beam_width = beam_width
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.stochastic = stochastic
        self.temperature = temperature
        self.n_restarts = n_restarts
        self.n_jobs = n_jobs

        self._cache: Dict[Tuple, float] = {}
        self.selected_features_: Optional[np.ndarray] = None
        self.best_score_: float = -np.inf

    def _score(self, X: np.ndarray, y: np.ndarray, features: FrozenSet) -> float:
        key = tuple(sorted(features))
        if not key:
            return -np.inf
        if key not in self._cache:
            from sklearn.model_selection import cross_val_score
            from sklearn.base import clone
            scores = cross_val_score(
                clone(self.estimator), X[:, key], y,
                cv=self.cv, scoring=self.scoring, n_jobs=self.n_jobs
            )
            self._cache[key] = float(scores.mean())
        return self._cache[key]

    def _select_beam(
        self,
        scored_candidates: List[Tuple[float, FrozenSet]],
    ) -> List[FrozenSet]:
        """Select top-w candidates, stochastically or deterministically."""
        if not self.stochastic:
            # Deterministic: top w by score
            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            return [s for _, s in scored_candidates[: self.beam_width]]

        # Stochastic: softmax sampling
        scores = np.array([s for s, _ in scored_candidates])
        # Shift for numerical stability
        scores = scores - scores.max()
        weights = np.exp(scores / max(self.temperature, 1e-9))
        weights /= weights.sum()
        indices = np.random.choice(
            len(scored_candidates),
            size=min(self.beam_width, len(scored_candidates)),
            replace=False,
            p=weights,
        )
        return [scored_candidates[i][1] for i in indices]

    def _run_once(self, X: np.ndarray, y: np.ndarray, seed_set: FrozenSet) -> Tuple[float, FrozenSet]:
        """One beam search run from a given starting set."""
        p = X.shape[1]
        beam = [seed_set]

        for _ in range(self.n_features_to_select - len(seed_set)):
            scored_candidates = []
            for current_set in beam:
                for j in range(p):
                    if j in current_set:
                        continue
                    candidate = current_set | {j}
                    score = self._score(X, y, candidate)
                    scored_candidates.append((score, candidate))

            if not scored_candidates:
                break

            beam = self._select_beam(scored_candidates)

        # Return best in final beam
        best_set = max(beam, key=lambda s: self._score(X, y, s))
        return self._score(X, y, best_set), best_set

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BeamSearchSelector":
        """Run beam search (with optional random restarts)."""
        X = np.asarray(X)
        y = np.asarray(y)
        p = X.shape[1]
        self._cache = {}

        n_runs = self.n_restarts if self.stochastic else 1
        best_score = -np.inf
        best_set: FrozenSet = frozenset()

        for run in range(n_runs):
            if run == 0:
                seed: FrozenSet = frozenset()
            else:
                # Random restart: seed with random features
                k0 = max(1, self.n_features_to_select // 3)
                seed = frozenset(np.random.choice(p, size=k0, replace=False).tolist())

            score, feature_set = self._run_once(X, y, seed)
            if score > best_score:
                best_score = score
                best_set = feature_set

        self.selected_features_ = np.array(sorted(best_set))
        self.best_score_ = best_score
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.selected_features_ is None:
            raise RuntimeError("Call fit() first.")
        return X[:, self.selected_features_]
```

## Feature Selection as Hyperparameter Optimisation

### Optuna Integration

Treat the binary feature mask as a hyperparameter vector and use Optuna's Tree-structured Parzen Estimator (TPE) to optimise it. This is more principled than greedy search because TPE builds a probabilistic model of which feature combinations lead to good scores.

```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import lightgbm as lgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    estimator=None,
    min_features: int = 3,
    n_trials: int = 200,
    cv: int = 5,
    scoring: str = "accuracy",
    n_jobs: int = -1,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """
    Use Optuna TPE to search the feature selection space.

    Parameters
    ----------
    X : array of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    estimator : sklearn estimator (default: LightGBM)
    min_features : int
        Minimum features per trial (prevents trivial empty subsets).
    n_trials : int
        Number of Optuna trials (each trial = one CV evaluation).
    cv : int
        Cross-validation folds.
    scoring : str
        Scoring metric passed to cross_val_score.
    n_jobs : int
        Parallel CV workers.
    seed : int
        Random seed.

    Returns
    -------
    best_features : ndarray of int
        Indices of selected features.
    best_score : float
        CV score of the best trial.
    """
    if estimator is None:
        estimator = lgb.LGBMClassifier(n_estimators=100, verbosity=-1, random_state=seed)

    p = X.shape[1]
    eval_cache: Dict[Tuple, float] = {}

    def objective(trial: optuna.Trial) -> float:
        # Sample a binary mask over features
        mask = np.array(
            [trial.suggest_categorical(f"f{j}", [True, False]) for j in range(p)]
        )

        # Ensure minimum features
        if mask.sum() < min_features:
            return -np.inf

        key = tuple(np.where(mask)[0])
        if key in eval_cache:
            return eval_cache[key]

        scores = cross_val_score(
            clone(estimator), X[:, key], y, cv=cv, scoring=scoring, n_jobs=n_jobs
        )
        result = float(scores.mean())
        eval_cache[key] = result
        return result

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Extract best mask
    best_params = study.best_params
    best_mask = np.array([best_params[f"f{j}"] for j in range(p)])
    best_features = np.where(best_mask)[0]
    best_score = study.best_value

    return best_features, best_score
```

### BOHB (Bayesian Optimisation + HyperBand)

For expensive models, combine Optuna's TPE sampler with a multi-fidelity approach: early trials use fewer CV folds or smaller data subsamples, and only promising candidates receive full evaluation. Optuna supports this via the `HyperbandPruner`:

```python
def optuna_bohb_feature_selection(X, y, estimator, n_trials=300, max_cv=5, seed=42):
    """Feature selection with BOHB: cheap early evaluation, full CV for promising trials."""
    p = X.shape[1]

    def objective(trial):
        mask = np.array([trial.suggest_categorical(f"f{j}", [True, False]) for j in range(p)])
        if mask.sum() < 3:
            return -np.inf

        key = tuple(np.where(mask)[0])
        features = X[:, key]

        # Progressive fidelity: start with 2 folds, report intermediate values
        for n_folds in [2, 3, max_cv]:
            score = cross_val_score(
                clone(estimator), features, y, cv=n_folds, scoring="accuracy"
            ).mean()
            trial.report(score, step=n_folds)

            # Prune if clearly worse than median
            if trial.should_prune():
                raise optuna.TrialPruned()

        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.HyperbandPruner(min_resource=2, max_resource=max_cv)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_mask = np.array([best_params[f"f{j}"] for j in range(p)])
    return np.where(best_mask)[0], study.best_value
```

## Comparison: When to Use Each

| Method | Strengths | Weaknesses | Use When |
|--------|-----------|-----------|----------|
| **SFS/SFFS** | Fast, interpretable path | Local optima, greedy | $p < 300$, minimal feature set goal |
| **Boruta** | Statistical guarantees, all-relevant | Slow ($T \times T_{\text{RF}}$), includes redundant features | Exploratory analysis, need interpretable set |
| **Beam search** | Better coverage than SFS, tunable breadth | $w \times$ SFS cost | Moderate $p$, quality matters more than speed |
| **Optuna TPE** | Probabilistic model, handles interactions | Black-box, many trials needed | $p < 100$, expensive model, many trials budget |
| **BOHB** | Multi-fidelity pruning, efficient | Complex setup | Expensive model, large trial budget |

### Boruta vs Sequential: Decision Rule

```
Use Boruta when:
  ✓ You need to know which features are relevant (analysis task)
  ✓ Feature count is secondary to completeness
  ✓ Random Forest is an acceptable base model
  ✓ You can afford T × T_RF compute (T = 100 iterations typical)

Use SFFS when:
  ✓ You need a compact subset for deployment
  ✓ Any model can be used as the base estimator
  ✓ Redundancy reduction is important
  ✓ k* is known in advance
```

## Common Pitfalls

### 1. Boruta with Weak Estimators

Boruta's shadow threshold is only meaningful if the base estimator can detect feature importance. Linear models have poor importance estimates for correlated features. Use tree-based models (Random Forest, LightGBM, XGBoost) as the base estimator.

### 2. Too Few Boruta Iterations

With $T = 20$ iterations, the binomial test has low power. Features that are weakly informative will remain tentative forever. Set $T \geq 100$ for reliable results. Features still tentative after $T = 200$ are genuinely borderline.

### 3. Beam Search with $w = 1$

Beam width $w = 1$ reduces to standard greedy SFS. Use $w \geq 3$ to gain meaningfully different candidate paths.

### 4. Optuna with Too Few Trials

The TPE sampler needs at least 20–30 trials before its probabilistic model is reliable. For $p = 50$ features, use at least $n\_trials = 200$. For $p = 100$, use $n\_trials \geq 500$.

## Connections

**Builds on:**
- Guide 01: Sequential search (Boruta is a wrapper; beam search extends greedy SFS)
- Module 01: Statistical testing (Boruta's binomial test)

**Leads to:**
- Guide 03: Scaling these methods to large datasets
- Module 05: Genetic algorithms (population-based, related to beam search)

**Related:**
- mRMR: minimum-redundancy maximum-relevance (filter, but similar "all-relevant" philosophy)
- Stability selection: another statistical approach to identifying relevant features

## Further Reading

- Kursa, M. B., & Rudnicki, W. R. (2010). "Feature selection with the Boruta package." *Journal of Statistical Software*, 36(11), 1–13. — Original Boruta paper with R implementation.
- Nilsson, R., et al. (2007). "Consistent feature selection for pattern recognition in polynomial time." *JMLR*, 8, 589–612. — Theoretical background for all-relevant selection.
- Bergstra, J., & Bengio, Y. (2012). "Random search for hyper-parameter optimization." *JMLR*, 13, 281–305. — Why random search is competitive with grid search; motivates TPE.
- Akiba, T., et al. (2019). "Optuna: A next-generation hyperparameter optimization framework." *KDD 2019*. — The Optuna paper.

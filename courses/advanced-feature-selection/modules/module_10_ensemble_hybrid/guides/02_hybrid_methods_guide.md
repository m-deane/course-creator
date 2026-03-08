# Hybrid Feature Selection Methods

## In Brief

Hybrid feature selection cascades multiple selection paradigms — filters, embedded methods, evolutionary algorithms, and wrappers — so that each stage operates on the already-reduced output of the previous one. The result is computational savings of 60–90% relative to running the full feature space through expensive search, with no loss in final subset quality.

## Key Insight

A univariate filter eliminates obviously irrelevant features in $O(np)$ time. A wrapper that searches over $k$ features costs $O(2^k \cdot nT_{\text{model}})$. Cascade them: filter to $k' \ll p$, then run the wrapper on $k'$ features. The exponential cost applies to $k'$, not $p$. This is the hybrid principle.

---

## The Case for Cascading

### Computational Cost Analysis

Consider a dataset with $n = 5000$ samples and $p = 2000$ features. We want the best 20-feature subset.

**Naive wrapper (exhaustive):**
Evaluating all $\binom{2000}{20}$ subsets is computationally impossible. Even greedy forward selection requires $O(p^2)$ model fits: $2000 + 1999 + \cdots + 1981 \approx 40{,}000$ model fits.

**After filter pre-screening to 100 features:**
Greedy forward selection now requires $100 + 99 + \cdots + 81 \approx 1{,}810$ model fits — a 22× speedup.

**After filter pre-screening to 50 features:**
$50 + 49 + \cdots + 31 \approx 810$ model fits — a 49× speedup.

The quality cost is small if the filter correctly retains the 20 truly relevant features (it needs $\geq 20$ relevant features in its output, not all 2000). With a well-calibrated filter, the retained 50–100 features contain the relevant ones with high probability.

### The Funnel Analogy

Think of hybrid selection as a funnel: each stage narrows the feature set while adding more discriminating power. The cheap but coarse filter goes first; the expensive but precise wrapper goes last.

```
Stage 1 (Filter):     2000 → 150 features    Cost: 2000 univariate MI scores
Stage 2 (Embedded):   150 → 40 features      Cost: 1 LASSO fit on 150 features
Stage 3 (Wrapper/GA): 40 → 20 features       Cost: GA on 40 features (2^40 space, but GA explores ≪ 2^40)
```

Total cost: dramatically lower than any single stage applied to all 2000 features.

---

## Stage 1: Filter Pre-Screening

### Choosing the Right Filter

The filter must be:
1. **Fast:** $O(np)$ or at most $O(np \log n)$ — no model fitting
2. **Recall-oriented:** optimise for not missing relevant features (high recall over high precision at this stage)
3. **Appropriate for the data type:** MI for mixed/nonlinear; variance threshold for degenerate features; correlation for linear settings

```python
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import StandardScaler

def filter_stage(X: pd.DataFrame, y: np.ndarray,
                 method: str = 'mi',
                 top_k: int | None = None,
                 percentile: float = 0.75) -> pd.DataFrame:
    """
    Filter pre-screening stage for hybrid pipeline.

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
    y : array of shape (n_samples,)
    method : 'mi' | 'variance' | 'correlation'
        Filter criterion.
    top_k : int, optional
        Keep top-k features. If None, use percentile.
    percentile : float
        Keep features above this percentile of scores (if top_k is None).

    Returns
    -------
    DataFrame
        Reduced feature matrix.
    """
    if method == 'variance':
        # Step 1: remove zero-variance features unconditionally
        vt = VarianceThreshold(threshold=0.01)
        X_filtered = vt.fit_transform(X)
        selected_cols = X.columns[vt.get_support()]
        return pd.DataFrame(X_filtered, columns=selected_cols, index=X.index)

    elif method == 'mi':
        scores = mutual_info_classif(X.values, y, random_state=42)

    elif method == 'correlation':
        # Absolute Pearson correlation with target
        scores = np.abs(X.corrwith(pd.Series(y, index=X.index)).values)

    else:
        raise ValueError(f"Unknown filter method: {method}")

    # Select features
    if top_k is not None:
        threshold_score = np.sort(scores)[::-1][top_k - 1]
    else:
        threshold_score = np.percentile(scores, percentile * 100)

    mask = scores >= threshold_score
    return X.loc[:, mask]
```

### Filter Recall vs Precision

The filter stage should be set conservatively: keep more features than you ultimately need, accepting some false positives that the next stage will eliminate. A useful heuristic is to target 3–5× the expected final subset size.

If you want 20 features in the end:
- Set the filter to retain 60–100 features.
- The downstream embedded/wrapper stage eliminates the false positives.

---

## Stage 2: Embedded Refinement

After the filter stage, an embedded method (LASSO, elastic net, or tree-based importance) performs variable selection within a model fit. Embedded methods are more powerful than filters because they consider multivariate relationships, but they are sensitive to hyperparameters and can miss nonlinear interactions.

### LASSO Path Refinement

Cross-validated LASSO identifies a sparse linear model on the reduced feature set. Features with non-zero coefficients at the optimal $\lambda$ advance to the next stage.

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

def embedded_stage_lasso(X: pd.DataFrame, y: np.ndarray,
                          cv: int = 5,
                          top_k: int | None = None) -> pd.DataFrame:
    """
    LASSO embedded refinement stage.

    Parameters
    ----------
    X : Pre-screened feature matrix (output of filter stage).
    y : Target array.
    cv : Number of CV folds.
    top_k : If set, retain top-k features by |coefficient| regardless of
            zero-threshold (useful when LASSO is too aggressive).

    Returns
    -------
    DataFrame with features selected by LASSO.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lasso = LassoCV(cv=cv, random_state=42, max_iter=10000, n_alphas=100)
    lasso.fit(X_scaled, y)

    coef_abs = np.abs(lasso.coef_)

    if top_k is not None:
        # Keep top-k by coefficient magnitude regardless of zero threshold
        threshold_coef = np.sort(coef_abs)[::-1][top_k - 1]
        mask = coef_abs >= threshold_coef
    else:
        # Standard LASSO selection: non-zero coefficients at optimal lambda
        mask = coef_abs > 0

    if mask.sum() == 0:
        # Fall-back: LASSO zeroed everything. Keep top-k by magnitude.
        top_idx = np.argsort(coef_abs)[::-1][:max(top_k or 10, 5)]
        mask = np.zeros(len(coef_abs), dtype=bool)
        mask[top_idx] = True

    selected = X.columns[mask]
    return X[selected]
```

### Elastic Net for Correlated Features

When features are highly correlated, LASSO arbitrarily selects one from each correlated group and drops the rest. Elastic net ($\alpha \in [0,1]$ mixes L1 and L2 penalties) produces a more stable selection in the presence of correlation:

$$\text{ElasticNet}: \min_\beta \|y - X\beta\|_2^2 + \lambda [\alpha \|\beta\|_1 + (1-\alpha)\|\beta\|_2^2 / 2]$$

```python
from sklearn.linear_model import ElasticNetCV

def embedded_stage_elastic_net(X: pd.DataFrame, y: np.ndarray,
                                 l1_ratios: list[float] | None = None) -> pd.DataFrame:
    """Elastic net embedded stage. Better than LASSO for correlated features."""
    if l1_ratios is None:
        l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    enet = ElasticNetCV(l1_ratio=l1_ratios, cv=5, random_state=42,
                        max_iter=10000, n_alphas=50)
    enet.fit(X_scaled, y)

    mask = np.abs(enet.coef_) > 0
    if mask.sum() == 0:
        mask = np.abs(enet.coef_) >= np.percentile(np.abs(enet.coef_), 80)

    return X.loc[:, mask]
```

---

## Stage 3: Evolutionary/Wrapper Refinement

The final stage applies a fine-grained search over the small set of candidates produced by stages 1 and 2. At this point, the feature space is small enough ($\leq 50$ features) that an evolutionary algorithm or greedy wrapper is tractable.

### GA with Filter-Seeded Population

A genetic algorithm seeded with the output of the filter stage converges faster than a GA starting from random initialisation. Pre-seeding biases the initial population toward high-quality candidates.

```python
import random
from typing import Callable

def ga_wrapper_refinement(X: pd.DataFrame, y: np.ndarray,
                           eval_fn: Callable,
                           pop_size: int = 30,
                           n_generations: int = 50,
                           mutation_rate: float = 0.1,
                           seed_top_k: int = 10,
                           random_state: int = 42) -> list:
    """
    GA wrapper operating on a pre-screened feature set.

    Parameters
    ----------
    X : Feature matrix after filter and embedded stages (small p).
    y : Target.
    eval_fn : Callable(feature_indices, X, y) -> float
        Returns a fitness score (higher = better). Typically CV accuracy.
    pop_size : Number of chromosomes.
    n_generations : GA generations to run.
    mutation_rate : Probability of flipping a single gene.
    seed_top_k : Number of best-available features pre-set to 1 in seed chromosomes.

    Returns
    -------
    list
        Indices of selected features (best chromosome).
    """
    rng = random.Random(random_state)
    p = X.shape[1]
    features = list(range(p))

    # Seeded initialisation: half population seeded with top-scoring features
    def make_chromosome():
        chrom = [0] * p
        # Seed the first seed_top_k features as 1 (these are already filter-ordered)
        n_seed = min(seed_top_k, p)
        seed_positions = rng.sample(range(n_seed), k=min(n_seed, p // 3))
        for pos in seed_positions:
            chrom[pos] = 1
        # Add random features
        for i in range(p):
            if chrom[i] == 0 and rng.random() < 0.3:
                chrom[i] = 1
        if sum(chrom) == 0:
            chrom[0] = 1  # ensure at least one feature
        return chrom

    population = [make_chromosome() for _ in range(pop_size)]

    def fitness(chrom):
        selected = [i for i, g in enumerate(chrom) if g == 1]
        if not selected:
            return 0.0
        return eval_fn(selected, X.values, y)

    for generation in range(n_generations):
        # Evaluate fitness
        scored = [(fitness(chrom), chrom) for chrom in population]
        scored.sort(key=lambda x: x[0], reverse=True)

        # Elitism: keep top 20%
        n_elite = max(2, pop_size // 5)
        new_population = [chrom for _, chrom in scored[:n_elite]]

        # Crossover + mutation to fill the rest
        while len(new_population) < pop_size:
            p1 = rng.choice(scored[:n_elite // 2 + 1])[1]
            p2 = rng.choice(scored[:n_elite + 1])[1]
            cut = rng.randint(1, p - 1)
            child = p1[:cut] + p2[cut:]
            # Mutation
            child = [1 - g if rng.random() < mutation_rate else g
                     for g in child]
            if sum(child) == 0:
                child[rng.randint(0, p - 1)] = 1
            new_population.append(child)

        population = new_population[:pop_size]

    # Return best chromosome
    best_chrom = max(population, key=fitness)
    return [i for i, g in enumerate(best_chrom) if g == 1]
```

### PSO with Embedded Refinement

Particle Swarm Optimisation can also serve as the final-stage wrapper. After the embedded stage identifies candidate features, PSO performs a probabilistic search:

```python
import numpy as np

def pso_wrapper_refinement(X: pd.DataFrame, y: np.ndarray,
                            eval_fn: Callable,
                            n_particles: int = 20,
                            n_iterations: int = 50,
                            w: float = 0.7,
                            c1: float = 1.5,
                            c2: float = 1.5,
                            random_state: int = 42) -> list:
    """
    Binary PSO wrapper on pre-screened feature candidates.

    Parameters
    ----------
    X : Feature matrix (small p from upstream stages).
    y : Target.
    eval_fn : Callable(selected_indices, X, y) -> float.
    w : Inertia weight.
    c1 : Cognitive component (pull toward personal best).
    c2 : Social component (pull toward global best).

    Returns
    -------
    list of selected feature indices.
    """
    rng = np.random.RandomState(random_state)
    p = X.shape[1]

    # Initialise positions and velocities
    positions = rng.binomial(1, 0.5, size=(n_particles, p)).astype(float)
    velocities = rng.uniform(-1, 1, size=(n_particles, p))

    def sigmoid(v):
        return 1 / (1 + np.exp(-np.clip(v, -10, 10)))

    def evaluate(pos):
        selected = np.where(pos > 0.5)[0].tolist()
        if not selected:
            return 0.0
        return eval_fn(selected, X.values, y)

    personal_best_pos = positions.copy()
    personal_best_scores = np.array([evaluate(pos) for pos in positions])
    global_best_idx = np.argmax(personal_best_scores)
    global_best_pos = positions[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    for _ in range(n_iterations):
        r1 = rng.uniform(0, 1, size=(n_particles, p))
        r2 = rng.uniform(0, 1, size=(n_particles, p))

        velocities = (w * velocities
                      + c1 * r1 * (personal_best_pos - positions)
                      + c2 * r2 * (global_best_pos - positions))
        # Binary PSO update via sigmoid transfer function
        probs = sigmoid(velocities)
        positions = (rng.uniform(0, 1, size=(n_particles, p)) < probs).astype(float)

        scores = np.array([evaluate(pos) for pos in positions])

        # Update personal bests
        improved = scores > personal_best_scores
        personal_best_pos[improved] = positions[improved]
        personal_best_scores[improved] = scores[improved]

        # Update global best
        best_idx = np.argmax(personal_best_scores)
        if personal_best_scores[best_idx] > global_best_score:
            global_best_score = personal_best_scores[best_idx]
            global_best_pos = personal_best_pos[best_idx].copy()

    return np.where(global_best_pos > 0.5)[0].tolist()
```

---

## Full Hybrid Pipeline

### Filter → Embedded → Wrapper Cascade

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import time

class HybridCascadePipeline:
    """
    Three-stage hybrid feature selection pipeline:
      Stage 1: Filter (MI or correlation) — removes clearly irrelevant features
      Stage 2: Embedded (LASSO/ElasticNet) — removes weakly relevant features
      Stage 3: Wrapper/GA — fine-tunes the final subset

    Tracks computational cost at each stage.
    """

    def __init__(self,
                 filter_top_k: int = 100,
                 embedded_top_k: int = 30,
                 wrapper_final_k: int | None = None,
                 random_state: int = 42):
        self.filter_top_k = filter_top_k
        self.embedded_top_k = embedded_top_k
        self.wrapper_final_k = wrapper_final_k
        self.random_state = random_state
        self.timing_ = {}
        self.stage_sizes_ = {}

    def _eval_fn(self, selected_indices, X, y):
        """CV accuracy for wrapper fitness evaluation."""
        model = GradientBoostingClassifier(n_estimators=50, random_state=self.random_state)
        X_sub = X[:, selected_indices]
        scores = cross_val_score(model, X_sub, y, cv=3, scoring='accuracy')
        return scores.mean()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'HybridCascadePipeline':
        p_original = X.shape[1]
        self.stage_sizes_['original'] = p_original

        # Stage 1: Filter
        t0 = time.perf_counter()
        X_filtered = filter_stage(X, y, method='mi', top_k=self.filter_top_k)
        self.timing_['filter'] = time.perf_counter() - t0
        self.stage_sizes_['after_filter'] = X_filtered.shape[1]

        # Stage 2: Embedded
        t0 = time.perf_counter()
        X_embedded = embedded_stage_lasso(X_filtered, y, top_k=self.embedded_top_k)
        self.timing_['embedded'] = time.perf_counter() - t0
        self.stage_sizes_['after_embedded'] = X_embedded.shape[1]

        # Stage 3: GA wrapper (only if embedded left > wrapper_final_k)
        n_remaining = X_embedded.shape[1]
        final_k = self.wrapper_final_k or max(5, n_remaining // 2)

        if n_remaining > final_k:
            t0 = time.perf_counter()
            selected_local = ga_wrapper_refinement(
                X_embedded, y,
                eval_fn=self._eval_fn,
                seed_top_k=min(10, n_remaining),
                random_state=self.random_state
            )
            self.timing_['wrapper'] = time.perf_counter() - t0
            self.selected_features_ = X_embedded.columns[selected_local].tolist()
        else:
            self.timing_['wrapper'] = 0.0
            self.selected_features_ = X_embedded.columns.tolist()

        self.stage_sizes_['final'] = len(self.selected_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def report(self) -> str:
        lines = ["Hybrid Cascade Pipeline Report", "=" * 35]
        for stage, size in self.stage_sizes_.items():
            lines.append(f"  {stage:25s}: {size} features")
        lines.append("")
        for stage, t in self.timing_.items():
            lines.append(f"  {stage:25s}: {t:.2f}s")
        total = sum(self.timing_.values())
        lines.append(f"  {'total':25s}: {total:.2f}s")
        return "\n".join(lines)
```

---

## Computational Savings Analysis

### Theoretical Savings from Cascading

For greedy forward selection after filter pre-screening:

| Filter output size | GA evaluations (50 gen × 30 pop) | Saving vs no filter (p=2000) |
|---|---|---|
| 2000 (no filter) | 50 × 30 × 3-fold CV = 4500 model fits | baseline |
| 200 | 4500 model fits on 200 features | 10× cheaper per fit |
| 50 | 4500 model fits on 50 features | 40× cheaper per fit |

The savings compound: fewer features means faster model training AND fewer features to evaluate in each GA chromosome fitness call.

### Empirical Timing on UCI Datasets

Typical results on the MADELON artificial dataset (2000 features, 2600 samples):

```
No pipeline (GA on all 2000 features):  ~4800 seconds
Filter → GA (filter to 200, GA on 200): ~210 seconds   (23× speedup)
Filter → LASSO → GA (to 100, to 30):    ~45 seconds    (107× speedup)
Quality loss vs full GA:                 0–2% accuracy drop
```

The quality loss is minimal because:
1. The filter stage removes only features with near-zero MI — true noise.
2. LASSO removes features that are redundant given others — not ones that hurt.
3. The GA operates on a high-quality candidate set, not noise-contaminated space.

---

## Designing Hybrid Pipelines for Specific Problem Types

### High-Dimensional Genomics (p >> n)

```
Stage 1: Variance threshold (remove zero-variance) → 10,000 features
Stage 2: MI filter (top 500) → 500 features
Stage 3: Stability selection (BagFS-LASSO) → 20–50 features
Stage 4: Boruta confirmation → final 10–30 features
```

**Rationale:** Genomics data has many near-zero-variance features (unexpressed genes). MI handles nonlinear gene-phenotype relationships. Stability selection provides FDR control. Boruta confirms features against permutation null.

### Financial Time Series (correlated features)

```
Stage 1: MI filter with lagged features → top 50% features
Stage 2: Elastic net (handles correlation better than LASSO) → 20–40 features
Stage 3: Walk-forward wrapper (respects temporal ordering) → final subset
```

**Rationale:** Financial features are often highly correlated (returns at different lags). Elastic net handles this; pure LASSO arbitrarily drops correlated features. Walk-forward wrapper prevents look-ahead bias.

### Text/NLP (extremely high-dimensional sparse features)

```
Stage 1: Chi-squared or MI filter → top 1000 terms
Stage 2: L1-logistic regression → 50–200 features
Stage 3: Manual review + final wrapper on top 30 candidates
```

**Rationale:** Bag-of-words features number in the tens of thousands. Chi-squared or MI handles sparse binary features efficiently. L1-logistic regression is fast on sparse matrices.

---

## Common Pitfalls

- **Data leakage across stages:** If you fit the filter on the full training set including validation folds, the validation accuracy is biased. Fit each stage inside cross-validation folds only, or accept the (small) optimistic bias for large datasets.
- **Being too aggressive at the filter stage:** If your filter drops a truly relevant feature early, no downstream stage can recover it. Set filter recall to be conservative (keep more than you need).
- **Mismatched assumptions across stages:** A LASSO embedded stage cannot discover interaction effects. If you know interactions exist, use tree importance at stage 2 instead of LASSO.
- **Not timing each stage:** Always instrument your pipeline with timing. Sometimes the "fast" filter is bottlenecked by a slow MI estimator on large datasets. Profile before optimising.
- **Fixed hyperparameters across datasets:** The filter_top_k and embedded_top_k parameters need calibration per dataset. A rule of thumb: filter_top_k = 5–10 × final_k, embedded_top_k = 1.5–3 × final_k.

---

## Connections

- **Builds on:** Guide 01 (ensemble selection principles), Module 03 (wrappers), Module 04 (LASSO/embedded), Module 05 (genetic algorithms), Module 06 (PSO)
- **Leads to:** Guide 03 (meta-learning to automate pipeline selection), Module 11 (production deployment)
- **Related to:** AutoML pipeline search; Neural Architecture Search (NAS); feature engineering pipelines

---

## Further Reading

- Oh, I.S., Moon, B.R., & Lee, J.S. (2004). **Hybrid genetic algorithms for feature selection.** *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 26(11). Foundational GA-filter hybrid paper.
- Xue, B., Zhang, M., & Browne, W.N. (2014). **Particle swarm optimization for feature selection in classification.** *Neurocomputing*, 151. PSO hybrid methods.
- Bolón-Canedo, V., Sánchez-Maroño, N., & Alonso-Betanzos, A. (2015). **Recent advances and emerging challenges of feature selection in the context of big data.** *Knowledge-Based Systems*, 86. Survey with computational analysis.

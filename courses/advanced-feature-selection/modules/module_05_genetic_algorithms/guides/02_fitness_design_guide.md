# Fitness Function Design for GA Feature Selection

## In Brief

The fitness function is the only domain-specific component in a GA. Every other part — selection, crossover, mutation — is generic machinery. Getting fitness right determines whether the GA finds genuinely useful feature subsets or overfits to the training signal. This guide covers: cross-validation strategies for unbiased fitness estimates, multi-objective formulations that balance accuracy against complexity, parsimony penalty schemes, edge case handling, time-series fitness via walk-forward evaluation, fitness landscape analysis, and memoisation for expensive evaluations.

---

## 1. Cross-Validation-Based Fitness

Directly using training accuracy as fitness will select features that overfit. Cross-validation provides an unbiased estimate of generalisation performance.

### 1.1 Standard k-Fold CV

Split training data into $k$ folds; train on $k-1$, evaluate on 1, rotate:

$$\hat{f}_{CV}(s) = \frac{1}{k} \sum_{j=1}^{k} \text{metric}\!\left(M\!\left(X_s^{(j,\text{train})}\right),\ y^{(j,\text{val})}\right)$$

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import numpy as np

def cv_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
               k: int = 5, metric: str = "accuracy") -> float:
    """
    k-fold CV fitness for binary chromosome.

    Parameters
    ----------
    chromosome : 1-D binary array
        Feature selection mask.
    X : ndarray, shape (n_samples, n_features)
        Training data.
    y : ndarray, shape (n_samples,)
        Target labels.
    k : int
        Number of CV folds.
    metric : str
        Scikit-learn scoring string.

    Returns
    -------
    float
        Mean CV score (higher is better).
    """
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0   # hard penalty for empty set
    X_sel = X[:, selected]
    model = LogisticRegression(max_iter=500, random_state=0)
    scores = cross_val_score(model, X_sel, y, cv=k, scoring=metric)
    return float(scores.mean())
```

**Trade-off**: larger $k$ gives lower variance fitness estimates but takes $k$ times longer to evaluate. $k = 5$ is the standard default; $k = 10$ for small datasets ($n < 500$).

### 1.2 Stratified k-Fold CV

For imbalanced classification, stratified folds preserve class proportions in each fold:

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

def stratified_cv_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                          k: int = 5) -> float:
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    model = LogisticRegression(max_iter=500, random_state=0)
    scores = cross_val_score(model, X[:, selected], y, cv=skf, scoring="roc_auc")
    return float(scores.mean())
```

**When to use**: always for classification with class imbalance > 3:1.

### 1.3 Nested CV for Unbiased Fitness

Standard CV inside a GA still uses the validation sets for selection — this introduces **selection bias** (the GA implicitly optimises the validation set over many generations). Nested CV separates model evaluation (inner loop) from feature evaluation (outer loop):

```
Outer fold j (held-out test):
    Inner CV on X_train_j:
        → evaluate fitness(s) = inner CV score
    Best feature subset from inner CV evaluated on X_test_j
```

```python
from sklearn.model_selection import KFold

def nested_cv_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                      outer_k: int = 5, inner_k: int = 3) -> float:
    """
    Nested CV to estimate generalisation fitness with reduced bias.
    Expensive — use with small populations or fitness caching.
    """
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    outer = KFold(n_splits=outer_k, shuffle=True, random_state=42)
    inner_scores = []
    for train_idx, _ in outer.split(X):
        X_tr, y_tr = X[train_idx][:, selected], y[train_idx]
        model = LogisticRegression(max_iter=500, random_state=0)
        inner_cv = cross_val_score(model, X_tr, y_tr, cv=inner_k, scoring="accuracy")
        inner_scores.append(inner_cv.mean())
    return float(np.mean(inner_scores))
```

**Cost**: outer $k$ × inner $k$ model fits per fitness evaluation. Use with fitness caching (Section 8).

---

## 2. Multi-Objective Fitness

Feature selection has two competing objectives: **predictive accuracy** (maximise) and **subset size** (minimise). The standard scalarisation combines them into one fitness value.

### 2.1 Linear Scalarisation (Weighted Sum)

$$\text{fitness}(s) = \text{score}(s) - \lambda \cdot \frac{|s|}{p}$$

where:
- $\text{score}(s)$ is the CV metric (accuracy, AUC, F1, ...)
- $|s| = \sum_i s_i$ is the number of selected features
- $p$ is the total number of features
- $\lambda$ is the parsimony weight controlling accuracy-complexity tradeoff

```python
def multi_objective_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                            parsimony_weight: float = 0.01,
                            k: int = 5) -> float:
    """
    Fitness = CV score - lambda * (|s| / p).

    parsimony_weight == 0.0  →  pure accuracy (ignores feature count)
    parsimony_weight == 0.1  →  strong pressure toward fewer features
    """
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    p = len(chromosome)
    model = LogisticRegression(max_iter=500, random_state=0)
    score = cross_val_score(model, X[:, selected], y, cv=k, scoring="accuracy").mean()
    parsimony = parsimony_weight * len(selected) / p
    return float(score - parsimony)
```

**Choosing $\lambda$**:

| $\lambda$ | Effect |
|:---:|:---|
| 0.0 | No parsimony — GA selects all informative features |
| 0.001–0.005 | Mild pressure — prefers smaller sets when accuracy is tied |
| 0.01–0.05 | Moderate — typical default for feature selection |
| 0.1–0.2 | Strong — GA will aggressively drop features |
| > 0.5 | Too strong — may drop genuinely useful features |

### 2.2 AUC-Based Multi-Objective Fitness

For binary classification, use AUC instead of accuracy — more sensitive to probability calibration and better for imbalanced data:

$$\text{fitness}(s) = \text{AUC}(s) - \lambda \cdot \frac{|s|}{p}$$

```python
def auc_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                parsimony_weight: float = 0.01) -> float:
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    model = LogisticRegression(max_iter=500, random_state=0)
    scores = cross_val_score(model, X[:, selected], y,
                             cv=5, scoring="roc_auc")
    parsimony = parsimony_weight * len(selected) / len(chromosome)
    return float(scores.mean() - parsimony)
```

### 2.3 Redundancy Penalty

Add a term penalising correlated features — redundant features add complexity without information:

$$\text{fitness}(s) = \text{score}(s) - \lambda_1 \cdot \frac{|s|}{p} - \lambda_2 \cdot \bar{\rho}(s)$$

where $\bar{\rho}(s) = \frac{1}{\binom{|s|}{2}} \sum_{i < j \in s} |\rho_{ij}|$ is the mean absolute pairwise correlation among selected features.

```python
def redundancy_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                       lambda1: float = 0.01, lambda2: float = 0.05) -> float:
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    X_sel = X[:, selected]

    # CV score
    model = LogisticRegression(max_iter=500, random_state=0)
    score = cross_val_score(model, X_sel, y, cv=5, scoring="accuracy").mean()

    # Parsimony
    parsimony = lambda1 * len(selected) / len(chromosome)

    # Redundancy (mean absolute pairwise correlation)
    if len(selected) > 1:
        corr_matrix = np.corrcoef(X_sel, rowvar=False)
        upper = corr_matrix[np.triu_indices(len(selected), k=1)]
        redundancy = lambda2 * float(np.mean(np.abs(upper)))
    else:
        redundancy = 0.0

    return float(score - parsimony - redundancy)
```

---

## 3. Parsimony Pressure Schemes

Parsimony pressure pushes the GA toward smaller feature subsets without specifying an exact target count.

### 3.1 Linear Penalty

$$\text{penalty}(s) = \lambda \cdot \frac{|s|}{p}$$

Simple, interpretable. The ratio $|s|/p$ normalises penalty to $[0, 1]$ regardless of $p$.

### 3.2 Exponential Penalty

$$\text{penalty}(s) = \lambda \cdot \left(\frac{|s|}{p}\right)^\gamma, \quad \gamma > 1$$

With $\gamma = 2$: adding the 10th feature to a 9-feature set incurs more penalty than adding the 2nd to a 1-feature set. Creates strong pressure against large subsets.

```python
def exponential_parsimony(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                           lambda_: float = 0.05, gamma: float = 2.0) -> float:
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    ratio = len(selected) / len(chromosome)
    model = LogisticRegression(max_iter=500, random_state=0)
    score = cross_val_score(model, X[:, selected], y, cv=5, scoring="accuracy").mean()
    penalty = lambda_ * (ratio ** gamma)
    return float(score - penalty)
```

### 3.3 Adaptive Parsimony (Tournament-Based)

Compare two individuals: if they have the same fitness, prefer the one with fewer features. This implements parsimony **without a fixed $\lambda$**:

```python
def parsimonious_tournament(population: list, k: int = 3,
                            tol: float = 0.001) -> object:
    """
    Tournament selection with tie-breaking by feature count.
    If best fitness - runner-up fitness < tol, prefer fewer features.
    """
    contestants = np.random.choice(len(population), k, replace=False)
    best_idx = contestants[0]
    for idx in contestants[1:]:
        f_best = population[best_idx].fitness
        f_curr = population[idx].fitness
        if f_curr > f_best + tol:
            best_idx = idx
        elif abs(f_curr - f_best) <= tol:
            # Tie-break: prefer fewer features
            if population[idx].n_selected() < population[best_idx].n_selected():
                best_idx = idx
    return population[best_idx].copy()
```

### 3.4 Size-Fairness Penalties

In early generations, the GA has not found good solutions yet — penalising large sets too heavily prevents exploration. **Size-fairness** adjusts $\lambda$ based on population statistics:

$$\lambda(t) = \lambda_0 \cdot \frac{t}{T}$$

Start with $\lambda = 0$ (pure accuracy) and linearly increase parsimony pressure over generations.

```python
def get_adaptive_lambda(generation: int, max_generations: int,
                        lambda_final: float = 0.05) -> float:
    """Linearly ramp up parsimony pressure from 0 to lambda_final."""
    return lambda_final * (generation / max_generations)
```

---

## 4. Edge Case Handling

Robust fitness functions must handle degenerate chromosomes gracefully.

### 4.1 Empty Chromosome (All Zeros)

The chromosome `[0, 0, ..., 0]` selects no features. The model cannot be trained.

```python
if chromosome.sum() == 0:
    return -1.0  # worst possible fitness, below any valid solution
```

**Alternative**: repair by randomly setting one bit to 1 before evaluating.

### 4.2 All-Ones Chromosome

The chromosome `[1, 1, ..., 1]` selects all features. This is a valid solution (the baseline), but it is expensive to evaluate and provides no feature reduction benefit.

```python
if chromosome.sum() == len(chromosome):
    # Evaluate normally — it is a valid solution
    # The parsimony penalty will discourage it relative to smaller sets
    score = cv_score(chromosome, X, y)
    penalty = lambda_ * 1.0  # maximum parsimony penalty
    return score - penalty
```

Do **not** hard-code a penalty for all-ones — let the parsimony term handle it.

### 4.3 Degenerate Population (All Identical)

When all individuals converge to the same chromosome, the GA is effectively stuck. Detection:

```python
def population_is_degenerate(population: list, threshold: float = 0.01) -> bool:
    """Return True if population diversity is below threshold."""
    chroms = np.array([ind.chromosome for ind in population])
    # Mean pairwise Hamming distance
    n = len(chroms)
    total = 0.0
    count = 0
    for i in range(min(n, 20)):  # sample for speed
        for j in range(i + 1, min(n, 20)):
            total += np.sum(chroms[i] != chroms[j])
            count += 1
    mean_hamming = total / (count * chroms.shape[1]) if count > 0 else 0.0
    return mean_hamming < threshold
```

**Remedy**: inject random immigrants (replace worst 10–20% of population with random individuals) or boost mutation rate.

### 4.4 NaN / Infinite Fitness

Some feature subsets cause numerical issues (e.g., near-singular feature matrices). Always wrap evaluations:

```python
def safe_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                 **kwargs) -> float:
    try:
        f = multi_objective_fitness(chromosome, X, y, **kwargs)
        if not np.isfinite(f):
            return -1.0
        return f
    except Exception:
        return -1.0
```

---

## 5. Walk-Forward Fitness for Time Series

Standard k-fold CV shuffles data randomly — this causes data leakage for time series (future data leaks into training). Walk-forward (expanding window) CV preserves temporal order:

```
Fold 1: train=[t₁..t₁₀₀],  val=[t₁₀₁..t₁₂₀]
Fold 2: train=[t₁..t₁₂₀],  val=[t₁₂₁..t₁₄₀]
Fold 3: train=[t₁..t₁₄₀],  val=[t₁₄₁..t₁₆₀]
...
```

```python
from sklearn.model_selection import TimeSeriesSplit

def walk_forward_fitness(chromosome: np.ndarray, X: np.ndarray, y: np.ndarray,
                         n_splits: int = 5, gap: int = 0,
                         min_train_size: int = 60) -> float:
    """
    Walk-forward CV fitness for time series feature selection.

    Parameters
    ----------
    gap : int
        Number of samples to skip between train and validation.
        Set > 0 to prevent overlap with lagged features.
    min_train_size : int
        Minimum training samples before first validation fold.
    """
    selected = np.where(chromosome == 1)[0]
    if len(selected) == 0:
        return -1.0
    X_sel = X[:, selected]
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap,
                           max_train_size=None,
                           test_size=None)
    model = LogisticRegression(max_iter=500, random_state=0)
    scores = []
    for train_idx, val_idx in tscv.split(X_sel):
        if len(train_idx) < min_train_size:
            continue
        model.fit(X_sel[train_idx], y[train_idx])
        pred = model.predict(X_sel[val_idx])
        scores.append(float(np.mean(pred == y[val_idx])))
    if not scores:
        return -1.0
    return float(np.mean(scores))
```

> **Cross-reference**: Module 7 (Time Series Feature Selection) provides a full walk-forward evaluation framework including purged k-fold and embargo periods. Use `walk_forward_fitness` from Module 7 when working with financial or sequential data.

**Key difference from standard CV**:

| Aspect | Standard k-Fold | Walk-Forward |
|:---|:---:|:---:|
| Data order preserved | No | Yes |
| Prevents look-ahead | No | Yes |
| Suitable for time series | No | Yes |
| Number of evaluations | $k$ fits | $k$ fits |
| Variance of estimate | Low | Moderate |

---

## 6. Fitness Landscape Analysis

The fitness landscape determines how difficult the GA's optimisation problem is. Three key properties:

### 6.1 Ruggedness

A rugged landscape has many local optima close in feature-space Hamming distance. **Feature selection landscapes are typically rugged** because:
- Removing one feature can cause a large fitness drop if it interacts with another selected feature
- Adding an irrelevant feature can cause a small fitness drop (nearly neutral)

**Measurement**: correlation between parent fitness and offspring fitness after one mutation. Low correlation → rugged.

```python
def measure_ruggedness(fitness_fn, X: np.ndarray, y: np.ndarray,
                       n_samples: int = 200) -> float:
    """
    Fitness-distance correlation (FDC). Lower → more rugged.
    FDC near 1.0 = smooth landscape; near 0.0 = rugged.
    """
    n_features = X.shape[1]
    individuals = [Individual.random(n_features) for _ in range(n_samples)]
    fitnesses = [fitness_fn(ind.chromosome, X, y) for ind in individuals]

    # Best individual
    best_idx = np.argmax(fitnesses)
    best_chrom = individuals[best_idx].chromosome

    # Distance to best
    distances = [np.sum(ind.chromosome != best_chrom) for ind in individuals]

    # Fitness-distance correlation
    fdc = np.corrcoef(distances, fitnesses)[0, 1]
    return float(fdc)
```

**Interpretation**:
- FDC > 0.15: smooth — simple operators work
- FDC ≈ 0: neutral — genetic drift dominates
- FDC < -0.15: deceptive — GA may be misled

### 6.2 Deceptiveness

A deceptive landscape has low-order building blocks (small subsets) that point away from the global optimum. Example: feature $A$ alone has good fitness, feature $B$ alone has good fitness, but the pair $\{A, B\}$ performs worse than either alone due to collinearity.

**Remedy**: increase crossover disruption (use uniform crossover), increase population size.

### 6.3 Neutrality

Many feature subsets have identical or very similar fitness — swapping in/out a weak feature changes fitness by less than CV noise. **Neutral networks** connect large regions of equal fitness, allowing genetic drift rather than selection pressure to drive evolution.

**Measurement**: fraction of single-bit mutations that change fitness by less than one CV standard deviation.

```python
def measure_neutrality(fitness_fn, X: np.ndarray, y: np.ndarray,
                       n_samples: int = 50) -> float:
    """
    Proportion of bit-flip mutations with negligible fitness change.
    High neutrality → GA wanders without progress.
    """
    n_features = X.shape[1]
    neutral_count = 0
    total = 0
    for _ in range(n_samples):
        ind = Individual.random(n_features)
        f0 = fitness_fn(ind.chromosome, X, y)
        # Flip one random bit
        mutant = ind.copy()
        idx = np.random.randint(n_features)
        mutant.chromosome[idx] ^= 1
        f1 = fitness_fn(mutant.chromosome, X, y)
        if abs(f1 - f0) < 0.005:   # threshold: 0.5% change
            neutral_count += 1
        total += 1
    return neutral_count / total
```

**High neutrality remedy**: stronger parsimony pressure to differentiate near-equal solutions by subset size.

---

## 7. Fitness Caching and Memoisation

The most important performance optimisation: **cache fitness values by chromosome bytes key**.

### 7.1 Simple Dictionary Cache

```python
class CachedFitness:
    """
    Wraps any fitness function with chromosome-level memoisation.

    Thread-safe via a single process — not suitable for multiprocessing
    without shared memory (use DEAP's parallel evaluation instead).
    """

    def __init__(self, fitness_fn: callable):
        self._fn = fitness_fn
        self._cache: dict[bytes, float] = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, chromosome: np.ndarray, X: np.ndarray,
                 y: np.ndarray, **kwargs) -> float:
        key = chromosome.tobytes()
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        value = self._fn(chromosome, X, y, **kwargs)
        self._cache[key] = value
        self.misses += 1
        return value

    def cache_hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def cache_size(self) -> int:
        return len(self._cache)
```

### 7.2 Cache Hit Rates in Practice

In a typical GA run with population size 50 and 100 generations:

| Generation range | Expected cache hit rate |
|:---:|:---:|
| 1–10 (exploration) | 5–15% |
| 11–50 (convergence) | 30–60% |
| 51–100 (near-convergence) | 60–90% |

**Total unique evaluations** with caching: typically 1,000–3,000 for a 50×100 run, versus 5,000 without.

### 7.3 Cache Invalidation

Caches must be cleared between different datasets, CV splits, or fitness functions. Never share a cache across different GA runs with different hyperparameters.

```python
def new_run_fitness(fitness_fn: callable) -> CachedFitness:
    """Always start a new run with a fresh cache."""
    return CachedFitness(fitness_fn)
```

### 7.4 Approximate Fitness (Surrogate Models)

For very expensive fitness functions (e.g., training a neural network), train a surrogate model (e.g., random forest) to predict fitness from chromosome features:

```python
from sklearn.ensemble import RandomForestRegressor

class SurrogateFitness:
    """
    Replace expensive fitness with a surrogate model prediction.
    Re-trains surrogate every `retrain_every` generations.
    """

    def __init__(self, true_fitness_fn: callable, retrain_every: int = 10,
                 n_initial_samples: int = 100):
        self._true_fn = true_fitness_fn
        self._retrain_every = retrain_every
        self._n_initial = n_initial_samples
        self._surrogate = RandomForestRegressor(n_estimators=50, random_state=0)
        self._chrom_data: list[np.ndarray] = []
        self._fitness_data: list[float] = []
        self._is_trained = False

    def _train_surrogate(self) -> None:
        if len(self._fitness_data) < self._n_initial:
            return
        X_train = np.array(self._chrom_data)
        y_train = np.array(self._fitness_data)
        self._surrogate.fit(X_train, y_train)
        self._is_trained = True

    def evaluate(self, chromosome: np.ndarray, X_data: np.ndarray,
                 y_data: np.ndarray, use_surrogate: bool = True) -> float:
        if not use_surrogate or not self._is_trained:
            # True evaluation
            f = self._true_fn(chromosome, X_data, y_data)
            self._chrom_data.append(chromosome.copy())
            self._fitness_data.append(f)
            return f
        # Surrogate prediction (fast)
        return float(self._surrogate.predict(chromosome.reshape(1, -1))[0])
```

---

## 8. Complete Fitness Design Example

Putting together a production-ready fitness function with all components:

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from functools import lru_cache

class FeatureSelectionFitness:
    """
    Production fitness function for GA feature selection.

    Features:
    - Stratified k-fold CV
    - Multi-objective: accuracy + parsimony + redundancy
    - Adaptive parsimony weight (increases over generations)
    - Fitness caching
    - Edge case handling
    - Walk-forward option for time series
    """

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 n_folds: int = 5,
                 parsimony_weight: float = 0.01,
                 redundancy_weight: float = 0.0,
                 time_series: bool = False,
                 adaptive_parsimony: bool = False):
        self.X = X
        self.y = y
        self.n_folds = n_folds
        self.parsimony_weight = parsimony_weight
        self.redundancy_weight = redundancy_weight
        self.time_series = time_series
        self.adaptive_parsimony = adaptive_parsimony
        self._cache: dict[bytes, float] = {}
        self._generation = 0
        self._max_generations = 100

    def set_generation(self, gen: int, max_gen: int) -> None:
        """Update generation for adaptive parsimony."""
        self._generation = gen
        self._max_generations = max_gen

    def _get_lambda(self) -> float:
        if self.adaptive_parsimony:
            return self.parsimony_weight * (self._generation / self._max_generations)
        return self.parsimony_weight

    def __call__(self, chromosome: np.ndarray) -> float:
        # Cache lookup
        key = chromosome.tobytes()
        if key in self._cache:
            return self._cache[key]

        # Edge case: empty chromosome
        selected = np.where(chromosome == 1)[0]
        if len(selected) == 0:
            self._cache[key] = -1.0
            return -1.0

        X_sel = self.X[:, selected]

        # CV evaluation
        try:
            model = LogisticRegression(max_iter=500, random_state=0)
            if self.time_series:
                from sklearn.model_selection import TimeSeriesSplit
                cv = TimeSeriesSplit(n_splits=self.n_folds)
            else:
                cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_sel, self.y, cv=cv, scoring="accuracy")
            score = float(scores.mean())
        except Exception:
            self._cache[key] = -1.0
            return -1.0

        # Parsimony penalty
        lam = self._get_lambda()
        parsimony = lam * len(selected) / len(chromosome)

        # Redundancy penalty (optional)
        if self.redundancy_weight > 0 and len(selected) > 1:
            corr = np.corrcoef(X_sel, rowvar=False)
            upper = corr[np.triu_indices(len(selected), k=1)]
            redundancy = self.redundancy_weight * float(np.mean(np.abs(upper)))
        else:
            redundancy = 0.0

        fitness = score - parsimony - redundancy

        if not np.isfinite(fitness):
            fitness = -1.0

        self._cache[key] = fitness
        return fitness
```

---

## Summary: Fitness Design Checklist

Before finalising a fitness function for production use:

- [ ] **CV strategy**: k-fold for iid data, stratified for imbalanced classes, walk-forward for time series
- [ ] **Metric**: accuracy for balanced, AUC for imbalanced, RMSE for regression
- [ ] **Parsimony**: include a weight $\lambda$ to control subset size — test $\lambda \in \{0.001, 0.01, 0.05\}$
- [ ] **Edge cases**: all-zero → -1.0, NaN/Inf → -1.0, exception handling
- [ ] **Caching**: always cache by `chromosome.tobytes()`
- [ ] **Adaptive parsimony**: ramp up $\lambda$ over generations for better exploration early
- [ ] **Redundancy**: add correlation penalty if features are expected to be correlated
- [ ] **Time series**: use `TimeSeriesSplit` with gap > 0 if data has temporal structure

## Key Takeaways

1. **CV inside GA** is standard; use stratified folds for classification.
2. **Fitness = score − λ|s|/p** is the workhorse formula; tune $\lambda$ via grid search on a holdout set.
3. **Exponential parsimony** punishes large subsets more aggressively than linear.
4. **Adaptive parsimony** (start at λ=0, ramp up) allows early exploration without premature feature dropping.
5. **Walk-forward CV** is mandatory for time series to prevent look-ahead bias.
6. **Fitness caching** by chromosome bytes key gives 60–90% speedup near convergence.
7. **Landscape analysis** (FDC, neutrality) informs operator choices — rugged landscapes need more mutation.
8. **Always handle edge cases**: empty chromosome, NaN fitness, degenerate populations.

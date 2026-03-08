# GA Fundamentals for Feature Selection

## In Brief

Genetic Algorithms (GAs) are population-based metaheuristics inspired by Darwinian evolution. For feature selection, they search the $2^p$ binary space by evolving populations of candidate subsets through selection, crossover, and mutation. Unlike filter methods (which score features independently) or wrappers (which greedily add/remove features), GAs explore combinations simultaneously and escape local optima through stochastic operators.

> **Why GAs for feature selection?** The search space is discrete, non-differentiable, and has complex feature interactions. GAs handle all three naturally — no gradient required, and crossover recombines interacting feature groups.

---

## 1. Binary Chromosome Encoding

Every candidate feature subset is encoded as a binary string called a **chromosome**. Each position (gene) corresponds to one feature:

```
Feature index:  0  1  2  3  4  5  6  7  8  9
Chromosome:    [1, 0, 1, 1, 0, 0, 1, 0, 1, 0]
               ↑     ↑  ↑     ↑        ↑
               selected features: {0, 2, 3, 6, 8}
```

- **Gene value 1**: feature is included in the model
- **Gene value 0**: feature is excluded
- **Chromosome length**: equals total number of features $p$
- **Search space**: $2^p$ possible chromosomes (one per feature subset)

### Why binary encoding?

Binary encoding is canonical for feature selection because:
1. The problem is naturally binary (include or exclude)
2. Standard crossover and mutation operators apply directly
3. The number of selected features is simply $\sum_i s_i$, making parsimony trivial to measure
4. All $2^p$ subsets are reachable from any starting point

### Python representation

```python
import numpy as np

class Individual:
    """Binary chromosome representing a feature subset."""

    def __init__(self, chromosome: np.ndarray):
        self.chromosome = chromosome.astype(np.int8)
        self.fitness: float | None = None

    @classmethod
    def random(cls, n_features: int, p_select: float = 0.5) -> "Individual":
        """Random initialisation with selection probability p_select."""
        chrom = (np.random.random(n_features) < p_select).astype(np.int8)
        # guarantee at least one feature
        if chrom.sum() == 0:
            chrom[np.random.randint(n_features)] = 1
        return cls(chrom)

    def selected_indices(self) -> np.ndarray:
        return np.where(self.chromosome == 1)[0]

    def n_selected(self) -> int:
        return int(self.chromosome.sum())

    def copy(self) -> "Individual":
        ind = Individual(self.chromosome.copy())
        ind.fitness = self.fitness
        return ind

    def __repr__(self) -> str:
        f = f"{self.fitness:.4f}" if self.fitness is not None else "None"
        return f"Individual(k={self.n_selected()}, fitness={f})"
```

---

## 2. Population Initialisation Strategies

The starting population should cover the search space broadly. Three strategies address different scenarios.

### 2.1 Uniform Random Initialisation

Each gene is set to 1 with probability $p = 0.5$, yielding expected subset size $p/2$.

```python
def init_random(pop_size: int, n_features: int) -> list[Individual]:
    return [Individual.random(n_features) for _ in range(pop_size)]
```

**Use when**: you have no prior knowledge of which features matter.

### 2.2 Biased Initialisation (using filter scores)

Use a univariate filter (mutual information, F-statistic) to compute feature relevance scores. Each gene $i$ is set to 1 with probability proportional to its score:

$$p_i = \frac{\text{score}_i - \min(\text{scores})}{\max(\text{scores}) - \min(\text{scores})} \cdot (p_{\max} - p_{\min}) + p_{\min}$$

where $p_{\min} = 0.1$ and $p_{\max} = 0.9$ keep all features reachable.

```python
from sklearn.feature_selection import mutual_info_classif

def init_biased(pop_size: int, X: np.ndarray, y: np.ndarray,
                p_min: float = 0.1, p_max: float = 0.9) -> list[Individual]:
    scores = mutual_info_classif(X, y)
    # Normalise to [p_min, p_max]
    s_min, s_max = scores.min(), scores.max()
    if s_max > s_min:
        probs = p_min + (scores - s_min) / (s_max - s_min) * (p_max - p_min)
    else:
        probs = np.full(len(scores), 0.5)

    population = []
    for _ in range(pop_size):
        chrom = (np.random.random(len(probs)) < probs).astype(np.int8)
        if chrom.sum() == 0:
            chrom[np.argmax(probs)] = 1
        population.append(Individual(chrom))
    return population
```

**Use when**: filter scores are cheap and you want faster initial convergence.

### 2.3 Guided Initialisation (subset seeding)

Seed part of the population with domain-expert subsets (e.g., features from a prior model or literature), then fill the rest randomly. This preserves known-good solutions while maintaining diversity.

```python
def init_guided(pop_size: int, n_features: int,
                seed_subsets: list[list[int]]) -> list[Individual]:
    population = []
    for subset in seed_subsets[:pop_size]:
        chrom = np.zeros(n_features, dtype=np.int8)
        chrom[subset] = 1
        population.append(Individual(chrom))
    # fill remainder randomly
    while len(population) < pop_size:
        population.append(Individual.random(n_features))
    return population
```

**Use when**: prior knowledge exists about important feature groups (e.g., from domain experts or previous runs).

---

## 3. The Generational Cycle

The GA iterates through a fixed cycle each generation:

```
┌─────────────────────────────────────────────────────────────┐
│  START: Initial population P₀ (random or biased)            │
│                                                              │
│  ┌─── Generation loop ──────────────────────────────────┐   │
│  │                                                       │   │
│  │  1. EVALUATE  fitness of all individuals in P_t       │   │
│  │         ↓                                             │   │
│  │  2. SELECT  parents from P_t                          │   │
│  │         ↓                                             │   │
│  │  3. CROSSOVER  pairs of parents → offspring           │   │
│  │         ↓                                             │   │
│  │  4. MUTATE  offspring with probability p_m            │   │
│  │         ↓                                             │   │
│  │  5. REPLACE  P_t with offspring → P_{t+1}             │   │
│  │         ↓                                             │   │
│  │  6. CHECK termination criterion                       │   │
│  │         ↓ (continue if not met)                       │   │
│  └───────────────────────────────────────────────────────┘   │
│                                                              │
│  END: Return best individual ever seen                        │
└─────────────────────────────────────────────────────────────┘
```

**Termination criteria** (use any combination):
- Maximum generations reached
- Best fitness plateau for $G$ consecutive generations
- Population diversity below threshold $\delta$
- Wall-clock time budget exceeded

---

## 4. Selection Operators

Selection chooses which individuals reproduce. The trade-off is **selection pressure**: high pressure exploits the current best solutions; low pressure explores more broadly.

### 4.1 Tournament Selection

Randomly sample $k$ individuals from the population and return the best. Repeat for each parent slot.

$$P(\text{individual } i \text{ is selected}) = P(i = \max_{j \in T} f_j)$$

where $T$ is a random subset of size $k$.

**Selection pressure analysis**: For a population of size $N$, the expected number of times the best individual is selected per generation with tournament size $k$ is approximately:

$$E[\text{copies of best}] \approx k \cdot \frac{1}{N}$$

| Tournament size $k$ | Selection pressure | Typical use |
|:---:|:---:|:---|
| 2 | Low | High diversity, slow convergence |
| 3–5 | Medium | Balanced (recommended default) |
| 7–10 | High | Fast convergence, premature risk |
| $N$ | Maximum | Equivalent to greedy selection |

```python
def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    """Select one individual via tournament of size k."""
    contestants = np.random.choice(len(population), size=k, replace=False)
    winner_idx = max(contestants, key=lambda i: population[i].fitness)
    return population[winner_idx].copy()
```

### 4.2 Roulette Wheel (Fitness-Proportionate) Selection

Each individual $i$ is selected with probability proportional to its fitness:

$$P(\text{select } i) = \frac{f_i}{\sum_{j=1}^{N} f_j}$$

**Limitation**: if one individual has fitness much higher than others, it monopolises reproduction. **Fitness scaling** (e.g., sigma scaling) mitigates this:

$$f'_i = \max\!\left(f_i - (\bar{f} - c \cdot \sigma_f),\ 0\right), \quad c \approx 2$$

```python
def roulette_select(population: list[Individual]) -> Individual:
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    # shift to ensure non-negative
    fitnesses -= fitnesses.min() - 1e-9
    probs = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), p=probs)
    return population[idx].copy()
```

### 4.3 Rank-Based Selection

Sort individuals by fitness and assign selection probability based on rank rather than raw fitness. For linear ranking with bias parameter $\eta^+ \in (1, 2]$:

$$P(\text{select rank } r) = \frac{1}{N}\left(\eta^+ - (\eta^+ - \eta^-)\frac{r - 1}{N - 1}\right)$$

where $\eta^- = 2 - \eta^+$ ensures probabilities sum to 1.

**Advantage**: immune to fitness scaling issues; selection pressure is constant regardless of fitness landscape shape.

```python
def rank_select(population: list[Individual], eta_plus: float = 1.5) -> Individual:
    N = len(population)
    sorted_idx = np.argsort([ind.fitness for ind in population])  # ascending
    eta_minus = 2.0 - eta_plus
    # rank 1 = worst, rank N = best
    ranks = np.arange(1, N + 1)
    probs = (1 / N) * (eta_minus + (eta_plus - eta_minus) * (ranks - 1) / (N - 1))
    probs /= probs.sum()
    chosen_rank = np.random.choice(N, p=probs)
    return population[sorted_idx[chosen_rank]].copy()
```

### 4.4 Stochastic Universal Sampling (SUS)

SUS selects $M$ individuals in a single pass by placing $M$ equally-spaced pointers on a roulette wheel:

$$\text{pointer}_j = r + \frac{j-1}{M}, \quad r \sim U\!\left[0, \frac{1}{M}\right]$$

**Advantage over roulette wheel**: zero spread — each individual is selected within $\pm 1$ of its expected count, eliminating sampling noise.

```python
def sus_select(population: list[Individual], n_select: int) -> list[Individual]:
    fitnesses = np.array([ind.fitness for ind in population], dtype=float)
    fitnesses -= fitnesses.min() - 1e-9
    total = fitnesses.sum()
    step = total / n_select
    start = np.random.uniform(0, step)
    pointers = [start + i * step for i in range(n_select)]

    selected = []
    cumulative = np.cumsum(fitnesses)
    idx = 0
    for ptr in pointers:
        while cumulative[idx] < ptr:
            idx += 1
        selected.append(population[idx].copy())
    return selected
```

---

## 5. Crossover Operators

Crossover (recombination) combines the genetic material of two parents to produce offspring. For feature selection, crossover merges different feature subsets.

### 5.1 Single-Point Crossover

A random point $c \in \{1, \ldots, p-1\}$ splits each parent. Children inherit opposite halves:

```
Parent 1:  [1 1 1 1 | 0 0 0 0]
Parent 2:  [0 0 0 0 | 1 1 1 1]
                ↑ c=4
Child 1:   [1 1 1 1 | 1 1 1 1]
Child 2:   [0 0 0 0 | 0 0 0 0]
```

**Feature selection implication**: genes near position 0 and genes near position $p$ tend to stay together across generations. Features with similar indices become **linkage-dependent** — which can be exploited if related features are grouped by index, or harmful if unrelated features are artificially linked.

```python
def single_point_crossover(p1: Individual, p2: Individual
                           ) -> tuple[Individual, Individual]:
    c = np.random.randint(1, len(p1.chromosome))
    c1 = Individual(np.concatenate([p1.chromosome[:c], p2.chromosome[c:]]))
    c2 = Individual(np.concatenate([p2.chromosome[:c], p1.chromosome[c:]]))
    return c1, c2
```

### 5.2 Two-Point Crossover

Two cut points $c_1 < c_2$ swap the middle segment:

```
Parent 1:  [1 1 | 0 0 0 | 1 1]
Parent 2:  [0 0 | 1 1 1 | 0 0]
              ↑c₁      ↑c₂
Child 1:   [1 1 | 1 1 1 | 1 1]
Child 2:   [0 0 | 0 0 0 | 0 0]
```

**Feature selection implication**: reduces positional bias compared to single-point. The middle block can represent a coherent feature group.

```python
def two_point_crossover(p1: Individual, p2: Individual
                        ) -> tuple[Individual, Individual]:
    pts = sorted(np.random.choice(range(1, len(p1.chromosome)), size=2, replace=False))
    c, d = pts
    c1 = Individual(np.concatenate([p1.chromosome[:c], p2.chromosome[c:d], p1.chromosome[d:]]))
    c2 = Individual(np.concatenate([p2.chromosome[:c], p1.chromosome[c:d], p2.chromosome[d:]]))
    return c1, c2
```

### 5.3 Uniform Crossover

Each gene is inherited independently from either parent with probability $p_u = 0.5$. A binary mask $M \in \{0,1\}^p$ determines inheritance:

```
Parent 1:  [1 0 1 1 0 1 0 1]
Parent 2:  [0 1 0 0 1 0 1 0]
Mask:      [1 0 0 1 1 0 1 0]
Child 1:   [1 1 0 1 0 0 0 0]   (from P1 where mask=1, P2 where mask=0)
Child 2:   [0 0 1 0 1 1 1 1]   (complement)
```

**Feature selection implication**: highest diversity — no positional bias, every combination of individual features from both parents is possible. Preferred when features are independent.

```python
def uniform_crossover(p1: Individual, p2: Individual,
                      swap_prob: float = 0.5) -> tuple[Individual, Individual]:
    mask = np.random.random(len(p1.chromosome)) < swap_prob
    c1 = Individual(np.where(mask, p1.chromosome, p2.chromosome))
    c2 = Individual(np.where(mask, p2.chromosome, p1.chromosome))
    return c1, c2
```

### 5.4 Feature-Group-Aware Crossover

When features form semantic groups (e.g., technical indicators from the same price series, lag features of the same variable), standard crossover may split coherent groups. **Block crossover** treats groups as atomic units:

```python
def block_crossover(p1: Individual, p2: Individual,
                    groups: list[list[int]]) -> tuple[Individual, Individual]:
    """
    Crossover that preserves feature groups.

    Parameters
    ----------
    groups : list of lists
        Each sub-list contains indices of a feature group.
        All features within a group are either all-in or all-out.
    """
    c1_chrom = p1.chromosome.copy()
    c2_chrom = p2.chromosome.copy()

    for group in groups:
        if np.random.random() < 0.5:
            # Swap this entire group between children
            c1_chrom[group] = p2.chromosome[group]
            c2_chrom[group] = p1.chromosome[group]

    return Individual(c1_chrom), Individual(c2_chrom)
```

**When to use**: time series feature engineering where lag-1, lag-2, lag-3 of a variable form a natural group; technical indicators computed from the same price window; polynomial feature expansions of the same raw feature.

---

## 6. Mutation Operators

Mutation introduces random variation to prevent premature convergence and enable the GA to escape local optima.

### 6.1 Standard Bit-Flip Mutation

Each gene flips independently with probability $p_m$:

$$s_i' = \begin{cases} 1 - s_i & \text{with probability } p_m \\ s_i & \text{with probability } 1 - p_m \end{cases}$$

**Recommended rate**: $p_m = 1/p$ gives an expected one bit flipped per chromosome — the standard choice for binary GAs.

```python
def bit_flip_mutate(ind: Individual, p_m: float | None = None) -> Individual:
    """In-place bit-flip mutation."""
    if p_m is None:
        p_m = 1.0 / len(ind.chromosome)
    mask = np.random.random(len(ind.chromosome)) < p_m
    ind.chromosome[mask] ^= 1  # XOR flip
    # repair: ensure at least one feature selected
    if ind.chromosome.sum() == 0:
        ind.chromosome[np.random.randint(len(ind.chromosome))] = 1
    ind.fitness = None  # invalidate cached fitness
    return ind
```

### 6.2 Adaptive Mutation Rates

Fixed mutation rates are a compromise. **Adaptive mutation** adjusts $p_m$ based on population state:

**Scheme 1 — Generational decay** (exploration → exploitation):
$$p_m(t) = p_{m,\text{init}} \cdot \left(1 - \frac{t}{T_{\max}}\right) + p_{m,\text{final}}$$

**Scheme 2 — Diversity-triggered** (increase $p_m$ when diversity falls):
$$p_m(t) = \begin{cases} p_{m,\text{high}} & \text{if } D(t) < D_{\min} \\ p_{m,\text{low}} & \text{otherwise} \end{cases}$$

where $D(t)$ is population diversity (e.g., mean pairwise Hamming distance).

**Scheme 3 — $1/\sqrt{p}$ rate**: for $p$ features, $p_m = 1/\sqrt{p}$ gives $\sqrt{p}$ expected flips — beneficial when good solutions are sparse.

```python
def adaptive_mutation_rate(generation: int, max_generations: int,
                           diversity: float, diversity_threshold: float = 0.1,
                           p_m_low: float = 0.01, p_m_high: float = 0.1) -> float:
    """Return adaptive mutation rate based on generation and diversity."""
    # Base: decay from high to low over generations
    decay_rate = p_m_high - (p_m_high - p_m_low) * (generation / max_generations)
    # Boost if diversity has collapsed
    if diversity < diversity_threshold:
        return min(p_m_high, decay_rate * 3.0)
    return decay_rate
```

### 6.3 Feature-Count-Preserving Mutation

Standard bit-flip changes the number of selected features. When you want to maintain approximately $k$ features, use **swap mutation**: randomly select one included feature and one excluded feature, and swap them:

$$\text{swap}(s) : \text{pick } i \in \{j: s_j = 1\},\ j \in \{k: s_k = 0\},\ \text{then } s_i \leftarrow 0,\ s_j \leftarrow 1$$

This leaves $\sum s_i$ unchanged.

```python
def swap_mutate(ind: Individual, n_swaps: int = 1) -> Individual:
    """Swap mutation — preserves number of selected features."""
    selected = np.where(ind.chromosome == 1)[0]
    unselected = np.where(ind.chromosome == 0)[0]
    if len(selected) == 0 or len(unselected) == 0:
        return ind
    n_swaps = min(n_swaps, len(selected), len(unselected))
    drop = np.random.choice(selected, size=n_swaps, replace=False)
    add = np.random.choice(unselected, size=n_swaps, replace=False)
    ind.chromosome[drop] = 0
    ind.chromosome[add] = 1
    ind.fitness = None
    return ind
```

---

## 7. Elitism

**Elitism** copies the $e$ best individuals from the current generation directly into the next, bypassing selection, crossover, and mutation. This guarantees the best-seen solution is never lost.

### Effect on convergence

| Elitism rate $e/N$ | Convergence | Diversity | Recommendation |
|:---:|:---:|:---:|:---|
| 0% | Slower | High | Risky — best solution can be lost |
| 1–5% | Balanced | Moderate | **Standard choice** |
| 10–20% | Fast | Low | Premature convergence risk |
| 50%+ | Very fast | Very low | Effectively local search |

```python
def apply_elitism(old_pop: list[Individual],
                  new_pop: list[Individual],
                  n_elite: int) -> list[Individual]:
    """Replace worst n_elite in new_pop with best n_elite from old_pop."""
    elites = sorted(old_pop, key=lambda x: x.fitness, reverse=True)[:n_elite]
    # Sort new population ascending (worst first)
    new_pop_sorted = sorted(new_pop, key=lambda x: x.fitness or -np.inf)
    # Replace worst with elite copies
    for i, elite in enumerate(elites):
        new_pop_sorted[i] = elite.copy()
    return new_pop_sorted
```

---

## 8. Complete GA Feature Selector from Scratch

Combining all components into a production-ready implementation:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class GAConfig:
    pop_size: int = 50
    n_generations: int = 100
    crossover_rate: float = 0.8
    mutation_rate: float | None = None   # None → 1/p
    tournament_size: int = 3
    n_elite: int = 2
    crossover_type: str = "uniform"      # single | two_point | uniform
    adaptive_mutation: bool = True
    patience: int = 20                   # early stopping generations
    parsimony_weight: float = 0.01


class GAFeatureSelector:
    """
    Genetic Algorithm feature selector — no DEAP dependency.

    Parameters
    ----------
    config : GAConfig
        Hyperparameter configuration.
    fitness_fn : callable, optional
        Custom fitness function f(individual, X, y) → float.
        Defaults to CV accuracy minus parsimony penalty.
    """

    def __init__(self, config: GAConfig = GAConfig(),
                 fitness_fn: Callable | None = None):
        self.config = config
        self._fitness_fn = fitness_fn or self._default_fitness
        self.history: dict = {
            "best_fitness": [], "mean_fitness": [], "diversity": [],
            "n_features": []
        }
        self.best_individual_: Individual | None = None
        self._cache: dict[bytes, float] = {}

    # ------------------------------------------------------------------ #
    # Fitness                                                              #
    # ------------------------------------------------------------------ #

    def _default_fitness(self, ind: Individual, X: np.ndarray,
                         y: np.ndarray) -> float:
        if ind.n_selected() == 0:
            return -1.0
        key = ind.chromosome.tobytes()
        if key in self._cache:
            return self._cache[key]
        model = LogisticRegression(max_iter=500, random_state=0)
        scores = cross_val_score(
            model, X[:, ind.selected_indices()], y, cv=5, scoring="accuracy"
        )
        penalty = self.config.parsimony_weight * ind.n_selected() / len(ind.chromosome)
        result = float(scores.mean()) - penalty
        self._cache[key] = result
        return result

    # ------------------------------------------------------------------ #
    # Crossover dispatch                                                   #
    # ------------------------------------------------------------------ #

    def _crossover(self, p1: Individual, p2: Individual
                   ) -> tuple[Individual, Individual]:
        t = self.config.crossover_type
        if t == "single":
            return single_point_crossover(p1, p2)
        elif t == "two_point":
            return two_point_crossover(p1, p2)
        else:
            return uniform_crossover(p1, p2)

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GAFeatureSelector":
        cfg = self.config
        n_features = X.shape[1]
        p_m = cfg.mutation_rate or (1.0 / n_features)

        # Initialise population
        population = [Individual.random(n_features) for _ in range(cfg.pop_size)]

        # Evaluate
        for ind in population:
            ind.fitness = self._fitness_fn(ind, X, y)

        best = max(population, key=lambda x: x.fitness).copy()
        stagnation = 0

        for gen in range(cfg.n_generations):
            # Adaptive mutation rate
            diversity = self._hamming_diversity(population)
            if cfg.adaptive_mutation:
                p_m_eff = adaptive_mutation_rate(
                    gen, cfg.n_generations, diversity
                )
            else:
                p_m_eff = p_m

            # Build next generation
            next_gen: list[Individual] = []

            # Elitism
            elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:cfg.n_elite]
            next_gen.extend([e.copy() for e in elites])

            # Offspring
            while len(next_gen) < cfg.pop_size:
                p1 = tournament_select(population, cfg.tournament_size)
                p2 = tournament_select(population, cfg.tournament_size)

                if np.random.random() < cfg.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                bit_flip_mutate(c1, p_m_eff)
                bit_flip_mutate(c2, p_m_eff)

                c1.fitness = self._fitness_fn(c1, X, y)
                next_gen.append(c1)
                if len(next_gen) < cfg.pop_size:
                    c2.fitness = self._fitness_fn(c2, X, y)
                    next_gen.append(c2)

            population = next_gen[:cfg.pop_size]

            # Update best
            gen_best = max(population, key=lambda x: x.fitness)
            if gen_best.fitness > best.fitness:
                best = gen_best.copy()
                stagnation = 0
            else:
                stagnation += 1

            # Record history
            fitnesses = [ind.fitness for ind in population]
            self.history["best_fitness"].append(best.fitness)
            self.history["mean_fitness"].append(float(np.mean(fitnesses)))
            self.history["diversity"].append(diversity)
            self.history["n_features"].append(best.n_selected())

            # Early stopping
            if stagnation >= cfg.patience:
                print(f"Early stopping at generation {gen + 1} (no improvement for {cfg.patience} gens)")
                break

        self.best_individual_ = best
        self.selected_features_ = best.selected_indices()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X[:, self.selected_features_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    @staticmethod
    def _hamming_diversity(population: list[Individual]) -> float:
        """Mean pairwise Hamming distance as diversity metric."""
        chroms = np.array([ind.chromosome for ind in population])
        n = len(chroms)
        if n < 2:
            return 0.0
        diffs = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diffs += np.sum(chroms[i] != chroms[j])
                count += 1
        return diffs / (count * chroms.shape[1])


# ------------------------------------------------------------------ #
# Quick usage example                                                 #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    cfg = GAConfig(pop_size=30, n_generations=50, parsimony_weight=0.01)
    selector = GAFeatureSelector(cfg)
    selector.fit(X_train, y_train)

    print(f"Selected {selector.best_individual_.n_selected()}/{X_train.shape[1]} features")
    print(f"Best fitness: {selector.best_individual_.fitness:.4f}")
    print(f"Selected indices: {selector.selected_features_}")
```

---

## 9. Operator Summary and Recommendations

| Scenario | Recommended configuration |
|:---|:---|
| General classification, $p < 50$ | Uniform crossover, tournament $k=3$, $p_m = 1/p$, elitism 2 |
| High-dimensional, $p > 200$ | Two-point crossover, rank selection, adaptive $p_m$, larger pop |
| Feature groups exist | Block crossover, higher parsimony weight |
| Fast convergence needed | High tournament $k$, higher elitism, SUS selection |
| Exploration required | Roulette wheel, low tournament $k$, diversity-triggered mutation |
| Fixed-size subset wanted | Swap mutation, initialise with target $k$ |

---

## Key Takeaways

1. **Binary chromosomes** encode feature subsets naturally — gene $i = 1$ means feature $i$ is selected.
2. **Initialisation matters** — biased starts from filter scores converge faster but may sacrifice diversity.
3. **Tournament selection** is the most robust default: simple to implement, tunable pressure via $k$.
4. **Uniform crossover** is preferred for feature selection when feature order has no semantic meaning.
5. **Mutation rate $1/p$** is the standard default; adaptive rates outperform on rugged landscapes.
6. **Elitism of 1–5%** prevents losing the best solution without sacrificing diversity.
7. **Fitness caching** is essential — most chromosomes repeat across generations; cache by bytes key.
8. **Early stopping** on fitness plateau saves computation without sacrificing solution quality.

# Convergence, Diversity, and Parameter Tuning for GA Feature Selection

## In Brief

A well-configured GA finds high-quality feature subsets efficiently. A poorly configured one either converges prematurely to a mediocre solution or never converges at all. This guide covers: convergence detection methods, diversity preservation techniques (crowding, fitness sharing, niching, island models), diagnosing and fixing premature convergence, systematic parameter tuning for feature selection, comparison against alternative search methods, and integration with the DEAP framework.

---

## 1. Convergence Detection

### 1.1 Fitness Plateau Detection

Track the best fitness across generations. A **plateau** occurs when the best fitness does not improve for $G$ consecutive generations.

```python
class ConvergenceMonitor:
    """
    Tracks GA convergence state and triggers early stopping.

    Parameters
    ----------
    patience : int
        Number of plateau generations before declaring convergence.
    min_delta : float
        Minimum improvement in best fitness to count as progress.
    """

    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self._best = -np.inf
        self._stagnation_count = 0
        self.converged = False
        self.history: list[float] = []

    def update(self, best_fitness: float) -> bool:
        """
        Update monitor with latest best fitness.
        Returns True if converged (stop signal).
        """
        self.history.append(best_fitness)
        if best_fitness > self._best + self.min_delta:
            self._best = best_fitness
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        if self._stagnation_count >= self.patience:
            self.converged = True
        return self.converged

    def generations_without_improvement(self) -> int:
        return self._stagnation_count
```

### 1.2 Population Diversity Metrics

Diversity measures how spread out the population is in chromosome space. Multiple metrics are useful:

**Hamming diversity** — mean pairwise Hamming distance normalised to $[0, 1]$:

$$D_H = \frac{1}{N(N-1)} \sum_{i \neq j} d_H(s_i, s_j) / p$$

where $d_H(s_i, s_j) = \sum_k |s_{ik} - s_{jk}|$ and $p$ is chromosome length.

```python
def hamming_diversity(population: list) -> float:
    """Mean pairwise Hamming distance, normalised to [0, 1]."""
    chroms = np.array([ind.chromosome for ind in population])
    n, p = chroms.shape
    # Efficient computation using matrix operations
    diffs = np.sum(chroms[:, None, :] != chroms[None, :, :], axis=2)
    # Upper triangle only
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    mean_dist = diffs[mask].mean() / p
    return float(mean_dist)
```

**Genotypic entropy** — Shannon entropy over each gene position:

$$H = \frac{1}{p} \sum_{i=1}^{p} H(p_i), \quad H(p_i) = -p_i \log p_i - (1-p_i)\log(1-p_i)$$

where $p_i$ is the proportion of individuals with gene $i = 1$.

```python
def genotypic_entropy(population: list) -> float:
    """Mean per-gene Shannon entropy, normalised to [0, 1]."""
    chroms = np.array([ind.chromosome for ind in population], dtype=float)
    p_ones = chroms.mean(axis=0)  # proportion of 1s per gene
    # Clip to avoid log(0)
    p_ones = np.clip(p_ones, 1e-10, 1 - 1e-10)
    entropy = -p_ones * np.log2(p_ones) - (1 - p_ones) * np.log2(1 - p_ones)
    return float(entropy.mean())  # normalised by log2(2) = 1
```

**Phenotypic diversity** — standard deviation of fitness values:

$$D_f = \text{std}(f_1, f_2, \ldots, f_N)$$

Low $D_f$ indicates the population has converged to similar fitness levels.

### 1.3 Generation Limit

The simplest termination criterion: run for exactly $T_{\max}$ generations. Use as a safety backstop when other criteria may not trigger:

```python
MAX_GENERATIONS = 200  # always terminate, even if not converged

for gen in range(MAX_GENERATIONS):
    # ... evolution step ...
    if convergence_monitor.update(best_fitness):
        print(f"Converged at generation {gen}")
        break
```

**Recommended compound termination**:
```python
def should_terminate(gen, max_gen, monitor, diversity, min_diversity=0.01):
    return (gen >= max_gen or
            monitor.converged or
            diversity < min_diversity)
```

---

## 2. Diversity Preservation

Without explicit diversity mechanisms, GAs on feature selection problems frequently lose diversity within 20–40 generations, causing premature convergence.

### 2.1 Crowding

Crowding limits competition to similar individuals: offspring compete against the most similar individual in the population (their **nearest neighbour** in Hamming space), rather than any random individual.

```python
def crowding_replacement(population: list, offspring: list,
                         crowding_factor: int = 3) -> list:
    """
    Deterministic crowding: each offspring replaces its most similar parent.

    For each offspring:
    1. Select crowding_factor random individuals from population
    2. Find the most similar (minimum Hamming distance)
    3. Replace only if offspring fitness ≥ similar individual's fitness
    """
    new_population = [ind.copy() for ind in population]
    for child in offspring:
        # Sample candidate replacements
        candidates_idx = np.random.choice(len(new_population),
                                          size=min(crowding_factor, len(new_population)),
                                          replace=False)
        # Find most similar
        distances = [np.sum(child.chromosome != new_population[i].chromosome)
                     for i in candidates_idx]
        closest_idx = candidates_idx[np.argmin(distances)]
        # Replace if offspring is better (or equal — stochastic version)
        if child.fitness >= new_population[closest_idx].fitness:
            new_population[closest_idx] = child.copy()
    return new_population
```

**Effect**: maintains phenotypic niches — different "types" of feature subsets coexist.

### 2.2 Fitness Sharing

Reduce the fitness of individuals that are close to many others, rewarding individuals in sparse regions of the search space:

$$f'(s_i) = \frac{f(s_i)}{\sum_j \text{sh}(d_H(s_i, s_j))}$$

where $\text{sh}(d)$ is the sharing function:

$$\text{sh}(d) = \begin{cases} 1 - (d / \sigma_{\text{share}})^\alpha & \text{if } d < \sigma_{\text{share}} \\ 0 & \text{otherwise} \end{cases}$$

```python
def apply_fitness_sharing(population: list,
                          sigma_share: float = 0.3,
                          alpha: float = 1.0) -> None:
    """
    Modify fitness in-place to apply sharing.

    Parameters
    ----------
    sigma_share : float
        Sharing radius (fraction of chromosome length).
        Individuals within this Hamming distance share fitness.
    alpha : float
        Shape of sharing function (1.0 = linear).
    """
    n = len(population)
    p = len(population[0].chromosome)
    sigma_abs = sigma_share * p  # convert to absolute Hamming distance

    niche_counts = np.ones(n)  # at minimum, each individual shares with itself
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = np.sum(population[i].chromosome != population[j].chromosome)
            if d < sigma_abs:
                sh = 1.0 - (d / sigma_abs) ** alpha
                niche_counts[i] += sh

    for i, ind in enumerate(population):
        ind.fitness = ind.fitness / niche_counts[i]
```

**Setting $\sigma_{\text{share}}$**: for $p$ features, $\sigma_{\text{share}} \approx 0.2$ means individuals within 20% of chromosome length share fitness. Start with $\sigma_{\text{share}} = 0.15$.

### 2.3 Niching via Speciation

Group individuals into species based on similarity. Each species evolves independently and only competes within its species for selection. Similar to island models but dynamic:

```python
def assign_species(population: list, n_species: int = 5,
                   sigma_share: float = 0.2) -> list[list[int]]:
    """
    Assign individuals to species using representative seeds.
    Returns list of species (each a list of individual indices).
    """
    p = len(population[0].chromosome)
    sigma_abs = sigma_share * p
    # Select species representatives (diverse seeds)
    reps = [0]
    for _ in range(n_species - 1):
        # Add most distant individual from existing reps
        max_min_dist = -1
        best_idx = 1
        for i in range(1, len(population)):
            min_dist = min(
                np.sum(population[i].chromosome != population[reps[r]].chromosome)
                for r in range(len(reps))
            )
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = i
        reps.append(best_idx)

    # Assign each individual to nearest representative
    species: list[list[int]] = [[] for _ in range(n_species)]
    for i, ind in enumerate(population):
        dists = [np.sum(ind.chromosome != population[reps[r]].chromosome)
                 for r in range(n_species)]
        species[np.argmin(dists)].append(i)
    return species
```

### 2.4 Island Models (Parallel GAs)

Run multiple independent GA subpopulations (islands) in parallel. Periodically **migrate** a few individuals between islands:

```python
import threading
from concurrent.futures import ThreadPoolExecutor

class IslandGA:
    """
    Parallel GA with island model for diversity preservation.

    Each island is an independent GA. Migration exchanges individuals
    every `migration_interval` generations.
    """

    def __init__(self, n_islands: int = 4,
                 island_pop_size: int = 25,
                 migration_interval: int = 10,
                 migration_rate: float = 0.1,
                 ga_config=None):
        self.n_islands = n_islands
        self.island_pop_size = island_pop_size
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        self.ga_config = ga_config or {}

    def migrate(self, islands: list[list]) -> None:
        """
        Ring topology migration: each island sends best individuals to next.
        migration_rate fraction of each island migrates.
        """
        n_migrants = max(1, int(self.island_pop_size * self.migration_rate))
        migrants = []
        for island in islands:
            # Select best n_migrants to emigrate
            best = sorted(island, key=lambda x: x.fitness, reverse=True)[:n_migrants]
            migrants.append([ind.copy() for ind in best])

        # Ring topology: island i receives from island (i-1) % n_islands
        for i, island in enumerate(islands):
            incoming = migrants[(i - 1) % self.n_islands]
            # Replace worst with incoming migrants
            worst_idx = sorted(range(len(island)),
                               key=lambda j: island[j].fitness)[:n_migrants]
            for k, idx in enumerate(worst_idx):
                island[idx] = incoming[k]

    def run(self, X: np.ndarray, y: np.ndarray,
            n_generations: int = 100, fitness_fn=None) -> list:
        """
        Run island GA and return combined population of best individuals.
        """
        from .ga_core import initialize_population, generation_step

        # Initialise islands
        islands = [
            initialize_population(self.island_pop_size, X.shape[1], X, y, fitness_fn)
            for _ in range(self.n_islands)
        ]

        for gen in range(n_generations):
            # Evolve each island independently
            for i, island in enumerate(islands):
                islands[i] = generation_step(island, X, y, fitness_fn,
                                             **self.ga_config)
            # Periodic migration
            if (gen + 1) % self.migration_interval == 0:
                self.migrate(islands)

        # Combine all islands, return unique best individuals
        all_individuals = [ind for island in islands for ind in island]
        all_individuals.sort(key=lambda x: x.fitness, reverse=True)
        return all_individuals
```

**Island model advantages**:
- Natural parallelism — each island runs on a CPU core
- Diversity maintained by geographic isolation
- Migration spreads good solutions without full population mixing

---

## 3. Premature Convergence: Diagnosis and Remedies

Premature convergence occurs when the population collapses to a local optimum before exploring the full search space.

### 3.1 Diagnosis

| Symptom | Measurement | Threshold |
|:---|:---|:---:|
| Fitness plateau | Stagnation generations | > 30 gens |
| Low Hamming diversity | $D_H$ | < 0.05 |
| Low genotypic entropy | $H$ | < 0.1 |
| Uniform population | Std of fitness | < 0.001 |
| All identical chromosomes | # unique chromosomes | < 5 |

```python
def diagnose_convergence(population: list, history: list[float],
                         gen: int) -> dict:
    """Return diagnostic report for current population state."""
    chroms = np.array([ind.chromosome for ind in population])
    fitnesses = np.array([ind.fitness for ind in population])
    return {
        "generation": gen,
        "hamming_diversity": hamming_diversity(population),
        "genotypic_entropy": genotypic_entropy(population),
        "fitness_std": float(fitnesses.std()),
        "n_unique": len(set(c.tobytes() for c in chroms)),
        "stagnation_gens": sum(1 for i in range(len(history) - 1, -1, -1)
                               if abs(history[i] - history[-1]) < 1e-5),
    }
```

### 3.2 Remedies

**Remedy 1 — Increase mutation rate temporarily**:
```python
if diversity < 0.05:
    p_m_effective = min(0.3, p_m * 5.0)  # 5× normal rate
else:
    p_m_effective = p_m
```

**Remedy 2 — Random immigrant injection**:
```python
def inject_immigrants(population: list, n_immigrants: int,
                      n_features: int, X, y, fitness_fn) -> list:
    """Replace worst n_immigrants with random new individuals."""
    immigrants = [Individual.random(n_features) for _ in range(n_immigrants)]
    for ind in immigrants:
        ind.fitness = fitness_fn(ind.chromosome, X, y)
    # Sort by fitness, replace worst
    population.sort(key=lambda x: x.fitness, reverse=True)
    return population[:-n_immigrants] + immigrants
```

**Remedy 3 — Population restart with best solution preserved**:
```python
def restart_with_elites(population: list, n_elite: int,
                        n_features: int, X, y, fitness_fn) -> list:
    """Restart population, keeping only top n_elite individuals."""
    elites = sorted(population, key=lambda x: x.fitness, reverse=True)[:n_elite]
    new_pop = [Individual.random(n_features) for _ in range(len(population) - n_elite)]
    for ind in new_pop:
        ind.fitness = fitness_fn(ind.chromosome, X, y)
    return elites + new_pop
```

**Remedy 4 — Reduce selection pressure**:

Lower tournament size from $k=5$ to $k=2$, or switch from tournament to rank selection temporarily.

---

## 4. Parameter Tuning for Feature Selection

### 4.1 Population Size $N$

**Theory**: $N$ should scale with the chromosome length $p$. Common rules:

- $N = 10p$ (conservative, good coverage)
- $N = \sqrt{2^p}$ (impractical for large $p$)
- $N = 50$ to $200$ (practical range for most feature selection problems)

```python
def recommend_pop_size(n_features: int, budget: int = 10000) -> int:
    """
    Recommend population size given feature count and evaluation budget.

    budget = total fitness evaluations allowed
    population_size = budget / n_generations (target ~100 generations)
    """
    target_gens = 100
    computed = budget // target_gens
    min_size = max(20, 2 * n_features)  # cover search space
    return min(computed, 200, max(min_size, computed))
```

| $p$ (features) | Recommended $N$ |
|:---:|:---:|
| 10–30 | 30–50 |
| 30–100 | 50–100 |
| 100–500 | 100–200 |
| > 500 | 200+ with surrogate fitness |

### 4.2 Generation Count $T$

Minimum generations needed scales with $\log N$ and problem difficulty. Practical recommendations:

| Population size $N$ | Recommended $T$ |
|:---:|:---:|
| 30 | 100–200 |
| 50 | 75–150 |
| 100 | 50–100 |

Always pair with early stopping (patience = 15–30 generations) to avoid unnecessary computation.

### 4.3 Crossover Rate $p_c$

The probability that two selected parents undergo crossover (vs. being copied unchanged):

- **Too low** ($p_c < 0.5$): insufficient genetic mixing, slow convergence
- **Too high** ($p_c > 0.95$): useful building blocks disrupted too frequently
- **Recommended**: $p_c = 0.7$–$0.9$ for feature selection

```python
CROSSOVER_RATE = 0.8  # default for feature selection
```

### 4.4 Mutation Rate $p_m$

- **Standard**: $p_m = 1/p$ — expected one bit flip per chromosome
- **Exploration**: $p_m = 3/p$ — faster escape from local optima
- **Exploitation**: $p_m = 0.5/p$ — conservative, slow convergence

For feature selection specifically, slightly higher than standard ($p_m = 1.5/p$) often works better due to the rugged landscape.

### 4.5 Recommended Parameter Ranges Summary

| Parameter | Default | Range | Notes |
|:---|:---:|:---:|:---|
| Population size $N$ | 50 | 30–200 | Scale with $p$ |
| Generations $T$ | 100 | 50–300 | Use early stopping |
| Crossover rate $p_c$ | 0.8 | 0.7–0.9 | Fixed or adaptive |
| Mutation rate $p_m$ | $1/p$ | $0.5/p$–$3/p$ | Adaptive recommended |
| Tournament size $k$ | 3 | 2–7 | Lower for diversity |
| Elitism count $e$ | 2 | 1–5 | Always include |
| Parsimony $\lambda$ | 0.01 | 0.001–0.1 | Tune on holdout |

---

## 5. GA vs. Alternative Search Methods

### 5.1 Comparison Framework

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

def evaluate_selector(selector, X_train, y_train, X_test, y_test,
                      method_name: str) -> dict:
    """Evaluate a fitted feature selector on train and test data."""
    t0 = time.time()
    X_sel_train = selector.transform(X_train)
    X_sel_test = selector.transform(X_test)
    t_transform = time.time() - t0

    model = LogisticRegression(max_iter=500, random_state=0)
    cv_scores = cross_val_score(model, X_sel_train, y_train, cv=5, scoring="accuracy")
    model.fit(X_sel_train, y_train)
    test_acc = model.score(X_sel_test, y_test)

    return {
        "method": method_name,
        "n_features": X_sel_train.shape[1],
        "cv_accuracy": cv_scores.mean(),
        "test_accuracy": test_acc,
        "fit_time_s": t_transform,
    }
```

### 5.2 Random Search Baseline

Random search provides a strong baseline — it is often surprisingly competitive because the search space is so large that any directed search method has high variance.

```python
def random_search_feature_selection(X: np.ndarray, y: np.ndarray,
                                    n_trials: int = 200,
                                    fitness_fn=None) -> np.ndarray:
    """
    Random search baseline: evaluate n_trials random chromosomes.
    Returns best chromosome found.
    """
    if fitness_fn is None:
        from .fitness import cv_fitness as fitness_fn

    best_chrom = None
    best_fitness = -np.inf

    for _ in range(n_trials):
        chrom = (np.random.random(X.shape[1]) > 0.5).astype(np.int8)
        if chrom.sum() == 0:
            chrom[0] = 1
        f = fitness_fn(chrom, X, y)
        if f > best_fitness:
            best_fitness = f
            best_chrom = chrom.copy()

    return best_chrom
```

**Important**: random search with $n_{\text{trials}} = N \times T$ evaluations (same budget as GA) is a fair comparison. GAs typically outperform random search on structured problems but the margin varies.

### 5.3 Exhaustive Search (Small $p$ Only)

For $p \leq 20$, exhaustive search over all $2^p$ subsets is feasible:

```python
from itertools import product

def exhaustive_search(X: np.ndarray, y: np.ndarray,
                      fitness_fn=None) -> np.ndarray:
    """
    Exhaustive search over all 2^p feature subsets.
    Only feasible for p <= 20.
    """
    p = X.shape[1]
    assert p <= 20, f"Exhaustive search infeasible for p={p}"

    best_chrom = None
    best_fitness = -np.inf

    for bits in product([0, 1], repeat=p):
        chrom = np.array(bits, dtype=np.int8)
        if chrom.sum() == 0:
            continue
        f = fitness_fn(chrom, X, y)
        if f > best_fitness:
            best_fitness = f
            best_chrom = chrom.copy()

    return best_chrom
```

### 5.4 Expected Performance on the Same Problem

For a 30-feature classification problem with 5-fold CV fitness, typical results:

| Method | N evaluations | Solution quality | Time |
|:---|:---:|:---:|:---:|
| Random search | 5,000 | Baseline | Fast |
| Greedy forward | ~30×k | Good locally | Moderate |
| GA (N=50, T=100) | ~1,500 (cached) | Often best | Moderate |
| Exhaustive ($p=30$) | $2^{30} \approx 10^9$ | Optimal | Infeasible |

GAs tend to outperform greedy methods on problems with feature interactions, and random search when the landscape has clear structure (fitness-distance correlation > 0.1).

---

## 6. DEAP Framework Integration

DEAP (Distributed Evolutionary Algorithms in Python) provides a flexible EA framework. The key components map directly to our custom implementation.

### 6.1 Installation and Setup

```bash
pip install deap
```

### 6.2 Complete DEAP GA for Feature Selection

```python
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target
N_FEATURES = X.shape[1]
PARSIMONY_WEIGHT = 0.01

# ------------------------------------------------------------------ #
# DEAP type registration                                               #
# ------------------------------------------------------------------ #
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Chromosome: each gene is 0 or 1
toolbox.register("attr_bool", np.random.randint, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_bool, N_FEATURES)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# ------------------------------------------------------------------ #
# Fitness function with parsimony                                      #
# ------------------------------------------------------------------ #
_fitness_cache: dict[tuple, float] = {}

def evaluate_deap(individual):
    """DEAP-compatible fitness function. Returns tuple (fitness,)."""
    chrom = np.array(individual, dtype=np.int8)
    key = chrom.tobytes()
    if key in _fitness_cache:
        return (_fitness_cache[key],)

    selected = np.where(chrom == 1)[0]
    if len(selected) == 0:
        _fitness_cache[key] = -1.0
        return (-1.0,)

    model = LogisticRegression(max_iter=500, random_state=0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X[:, selected], y, cv=cv, scoring="accuracy")
    fitness = float(scores.mean()) - PARSIMONY_WEIGHT * len(selected) / N_FEATURES

    _fitness_cache[key] = fitness
    return (fitness,)

toolbox.register("evaluate", evaluate_deap)

# ------------------------------------------------------------------ #
# Operators                                                            #
# ------------------------------------------------------------------ #
toolbox.register("mate", tools.cxUniform, indpb=0.5)        # uniform crossover
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0/N_FEATURES)  # bit-flip
toolbox.register("select", tools.selTournament, tournsize=3)  # tournament

# ------------------------------------------------------------------ #
# Hall of Fame and Statistics                                          #
# ------------------------------------------------------------------ #
hof = tools.HallOfFame(5)  # track top 5 solutions ever

stats = tools.Statistics(lambda ind: ind.fitness.values[0])
stats.register("max", np.max)
stats.register("mean", np.mean)
stats.register("std", np.std)

# ------------------------------------------------------------------ #
# Run evolution                                                        #
# ------------------------------------------------------------------ #
POP_SIZE = 50
N_GENERATIONS = 100
CROSSOVER_RATE = 0.8
MUTATION_RATE = 1.0  # each individual is mutated (operator uses indpb)

population = toolbox.population(n=POP_SIZE)
population, logbook = algorithms.eaSimple(
    population, toolbox,
    cxpb=CROSSOVER_RATE,
    mutpb=MUTATION_RATE,
    ngen=N_GENERATIONS,
    stats=stats,
    halloffame=hof,
    verbose=True
)

# ------------------------------------------------------------------ #
# Results                                                             #
# ------------------------------------------------------------------ #
best = hof[0]
selected = np.where(np.array(best) == 1)[0]
print(f"\nBest solution: {len(selected)} features, fitness={best.fitness.values[0]:.4f}")
print(f"Selected features: {selected}")
print(f"\nTop 5 Hall of Fame:")
for i, ind in enumerate(hof):
    sel = np.where(np.array(ind) == 1)[0]
    print(f"  {i+1}. {len(sel)} features, fitness={ind.fitness.values[0]:.4f}")
```

### 6.3 DEAP Custom Operators

DEAP makes it easy to plug in custom operators:

```python
# Feature-count-preserving mutation (swap mutation)
def deap_swap_mutate(individual, n_swaps=1):
    chrom = np.array(individual, dtype=np.int8)
    selected = np.where(chrom == 1)[0]
    unselected = np.where(chrom == 0)[0]
    if len(selected) > 0 and len(unselected) > 0:
        n = min(n_swaps, len(selected), len(unselected))
        drop = np.random.choice(selected, n, replace=False)
        add = np.random.choice(unselected, n, replace=False)
        chrom[drop] = 0
        chrom[add] = 1
    for i, val in enumerate(chrom):
        individual[i] = int(val)
    return (individual,)

toolbox.register("mutate_swap", deap_swap_mutate, n_swaps=2)
```

### 6.4 DEAP Parallelisation

DEAP integrates with Python's `multiprocessing` for parallel fitness evaluation:

```python
import multiprocessing

if __name__ == "__main__":
    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    population = toolbox.population(n=POP_SIZE)
    population, logbook = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.8, mutpb=1.0, ngen=100,
        stats=stats, halloffame=hof, verbose=True
    )
    pool.close()
```

**Speedup**: near-linear with number of cores for CPU-bound fitness functions. Note that the fitness cache cannot be shared across processes — each process maintains its own cache.

### 6.5 DEAP vs Custom Implementation

| Aspect | Custom (Guide 01) | DEAP |
|:---|:---:|:---:|
| Learning overhead | None | Moderate |
| Flexibility | Maximum | High |
| Parallelism | Manual | Built-in |
| Algorithms available | 1 (eaSimple) | Many (NSGA-II, CMA-ES, ...) |
| Debugging | Easier | Moderate |
| Production use | ✓ | ✓ |

**Recommendation**: use the custom implementation for learning and full control; use DEAP when you need multi-objective algorithms (NSGA-II for Pareto front), island models, or easy parallelism.

---

## 7. Complete Parameter Tuning Workflow

```python
import itertools
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def parameter_sweep(X_train, y_train, X_val, y_val, param_grid: dict) -> pd.DataFrame:
    """
    Systematic parameter sweep for GA feature selection.

    Parameters
    ----------
    param_grid : dict
        Keys: parameter names. Values: lists of values to try.
        e.g., {'pop_size': [30, 50], 'mutation_rate': [0.01, 0.05]}

    Returns
    -------
    results : DataFrame
        One row per parameter combination with performance metrics.
    """
    from .ga_selector import GAFeatureSelector, GAConfig
    from .fitness import cv_fitness

    keys = list(param_grid.keys())
    values = list(param_grid.values())
    results = []

    for combo in itertools.product(*values):
        params = dict(zip(keys, combo))
        cfg = GAConfig(**params)
        selector = GAFeatureSelector(cfg)

        t0 = __import__("time").time()
        selector.fit(X_train, y_train)
        fit_time = __import__("time").time() - t0

        X_val_sel = X_val[:, selector.selected_features_]
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=500, random_state=0)
        model.fit(X_train[:, selector.selected_features_], y_train)
        val_acc = model.score(X_val_sel, y_val)

        results.append({
            **params,
            "n_features": selector.best_individual_.n_selected(),
            "best_fitness": selector.best_individual_.fitness,
            "val_accuracy": val_acc,
            "fit_time_s": fit_time,
        })
        print(f"  {params} → val_acc={val_acc:.4f}, n_feat={results[-1]['n_features']}")

    return pd.DataFrame(results)


# Example usage
if __name__ == "__main__":
    data = load_breast_cancer()
    X_tr, X_te, y_tr, y_te = train_test_split(
        StandardScaler().fit_transform(data.data), data.target,
        test_size=0.2, random_state=42, stratify=data.target
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=0, stratify=y_tr
    )

    grid = {
        "pop_size": [30, 50, 100],
        "mutation_rate": [None, 0.05, 0.1],  # None = 1/p
        "crossover_rate": [0.7, 0.9],
        "parsimony_weight": [0.005, 0.01, 0.05],
    }
    results_df = parameter_sweep(X_tr, y_tr, X_val, y_val, grid)
    print(results_df.sort_values("val_accuracy", ascending=False).head(10))
```

---

## Key Takeaways

1. **Convergence detection** requires both fitness plateau monitoring and diversity metrics — use both.
2. **Hamming diversity** and genotypic entropy are complementary: diversity catches chromosome-level collapse, entropy catches gene-level fixation.
3. **Crowding** preserves multiple niches without a fixed radius; **fitness sharing** penalises crowded regions.
4. **Island models** are the most effective diversity strategy and provide natural parallelism.
5. **Premature convergence remedy priority**: (1) reduce selection pressure, (2) boost mutation, (3) inject immigrants, (4) restart with elites.
6. **Population size** should scale with $p$ (features): $N \approx \max(50, 1.5p)$ is a practical rule.
7. **Mutation rate $1/p$** with adaptive adjustment based on diversity outperforms any fixed rate.
8. **DEAP** provides NSGA-II for Pareto-front feature selection and built-in parallelism.
9. **Random search with equal budget** is the minimal baseline every GA implementation should beat.
10. **Systematic parameter sweeps** on a validation set (separate from test) prevent test set overfitting in the tuning process.

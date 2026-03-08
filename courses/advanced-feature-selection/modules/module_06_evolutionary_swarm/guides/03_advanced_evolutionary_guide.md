# Advanced Evolutionary Methods for Feature Selection

## Memetic Algorithms: Hybrid Global-Local Search

Memetic algorithms (MAs) combine population-based global search (GA, PSO, DE) with individual-level local search. The term was coined by Moscato (1989), inspired by Dawkins' "meme" — a unit of cultural learning that individuals refine through their own experience before passing on.

### Why Hybridise?

Global search (GA) finds promising regions of the search space but is coarse — it does not exploit a good solution found by crossover/mutation. Local search refines solutions within the neighbourhood of a promising point but is myopic — it terminates at the first local optimum. Combining them avoids both failure modes.

**Memetic algorithm structure**:
```
Initialise population
Evaluate fitness
While termination criterion not met:
    Apply global search operators (selection, crossover, mutation)
    Evaluate offspring
    Apply local search to each (or a fraction of) offspring
    Re-evaluate after local improvement
    Select survivors (elitist or generational)
```

### Local Search Options for Feature Selection

**Hill Climbing (HC)**: single-step bit-flip search. At each step, evaluate all single-bit-flip neighbours, move to the best improvement.

- Time per call: $O(D \cdot \text{eval\_cost})$
- Terminates at local optima (no worse-solution acceptance)
- Fast but shallow

```python
def hill_climb(individual: list, evaluate_fn, max_steps: int = 50) -> list:
    """Greedy bit-flip hill climbing for binary feature selection."""
    current = list(individual)
    current_fit = evaluate_fn(current)

    for _ in range(max_steps):
        improved = False
        # Try flipping each bit
        for j in range(len(current)):
            neighbour = list(current)
            neighbour[j] = 1 - neighbour[j]
            if sum(neighbour) == 0:
                continue  # require at least one feature
            neighbour_fit = evaluate_fn(neighbour)
            if neighbour_fit < current_fit:
                current = neighbour
                current_fit = neighbour_fit
                improved = True
                break  # first-improvement strategy
        if not improved:
            break  # local optimum

    return current
```

**Simulated Annealing (SA)**: probabilistic acceptance of worse solutions. Allows escape from local optima by accepting worse solutions with probability $\exp(-\Delta f / T)$ where $T$ is the "temperature", decayed over iterations.

```python
import math
import random

def simulated_annealing(
    individual: list,
    evaluate_fn,
    T_init: float = 1.0,
    T_min: float = 0.01,
    cooling_rate: float = 0.95,
    max_steps: int = 200,
    random_state: int = 42,
) -> list:
    """Simulated annealing local search for binary feature selection."""
    rng = random.Random(random_state)
    current = list(individual)
    current_fit = evaluate_fn(current)
    best = list(current)
    best_fit = current_fit
    T = T_init

    step = 0
    while T > T_min and step < max_steps:
        j = rng.randint(0, len(current) - 1)
        neighbour = list(current)
        neighbour[j] = 1 - neighbour[j]
        if sum(neighbour) == 0:
            step += 1
            continue
        neighbour_fit = evaluate_fn(neighbour)
        delta = neighbour_fit - current_fit

        if delta < 0 or rng.random() < math.exp(-delta / T):
            current = neighbour
            current_fit = neighbour_fit

        if current_fit < best_fit:
            best = list(current)
            best_fit = current_fit

        T *= cooling_rate
        step += 1

    return best
```

### Lamarckian vs Baldwinian Learning

Two philosophies for incorporating local search results back into the population:

**Lamarckian**: replace the individual in the population with the locally improved version. The chromosome changes — improved alleles are directly inherited by offspring.

**Baldwinian**: use the locally improved fitness for selection but do **not** replace the chromosome. The individual retains its original genotype but competes for survival using its improved phenotype fitness.

| Aspect | Lamarckian | Baldwinian |
|---|---|---|
| Fitness used for selection | Local optimum fitness | Local optimum fitness |
| Chromosome passed to offspring | Locally improved | Original (unmodified) |
| Convergence speed | Faster | Slower |
| Diversity preservation | Lower | Higher |
| Biological justification | None (Lamarck was wrong) | Natural selection |

**Practical recommendation**: Lamarckian is more common in feature selection because convergence speed matters more than biological fidelity. Baldwinian is preferred when premature convergence is a known problem.

---

## Cooperative Co-Evolution

**Reference**: Potter & De Jong (1994). *A cooperative coevolutionary approach to function optimization*. PPSN, 249–257.

### Motivation

For $D > 100$ features, a single population searching the full $2^D$ space has poor density. Cooperative co-evolution (CC) decomposes the feature set into $k$ groups (subcomponents), evolves a separate population for each subcomponent, and evaluates fitness using a **collaboration** with the best-known solution from each other subcomponent.

### CC Algorithm for Feature Selection

```
Decompose features into k subsets: G_1, G_2, ..., G_k
Initialise k populations: P_1, P_2, ..., P_k
Initialise representative solutions: r_1, r_2, ..., r_k

For each cycle:
    For each subpopulation P_i:
        For each individual x_i in P_i:
            Construct full solution: s = merge(x_i, r_1, ..., r_{i-1}, r_{i+1}, ..., r_k)
            Evaluate fitness of s
        Apply selection, crossover, mutation to P_i
        Update representative r_i = best individual in P_i
```

### Decomposition Strategies

| Strategy | How | When |
|---|---|---|
| **Static random** | Randomly split features into $k$ equal groups | Simple baseline |
| **Static domain** | Split by feature type (e.g., technical vs fundamental) | Domain knowledge available |
| **Dynamic** | Re-partition every $N$ generations | Features with high interaction detected |
| **Interaction-based** | Group features with high mutual information | Correlated features |

```python
import numpy as np
from typing import List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)
N_FEATURES = X_scaled.shape[1]

def evaluate_full_mask(mask: np.ndarray) -> float:
    if not mask.any():
        return 1.0
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled[:, mask], y, cv=cv, scoring="accuracy")
    return 1.0 - scores.mean()

def cooperative_coevolution_feature_selection(
    n_subpops: int = 3,
    subpop_size: int = 20,
    n_cycles: int = 10,
    n_generations_per_cycle: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)

    # Random static decomposition
    feature_indices = np.arange(N_FEATURES)
    rng.shuffle(feature_indices)
    subgroups = np.array_split(feature_indices, n_subpops)

    # Initialise sub-populations (binary)
    subpops = [
        rng.integers(0, 2, size=(subpop_size, len(sg)))
        for sg in subgroups
    ]

    # Initialise representatives (best individual per subpopulation)
    representatives = [
        rng.integers(0, 2, size=len(sg)) for sg in subgroups
    ]

    def build_full_mask(subpop_idx: int, individual: np.ndarray) -> np.ndarray:
        """Merge this individual with current representatives of other subpops."""
        full_mask = np.zeros(N_FEATURES, dtype=bool)
        for k, (sg, rep) in enumerate(zip(subgroups, representatives)):
            bits = individual if k == subpop_idx else rep
            full_mask[sg] = bits.astype(bool)
        return full_mask

    global_best_mask = None
    global_best_fit = np.inf

    for cycle in range(n_cycles):
        for subpop_idx in range(n_subpops):
            sg = subgroups[subpop_idx]
            pop = subpops[subpop_idx]

            # Evaluate all individuals via collaboration
            fitness = np.array([
                evaluate_full_mask(build_full_mask(subpop_idx, ind))
                for ind in pop
            ])

            # Simple (1+1) EA for each subpopulation
            for _ in range(n_generations_per_cycle):
                for i in range(subpop_size):
                    # Bit-flip mutation (p=1/subgroup_size)
                    mutant = pop[i].copy()
                    flip_prob = 1.0 / len(sg)
                    flip_mask = rng.random(len(sg)) < flip_prob
                    mutant[flip_mask] ^= 1
                    if not mutant.any():
                        mutant[rng.integers(len(sg))] = 1

                    mut_fit = evaluate_full_mask(
                        build_full_mask(subpop_idx, mutant)
                    )
                    if mut_fit < fitness[i]:
                        pop[i] = mutant
                        fitness[i] = mut_fit

            subpops[subpop_idx] = pop

            # Update representative
            best_in_subpop = pop[np.argmin(fitness)]
            representatives[subpop_idx] = best_in_subpop

            # Check global best
            full_mask = build_full_mask(subpop_idx, best_in_subpop)
            best_fit_here = fitness.min()
            if best_fit_here < global_best_fit:
                global_best_fit = best_fit_here
                global_best_mask = full_mask.copy()

        print(f"Cycle {cycle+1:2d} | best_error={global_best_fit:.4f}")

    return global_best_mask
```

---

## CMA-ES: Covariance Matrix Adaptation

**Reference**: Hansen & Ostermeier (2001). *Completely Derandomized Self-Adaptation in Evolution Strategies*. Evolutionary Computation, 9(2), 159–195.

CMA-ES is an evolution strategy that adapts a full covariance matrix over the search space, learning the correlation structure among variables. It is one of the most powerful algorithms for continuous black-box optimisation.

### Continuous Relaxation for Feature Selection

Feature selection is inherently discrete, but CMA-ES operates on continuous vectors. The continuous relaxation: evolve a vector $\boldsymbol{\mu} \in \mathbb{R}^D$; convert to binary via threshold after sigmoid:

$$x_j = \begin{cases} 1 & \text{if } \sigma(z_j) > 0.5 \\ 0 & \text{otherwise} \end{cases}, \quad z_j \sim \mathcal{N}(\mu_j, \mathbf{C})$$

CMA-ES learns which feature combinations tend to improve fitness by adapting $\mathbf{C}$ — the covariance matrix captures feature interactions.

```python
# Using the cma library: pip install cma
import cma
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)
N_FEATURES = X_scaled.shape[1]

def cma_fitness(z: list) -> float:
    """CMA-ES fitness function: continuous vector to binary mask."""
    z_arr = np.array(z)
    mask = (1.0 / (1.0 + np.exp(-z_arr))) > 0.5
    if not mask.any():
        return 1.0
    clf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled[:, mask], y, cv=cv, scoring="accuracy")
    return 1.0 - scores.mean()

# CMA-ES: start from zero (neutral prior) with sigma=0.5
x0 = [0.0] * N_FEATURES
sigma0 = 0.5
opts = cma.CMAOptions()
opts.set("maxiter", 100)
opts.set("seed", 42)
opts.set("verbose", -9)  # suppress output

es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
while not es.stop():
    solutions = es.ask()
    fitnesses = [cma_fitness(z) for z in solutions]
    es.tell(solutions, fitnesses)

best_z = es.result.xbest
best_mask = (1.0 / (1.0 + np.exp(-np.array(best_z)))) > 0.5
print(f"CMA-ES best: error={es.result.fbest:.4f}, "
      f"features={best_mask.sum()}/{N_FEATURES}")
```

### Why CMA-ES for Feature Selection?

CMA-ES adapts the search distribution to the fitness landscape, learning which directions (feature combinations) lead to improvement. This is particularly useful when features are highly correlated — CMA-ES can learn to jointly select/deselect correlated features.

**Limitation**: CMA-ES is designed for continuous spaces. The sigmoid mapping introduces a discretisation step that loses gradient information. For large $D$ (> 200), the $D \times D$ covariance matrix update becomes expensive ($O(D^2)$ per generation).

---

## Natural Evolution Strategies (NES) and OpenAI-ES

**Reference**: Wierstra et al. (2014). *Natural evolution strategies*. JMLR, 15, 949–980.

NES parameterises the search distribution as $p(\mathbf{z} | \boldsymbol{\theta})$ and follows the **natural gradient** of expected fitness:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t + \eta \cdot \mathbf{F}^{-1} \nabla_{\boldsymbol{\theta}} \mathbb{E}[f(\mathbf{z})]$$

where $\mathbf{F}$ is the Fisher information matrix (the natural gradient adapts the step to the geometry of the parameter space).

**OpenAI-ES** (Salimans et al., 2017) simplifies NES by using a fixed isotropic Gaussian and estimating the gradient via perturbations:

$$\nabla_{\boldsymbol{\mu}} \mathbb{E}[f] \approx \frac{1}{n\sigma} \sum_{i=1}^n f(\boldsymbol{\mu} + \sigma \boldsymbol{\epsilon}_i) \boldsymbol{\epsilon}_i, \quad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, \mathbf{I})$$

Applied to feature selection with sigmoid mapping, OpenAI-ES is embarrassingly parallel (each $\boldsymbol{\epsilon}_i$ evaluation is independent) and scales to large $D$.

---

## Surrogate-Assisted Evolutionary Optimisation

**Motivation**: CV evaluation is expensive ($O(k \cdot \text{model training time})$ per individual). Surrogate models replace the expensive true fitness function with a cheap approximation during most evaluations.

### Architecture

```
Population
    │
    ├──► True fitness (expensive): evaluate EVERY ~20 generations
    │                              or the best K individuals
    │
    └──► Surrogate fitness (cheap): evaluate all individuals
         - Random Forest regressor of true fitness
         - Gaussian Process (uncertainty estimates)
         - XGBoost
```

### Surrogate-Assisted GA

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
import random

data = load_breast_cancer()
X, y = data.data, data.target
X_scaled = StandardScaler().fit_transform(X)
N_FEATURES = X_scaled.shape[1]

def true_fitness(individual: list) -> float:
    mask = np.array(individual, dtype=bool)
    if not mask.any():
        return 1.0
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_scaled[:, mask], y, cv=cv, scoring="accuracy")
    return 1.0 - scores.mean()

def surrogate_assisted_ga(
    population_size: int = 50,
    n_generations: int = 30,
    surrogate_update_interval: int = 5,
    elite_fraction: float = 0.2,
    random_state: int = 42,
) -> list:
    rng = random.Random(random_state)
    np_rng = np.random.default_rng(random_state)

    # Initialise population
    population = [
        [rng.randint(0, 1) for _ in range(N_FEATURES)]
        for _ in range(population_size)
    ]

    # Initial true fitness evaluation
    true_archive_X = []
    true_archive_y = []
    fitness = []
    for ind in population:
        f = true_fitness(ind)
        fitness.append(f)
        true_archive_X.append(ind[:])
        true_archive_y.append(f)

    surrogate = RandomForestRegressor(n_estimators=50, random_state=42)
    surrogate.fit(true_archive_X, true_archive_y)

    best_solution = population[np.argmin(fitness)]
    best_fit = min(fitness)

    for gen in range(n_generations):
        # Generate offspring
        new_population = []
        for _ in range(population_size):
            # Tournament selection
            p1, p2 = rng.sample(range(population_size), 2)
            parent1 = population[p1] if fitness[p1] < fitness[p2] else population[p2]
            p3, p4 = rng.sample(range(population_size), 2)
            parent2 = population[p3] if fitness[p3] < fitness[p4] else population[p4]

            # Single-point crossover
            cx_point = rng.randint(1, N_FEATURES - 1)
            child = parent1[:cx_point] + parent2[cx_point:]

            # Bit-flip mutation
            child = [1 - b if rng.random() < 1.0 / N_FEATURES else b for b in child]
            if sum(child) == 0:
                child[rng.randint(0, N_FEATURES - 1)] = 1
            new_population.append(child)

        # Evaluate new population using surrogate
        surrogate_fitness = surrogate.predict(new_population)

        # Select elite fraction for true fitness evaluation
        n_elite = max(2, int(elite_fraction * population_size))
        elite_indices = np.argsort(surrogate_fitness)[:n_elite]

        true_fitness_values = {}
        for idx in elite_indices:
            f = true_fitness(new_population[idx])
            true_fitness_values[idx] = f
            true_archive_X.append(new_population[idx][:])
            true_archive_y.append(f)

        # Update surrogate periodically
        if gen % surrogate_update_interval == 0 and len(true_archive_y) > 10:
            surrogate.fit(true_archive_X, true_archive_y)

        # Combine surrogate and true fitness for population update
        combined_fitness = []
        for i, ind in enumerate(new_population):
            if i in true_fitness_values:
                combined_fitness.append(true_fitness_values[i])
            else:
                combined_fitness.append(float(surrogate.predict([ind])[0]))

        # Elitist selection: keep best from old + new
        all_inds = population + new_population
        all_fit = fitness + combined_fitness
        top_indices = np.argsort(all_fit)[:population_size]
        population = [all_inds[i] for i in top_indices]
        fitness = [all_fit[i] for i in top_indices]

        if fitness[0] < best_fit:
            best_fit = fitness[0]
            best_solution = population[0][:]

        n_true_evals = len(true_fitness_values)
        print(f"Gen {gen:3d} | best={best_fit:.4f} | "
              f"true_evals={n_true_evals}/{population_size}")

    return best_solution
```

**Savings**: instead of `population_size × n_generations` true evaluations, surrogate-assisted GA uses `population_size + n_elite × n_generations` ≈ 4–5× fewer expensive evaluations.

---

## Estimation of Distribution Algorithms (EDA)

EDAs replace crossover/mutation with explicit probability model estimation and sampling.

### PBIL (Population-Based Incremental Learning)

Maintains a probability vector $\mathbf{p} \in [0,1]^D$ where $p_j$ = probability of feature $j$ being selected:

```python
import numpy as np

def pbil_feature_selection(
    X: np.ndarray, y: np.ndarray,
    n_samples: int = 50,
    n_generations: int = 50,
    learning_rate: float = 0.1,
    mutation_prob: float = 0.02,
    mutation_shift: float = 0.05,
    random_state: int = 42,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    D = X.shape[1]
    prob_vector = np.full(D, 0.5)  # uniform initialisation

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    def evaluate(mask):
        if not mask.any():
            return 1.0
        clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        return 1.0 - cross_val_score(clf, X[:, mask], y, cv=cv, scoring="accuracy").mean()

    best_mask = None
    best_fit = np.inf

    for gen in range(n_generations):
        # Sample population from current probability vector
        samples = (rng.random((n_samples, D)) < prob_vector).astype(bool)
        fitness = np.array([evaluate(s) for s in samples])

        # Find best sample
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_fit:
            best_fit = fitness[best_idx]
            best_mask = samples[best_idx].copy()

        # Update probability vector toward best sample (Lamarckian)
        best_sample = samples[best_idx].astype(float)
        prob_vector = (
            (1 - learning_rate) * prob_vector + learning_rate * best_sample
        )

        # Mutation: random perturbation of probability vector
        mut_mask = rng.random(D) < mutation_prob
        prob_vector[mut_mask] = (
            prob_vector[mut_mask] * (1 - mutation_shift)
            + rng.random(mut_mask.sum()) * mutation_shift
        )
        prob_vector = np.clip(prob_vector, 0.05, 0.95)

        if gen % 10 == 0:
            print(f"Gen {gen:3d} | best_error={best_fit:.4f}, "
                  f"prob_mean={prob_vector.mean():.3f}")

    return best_mask
```

**UMDA** (Univariate Marginal Distribution Algorithm): estimates a factored probability model from the top-$k$ solutions each generation. Assumes feature independence.

**BOA** (Bayesian Optimisation Algorithm): learns a Bayesian network over features, capturing pairwise and higher-order dependencies. More powerful but computationally expensive.

---

## Island Models and Migration

Island models run multiple sub-populations in parallel ("islands"), each using potentially different operators or parameter settings:

```
Island 1 (GA, p_c=0.8)    Island 2 (PSO)    Island 3 (DE/best/1)
         │                       │                    │
         └───────────────────────┘                    │
                   Migration every K generations      │
                   (best M individuals) ──────────────┘
```

**Migration policies**:
- **Ring**: island sends to next island in ring
- **Random**: select random destination
- **Best-first**: send to the island with the best current solution

**Migration parameters**:
- **Interval** ($K$): every $K$ generations (typical: 10–50)
- **Migration rate** ($m$): fraction of population migrated (typical: 5–10%)
- **Selection**: random migrants vs elite migrants

**Heterogeneous islands** improve diversity: different operators prevent all islands from converging to the same local optimum.

```python
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def run_island(island_config: dict) -> dict:
    """Run one island's evolution and return its best solution."""
    # Each island runs independently for K generations
    algorithm = island_config["algorithm"]  # "ga", "pso", or "de"
    results = algorithm(**island_config["params"])
    return {"best_solution": results["best"], "best_fitness": results["fitness"]}

def island_model(
    n_islands: int = 4,
    n_cycles: int = 10,
    migration_interval: int = 5,
    migration_size: int = 2,
):
    """Island model with heterogeneous operators and ring migration."""
    # Each island has its own population and operator configuration
    island_populations = [
        np.random.default_rng(seed=i).integers(0, 2, size=(40, N_FEATURES))
        for i in range(n_islands)
    ]

    for cycle in range(n_cycles):
        # Evolve each island for migration_interval generations
        # (in practice, call island-specific evolve functions here)

        # Ring migration: send best migration_size individuals to next island
        migrants = []
        for i in range(n_islands):
            # Select elite migrants from island i
            migrants.append(island_populations[i][:migration_size].copy())

        for i in range(n_islands):
            next_island = (i + 1) % n_islands
            # Replace worst individuals with incoming migrants
            island_populations[next_island][-migration_size:] = migrants[i]

        print(f"Cycle {cycle+1} complete — migration applied")
```

---

## Key Takeaways

- **Memetic algorithms** combine global (GA/PSO) search with local search (hill climbing, SA). Lamarckian learning (replace chromosome) converges faster; Baldwinian (keep original chromosome) preserves diversity.
- **Cooperative co-evolution** decomposes the feature space into independently evolving subpopulations. Effective for $D > 100$ when features have weak inter-group dependencies.
- **CMA-ES** applies a powerful continuous optimiser to feature selection via sigmoid mapping. Learns feature interaction structure through the covariance matrix.
- **NES/OpenAI-ES** follow the natural gradient of expected fitness — embarrassingly parallel and scale to large $D$.
- **Surrogate-assisted evolution** replaces expensive CV evaluation with a cheap model for most evaluations, typically achieving 4–5× speedup with minimal quality loss.
- **EDAs** (PBIL, UMDA, BOA) build explicit probabilistic models of good solutions instead of using genetic operators.
- **Island models** run heterogeneous sub-populations in parallel with periodic migration, combining diversity and quality.

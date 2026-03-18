# Genetic Algorithm Components: The Complete Framework

## In Brief

A genetic algorithm for feature selection consists of six interacting components: initialization, fitness evaluation, selection, crossover, mutation, and replacement. These components form a cycle that repeats for a fixed number of generations or until a convergence criterion is met. Understanding how each component works and how to configure it is the foundation for building effective GAs in practice.

## Key Insight

The GA components do not function independently — they interact as a system. Strong selection pressure combined with low mutation creates fast but brittle convergence. Weak pressure with high mutation creates slow, diffuse search. The art of GA parameter setting is balancing exploitation (using what you already know is good) against exploration (discovering new, potentially better regions).

## Formal Definition

Given:
- Feature matrix $X \in \mathbb{R}^{n \times p}$, target $y \in \mathbb{R}^n$
- Population size $N$, max generations $G$
- Crossover probability $p_c$, mutation probability $p_m$
- Elitism count $e$

The GA minimizes $f(s) = \text{CV\_Error}(s) + \lambda \|s\|_0$ over $s \in \{0,1\}^p$ by the following iteration:

$$\mathcal{P}_{t+1} = \text{Replace}(\mathcal{P}_t, \text{Mutate}(\text{Crossover}(\text{Select}(\mathcal{P}_t))))$$

with evaluation $f(s)$ applied before each selection step.

## Intuitive Explanation

Think of a GA as an evolution simulation of competing feature teams:

1. **Initialization**: Recruit 50 random feature teams from the available pool.
2. **Evaluation**: Test each team's predictive power on held-out data.
3. **Selection**: The best-performing teams are chosen to "reproduce" — their feature combinations will influence the next generation.
4. **Crossover**: Merge two selected teams, taking some features from each parent.
5. **Mutation**: Randomly swap out a feature to try something new.
6. **Replacement**: The best teams from the old and new generations survive to form the next generation's starting lineup.

After 50–200 such rounds, the surviving teams represent the best feature combinations the search has discovered.

## Code Implementation

```python
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge


# ─── CORE DATA STRUCTURES ─────────────────────────────────────────────────────

@dataclass
class Individual:
    """A binary-encoded feature selection candidate with its fitness."""
    chromosome: np.ndarray   # dtype int8, values in {0, 1}
    fitness: Optional[float] = None

    @property
    def selected_features(self) -> List[int]:
        return np.where(self.chromosome == 1)[0].tolist()

    @property
    def num_features(self) -> int:
        return int(self.chromosome.sum())

    def copy(self) -> "Individual":
        return Individual(
            chromosome=self.chromosome.copy(),
            fitness=self.fitness,
        )


class Population:
    """A collection of individuals with factory methods."""

    def __init__(self, individuals: List[Individual]):
        self.individuals = individuals

    @classmethod
    def random(
        cls,
        pop_size: int,
        n_features: int,
        init_prob: float = 0.5,
        min_features: int = 1,
        rng: Optional[np.random.Generator] = None,
    ) -> "Population":
        """Create a random population, guaranteeing min_features per individual."""
        rng = rng or np.random.default_rng()
        individuals = []
        for _ in range(pop_size):
            chromosome = (rng.random(n_features) < init_prob).astype(np.int8)
            while chromosome.sum() < min_features:
                chromosome[rng.integers(n_features)] = 1
            individuals.append(Individual(chromosome=chromosome))
        return cls(individuals)

    def __len__(self) -> int:
        return len(self.individuals)

    def best(self) -> Individual:
        """Return the individual with the lowest (best) fitness."""
        return min(
            [ind for ind in self.individuals if ind.fitness is not None],
            key=lambda x: x.fitness,
        )

    def sorted_individuals(self) -> List[Individual]:
        return sorted(
            [ind for ind in self.individuals if ind.fitness is not None],
            key=lambda x: x.fitness,
        )


# ─── COMPONENT 1: EVALUATION ──────────────────────────────────────────────────

def evaluate_individual(
    individual: Individual,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    cv_folds: int = 5,
    feature_penalty: float = 0.01,
) -> float:
    """
    Evaluate a feature subset using cross-validated MSE plus complexity penalty.

    Returns infinity for empty selections to prevent crashes.
    """
    selected = individual.selected_features
    if len(selected) == 0:
        return float("inf")

    X_selected = X[:, selected]
    model = model_fn()
    scores = cross_val_score(
        model, X_selected, y, cv=cv_folds,
        scoring="neg_mean_squared_error"
    )
    error = -scores.mean()
    penalty = feature_penalty * len(selected) / len(individual.chromosome)
    return error + penalty


def evaluate_population(
    population: Population,
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    cv_folds: int = 5,
    cache: Optional[dict] = None,
) -> None:
    """Evaluate all unevaluated individuals in-place. Uses caching if provided."""
    for ind in population.individuals:
        if ind.fitness is not None:
            continue   # Already evaluated (e.g., elite from previous generation)

        key = tuple(ind.chromosome.tolist()) if cache is not None else None
        if cache is not None and key in cache:
            ind.fitness = cache[key]
        else:
            ind.fitness = evaluate_individual(ind, X, y, model_fn, cv_folds)
            if cache is not None:
                cache[key] = ind.fitness


# ─── COMPONENT 2: SELECTION ───────────────────────────────────────────────────

def tournament_selection(
    population: Population,
    k: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """Select one parent by tournament: best of k random individuals."""
    rng = rng or np.random.default_rng()
    evaluated = [ind for ind in population.individuals if ind.fitness is not None]
    k = min(k, len(evaluated))
    tournament = [evaluated[i] for i in rng.choice(len(evaluated), k, replace=False)]
    return min(tournament, key=lambda x: x.fitness).copy()


# ─── COMPONENT 3: CROSSOVER ───────────────────────────────────────────────────

def uniform_crossover(
    parent1: Individual,
    parent2: Individual,
    crossover_prob: float = 0.8,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Individual, Individual]:
    """Uniform crossover: each gene independently from either parent."""
    rng = rng or np.random.default_rng()
    if rng.random() > crossover_prob:
        return parent1.copy(), parent2.copy()

    n = len(parent1.chromosome)
    mask = rng.integers(0, 2, n, dtype=bool)
    c1 = np.where(mask, parent2.chromosome, parent1.chromosome)
    c2 = np.where(mask, parent1.chromosome, parent2.chromosome)
    return Individual(c1.astype(np.int8)), Individual(c2.astype(np.int8))


# ─── COMPONENT 4: MUTATION ────────────────────────────────────────────────────

def bit_flip_mutation(
    individual: Individual,
    mutation_prob: Optional[float] = None,
    min_features: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Individual:
    """Bit-flip mutation with default rate 1/n. Enforces min_features."""
    rng = rng or np.random.default_rng()
    n = len(individual.chromosome)
    rate = mutation_prob if mutation_prob is not None else 1.0 / n

    mutant = individual.copy()
    flip = rng.random(n) < rate
    mutant.chromosome = (mutant.chromosome ^ flip.astype(np.int8))

    while mutant.num_features < min_features:
        zeros = np.where(mutant.chromosome == 0)[0]
        if len(zeros) == 0:
            break
        mutant.chromosome[rng.choice(zeros)] = 1

    mutant.fitness = None
    return mutant


# ─── COMPONENT 5: REPLACEMENT ────────────────────────────────────────────────

def generational_replacement(
    old_population: Population,
    offspring: List[Individual],
    elitism: int = 2,
) -> Population:
    """
    Replace population with offspring, keeping 'elitism' best individuals.

    Elitism ensures the best solution found so far is never discarded.
    """
    elite = old_population.sorted_individuals()[:elitism]
    elite_copies = [ind.copy() for ind in elite]

    new_size = len(old_population)
    fill_count = new_size - elitism
    new_individuals = elite_copies + offspring[:fill_count]
    return Population(new_individuals)


def steady_state_replacement(
    population: Population,
    offspring: List[Individual],
) -> Population:
    """
    Merge population and offspring, keep the best N.

    More conservative than generational replacement — good solutions
    from any generation survive as long as they remain competitive.
    """
    combined = population.individuals + offspring
    combined_sorted = sorted(
        [ind for ind in combined if ind.fitness is not None],
        key=lambda x: x.fitness,
    )
    return Population(combined_sorted[:len(population)])


# ─── COMPLETE GA LOOP ─────────────────────────────────────────────────────────

def run_ga(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable = None,
    pop_size: int = 50,
    n_generations: int = 100,
    tournament_size: int = 3,
    crossover_prob: float = 0.8,
    mutation_prob: Optional[float] = None,   # None → 1/n
    elitism: int = 2,
    feature_penalty: float = 0.01,
    cv_folds: int = 5,
    random_state: int = 42,
    verbose: bool = False,
) -> dict:
    """
    Run a complete GA for feature selection.

    Returns a dict with the best individual, fitness history, and
    feature selection frequency across the final population.
    """
    if model_fn is None:
        model_fn = lambda: Ridge(alpha=1.0)

    rng = np.random.default_rng(random_state)
    n_features = X.shape[1]
    cache = {}   # Fitness cache: chromosome tuple → fitness

    population = Population.random(pop_size, n_features, rng=rng)
    evaluate_population(population, X, y, model_fn, cv_folds, cache)

    best_history = []
    avg_history = []

    for gen in range(n_generations):
        offspring = []

        while len(offspring) < pop_size - elitism:
            p1 = tournament_selection(population, k=tournament_size, rng=rng)
            p2 = tournament_selection(population, k=tournament_size, rng=rng)
            c1, c2 = uniform_crossover(p1, p2, crossover_prob, rng=rng)
            c1 = bit_flip_mutation(c1, mutation_prob, min_features=1, rng=rng)
            c2 = bit_flip_mutation(c2, mutation_prob, min_features=1, rng=rng)
            offspring.extend([c1, c2])

        # Evaluate offspring
        for ind in offspring:
            key = tuple(ind.chromosome.tolist())
            if key in cache:
                ind.fitness = cache[key]
            else:
                ind.fitness = evaluate_individual(
                    ind, X, y, model_fn, cv_folds, feature_penalty
                )
                cache[key] = ind.fitness

        population = generational_replacement(population, offspring, elitism)

        fitnesses = [
            ind.fitness for ind in population.individuals
            if ind.fitness is not None
        ]
        best_fitness = min(fitnesses)
        avg_fitness = float(np.mean(fitnesses))
        best_history.append(best_fitness)
        avg_history.append(avg_fitness)

        if verbose and gen % 10 == 0:
            best_ind = population.best()
            print(f"Gen {gen:4d}: best={best_fitness:.4f}, "
                  f"avg={avg_fitness:.4f}, "
                  f"features={best_ind.num_features}")

    best_individual = population.best()

    return {
        "best_individual": best_individual,
        "selected_features": best_individual.selected_features,
        "best_fitness": best_individual.fitness,
        "best_history": best_history,
        "avg_history": avg_history,
        "cache_size": len(cache),
    }


if __name__ == "__main__":
    from sklearn.datasets import make_regression

    X, y = make_regression(
        n_samples=300, n_features=30,
        n_informative=8, noise=20, random_state=42
    )

    result = run_ga(
        X, y,
        pop_size=50,
        n_generations=50,
        verbose=True,
    )

    print(f"\nBest solution:")
    print(f"  Selected features: {result['selected_features']}")
    print(f"  Fitness (CV MSE + penalty): {result['best_fitness']:.4f}")
    print(f"  Unique evaluations: {result['cache_size']}")
```

## Common Pitfalls

**Pitfall 1: Evaluating elite individuals from the previous generation.** Elite individuals carried over from the previous generation already have valid fitness values. Re-evaluating them wastes compute and, if CV involves any randomness, may change their fitness, undermining the purpose of elitism. Check `ind.fitness is not None` before evaluating.

**Pitfall 2: Not caching fitness values.** Duplicate chromosomes frequently appear in a GA population through crossover. Without caching, the same chromosome may be re-evaluated dozens of times across generations, each time training a model unnecessarily. A simple dictionary keyed on the chromosome tuple typically saves 20–40% of evaluations.

**Pitfall 3: Applying elitism after replacement rather than before.** The purpose of elitism is to ensure the best individuals from the current generation survive unchanged into the next. If you apply genetic operators to the elite individuals, you lose this guarantee. Always copy elite individuals before applying any operator.

**Pitfall 4: Population size too small.** A population of 10 individuals provides insufficient genetic diversity for the crossover operator to work effectively. The GA will behave like repeated random restarts of a local search. Rule of thumb: population size $\approx 5–10 \times p$ for the feature selection problem.

**Pitfall 5: Ignoring the computational budget.** Each generation requires approximately `pop_size × T(fitness)` evaluations, where `T(fitness)` is the time for one cross-validated fitness call. For a 5-fold CV with a random forest model taking 2 seconds, 50 individuals × 100 generations = 5,000 evaluations × 2s = ~2.8 hours. Plan accordingly or use parallelization.

## Connections

**Builds on:**
- Module 00: Feature selection problem formulation
- Module 01 (encoding guide): Binary chromosome representation
- Module 00 (evolutionary operators guide): Preview of all three operator types

**Leads to:**
- Module 01 (selection guide): Tournament, roulette, rank, SUS in depth
- Module 01 (genetic operators guide): Crossover and mutation in depth
- Module 02: Fitness function design with cross-validation
- Module 04 (DEAP guide): Implementing this framework using the DEAP library

**Related:**
- Evolutionary strategies (CMA-ES): real-valued counterpart
- Differential evolution: vector-difference based crossover
- Particle swarm optimization: velocity-based alternative

## Practice Problems

1. **Elitism experiment:** Run `run_ga` with `elitism=0` and `elitism=2` on the same synthetic dataset (30 features, 8 informative) with 20 repetitions each, using different random seeds. Compare the final best fitness distribution. Does elitism reduce variance in the results?

2. **Cache hit rate:** Modify `run_ga` to track the cache hit rate per generation. Plot this rate across 100 generations. Does it increase over time as the population converges?

3. **Component ablation:** Remove one component at a time and measure the impact. Run four variants: (a) full GA, (b) no crossover (mutation only), (c) no mutation (crossover only), (d) random search (no selection pressure). How does each component contribute to performance?

4. **Population size sensitivity:** Run `run_ga` with `pop_size ∈ {10, 20, 50, 100, 200}` on a 50-feature problem. For each size, record the best fitness after 100 generations and total runtime. Plot the pareto frontier of fitness vs time.

5. **Convergence diagnosis:** Plot both `best_history` and `avg_history` from `run_ga`. A healthy run shows both curves decreasing with the gap between them narrowing. Identify the generation at which the GA appears to have converged. What would you change if convergence happened in generation 5 versus generation 95?

## Further Reading

- Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press. (Accessible overview of all components.)
- De Jong, K. A. (1975). *An Analysis of the Behavior of a Class of Genetic Adaptive Systems*. PhD thesis, University of Michigan.
- Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms*. Wiley.
- Whitley, D. (1994). A genetic algorithm tutorial. *Statistics and Computing*, 4, 65–85.

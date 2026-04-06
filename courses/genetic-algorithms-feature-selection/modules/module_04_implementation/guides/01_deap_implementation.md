# Implementing GAs with DEAP

> **Reading time:** ~8 min | **Module:** 4 — Implementation | **Prerequisites:** Modules 1-3

## Introduction to DEAP

DEAP (Distributed Evolutionary Algorithms in Python) is a flexible framework for evolutionary computation.

```bash
pip install deap
```

![GA Lifecycle](./ga_lifecycle.svg)

<div class="callout-key">

**Key Concept Summary:** DEAP is not a black-box GA library -- it is a construction kit. You assemble a GA from four building blocks: **creator** (defines your data types), **toolbox** (registers your operators), **algorithms** (orchestrates the evolutionary loop), and **tools** (provides ready-made operators). Understanding this architecture lets you swap any component -- selection, crossover, mutation, fitness -- without rewriting the loop. Think of DEAP as a recipe framework: you declare the ingredients and the framework cooks them in the right order.

</div>

## DEAP Mental Model

Before writing any code, understand how DEAP's four modules work together. Each module has a distinct responsibility:

**1. Creator -- Defines *what* your GA works with.** The `creator` module creates new Python classes for your individuals and their fitness. When you call `creator.create("FitnessMin", base.Fitness, weights=(-1.0,))`, you are defining a new type that says "fitness is a single number, and lower is better." When you call `creator.create("Individual", list, fitness=creator.FitnessMin)`, you are saying "an individual is a Python list that carries a FitnessMin attribute." This is pure type definition -- no algorithms, no logic.

**2. Toolbox -- Defines *how* your GA operates.** The `toolbox` is a registry of functions. You register every operation the GA needs: how to create a random gene (`attr_bool`), how to assemble genes into an individual (`individual`), how to build a population (`population`), how to evaluate fitness (`evaluate`), how to select parents (`select`), how to cross parents (`mate`), and how to mutate offspring (`mutate`). The toolbox does not run anything -- it stores your recipes.

**3. Algorithms -- Orchestrates *when* things happen.** The `algorithms` module provides pre-built evolutionary loops like `eaSimple` (standard generational GA). The algorithm calls your toolbox functions in the canonical order: evaluate -> select -> mate -> mutate -> evaluate -> repeat. You can also write your own loop for full control.

**4. Tools -- Provides *ready-made* operators.** The `tools` module contains implementations of common operators: `selTournament`, `cxUniform`, `mutFlipBit`, `HallOfFame`, `Statistics`, and many others. You pick the ones you need and register them in the toolbox.

The analogy: **the toolbox is a recipe that defines your GA's ingredients -- what an individual looks like, how to select parents, how to combine them, how to mutate offspring.** The algorithm is the chef who follows the recipe. The creator defines the types of dishes. The tools are the pre-made ingredients you can use.

**Conceptual flow:**

```
Creator                    Toolbox                     Algorithm
  |                          |                            |
  |-- defines Individual --> registers evaluate,    -->  calls toolbox
  |-- defines Fitness        select, mate, mutate        in a loop:
                             with specific operators      evaluate -> select
                                                          -> mate -> mutate
                                                          -> repeat
```

<div class="callout-insight">

DEAP's toolbox pattern is what makes it powerful for research and prototyping. Want to switch from tournament selection to rank-based? Change one line: `toolbox.register("select", tools.selRoulette)`. Want a custom fitness function? Register it: `toolbox.register("evaluate", my_custom_fitness)`. The algorithm loop does not change -- it just calls `toolbox.select`, `toolbox.mate`, and `toolbox.mutate` regardless of what is registered behind those names.

</div>

<div class="flow">
<div class="flow-step mint">Define Types</div>
<div class="flow-arrow">→</div>
<div class="flow-step blue">Build Toolbox</div>
<div class="flow-arrow">→</div>
<div class="flow-step amber">Create Population</div>
<div class="flow-arrow">→</div>
<div class="flow-step lavender">Run Evolution</div>
<div class="flow-arrow">→</div>
<div class="flow-step mint">Extract Best</div>
</div>

## Basic GA Setup

### Defining Types


<span class="filename">example.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
import numpy as np
import random
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Define fitness (minimize)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# Define individual
creator.create("Individual", list, fitness=creator.FitnessMin)
```

</div>


### Creating the Toolbox


<span class="filename">setup_toolbox.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def setup_toolbox(n_features: int, X: np.ndarray, y: np.ndarray):
    """
    Setup DEAP toolbox for feature selection.
    """
    toolbox = base.Toolbox()

    # Attribute generator (0 or 1)
    toolbox.register("attr_bool", random.randint, 0, 1)

    # Individual creator
    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_bool,
        n=n_features
    )

    # Population creator
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual
    )

    # Fitness function
    def evaluate(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) == 0:
            return (float('inf'),)

        X_selected = X[:, selected]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        scores = cross_val_score(
            model, X_selected, y,
            cv=5, scoring='neg_mean_squared_error'
        )
        return (-scores.mean(),)

    toolbox.register("evaluate", evaluate)

    # Genetic operators
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox
```

</div>

## Parameter Selection Guide

The `setup_toolbox` function above uses specific values for `pop_size`, `indpb`, `tournsize`, and `mutation_prob`. These are not arbitrary -- each has a heuristic justification. The table below summarizes the recommended defaults and the reasoning behind them.

| Parameter | Recommended Range | Heuristic | Reasoning |
|---|---|---|---|
| **pop_size** | 2-5x `n_features` | For 50 features, use 100-250 | Need diversity proportional to search space size ($2^n$). Too small and the GA converges prematurely; too large and each generation is expensive. |
| **mutation_rate** (`indpb` for `mutFlipBit`) | `1/n_features` | For 50 features, use 0.02 | On average, one feature flips per individual. This is the minimal perturbation that still explores -- like a small learning rate in gradient descent. |
| **crossover_prob** (`cxpb`) | 0.6-0.9 | Default: 0.8 | Crossover is the primary search operator that combines building blocks from parents. Below 0.6, the GA relies too heavily on mutation alone. |
| **tournament_size** (`tournsize`) | 3-7 | Default: 3 | Balances selection pressure and diversity. Size 3 means the best of 3 random individuals becomes a parent -- moderate pressure. Size 7 is aggressive, suitable for large populations. |

<div class="callout-warning">

These parameters interact. If you increase tournament size (higher selection pressure), the population loses diversity faster, so you may need a larger population or higher mutation rate to compensate. If you lower mutation rate, crossover becomes the sole source of novelty -- make sure crossover probability is high (0.8+). Never tune one parameter in isolation.

</div>

## Running the GA

### Simple Evolution

```python
def run_ga(
    X: np.ndarray,
    y: np.ndarray,
    pop_size: int = 50,
    n_generations: int = 50,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.2,
    verbose: bool = True
):
    """
    Run genetic algorithm for feature selection.
    """
    n_features = X.shape[1]
    toolbox = setup_toolbox(n_features, X, y)

    # Create initial population
    population = toolbox.population(n=pop_size)

    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)
    stats.register("std", np.std)

    # Hall of Fame (best individuals)
    hof = tools.HallOfFame(5)

    # Run evolution
    population, logbook = algorithms.eaSimple(
        population,
        toolbox,
        cxpb=crossover_prob,
        mutpb=mutation_prob,
        ngen=n_generations,
        stats=stats,
        halloffame=hof,
        verbose=verbose
    )

    # Get best solution
    best = hof[0]
    selected_features = [i for i, bit in enumerate(best) if bit == 1]

    return {
        'selected_features': selected_features,
        'fitness': best.fitness.values[0],
        'logbook': logbook,
        'hall_of_fame': hof
    }

# Example usage
np.random.seed(42)
X = np.random.randn(500, 30)  # 30 features
y = X[:, 0] + 2*X[:, 5] + 0.5*X[:, 10] + np.random.randn(500)*0.1

result = run_ga(X, y, pop_size=50, n_generations=30)
print(f"Selected features: {result['selected_features']}")
print(f"Best fitness: {result['fitness']:.4f}")
```

## Custom Evolution Loop


<span class="filename">custom_ga.py</span>
</div>

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>

```python
def custom_ga(
    X: np.ndarray,
    y: np.ndarray,
    pop_size: int = 50,
    n_generations: int = 50,
    crossover_prob: float = 0.8,
    mutation_prob: float = 0.2,
    elitism: int = 2,
    early_stopping: int = 10
):
    """
    Custom GA with elitism and early stopping.
    """
    n_features = X.shape[1]
    toolbox = setup_toolbox(n_features, X, y)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Track best
    best_fitness = float('inf')
    generations_without_improvement = 0
    history = []

    for gen in range(n_generations):
        # Select elite
        elite = tools.selBest(population, elitism)

        # Select parents
        offspring = toolbox.select(population, len(population) - elitism)
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < crossover_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < mutation_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Ensure at least one feature selected
        for ind in offspring:
            if sum(ind) == 0:
                ind[random.randint(0, len(ind)-1)] = 1
                del ind.fitness.values

        # Evaluate new individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace population (elite + offspring)
        population[:] = elite + offspring

        # Track progress
        current_best = min(ind.fitness.values[0] for ind in population)
        avg_fitness = np.mean([ind.fitness.values[0] for ind in population])

        history.append({
            'generation': gen,
            'best': current_best,
            'average': avg_fitness,
            'n_features_best': sum(tools.selBest(population, 1)[0])
        })

        # Early stopping
        if current_best < best_fitness:
            best_fitness = current_best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= early_stopping:
            print(f"Early stopping at generation {gen}")
            break

    # Return results
    best_ind = tools.selBest(population, 1)[0]
    return {
        'selected_features': [i for i, bit in enumerate(best_ind) if bit == 1],
        'fitness': best_ind.fitness.values[0],
        'history': history
    }
```

</div>

## Time Series Feature Selection

<div class="callout-danger">

<strong>Danger:</strong> DEAP's creator.create() modifies global state. Calling it twice with the same name (e.g., "FitnessMin") raises an error. In notebooks, restart the kernel or add a guard: if not hasattr(creator, "FitnessMin").

</div>

<div class="callout-warning">

⚠️ **Warning:** Never use standard `cross_val_score` with default `KFold` for time series data. Always pass a `TimeSeriesSplit` splitter to prevent look-ahead bias, which produces unrealistically optimistic fitness estimates.

</div>

```python
from sklearn.model_selection import TimeSeriesSplit

def setup_timeseries_toolbox(n_features: int, X: np.ndarray, y: np.ndarray):
    """
    Setup toolbox with time series cross-validation.
    """
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register(
        "individual", tools.initRepeat,
        creator.Individual, toolbox.attr_bool, n=n_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_timeseries(individual):
        selected = [i for i, bit in enumerate(individual) if bit == 1]
        if len(selected) == 0:
            return (float('inf'),)

        X_selected = X[:, selected]

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        errors = []

        for train_idx, test_idx in tscv.split(X_selected):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            errors.append(np.mean((y_test - pred) ** 2))

        # Penalty for feature count
        feature_penalty = 0.01 * len(selected)

        return (np.mean(errors) + feature_penalty,)

    toolbox.register("evaluate", evaluate_timeseries)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox
```

## Parallel Evaluation

```python
from multiprocessing import Pool

def run_parallel_ga(X, y, pop_size=50, n_generations=50, n_processes=4):
    """
    Run GA with parallel fitness evaluation.
    """
    n_features = X.shape[1]
    toolbox = setup_toolbox(n_features, X, y)

    # Register parallel map
    pool = Pool(processes=n_processes)
    toolbox.register("map", pool.map)

    try:
        population = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(5)

        population, logbook = algorithms.eaSimple(
            population, toolbox,
            cxpb=0.8, mutpb=0.2,
            ngen=n_generations,
            halloffame=hof,
            verbose=True
        )

        best = hof[0]
        return {
            'selected_features': [i for i, b in enumerate(best) if b == 1],
            'fitness': best.fitness.values[0]
        }
    finally:
        pool.close()
        pool.join()
```

## Visualization

```python
import matplotlib.pyplot as plt

def plot_evolution(history: list):
    """
    Plot GA evolution progress.
    """
    generations = [h['generation'] for h in history]
    best_fitness = [h['best'] for h in history]
    avg_fitness = [h['average'] for h in history]
    n_features = [h['n_features_best'] for h in history]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Fitness over generations
    ax1.plot(generations, best_fitness, 'b-', label='Best', linewidth=2)
    ax1.plot(generations, avg_fitness, 'r--', label='Average', alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness (MSE)')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Feature count over generations
    ax2.plot(generations, n_features, 'g-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Number of Features')
    ax2.set_title('Feature Count in Best Solution')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Usage
result = custom_ga(X, y)
plot_evolution(result['history'])
```

## Key Takeaways

<div class="callout-key">

🔑 **Key Points**

1. **DEAP provides flexibility** - customize every component

2. **Toolbox pattern** organizes all GA components -- creator defines types, toolbox registers operators, algorithms orchestrate, tools provide implementations

3. **Hall of Fame** preserves best solutions across all generations

4. **Early stopping** prevents wasted computation when the GA has converged

5. **Parallel evaluation** speeds up fitness computation linearly with cores

6. **Parameter choices matter** -- pop_size, mutation_rate, crossover_prob, and tournament_size interact as a system, not independently

</div>

## Practice Problems

1. **DEAP Architecture**
   Explain in your own words what happens when you call `algorithms.eaSimple(population, toolbox, ...)`. Which toolbox functions does it call, and in what order? What is the role of `stats` and `halloffame`?

2. **Parameter Reasoning**
   You have a dataset with 200 features. Using the heuristics from the Parameter Selection Guide, specify: population size, mutation rate, crossover probability, and tournament size. Justify each choice in one sentence.

3. **Creator Global State**
   Why does DEAP's `creator.create()` cause problems in Jupyter notebooks? What happens if you call `creator.create("FitnessMin", ...)` twice? How do you work around this?

4. **Toolbox Swap**
   You have a working GA using `tools.selTournament` and want to switch to rank-based selection. Which line(s) do you change? Do you need to modify the algorithm loop?

5. **Fitness Function Design**
   The `evaluate` function in `setup_toolbox` returns `(float('inf'),)` when no features are selected. Why is this necessary? What would happen if it returned `(0.0,)` instead?

---

**Next:** [Companion Slides](./01_deap_implementation_slides.md) | [Notebook](../notebooks/01_deap_ga.ipynb)

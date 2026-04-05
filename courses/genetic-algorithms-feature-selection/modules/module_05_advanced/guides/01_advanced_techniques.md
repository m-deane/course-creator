# Advanced GA Techniques

> **Reading time:** ~7 min | **Module:** 5 — Advanced Topics | **Prerequisites:** Modules 1-4

![Fitness Landscape](./fitness_landscape.svg)

<div class="callout-danger">
<strong>Danger:</strong> Running NSGA-II with insufficient population size (e.g., 20 individuals for a 100-feature problem) produces a sparse, unreliable Pareto front. Use at least 2-3x the number of features as population size for meaningful multi-objective results.
</div>

<div class="callout-warning">
<strong>Warning:</strong> DEAP's creator.create() uses global state. If you define FitnessMulti for NSGA-II after previously defining FitnessMin for single-objective, both remain active. Clear previous definitions or use separate modules.
</div>

## Multi-Objective Optimization

### NSGA-II for Feature Selection

NSGA-II (Non-dominated Sorting Genetic Algorithm II) solves feature selection as a two-objective problem: minimize prediction error while simultaneously minimizing the number of features. The Pareto front it produces gives you a set of non-dominated trade-off solutions, letting you choose how much accuracy to trade for simplicity.

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">setup_nsga2_toolbox.py</span>
</div>

```python
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Multi-objective: minimize error AND feature count
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)

def setup_nsga2_toolbox(n_features, X, y):
    """Setup DEAP toolbox for NSGA-II."""
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", np.random.randint, 0, 2)
    toolbox.register(
        "individual", tools.initRepeat,
        creator.Individual, toolbox.attr_bool, n_features
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual):
        selected = [i for i, b in enumerate(individual) if b == 1]
        if len(selected) == 0:
            return (float('inf'), float('inf'))

        X_sel = X[:, selected]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_sel, y, cv=5, scoring='neg_mean_squared_error')

        return (-scores.mean(), len(selected))

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selNSGA2)

    return toolbox

def run_nsga2(X, y, pop_size=100, n_generations=50):
    """Run NSGA-II for multi-objective feature selection."""
    toolbox = setup_nsga2_toolbox(X.shape[1], X, y)

    population = toolbox.population(n=pop_size)

    # Evaluate initial population
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    for gen in range(n_generations):
        offspring = tools.selTournamentDCD(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < 0.9:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < 0.1:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Ensure valid individuals
        for ind in offspring:
            if sum(ind) == 0:
                ind[np.random.randint(len(ind))] = 1
                del ind.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        population = toolbox.select(population + offspring, pop_size)

    # Extract Pareto front
    pareto_front = tools.sortNondominated(population, len(population), first_front_only=True)[0]

    return {
        'pareto_front': pareto_front,
        'solutions': [
            {
                'features': [i for i, b in enumerate(ind) if b == 1],
                'error': ind.fitness.values[0],
                'n_features': ind.fitness.values[1]
            }
            for ind in pareto_front
        ]
    }
```
</div>


<div class="callout-insight">

💡 **Key Insight:** The Pareto front from NSGA-II gives you a menu of accuracy-vs-parsimony tradeoffs rather than a single answer. In production, you pick the "knee point" -- the solution where adding one more feature yields diminishing accuracy gains.

</div>

## Hybrid Methods

### GA + Local Search

```python
def local_search(individual, X, y, n_iterations=10):
    """
    Hill climbing to refine GA solution.
    """
    best = individual.copy()
    best_fitness = evaluate_fitness(best, X, y)

    for _ in range(n_iterations):
        # Try flipping each bit
        for i in range(len(best)):
            neighbor = best.copy()
            neighbor[i] = 1 - neighbor[i]

            # Ensure at least one feature
            if sum(neighbor) == 0:
                continue

            fitness = evaluate_fitness(neighbor, X, y)

            if fitness < best_fitness:
                best = neighbor
                best_fitness = fitness
                break  # First improvement

    return best, best_fitness

def memetic_ga(X, y, pop_size=50, n_generations=30, local_search_freq=5):
    """
    Memetic algorithm: GA + periodic local search.
    """
    toolbox = setup_toolbox(X.shape[1], X, y)
    population = toolbox.population(n=pop_size)

    # Evaluate
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(n_generations):
        # Standard GA operations
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(c1, c2)
            del c1.fitness.values, c2.fitness.values

        for m in offspring:
            toolbox.mutate(m)
            del m.fitness.values

        # Evaluate offspring
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # Local search on best individuals periodically
        if gen % local_search_freq == 0:
            best_inds = tools.selBest(offspring, 5)
            for ind in best_inds:
                improved, fitness = local_search(list(ind), X, y)
                ind[:] = improved
                ind.fitness.values = (fitness,)

        population[:] = offspring

    return tools.selBest(population, 1)[0]
```

### GA + Filter Methods

```python
from sklearn.feature_selection import mutual_info_regression

def filter_guided_initialization(X, y, pop_size, top_k=None):
    """
    Initialize population using filter method rankings.
    """
    n_features = X.shape[1]
    top_k = top_k or n_features // 2

    # Compute mutual information scores
    mi_scores = mutual_info_regression(X, y)
    top_features = np.argsort(mi_scores)[::-1][:top_k]

    population = []

    # Half population guided by MI scores
    for _ in range(pop_size // 2):
        chromosome = np.zeros(n_features, dtype=int)
        # Select random subset of top features
        n_select = np.random.randint(3, min(top_k, 15))
        selected = np.random.choice(top_features, n_select, replace=False)
        chromosome[selected] = 1
        population.append(creator.Individual(chromosome.tolist()))

    # Half population random
    for _ in range(pop_size - len(population)):
        chromosome = (np.random.random(n_features) < 0.3).astype(int)
        if chromosome.sum() == 0:
            chromosome[np.random.choice(top_features)] = 1
        population.append(creator.Individual(chromosome.tolist()))

    return population
```

## Ensemble Feature Selection

### Multiple GA Runs

```python
def ensemble_ga(X, y, n_runs=10, **ga_params):
    """
    Run GA multiple times and aggregate results.
    """
    n_features = X.shape[1]
    feature_votes = np.zeros(n_features)

    all_results = []
    for run in range(n_runs):
        np.random.seed(run)
        result = run_ga(X, y, **ga_params)
        all_results.append(result)

        for feat in result['selected_features']:
            feature_votes[feat] += 1

    # Features selected in >50% of runs
    consensus_features = np.where(feature_votes > n_runs / 2)[0].tolist()

    return {
        'consensus_features': consensus_features,
        'feature_votes': feature_votes,
        'all_results': all_results
    }
```

### Stacked Selection

```python
def stacked_feature_selection(X, y, models, ga_params):
    """
    Select features that work well across multiple models.
    """
    n_features = X.shape[1]
    model_selections = {}

    for model_name, model_fn in models.items():
        # Create fitness function for this model
        def fitness_for_model(individual):
            selected = [i for i, b in enumerate(individual) if b == 1]
            if not selected:
                return (float('inf'),)
            X_sel = X[:, selected]
            scores = cross_val_score(model_fn(), X_sel, y, cv=5, scoring='neg_mean_squared_error')
            return (-scores.mean(),)

        # Run GA
        result = run_ga_with_fitness(fitness_for_model, n_features, **ga_params)
        model_selections[model_name] = set(result['selected_features'])

    # Find features selected across all models
    common_features = set.intersection(*model_selections.values())

    # Find features selected by at least half of models
    feature_counts = {}
    for features in model_selections.values():
        for f in features:
            feature_counts[f] = feature_counts.get(f, 0) + 1

    robust_features = [f for f, count in feature_counts.items() if count >= len(models) / 2]

    return {
        'common_features': list(common_features),
        'robust_features': robust_features,
        'model_selections': model_selections
    }
```

## Constraint Handling

### Feature Group Constraints

```python
def constrained_ga(X, y, feature_groups, min_per_group=1, max_per_group=3):
    """
    GA with constraints on feature group selection.

    Args:
        feature_groups: dict mapping group name to feature indices
    """
    n_features = X.shape[1]

    def repair_individual(individual):
        """Repair individual to satisfy constraints."""
        ind = list(individual)

        for group_name, group_indices in feature_groups.items():
            selected_in_group = [i for i in group_indices if ind[i] == 1]

            # Too few selected
            while len(selected_in_group) < min_per_group:
                available = [i for i in group_indices if ind[i] == 0]
                if not available:
                    break
                add_idx = np.random.choice(available)
                ind[add_idx] = 1
                selected_in_group.append(add_idx)

            # Too many selected
            while len(selected_in_group) > max_per_group:
                remove_idx = np.random.choice(selected_in_group)
                ind[remove_idx] = 0
                selected_in_group.remove(remove_idx)

        return ind

    def constrained_fitness(individual):
        repaired = repair_individual(individual)
        selected = [i for i, b in enumerate(repaired) if b == 1]

        if not selected:
            return (float('inf'),)

        X_sel = X[:, selected]
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        scores = cross_val_score(model, X_sel, y, cv=5, scoring='neg_mean_squared_error')

        return (-scores.mean(),)

    # Run GA with constrained fitness
    toolbox = setup_toolbox(n_features, X, y)
    toolbox.register("evaluate", constrained_fitness)

    population = toolbox.population(n=50)
    # Repair initial population
    population = [creator.Individual(repair_individual(ind)) for ind in population]

    # Run evolution
    result = algorithms.eaSimple(
        population, toolbox,
        cxpb=0.8, mutpb=0.2, ngen=30,
        verbose=False
    )

    best = tools.selBest(population, 1)[0]
    return {
        'selected_features': [i for i, b in enumerate(best) if b == 1],
        'fitness': best.fitness.values[0]
    }
```

## Performance Optimization

### Surrogate Model Fitness

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class SurrogateFitness:
    """
    Use surrogate model to approximate expensive fitness evaluations.
    """

    def __init__(self, X, y, n_initial=50):
        self.X = X
        self.y = y
        self.n_features = X.shape[1]

        # Cache of exact evaluations
        self.exact_cache = {}

        # Surrogate model
        self.surrogate = GaussianProcessRegressor(kernel=RBF())
        self.surrogate_X = []
        self.surrogate_y = []

        # Initial exact evaluations
        self._initialize_surrogate(n_initial)

    def _exact_evaluation(self, individual):
        """Compute exact fitness."""
        key = tuple(individual)
        if key not in self.exact_cache:
            selected = [i for i, b in enumerate(individual) if b == 1]
            if not selected:
                self.exact_cache[key] = float('inf')
            else:
                X_sel = self.X[:, selected]
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                scores = cross_val_score(model, X_sel, self.y, cv=5, scoring='neg_mean_squared_error')
                self.exact_cache[key] = -scores.mean()
        return self.exact_cache[key]

    def _initialize_surrogate(self, n_samples):
        """Initialize surrogate with random evaluations."""
        for _ in range(n_samples):
            individual = (np.random.random(self.n_features) < 0.3).astype(int).tolist()
            fitness = self._exact_evaluation(individual)

            self.surrogate_X.append(individual)
            self.surrogate_y.append(fitness)

        self.surrogate.fit(self.surrogate_X, self.surrogate_y)

    def evaluate(self, individual, use_surrogate=True):
        """Evaluate with surrogate or exact computation."""
        if use_surrogate and len(self.surrogate_X) > 20:
            # Use surrogate
            pred, std = self.surrogate.predict([individual], return_std=True)

            # If uncertain, compute exactly
            if std[0] > 0.5:  # Uncertainty threshold
                fitness = self._exact_evaluation(individual)
                self._update_surrogate(individual, fitness)
            else:
                fitness = pred[0]
        else:
            fitness = self._exact_evaluation(individual)
            self._update_surrogate(individual, fitness)

        return (fitness,)

    def _update_surrogate(self, individual, fitness):
        """Update surrogate model with new evaluation."""
        self.surrogate_X.append(list(individual))
        self.surrogate_y.append(fitness)
        self.surrogate.fit(self.surrogate_X, self.surrogate_y)
```

## Key Takeaways

<div class="callout-key">
🔑 **Key Points**

1. **NSGA-II** handles multi-objective optimization naturally

2. **Hybrid methods** combine global and local search effectively

3. **Ensemble approaches** improve selection stability

4. **Constraints** can be handled through repair operators

5. **Surrogate models** reduce computational cost for expensive fitness functions
</div>
---

**Next:** [Companion Slides](./01_advanced_techniques_slides.md) | [Notebook](../notebooks/01_nsga2_features.ipynb)

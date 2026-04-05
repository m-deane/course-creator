# Evolutionary Operators: Selection, Crossover, and Mutation

> **Reading time:** ~7 min | **Module:** 0 — Foundations | **Prerequisites:** 01 Optimization Basics

## Introduction

<div class="callout-key">
<strong>Key Takeaway:</strong> The three operators -- selection, crossover, mutation -- form a balanced system. Selection drives convergence, crossover combines good building blocks, and mutation prevents stagnation. Removing or misconfiguring any one breaks the system.
</div>

Genetic algorithms evolve solutions through three fundamental operators:
1. **Selection**: Choose parents for reproduction
2. **Crossover**: Combine parent traits to create offspring
3. **Mutation**: Introduce random variation

Understanding these operators is essential for effective GA design.


![Feature Selection Pipeline](./feature_selection_pipeline.svg)

<div class="flow">
<div class="flow-step mint">Selection</div>
<div class="flow-arrow">→</div>
<div class="flow-step blue">Crossover</div>
<div class="flow-arrow">→</div>
<div class="flow-step amber">Mutation</div>
<div class="flow-arrow">→</div>
<div class="flow-step lavender">Offspring</div>
</div>

## Selection Operators

### Tournament Selection

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">tournament_selection.py</span>
</div>

```python
import numpy as np
import random

def tournament_selection(population, fitness_scores, tournament_size=3):
    """
    Tournament selection: Select best individual from random tournament.

    Parameters:
    -----------
    population : list
        List of individuals (chromosomes)
    fitness_scores : list
        Fitness score for each individual
    tournament_size : int
        Number of individuals in each tournament

    Returns:
    --------
    Individual selected as parent
    """
    # Random sample for tournament
    tournament_indices = random.sample(range(len(population)), tournament_size)

    # Find best in tournament
    best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])

    return population[best_idx]

# Example
np.random.seed(42)
population = [[random.randint(0, 1) for _ in range(10)] for _ in range(20)]
fitness_scores = [sum(ind) for ind in population]  # Count ones

selected = tournament_selection(population, fitness_scores, tournament_size=3)
print(f"Selected individual: {selected}")
print(f"Fitness: {sum(selected)}")
```
</div>


### Roulette Wheel Selection

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">roulette_wheel.py</span>
</div>

```python
def roulette_wheel_selection(population, fitness_scores):
    """
    Roulette wheel (fitness proportionate) selection.

    Probability of selection proportional to fitness.
    """
    # Handle negative fitness
    min_fitness = min(fitness_scores)
    adjusted_fitness = [f - min_fitness + 1 for f in fitness_scores]

    total_fitness = sum(adjusted_fitness)
    probabilities = [f / total_fitness for f in adjusted_fitness]

    # Spin the wheel
    cumulative = 0
    spin = random.random()

    for i, prob in enumerate(probabilities):
        cumulative += prob
        if spin <= cumulative:
            return population[i]

    return population[-1]  # Fallback

# Vectorized version using numpy
def roulette_wheel_vectorized(population, fitness_scores, n_select=1):
    """Efficient roulette wheel selection."""
    fitness_arr = np.array(fitness_scores)
    fitness_arr = fitness_arr - fitness_arr.min() + 1e-6  # Make positive

    probabilities = fitness_arr / fitness_arr.sum()

    selected_indices = np.random.choice(
        len(population), size=n_select, p=probabilities
    )

    return [population[i] for i in selected_indices]

# Example
selected = roulette_wheel_selection(population, fitness_scores)
print(f"Roulette selected: {selected}")
```

</div>

### Rank Selection

```python
def rank_selection(population, fitness_scores, selection_pressure=1.5):
    """
    Rank-based selection.

    Selection probability based on rank, not raw fitness.
    More robust to fitness scaling issues.
    """
    n = len(population)

    # Rank individuals (1 = worst, n = best)
    ranked_indices = np.argsort(fitness_scores)
    ranks = np.empty_like(ranked_indices)
    ranks[ranked_indices] = np.arange(1, n + 1)

    # Linear ranking probabilities
    # P(rank i) = (2 - s + 2(s-1)(i-1)/(n-1)) / n
    # where s is selection pressure (1 < s <= 2)
    s = selection_pressure
    probabilities = (2 - s + 2 * (s - 1) * (ranks - 1) / (n - 1)) / n

    # Normalize
    probabilities = probabilities / probabilities.sum()

    selected_idx = np.random.choice(n, p=probabilities)
    return population[selected_idx]

# Example
selected = rank_selection(population, fitness_scores)
print(f"Rank selected: {selected}")
```

![Crossover Types](./crossover_types.svg)

## Crossover Operators

<div class="callout-insight">

💡 **Key Insight:** For feature selection, uniform crossover typically outperforms single-point and two-point crossover because features are unordered -- there is no positional structure to preserve.

</div>

### Single-Point Crossover

```python
def single_point_crossover(parent1, parent2):
    """
    Single-point crossover: Split at one random point.
    """
    n = len(parent1)
    point = random.randint(1, n - 1)

    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]

    return child1, child2

# Example
p1 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
p2 = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

c1, c2 = single_point_crossover(p1, p2)
print(f"Parent 1: {p1}")
print(f"Parent 2: {p2}")
print(f"Child 1:  {c1}")
print(f"Child 2:  {c2}")
```

### Two-Point Crossover

```python
def two_point_crossover(parent1, parent2):
    """
    Two-point crossover: Exchange segment between two points.
    """
    n = len(parent1)
    point1, point2 = sorted(random.sample(range(1, n), 2))

    child1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
    child2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

    return child1, child2

c1, c2 = two_point_crossover(p1, p2)
print(f"\nTwo-point crossover:")
print(f"Child 1: {c1}")
print(f"Child 2: {c2}")
```

### Uniform Crossover

```python
def uniform_crossover(parent1, parent2, swap_prob=0.5):
    """
    Uniform crossover: Each gene swapped independently.
    """
    child1 = []
    child2 = []

    for g1, g2 in zip(parent1, parent2):
        if random.random() < swap_prob:
            child1.append(g2)
            child2.append(g1)
        else:
            child1.append(g1)
            child2.append(g2)

    return child1, child2

c1, c2 = uniform_crossover(p1, p2)
print(f"\nUniform crossover:")
print(f"Child 1: {c1}")
print(f"Child 2: {c2}")
```

![Mutation Types](./mutation_types.svg)

## Mutation Operators

<div class="callout-warning">

⚠️ **Warning:** Setting mutation rate too high (above 0.1 per gene) effectively degrades the GA to random search. The rule of thumb is `1/n` where `n` is chromosome length, which flips approximately one gene per individual.

</div>

### Bit Flip Mutation

```python
def bit_flip_mutation(individual, mutation_rate=0.1):
    """
    Bit flip mutation for binary chromosomes.
    """
    mutant = individual.copy()

    for i in range(len(mutant)):
        if random.random() < mutation_rate:
            mutant[i] = 1 - mutant[i]  # Flip bit

    return mutant

# Example
original = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
mutated = bit_flip_mutation(original, mutation_rate=0.2)
print(f"Original: {original}")
print(f"Mutated:  {mutated}")
print(f"Bits flipped: {sum(a != b for a, b in zip(original, mutated))}")
```

### Gaussian Mutation (Real-Valued)

```python
def gaussian_mutation(individual, mutation_rate=0.1, sigma=0.1):
    """
    Gaussian mutation for real-valued chromosomes.
    """
    mutant = individual.copy()

    for i in range(len(mutant)):
        if random.random() < mutation_rate:
            mutant[i] += np.random.normal(0, sigma)

    return mutant

# Example with real values
real_individual = [0.5, 0.3, 0.8, 0.2, 0.6]
mutated_real = gaussian_mutation(real_individual, mutation_rate=0.3, sigma=0.1)
print(f"\nReal-valued mutation:")
print(f"Original: {[f'{x:.3f}' for x in real_individual]}")
print(f"Mutated:  {[f'{x:.3f}' for x in mutated_real]}")
```

### Swap Mutation (Permutations)

```python
def swap_mutation(individual):
    """
    Swap mutation for permutation chromosomes.
    """
    mutant = individual.copy()
    i, j = random.sample(range(len(mutant)), 2)
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant

# Example with permutation
perm = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mutated_perm = swap_mutation(perm)
print(f"\nPermutation mutation:")
print(f"Original: {perm}")
print(f"Mutated:  {mutated_perm}")
```

## Feature Selection Operators

For feature selection, chromosomes are binary vectors where 1 = feature selected:

<div class="code-window">
<div class="code-header">
<div class="dots"><span class="dot-red"></span><span class="dot-yellow"></span><span class="dot-green"></span></div>
<span class="filename">feature_selection_operators.py</span>
</div>

```python
class FeatureSelectionOperators:
    """Operators specialized for feature selection."""

    def __init__(self, n_features, min_features=1, max_features=None):
        self.n_features = n_features
        self.min_features = min_features
        self.max_features = max_features or n_features

    def crossover(self, parent1, parent2, method='uniform'):
        """Feature selection crossover with constraints."""
        if method == 'uniform':
            child1, child2 = uniform_crossover(parent1, parent2)
        elif method == 'single':
            child1, child2 = single_point_crossover(parent1, parent2)
        else:
            child1, child2 = two_point_crossover(parent1, parent2)

        # Enforce constraints
        child1 = self._enforce_constraints(child1)
        child2 = self._enforce_constraints(child2)

        return child1, child2

    def mutate(self, individual, mutation_rate=0.1):
        """Feature selection mutation with constraints."""
        mutant = bit_flip_mutation(individual, mutation_rate)
        return self._enforce_constraints(mutant)

    def _enforce_constraints(self, individual):
        """Ensure min/max features selected."""
        n_selected = sum(individual)

        if n_selected < self.min_features:
            # Add random features
            zeros = [i for i, x in enumerate(individual) if x == 0]
            to_add = random.sample(zeros, self.min_features - n_selected)
            for i in to_add:
                individual[i] = 1

        elif n_selected > self.max_features:
            # Remove random features
            ones = [i for i, x in enumerate(individual) if x == 1]
            to_remove = random.sample(ones, n_selected - self.max_features)
            for i in to_remove:
                individual[i] = 0

        return individual

# Example
fs_ops = FeatureSelectionOperators(n_features=20, min_features=5, max_features=15)

# Create parents
parent1 = [random.randint(0, 1) for _ in range(20)]
parent2 = [random.randint(0, 1) for _ in range(20)]

child1, child2 = fs_ops.crossover(parent1, parent2)
mutated = fs_ops.mutate(child1)

print(f"Parent 1 features: {sum(parent1)}")
print(f"Parent 2 features: {sum(parent2)}")
print(f"Child 1 features: {sum(child1)}")
print(f"Mutated features: {sum(mutated)}")
```

</div>

## Adaptive Operators

### Self-Adaptive Mutation Rate

```python
class AdaptiveMutation:
    """Mutation rate that adapts based on population diversity."""

    def __init__(self, base_rate=0.1, min_rate=0.01, max_rate=0.5):
        self.base_rate = base_rate
        self.min_rate = min_rate
        self.max_rate = max_rate

    def get_rate(self, population):
        """Calculate adaptive mutation rate."""
        # Measure diversity as average Hamming distance
        n = len(population)
        total_distance = 0
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                distance = sum(a != b for a, b in zip(population[i], population[j]))
                total_distance += distance
                count += 1

        avg_diversity = total_distance / count if count > 0 else 0
        max_possible = len(population[0])

        # Low diversity -> high mutation, high diversity -> low mutation
        diversity_ratio = avg_diversity / max_possible
        rate = self.base_rate * (1 - diversity_ratio) + self.max_rate * (1 - diversity_ratio)

        return np.clip(rate, self.min_rate, self.max_rate)

adaptive = AdaptiveMutation()
rate = adaptive.get_rate(population)
print(f"Adaptive mutation rate: {rate:.4f}")
```

## Operator Comparison

```python
def compare_operators(n_trials=100):
    """Compare effectiveness of different operators."""
    results = {
        'single_point': [],
        'two_point': [],
        'uniform': []
    }

    for _ in range(n_trials):
        # Random parents
        p1 = [random.randint(0, 1) for _ in range(100)]
        p2 = [random.randint(0, 1) for _ in range(100)]

        # Measure offspring diversity
        c1_sp, c2_sp = single_point_crossover(p1, p2)
        c1_tp, c2_tp = two_point_crossover(p1, p2)
        c1_u, c2_u = uniform_crossover(p1, p2)

        # Diversity = distance from parents
        def parent_distance(child, p1, p2):
            d1 = sum(a != b for a, b in zip(child, p1))
            d2 = sum(a != b for a, b in zip(child, p2))
            return (d1 + d2) / 2

        results['single_point'].append(parent_distance(c1_sp, p1, p2))
        results['two_point'].append(parent_distance(c1_tp, p1, p2))
        results['uniform'].append(parent_distance(c1_u, p1, p2))

    print("Crossover Operator Comparison (offspring diversity):")
    for op, distances in results.items():
        print(f"  {op}: mean={np.mean(distances):.1f}, std={np.std(distances):.1f}")

compare_operators()
```

## Key Takeaways

<div class="callout-key">
🔑 **Key Points**

1. **Selection pressure** controls exploitation vs exploration balance

2. **Tournament selection** is robust and easy to tune via tournament size

3. **Crossover operators** differ in how they mix parent genetic material

4. **Mutation rate** should be low enough to preserve good solutions but high enough to maintain diversity

5. **Constraint handling** is essential for feature selection problems

6. **Adaptive operators** can improve performance by responding to population state
</div>
---

**Next:** [Companion Slides](./03_evolutionary_operators_slides.md) | [Notebook](../notebooks/01_selection_comparison.ipynb)

# Selection Operators for Genetic Algorithms

## In Brief

Selection operators determine which individuals from the current population reproduce to create the next generation. Tournament, roulette wheel, and rank selection represent different strategies for balancing exploitation of good solutions with exploration of the search space.

> 💡 **Key Insight:** Selection pressure controls the trade-off between convergence speed and population diversity. High pressure (large tournaments, steep rank weighting) accelerates convergence but risks premature convergence to local optima. Low pressure maintains diversity but may converge too slowly.

## Formal Definition

### Selection Probability

For individual $i$ with fitness $f_i$, the selection probability depends on the method:

**Roulette Wheel (Fitness Proportionate)**:
$$P(i) = \frac{w(f_i)}{\sum_{j=1}^{N} w(f_j)}$$

where $w(f)$ is a fitness transformation (e.g., $w(f) = f_{max} - f$ for minimization).

**Rank-Based**:
$$P(i) = \frac{r_i}{\sum_{j=1}^{N} r_j}$$

where $r_i$ is the rank of individual $i$ (best = highest rank).

**Tournament**:
$$P_{\text{win}}(i) = \begin{cases} 1 & \text{if } f_i = \min_{j \in T} f_j \\ 0 & \text{otherwise} \end{cases}$$

where $T$ is the tournament of size $k$.

### Selection Pressure

Quantified by the expected number of offspring for the best individual:
$$s = \frac{\text{E}[\text{offspring of best}]}{\text{E}[\text{offspring per individual}]} = \frac{N \cdot P(\text{best})}{1}$$

Higher $s$ means stronger selection pressure.

## Intuitive Explanation

Think of selection as choosing participants for a breeding program:

**Tournament Selection** is like a sports playoff. Randomly pick $k$ individuals, the best one advances. Larger tournaments favor top performers (high pressure), smaller tournaments give more chances to weaker individuals (low pressure).

**Roulette Wheel** is like a lottery where better individuals get more tickets. An individual with 2x the fitness gets 2x the chance of selection. Problem: If one individual is much better, it dominates (premature convergence).

**Rank Selection** is like giving medals: 1st place gets $N$ points, 2nd gets $N-1$, etc., regardless of the actual performance gap. This prevents super-fit individuals from dominating while still favoring better solutions.

**When to use each:**
- **Tournament**: Default choice, robust, tunable pressure, works for minimization/maximization
- **Roulette**: When fitness scale is meaningful and well-distributed
- **Rank**: When fitnesses have outliers or very different scales

## Code Implementation

### Tournament Selection

```python
import numpy as np
from typing import List, Protocol, Optional
from dataclasses import dataclass

class Individual(Protocol):
    """Protocol for any individual with fitness."""
    fitness: float
    def copy(self) -> 'Individual': ...


def tournament_selection(
    population: List[Individual],
    tournament_size: int = 3,
    minimize: bool = True
) -> Individual:
    """
    Select one individual via tournament selection.

    Parameters
    ----------
    population : List[Individual]
        Current population
    tournament_size : int
        Number of individuals in tournament (k)
        Larger k = higher selection pressure
    minimize : bool
        True if lower fitness is better, False otherwise

    Returns
    -------
    Individual
        Selected individual (not a copy)

    Notes
    -----
    Selection pressure s ≈ tournament_size for large populations
    Typical values: k ∈ {2, 3, 5, 7}
    """
    # Randomly sample tournament participants
    if tournament_size > len(population):
        tournament_size = len(population)

    tournament = np.random.choice(population, size=tournament_size, replace=False)

    # Select best (lowest fitness for minimization)
    if minimize:
        winner = min(tournament, key=lambda ind: ind.fitness)
    else:
        winner = max(tournament, key=lambda ind: ind.fitness)

    return winner


def select_parents_tournament(
    population: List[Individual],
    n_parents: int,
    tournament_size: int = 3,
    minimize: bool = True
) -> List[Individual]:
    """
    Select multiple parents using tournament selection.

    Parents can be selected multiple times (with replacement).
    """
    return [
        tournament_selection(population, tournament_size, minimize)
        for _ in range(n_parents)
    ]


# Adaptive tournament size
def adaptive_tournament_selection(
    population: List[Individual],
    generation: int,
    max_generations: int,
    min_size: int = 2,
    max_size: int = 7,
    minimize: bool = True
) -> Individual:
    """
    Tournament selection with increasing size over generations.

    Starts with low pressure (small tournaments) for exploration,
    increases pressure (larger tournaments) for exploitation.
    """
    progress = generation / max_generations
    size = int(min_size + progress * (max_size - min_size))
    return tournament_selection(population, tournament_size=size, minimize=minimize)
```

### Roulette Wheel Selection

```python
def roulette_wheel_selection(
    population: List[Individual],
    minimize: bool = True
) -> Individual:
    """
    Fitness-proportionate selection (roulette wheel).

    Parameters
    ----------
    population : List[Individual]
        Current population
    minimize : bool
        True if lower fitness is better

    Returns
    -------
    Individual
        Selected individual

    Notes
    -----
    - Requires non-negative fitnesses after transformation
    - Sensitive to fitness scaling
    - Can lead to premature convergence if one individual dominates
    """
    fitnesses = np.array([ind.fitness for ind in population])

    # Transform fitnesses for minimization
    if minimize:
        # Invert: lower fitness → higher selection probability
        # Add small constant to avoid division by zero
        max_fitness = fitnesses.max()
        transformed = max_fitness - fitnesses + 1e-10
    else:
        # Ensure positive
        min_fitness = fitnesses.min()
        if min_fitness < 0:
            transformed = fitnesses - min_fitness + 1e-10
        else:
            transformed = fitnesses + 1e-10

    # Convert to probabilities
    probabilities = transformed / transformed.sum()

    # Select
    idx = np.random.choice(len(population), p=probabilities)
    return population[idx]


def stochastic_universal_sampling(
    population: List[Individual],
    n_select: int,
    minimize: bool = True
) -> List[Individual]:
    """
    Stochastic Universal Sampling (SUS) - better than repeated roulette.

    Single spin with evenly-spaced pointers ensures low variance in
    selection counts (close to expected values).

    Parameters
    ----------
    population : List[Individual]
        Current population
    n_select : int
        Number of individuals to select
    minimize : bool
        True if lower fitness is better

    Returns
    -------
    List[Individual]
        Selected individuals
    """
    fitnesses = np.array([ind.fitness for ind in population])

    # Transform fitnesses
    if minimize:
        max_fitness = fitnesses.max()
        transformed = max_fitness - fitnesses + 1e-10
    else:
        min_fitness = fitnesses.min()
        if min_fitness < 0:
            transformed = fitnesses - min_fitness + 1e-10
        else:
            transformed = fitnesses + 1e-10

    # Cumulative fitness
    cumulative = np.cumsum(transformed)
    total_fitness = cumulative[-1]

    # Generate evenly-spaced pointers
    pointer_distance = total_fitness / n_select
    start = np.random.uniform(0, pointer_distance)
    pointers = [start + i * pointer_distance for i in range(n_select)]

    # Select individuals
    selected = []
    for pointer in pointers:
        idx = np.searchsorted(cumulative, pointer)
        idx = min(idx, len(population) - 1)  # Handle edge case
        selected.append(population[idx])

    return selected
```

### Rank Selection

```python
def rank_selection(
    population: List[Individual],
    minimize: bool = True,
    selection_pressure: float = 2.0
) -> Individual:
    """
    Rank-based selection.

    Parameters
    ----------
    population : List[Individual]
        Current population
    minimize : bool
        True if lower fitness is better
    selection_pressure : float
        Controls selection pressure via linear ranking
        Range: [1.0, 2.0], where 2.0 = max pressure
        Default 2.0 gives probabilities: best = 2/N, worst = 0

    Returns
    -------
    Individual
        Selected individual

    Notes
    -----
    Linear ranking: P(rank r) = (1/N) * (selection_pressure - 2*(selection_pressure-1)*(r-1)/(N-1))
    where r=1 is best, r=N is worst
    """
    # Sort population by fitness
    if minimize:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    else:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    N = len(population)

    # Linear ranking probabilities
    # Best (rank 1) gets highest probability
    # Worst (rank N) gets lowest probability
    probabilities = np.zeros(N)
    for i in range(N):
        rank = i + 1  # 1-indexed
        probabilities[i] = (1/N) * (selection_pressure - 2*(selection_pressure-1)*(rank-1)/(N-1))

    # Normalize (should already sum to 1, but ensure numerical stability)
    probabilities = probabilities / probabilities.sum()

    # Select
    idx = np.random.choice(N, p=probabilities)
    return sorted_pop[idx]


def exponential_rank_selection(
    population: List[Individual],
    minimize: bool = True,
    base: float = 0.95
) -> Individual:
    """
    Exponential rank-based selection.

    Stronger selection pressure than linear ranking.
    P(rank r) ∝ base^(r-1)

    Parameters
    ----------
    base : float
        Exponential base, typically in [0.9, 0.99]
        Lower = higher selection pressure
    """
    # Sort population
    if minimize:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness)
    else:
        sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)

    N = len(population)

    # Exponential ranking: best gets base^0, worst gets base^(N-1)
    ranks = np.arange(N)
    weights = base ** ranks
    probabilities = weights / weights.sum()

    idx = np.random.choice(N, p=probabilities)
    return sorted_pop[idx]
```

### Comparison and Analysis

```python
@dataclass
class SimpleIndividual:
    """Simple individual for testing."""
    fitness: float
    id: int

    def copy(self):
        return SimpleIndividual(self.fitness, self.id)


def analyze_selection_methods():
    """
    Compare selection methods on diversity and selection pressure.
    """
    import matplotlib.pyplot as plt
    from collections import Counter

    # Create population with varying fitness
    np.random.seed(42)
    population = [
        SimpleIndividual(fitness=np.random.exponential(scale=1.0), id=i)
        for i in range(50)
    ]

    n_selections = 1000
    methods = {
        'Tournament (k=2)': lambda: tournament_selection(population, tournament_size=2),
        'Tournament (k=5)': lambda: tournament_selection(population, tournament_size=5),
        'Roulette Wheel': lambda: roulette_wheel_selection(population),
        'Rank (pressure=1.5)': lambda: rank_selection(population, selection_pressure=1.5),
        'Rank (pressure=2.0)': lambda: rank_selection(population, selection_pressure=2.0),
    }

    # Run selections
    results = {}
    for name, method in methods.items():
        selected_ids = [method().id for _ in range(n_selections)]
        results[name] = Counter(selected_ids)

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Sort population by fitness for consistent plotting
    sorted_pop = sorted(population, key=lambda x: x.fitness)
    fitness_order = [ind.id for ind in sorted_pop]

    for idx, (name, counts) in enumerate(results.items()):
        ax = axes[idx]

        # Get selection counts in fitness order
        selection_counts = [counts.get(id, 0) for id in fitness_order]

        ax.bar(range(len(selection_counts)), selection_counts)
        ax.set_title(name)
        ax.set_xlabel('Individual (sorted by fitness)')
        ax.set_ylabel('Times Selected')
        ax.axhline(y=n_selections/len(population), color='r', linestyle='--',
                   label='Uniform')
        ax.legend()

    # Hide unused subplot
    axes[-1].axis('off')

    plt.tight_layout()
    plt.savefig('selection_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Calculate statistics
    print("\nSelection Statistics")
    print("=" * 70)
    print(f"{'Method':<25} {'Best Selected':<15} {'Unique Selected':<20}")
    print("-" * 70)

    best_id = sorted_pop[0].id  # Best individual

    for name, counts in results.items():
        best_count = counts.get(best_id, 0)
        unique_count = len(counts)
        print(f"{name:<25} {best_count:<15} {unique_count:<20}")


def measure_selection_pressure():
    """
    Measure theoretical vs. empirical selection pressure.
    """
    # Create population
    population = [SimpleIndividual(fitness=i, id=i) for i in range(100)]
    n_trials = 10000

    methods = [
        ('Tournament k=2', lambda: tournament_selection(population, tournament_size=2)),
        ('Tournament k=3', lambda: tournament_selection(population, tournament_size=3)),
        ('Tournament k=5', lambda: tournament_selection(population, tournament_size=5)),
        ('Tournament k=7', lambda: tournament_selection(population, tournament_size=7)),
        ('Roulette', lambda: roulette_wheel_selection(population)),
        ('Rank (p=1.5)', lambda: rank_selection(population, selection_pressure=1.5)),
        ('Rank (p=2.0)', lambda: rank_selection(population, selection_pressure=2.0)),
    ]

    print("\nSelection Pressure Analysis")
    print("=" * 70)
    print(f"{'Method':<20} {'P(best)':<15} {'Pressure s':<15}")
    print("-" * 70)

    for name, method in methods:
        # Count selections of best individual
        best_count = sum(1 for _ in range(n_trials) if method().id == 0)
        p_best = best_count / n_trials

        # Selection pressure = N * P(best)
        pressure = len(population) * p_best

        print(f"{name:<20} {p_best:<15.4f} {pressure:<15.2f}")


if __name__ == "__main__":
    analyze_selection_methods()
    measure_selection_pressure()
```

## Common Pitfalls

### 1. Wrong Fitness Transformation for Minimization

**Problem**: Using raw fitness for roulette wheel in minimization problems.

```python
# Bad - higher fitness gets higher probability!
def bad_roulette(population):
    fitnesses = np.array([ind.fitness for ind in population])
    probs = fitnesses / fitnesses.sum()  # WRONG for minimization
    return np.random.choice(population, p=probs)

# Good - invert fitness
def good_roulette(population):
    fitnesses = np.array([ind.fitness for ind in population])
    max_fit = fitnesses.max()
    inverted = max_fit - fitnesses + 1e-10
    probs = inverted / inverted.sum()
    return np.random.choice(population, p=probs)
```

### 2. Tournament Selection with Replacement

**Problem**: Selecting same individual multiple times in tournament.

```python
# Bad - can select same individual twice
tournament = [population[np.random.randint(len(population))]
              for _ in range(k)]

# Good - sample without replacement
tournament = np.random.choice(population, size=k, replace=False)
```

### 3. Premature Convergence from High Pressure

**Problem**: Tournament too large or selection pressure too high too early.

**Solution**: Start with low pressure, increase over time.

```python
def adaptive_pressure_selection(population, generation, max_gen):
    """Gradually increase selection pressure."""
    # Start with k=2, end with k=7
    progress = generation / max_gen
    k = int(2 + 5 * progress)
    return tournament_selection(population, tournament_size=k)
```

### 4. Negative Fitness in Roulette Wheel

**Problem**: Roulette wheel fails with negative fitnesses.

```python
# Bad - negative probabilities!
def bad_roulette_negative(population):
    fitnesses = np.array([ind.fitness for ind in population])
    probs = fitnesses / fitnesses.sum()  # Fails if any negative

# Good - shift to positive range
def good_roulette_negative(population):
    fitnesses = np.array([ind.fitness for ind in population])
    min_fit = fitnesses.min()
    shifted = fitnesses - min_fit + 1  # All positive
    probs = shifted / shifted.sum()
```

## Connections

### Prerequisites
- Basic probability and statistics
- Population-based optimization concepts
- Understanding of fitness functions

### Leads To
- Crossover operators
- Mutation operators
- Replacement strategies
- Multi-objective optimization (Pareto selection)

### Related Concepts
- Exploration vs. exploitation trade-off
- Diversity maintenance
- Fitness sharing and niching
- Island models and migration

## Practice Problems

### Problem 1: Elitist Tournament

Implement tournament selection that always includes the best individual.

```python
def elitist_tournament(population: List[Individual],
                      tournament_size: int = 3) -> Individual:
    """
    Tournament that always includes the current best.
    Ensures best individual has high selection probability.
    """
    # Your implementation here
    pass
```

### Problem 2: Boltzmann Selection

Implement selection based on Boltzmann distribution (simulated annealing-inspired).

```python
def boltzmann_selection(population: List[Individual],
                       temperature: float = 1.0,
                       minimize: bool = True) -> Individual:
    """
    Selection probability: P(i) ∝ exp(-fitness_i / temperature)

    High temperature → more random (exploration)
    Low temperature → more greedy (exploitation)
    """
    # Your implementation here
    pass
```

### Problem 3: Comparative Analysis

Compare selection methods on a test problem.

```python
def compare_on_test_problem():
    """
    Create a test problem and compare:
    - Convergence speed (generations to solution)
    - Diversity over time (unique individuals)
    - Success rate (finding global optimum)

    Test with different fitness landscapes:
    1. Unimodal (single peak)
    2. Multimodal (multiple peaks)
    3. Deceptive (misleading gradients)
    """
    pass
```

### Problem 4: Diversity-Preserving Selection

Implement selection that maintains population diversity.

```python
def fitness_sharing_selection(population: List[Individual],
                              sigma: float = 1.0) -> Individual:
    """
    Reduce fitness of individuals in crowded regions.

    Shared fitness: f'(i) = f(i) / Σ_j sh(d(i,j))
    where sh(d) = 1 - (d/sigma) if d < sigma, else 0
    """
    # Your implementation here
    pass
```

### Problem 5: Multi-Objective Selection

Implement selection for multi-objective optimization (NSGA-II inspired).

```python
def non_dominated_tournament(population: List[Individual],
                            tournament_size: int = 3) -> Individual:
    """
    Tournament based on:
    1. Non-domination rank (lower is better)
    2. Crowding distance (higher is better) as tiebreaker

    Assumes individual has .rank and .crowding_distance attributes
    """
    # Your implementation here
    pass
```

## Further Reading

### Academic Papers

- Goldberg, D.E., & Deb, K. (1991). "A Comparative Analysis of Selection Schemes Used in Genetic Algorithms". Foundations of Genetic Algorithms, 1, 69-93.
  - Comprehensive comparison of selection methods

- Miller, B.L., & Goldberg, D.E. (1995). "Genetic Algorithms, Tournament Selection, and the Effects of Noise". Complex Systems, 9(3), 193-212.
  - Analysis of tournament selection under noise

- Baker, J.E. (1987). "Reducing Bias and Inefficiency in the Selection Algorithm". Genetic Algorithms and their Applications, 14-21.
  - Introduction of Stochastic Universal Sampling

### Books

- Eiben, A.E., & Smith, J.E. (2015). "Introduction to Evolutionary Computing" (2nd ed.)
  - Chapter 4: Selection mechanisms

- Goldberg, D.E. (1989). "Genetic Algorithms in Search, Optimization, and Machine Learning"
  - Chapter 3: Selection and reproduction

### Online Resources

- DEAP Selection Documentation: https://deap.readthedocs.io/en/master/api/tools.html#selection
  - Practical implementations of all selection methods

- Genetic Algorithm Tutorial: http://www.obitko.com/tutorials/genetic-algorithms/selection.php
  - Interactive visualizations

### Key Insights from Literature

1. **Tournament selection is most robust** across different fitness landscapes
2. **Selection pressure should adapt** over the run (start low, increase)
3. **SUS is superior to repeated roulette** for proportionate selection
4. **Rank selection prevents premature convergence** better than fitness-proportionate
